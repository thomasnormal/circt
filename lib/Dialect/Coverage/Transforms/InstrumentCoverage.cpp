//===- InstrumentCoverage.cpp - Insert coverage instrumentation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass instruments hardware designs with coverage collection operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Coverage/CoverageOps.h"
#include "circt/Dialect/Coverage/CoveragePasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"

namespace circt {
namespace coverage {
#define GEN_PASS_DEF_INSTRUMENTCOVERAGE
#include "circt/Dialect/Coverage/CoveragePasses.h.inc"
} // namespace coverage
} // namespace circt

using namespace circt;
using namespace coverage;
using namespace hw;

namespace {
struct InstrumentCoveragePass
    : public coverage::impl::InstrumentCoverageBase<InstrumentCoveragePass> {
  using InstrumentCoverageBase::InstrumentCoverageBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Walk all HW modules and instrument them
    moduleOp.walk([&](hw::HWModuleOp hwModule) {
      instrumentModule(hwModule);
    });
  }

  void instrumentModule(hw::HWModuleOp moduleOp) {
    std::string moduleName = moduleOp.getName().str();
    std::string hierarchy = hierarchyPrefix.empty()
                                ? moduleName
                                : (hierarchyPrefix + "." + moduleName);

    // Insert line coverage at the start of each module
    if (instrumentLine) {
      OpBuilder builder(moduleOp.getBody());
      builder.setInsertionPointToStart(moduleOp.getBodyBlock());

      // Get location info if available
      auto loc = moduleOp.getLoc();
      std::string filename = "unknown";
      int32_t line = 0;

      if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
        filename = fileLoc.getFilename().str();
        line = fileLoc.getLine();
      }

      LineCoverageOp::create(builder, loc, filename, line,
                             StringAttr::get(&getContext(), moduleName));
      ++numLineCoverageOps;
    }

    // Insert toggle coverage for module ports
    if (instrumentToggle) {
      instrumentModulePorts(moduleOp, hierarchy);
    }

    // Walk operations looking for things to instrument
    moduleOp.walk([&](Operation *op) {
      if (instrumentLine) {
        instrumentLineForOp(op, hierarchy);
      }
      if (instrumentToggle) {
        instrumentToggleForOp(op, hierarchy);
      }
      if (instrumentBranch) {
        instrumentBranchForOp(op, hierarchy);
      }
    });
  }

  void instrumentModulePorts(hw::HWModuleOp moduleOp, StringRef hierarchy) {
    OpBuilder builder(moduleOp.getBody());
    builder.setInsertionPointToStart(moduleOp.getBodyBlock());
    auto loc = moduleOp.getLoc();

    ModulePortInfo ports(moduleOp.getPortList());

    // Instrument input ports
    for (auto [port, arg] :
         llvm::zip(ports.getInputs(), moduleOp.getBodyBlock()->getArguments())) {
      // Only instrument integer types
      if (!isa<IntegerType>(arg.getType()))
        continue;

      std::string name = (hierarchy + "." + port.getName()).str();
      ToggleCoverageOp::create(builder, loc, arg, port.getName(),
                               StringAttr::get(&getContext(), hierarchy));
      ++numToggleCoverageOps;
    }

    // Find the output op to instrument output ports
    auto *outputOp = moduleOp.getBodyBlock()->getTerminator();
    builder.setInsertionPoint(outputOp);

    for (auto [port, result] :
         llvm::zip(ports.getOutputs(), outputOp->getOperands())) {
      if (!isa<IntegerType>(result.getType()))
        continue;

      std::string name = (hierarchy + "." + port.getName()).str();
      ToggleCoverageOp::create(builder, loc, result, port.getName(),
                               StringAttr::get(&getContext(), hierarchy));
      ++numToggleCoverageOps;
    }
  }

  void instrumentLineForOp(Operation *op, StringRef hierarchy) {
    // Skip coverage ops themselves
    if (isa<coverage::LineCoverageOp, coverage::ToggleCoverageOp,
            coverage::BranchCoverageOp, coverage::CoverageGroupOp>(op))
      return;

    // Instrument operations that represent statements
    // For now, instrument seq.compreg and hw.instance operations
    if (!isa<seq::CompRegOp, hw::InstanceOp>(op))
      return;

    auto loc = op->getLoc();
    std::string filename = "unknown";
    int32_t line = 0;

    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      filename = fileLoc.getFilename().str();
      line = fileLoc.getLine();
    }

    OpBuilder builder(op);
    LineCoverageOp::create(builder, loc, filename, line,
                           StringAttr::get(&getContext(), hierarchy));
    ++numLineCoverageOps;
  }

  void instrumentToggleForOp(Operation *op, StringRef hierarchy) {
    // Instrument register outputs for toggle coverage
    if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
      auto result = regOp.getResult();
      if (!isa<IntegerType>(result.getType()))
        return;

      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);

      StringRef name = "";
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        name = nameAttr.getValue();
      else if (auto nameAttr = op->getAttrOfType<StringAttr>("sv.namehint"))
        name = nameAttr.getValue();

      if (!name.empty()) {
        ToggleCoverageOp::create(builder, op->getLoc(), result, name,
                                 StringAttr::get(&getContext(), hierarchy));
        ++numToggleCoverageOps;
      }
    }
  }

  void instrumentBranchForOp(Operation *op, StringRef hierarchy) {
    // Instrument mux operations as branch points
    if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
      auto cond = muxOp.getCond();
      if (!isa<IntegerType>(cond.getType()) ||
          cast<IntegerType>(cond.getType()).getWidth() != 1)
        return;

      auto loc = op->getLoc();
      std::string filename = "";
      int32_t line = 0;

      if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
        filename = fileLoc.getFilename().str();
        line = fileLoc.getLine();
      }

      OpBuilder builder(op);
      // Create unique IDs based on operation position
      static int32_t branchCounter = 0;
      int32_t trueId = branchCounter++;
      int32_t falseId = branchCounter++;

      std::string name = "branch_" + std::to_string(trueId / 2);
      auto filenameAttr = filename.empty()
                              ? StringAttr()
                              : StringAttr::get(&getContext(), filename);
      auto lineAttr =
          line > 0 ? IntegerAttr::get(IntegerType::get(&getContext(), 32), line)
                   : IntegerAttr();

      BranchCoverageOp::create(builder, loc, cond, name, trueId, falseId,
                               filenameAttr, lineAttr);
      ++numBranchCoverageOps;
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
coverage::createInstrumentCoveragePass(const InstrumentCoverageOptions &options) {
  return std::make_unique<InstrumentCoveragePass>(options);
}
