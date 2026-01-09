//===- ExportCoverageData.cpp - Export coverage metadata ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass exports coverage instrumentation metadata for runtime use.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Coverage/CoverageOps.h"
#include "circt/Dialect/Coverage/CoveragePasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace circt {
namespace coverage {
#define GEN_PASS_DEF_EXPORTCOVERAGEDATA
#include "circt/Dialect/Coverage/CoveragePasses.h.inc"
} // namespace coverage
} // namespace circt

using namespace circt;
using namespace coverage;

namespace {
struct ExportCoverageDataPass
    : public coverage::impl::ExportCoverageDataBase<ExportCoverageDataPass> {
  using ExportCoverageDataBase::ExportCoverageDataBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Collect all coverage points
    std::vector<LineCoverageOp> lineCoverageOps;
    std::vector<ToggleCoverageOp> toggleCoverageOps;
    std::vector<BranchCoverageOp> branchCoverageOps;

    moduleOp.walk([&](Operation *op) {
      if (auto lineOp = dyn_cast<LineCoverageOp>(op))
        lineCoverageOps.push_back(lineOp);
      else if (auto toggleOp = dyn_cast<ToggleCoverageOp>(op))
        toggleCoverageOps.push_back(toggleOp);
      else if (auto branchOp = dyn_cast<BranchCoverageOp>(op))
        branchCoverageOps.push_back(branchOp);
    });

    if (format == "json") {
      exportJSON(lineCoverageOps, toggleCoverageOps, branchCoverageOps);
    } else {
      exportBinary(lineCoverageOps, toggleCoverageOps, branchCoverageOps);
    }
  }

  void exportJSON(const std::vector<LineCoverageOp> &lineCoverageOps,
                  const std::vector<ToggleCoverageOp> &toggleCoverageOps,
                  const std::vector<BranchCoverageOp> &branchCoverageOps) {
    llvm::json::Object root;

    // Export line coverage points
    llvm::json::Array lineArray;
    for (size_t i = 0; i < lineCoverageOps.size(); ++i) {
      auto op = lineCoverageOps[i];
      llvm::json::Object point;
      point["id"] = static_cast<int64_t>(i);
      point["filename"] = op.getFilename().str();
      point["line"] = static_cast<int64_t>(op.getLine());
      if (op.getTag())
        point["tag"] = op.getTag()->str();
      lineArray.push_back(std::move(point));
    }
    root["line_coverage"] = std::move(lineArray);

    // Export toggle coverage points
    llvm::json::Array toggleArray;
    for (size_t i = 0; i < toggleCoverageOps.size(); ++i) {
      auto op = toggleCoverageOps[i];
      llvm::json::Object point;
      point["id"] = static_cast<int64_t>(i);
      point["name"] = op.getName().str();
      point["width"] = static_cast<int64_t>(op.getSignalWidth());
      if (op.getHierarchy())
        point["hierarchy"] = op.getHierarchy()->str();
      toggleArray.push_back(std::move(point));
    }
    root["toggle_coverage"] = std::move(toggleArray);

    // Export branch coverage points
    llvm::json::Array branchArray;
    for (size_t i = 0; i < branchCoverageOps.size(); ++i) {
      auto op = branchCoverageOps[i];
      llvm::json::Object point;
      point["id"] = static_cast<int64_t>(i);
      point["name"] = op.getName().str();
      point["true_id"] = static_cast<int64_t>(op.getTrueId());
      point["false_id"] = static_cast<int64_t>(op.getFalseId());
      if (op.getFilename())
        point["filename"] = op.getFilename()->str();
      if (op.getLine())
        point["line"] = static_cast<int64_t>(*op.getLine());
      branchArray.push_back(std::move(point));
    }
    root["branch_coverage"] = std::move(branchArray);

    // Summary
    llvm::json::Object summary;
    summary["total_line_points"] =
        static_cast<int64_t>(lineCoverageOps.size());
    summary["total_toggle_points"] =
        static_cast<int64_t>(toggleCoverageOps.size());
    summary["total_branch_points"] =
        static_cast<int64_t>(branchCoverageOps.size());
    root["summary"] = std::move(summary);

    // Write to file
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFile, ec);
    if (ec) {
      emitError(getOperation().getLoc())
          << "failed to open output file: " << ec.message();
      signalPassFailure();
      return;
    }

    os << llvm::json::Value(std::move(root));
  }

  void exportBinary(const std::vector<LineCoverageOp> &lineCoverageOps,
                    const std::vector<ToggleCoverageOp> &toggleCoverageOps,
                    const std::vector<BranchCoverageOp> &branchCoverageOps) {
    // Binary format:
    // Header:
    //   4 bytes: magic "CCOV"
    //   4 bytes: version (1)
    //   4 bytes: number of line coverage points
    //   4 bytes: number of toggle coverage points
    //   4 bytes: number of branch coverage points
    // Then data for each type follows

    std::error_code ec;
    llvm::raw_fd_ostream os(outputFile, ec);
    if (ec) {
      emitError(getOperation().getLoc())
          << "failed to open output file: " << ec.message();
      signalPassFailure();
      return;
    }

    // Write header
    os << "CCOV";
    uint32_t version = 1;
    uint32_t numLine = lineCoverageOps.size();
    uint32_t numToggle = toggleCoverageOps.size();
    uint32_t numBranch = branchCoverageOps.size();

    os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    os.write(reinterpret_cast<const char *>(&numLine), sizeof(numLine));
    os.write(reinterpret_cast<const char *>(&numToggle), sizeof(numToggle));
    os.write(reinterpret_cast<const char *>(&numBranch), sizeof(numBranch));

    // Write line coverage data
    for (const auto &op : lineCoverageOps) {
      uint32_t line = op.getLine();
      os.write(reinterpret_cast<const char *>(&line), sizeof(line));
      auto filename = op.getFilename();
      uint32_t filenameLen = filename.size();
      os.write(reinterpret_cast<const char *>(&filenameLen),
               sizeof(filenameLen));
      os << filename;
    }

    // Write toggle coverage data
    for (const auto &op : toggleCoverageOps) {
      uint32_t width = op.getSignalWidth();
      os.write(reinterpret_cast<const char *>(&width), sizeof(width));
      auto name = op.getName();
      uint32_t nameLen = name.size();
      os.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
      os << name;
    }

    // Write branch coverage data
    for (const auto &op : branchCoverageOps) {
      uint32_t trueId = op.getTrueId();
      uint32_t falseId = op.getFalseId();
      os.write(reinterpret_cast<const char *>(&trueId), sizeof(trueId));
      os.write(reinterpret_cast<const char *>(&falseId), sizeof(falseId));
      auto name = op.getName();
      uint32_t nameLen = name.size();
      os.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
      os << name;
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
coverage::createExportCoverageDataPass(const ExportCoverageDataOptions &options) {
  return std::make_unique<ExportCoverageDataPass>(options);
}
