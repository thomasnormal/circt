//===- HWExternalizeModules.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts selected hw.module operations to hw.module.extern.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWEXTERNALIZEMODULES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct HWExternalizeModulesPass
    : public circt::hw::impl::HWExternalizeModulesBase<
          HWExternalizeModulesPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void HWExternalizeModulesPass::runOnOperation() {
  llvm::StringSet<> requestedNames;
  for (const std::string &entry : moduleNames) {
    SmallVector<StringRef> tokens;
    StringRef(entry).split(tokens, ",", /*MaxSplit=*/-1,
                           /*KeepEmpty=*/false);
    for (StringRef token : tokens) {
      StringRef trimmed = token.trim();
      if (trimmed.empty())
        continue;
      requestedNames.insert(trimmed);
    }
  }

  if (requestedNames.empty())
    return;

  llvm::StringSet<> matchedNames;
  mlir::ModuleOp module = getOperation();
  for (auto hwModule : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    StringRef moduleName = hwModule.getName();
    if (!requestedNames.contains(moduleName))
      continue;
    matchedNames.insert(moduleName);

    OpBuilder builder(hwModule);
    HWModuleExternOp::create(builder, hwModule.getLoc(), hwModule.getNameAttr(),
                             hwModule.getPortList(),
                             hw::getVerilogModuleNameAttr(hwModule).getValue(),
                             hwModule.getParameters());
    hwModule.erase();
  }

  if (allowMissing)
    return;

  SmallVector<std::string> missing;
  missing.reserve(requestedNames.size());
  for (const auto &it : requestedNames) {
    if (!matchedNames.contains(it.getKey()))
      missing.push_back(it.getKey().str());
  }
  if (missing.empty())
    return;

  llvm::sort(missing);
  module.emitError() << "missing requested modules: " << llvm::join(missing, ",");
  signalPassFailure();
}
