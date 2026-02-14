//===- HWStopatSymbolic.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Apply stopat-style symbolic cutpoints to selected instance output nets.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWSTOPATSYMBOLIC
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct StopatSelector {
  std::string key;
  std::string instanceName;
  std::string portName;
};

struct HWStopatSymbolicPass
    : public circt::hw::impl::HWStopatSymbolicBase<HWStopatSymbolicPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static std::optional<StopatSelector> parseStopatSelector(StringRef raw,
                                                         std::string &error) {
  StringRef token = raw.trim();
  if (token.empty()) {
    error = "empty selector";
    return std::nullopt;
  }

  token.consume_front("*");
  if (token.empty()) {
    error = "selector missing 'inst.port' payload";
    return std::nullopt;
  }

  SmallVector<StringRef> parts;
  token.split(parts, '.', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.size() != 2) {
    error = "unsupported selector form (expected 'inst.port' or '*inst.port')";
    return std::nullopt;
  }

  StringRef instanceName = parts[0].trim();
  StringRef portName = parts[1].trim();
  if (instanceName.empty() || portName.empty()) {
    error = "selector requires non-empty instance and port names";
    return std::nullopt;
  }

  StopatSelector out;
  out.instanceName = instanceName.str();
  out.portName = portName.str();
  out.key = (instanceName + "." + portName).str();
  return out;
}

void HWStopatSymbolicPass::runOnOperation() {
  llvm::StringMap<StopatSelector> selectorsByKey;
  for (const std::string &entry : targets) {
    SmallVector<StringRef> tokens;
    StringRef(entry).split(tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef token : tokens) {
      std::string parseError;
      auto selector = parseStopatSelector(token, parseError);
      if (!selector) {
        getOperation()->emitError()
            << "invalid --hw-stopat-symbolic selector '" << token.trim()
            << "': " << parseError;
        signalPassFailure();
        return;
      }
      selectorsByKey.try_emplace(selector->key, *selector);
    }
  }

  if (selectorsByKey.empty())
    return;

  llvm::StringMap<unsigned> matchCounts;
  for (auto &it : selectorsByKey)
    matchCounts.try_emplace(it.getKey(), 0);

  mlir::ModuleOp module = getOperation();
  module.walk([&](hw::InstanceOp instance) {
    StringRef instanceName = instance.getInstanceName();
    for (auto &it : selectorsByKey) {
      const StopatSelector &selector = it.second;
      if (instanceName != selector.instanceName)
        continue;
      for (auto [idx, portAttr] : llvm::enumerate(instance.getResultNames())) {
        auto portName = mlir::cast<mlir::StringAttr>(portAttr).getValue();
        if (portName != selector.portName)
          continue;
        OpBuilder builder(instance);
        builder.setInsertionPointAfter(instance);
        auto sym = verif::SymbolicValueOp::create(builder, instance.getLoc(),
                                                  instance.getResult(idx).getType());
        instance.getResult(idx).replaceAllUsesWith(sym.getResult());
        matchCounts[selector.key] += 1;
      }
    }
  });

  if (allowUnmatched)
    return;

  SmallVector<std::string> unmatched;
  for (const auto &it : matchCounts)
    if (it.second == 0)
      unmatched.push_back(it.first().str());
  if (unmatched.empty())
    return;

  llvm::sort(unmatched);
  module.emitError() << "unmatched stopat selectors: " << llvm::join(unmatched, ",");
  signalPassFailure();
}
