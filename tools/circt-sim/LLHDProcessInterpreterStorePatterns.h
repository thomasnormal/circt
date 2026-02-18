//===- LLHDProcessInterpreterStorePatterns.h - Store pattern matching -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal helper matchers for recognizing common LLVM/HW store patterns used
// by module-level initialization and interface propagation logic.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERSTOREPATTERNS_H
#define CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERSTOREPATTERNS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace circt {
namespace sim {

struct InterfaceTriStateStorePattern {
  uint64_t condAddr = 0;
  uint64_t srcAddr = 0;
  unsigned condBitIndex = 0;
  mlir::Value elseValue;
};

inline bool hasSingleFieldIndex(llvm::ArrayRef<int64_t> pos, unsigned idx) {
  return pos.size() == 1 && pos.front() == static_cast<int64_t>(idx);
}

template <typename ResolveAddrFn>
inline bool matchFourStateCopyStore(mlir::Value storeValue,
                                    ResolveAddrFn &&resolveAddr,
                                    uint64_t &srcAddr) {
  srcAddr = 0;

  if (auto loadOp = storeValue.getDefiningOp<mlir::LLVM::LoadOp>()) {
    srcAddr = resolveAddr(loadOp.getAddr());
    return srcAddr != 0;
  }

  auto insertUnknown = storeValue.getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertUnknown || !hasSingleFieldIndex(insertUnknown.getPosition(), 1))
    return false;
  auto insertValue =
      insertUnknown.getContainer().getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertValue || !hasSingleFieldIndex(insertValue.getPosition(), 0))
    return false;

  auto valueExtract =
      insertValue.getValue().getDefiningOp<mlir::LLVM::ExtractValueOp>();
  auto unknownExtract =
      insertUnknown.getValue().getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!valueExtract || !unknownExtract)
    return false;
  if (!hasSingleFieldIndex(valueExtract.getPosition(), 0) ||
      !hasSingleFieldIndex(unknownExtract.getPosition(), 1))
    return false;
  if (valueExtract.getContainer() != unknownExtract.getContainer())
    return false;

  auto srcLoad =
      valueExtract.getContainer().getDefiningOp<mlir::LLVM::LoadOp>();
  if (!srcLoad)
    return false;
  srcAddr = resolveAddr(srcLoad.getAddr());
  return srcAddr != 0;
}

template <typename ResolveAddrFn>
inline bool matchFourStateStructCreateLoad(mlir::Value value,
                                           ResolveAddrFn &&resolveAddr,
                                           uint64_t &srcAddr) {
  srcAddr = 0;
  auto createOp = value.getDefiningOp<circt::hw::StructCreateOp>();
  if (!createOp || createOp.getNumOperands() != 2)
    return false;

  auto valueExtract =
      createOp.getOperand(0).getDefiningOp<mlir::LLVM::ExtractValueOp>();
  auto unknownExtract =
      createOp.getOperand(1).getDefiningOp<mlir::LLVM::ExtractValueOp>();
  if (!valueExtract || !unknownExtract)
    return false;
  if (!hasSingleFieldIndex(valueExtract.getPosition(), 0) ||
      !hasSingleFieldIndex(unknownExtract.getPosition(), 1))
    return false;
  if (valueExtract.getContainer() != unknownExtract.getContainer())
    return false;

  auto srcLoad =
      valueExtract.getContainer().getDefiningOp<mlir::LLVM::LoadOp>();
  if (!srcLoad)
    return false;
  srcAddr = resolveAddr(srcLoad.getAddr());
  return srcAddr != 0;
}

template <typename ResolveSignalFn>
inline bool matchFourStateProbeCopyStore(mlir::Value storeValue,
                                         ResolveSignalFn &&resolveSignal,
                                         SignalId &srcSignalId) {
  srcSignalId = 0;

  auto insertUnknown = storeValue.getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertUnknown || !hasSingleFieldIndex(insertUnknown.getPosition(), 1))
    return false;
  auto insertValue =
      insertUnknown.getContainer().getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertValue || !hasSingleFieldIndex(insertValue.getPosition(), 0))
    return false;

  auto valueExtract =
      insertValue.getValue().getDefiningOp<circt::hw::StructExtractOp>();
  auto unknownExtract =
      insertUnknown.getValue().getDefiningOp<circt::hw::StructExtractOp>();
  if (!valueExtract || !unknownExtract)
    return false;
  if (valueExtract.getFieldName() != "value" ||
      unknownExtract.getFieldName() != "unknown")
    return false;
  if (valueExtract.getInput() != unknownExtract.getInput())
    return false;

  auto probeOp = valueExtract.getInput().getDefiningOp<circt::llhd::ProbeOp>();
  if (!probeOp)
    return false;
  srcSignalId = resolveSignal(probeOp.getSignal());
  return srcSignalId != 0;
}

template <typename ResolveAddrFn>
inline bool matchInterfaceTriStateStore(mlir::Value storeValue,
                                        ResolveAddrFn &&resolveAddr,
                                        InterfaceTriStateStorePattern &pattern) {
  pattern = InterfaceTriStateStorePattern{};

  auto insertUnknown = storeValue.getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertUnknown || !hasSingleFieldIndex(insertUnknown.getPosition(), 1))
    return false;
  auto insertValue =
      insertUnknown.getContainer().getDefiningOp<mlir::LLVM::InsertValueOp>();
  if (!insertValue || !hasSingleFieldIndex(insertValue.getPosition(), 0))
    return false;

  auto valueExtract =
      insertValue.getValue().getDefiningOp<circt::hw::StructExtractOp>();
  auto unknownExtract =
      insertUnknown.getValue().getDefiningOp<circt::hw::StructExtractOp>();
  if (!valueExtract || !unknownExtract)
    return false;
  if (valueExtract.getFieldName() != "value" ||
      unknownExtract.getFieldName() != "unknown")
    return false;
  if (valueExtract.getInput() != unknownExtract.getInput())
    return false;

  auto ifOp = valueExtract.getInput().getDefiningOp<mlir::scf::IfOp>();
  if (!ifOp)
    return false;

  mlir::Value condValue = ifOp.getCondition();
  if (auto condExtract = condValue.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
    if (!hasSingleFieldIndex(condExtract.getPosition(), 0))
      return false;
    auto condLoad =
        condExtract.getContainer().getDefiningOp<mlir::LLVM::LoadOp>();
    if (!condLoad)
      return false;
    pattern.condAddr = resolveAddr(condLoad.getAddr());
    pattern.condBitIndex = 0;
  } else if (auto condLoad = condValue.getDefiningOp<mlir::LLVM::LoadOp>()) {
    pattern.condAddr = resolveAddr(condLoad.getAddr());
    pattern.condBitIndex = 0;
  } else {
    return false;
  }
  if (pattern.condAddr == 0)
    return false;

  auto thenYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  auto elseYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
  if (!thenYield || !elseYield || thenYield.getNumOperands() != 1 ||
      elseYield.getNumOperands() != 1)
    return false;

  uint64_t srcAddr = 0;
  if (!matchFourStateStructCreateLoad(thenYield.getOperand(0), resolveAddr,
                                      srcAddr) &&
      !matchFourStateCopyStore(thenYield.getOperand(0), resolveAddr, srcAddr))
    return false;
  if (srcAddr == 0)
    return false;

  pattern.srcAddr = srcAddr;
  pattern.elseValue = elseYield.getOperand(0);
  return true;
}

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERSTOREPATTERNS_H
