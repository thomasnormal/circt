//===- CoverageOps.cpp - Coverage dialect operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Coverage/CoverageOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace circt;
using namespace coverage;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ToggleCoverageOp
//===----------------------------------------------------------------------===//

unsigned ToggleCoverageOp::getSignalWidth() {
  Type signalType = getSignal().getType();
  if (auto intType = dyn_cast<IntegerType>(signalType))
    return intType.getWidth();
  // Default to 1 for non-integer types
  return 1;
}

//===----------------------------------------------------------------------===//
// FSMTransitionCoverageOp
//===----------------------------------------------------------------------===//

LogicalResult FSMTransitionCoverageOp::verify() {
  // Verify that prev_state and next_state have the same type
  if (getPrevState().getType() != getNextState().getType())
    return emitOpError("prev_state and next_state must have the same type");

  // Verify that num_states is reasonable for the bit width
  unsigned width =
      cast<IntegerType>(getPrevState().getType()).getWidth();
  unsigned maxStates = 1u << width;
  if (getNumStates() > static_cast<int32_t>(maxStates))
    return emitOpError("num_states (")
           << getNumStates() << ") exceeds maximum for " << width
           << "-bit state (" << maxStates << ")";

  return success();
}

// Operation implementations generated from `Coverage.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Coverage/Coverage.cpp.inc"

void CoverageDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Coverage/Coverage.cpp.inc"
      >();
}
