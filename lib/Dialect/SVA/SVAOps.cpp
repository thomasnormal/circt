//===- SVAOps.cpp - SVA dialect operations implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SVA/SVAOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace sva;
using namespace mlir;

//===----------------------------------------------------------------------===//
// SequenceDelayOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceDelayOp::verify() {
  // Delay must be non-negative
  if (getDelay() < 0)
    return emitOpError("delay must be non-negative");

  // Length (if specified) must be non-negative
  if (auto length = getLength())
    if (*length < 0)
      return emitOpError("length must be non-negative");

  return success();
}

//===----------------------------------------------------------------------===//
// SequenceRepeatOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceRepeatOp::verify() {
  // Base must be non-negative
  if (getBase() < 0)
    return emitOpError("base repetition count must be non-negative");

  // More (if specified) must be non-negative
  if (auto more = getMore())
    if (*more < 0)
      return emitOpError("additional repetition count must be non-negative");

  return success();
}

//===----------------------------------------------------------------------===//
// SequenceConcatOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceConcatOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one input sequence");
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceOrOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceOrOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two input sequences");
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceAndOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceAndOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two input sequences");
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceIntersectOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceIntersectOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two input sequences");
  return success();
}

//===----------------------------------------------------------------------===//
// PropertyAndOp
//===----------------------------------------------------------------------===//

LogicalResult PropertyAndOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two input properties");
  return success();
}

//===----------------------------------------------------------------------===//
// PropertyOrOp
//===----------------------------------------------------------------------===//

LogicalResult PropertyOrOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two input properties");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

// Enum attribute implementation generated from `SVAOps.td`
#include "circt/Dialect/SVA/SVAEnums.cpp.inc"

// Operation implementation generated from `SVAOps.td`
#define GET_OP_CLASSES
#include "circt/Dialect/SVA/SVA.cpp.inc"
