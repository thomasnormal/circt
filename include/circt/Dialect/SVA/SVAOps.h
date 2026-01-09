//===- SVAOps.h - SVA dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SVA_SVAOPS_H
#define CIRCT_DIALECT_SVA_SVAOPS_H

#include "circt/Dialect/SVA/SVADialect.h"
#include "circt/Dialect/SVA/SVATypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Enum definitions generated from `SVAOps.td`
#include "circt/Dialect/SVA/SVAEnums.h.inc"

// Operation definitions generated from `SVAOps.td`
#define GET_OP_CLASSES
#include "circt/Dialect/SVA/SVA.h.inc"

#endif // CIRCT_DIALECT_SVA_SVAOPS_H
