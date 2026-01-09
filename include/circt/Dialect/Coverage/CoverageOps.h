//===- CoverageOps.h - Coverage dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COVERAGE_COVERAGEOPS_H
#define CIRCT_DIALECT_COVERAGE_COVERAGEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Coverage/CoverageDialect.h"

// Operation definitions generated from `Coverage.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Coverage/Coverage.h.inc"

#endif // CIRCT_DIALECT_COVERAGE_COVERAGEOPS_H
