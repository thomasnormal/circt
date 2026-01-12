//===- SVADialect.cpp - SVA dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SVA/SVADialect.h"
#include "circt/Dialect/SVA/SVAOps.h"
#include "circt/Dialect/SVA/SVATypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace sva;

void SVADialect::initialize() {
  registerTypes();
  registerOps();
}

void SVADialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SVA/SVATypes.cpp.inc"
      >();
}

void SVADialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SVA/SVA.cpp.inc"
      >();
}

// Dialect implementation generated from `SVADialect.td`
#include "circt/Dialect/SVA/SVADialect.cpp.inc"

// Type implementation generated from `SVATypes.td`
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SVA/SVATypes.cpp.inc"
