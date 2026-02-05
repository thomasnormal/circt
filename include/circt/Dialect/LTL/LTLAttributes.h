//===- LTLAttributes.h - Declare LTL dialect attributes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLATTRIBUTES_H
#define CIRCT_DIALECT_LTL_LTLATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "circt/Dialect/LTL/LTLDialect.h"
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LTL/LTLAttributes.h.inc"

#endif // CIRCT_DIALECT_LTL_LTLATTRIBUTES_H
