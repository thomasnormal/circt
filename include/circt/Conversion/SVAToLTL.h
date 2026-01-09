//===- SVAToLTL.h - SVA to LTL conversion pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass to convert SVA dialect operations to LTL and
// Verif dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SVATOLTL_H
#define CIRCT_CONVERSION_SVATOLTL_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {

#define GEN_PASS_DECL_LOWERSVATOLTL
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerSVAToLTLPass();

} // namespace circt

#endif // CIRCT_CONVERSION_SVATOLTL_H
