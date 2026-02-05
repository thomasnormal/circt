//===- SMTModel.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for normalizing and printing SMT model values.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SMTMODEL_H
#define CIRCT_SUPPORT_SMTMODEL_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace circt {

std::string normalizeSMTModelValue(llvm::StringRef value);

} // namespace circt

extern "C" void circt_smt_print_model_header();
extern "C" void circt_smt_print_model_value(const char *name,
                                            const char *value);

#endif // CIRCT_SUPPORT_SMTMODEL_H
