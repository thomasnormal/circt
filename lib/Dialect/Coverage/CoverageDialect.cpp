//===- CoverageDialect.cpp - Coverage dialect implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Coverage/CoverageDialect.h"
#include "circt/Dialect/Coverage/CoverageOps.h"

using namespace circt;
using namespace coverage;

void CoverageDialect::initialize() { registerOps(); }

// Dialect implementation generated from `CoverageDialect.td`
#include "circt/Dialect/Coverage/CoverageDialect.cpp.inc"
