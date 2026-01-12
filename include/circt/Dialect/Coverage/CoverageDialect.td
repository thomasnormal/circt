//===- CoverageDialect.td - Coverage dialect definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Coverage dialect for code coverage instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COVERAGE_COVERAGEDIALECT_TD
#define CIRCT_DIALECT_COVERAGE_COVERAGEDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def CoverageDialect : Dialect {
  let name = "coverage";
  let summary = "Coverage instrumentation dialect for simulation";
  let description = [{
    The Coverage dialect provides operations for instrumenting hardware designs
    with coverage collection capabilities. This includes:

    - Line coverage: Track which lines of source code are executed
    - Toggle coverage: Track signal transitions (0->1 and 1->0)
    - Branch coverage: Track which branches are taken (future)
    - FSM coverage: Track state machine transitions (future)

    The dialect is designed to work with simulation flows and can export
    coverage data in standard formats for analysis.
  }];
  let cppNamespace = "circt::coverage";

  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
    void registerOps();
  }];
}

#endif // CIRCT_DIALECT_COVERAGE_COVERAGEDIALECT_TD
