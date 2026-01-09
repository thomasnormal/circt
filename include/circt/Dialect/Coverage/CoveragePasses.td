//===- CoveragePasses.td - Coverage dialect passes ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_TD
#define CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_TD

include "mlir/Pass/PassBase.td"

def InstrumentCoverage : Pass<"coverage-instrument", "mlir::ModuleOp"> {
  let summary = "Insert coverage instrumentation operations";
  let description = [{
    This pass instruments the design with coverage collection operations.
    It can insert:

    - Line coverage operations at statement boundaries
    - Toggle coverage operations for selected signals
    - Branch coverage operations for conditionals

    Options control which types of coverage are enabled and which signals
    or statements are instrumented.

    Example:
    ```
    circt-opt --coverage-instrument=line=true,toggle=true input.mlir
    ```
  }];

  let dependentDialects = [
    "circt::comb::CombDialect",
    "circt::coverage::CoverageDialect",
    "circt::hw::HWDialect",
    "circt::seq::SeqDialect"
  ];

  let options = [
    Option<"instrumentLine", "line", "bool", "true",
           "Insert line coverage instrumentation">,
    Option<"instrumentToggle", "toggle", "bool", "true",
           "Insert toggle coverage for ports and wires">,
    Option<"instrumentBranch", "branch", "bool", "false",
           "Insert branch coverage for conditionals">,
    Option<"hierarchyPrefix", "hierarchy", "std::string", "\"\"",
           "Prefix for hierarchy paths in coverage data">
  ];

  let statistics = [
    Statistic<"numLineCoverageOps", "line-coverage-ops",
      "Number of line coverage operations inserted">,
    Statistic<"numToggleCoverageOps", "toggle-coverage-ops",
      "Number of toggle coverage operations inserted">,
    Statistic<"numBranchCoverageOps", "branch-coverage-ops",
      "Number of branch coverage operations inserted">
  ];
}

def ExportCoverageData : Pass<"coverage-export", "mlir::ModuleOp"> {
  let summary = "Export coverage instrumentation metadata";
  let description = [{
    Collects all coverage instrumentation points in the design and exports
    metadata that can be used by the runtime to map coverage counters to
    source locations.

    The output format can be JSON or a binary format suitable for runtime
    consumption.
  }];

  let dependentDialects = ["circt::coverage::CoverageDialect"];

  let options = [
    Option<"outputFile", "output", "std::string", "\"coverage.json\"",
           "Output file for coverage metadata">,
    Option<"format", "format", "std::string", "\"json\"",
           "Output format (json or binary)">
  ];
}

#endif // CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_TD
