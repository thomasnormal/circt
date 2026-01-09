//===- CoveragePasses.h - Coverage dialect passes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_H
#define CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace coverage {

#define GEN_PASS_DECL
#include "circt/Dialect/Coverage/CoveragePasses.h.inc"

std::unique_ptr<mlir::Pass>
createInstrumentCoveragePass(const InstrumentCoverageOptions &options = {});

std::unique_ptr<mlir::Pass>
createExportCoverageDataPass(const ExportCoverageDataOptions &options = {});

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Coverage/CoveragePasses.h.inc"

} // namespace coverage
} // namespace circt

#endif // CIRCT_DIALECT_COVERAGE_COVERAGEPASSES_H
