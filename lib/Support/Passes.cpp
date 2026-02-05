//===- Passes.cpp - Pass Utilities ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace circt;

std::unique_ptr<Pass> circt::createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  return mlir::createCanonicalizerPass(config);
}

std::unique_ptr<Pass> circt::createBottomUpSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(false);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  // Put a conservative cap on rewrites to prevent pathological behavior from
  // consuming unbounded memory/time in large tool pipelines.
  config.setMaxNumRewrites(200000);
  return mlir::createCanonicalizerPass(config);
}

std::unique_ptr<Pass> circt::createBottomUpCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(false);
  return mlir::createCanonicalizerPass(config);
}
