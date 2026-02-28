//===- NativeMutationPlanner.h - Native mutation planning -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for deterministic CIRCT-only mutation planning used by circt-mut.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_MUT_NATIVEMUTATIONPLANNER_H
#define CIRCT_TOOLS_CIRCT_MUT_NATIVEMUTATIONPLANNER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace circt::mut {

struct NativeMutationPlannerConfig {
  enum class Policy { Legacy, Weighted };

  Policy policy = Policy::Legacy;
  int pickCoverPercent = 80;
  int weightCover = 500;
  int weightPQW = 100;
  int weightPQB = 100;
  int weightPQC = 100;
  int weightPQS = 100;
  int weightPQMW = 100;
  int weightPQMB = 100;
  int weightPQMC = 100;
  int weightPQMS = 100;
};

bool parseNativeMutationPlannerConfig(
    llvm::ArrayRef<std::string> cfgEntries, NativeMutationPlannerConfig &config,
    std::string &error);

bool computeOrderedNativeMutationOps(llvm::StringRef designText,
                                     llvm::SmallVectorImpl<std::string> &orderedOps,
                                     std::string &error);

bool hasNativeMutationPatternForOp(llvm::StringRef designText,
                                   llvm::StringRef op);

bool emitNativeMutationPlan(llvm::ArrayRef<std::string> orderedOps,
                            llvm::StringRef designText, uint64_t count,
                            uint64_t seed,
                            const NativeMutationPlannerConfig &config,
                            llvm::raw_ostream &out, std::string &error);

bool applyNativeMutationLabel(llvm::StringRef designText, llvm::StringRef label,
                              std::string &mutatedText, bool &changed,
                              std::string &error);

} // namespace circt::mut

#endif // CIRCT_TOOLS_CIRCT_MUT_NATIVEMUTATIONPLANNER_H
