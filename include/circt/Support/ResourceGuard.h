//===- ResourceGuard.h - Memory usage safeguards ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for installing coarse process-wide resource safeguards. These are
// meant to prevent catastrophic "runaway" behavior (e.g. pass pipelines that
// allocate unbounded memory or get stuck) from taking down a machine.
//
// The guard is intentionally simple: users can configure a maximum RSS or
// malloc-heap usage via command line flags (or environment variables), after
// which the process terminates.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_RESOURCEGUARD_H
#define CIRCT_SUPPORT_RESOURCEGUARD_H

#include <cstdint>
#include <optional>

namespace llvm {
class StringRef;
namespace cl {
class OptionCategory;
} // namespace cl
} // namespace llvm

namespace circt {

/// Parse a string containing an unsigned integer number of megabytes.
std::optional<uint64_t> parseMegabytes(llvm::StringRef text);

/// Return the command line category used for resource guard options.
llvm::cl::OptionCategory &getResourceGuardCategory();

/// Install the resource guard based on command line options and environment
/// variables. This should be called after command line parsing.
void installResourceGuard();

} // namespace circt

#endif // CIRCT_SUPPORT_RESOURCEGUARD_H
