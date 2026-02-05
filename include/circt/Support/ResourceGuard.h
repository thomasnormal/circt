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

/// Set a best-effort label for the current "phase" of execution. If the
/// resource guard triggers, it will include this label in its diagnostic to
/// help narrow down where memory was consumed.
///
/// This is intended for coarse-grained tool-level phases (e.g. parsing,
/// pass pipeline, SMT export, solver run), not per-operation tracing.
void setResourceGuardPhase(llvm::StringRef phase);

/// Return the command line category used for resource guard options.
llvm::cl::OptionCategory &getResourceGuardCategory();

/// Install the resource guard based on command line options and environment
/// variables. This should be called after command line parsing.
///
/// In addition to RSS/malloc/VMem limits, the guard optionally supports a
/// wall-clock limit that aborts the process if it runs longer than the
/// configured duration. This is intended as a last-resort safeguard against
/// hangs.
void installResourceGuard();

} // namespace circt

#endif // CIRCT_SUPPORT_RESOURCEGUARD_H
