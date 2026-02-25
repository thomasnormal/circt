//===- LowerTaggedIndirectCalls.h - Tagged vtable call lowering --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SIM_COMPILE_LOWER_TAGGED_INDIRECT_CALLS_H
#define CIRCT_SIM_COMPILE_LOWER_TAGGED_INDIRECT_CALLS_H

namespace llvm {
class Module;
}

/// Rewrite indirect calls through tagged synthetic vtable addresses
/// (0xF0000000+N) into lookups through @__circt_sim_func_entries[fid].
void runLowerTaggedIndirectCalls(llvm::Module &M);

#endif // CIRCT_SIM_COMPILE_LOWER_TAGGED_INDIRECT_CALLS_H
