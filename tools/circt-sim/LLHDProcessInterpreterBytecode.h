//===- LLHDProcessInterpreterBytecode.h - Bytecode types ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the bytecode process interpreter: MicroOpKind, MicroOp, and
// BytecodeProgram. Shared between the header (for unique_ptr completeness)
// and the implementation file.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERBYTECODE_H
#define CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERBYTECODE_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace circt {
namespace sim {

enum class MicroOpKind : uint8_t {
  // Signal I/O
  Probe,  // destReg = signalValue[signalId]
  Drive,  // schedule signalUpdate(signalId, regs[srcReg1])

  // Constants
  Const, // destReg = immediate

  // Arithmetic/logic (comb dialect)
  Add,
  Sub,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  Mul,
  ICmpEq,
  ICmpNe,

  // Control flow
  Jump,     // goto targetBlock
  BranchIf, // if regs[srcReg1] goto trueBlock else goto falseBlock

  // Process control
  Wait, // suspend process, re-enqueue on signal changes
  Halt, // terminate process

  // Misc
  Not,    // destReg = ~regs[srcReg1] (truncated to width)
  Trunci, // destReg = regs[srcReg1] & ((1<<width)-1)
  Zext,   // destReg = regs[srcReg1] (already zero-extended as uint64)
  Sext,   // destReg = sign-extend regs[srcReg1] from srcWidth to dstWidth
  Mux,    // destReg = regs[srcReg1] ? regs[srcReg2] : regs[srcReg3]
};

struct MicroOp {
  MicroOpKind kind;
  uint8_t destReg = 0;   // destination virtual register
  uint8_t srcReg1 = 0;   // source register 1
  uint8_t srcReg2 = 0;   // source register 2 (or true-block for BranchIf)
  uint8_t srcReg3 = 0;   // source register 3 (or false-block for BranchIf)
  uint8_t width = 0;     // bit width for truncation/extension
  uint16_t padding = 0;
  uint32_t signalId = 0; // for Probe/Drive
  uint64_t immediate = 0; // for Const, or targetBlock for Jump
};

/// A compiled bytecode program for a single LLHD process.
struct BytecodeProgram {
  /// The micro-ops, organized by block. blockOffsets[i] gives the index
  /// of the first op in block i. blockOffsets[numBlocks] = ops.size().
  llvm::SmallVector<MicroOp, 32> ops;
  llvm::SmallVector<uint32_t, 8> blockOffsets;

  /// Number of virtual registers needed.
  uint8_t numRegs = 0;

  /// Sensitivity list for the wait op (pre-resolved signal IDs).
  llvm::SmallVector<SignalId, 4> waitSignals;
  llvm::SmallVector<EdgeType, 4> waitEdges;

  /// The block index that contains the wait op (loop head).
  uint32_t waitBlockIndex = 0;

  /// The block index to resume at after wait (destination of wait).
  uint32_t resumeBlockIndex = 0;

  /// Whether the program is valid and can be executed.
  bool valid = false;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETERBYTECODE_H
