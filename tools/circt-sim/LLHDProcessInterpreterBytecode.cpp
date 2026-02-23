//===- LLHDProcessInterpreterBytecode.cpp - Bytecode process interpreter --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-compiled micro-op bytecode interpreter for LLHD processes. Walks process
// IR once at init time to build a flat array of MicroOps with pre-resolved
// signal IDs and integer virtual registers. Executes via a tight switch loop
// without MLIR op access, string lookups, DenseMap, or APInt.
//
// Target: 128-320x improvement over MLIR-walking interpreter for common RTL
// processes (64K → 200-500 instructions/delta cycle).
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "JITBlockCompiler.h"
#include "JITCompileManager.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <map>

#define DEBUG_TYPE "llhd-bytecode"

using namespace circt::sim;
using namespace circt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Bytecode compiler: IR → MicroOps
//===----------------------------------------------------------------------===//

namespace {

class BytecodeCompiler {
public:
  using SignalResolverFn = std::function<SignalId(Value)>;

  BytecodeCompiler(SignalResolverFn resolver, ProcessScheduler &sched)
      : resolveSignalId(std::move(resolver)), scheduler(sched) {}

  /// Try to compile a process to bytecode. Returns nullopt if the process
  /// contains unsupported operations.
  std::optional<BytecodeProgram> compile(llhd::ProcessOp processOp);

  /// Name of the first op that caused compilation failure (diagnostic).
  std::string failedOpName;

private:
  /// Assign a virtual register to an SSA value. Returns the register index.
  uint8_t getOrAssignReg(Value value) {
    auto it = valueToReg.find(value);
    if (it != valueToReg.end())
      return it->second;
    uint8_t reg = nextReg++;
    if (reg >= 255) { // Limit virtual registers (uint8_t max)
      tooManyRegs = true;
      return 0;
    }
    valueToReg[value] = reg;
    return reg;
  }

  /// Get the block index for a block.
  uint32_t getBlockIndex(Block *block) {
    auto it = blockToIndex.find(block);
    assert(it != blockToIndex.end() && "Block not indexed");
    return it->second;
  }

  /// Try to compile a single operation. Returns false if unsupported.
  bool compileOp(Operation *op, BytecodeProgram &program);

  /// Try to compile a probe operation.
  bool compileProbe(llhd::ProbeOp probeOp, BytecodeProgram &program);

  /// Try to compile a drive operation.
  bool compileDrive(llhd::DriveOp driveOp, BytecodeProgram &program);

  /// Emit parallel phi copies that are safe against register aliasing.
  /// When copies form cycles (e.g., swap: a→b, b→a), temporary registers
  /// are used to break the dependency.
  void emitParallelCopies(
      llvm::ArrayRef<std::pair<uint8_t, uint8_t>> copies,
      BytecodeProgram &program);

  /// Deferred auxiliary block for cf.cond_br with block arguments.
  /// Each aux block performs parallel phi copies then jumps to the real target.
  struct DeferredAuxBlock {
    llvm::SmallVector<std::pair<uint8_t, uint8_t>> copies;
    uint32_t targetBlockIndex;
  };

  SignalResolverFn resolveSignalId;
  ProcessScheduler &scheduler;
  llvm::DenseMap<Value, uint8_t> valueToReg;
  llvm::DenseMap<Block *, uint32_t> blockToIndex;
  llvm::SmallVector<DeferredAuxBlock> deferredAuxBlocks;
  uint32_t nextAuxBlockIndex = 0;
  uint8_t nextReg = 0;
  bool tooManyRegs = false;
};

void BytecodeCompiler::emitParallelCopies(
    llvm::ArrayRef<std::pair<uint8_t, uint8_t>> copies,
    BytecodeProgram &program) {
  // Filter out no-op copies (src == dest).
  llvm::SmallVector<std::pair<uint8_t, uint8_t>> filtered;
  for (auto [src, dest] : copies) {
    if (src != dest)
      filtered.push_back({src, dest});
  }
  if (filtered.empty())
    return;

  // Check if any copy's source register is also a destination of another copy.
  // If so, sequential execution would clobber values (the parallel copy problem).
  llvm::SmallDenseSet<uint8_t, 8> destRegs;
  for (auto [src, dest] : filtered)
    destRegs.insert(dest);
  bool hasAliasing = false;
  for (auto [src, dest] : filtered) {
    if (destRegs.count(src)) {
      hasAliasing = true;
      break;
    }
  }

  if (!hasAliasing) {
    // No aliasing: safe to emit direct copies.
    for (auto [src, dest] : filtered) {
      MicroOp copyOp;
      copyOp.kind = MicroOpKind::Trunci;
      copyOp.destReg = dest;
      copyOp.srcReg1 = src;
      copyOp.width = 64;
      program.ops.push_back(copyOp);
    }
    return;
  }

  // Aliasing detected: snapshot all sources into temps, then copy temps to
  // destinations. This breaks any read-after-write dependencies.
  llvm::SmallVector<uint8_t, 4> temps;
  for (auto [src, dest] : filtered) {
    uint8_t temp = nextReg++;
    if (nextReg >= 255) {
      tooManyRegs = true;
      return;
    }
    temps.push_back(temp);
    MicroOp copyOp;
    copyOp.kind = MicroOpKind::Trunci;
    copyOp.destReg = temp;
    copyOp.srcReg1 = src;
    copyOp.width = 64;
    program.ops.push_back(copyOp);
  }
  for (size_t i = 0; i < filtered.size(); ++i) {
    MicroOp copyOp;
    copyOp.kind = MicroOpKind::Trunci;
    copyOp.destReg = filtered[i].second;
    copyOp.srcReg1 = temps[i];
    copyOp.width = 64;
    program.ops.push_back(copyOp);
  }
}

bool BytecodeCompiler::compileProbe(llhd::ProbeOp probeOp,
                                     BytecodeProgram &program) {
  SignalId sigId = resolveSignalId(probeOp.getSignal());
  if (sigId == 0)
    return false;

  // Only support signals ≤64 bits wide.
  const auto &sigVal = scheduler.getSignalValue(sigId);
  if (sigVal.getWidth() > 64)
    return false;

  MicroOp op;
  op.kind = MicroOpKind::Probe;
  op.destReg = getOrAssignReg(probeOp.getResult());
  op.signalId = sigId;
  op.width = sigVal.getWidth();
  program.ops.push_back(op);
  return !tooManyRegs;
}

bool BytecodeCompiler::compileDrive(llhd::DriveOp driveOp,
                                     BytecodeProgram &program) {
  // Only support unconditional drives (no enable).
  if (driveOp.getEnable())
    return false;

  SignalId sigId = resolveSignalId(driveOp.getSignal());
  if (sigId == 0)
    return false;

  const auto &sigVal = scheduler.getSignalValue(sigId);
  if (sigVal.getWidth() > 64)
    return false;

  // Only support epsilon delay (0ns, 1d, 0e).
  // We'll check the delay value at runtime via the interpreter's
  // constant_time resolution, but for now just accept it.

  MicroOp op;
  op.kind = MicroOpKind::Drive;
  op.srcReg1 = getOrAssignReg(driveOp.getValue());
  op.signalId = sigId;
  op.width = sigVal.getWidth();
  program.ops.push_back(op);
  return !tooManyRegs;
}

bool BytecodeCompiler::compileOp(Operation *op, BytecodeProgram &program) {
  // hw.constant
  if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
    APInt val = constOp.getValue();
    if (val.getBitWidth() > 64)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::Const;
    mop.destReg = getOrAssignReg(constOp.getResult());
    mop.immediate = val.getZExtValue();
    mop.width = val.getBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // llhd.prb
  if (auto probeOp = dyn_cast<llhd::ProbeOp>(op))
    return compileProbe(probeOp, program);

  // llhd.drv
  if (auto driveOp = dyn_cast<llhd::DriveOp>(op))
    return compileDrive(driveOp, program);

  // llhd.constant_time — skip, delay is handled by the drive/wait logic.
  if (isa<llhd::ConstantTimeOp>(op))
    return true;

  // llhd.wait — record the wait and emit a Wait micro-op.
  if (auto waitOp = dyn_cast<llhd::WaitOp>(op)) {
    // Collect observed signals.
    for (Value observed : waitOp.getObserved()) {
      SignalId sigId = resolveSignalId(observed);
      if (sigId == 0) {
        // Try to trace through probes.
        if (auto probeOp = observed.getDefiningOp<llhd::ProbeOp>()) {
          sigId = resolveSignalId(probeOp.getSignal());
        }
      }
      if (sigId != 0) {
        program.waitSignals.push_back(sigId);
        program.waitEdges.push_back(EdgeType::AnyEdge);
      }
    }

    // Record which block the wait resumes to.
    if (waitOp.getDest()) {
      auto destIt = blockToIndex.find(waitOp.getDest());
      if (destIt != blockToIndex.end())
        program.resumeBlockIndex = destIt->second;
    }

    MicroOp mop;
    mop.kind = MicroOpKind::Wait;
    program.ops.push_back(mop);
    return true;
  }

  // llhd.halt
  if (isa<llhd::HaltOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Halt;
    program.ops.push_back(mop);
    return true;
  }

  // cf.br (unconditional branch)
  if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
    // Handle block arguments (phi values) with parallel copy semantics
    // to avoid register aliasing (e.g., swap: cf.br ^bb(%b, %a)).
    Block *dest = brOp.getDest();
    auto destArgs = dest->getArguments();
    auto branchOperands = brOp.getDestOperands();
    llvm::SmallVector<std::pair<uint8_t, uint8_t>, 4> copies;
    for (size_t i = 0; i < destArgs.size(); ++i) {
      uint8_t srcReg = getOrAssignReg(branchOperands[i]);
      uint8_t destReg = getOrAssignReg(destArgs[i]);
      copies.push_back({srcReg, destReg});
    }
    emitParallelCopies(copies, program);
    if (tooManyRegs)
      return false;

    MicroOp mop;
    mop.kind = MicroOpKind::Jump;
    mop.immediate = getBlockIndex(dest);
    program.ops.push_back(mop);
    return true;
  }

  // cf.cond_br (conditional branch)
  if (auto condBrOp = dyn_cast<cf::CondBranchOp>(op)) {
    bool hasTrueArgs = !condBrOp.getTrueDestOperands().empty();
    bool hasFalseArgs = !condBrOp.getFalseDestOperands().empty();

    if (!hasTrueArgs && !hasFalseArgs) {
      // No block arguments: simple BranchIf.
      MicroOp mop;
      mop.kind = MicroOpKind::BranchIf;
      mop.srcReg1 = getOrAssignReg(condBrOp.getCondition());
      mop.srcReg2 =
          static_cast<uint8_t>(getBlockIndex(condBrOp.getTrueDest()));
      mop.srcReg3 =
          static_cast<uint8_t>(getBlockIndex(condBrOp.getFalseDest()));
      program.ops.push_back(mop);
      return !tooManyRegs;
    }

    // Block arguments present: use auxiliary blocks to perform parallel phi
    // copies before jumping to the real destination. This avoids the register
    // aliasing problem where sequential copies clobber values needed by
    // subsequent copies (e.g., swap patterns).
    //
    // For each branch side that has operands, we create an auxiliary block:
    //   aux_block: parallel_copy(operands → dest_block_args); Jump dest_block
    // If a branch side has no operands, we target the real block directly.

    // Determine targets: real block index or deferred aux block index.
    uint8_t trueTarget, falseTarget;

    if (hasTrueArgs) {
      if (nextAuxBlockIndex >= 256)
        return false; // BranchIf targets are uint8_t
      trueTarget = static_cast<uint8_t>(nextAuxBlockIndex++);
      DeferredAuxBlock aux;
      auto destArgs = condBrOp.getTrueDest()->getArguments();
      auto operands = condBrOp.getTrueDestOperands();
      for (size_t i = 0; i < destArgs.size(); ++i)
        aux.copies.push_back(
            {getOrAssignReg(operands[i]), getOrAssignReg(destArgs[i])});
      aux.targetBlockIndex = getBlockIndex(condBrOp.getTrueDest());
      deferredAuxBlocks.push_back(std::move(aux));
    } else {
      trueTarget =
          static_cast<uint8_t>(getBlockIndex(condBrOp.getTrueDest()));
    }

    if (hasFalseArgs) {
      if (nextAuxBlockIndex >= 256)
        return false;
      falseTarget = static_cast<uint8_t>(nextAuxBlockIndex++);
      DeferredAuxBlock aux;
      auto destArgs = condBrOp.getFalseDest()->getArguments();
      auto operands = condBrOp.getFalseDestOperands();
      for (size_t i = 0; i < destArgs.size(); ++i)
        aux.copies.push_back(
            {getOrAssignReg(operands[i]), getOrAssignReg(destArgs[i])});
      aux.targetBlockIndex = getBlockIndex(condBrOp.getFalseDest());
      deferredAuxBlocks.push_back(std::move(aux));
    } else {
      falseTarget =
          static_cast<uint8_t>(getBlockIndex(condBrOp.getFalseDest()));
    }

    MicroOp mop;
    mop.kind = MicroOpKind::BranchIf;
    mop.srcReg1 = getOrAssignReg(condBrOp.getCondition());
    mop.srcReg2 = trueTarget;
    mop.srcReg3 = falseTarget;
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.add
  if (auto addOp = dyn_cast<comb::AddOp>(op)) {
    if (addOp.getInputs().size() != 2)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::Add;
    mop.destReg = getOrAssignReg(addOp.getResult());
    mop.srcReg1 = getOrAssignReg(addOp.getInputs()[0]);
    mop.srcReg2 = getOrAssignReg(addOp.getInputs()[1]);
    mop.width = addOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.sub
  if (auto subOp = dyn_cast<comb::SubOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Sub;
    mop.destReg = getOrAssignReg(subOp.getResult());
    mop.srcReg1 = getOrAssignReg(subOp.getLhs());
    mop.srcReg2 = getOrAssignReg(subOp.getRhs());
    mop.width = subOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.mul
  if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
    if (mulOp.getInputs().size() != 2)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::Mul;
    mop.destReg = getOrAssignReg(mulOp.getResult());
    mop.srcReg1 = getOrAssignReg(mulOp.getInputs()[0]);
    mop.srcReg2 = getOrAssignReg(mulOp.getInputs()[1]);
    mop.width = mulOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.shl
  if (auto shlOp = dyn_cast<comb::ShlOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Shl;
    mop.destReg = getOrAssignReg(shlOp.getResult());
    mop.srcReg1 = getOrAssignReg(shlOp.getLhs());
    mop.srcReg2 = getOrAssignReg(shlOp.getRhs());
    mop.width = shlOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.shru
  if (auto shrOp = dyn_cast<comb::ShrUOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Shr;
    mop.destReg = getOrAssignReg(shrOp.getResult());
    mop.srcReg1 = getOrAssignReg(shrOp.getLhs());
    mop.srcReg2 = getOrAssignReg(shrOp.getRhs());
    mop.width = shrOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.xor
  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    if (xorOp.getInputs().size() != 2)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::Xor;
    mop.destReg = getOrAssignReg(xorOp.getResult());
    mop.srcReg1 = getOrAssignReg(xorOp.getInputs()[0]);
    mop.srcReg2 = getOrAssignReg(xorOp.getInputs()[1]);
    mop.width = xorOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.and
  if (auto andOp = dyn_cast<comb::AndOp>(op)) {
    if (andOp.getInputs().size() != 2)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::And;
    mop.destReg = getOrAssignReg(andOp.getResult());
    mop.srcReg1 = getOrAssignReg(andOp.getInputs()[0]);
    mop.srcReg2 = getOrAssignReg(andOp.getInputs()[1]);
    mop.width = andOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.or
  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    if (orOp.getInputs().size() != 2)
      return false;
    MicroOp mop;
    mop.kind = MicroOpKind::Or;
    mop.destReg = getOrAssignReg(orOp.getResult());
    mop.srcReg1 = getOrAssignReg(orOp.getInputs()[0]);
    mop.srcReg2 = getOrAssignReg(orOp.getInputs()[1]);
    mop.width = orOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.icmp
  if (auto icmpOp = dyn_cast<comb::ICmpOp>(op)) {
    MicroOp mop;
    switch (icmpOp.getPredicate()) {
    case comb::ICmpPredicate::eq:
    case comb::ICmpPredicate::ceq:
    case comb::ICmpPredicate::weq:
      mop.kind = MicroOpKind::ICmpEq;
      break;
    case comb::ICmpPredicate::ne:
    case comb::ICmpPredicate::cne:
    case comb::ICmpPredicate::wne:
      mop.kind = MicroOpKind::ICmpNe;
      break;
    default:
      return false; // Unsupported predicate
    }
    mop.destReg = getOrAssignReg(icmpOp.getResult());
    mop.srcReg1 = getOrAssignReg(icmpOp.getLhs());
    mop.srcReg2 = getOrAssignReg(icmpOp.getRhs());
    mop.width = icmpOp.getLhs().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // comb.mux
  if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Mux;
    mop.destReg = getOrAssignReg(muxOp.getResult());
    mop.srcReg1 = getOrAssignReg(muxOp.getCond());
    mop.srcReg2 = getOrAssignReg(muxOp.getTrueValue());
    mop.srcReg3 = getOrAssignReg(muxOp.getFalseValue());
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // arith.trunci
  if (auto truncOp = dyn_cast<arith::TruncIOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Trunci;
    mop.destReg = getOrAssignReg(truncOp.getResult());
    mop.srcReg1 = getOrAssignReg(truncOp.getIn());
    mop.width = truncOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // arith.extui
  if (auto extOp = dyn_cast<arith::ExtUIOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Zext;
    mop.destReg = getOrAssignReg(extOp.getResult());
    mop.srcReg1 = getOrAssignReg(extOp.getIn());
    mop.width = extOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // arith.extsi
  if (auto extOp = dyn_cast<arith::ExtSIOp>(op)) {
    MicroOp mop;
    mop.kind = MicroOpKind::Sext;
    mop.destReg = getOrAssignReg(extOp.getResult());
    mop.srcReg1 = getOrAssignReg(extOp.getIn());
    // Store source width in srcReg2 and dest width in width
    unsigned srcWidth = extOp.getIn().getType().getIntOrFloatBitWidth();
    mop.srcReg2 = static_cast<uint8_t>(srcWidth);
    mop.width = extOp.getResult().getType().getIntOrFloatBitWidth();
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // llhd.int_to_time — skip, consumed by wait delay logic.
  if (isa<llhd::IntToTimeOp>(op))
    return true;

  // hw.struct_extract — extract a field from a struct via shift + mask.
  if (auto extractOp = dyn_cast<hw::StructExtractOp>(op)) {
    auto structTy =
        hw::type_cast<hw::StructType>(extractOp.getInput().getType());
    unsigned totalWidth = hw::getBitWidth(structTy);
    if (totalWidth > 64 || totalWidth == 0)
      return false;
    StringRef fieldName = extractOp.getFieldName();
    unsigned fieldOffset = 0;
    unsigned fieldWidth = 0;
    for (auto &field : structTy.getElements()) {
      unsigned fw = hw::getBitWidth(field.type);
      if (field.name == fieldName) {
        fieldWidth = fw;
        break;
      }
      fieldOffset += fw;
    }
    if (fieldWidth == 0 || fieldWidth > 64)
      return false;
    // HW convention: fields MSB-first, so shift = total - offset - width.
    unsigned shiftAmount = totalWidth - fieldOffset - fieldWidth;
    MicroOp mop;
    mop.kind = MicroOpKind::StructExtract;
    mop.destReg = getOrAssignReg(extractOp.getResult());
    mop.srcReg1 = getOrAssignReg(extractOp.getInput());
    mop.width = static_cast<uint8_t>(fieldWidth);
    mop.immediate = shiftAmount;
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // hw.struct_create — pack fields into a struct (2-field structs only).
  if (auto createOp = dyn_cast<hw::StructCreateOp>(op)) {
    auto structTy = hw::type_cast<hw::StructType>(createOp.getType());
    auto fields = structTy.getElements();
    if (fields.size() != 2)
      return false; // Only support 2-field structs (FourState pattern).
    unsigned totalWidth = hw::getBitWidth(structTy);
    if (totalWidth > 64 || totalWidth == 0)
      return false;
    unsigned field1Width = hw::getBitWidth(fields[1].type);
    if (field1Width > 64)
      return false;
    // HW: field0 at MSB, field1 at LSB.
    // dest = (field0 << field1Width) | field1
    MicroOp mop;
    mop.kind = MicroOpKind::StructCreate2;
    mop.destReg = getOrAssignReg(createOp.getResult());
    mop.srcReg1 = getOrAssignReg(createOp.getInput()[0]);
    mop.srcReg2 = getOrAssignReg(createOp.getInput()[1]);
    mop.width = static_cast<uint8_t>(field1Width);
    mop.immediate = totalWidth;
    program.ops.push_back(mop);
    return !tooManyRegs;
  }

  // Unsupported operation.
  LLVM_DEBUG(llvm::dbgs() << "[Bytecode] Unsupported op: " << op->getName()
                          << "\n");
  failedOpName = op->getName().getStringRef().str();
  return false;
}

std::optional<BytecodeProgram>
BytecodeCompiler::compile(llhd::ProcessOp processOp) {
  BytecodeProgram program;
  Region &body = processOp.getBody();

  // Index all blocks.
  uint32_t blockIdx = 0;
  for (Block &block : body) {
    blockToIndex[&block] = blockIdx++;
  }
  // Auxiliary blocks (for cf.cond_br phi copies) start after MLIR blocks.
  nextAuxBlockIndex = blockIdx;

  // Compile each block.
  for (Block &block : body) {
    uint32_t blockIndex = blockToIndex[&block];
    program.blockOffsets.push_back(program.ops.size());

    // Assign registers to block arguments.
    for (auto arg : block.getArguments()) {
      getOrAssignReg(arg);
    }

    // Compile each operation in the block.
    for (Operation &op : block) {
      if (!compileOp(&op, program)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Bytecode] Failed to compile process: unsupported op "
                   << op.getName() << " in block " << blockIndex << "\n");
        return std::nullopt;
      }
      if (tooManyRegs) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Bytecode] Too many virtual registers (>128)\n");
        return std::nullopt;
      }
    }
  }

  // Emit deferred auxiliary blocks for cf.cond_br with block arguments.
  // Each aux block does: parallel phi copies → Jump to real target.
  for (auto &aux : deferredAuxBlocks) {
    program.blockOffsets.push_back(program.ops.size());
    emitParallelCopies(aux.copies, program);
    if (tooManyRegs) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Bytecode] Too many regs in aux block phi copies\n");
      return std::nullopt;
    }
    MicroOp jumpOp;
    jumpOp.kind = MicroOpKind::Jump;
    jumpOp.immediate = aux.targetBlockIndex;
    program.ops.push_back(jumpOp);
  }

  // Sentinel: end of last block.
  program.blockOffsets.push_back(program.ops.size());
  program.numRegs = nextReg;

  // Find the wait block (the block containing a Wait micro-op).
  for (uint32_t bi = 0; bi < program.blockOffsets.size() - 1; ++bi) {
    uint32_t start = program.blockOffsets[bi];
    uint32_t end = program.blockOffsets[bi + 1];
    for (uint32_t i = start; i < end; ++i) {
      if (program.ops[i].kind == MicroOpKind::Wait) {
        program.waitBlockIndex = bi;
        break;
      }
    }
  }

  // Validate: must have at least one wait and some signals to observe.
  if (program.waitSignals.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Bytecode] No wait signals found, rejecting\n");
    return std::nullopt;
  }

  program.valid = true;

  LLVM_DEBUG({
    llvm::dbgs() << "[Bytecode] Compiled process: " << program.ops.size()
                 << " micro-ops, " << (int)program.numRegs << " regs, "
                 << program.waitSignals.size() << " wait signals, "
                 << (program.blockOffsets.size() - 1) << " blocks\n";
  });

  return program;
}

} // namespace

//===----------------------------------------------------------------------===//
// Bytecode executor
//===----------------------------------------------------------------------===//

namespace {

/// A pending drive from bytecode execution, batched for a single Event.
struct PendingDrive {
  uint32_t signalId;
  uint32_t width;
  uint64_t value;
};

/// Execute a compiled bytecode program starting from the given block.
/// Returns true if the process suspended (hit Wait), false if it halted or
/// encountered an error.
bool executeBytecodeProgram(const BytecodeProgram &program,
                            ProcessScheduler &scheduler,
                            uint32_t startBlock) {
  uint64_t regs[256];
  std::memset(regs, 0, sizeof(regs));

  // Collect drives to batch into a single Event at the end.
  llvm::SmallVector<PendingDrive, 4> pendingDrives;

  uint32_t currentBlock = startBlock;

  for (;;) {
    if (currentBlock >= program.blockOffsets.size() - 1)
      return false; // Invalid block

    uint32_t opStart = program.blockOffsets[currentBlock];
    uint32_t opEnd = program.blockOffsets[currentBlock + 1];

    for (uint32_t ip = opStart; ip < opEnd; ++ip) {
      const MicroOp &op = program.ops[ip];

      switch (op.kind) {
      case MicroOpKind::Probe: {
        const SignalValue &sv = scheduler.getSignalValue(op.signalId);
        if (sv.isUnknown())
          regs[op.destReg] = 0; // Treat X as 0 for bytecode
        else
          regs[op.destReg] = sv.getValue();
        break;
      }

      case MicroOpKind::Drive: {
        uint64_t val = regs[op.srcReg1];
        // Mask to signal width.
        if (op.width < 64)
          val &= (1ULL << op.width) - 1;
        pendingDrives.push_back({op.signalId, op.width, val});
        break;
      }

      case MicroOpKind::Const:
        regs[op.destReg] = op.immediate;
        break;

      case MicroOpKind::Add: {
        uint64_t result = regs[op.srcReg1] + regs[op.srcReg2];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::Sub: {
        uint64_t result = regs[op.srcReg1] - regs[op.srcReg2];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::And:
        regs[op.destReg] = regs[op.srcReg1] & regs[op.srcReg2];
        break;

      case MicroOpKind::Or:
        regs[op.destReg] = regs[op.srcReg1] | regs[op.srcReg2];
        break;

      case MicroOpKind::Xor:
        regs[op.destReg] = regs[op.srcReg1] ^ regs[op.srcReg2];
        break;

      case MicroOpKind::Shl: {
        uint64_t result = regs[op.srcReg1] << regs[op.srcReg2];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::Shr:
        regs[op.destReg] = regs[op.srcReg1] >> regs[op.srcReg2];
        break;

      case MicroOpKind::Mul: {
        uint64_t result = regs[op.srcReg1] * regs[op.srcReg2];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::ICmpEq:
        regs[op.destReg] = (regs[op.srcReg1] == regs[op.srcReg2]) ? 1 : 0;
        break;

      case MicroOpKind::ICmpNe:
        regs[op.destReg] = (regs[op.srcReg1] != regs[op.srcReg2]) ? 1 : 0;
        break;

      case MicroOpKind::Not: {
        uint64_t result = ~regs[op.srcReg1];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::Trunci: {
        uint64_t result = regs[op.srcReg1];
        if (op.width > 0 && op.width < 64)
          result &= (1ULL << op.width) - 1;
        regs[op.destReg] = result;
        break;
      }

      case MicroOpKind::Zext:
        // Already zero-extended as uint64_t.
        regs[op.destReg] = regs[op.srcReg1];
        break;

      case MicroOpKind::Sext: {
        uint64_t val = regs[op.srcReg1];
        uint8_t srcWidth = op.srcReg2; // Source width stored in srcReg2
        if (srcWidth > 0 && srcWidth < 64) {
          uint64_t signBit = 1ULL << (srcWidth - 1);
          if (val & signBit) {
            // Sign extend: fill upper bits with 1s
            uint64_t mask = ~((1ULL << srcWidth) - 1);
            val |= mask;
          }
          if (op.width > 0 && op.width < 64)
            val &= (1ULL << op.width) - 1;
        }
        regs[op.destReg] = val;
        break;
      }

      case MicroOpKind::Mux:
        regs[op.destReg] =
            regs[op.srcReg1] ? regs[op.srcReg2] : regs[op.srcReg3];
        break;

      case MicroOpKind::StructExtract: {
        // Extract field: shift right by immediate, mask to width bits.
        uint64_t mask =
            (op.width >= 64) ? ~0ULL : ((1ULL << op.width) - 1);
        regs[op.destReg] = (regs[op.srcReg1] >> op.immediate) & mask;
        break;
      }

      case MicroOpKind::StructCreate2: {
        // Pack 2 fields: field0 at MSB (shifted left by width), field1 at LSB.
        // width = field1Width (shift amount for field0).
        // immediate = totalWidth (for masking the result).
        uint64_t field0Mask =
            (op.immediate - op.width >= 64)
                ? ~0ULL
                : ((1ULL << (op.immediate - op.width)) - 1);
        uint64_t field1Mask =
            (op.width >= 64) ? ~0ULL : ((1ULL << op.width) - 1);
        regs[op.destReg] = ((regs[op.srcReg1] & field0Mask) << op.width) |
                           (regs[op.srcReg2] & field1Mask);
        break;
      }

      case MicroOpKind::Jump:
        currentBlock = static_cast<uint32_t>(op.immediate);
        goto next_block;

      case MicroOpKind::BranchIf:
        currentBlock = regs[op.srcReg1] ? op.srcReg2 : op.srcReg3;
        goto next_block;

      case MicroOpKind::Wait: {
        // Queue all pending drives via the fast path (no Event overhead).
        for (const auto &d : pendingDrives)
          scheduler.queueSignalUpdateFast(d.signalId, d.value, d.width);
        return true; // Suspended
      }

      case MicroOpKind::Halt: {
        for (const auto &d : pendingDrives)
          scheduler.queueSignalUpdateFast(d.signalId, d.value, d.width);
        return false; // Process terminated
      }
      }
    }

    // Fell through block without a terminator — should not happen.
    return false;

  next_block:
    continue;
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Integration with LLHDProcessInterpreter
//===----------------------------------------------------------------------===//

/// Print bytecode compilation statistics. Called before _exit().
void LLHDProcessInterpreter::printBytecodeStats() const {
  if (bytecodeAttempted == 0)
    return;
  llvm::errs() << "\n[Bytecode Stats] " << bytecodeCompiled << "/"
               << bytecodeAttempted << " processes compiled to bytecode\n";
  if (!bytecodeFailedOps.empty()) {
    llvm::errs() << "[Bytecode Stats] Ops causing fallback:\n";
    std::vector<std::pair<std::string, unsigned>> sorted(
        bytecodeFailedOps.begin(), bytecodeFailedOps.end());
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    for (auto &[name, count] : sorted)
      llvm::errs() << "  " << name << ": " << count << "\n";
  }
}

void LLHDProcessInterpreter::printCompileReport() const {
  uint64_t total =
      activationsAOTCallback + activationsBytecode + activationsInterpreter;
  uint64_t divisor = total > 0 ? total : 1;

  llvm::errs() << "\n=== Compile Coverage Report ===\n";
  llvm::errs() << "Process activations by dispatch path:\n";
  llvm::errs() << "  AOT callback:  " << activationsAOTCallback << " ("
               << llvm::format("%.1f", 100.0 * activationsAOTCallback / divisor)
               << "%)\n";
  llvm::errs() << "  Bytecode:      " << activationsBytecode << " ("
               << llvm::format("%.1f", 100.0 * activationsBytecode / divisor)
               << "%)\n";
  llvm::errs() << "  Interpreter:   " << activationsInterpreter << " ("
               << llvm::format("%.1f", 100.0 * activationsInterpreter / divisor)
               << "%)\n";
  llvm::errs() << "  Total:         " << total << "\n";

  // ExecModel breakdown: count processes per model.
  llvm::DenseMap<ExecModel, unsigned> modelCounts;
  for (auto &[procId, model] : processExecModels)
    modelCounts[model]++;
  if (!modelCounts.empty()) {
    llvm::errs() << "ExecModel breakdown (registered processes):\n";
    auto modelName = [](ExecModel m) -> llvm::StringRef {
      switch (m) {
      case ExecModel::CallbackStaticObserved:  return "CallbackStaticObserved";
      case ExecModel::CallbackDynamicWait:     return "CallbackDynamicWait";
      case ExecModel::CallbackTimeOnly:        return "CallbackTimeOnly";
      case ExecModel::OneShotCallback:         return "OneShotCallback";
      case ExecModel::Coroutine:               return "Coroutine";
      }
      return "Unknown";
    };
    for (auto m : {ExecModel::CallbackStaticObserved, ExecModel::CallbackDynamicWait,
                   ExecModel::CallbackTimeOnly, ExecModel::OneShotCallback,
                   ExecModel::Coroutine}) {
      unsigned cnt = modelCounts.lookup(m);
      if (cnt > 0)
        llvm::errs() << "  " << modelName(m) << ": " << cnt << "\n";
    }
  }

  // AOT rejection reasons (populated during AOT compilation if aotEnabled).
  if (aotCompiler && !aotCompiler->rejectionStats.empty()) {
    llvm::errs() << "AOT rejection reasons:\n";
    std::vector<std::pair<std::string, unsigned>> sorted;
    for (auto &entry : aotCompiler->rejectionStats)
      sorted.emplace_back(entry.getKey().str(), entry.getValue());
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    for (auto &[name, count] : sorted)
      llvm::errs() << "  " << name << ": " << count << "\n";
  }

  llvm::errs() << "===============================\n";
}

/// Try to compile a process to bytecode and store the result.
bool LLHDProcessInterpreter::tryCompileProcessBytecode(
    ProcessId procId, ProcessExecutionState &state) {
  auto processOp = state.getProcessOp();
  if (!processOp)
    return false;

  ++bytecodeAttempted;

  BytecodeCompiler compiler(
      [this](Value v) { return resolveSignalId(v); }, scheduler);
  auto program = compiler.compile(processOp);
  if (!program) {
    if (!compiler.failedOpName.empty())
      bytecodeFailedOps[compiler.failedOpName]++;
    else
      bytecodeFailedOps["(unknown/too-many-regs)"]++;
    return false;
  }

  ++bytecodeCompiled;

  // Store the compiled program.
  auto prog = std::make_unique<BytecodeProgram>(std::move(*program));
  bytecodeProgramMap[procId] = std::move(prog);

  LLVM_DEBUG(llvm::dbgs() << "[Bytecode] Installed bytecode for process "
                          << procId << "\n");
  return true;
}

/// Execute a process using its compiled bytecode program.
/// Returns true if execution succeeded (process suspended or halted).
bool LLHDProcessInterpreter::executeBytecodeProcess(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  auto progIt = bytecodeProgramMap.find(procId);
  if (progIt == bytecodeProgramMap.end() || !progIt->second ||
      !progIt->second->valid) {
    thunkState.deoptRequested = true;
    return false;
  }

  const BytecodeProgram &program = *progIt->second;

  // Determine start block: on first activation (token 0), start from
  // block 0 (entry). On subsequent activations (token >= 1), start from
  // the resume block (after the wait op).
  uint32_t startBlock = 0;
  if (thunkState.resumeToken >= 1) {
    startBlock = program.resumeBlockIndex;
  }

  bool suspended = executeBytecodeProgram(program, scheduler, startBlock);

  if (suspended) {
    if (thunkState.resumeToken == 0) {
      // First activation: do full sensitivity registration.
      SensitivityList waitList;
      for (size_t i = 0; i < program.waitSignals.size(); ++i) {
        waitList.addEdge(program.waitSignals[i], program.waitEdges[i]);
      }
      scheduler.suspendProcessForEvents(procId, waitList);

      // Initialize the cached wait state for the process skip optimization.
      state.lastWaitHadDelay = false;
      state.lastWaitHasEdge = false;
      state.lastSensitivityEntries = waitList.getEntries();
      state.lastSensitivityValid = true;
      state.lastSensitivityValues.resize(state.lastSensitivityEntries.size());
      for (size_t i = 0; i < state.lastSensitivityEntries.size(); ++i) {
        state.lastSensitivityValues[i] =
            scheduler.getSignalValue(state.lastSensitivityEntries[i].signalId);
      }
    } else {
      // Subsequent activations: just set state back to Waiting.
      // Sensitivity list and signal registrations are unchanged.
      scheduler.resuspendProcessFast(procId);
    }

    // Advance resume token so next activation starts from resume block.
    thunkState.resumeToken = std::max(thunkState.resumeToken, uint64_t{1});
  } else {
    // Process halted.
    state.halted = true;
  }

  return true;
}

