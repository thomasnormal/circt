//===- LowerTaggedIndirectCalls.cpp - Rewrite tagged vtable calls -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLVM IR pass that transforms indirect calls through tagged synthetic vtable
// addresses (0xF0000000+N) into lookups through the unified entry table
// @__circt_sim_func_entries[fid].
//
// The interpreter assigns vtable entries synthetic addresses of the form
// 0xF0000000+N. When AOT-compiled code loads these addresses and calls through
// them, the call would crash (they're not real pointers). This pass intercepts
// such calls at compile time by inserting a runtime check:
//
//   if (fp >= 0xF0000000 && fp < 0x100000000)
//     call @__circt_sim_func_entries[fp - 0xF0000000](args...)
//   else
//     call fp(args...)  // real function pointer, call directly
//
//===----------------------------------------------------------------------===//

#include "LowerTaggedIndirectCalls.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// Look up the @__circt_sim_func_entries global.
/// The definition with initializer is emitted by synthesizeDescriptor.
/// If it is missing, tagged-indirect lowering must be skipped to avoid
/// introducing unresolved references in modules that have no FuncId table.
static GlobalVariable *getFuncEntriesGlobalIfPresent(Module &M) {
  return M.getGlobalVariable("__circt_sim_func_entries");
}

/// Lower a single indirect call/invoke through the tagged dispatch check.
/// Returns true if the instruction was transformed.
static bool lowerIndirectCall(CallBase *CB, GlobalVariable *funcEntries) {
  Value *calledOp = CB->getCalledOperand();
  if (!calledOp)
    return false;

  // Skip direct calls (already have a known target).
  if (CB->getCalledFunction())
    return false;

  // Skip inline asm.
  if (isa<InlineAsm>(calledOp))
    return false;

  auto &ctx = CB->getContext();
  IRBuilder<> builder(CB);

  auto *i64Ty = Type::getInt64Ty(ctx);
  auto *i32Ty = Type::getInt32Ty(ctx);
  auto *ptrTy = PointerType::get(ctx, 0);

  // %fp_int = ptrtoint ptr %fp to i64
  Value *fpInt = builder.CreatePtrToInt(calledOp, i64Ty, "fp_int");

  // %is_tagged = icmp uge i64 %fp_int, 0xF0000000
  Value *isGe = builder.CreateICmpUGE(
      fpInt, ConstantInt::get(i64Ty, 0xF0000000ULL), "is_ge_tag");

  // %is_low = icmp ult i64 %fp_int, 0x100000000
  Value *isLt = builder.CreateICmpULT(
      fpInt, ConstantInt::get(i64Ty, 0x100000000ULL), "is_lt_4g");

  // %tagged = and i1 %is_tagged, %is_low
  Value *isTagged = builder.CreateAnd(isGe, isLt, "is_tagged");

  // Split the basic block.
  BasicBlock *origBB = CB->getParent();
  Function *F = origBB->getParent();

  // For invoke instructions, we need special handling due to the unwind dest.
  if (auto *II = dyn_cast<InvokeInst>(CB)) {
    BasicBlock *normalDest = II->getNormalDest();
    BasicBlock *unwindDest = II->getUnwindDest();

    BasicBlock *taggedBB =
        BasicBlock::Create(ctx, "tagged_call", F, normalDest);
    BasicBlock *directBB =
        BasicBlock::Create(ctx, "direct_call", F, normalDest);
    BasicBlock *mergeBB =
        BasicBlock::Create(ctx, "merge_call", F, normalDest);

    builder.CreateCondBr(isTagged, taggedBB, directBB);

    // Tagged path: decode fid, load from entry table, invoke.
    builder.SetInsertPoint(taggedBB);
    Value *fidI64 = builder.CreateSub(
        fpInt, ConstantInt::get(i64Ty, 0xF0000000ULL), "fid_i64");
    Value *fid = builder.CreateTrunc(fidI64, i32Ty, "fid");
    Value *entryGEP = builder.CreateInBoundsGEP(
        funcEntries->getValueType(), funcEntries,
        {ConstantInt::get(i64Ty, 0), fid}, "entry_ptr");
    Value *entry = builder.CreateLoad(ptrTy, entryGEP, "entry");

    SmallVector<Value *, 8> args(II->arg_begin(), II->arg_end());
    Twine taggedName = II->getType()->isVoidTy() ? Twine("") : Twine("tagged_ret");
    InvokeInst *taggedInvoke = builder.CreateInvoke(
        II->getFunctionType(), entry, mergeBB, unwindDest, args, taggedName);
    taggedInvoke->setCallingConv(II->getCallingConv());
    taggedInvoke->setAttributes(II->getAttributes());

    // Direct path: invoke original pointer.
    builder.SetInsertPoint(directBB);
    Twine directName = II->getType()->isVoidTy() ? Twine("") : Twine("direct_ret");
    InvokeInst *directInvoke = builder.CreateInvoke(
        II->getFunctionType(), calledOp, mergeBB, unwindDest, args,
        directName);
    directInvoke->setCallingConv(II->getCallingConv());
    directInvoke->setAttributes(II->getAttributes());

    // Merge: PHI for non-void returns.
    builder.SetInsertPoint(mergeBB, mergeBB->begin());
    if (!II->getType()->isVoidTy()) {
      PHINode *phi = builder.CreatePHI(II->getType(), 2, "call_ret");
      phi->addIncoming(taggedInvoke, taggedInvoke->getNormalDest() == mergeBB
                                         ? taggedBB
                                         : taggedInvoke->getParent());
      phi->addIncoming(directInvoke, directInvoke->getNormalDest() == mergeBB
                                         ? directBB
                                         : directInvoke->getParent());
      II->replaceAllUsesWith(phi);
    }
    // Branch from merge to original normal dest.
    builder.CreateBr(normalDest);

    // Fix PHI nodes in normalDest: origBB → mergeBB
    for (auto &Phi : normalDest->phis()) {
      int Idx = Phi.getBasicBlockIndex(origBB);
      if (Idx >= 0)
        Phi.setIncomingBlock(Idx, mergeBB);
    }

    // Fix PHI nodes in unwindDest: origBB → {taggedBB, directBB}
    for (auto &Phi : unwindDest->phis()) {
      int Idx = Phi.getBasicBlockIndex(origBB);
      if (Idx >= 0) {
        llvm::Value *Val = Phi.getIncomingValue(Idx);
        Phi.setIncomingBlock(Idx, taggedBB);
        Phi.addIncoming(Val, directBB);
      }
    }

    // Remove the original invoke.
    II->eraseFromParent();
    return true;
  }

  // Regular call instruction.
  auto *CI = cast<CallInst>(CB);

  // Split: origBB -> {taggedBB, directBB} -> mergeBB -> rest of origBB
  BasicBlock *restBB = origBB->splitBasicBlock(CI, "rest");

  // Remove the unconditional branch that splitBasicBlock created.
  origBB->getTerminator()->eraseFromParent();

  BasicBlock *taggedBB = BasicBlock::Create(ctx, "tagged_call", F, restBB);
  BasicBlock *directBB = BasicBlock::Create(ctx, "direct_call", F, restBB);
  BasicBlock *mergeBB = BasicBlock::Create(ctx, "merge_call", F, restBB);

  // Terminate origBB with conditional branch.
  builder.SetInsertPoint(origBB);
  builder.CreateCondBr(isTagged, taggedBB, directBB);

  // Tagged path.
  builder.SetInsertPoint(taggedBB);
  Value *fidI64 = builder.CreateSub(
      fpInt, ConstantInt::get(i64Ty, 0xF0000000ULL), "fid_i64");
  Value *fid = builder.CreateTrunc(fidI64, i32Ty, "fid");
  Value *entryGEP = builder.CreateInBoundsGEP(
      funcEntries->getValueType(), funcEntries,
      {ConstantInt::get(i64Ty, 0), fid}, "entry_ptr");
  Value *entry = builder.CreateLoad(ptrTy, entryGEP, "entry");

  SmallVector<Value *, 8> args(CI->arg_begin(), CI->arg_end());
  Twine taggedName = CI->getType()->isVoidTy() ? Twine("") : Twine("tagged_ret");
  CallInst *taggedCall =
      builder.CreateCall(CI->getFunctionType(), entry, args, taggedName);
  taggedCall->setCallingConv(CI->getCallingConv());
  taggedCall->setAttributes(CI->getAttributes());
  taggedCall->setTailCallKind(CI->getTailCallKind());
  builder.CreateBr(mergeBB);

  // Direct path.
  builder.SetInsertPoint(directBB);
  Twine directName = CI->getType()->isVoidTy() ? Twine("") : Twine("direct_ret");
  CallInst *directCall =
      builder.CreateCall(CI->getFunctionType(), calledOp, args, directName);
  directCall->setCallingConv(CI->getCallingConv());
  directCall->setAttributes(CI->getAttributes());
  directCall->setTailCallKind(CI->getTailCallKind());
  builder.CreateBr(mergeBB);

  // Merge.
  builder.SetInsertPoint(mergeBB);
  if (!CI->getType()->isVoidTy()) {
    PHINode *phi = builder.CreatePHI(CI->getType(), 2, "call_ret");
    phi->addIncoming(taggedCall, taggedBB);
    phi->addIncoming(directCall, directBB);
    CI->replaceAllUsesWith(phi);
  }
  builder.CreateBr(restBB);

  CI->eraseFromParent();
  return true;
}

void runLowerTaggedIndirectCalls(Module &M) {
  auto *funcEntries = getFuncEntriesGlobalIfPresent(M);
  if (!funcEntries || funcEntries->isDeclaration())
    return;

  // Collect all indirect calls first (we'll modify the IR as we go).
  SmallVector<CallBase *, 64> indirectCalls;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;
        if (CB->getCalledFunction())
          continue; // direct call
        if (isa<InlineAsm>(CB->getCalledOperand()))
          continue;
        indirectCalls.push_back(CB);
      }
    }
  }

  unsigned lowered = 0;
  for (auto *CB : indirectCalls) {
    if (lowerIndirectCall(CB, funcEntries))
      ++lowered;
  }

  if (lowered > 0) {
    errs() << "[circt-sim-compile] LowerTaggedIndirectCalls: lowered "
           << lowered << " indirect calls\n";
  }
}
