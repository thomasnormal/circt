//===- circt-sim-compile.cpp - AOT compiler for circt-sim --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ahead-of-time compiler that takes MLIR input and produces a shared object
// (.so) loadable by circt-sim at runtime. The .so exports a single
// CirctSimCompiledModule descriptor via circt_sim_get_compiled_module().
//
// Pipeline:
//   1. Parse MLIR (bytecode or text)
//   2. Identify compilable func.func bodies
//   3. Lower arith/cf/func → LLVM dialect (in-place)
//   4. Synthesize the CirctSimCompiledModule descriptor as LLVM IR globals
//   5. Translate LLVM dialect → LLVM IR
//   6. Compile LLVM IR → .o via TargetMachine
//   7. Link .o → .so via clang -shared
//
//===----------------------------------------------------------------------===//

#include "LowerTaggedIndirectCalls.h"
#include "circt/Runtime/CirctSimABI.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/FourStateUtils.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include <chrono>
#include <cctype>
#include <cstdlib>

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                  llvm::cl::desc("<input>"),
                                                  llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::opt<int>
    optLevel("O", llvm::cl::desc("Optimization level (0-3)"),
             llvm::cl::init(1));

static llvm::cl::opt<bool>
    emitLLVM("emit-llvm", llvm::cl::desc("Emit LLVM IR instead of .so"),
             llvm::cl::init(false));

static llvm::cl::opt<bool>
    emitObject("emit-obj", llvm::cl::desc("Emit .o instead of .so"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    verbose("v", llvm::cl::desc("Verbose output"), llvm::cl::init(false));

static llvm::cl::opt<bool> emitSnapshot(
    "emit-snapshot",
    llvm::cl::desc("Emit a snapshot directory (design.mlirbc + native.so + "
                   "meta.json) instead of a bare .so"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Lowering: func compilability check
//===----------------------------------------------------------------------===//

static Value unwrapPointerFromRefCast(Value value);
static Type convertToLLVMCompatibleType(Type type, MLIRContext *ctx);
static bool isLLVMCompatibleRefValue(Value value, MLIRContext *ctx);

/// Check whether a func.func body contains only ops that can be lowered to
/// LLVM dialect. Only arith/cf/func/LLVM ops are accepted. Aggregate types
/// in signatures are allowed — they will be flattened by
/// flattenAggregateFunctionABIs() after lowering to LLVM dialect.
static bool isFuncBodyCompilable(func::FuncOp funcOp,
                                  std::string *rejectionReason = nullptr) {
  if (funcOp.isExternal()) {
    if (rejectionReason)
      *rejectionReason = "external_declaration";
    return false;
  }

  bool compilable = true;
  funcOp.walk([&](Operation *op) {
    if (isa<arith::ArithDialect, cf::ControlFlowDialect, LLVM::LLVMDialect,
            func::FuncDialect, scf::SCFDialect>(op->getDialect()))
      return WalkResult::advance();
    // Allow specific ops from other dialects that we can lower.
    if (isa<hw::ConstantOp, hw::BitcastOp, comb::ExtractOp, comb::AndOp,
            comb::OrOp, comb::XorOp, comb::ICmpOp, comb::AddOp, comb::SubOp,
            comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp,
            comb::MulOp, comb::MuxOp, comb::ConcatOp,
            comb::ReplicateOp,
            comb::ShlOp, comb::ShrUOp, comb::ShrSOp,
            hw::StructCreateOp, hw::StructExtractOp,
            hw::AggregateConstantOp>(op))
      return WalkResult::advance();
    if (auto sigOp = dyn_cast<llhd::SignalOp>(op)) {
      auto refTy = dyn_cast<llhd::RefType>(sigOp.getResult().getType());
      Type nestedTy = refTy ? refTy.getNestedType() : Type();
      Type llvmTy = nestedTy ? convertToLLVMCompatibleType(nestedTy, funcOp.getContext())
                             : Type();
      if (llvmTy && LLVM::isCompatibleType(llvmTy))
        return WalkResult::advance();
      if (rejectionReason)
        *rejectionReason = "llhd.sig";
      compilable = false;
      return WalkResult::interrupt();
    }
    if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(op)) {
      bool allArrayGetUsers = true;
      for (Operation *user : arrayCreateOp->getUsers()) {
        auto getOp = dyn_cast<hw::ArrayGetOp>(user);
        if (!getOp || getOp.getInput() != arrayCreateOp.getResult()) {
          allArrayGetUsers = false;
          break;
        }
      }
      if (allArrayGetUsers)
        return WalkResult::advance();
    }
    if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(op)) {
      if (arrayGetOp.getInput().getDefiningOp<hw::ArrayCreateOp>() &&
          isa<IntegerType>(arrayGetOp.getIndex().getType()) &&
          LLVM::isCompatibleType(arrayGetOp.getResult().getType()))
        return WalkResult::advance();
    }
    if (auto currentTimeOp = dyn_cast<llhd::CurrentTimeOp>(op)) {
      bool allResultsDead = true;
      for (Value result : currentTimeOp->getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        return WalkResult::advance();
      bool onlyTimeToIntUsers = true;
      for (Operation *user : currentTimeOp->getUsers())
        onlyTimeToIntUsers &= isa<llhd::TimeToIntOp>(user);
      if (onlyTimeToIntUsers)
        return WalkResult::advance();
    }
    if (auto constTimeOp = dyn_cast<llhd::ConstantTimeOp>(op)) {
      bool allResultsDead = true;
      for (Value result : constTimeOp->getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        return WalkResult::advance();
      bool onlyTimeToIntUsers = true;
      for (Operation *user : constTimeOp->getUsers())
        onlyTimeToIntUsers &= isa<llhd::TimeToIntOp>(user);
      if (onlyTimeToIntUsers)
        return WalkResult::advance();
      bool onlyDriveUsers = true;
      for (Operation *user : constTimeOp->getUsers())
        onlyDriveUsers &= isa<llhd::DriveOp>(user);
      if (onlyDriveUsers)
        return WalkResult::advance();
    }
    if (isa<llhd::DriveOp>(op))
      return WalkResult::advance();
    if (auto sigExtractOp = dyn_cast<llhd::SigExtractOp>(op)) {
      auto rejectSigExtract = [&]() {
        if (rejectionReason)
          *rejectionReason = "llhd.sig.extract";
        compilable = false;
        return WalkResult::interrupt();
      };
      bool supportedInput = false;
      if (unwrapPointerFromRefCast(sigExtractOp.getInput())) {
        supportedInput = true;
      } else if (auto sigStructExtract =
                     sigExtractOp.getInput().getDefiningOp<llhd::SigStructExtractOp>()) {
        if (unwrapPointerFromRefCast(sigStructExtract.getInput())) {
          supportedInput = true;
        } else if (auto sigOp =
                       sigStructExtract.getInput().getDefiningOp<llhd::SignalOp>()) {
          auto sigRefTy = dyn_cast<llhd::RefType>(sigOp.getResult().getType());
          Type sigNestedTy = sigRefTy ? sigRefTy.getNestedType() : Type();
          Type sigLLVMTy =
              sigNestedTy
                  ? convertToLLVMCompatibleType(sigNestedTy, funcOp.getContext())
                  : Type();
          supportedInput = sigLLVMTy && LLVM::isCompatibleType(sigLLVMTy);
        } else if (isLLVMCompatibleRefValue(sigStructExtract.getInput(),
                                            funcOp.getContext())) {
          supportedInput = true;
        }
      }
      if (!supportedInput)
        return rejectSigExtract();
      auto inRefTy = dyn_cast<llhd::RefType>(sigExtractOp.getInput().getType());
      auto outRefTy = dyn_cast<llhd::RefType>(sigExtractOp.getResult().getType());
      auto inIntTy = inRefTy ? dyn_cast<IntegerType>(inRefTy.getNestedType())
                             : IntegerType();
      auto outIntTy = outRefTy ? dyn_cast<IntegerType>(outRefTy.getNestedType())
                               : IntegerType();
      if (!inIntTy || !outIntTy)
        return rejectSigExtract();
      bool allSupportedUsers = true;
      for (Operation *user : sigExtractOp->getUsers()) {
        auto drvOp = dyn_cast<llhd::DriveOp>(user);
        if (!drvOp || drvOp.getSignal() != sigExtractOp.getResult() ||
            drvOp.getEnable()) {
          allSupportedUsers = false;
          break;
        }
        auto delayConst = drvOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
        if (!delayConst) {
          allSupportedUsers = false;
          break;
        }
        auto delayAttr = dyn_cast<llhd::TimeAttr>(delayConst.getValueAttr());
        if (!delayAttr || delayAttr.getTime() != 0 || delayAttr.getDelta() != 0 ||
            delayAttr.getEpsilon() > 1) {
          allSupportedUsers = false;
          break;
        }
        auto valueIntTy = dyn_cast<IntegerType>(drvOp.getValue().getType());
        if (!valueIntTy || valueIntTy.getWidth() != outIntTy.getWidth()) {
          allSupportedUsers = false;
          break;
        }
      }
      if (allSupportedUsers)
        return WalkResult::advance();
      return rejectSigExtract();
    }
    if (auto sigStructExtractOp = dyn_cast<llhd::SigStructExtractOp>(op)) {
      auto rejectSigStructExtract = [&]() {
        if (rejectionReason)
          *rejectionReason = "llhd.sig.struct_extract";
        compilable = false;
        return WalkResult::interrupt();
      };
      Value basePtr = unwrapPointerFromRefCast(sigStructExtractOp.getInput());
      if (!basePtr) {
        auto sigOp =
            sigStructExtractOp.getInput().getDefiningOp<llhd::SignalOp>();
        bool compatibleRefInput = false;
        if (sigOp) {
          auto sigRefTy = dyn_cast<llhd::RefType>(sigOp.getResult().getType());
          Type sigNestedTy = sigRefTy ? sigRefTy.getNestedType() : Type();
          Type sigLLVMTy = sigNestedTy
                               ? convertToLLVMCompatibleType(sigNestedTy,
                                                             funcOp.getContext())
                               : Type();
          compatibleRefInput = sigLLVMTy && LLVM::isCompatibleType(sigLLVMTy);
        } else {
          compatibleRefInput =
              isLLVMCompatibleRefValue(sigStructExtractOp.getInput(),
                                       funcOp.getContext());
        }
        if (!compatibleRefInput)
          return rejectSigStructExtract();
      }
      auto inRefTy =
          dyn_cast<llhd::RefType>(sigStructExtractOp.getInput().getType());
      auto outRefTy =
          dyn_cast<llhd::RefType>(sigStructExtractOp.getResult().getType());
      auto inStructTy =
          inRefTy ? dyn_cast<hw::StructType>(inRefTy.getNestedType())
                  : hw::StructType();
      auto outIntTy =
          outRefTy ? dyn_cast<IntegerType>(outRefTy.getNestedType())
                   : IntegerType();
      if (!inStructTy || !outIntTy)
        return rejectSigStructExtract();
      auto fieldTy =
          dyn_cast_or_null<IntegerType>(inStructTy.getFieldType(
              sigStructExtractOp.getFieldAttr()));
      if (!fieldTy || fieldTy.getWidth() != outIntTy.getWidth())
        return rejectSigStructExtract();
      bool allSupportedUsers = true;
      for (Operation *user : sigStructExtractOp->getUsers()) {
        if (auto sigExtract = dyn_cast<llhd::SigExtractOp>(user)) {
          if (sigExtract.getInput() != sigStructExtractOp.getResult()) {
            allSupportedUsers = false;
            break;
          }
          auto nestedOutRefTy =
              dyn_cast<llhd::RefType>(sigExtract.getResult().getType());
          auto nestedOutIntTy = nestedOutRefTy
                                    ? dyn_cast<IntegerType>(
                                          nestedOutRefTy.getNestedType())
                                    : IntegerType();
          if (!nestedOutIntTy) {
            allSupportedUsers = false;
            break;
          }
          for (Operation *nestedUser : sigExtract->getUsers()) {
            auto drvOp = dyn_cast<llhd::DriveOp>(nestedUser);
            if (!drvOp || drvOp.getSignal() != sigExtract.getResult() ||
                drvOp.getEnable()) {
              allSupportedUsers = false;
              break;
            }
            auto delayConst =
                drvOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
            if (!delayConst) {
              allSupportedUsers = false;
              break;
            }
            auto delayAttr = dyn_cast<llhd::TimeAttr>(delayConst.getValueAttr());
            if (!delayAttr || delayAttr.getTime() != 0 ||
                delayAttr.getDelta() != 0 || delayAttr.getEpsilon() > 1) {
              allSupportedUsers = false;
              break;
            }
            auto valueIntTy = dyn_cast<IntegerType>(drvOp.getValue().getType());
            if (!valueIntTy ||
                valueIntTy.getWidth() != nestedOutIntTy.getWidth()) {
              allSupportedUsers = false;
              break;
            }
          }
          if (!allSupportedUsers)
            break;
          continue;
        }
        if (auto prbOp = dyn_cast<llhd::ProbeOp>(user)) {
          if (prbOp.getSignal() != sigStructExtractOp.getResult() ||
              !isa<IntegerType>(prbOp.getResult().getType())) {
            allSupportedUsers = false;
            break;
          }
          continue;
        }
        auto drvOp = dyn_cast<llhd::DriveOp>(user);
        if (!drvOp || drvOp.getSignal() != sigStructExtractOp.getResult() ||
            drvOp.getEnable()) {
          allSupportedUsers = false;
          break;
        }
        auto delayConst = drvOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
        if (!delayConst) {
          allSupportedUsers = false;
          break;
        }
        auto delayAttr = dyn_cast<llhd::TimeAttr>(delayConst.getValueAttr());
        if (!delayAttr || delayAttr.getTime() != 0 || delayAttr.getDelta() != 0 ||
            delayAttr.getEpsilon() > 1) {
          allSupportedUsers = false;
          break;
        }
        auto valueIntTy = dyn_cast<IntegerType>(drvOp.getValue().getType());
        if (!valueIntTy || valueIntTy.getWidth() != outIntTy.getWidth()) {
          allSupportedUsers = false;
          break;
        }
      }
      if (allSupportedUsers)
        return WalkResult::advance();
      return rejectSigStructExtract();
    }
    if (auto prbOp = dyn_cast<llhd::ProbeOp>(op)) {
      Value signalPtr = prbOp.getSignal();
      if (!isa<LLVM::LLVMPointerType>(signalPtr.getType())) {
        signalPtr = unwrapPointerFromRefCast(signalPtr);
        if (!signalPtr) {
          if (auto sigOp = prbOp.getSignal().getDefiningOp<llhd::SignalOp>()) {
            auto sigRefTy = dyn_cast<llhd::RefType>(sigOp.getResult().getType());
            Type sigNestedTy = sigRefTy ? sigRefTy.getNestedType() : Type();
            Type sigLLVMTy =
                sigNestedTy ? convertToLLVMCompatibleType(sigNestedTy,
                                                          funcOp.getContext())
                            : Type();
            if (!sigLLVMTy || !LLVM::isCompatibleType(sigLLVMTy)) {
              if (rejectionReason)
                *rejectionReason = "llhd.prb";
              compilable = false;
              return WalkResult::interrupt();
            }
          } else
          if (auto sigStructExtract =
                  prbOp.getSignal().getDefiningOp<llhd::SigStructExtractOp>()) {
            auto inRefTy =
                dyn_cast<llhd::RefType>(sigStructExtract.getInput().getType());
            auto outRefTy =
                dyn_cast<llhd::RefType>(sigStructExtract.getResult().getType());
            auto inStructTy =
                inRefTy ? dyn_cast<hw::StructType>(inRefTy.getNestedType())
                        : hw::StructType();
            auto outIntTy =
                outRefTy ? dyn_cast<IntegerType>(outRefTy.getNestedType())
                         : IntegerType();
            Value basePtr = unwrapPointerFromRefCast(sigStructExtract.getInput());
            auto fieldTy = inStructTy ? dyn_cast_or_null<IntegerType>(
                                            inStructTy.getFieldType(
                                                sigStructExtract.getFieldAttr()))
                                      : IntegerType();
            bool compatibleRefInput =
                basePtr ||
                isLLVMCompatibleRefValue(sigStructExtract.getInput(),
                                         funcOp.getContext());
            if (!compatibleRefInput || !inStructTy || !outIntTy || !fieldTy ||
                fieldTy.getWidth() != outIntTy.getWidth()) {
              if (rejectionReason)
                *rejectionReason = "llhd.prb";
              compilable = false;
              return WalkResult::interrupt();
            }
          } else {
            if (rejectionReason)
              *rejectionReason = "llhd.prb";
            compilable = false;
            return WalkResult::interrupt();
          }
        }
      }
      Type prbLLVMTy =
          convertToLLVMCompatibleType(prbOp.getResult().getType(),
                                      funcOp.getContext());
      if (!prbLLVMTy || !LLVM::isCompatibleType(prbLLVMTy)) {
        if (rejectionReason)
          *rejectionReason = "llhd.prb";
        compilable = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto timeToIntOp = dyn_cast<llhd::TimeToIntOp>(op)) {
      Value input = timeToIntOp.getOperand();
      if (isa<IntegerType>(input.getType()))
        return WalkResult::advance();
      if (auto *defOp = input.getDefiningOp();
          defOp && isa<llhd::CurrentTimeOp, llhd::ConstantTimeOp,
                        llhd::IntToTimeOp>(defOp))
        return WalkResult::advance();
    }
    if (auto intToTimeOp = dyn_cast<llhd::IntToTimeOp>(op)) {
      bool allResultsDead = true;
      for (Value result : intToTimeOp->getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        return WalkResult::advance();
      bool onlyTimeToIntUsers = true;
      for (Operation *user : intToTimeOp->getUsers())
        onlyTimeToIntUsers &= isa<llhd::TimeToIntOp>(user);
      if (onlyTimeToIntUsers)
        return WalkResult::advance();
    }
    // Allow unrealized_conversion_cast only when types match (foldable).
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      bool allResultsDead = true;
      for (Value result : castOp.getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        return WalkResult::advance();
      if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
        Type inTy = castOp.getOperand(0).getType();
        Type outTy = castOp.getResult(0).getType();
        if (inTy == outTy)
          return WalkResult::advance();
        // Accept ptr -> function casts used by func.call_indirect lowering.
        if (isa<LLVM::LLVMPointerType>(inTy) && isa<FunctionType>(outTy))
          return WalkResult::advance();
        // Accept llvm.struct <-> hw.struct casts when field extraction
        // patterns can be folded before LLVM translation.
        if ((isa<LLVM::LLVMStructType>(inTy) && isa<hw::StructType>(outTy)) ||
            (isa<hw::StructType>(inTy) && isa<LLVM::LLVMStructType>(outTy)))
          return WalkResult::advance();
        // Accept llhd.ref <-> ptr casts; micro-module ABI canonicalization
        // rewrites function ref arguments to !llvm.ptr and lets these fold.
        if ((isa<LLVM::LLVMPointerType>(inTy) && isa<llhd::RefType>(outTy)) ||
            (isa<llhd::RefType>(inTy) && isa<LLVM::LLVMPointerType>(outTy)))
          return WalkResult::advance();
        // Accept simple pointer/integer cast pairs.
        if ((isa<LLVM::LLVMPointerType>(inTy) && isa<IntegerType>(outTy)) ||
            (isa<IntegerType>(inTy) && isa<LLVM::LLVMPointerType>(outTy)) ||
            (isa<LLVM::LLVMPointerType>(inTy) &&
             isa<LLVM::LLVMPointerType>(outTy)) ||
            (isa<IntegerType>(inTy) && isa<IntegerType>(outTy)))
          return WalkResult::advance();
        if (rejectionReason) {
          std::string inTyStr, outTyStr;
          llvm::raw_string_ostream inOS(inTyStr), outOS(outTyStr);
          inTy.print(inOS);
          outTy.print(outOS);
          *rejectionReason =
              ("builtin.unrealized_conversion_cast:" + inOS.str() + "->" +
               outOS.str());
        }
        compilable = false;
        return WalkResult::interrupt();
      }
      if (rejectionReason)
        *rejectionReason = "builtin.unrealized_conversion_cast:unsupported_arity";
      compilable = false;
      return WalkResult::interrupt();
    }
    if (auto ciOp = dyn_cast<func::CallIndirectOp>(op)) {
      // Accept call_indirect when callee is function-typed and result count is
      // LLVM-call compatible. The lowering pass converts this to llvm.call.
      if (isa<FunctionType>(ciOp.getCallee().getType()) &&
          ciOp.getNumResults() <= 1)
        return WalkResult::advance();
    }
    if (rejectionReason)
      *rejectionReason = op->getName().getStringRef().str();
    compilable = false;
    return WalkResult::interrupt();
  });
  return compilable;
}

static bool isModuleInitSkippableOp(Operation *op) {
  return isa<llhd::ProcessOp, seq::InitialOp, llhd::CombinationalOp,
             llhd::SignalOp, hw::InstanceOp, hw::OutputOp, llhd::DriveOp,
             llhd::ConstantTimeOp, llhd::CurrentTimeOp, sim::FormatLiteralOp,
             sim::FormatHexOp, sim::FormatDecOp, sim::FormatBinOp,
             sim::FormatOctOp, sim::FormatCharOp, sim::FormatGeneralOp,
             sim::FormatFloatOp, sim::FormatScientificOp,
             sim::FormatStringConcatOp, sim::FormatDynStringOp>(op);
}

static bool isNativeModuleInitAllowedCall(LLVM::CallOp callOp) {
  auto callee = callOp.getCallee();
  if (!callee)
    return false;
  llvm::StringRef name = *callee;
  return name == "memset" || name == "memcpy" || name == "memmove" ||
         name == "malloc" || name == "calloc" ||
         name.starts_with("llvm.memset.") ||
         name.starts_with("llvm.memcpy.") || name.starts_with("llvm.memmove.");
}

static bool isNativeModuleInitOp(Operation *op) {
  if (auto callOp = dyn_cast<LLVM::CallOp>(op))
    return isNativeModuleInitAllowedCall(callOp);
  // Keep this intentionally conservative in the first step: only memory and
  // constant/aggregate pointer plumbing ops that are known to be self-contained
  // and safe to run before interpreter dispatch wiring.
  return isa<LLVM::AllocaOp, LLVM::StoreOp, LLVM::ConstantOp,
             arith::ConstantOp, hw::ConstantOp, hw::AggregateConstantOp,
             LLVM::UndefOp, LLVM::ZeroOp, LLVM::AddressOfOp, LLVM::LoadOp,
             LLVM::InsertValueOp, LLVM::ExtractValueOp, LLVM::GEPOp,
             llhd::ProbeOp, UnrealizedConversionCastOp>(op);
}

/// Return true when a module-level llhd.prb from a hw.module block argument
/// can be lowered to a native-init runtime helper call.
static bool isSupportedNativeModuleInitBlockArgProbe(llhd::ProbeOp probeOp,
                                                     Block &moduleBody) {
  auto blockArg = dyn_cast<BlockArgument>(probeOp.getSignal());
  if (!blockArg || blockArg.getOwner() != &moduleBody)
    return false;

  Type resultTy = probeOp.getResult().getType();
  if (auto intTy = dyn_cast<IntegerType>(resultTy))
    return intTy.getWidth() <= 64;
  if (isa<LLVM::LLVMPointerType>(resultTy))
    return true;
  return false;
}

/// Return true when llhd.prb directly probes a skipped module-level llhd.sig
/// whose value is still the signal init operand (no module-level mutations).
static bool isSupportedNativeModuleInitSignalProbe(llhd::ProbeOp probeOp,
                                                   Block &moduleBody) {
  auto signalOp = probeOp.getSignal().getDefiningOp<llhd::SignalOp>();
  if (!signalOp || signalOp->getBlock() != &moduleBody)
    return false;

  // Keep this conservative: allow probe-only module-level reads plus
  // non-mutating connectivity uses through hw.instance operands.
  for (Operation *user : signalOp->getUsers()) {
    if (user->getBlock() != &moduleBody)
      continue;
    if (!isa<llhd::ProbeOp, hw::InstanceOp>(user))
      return false;
  }
  return true;
}

static std::string encodeModuleInitSymbol(llvm::StringRef moduleName) {
  std::string out = "__circt_sim_module_init__";
  out.reserve(out.size() + moduleName.size() * 3);
  static constexpr char hex[] = "0123456789ABCDEF";
  for (unsigned char c : moduleName.bytes()) {
    if (std::isalnum(c) || c == '_') {
      out.push_back(static_cast<char>(c));
      continue;
    }
    out.push_back('_');
    out.push_back(hex[(c >> 4) & 0xF]);
    out.push_back(hex[c & 0xF]);
  }
  return out;
}

struct NativeModuleInitSynthesisStats {
  unsigned totalModules = 0;
  unsigned emittedModules = 0;
  llvm::StringMap<unsigned> skipReasons;
};

static uint64_t toFemtoseconds(llhd::TimeAttr timeAttr) {
  uint64_t realFs = timeAttr.getTime();
  llvm::StringRef unit = timeAttr.getTimeUnit();
  if (unit == "ps")
    realFs *= 1000;
  else if (unit == "ns")
    realFs *= 1000000;
  else if (unit == "us")
    realFs *= 1000000000ULL;
  else if (unit == "ms")
    realFs *= 1000000000000ULL;
  else if (unit == "s")
    realFs *= 1000000000000000ULL;
  return realFs;
}

static Value unwrapPointerFromRefCast(Value value) {
  if (isa<LLVM::LLVMPointerType>(value.getType()))
    return value;
  auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!castOp || castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
    return {};
  Value src = castOp.getOperand(0);
  if (isa<LLVM::LLVMPointerType>(src.getType()))
    return src;
  return {};
}

static Type convertToLLVMCompatibleType(Type type, MLIRContext *ctx) {
  if (isa<IntegerType, LLVM::LLVMPointerType, LLVM::LLVMStructType,
          LLVM::LLVMArrayType>(type))
    return type;
  if (auto hwStructTy = dyn_cast<hw::StructType>(type)) {
    SmallVector<Type> fieldTypes;
    fieldTypes.reserve(hwStructTy.getElements().size());
    for (auto field : hwStructTy.getElements()) {
      Type converted = convertToLLVMCompatibleType(field.type, ctx);
      if (!converted || !LLVM::isCompatibleType(converted))
        return {};
      fieldTypes.push_back(converted);
    }
    return LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
  }
  if (auto hwArrayTy = dyn_cast<hw::ArrayType>(type)) {
    Type convertedElem =
        convertToLLVMCompatibleType(hwArrayTy.getElementType(), ctx);
    if (!convertedElem || !LLVM::isCompatibleType(convertedElem))
      return {};
    return LLVM::LLVMArrayType::get(convertedElem, hwArrayTy.getNumElements());
  }
  return {};
}

static bool isLLVMCompatibleRefValue(Value value, MLIRContext *ctx) {
  auto refTy = dyn_cast<llhd::RefType>(value.getType());
  if (!refTy)
    return false;
  Type llvmTy = convertToLLVMCompatibleType(refTy.getNestedType(), ctx);
  return llvmTy && LLVM::isCompatibleType(llvmTy);
}

/// Canonicalize func.func argument ABIs by replacing !llhd.ref<...> arguments
/// with !llvm.ptr in the compilable micro-module. This removes common
/// ptr<->ref cast wrappers around helper calls.
static void canonicalizeLLHDRefArgumentABIs(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);

  struct FuncRewriteInfo {
    FunctionType newType;
  };
  llvm::StringMap<FuncRewriteInfo> rewriteMap;

  moduleOp.walk([&](func::FuncOp funcOp) {
    auto oldType = funcOp.getFunctionType();
    bool changed = false;
    SmallVector<Type> newInputs;
    newInputs.reserve(oldType.getNumInputs());
    for (Type inputTy : oldType.getInputs()) {
      if (isa<llhd::RefType>(inputTy)) {
        newInputs.push_back(ptrTy);
        changed = true;
      } else {
        newInputs.push_back(inputTy);
      }
    }
    if (!changed)
      return;
    auto newType = FunctionType::get(ctx, newInputs, oldType.getResults());
    rewriteMap[funcOp.getSymName()] = {newType};
  });

  if (rewriteMap.empty())
    return;

  auto requiresRefOperand = [&](OpOperand &use) -> bool {
    Operation *owner = use.getOwner();
    unsigned operandNo = use.getOperandNumber();
    if (isa<llhd::ProbeOp>(owner))
      return operandNo == 0;
    if (isa<llhd::DriveOp>(owner))
      return operandNo == 0;
    if (isa<llhd::SigExtractOp>(owner))
      return operandNo == 0;
    if (isa<llhd::SigStructExtractOp>(owner))
      return operandNo == 0;
    if (isa<llhd::SigArraySliceOp>(owner))
      return operandNo == 0;
    return false;
  };

  // Rewrite function signatures and entry block argument types.
  for (auto &kv : rewriteMap) {
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(kv.first());
    if (!funcOp)
      continue;
    auto oldType = funcOp.getFunctionType();
    auto newType = kv.second.newType;
    funcOp.setFunctionType(newType);
    if (funcOp.isExternal())
      continue;
    Block &entry = funcOp.getBody().front();
    for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
      Type oldInputTy = oldType.getInput(i);
      Type newInputTy = newType.getInput(i);
      if (entry.getArgument(i).getType() != newInputTy)
        entry.getArgument(i).setType(newInputTy);
      if (!isa<llhd::RefType>(oldInputTy) || !isa<LLVM::LLVMPointerType>(newInputTy))
        continue;
      BlockArgument arg = entry.getArgument(i);
      SmallVector<OpOperand *> llhdRefUses;
      for (OpOperand &use : arg.getUses())
        if (requiresRefOperand(use))
          llhdRefUses.push_back(&use);
      if (llhdRefUses.empty())
        continue;
      OpBuilder entryBuilder = OpBuilder::atBlockBegin(&entry);
      auto refCast = entryBuilder.create<UnrealizedConversionCastOp>(
          funcOp.getLoc(), TypeRange{oldInputTy}, ValueRange{arg});
      Value refView = refCast.getResult(0);
      for (OpOperand *use : llhdRefUses)
        use->set(refView);
    }
  }

  // Rewrite direct call sites to match updated signatures.
  SmallVector<func::CallOp> callsToRewrite;
  moduleOp.walk([&](func::CallOp callOp) {
    if (rewriteMap.count(callOp.getCallee()))
      callsToRewrite.push_back(callOp);
  });

  for (auto callOp : callsToRewrite) {
    auto it = rewriteMap.find(callOp.getCallee());
    if (it == rewriteMap.end())
      continue;
    auto newType = it->second.newType;

    bool needsRewrite = false;
    for (unsigned i = 0, e = callOp.getNumOperands(); i < e; ++i) {
      if (callOp.getOperand(i).getType() != newType.getInput(i)) {
        needsRewrite = true;
        break;
      }
    }
    if (!needsRewrite)
      continue;

    OpBuilder builder(callOp);
    SmallVector<Value> newOperands;
    newOperands.reserve(callOp.getNumOperands());
    for (unsigned i = 0, e = callOp.getNumOperands(); i < e; ++i) {
      Value operand = callOp.getOperand(i);
      Type targetTy = newType.getInput(i);
      if (operand.getType() == targetTy) {
        newOperands.push_back(operand);
        continue;
      }
      if (isa<LLVM::LLVMPointerType>(targetTy)) {
        if (Value ptr = unwrapPointerFromRefCast(operand)) {
          newOperands.push_back(ptr);
          continue;
        }
      }
      auto bridgeCast = builder.create<UnrealizedConversionCastOp>(
          callOp.getLoc(), TypeRange{targetTy}, ValueRange{operand});
      newOperands.push_back(bridgeCast.getResult(0));
    }

    auto newCall = builder.create<func::CallOp>(callOp.getLoc(),
                                                callOp.getCallee(),
                                                callOp.getResultTypes(),
                                                newOperands);
    callOp.replaceAllUsesWith(newCall.getResults());
    callOp.erase();
  }

  // Rewrite call_indirect sites whose function type still carries !llhd.ref
  // argument types. This keeps verifier invariants after operand rewriting.
  SmallVector<func::CallIndirectOp> indirectCallsToRewrite;
  moduleOp.walk([&](func::CallIndirectOp callOp) {
    auto calleeTy = dyn_cast<FunctionType>(callOp.getCallee().getType());
    if (!calleeTy)
      return;
    bool needsRewrite = false;
    for (Type inputTy : calleeTy.getInputs())
      needsRewrite |= isa<llhd::RefType>(inputTy);
    if (needsRewrite)
      indirectCallsToRewrite.push_back(callOp);
  });

  for (auto callOp : indirectCallsToRewrite) {
    auto oldTy = dyn_cast<FunctionType>(callOp.getCallee().getType());
    if (!oldTy)
      continue;

    SmallVector<Type> newInputs;
    bool changed = false;
    newInputs.reserve(oldTy.getNumInputs());
    for (Type inputTy : oldTy.getInputs()) {
      if (isa<llhd::RefType>(inputTy)) {
        newInputs.push_back(ptrTy);
        changed = true;
      } else {
        newInputs.push_back(inputTy);
      }
    }
    if (!changed)
      continue;

    auto newTy = FunctionType::get(ctx, newInputs, oldTy.getResults());
    OpBuilder builder(callOp);

    Value oldCallee = callOp.getCallee();
    Value newCallee = oldCallee;
    if (oldCallee.getType() != newTy) {
      if (auto oldCast = oldCallee.getDefiningOp<UnrealizedConversionCastOp>();
          oldCast && oldCast.getNumOperands() == 1 && oldCast.getNumResults() == 1 &&
          isa<LLVM::LLVMPointerType>(oldCast.getOperand(0).getType())) {
        auto newCast = builder.create<UnrealizedConversionCastOp>(
            callOp.getLoc(), TypeRange{newTy},
            ValueRange{oldCast.getOperand(0)});
        newCallee = newCast.getResult(0);
      } else {
        auto newCast = builder.create<UnrealizedConversionCastOp>(
            callOp.getLoc(), TypeRange{newTy}, ValueRange{oldCallee});
        newCallee = newCast.getResult(0);
      }
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(callOp.getArgOperands().size());
    for (unsigned i = 0, e = callOp.getArgOperands().size(); i < e; ++i) {
      Value operand = callOp.getArgOperands()[i];
      Type targetTy = newTy.getInput(i);
      if (operand.getType() == targetTy) {
        newOperands.push_back(operand);
        continue;
      }
      if (isa<LLVM::LLVMPointerType>(targetTy)) {
        if (Value ptr = unwrapPointerFromRefCast(operand)) {
          newOperands.push_back(ptr);
          continue;
        }
      }
      auto bridgeCast = builder.create<UnrealizedConversionCastOp>(
          callOp.getLoc(), TypeRange{targetTy}, ValueRange{operand});
      newOperands.push_back(bridgeCast.getResult(0));
    }

    auto newCall = builder.create<func::CallIndirectOp>(
        callOp.getLoc(), newCallee, ValueRange{newOperands});
    callOp.replaceAllUsesWith(newCall.getResults());
    callOp.erase();
  }
}

/// Synthesize optional native module-init functions from hw.module top-level
/// LLVM-style init ops. Returns synthesis stats including skip reasons.
static NativeModuleInitSynthesisStats
synthesizeNativeModuleInitFunctions(ModuleOp sourceModule,
                                    ModuleOp microModule) {
  NativeModuleInitSynthesisStats stats;
  OpBuilder moduleBuilder(microModule.getContext());
  constexpr llvm::StringLiteral kProbePortHelperName =
      "__circt_sim_module_init_probe_port_raw";

  auto ensureProbePortHelperDecl = [&](Location loc) -> func::FuncOp {
    if (auto existing =
            microModule.lookupSymbol<func::FuncOp>(kProbePortHelperName))
      return existing;
    moduleBuilder.setInsertionPointToEnd(microModule.getBody());
    auto helperTy = moduleBuilder.getFunctionType(
        {moduleBuilder.getI64Type()}, {moduleBuilder.getI64Type()});
    auto decl = func::FuncOp::create(moduleBuilder, loc, kProbePortHelperName,
                                     helperTy);
    decl.setPrivate();
    return decl;
  };

  for (auto hwModule : sourceModule.getOps<hw::HWModuleOp>()) {
    ++stats.totalModules;
    Block &body = hwModule.getBody().front();
    llvm::SmallVector<Operation *> opsToClone;
    bool unsupported = false;
    std::string skipReason;

    for (Operation &op : body.getOperations()) {
      Operation *opPtr = &op;
      if (isModuleInitSkippableOp(opPtr))
        continue;
      if (!isNativeModuleInitOp(opPtr)) {
        if (auto callOp = dyn_cast<LLVM::CallOp>(opPtr)) {
          if (auto callee = callOp.getCallee())
            skipReason = ("unsupported_call:" + callee->str());
          else
            skipReason = "unsupported_call:indirect";
        } else {
          skipReason = ("unsupported_op:" +
                        opPtr->getName().getStringRef().str());
        }
        unsupported = true;
        break;
      }
      if (opPtr->getNumRegions() != 0) {
        skipReason = ("op_has_region:" + opPtr->getName().getStringRef().str());
        unsupported = true;
        break;
      }

      // Reject dependencies on hw.module block arguments or skipped ops.
      for (Value operand : opPtr->getOperands()) {
        if (isa<BlockArgument>(operand)) {
          if (auto probeOp = dyn_cast<llhd::ProbeOp>(opPtr)) {
            if (operand == probeOp.getSignal() &&
                isSupportedNativeModuleInitBlockArgProbe(probeOp, body))
              continue;
          }
          skipReason = ("operand_block_arg:" +
                        opPtr->getName().getStringRef().str());
          unsupported = true;
          break;
        }
        if (Operation *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == &body) {
            if (isModuleInitSkippableOp(defOp)) {
              if (auto probeOp = dyn_cast<llhd::ProbeOp>(opPtr)) {
                if (defOp == probeOp.getSignal().getDefiningOp() &&
                    isSupportedNativeModuleInitSignalProbe(probeOp, body))
                  continue;
              }
              skipReason = ("operand_dep_skipped:" +
                            defOp->getName().getStringRef().str());
              unsupported = true;
              break;
            }
            if (!isNativeModuleInitOp(defOp)) {
              skipReason = ("operand_dep_unsupported:" +
                            defOp->getName().getStringRef().str());
              unsupported = true;
              break;
            }
          }
        }
      }
      if (unsupported)
        break;
      opsToClone.push_back(opPtr);
    }

    if (unsupported || opsToClone.empty()) {
      if (unsupported) {
        if (skipReason.empty())
          skipReason = "unsupported:unknown";
        ++stats.skipReasons[skipReason];
      }
      continue;
    }

    std::string symName = encodeModuleInitSymbol(hwModule.getName());
    moduleBuilder.setInsertionPointToEnd(microModule.getBody());
    auto initType = moduleBuilder.getFunctionType({}, {});
    auto initFunc =
        func::FuncOp::create(moduleBuilder, hwModule.getLoc(), symName, initType);
    Block *entry = initFunc.addEntryBlock();
    OpBuilder initBuilder(entry, entry->begin());
    IRMapping mapping;
    bool cloneUnsupported = false;
    std::string cloneSkipReason;
    for (Operation *op : opsToClone) {
      if (auto probeOp = dyn_cast<llhd::ProbeOp>(op)) {
        auto blockArg = dyn_cast<BlockArgument>(probeOp.getSignal());
        if (blockArg && blockArg.getOwner() == &body) {
          if (!isSupportedNativeModuleInitBlockArgProbe(probeOp, body)) {
            cloneUnsupported = true;
            cloneSkipReason = "operand_block_arg:llhd.prb";
            break;
          }

          auto helperDecl =
              ensureProbePortHelperDecl(probeOp.getLoc());
          Value portIdx = arith::ConstantOp::create(
              initBuilder, probeOp.getLoc(),
              initBuilder.getI64IntegerAttr(blockArg.getArgNumber()));
          auto helperCall = initBuilder.create<func::CallOp>(
              probeOp.getLoc(), helperDecl.getName(),
              TypeRange{initBuilder.getI64Type()}, ValueRange{portIdx});
          Value rawVal = helperCall.getResult(0);
          Value mappedResult = rawVal;
          Type resultTy = probeOp.getResult().getType();
          if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
            if (intTy.getWidth() < 64) {
              mappedResult = arith::TruncIOp::create(initBuilder, probeOp.getLoc(),
                                                     intTy, rawVal);
            } else if (intTy.getWidth() > 64) {
              mappedResult = arith::ExtUIOp::create(initBuilder, probeOp.getLoc(),
                                                    intTy, rawVal);
            }
          } else if (isa<LLVM::LLVMPointerType>(resultTy)) {
            auto cast = initBuilder.create<UnrealizedConversionCastOp>(
                probeOp.getLoc(), TypeRange{resultTy}, ValueRange{rawVal});
            mappedResult = cast.getResult(0);
          } else {
            cloneUnsupported = true;
            cloneSkipReason = "operand_block_arg:llhd.prb";
            break;
          }
          mapping.map(probeOp.getResult(), mappedResult);
          continue;
        }
        if (isSupportedNativeModuleInitSignalProbe(probeOp, body)) {
          auto signalOp = probeOp.getSignal().getDefiningOp<llhd::SignalOp>();
          Value signalInit = signalOp.getInit();
          Value mappedInit = signalInit;
          if (mapping.contains(signalInit))
            mappedInit = mapping.lookup(signalInit);
          if (mappedInit.getType() != probeOp.getResult().getType()) {
            auto cast = initBuilder.create<UnrealizedConversionCastOp>(
                probeOp.getLoc(), TypeRange{probeOp.getResult().getType()},
                ValueRange{mappedInit});
            mappedInit = cast.getResult(0);
          }
          mapping.map(probeOp.getResult(), mappedInit);
          continue;
        }
      }
      initBuilder.clone(*op, mapping);
    }
    if (cloneUnsupported) {
      if (cloneSkipReason.empty())
        cloneSkipReason = "unsupported:unknown";
      ++stats.skipReasons[cloneSkipReason];
      initFunc.erase();
      continue;
    }
    initBuilder.create<func::ReturnOp>(hwModule.getLoc());
    ++stats.emittedModules;
  }

  return stats;
}

/// Lower SCF control-flow ops (if/for/while/parallel/execute_region/...) to
/// cf.* before custom arith/cf/func→LLVM rewriting.
///
/// Transactional semantics: conversion runs on a clone and only commits on
/// success. On failure, the original module remains unchanged so later passes
/// can safely continue with residual-op stripping.
static bool lowerSCFToCF(ModuleOp &microModule) {
  ModuleOp converted = cast<ModuleOp>(microModule->clone());
  PassManager pm(microModule.getContext());
  pm.addNestedPass<func::FuncOp>(createSCFToControlFlowPass());
  if (failed(pm.run(converted))) {
    converted.erase();
    return false;
  }
  microModule->erase();
  microModule = converted;
  return true;
}

//===----------------------------------------------------------------------===//
// Lowering: clone referenced declarations into micro-module
//===----------------------------------------------------------------------===//

static void cloneReferencedDeclarations(ModuleOp microModule,
                                        ModuleOp parentModule,
                                        IRMapping &mapping) {
  llvm::DenseSet<llvm::StringRef> clonedSymbols;
  bool changed = true;

  while (changed) {
    changed = false;
    llvm::SmallVector<llvm::StringRef, 8> needed;

    microModule.walk([&](Operation *op) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
        if (auto callee = callOp.getCallee())
          if (!clonedSymbols.contains(*callee) &&
              !microModule.lookupSymbol(*callee))
            needed.push_back(*callee);
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        auto callee = callOp.getCallee();
        if (!clonedSymbols.contains(callee) &&
            !microModule.lookupSymbol(callee))
          needed.push_back(callee);
      }
      if (auto addrOp = dyn_cast<LLVM::AddressOfOp>(op)) {
        auto name = addrOp.getGlobalName();
        if (!clonedSymbols.contains(name) && !microModule.lookupSymbol(name))
          needed.push_back(name);
      }
    });

    for (auto name : needed) {
      if (clonedSymbols.contains(name))
        continue;
      clonedSymbols.insert(name);

      auto *srcOp = parentModule.lookupSymbol(name);
      if (!srcOp)
        continue;

      OpBuilder builder(microModule.getContext());
      builder.setInsertionPointToEnd(microModule.getBody());

      if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(srcOp)) {
        auto cloned = builder.clone(*llvmFunc, mapping);
        if (auto clonedFunc = dyn_cast<LLVM::LLVMFuncOp>(cloned)) {
          if (!clonedFunc.getBody().empty()) {
            for (Block &block : clonedFunc.getBody())
              block.dropAllDefinedValueUses();
            clonedFunc.getBody().dropAllReferences();
            clonedFunc.getBody().getBlocks().clear();
          }
          auto linkage = clonedFunc.getLinkage();
          if (linkage != LLVM::Linkage::External &&
              linkage != LLVM::Linkage::ExternWeak)
            clonedFunc.setLinkage(LLVM::Linkage::External);
        }
        changed = true;
      } else if (auto funcFunc = dyn_cast<func::FuncOp>(srcOp)) {
        auto decl = func::FuncOp::create(builder, funcFunc.getLoc(),
                                         funcFunc.getSymName(),
                                         funcFunc.getFunctionType());
        // External func.func declarations cannot be public; keep cloned
        // declarations private to avoid verifier errors in large UVM modules.
        decl.setPrivate();
        changed = true;
      } else if (auto globalOp = dyn_cast<LLVM::GlobalOp>(srcOp)) {
        builder.clone(*globalOp, mapping);
        changed = true;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Lowering: arith/cf/func → LLVM dialect (in-place)
//===----------------------------------------------------------------------===//

static LLVM::ICmpPredicate convertCmpPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return LLVM::ICmpPredicate::eq;
  case arith::CmpIPredicate::ne:
    return LLVM::ICmpPredicate::ne;
  case arith::CmpIPredicate::slt:
    return LLVM::ICmpPredicate::slt;
  case arith::CmpIPredicate::sle:
    return LLVM::ICmpPredicate::sle;
  case arith::CmpIPredicate::sgt:
    return LLVM::ICmpPredicate::sgt;
  case arith::CmpIPredicate::sge:
    return LLVM::ICmpPredicate::sge;
  case arith::CmpIPredicate::ult:
    return LLVM::ICmpPredicate::ult;
  case arith::CmpIPredicate::ule:
    return LLVM::ICmpPredicate::ule;
  case arith::CmpIPredicate::ugt:
    return LLVM::ICmpPredicate::ugt;
  case arith::CmpIPredicate::uge:
    return LLVM::ICmpPredicate::uge;
  }
  llvm_unreachable("unhandled arith::CmpIPredicate");
}

static bool lowerFuncArithCfToLLVM(ModuleOp microModule,
                                    MLIRContext &mlirContext) {
  IRRewriter rewriter(&mlirContext);
  auto ensurePrivateFuncDecl = [&](llvm::StringRef name, FunctionType type) {
    if (microModule.lookupSymbol<func::FuncOp>(name) ||
        microModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return;
    OpBuilder builder(microModule.getContext());
    builder.setInsertionPointToEnd(microModule.getBody());
    auto decl = func::FuncOp::create(builder, UnknownLoc::get(&mlirContext),
                                     name, type);
    decl.setPrivate();
  };
  auto convertIntegerWidth = [&](Location loc, Value value,
                                 Type targetType) -> Value {
    if (value.getType() == targetType)
      return value;
    auto srcIntTy = dyn_cast<IntegerType>(value.getType());
    auto dstIntTy = dyn_cast<IntegerType>(targetType);
    if (!srcIntTy || !dstIntTy)
      return {};
    if (srcIntTy.getWidth() < dstIntTy.getWidth())
      return rewriter.create<arith::ExtUIOp>(loc, targetType, value);
    if (srcIntTy.getWidth() > dstIntTy.getWidth())
      return rewriter.create<arith::TruncIOp>(loc, targetType, value);
    return value;
  };
  auto materializeLLVMValue = [&](Location loc, Value value, Type targetType,
                                  const auto &self) -> Value {
    if (value.getType() == targetType)
      return value;
    if (!targetType || !LLVM::isCompatibleType(targetType))
      return {};

    if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
        if (Value peeled = self(loc, castOp.getOperand(0), targetType, self))
          return peeled;
      }
    }

    if (auto targetIntTy = dyn_cast<IntegerType>(targetType)) {
      if (isa<IntegerType>(value.getType()))
        return convertIntegerWidth(loc, value, targetIntTy);
    }

    if (auto targetStructTy = dyn_cast<LLVM::LLVMStructType>(targetType)) {
      if (targetStructTy.isOpaque())
        return {};
      auto hwStructTy = dyn_cast<hw::StructType>(value.getType());
      if (!hwStructTy)
        return {};
      auto targetFields = targetStructTy.getBody();
      auto hwFields = hwStructTy.getElements();
      if (targetFields.size() != hwFields.size())
        return {};

      if (auto createOp = value.getDefiningOp<hw::StructCreateOp>()) {
        if (createOp.getNumOperands() != hwFields.size())
          return {};
        Value agg = LLVM::UndefOp::create(rewriter, loc, targetStructTy);
        for (unsigned i = 0; i < hwFields.size(); ++i) {
          Value field =
              self(loc, createOp.getOperand(i), targetFields[i], self);
          if (!field)
            return {};
          agg = LLVM::InsertValueOp::create(
              rewriter, loc, agg, field, ArrayRef<int64_t>{(int64_t)i});
        }
        return agg;
      }

      if (auto aggConst = value.getDefiningOp<hw::AggregateConstantOp>()) {
        auto fields = aggConst.getFields();
        if (fields.size() != hwFields.size())
          return {};
        Value agg = LLVM::UndefOp::create(rewriter, loc, targetStructTy);
        for (unsigned i = 0; i < fields.size(); ++i) {
          auto intAttr = dyn_cast<IntegerAttr>(fields[i]);
          auto intTy = dyn_cast<IntegerType>(targetFields[i]);
          if (!intAttr || !intTy)
            return {};
          Value c = arith::ConstantOp::create(rewriter, loc, intAttr);
          Value field = self(loc, c, targetFields[i], self);
          if (!field)
            return {};
          agg = LLVM::InsertValueOp::create(
              rewriter, loc, agg, field, ArrayRef<int64_t>{(int64_t)i});
        }
        return agg;
      }

      if (auto bitcastOp = value.getDefiningOp<hw::BitcastOp>()) {
        auto inIntTy = dyn_cast<IntegerType>(bitcastOp.getInput().getType());
        if (!inIntTy)
          return {};
        unsigned totalWidth = 0;
        SmallVector<IntegerType> targetIntFields;
        targetIntFields.reserve(targetFields.size());
        for (Type fieldTy : targetFields) {
          auto intTy = dyn_cast<IntegerType>(fieldTy);
          if (!intTy)
            return {};
          targetIntFields.push_back(intTy);
          totalWidth += intTy.getWidth();
        }
        if (inIntTy.getWidth() != totalWidth)
          return {};
        Value raw = bitcastOp.getInput();
        Value agg = LLVM::UndefOp::create(rewriter, loc, targetStructTy);
        unsigned consumedHigh = 0;
        for (unsigned i = 0; i < targetIntFields.size(); ++i) {
          unsigned width = targetIntFields[i].getWidth();
          unsigned lowBit = totalWidth - consumedHigh - width;
          Value slice;
          if (lowBit == 0 && width == inIntTy.getWidth()) {
            slice = raw;
          } else {
            slice = raw;
            if (lowBit != 0) {
              Value shiftAmt = arith::ConstantOp::create(
                  rewriter, loc,
                  rewriter.getIntegerAttr(inIntTy, lowBit));
              slice = arith::ShRUIOp::create(rewriter, loc, slice, shiftAmt);
            }
            if (width < inIntTy.getWidth())
              slice = arith::TruncIOp::create(rewriter, loc, targetIntFields[i],
                                              slice);
          }
          Value field = self(loc, slice, targetFields[i], self);
          if (!field)
            return {};
          agg = LLVM::InsertValueOp::create(
              rewriter, loc, agg, field, ArrayRef<int64_t>{(int64_t)i});
          consumedHigh += width;
        }
        return agg;
      }
      return {};
    }

    if (auto targetArrayTy = dyn_cast<LLVM::LLVMArrayType>(targetType)) {
      auto hwArrayTy = dyn_cast<hw::ArrayType>(value.getType());
      if (!hwArrayTy || hwArrayTy.getNumElements() != targetArrayTy.getNumElements())
        return {};
      auto createOp = value.getDefiningOp<hw::ArrayCreateOp>();
      if (!createOp || createOp.getNumOperands() != hwArrayTy.getNumElements())
        return {};
      Value agg = LLVM::UndefOp::create(rewriter, loc, targetArrayTy);
      for (unsigned logicalIdx = 0, e = hwArrayTy.getNumElements();
           logicalIdx < e; ++logicalIdx) {
        unsigned opIdx = e - 1 - logicalIdx;
        Value elem = self(loc, createOp.getOperand(opIdx),
                          targetArrayTy.getElementType(), self);
        if (!elem)
          return {};
        agg = LLVM::InsertValueOp::create(
            rewriter, loc, agg, elem, ArrayRef<int64_t>{(int64_t)logicalIdx});
      }
      return agg;
    }

    return {};
  };

  // Phase 0: Lower hw.constant → arith.constant, comb ops → arith ops,
  // and resolve unrealized_conversion_cast before the main arith→LLVM pass.
  {
    llvm::SmallVector<Operation *> preOps;
    microModule.walk([&](Operation *op) {
      if (isa<hw::ConstantOp, hw::BitcastOp, hw::StructCreateOp,
              hw::StructExtractOp, hw::AggregateConstantOp, hw::ArrayCreateOp,
              hw::ArrayGetOp, comb::ExtractOp,
              comb::AndOp, comb::OrOp, comb::XorOp, comb::ICmpOp,
              comb::AddOp, comb::SubOp, comb::DivUOp, comb::DivSOp,
              comb::ModUOp, comb::ModSOp, comb::MulOp,
              comb::MuxOp,
              comb::ConcatOp, comb::ReplicateOp, comb::ShlOp,
              comb::ShrUOp, comb::ShrSOp, llhd::ConstantTimeOp,
              llhd::CurrentTimeOp, llhd::TimeToIntOp, llhd::IntToTimeOp,
              llhd::SignalOp, llhd::DriveOp, llhd::ProbeOp, llhd::SigExtractOp,
              llhd::SigStructExtractOp,
              UnrealizedConversionCastOp>(op))
        preOps.push_back(op);
    });
    for (auto *op : preOps) {
      rewriter.setInsertionPoint(op);
      auto loc = op->getLoc();

      if (auto hwConst = dyn_cast<hw::ConstantOp>(op)) {
        auto arithConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(hwConst.getType(),
                                         hwConst.getValue()));
        hwConst.replaceAllUsesWith(arithConst.getResult());
        rewriter.eraseOp(hwConst);
      } else if (auto timeToIntOp = dyn_cast<llhd::TimeToIntOp>(op)) {
        Value input = timeToIntOp.getOperand();
        Value replacement;
        Type resultTy = timeToIntOp.getResult().getType();
        if (auto currentTimeOp = input.getDefiningOp<llhd::CurrentTimeOp>()) {
          auto getTimeTy = rewriter.getFunctionType({}, {rewriter.getI64Type()});
          ensurePrivateFuncDecl("__circt_sim_current_time_fs", getTimeTy);
          auto call = rewriter.create<func::CallOp>(
              loc, "__circt_sim_current_time_fs",
              TypeRange{rewriter.getI64Type()},
              ValueRange{});
          replacement = call.getResult(0);
          replacement = convertIntegerWidth(loc, replacement, resultTy);
          if (!replacement)
            continue;
          timeToIntOp.replaceAllUsesWith(replacement);
          rewriter.eraseOp(timeToIntOp);
          if (currentTimeOp->use_empty())
            rewriter.eraseOp(currentTimeOp);
          continue;
        }
        if (auto constTimeOp = input.getDefiningOp<llhd::ConstantTimeOp>()) {
          auto timeAttr = llvm::dyn_cast<llhd::TimeAttr>(constTimeOp.getValueAttr());
          if (!timeAttr)
            continue;
          uint64_t fs = toFemtoseconds(timeAttr);
          replacement = arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(rewriter.getI64Type(), fs))
                            .getResult();
          replacement = convertIntegerWidth(loc, replacement, resultTy);
          if (!replacement)
            continue;
          timeToIntOp.replaceAllUsesWith(replacement);
          rewriter.eraseOp(timeToIntOp);
          if (constTimeOp->use_empty())
            rewriter.eraseOp(constTimeOp);
          continue;
        }
        if (auto intToTimeOp = input.getDefiningOp<llhd::IntToTimeOp>()) {
          Value raw = intToTimeOp.getOperand();
          if (raw.getType() != resultTy) {
            auto rawIntTy = dyn_cast<IntegerType>(raw.getType());
            auto resIntTy = dyn_cast<IntegerType>(resultTy);
            if (rawIntTy && resIntTy) {
              if (rawIntTy.getWidth() < resIntTy.getWidth())
                raw = rewriter.create<arith::ExtUIOp>(loc, resultTy, raw);
              else if (rawIntTy.getWidth() > resIntTy.getWidth())
                raw = rewriter.create<arith::TruncIOp>(loc, resultTy, raw);
            }
          }
          if (raw.getType() == resultTy) {
            timeToIntOp.replaceAllUsesWith(raw);
            rewriter.eraseOp(timeToIntOp);
            if (intToTimeOp->use_empty())
              rewriter.eraseOp(intToTimeOp);
            continue;
          }
        }
      } else if (auto intToTimeOp = dyn_cast<llhd::IntToTimeOp>(op)) {
        bool allResultsDead = true;
        for (Value result : intToTimeOp->getResults())
          allResultsDead &= result.use_empty();
        if (allResultsDead) {
          rewriter.eraseOp(intToTimeOp);
          continue;
        }
      } else if (auto sigOp = dyn_cast<llhd::SignalOp>(op)) {
        auto refTy = dyn_cast<llhd::RefType>(sigOp.getResult().getType());
        Type nestedTy = refTy ? refTy.getNestedType() : Type();
        Type llvmNestedTy =
            nestedTy ? convertToLLVMCompatibleType(nestedTy, &mlirContext)
                     : Type();
        if (!llvmNestedTy || !LLVM::isCompatibleType(llvmNestedTy))
          continue;
        Value initVal =
            materializeLLVMValue(loc, sigOp.getInit(), llvmNestedTy,
                                 materializeLLVMValue);
        if (!initVal)
          continue;
        auto one = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));
        auto alloca = LLVM::AllocaOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(&mlirContext),
            llvmNestedTy, one);
        rewriter.create<LLVM::StoreOp>(loc, initVal, alloca);
        auto refCast = rewriter.create<UnrealizedConversionCastOp>(
            loc, TypeRange{sigOp.getResult().getType()}, ValueRange{alloca});
        sigOp.getResult().replaceAllUsesWith(refCast.getResult(0));
        rewriter.eraseOp(sigOp);
        continue;
      } else if (auto sigStructExtractOp = dyn_cast<llhd::SigStructExtractOp>(op)) {
        Value basePtr = unwrapPointerFromRefCast(sigStructExtractOp.getInput());
        if (!basePtr)
          continue;
        auto inRefTy =
            dyn_cast<llhd::RefType>(sigStructExtractOp.getInput().getType());
        auto outRefTy =
            dyn_cast<llhd::RefType>(sigStructExtractOp.getResult().getType());
        auto inStructTy =
            inRefTy ? dyn_cast<hw::StructType>(inRefTy.getNestedType())
                    : hw::StructType();
        auto outIntTy =
            outRefTy ? dyn_cast<IntegerType>(outRefTy.getNestedType())
                     : IntegerType();
        if (!inStructTy || !outIntTy)
          continue;
        auto fieldIndex = inStructTy.getFieldIndex(sigStructExtractOp.getFieldAttr());
        Type fieldType = inStructTy.getFieldType(sigStructExtractOp.getFieldAttr());
        if (!fieldIndex || fieldType != outIntTy)
          continue;
        Type llvmStructTy =
            convertToLLVMCompatibleType(inStructTy, &mlirContext);
        if (!llvmStructTy || !isa<LLVM::LLVMStructType>(llvmStructTy))
          continue;
        SmallVector<LLVM::GEPArg> gepIndices;
        gepIndices.push_back(0);
        gepIndices.push_back(static_cast<int32_t>(*fieldIndex));
        auto fieldPtr = LLVM::GEPOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(&mlirContext),
            llvmStructTy, basePtr, gepIndices);
        auto fieldRef = rewriter.create<UnrealizedConversionCastOp>(
            loc, TypeRange{sigStructExtractOp.getResult().getType()},
            ValueRange{fieldPtr});
        sigStructExtractOp.getResult().replaceAllUsesWith(fieldRef.getResult(0));
        rewriter.eraseOp(sigStructExtractOp);
        continue;
      } else if (auto prbOp = dyn_cast<llhd::ProbeOp>(op)) {
        Type loadTy =
            convertToLLVMCompatibleType(prbOp.getResult().getType(),
                                        &mlirContext);
        if (!loadTy || !LLVM::isCompatibleType(loadTy))
          continue;
        Value signalPtr = prbOp.getSignal();
        if (!isa<LLVM::LLVMPointerType>(signalPtr.getType())) {
          signalPtr = unwrapPointerFromRefCast(signalPtr);
          if (!signalPtr)
            continue;
        }
        Value loaded = rewriter.create<LLVM::LoadOp>(loc, loadTy, signalPtr);
        if (prbOp.getResult().getType() == loadTy) {
          prbOp.replaceAllUsesWith(loaded);
        } else {
          auto cast = rewriter.create<UnrealizedConversionCastOp>(
              loc, TypeRange{prbOp.getResult().getType()}, ValueRange{loaded});
          prbOp.replaceAllUsesWith(cast.getResult(0));
        }
        rewriter.eraseOp(prbOp);
        continue;
      } else if (auto drvOp = dyn_cast<llhd::DriveOp>(op)) {
        // Lower pointer-backed immediate/delta drives to a plain store.
        if (drvOp.getEnable())
          continue;
        auto delayConst = drvOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
        if (!delayConst)
          continue;
        auto delayAttr = dyn_cast<llhd::TimeAttr>(delayConst.getValueAttr());
        if (!delayAttr)
          continue;
        bool isNearImmediate = toFemtoseconds(delayAttr) == 0 &&
                               delayAttr.getDelta() == 0 &&
                               delayAttr.getEpsilon() <= 1;
        if (!isNearImmediate)
          continue;

        // Lower drives to extracted refs as read-modify-write on the base
        // integer signal: drv(sig.extract(base, low), val) =>
        //   old = load base; new = insertbits(old, val, low); store new.
        if (auto sigExtract = drvOp.getSignal().getDefiningOp<llhd::SigExtractOp>()) {
          Value basePtr = unwrapPointerFromRefCast(sigExtract.getInput());
          if (!basePtr)
            continue;
          auto inRefTy = dyn_cast<llhd::RefType>(sigExtract.getInput().getType());
          auto outRefTy = dyn_cast<llhd::RefType>(sigExtract.getResult().getType());
          auto inIntTy = inRefTy ? dyn_cast<IntegerType>(inRefTy.getNestedType())
                                 : IntegerType();
          auto outIntTy = outRefTy ? dyn_cast<IntegerType>(outRefTy.getNestedType())
                                   : IntegerType();
          auto valIntTy = dyn_cast<IntegerType>(drvOp.getValue().getType());
          if (!inIntTy || !outIntTy || !valIntTy ||
              valIntTy.getWidth() != outIntTy.getWidth())
            continue;

          Value lowBit = sigExtract.getLowBit();
          auto lowTy = dyn_cast<IntegerType>(lowBit.getType());
          if (!lowTy)
            continue;
          unsigned baseWidth = inIntTy.getWidth();
          if (lowTy.getWidth() < baseWidth)
            lowBit = rewriter.create<arith::ExtUIOp>(loc, inIntTy, lowBit);
          else if (lowTy.getWidth() > baseWidth)
            lowBit = rewriter.create<arith::TruncIOp>(loc, inIntTy, lowBit);
          if (lowBit.getType() != inIntTy)
            continue;

          Value oldValue = rewriter.create<LLVM::LoadOp>(loc, inIntTy, basePtr);
          Value extValue = drvOp.getValue();
          if (extValue.getType() != inIntTy)
            extValue = rewriter.create<arith::ExtUIOp>(loc, inIntTy, extValue);
          Value shiftedValue =
              rewriter.create<arith::ShLIOp>(loc, extValue, lowBit);

          APInt maskBits = APInt::getLowBitsSet(baseWidth, outIntTy.getWidth());
          Value mask = arith::ConstantOp::create(
                           rewriter, loc,
                           rewriter.getIntegerAttr(inIntTy, maskBits))
                           .getResult();
          Value shiftedMask = rewriter.create<arith::ShLIOp>(loc, mask, lowBit);
          Value allOnes = arith::ConstantOp::create(
                              rewriter, loc,
                              rewriter.getIntegerAttr(inIntTy, APInt::getAllOnes(baseWidth)))
                              .getResult();
          Value invMask = rewriter.create<arith::XOrIOp>(loc, shiftedMask, allOnes);
          Value cleared = rewriter.create<arith::AndIOp>(loc, oldValue, invMask);
          Value inserted = rewriter.create<arith::AndIOp>(loc, shiftedValue, shiftedMask);
          Value merged = rewriter.create<arith::OrIOp>(loc, cleared, inserted);
          rewriter.create<LLVM::StoreOp>(loc, merged, basePtr);
          rewriter.eraseOp(drvOp);
          if (sigExtract->use_empty())
            rewriter.eraseOp(sigExtract);
          if (delayConst->use_empty())
            rewriter.eraseOp(delayConst);
          continue;
        }

        Value signalPtr = drvOp.getSignal();
        if (!isa<LLVM::LLVMPointerType>(signalPtr.getType())) {
          signalPtr = unwrapPointerFromRefCast(signalPtr);
          if (!signalPtr)
            continue;
        }
        Type storeTy = drvOp.getValue().getType();
        if (auto refTy = dyn_cast<llhd::RefType>(drvOp.getSignal().getType())) {
          Type converted = convertToLLVMCompatibleType(refTy.getNestedType(),
                                                       &mlirContext);
          if (converted && LLVM::isCompatibleType(converted))
            storeTy = converted;
        }
        Value storeVal = drvOp.getValue();
        if (storeVal.getType() != storeTy) {
          storeVal = materializeLLVMValue(loc, storeVal, storeTy,
                                          materializeLLVMValue);
          if (!storeVal)
            continue;
        }
        rewriter.create<LLVM::StoreOp>(loc, storeVal, signalPtr);
        rewriter.eraseOp(drvOp);
        continue;
      } else if (isa<llhd::ConstantTimeOp, llhd::CurrentTimeOp>(op)) {
        bool allResultsDead = true;
        for (Value result : op->getResults())
          allResultsDead &= result.use_empty();
        if (allResultsDead) {
          rewriter.eraseOp(op);
          continue;
        }
      } else if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(op)) {
        auto arrayCreateOp =
            arrayGetOp.getInput().getDefiningOp<hw::ArrayCreateOp>();
        auto idxTy = dyn_cast<IntegerType>(arrayGetOp.getIndex().getType());
        if (!arrayCreateOp || !idxTy ||
            !LLVM::isCompatibleType(arrayGetOp.getResult().getType()))
          continue;
        unsigned numElems = arrayCreateOp.getNumOperands();
        if (numElems == 0)
          continue;
        auto getElemForLogicalIndex = [&](unsigned idx) -> Value {
          unsigned opIdx = numElems - 1 - idx;
          return arrayCreateOp.getOperand(opIdx);
        };
        Value result = getElemForLogicalIndex(0);
        for (unsigned idx = 1; idx < numElems; ++idx) {
          auto idxConst = arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(idxTy, idx));
          auto cmp = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, arrayGetOp.getIndex(),
              idxConst);
          Value candidate = getElemForLogicalIndex(idx);
          result = LLVM::SelectOp::create(
              rewriter, loc, arrayGetOp.getResult().getType(), cmp, candidate,
              result);
        }
        arrayGetOp.replaceAllUsesWith(result);
        rewriter.eraseOp(arrayGetOp);
        continue;
      } else if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
        Value input = extractOp.getInput();
        unsigned lowBit = extractOp.getLowBit();
        auto resultType = extractOp.getType();
        if (lowBit != 0) {
          auto shift = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(input.getType(), lowBit));
          input = rewriter.create<arith::ShRUIOp>(loc, input, shift);
        }
        auto trunc = rewriter.create<arith::TruncIOp>(loc, resultType, input);
        extractOp.replaceAllUsesWith(trunc.getResult());
        rewriter.eraseOp(extractOp);
      } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
        // comb.and is variadic — fold left.
        auto inputs = andOp.getInputs();
        Value result = inputs[0];
        for (unsigned i = 1; i < inputs.size(); ++i)
          result = rewriter.create<arith::AndIOp>(loc, result, inputs[i]);
        andOp.replaceAllUsesWith(result);
        rewriter.eraseOp(andOp);
      } else if (auto icmpOp = dyn_cast<comb::ICmpOp>(op)) {
        arith::CmpIPredicate pred;
        switch (icmpOp.getPredicate()) {
        case comb::ICmpPredicate::eq:
        case comb::ICmpPredicate::ceq:
        case comb::ICmpPredicate::weq:
          pred = arith::CmpIPredicate::eq;
          break;
        case comb::ICmpPredicate::ne:
        case comb::ICmpPredicate::cne:
        case comb::ICmpPredicate::wne:
          pred = arith::CmpIPredicate::ne;
          break;
        case comb::ICmpPredicate::slt:
          pred = arith::CmpIPredicate::slt;
          break;
        case comb::ICmpPredicate::sle:
          pred = arith::CmpIPredicate::sle;
          break;
        case comb::ICmpPredicate::sgt:
          pred = arith::CmpIPredicate::sgt;
          break;
        case comb::ICmpPredicate::sge:
          pred = arith::CmpIPredicate::sge;
          break;
        case comb::ICmpPredicate::ult:
          pred = arith::CmpIPredicate::ult;
          break;
        case comb::ICmpPredicate::ule:
          pred = arith::CmpIPredicate::ule;
          break;
        case comb::ICmpPredicate::ugt:
          pred = arith::CmpIPredicate::ugt;
          break;
        case comb::ICmpPredicate::uge:
          pred = arith::CmpIPredicate::uge;
          break;
        }
        auto result = rewriter.create<arith::CmpIOp>(loc, pred,
                                                       icmpOp.getLhs(),
                                                       icmpOp.getRhs());
        icmpOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(icmpOp);
      } else if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
        auto inputs = xorOp.getInputs();
        Value result = inputs[0];
        for (unsigned i = 1; i < inputs.size(); ++i)
          result = rewriter.create<arith::XOrIOp>(loc, result, inputs[i]);
        xorOp.replaceAllUsesWith(result);
        rewriter.eraseOp(xorOp);
      } else if (auto orOp = dyn_cast<comb::OrOp>(op)) {
        auto inputs = orOp.getInputs();
        Value result = inputs[0];
        for (unsigned i = 1; i < inputs.size(); ++i)
          result = rewriter.create<arith::OrIOp>(loc, result, inputs[i]);
        orOp.replaceAllUsesWith(result);
        rewriter.eraseOp(orOp);
      } else if (auto addOp = dyn_cast<comb::AddOp>(op)) {
        auto inputs = addOp.getInputs();
        Value result = inputs[0];
        for (unsigned i = 1; i < inputs.size(); ++i)
          result = rewriter.create<arith::AddIOp>(loc, result, inputs[i]);
        addOp.replaceAllUsesWith(result);
        rewriter.eraseOp(addOp);
      } else if (auto subOp = dyn_cast<comb::SubOp>(op)) {
        auto result = rewriter.create<arith::SubIOp>(loc, subOp.getLhs(),
                                                      subOp.getRhs());
        subOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(subOp);
      } else if (auto divOp = dyn_cast<comb::DivUOp>(op)) {
        auto result = rewriter.create<arith::DivUIOp>(loc, divOp.getLhs(),
                                                       divOp.getRhs());
        divOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(divOp);
      } else if (auto divOp = dyn_cast<comb::DivSOp>(op)) {
        auto result = rewriter.create<arith::DivSIOp>(loc, divOp.getLhs(),
                                                      divOp.getRhs());
        divOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(divOp);
      } else if (auto modOp = dyn_cast<comb::ModUOp>(op)) {
        auto result = rewriter.create<arith::RemUIOp>(loc, modOp.getLhs(),
                                                      modOp.getRhs());
        modOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(modOp);
      } else if (auto modOp = dyn_cast<comb::ModSOp>(op)) {
        auto result = rewriter.create<arith::RemSIOp>(loc, modOp.getLhs(),
                                                      modOp.getRhs());
        modOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(modOp);
      } else if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
        auto inputs = mulOp.getInputs();
        Value result = inputs[0];
        for (unsigned i = 1; i < inputs.size(); ++i)
          result = rewriter.create<arith::MulIOp>(loc, result, inputs[i]);
        mulOp.replaceAllUsesWith(result);
        rewriter.eraseOp(mulOp);
      } else if (auto shlOp = dyn_cast<comb::ShlOp>(op)) {
        auto result =
            rewriter.create<arith::ShLIOp>(loc, shlOp.getLhs(), shlOp.getRhs());
        shlOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(shlOp);
      } else if (auto shruOp = dyn_cast<comb::ShrUOp>(op)) {
        auto result = rewriter.create<arith::ShRUIOp>(loc, shruOp.getLhs(),
                                                       shruOp.getRhs());
        shruOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(shruOp);
      } else if (auto shrsOp = dyn_cast<comb::ShrSOp>(op)) {
        auto result = rewriter.create<arith::ShRSIOp>(loc, shrsOp.getLhs(),
                                                       shrsOp.getRhs());
        shrsOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(shrsOp);
      } else if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
        auto result = rewriter.create<arith::SelectOp>(
            loc, muxOp.getCond(), muxOp.getTrueValue(),
            muxOp.getFalseValue());
        muxOp.replaceAllUsesWith(result.getResult());
        rewriter.eraseOp(muxOp);
      } else if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
        // concat(a, b, c) = (a << (bw+cw)) | (b << cw) | c
        // Operands are MSB-first.
        auto resultType = concatOp.getResult().getType();
        unsigned numOps = concatOp.getNumOperands();
        Value result = rewriter.create<arith::ExtUIOp>(
            loc, resultType, concatOp.getOperand(numOps - 1));
        unsigned shift =
            concatOp.getOperand(numOps - 1).getType().getIntOrFloatBitWidth();
        for (int i = numOps - 2; i >= 0; --i) {
          auto extended = rewriter.create<arith::ExtUIOp>(
              loc, resultType, concatOp.getOperand(i));
          auto shiftAmt = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(resultType, shift));
          auto shifted =
              rewriter.create<arith::ShLIOp>(loc, extended, shiftAmt);
          result = rewriter.create<arith::OrIOp>(loc, result, shifted);
          shift +=
              concatOp.getOperand(i).getType().getIntOrFloatBitWidth();
        }
        concatOp.replaceAllUsesWith(result);
        rewriter.eraseOp(concatOp);
      } else if (auto repOp = dyn_cast<comb::ReplicateOp>(op)) {
        auto resultType = repOp.getResult().getType();
        unsigned inputWidth =
            repOp.getInput().getType().getIntOrFloatBitWidth();
        unsigned resultWidth = resultType.getIntOrFloatBitWidth();
        unsigned count = resultWidth / inputWidth;
        Value result = rewriter.create<arith::ExtUIOp>(loc, resultType,
                                                        repOp.getInput());
        unsigned shift = inputWidth;
        for (unsigned i = 1; i < count; ++i) {
          auto extended = rewriter.create<arith::ExtUIOp>(loc, resultType,
                                                           repOp.getInput());
          auto shiftAmt = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(resultType, shift));
          auto shifted =
              rewriter.create<arith::ShLIOp>(loc, extended, shiftAmt);
          result = rewriter.create<arith::OrIOp>(loc, result, shifted);
          shift += inputWidth;
        }
        repOp.replaceAllUsesWith(result);
        rewriter.eraseOp(repOp);
      } else if (auto bitcastOp = dyn_cast<hw::BitcastOp>(op)) {
        // If both types are the same, just fold away.
        if (bitcastOp.getInput().getType() ==
            bitcastOp.getResult().getType()) {
          bitcastOp.replaceAllUsesWith(bitcastOp.getInput());
          rewriter.eraseOp(bitcastOp);
        } else {
          // If both are integers of same width, use arith.bitcast.
          auto inTy = bitcastOp.getInput().getType();
          auto outTy = bitcastOp.getResult().getType();
          if (inTy.isIntOrFloat() && outTy.isIntOrFloat() &&
              inTy.getIntOrFloatBitWidth() == outTy.getIntOrFloatBitWidth()) {
            auto result = rewriter.create<arith::BitcastOp>(loc, outTy,
                                                             bitcastOp.getInput());
            bitcastOp.replaceAllUsesWith(result.getResult());
            rewriter.eraseOp(bitcastOp);
          }
        }
      } else if (auto extractOp = dyn_cast<hw::StructExtractOp>(op)) {
        auto lowerExtractFromFields = [&](Value lhs, Value rhs) {
          StringRef fieldName = extractOp.getFieldName();
          if (fieldName == "value")
            extractOp.getResult().replaceAllUsesWith(lhs);
          else if (fieldName == "unknown")
            extractOp.getResult().replaceAllUsesWith(rhs);
          else
            return;
          rewriter.eraseOp(extractOp);
        };

        // Fold extract(cast(llvm.struct -> hw.struct), "field") to
        // llvm.extractvalue on the original llvm struct value.
        if (auto castOp = extractOp.getInput()
                              .getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
            auto srcTy =
                dyn_cast<LLVM::LLVMStructType>(castOp.getOperand(0).getType());
            auto dstTy =
                dyn_cast<hw::StructType>(castOp.getResult(0).getType());
            if (srcTy && dstTy && !srcTy.isOpaque()) {
              if (auto fieldIndex =
                      dstTy.getFieldIndex(extractOp.getFieldName())) {
                unsigned idx = *fieldIndex;
                auto body = srcTy.getBody();
                if (idx < body.size()) {
                  Type srcFieldTy = body[idx];
                  if (srcFieldTy == extractOp.getResult().getType()) {
                    llvm::SmallVector<int64_t, 1> positions{
                        static_cast<int64_t>(idx)};
                    auto llvmExtract = rewriter.create<LLVM::ExtractValueOp>(
                        loc, srcFieldTy, castOp.getOperand(0), positions);
                    extractOp.getResult().replaceAllUsesWith(
                        llvmExtract.getResult());
                    rewriter.eraseOp(extractOp);
                    if (castOp->use_empty())
                      rewriter.eraseOp(castOp);
                    continue;
                  }
                }
              }
            }
          }
        }

        // Generic fold: extract(struct_create(...), "field") -> operand.
        if (auto createOp =
                extractOp.getInput().getDefiningOp<hw::StructCreateOp>()) {
          if (auto structTy = dyn_cast<hw::StructType>(createOp.getType())) {
            if (auto fieldIndex = structTy.getFieldIndex(extractOp.getFieldName())) {
              unsigned idx = *fieldIndex;
              if (idx < createOp.getNumOperands()) {
                extractOp.getResult().replaceAllUsesWith(createOp.getOperand(idx));
                rewriter.eraseOp(extractOp);
                continue;
              }
            }
          }
          if (getFourStateValueWidth(createOp.getType()) &&
              createOp.getNumOperands() == 2) {
            lowerExtractFromFields(createOp.getOperand(0), createOp.getOperand(1));
            continue;
          }
        } else if (auto aggConst = extractOp.getInput()
                                      .getDefiningOp<hw::AggregateConstantOp>()) {
          // Generic fold: extract(aggregate_constant[...], "field") -> constant.
          if (auto structTy = dyn_cast<hw::StructType>(aggConst.getType())) {
            if (auto fieldIndex = structTy.getFieldIndex(extractOp.getFieldName())) {
              auto fields = aggConst.getFields();
              unsigned idx = *fieldIndex;
              if (idx < fields.size()) {
                auto intAttr = dyn_cast<IntegerAttr>(fields[idx]);
                if (intAttr) {
                  auto c = arith::ConstantOp::create(rewriter, loc, intAttr);
                  extractOp.getResult().replaceAllUsesWith(c.getResult());
                  rewriter.eraseOp(extractOp);
                  continue;
                }
              }
            }
          }
          if (getFourStateValueWidth(aggConst.getType())) {
            auto fields = aggConst.getFields();
            if (fields.size() == 2) {
              auto mkConst = [&](Attribute attr) -> Value {
                auto intAttr = dyn_cast<IntegerAttr>(attr);
                if (!intAttr)
                  return {};
                return arith::ConstantOp::create(rewriter, loc, intAttr)
                    .getResult();
              };
              Value valField = mkConst(fields[0]);
              Value unknownField = mkConst(fields[1]);
              if (valField && unknownField) {
                lowerExtractFromFields(valField, unknownField);
                continue;
              }
            }
          }
        }
      } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
        // Cancel chained casts when they round-trip through an intermediate
        // type (e.g. ptr -> !llhd.ref<T> -> ptr).
        if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
          Value in = castOp.getOperand(0);
          Type outTy = castOp.getResult(0).getType();
          if (auto prevCast = in.getDefiningOp<UnrealizedConversionCastOp>()) {
            if (prevCast.getNumOperands() == 1 && prevCast.getNumResults() == 1 &&
                prevCast.getOperand(0).getType() == outTy) {
              castOp.getResult(0).replaceAllUsesWith(prevCast.getOperand(0));
              rewriter.eraseOp(castOp);
              if (prevCast->use_empty())
                rewriter.eraseOp(prevCast);
              continue;
            }
          }
        }
        // Dead casts are no-ops; keep them from forcing whole-function
        // rejection in the compilability filter.
        bool allResultsDead = true;
        for (Value result : castOp.getResults())
          allResultsDead &= result.use_empty();
        if (allResultsDead) {
          rewriter.eraseOp(castOp);
          continue;
        }
        // If single input → single output with same type, fold away.
        if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1 &&
            castOp.getOperand(0).getType() == castOp.getResult(0).getType()) {
          castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
          rewriter.eraseOp(castOp);
        }
        // Otherwise leave in place — will be stripped later if unresolvable.
      }
    }

    // Drop dead 4-state helper ops after struct_extract folding.
    llvm::SmallVector<Operation *> deadStructOps;
    microModule.walk([&](Operation *op) {
      if (op->use_empty() &&
          isa<hw::StructCreateOp, hw::AggregateConstantOp, hw::BitcastOp>(op))
        deadStructOps.push_back(op);
    });
    for (Operation *op : deadStructOps) {
      op->erase();
    }

    llvm::SmallVector<Operation *> deadCasts;
    microModule.walk([&](UnrealizedConversionCastOp castOp) {
      bool allResultsDead = true;
      for (Value result : castOp.getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        deadCasts.push_back(castOp.getOperation());
    });
    for (Operation *op : deadCasts)
      op->erase();

    llvm::SmallVector<Operation *> deadTimeOps;
    microModule.walk([&](Operation *op) {
      if (!isa<llhd::ConstantTimeOp, llhd::CurrentTimeOp>(op))
        return;
      bool allResultsDead = true;
      for (Value result : op->getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        deadTimeOps.push_back(op);
    });
    for (Operation *op : deadTimeOps)
      op->erase();

    llvm::SmallVector<Operation *> deadSigExtractOps;
    microModule.walk([&](llhd::SigExtractOp sigExtractOp) {
      if (sigExtractOp->use_empty())
        deadSigExtractOps.push_back(sigExtractOp.getOperation());
    });
    for (Operation *op : deadSigExtractOps)
      op->erase();

    llvm::SmallVector<Operation *> deadSigStructExtractOps;
    microModule.walk([&](llhd::SigStructExtractOp sigStructExtractOp) {
      if (sigStructExtractOp->use_empty())
        deadSigStructExtractOps.push_back(sigStructExtractOp.getOperation());
    });
    for (Operation *op : deadSigStructExtractOps)
      op->erase();

    llvm::SmallVector<Operation *> deadArrayCreateOps;
    microModule.walk([&](hw::ArrayCreateOp arrayCreateOp) {
      if (arrayCreateOp->use_empty())
        deadArrayCreateOps.push_back(arrayCreateOp.getOperation());
    });
    for (Operation *op : deadArrayCreateOps)
      op->erase();
  }

  // Phase 1: Rewrite arith/cf/func ops to LLVM equivalents.
  llvm::SmallVector<Operation *> toRewrite;
  microModule.walk([&](Operation *op) {
    auto *dialect = op->getDialect();
    if (dialect && (isa<arith::ArithDialect>(dialect) ||
                    isa<cf::ControlFlowDialect>(dialect)))
      toRewrite.push_back(op);
    else if (isa<func::ReturnOp, func::CallOp, func::CallIndirectOp,
                 UnrealizedConversionCastOp>(op))
      toRewrite.push_back(op);
  });

  for (auto *op : toRewrite) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();

    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto r =
          rewriter.create<LLVM::ConstantOp>(loc, c.getType(), c.getValue());
      c.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(c);
    } else if (auto o = dyn_cast<arith::AddIOp>(op)) {
      auto r = rewriter.create<LLVM::AddOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::SubIOp>(op)) {
      auto r = rewriter.create<LLVM::SubOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::MulIOp>(op)) {
      auto r = rewriter.create<LLVM::MulOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::DivSIOp>(op)) {
      auto r = rewriter.create<LLVM::SDivOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::DivUIOp>(op)) {
      auto r = rewriter.create<LLVM::UDivOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::RemSIOp>(op)) {
      auto r = rewriter.create<LLVM::SRemOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::RemUIOp>(op)) {
      auto r = rewriter.create<LLVM::URemOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::AndIOp>(op)) {
      auto r = rewriter.create<LLVM::AndOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::OrIOp>(op)) {
      auto r = rewriter.create<LLVM::OrOp>(loc, o.getType(), o.getLhs(),
                                            o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::XOrIOp>(op)) {
      auto r = rewriter.create<LLVM::XOrOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShLIOp>(op)) {
      auto r = rewriter.create<LLVM::ShlOp>(loc, o.getType(), o.getLhs(),
                                             o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShRSIOp>(op)) {
      auto r = rewriter.create<LLVM::AShrOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShRUIOp>(op)) {
      auto r = rewriter.create<LLVM::LShrOp>(loc, o.getType(), o.getLhs(),
                                              o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ExtSIOp>(op)) {
      auto r = rewriter.create<LLVM::SExtOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ExtUIOp>(op)) {
      auto r = rewriter.create<LLVM::ZExtOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::TruncIOp>(op)) {
      auto r = rewriter.create<LLVM::TruncOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::BitcastOp>(op)) {
      auto r = rewriter.create<LLVM::BitcastOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::CmpIOp>(op)) {
      auto r = rewriter.create<LLVM::ICmpOp>(
          loc, convertCmpPredicate(o.getPredicate()), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::SelectOp>(op)) {
      auto r = rewriter.create<LLVM::SelectOp>(
          loc, o.getType(), o.getCondition(), o.getTrueValue(),
          o.getFalseValue());
      o.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::IndexCastOp>(op)) {
      o.replaceAllUsesWith(o.getIn());
      rewriter.eraseOp(o);
    } else if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
      rewriter.create<LLVM::BrOp>(loc, brOp.getDestOperands(),
                                   brOp.getDest());
      rewriter.eraseOp(brOp);
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(op)) {
      rewriter.create<LLVM::CondBrOp>(
          loc, condBrOp.getCondition(), condBrOp.getTrueDest(),
          condBrOp.getTrueDestOperands(), condBrOp.getFalseDest(),
          condBrOp.getFalseDestOperands());
      rewriter.eraseOp(condBrOp);
    } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
      rewriter.create<LLVM::ReturnOp>(loc, retOp.getOperands());
      rewriter.eraseOp(retOp);
    } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
        continue;
      Value input = castOp.getOperand(0);
      Type inTy = input.getType();
      Type outTy = castOp.getResult(0).getType();
      Value converted;

      if (inTy == outTy) {
        converted = input;
      } else if (isa<LLVM::LLVMPointerType>(inTy) && isa<IntegerType>(outTy)) {
        converted = rewriter.create<LLVM::PtrToIntOp>(loc, outTy, input);
      } else if (isa<IntegerType>(inTy) && isa<LLVM::LLVMPointerType>(outTy)) {
        converted = rewriter.create<LLVM::IntToPtrOp>(loc, outTy, input);
      } else if (isa<LLVM::LLVMPointerType>(inTy) &&
                 isa<LLVM::LLVMPointerType>(outTy)) {
        converted = rewriter.create<LLVM::BitcastOp>(loc, outTy, input);
      } else if (auto inIntTy = dyn_cast<IntegerType>(inTy)) {
        auto outIntTy = dyn_cast<IntegerType>(outTy);
        if (!outIntTy)
          continue;
        unsigned inWidth = inIntTy.getWidth();
        unsigned outWidth = outIntTy.getWidth();
        if (inWidth == outWidth)
          converted = input;
        else if (inWidth < outWidth)
          converted = rewriter.create<LLVM::ZExtOp>(loc, outTy, input);
        else
          converted = rewriter.create<LLVM::TruncOp>(loc, outTy, input);
      } else {
        continue;
      }

      castOp.getResult(0).replaceAllUsesWith(converted);
      rewriter.eraseOp(castOp);
    } else if (auto callIndirectOp = dyn_cast<func::CallIndirectOp>(op)) {
      // Lower:
      //   %fn = unrealized_conversion_cast %ptr : !llvm.ptr to (...) -> ...
      //   %r = func.call_indirect %fn(%args)
      // to:
      //   %r = llvm.call %ptr(%args) : !llvm.ptr, (...) -> ...
      auto calleeTy = dyn_cast<FunctionType>(callIndirectOp.getCallee().getType());
      if (!calleeTy || calleeTy.getNumResults() > 1)
        continue;

      Value calleePtr = callIndirectOp.getCallee();
      auto calleeCast = calleePtr.getDefiningOp<UnrealizedConversionCastOp>();
      if (calleeCast && calleeCast.getNumOperands() == 1 &&
          calleeCast.getNumResults() == 1 &&
          isa<LLVM::LLVMPointerType>(calleeCast.getOperand(0).getType()) &&
          isa<FunctionType>(calleeCast.getResult(0).getType()))
        calleePtr = calleeCast.getOperand(0);

      if (!isa<LLVM::LLVMPointerType>(calleePtr.getType()))
        continue;

      SmallVector<Type> llvmArgTypes;
      bool unsupported = false;
      for (Type argTy : calleeTy.getInputs()) {
        if (!LLVM::isCompatibleType(argTy)) {
          unsupported = true;
          break;
        }
        llvmArgTypes.push_back(argTy);
      }
      Type llvmRetTy = LLVM::LLVMVoidType::get(&mlirContext);
      if (!unsupported && calleeTy.getNumResults() == 1) {
        llvmRetTy = calleeTy.getResult(0);
        if (!LLVM::isCompatibleType(llvmRetTy))
          unsupported = true;
      }
      if (unsupported)
        continue;

      auto llvmFuncTy =
          LLVM::LLVMFunctionType::get(llvmRetTy, llvmArgTypes,
                                      /*isVarArg=*/false);

      SmallVector<Value> callOperands;
      callOperands.push_back(calleePtr);
      for (auto [arg, expectedTy] :
           llvm::zip(callIndirectOp.getArgOperands(), llvmArgTypes)) {
        if (arg.getType() != expectedTy) {
          unsupported = true;
          break;
        }
        callOperands.push_back(arg);
      }
      if (unsupported)
        continue;

      auto newCall = rewriter.create<LLVM::CallOp>(loc, llvmFuncTy, callOperands);
      callIndirectOp.replaceAllUsesWith(newCall.getResults());
      rewriter.eraseOp(callIndirectOp);

      // Cast becomes dead after lowering the only call_indirect use.
      if (calleeCast && calleeCast->use_empty())
        rewriter.eraseOp(calleeCast);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto newCall = rewriter.create<LLVM::CallOp>(
          loc, callOp.getResultTypes(), callOp.getCallee(),
          callOp.getOperands());
      callOp.replaceAllUsesWith(newCall.getResults());
      rewriter.eraseOp(callOp);
    }
  }

  // Phase 2: Convert func.func → llvm.func by body splicing.
  llvm::SmallVector<func::FuncOp> funcOps;
  microModule.walk([&](func::FuncOp op) { funcOps.push_back(op); });

  for (auto funcOp : funcOps) {
    auto funcType = funcOp.getFunctionType();
    std::string origName = funcOp.getSymName().str();
    auto loc = funcOp.getLoc();

    SmallVector<Type> argTypes(funcType.getInputs());
    Type retType = funcType.getNumResults() == 0
                       ? LLVM::LLVMVoidType::get(&mlirContext)
                       : funcType.getResult(0);
    auto llvmFuncType = LLVM::LLVMFunctionType::get(retType, argTypes);

    if (funcOp.isExternal()) {
      rewriter.setInsertionPoint(funcOp);
      rewriter.eraseOp(funcOp);
      rewriter.setInsertionPointToEnd(microModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(loc, origName, llvmFuncType);
      continue;
    }

    rewriter.setInsertionPoint(funcOp);
    std::string tmpName = "__tmp_compile_" + origName;
    auto llvmFunc =
        rewriter.create<LLVM::LLVMFuncOp>(loc, tmpName, llvmFuncType);

    auto &srcRegion = funcOp.getBody();
    auto &dstRegion = llvmFunc.getBody();
    dstRegion.getBlocks().splice(dstRegion.end(), srcRegion.getBlocks());
    rewriter.eraseOp(funcOp);
    llvmFunc.setSymName(origName);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Lowering: strip functions with residual non-LLVM ops
//===----------------------------------------------------------------------===//

/// Returns true if all parameter and result types of an LLVM function type are
/// compatible with LLVM IR translation.
static bool hasAllLLVMCompatibleTypes(LLVM::LLVMFunctionType funcTy) {
  if (!LLVM::isCompatibleType(funcTy.getReturnType()))
    return false;
  for (unsigned i = 0; i < funcTy.getNumParams(); ++i)
    if (!LLVM::isCompatibleType(funcTy.getParamType(i)))
      return false;
  return true;
}

static llvm::SmallVector<std::string>
stripNonLLVMFunctions(ModuleOp microModule) {
  llvm::SmallVector<std::string> stripped;
  llvm::SmallVector<LLVM::LLVMFuncOp> toStrip;

  microModule.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal()) {
      // External declarations with non-LLVM param/result types (e.g.
      // !hw.struct, !llhd.ref, !hw.inout) cannot be translated to LLVM IR
      // and must be erased before the translation step.
      if (!hasAllLLVMCompatibleTypes(func.getFunctionType()))
        toStrip.push_back(func);
      return;
    }
    // Defined functions whose signature contains non-LLVM types (e.g.
    // !llhd.ref) also fail LLVM IR translation — the entry block arguments
    // inherit the function param types and are not reachable by the inner-op
    // walk below, so we must check the function type here first.
    if (!hasAllLLVMCompatibleTypes(func.getFunctionType())) {
      toStrip.push_back(func);
      return;
    }
    bool hasNonLLVM = false;
    func.walk([&](Operation *op) {
      if (!isa<LLVM::LLVMDialect>(op->getDialect())) {
        hasNonLLVM = true;
        return WalkResult::interrupt();
      }
      for (auto ty : op->getResultTypes()) {
        if (!LLVM::isCompatibleType(ty)) {
          hasNonLLVM = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (hasNonLLVM)
      toStrip.push_back(func);
  });

  for (auto func : toStrip) {
    stripped.push_back(func.getSymName().str());
    if (func.isExternal() || !hasAllLLVMCompatibleTypes(func.getFunctionType())) {
      // Erase outright: external declarations have no body to clear, and
      // defined functions with non-LLVM-typed signatures cannot be demoted to
      // a valid external declaration (the decl would still carry non-LLVM
      // param types and fail LLVM IR translation).
      func.erase();
    } else {
      // For defined functions whose body contains non-LLVM ops (but whose
      // signature is LLVM-compatible), demote to an external declaration so
      // that any remaining call sites see a valid (if unresolved) symbol.
      for (Block &block : func.getBody())
        block.dropAllDefinedValueUses();
      func.getBody().dropAllReferences();
      func.getBody().getBlocks().clear();
      func.setLinkage(LLVM::Linkage::External);
    }
  }

  // Also erase LLVM::GlobalOp entries whose type is not LLVM-compatible.
  // These can be cloned in by cloneReferencedDeclarations() when the original
  // global was declared with a non-LLVM type (e.g. a hw.struct global).
  llvm::SmallVector<LLVM::GlobalOp> globalsToErase;
  microModule.walk([&](LLVM::GlobalOp global) {
    if (!LLVM::isCompatibleType(global.getType()))
      globalsToErase.push_back(global);
  });
  for (auto global : globalsToErase) {
    stripped.push_back(global.getSymName().str());
    global.erase();
  }

  // Fix globals whose initializer value attribute is type-mismatched.
  // For example, a covergroup handle global may have type !llvm.ptr but an
  // IntegerAttr(i64, 0) initializer. When LLVM IR translation calls
  // getLLVMConstant(), it tries llvmType->getIntegerBitWidth() on a pointer
  // type, which asserts. Replace such mismatched initializers with ZeroAttr.
  microModule.walk([&](LLVM::GlobalOp global) {
    auto valueAttr = global.getValueOrNull();
    if (!valueAttr)
      return; // No value attribute — OK (uses initializer region or undef).
    // ZeroAttr and UndefAttr are universally safe.
    if (isa<LLVM::ZeroAttr>(valueAttr) || isa<LLVM::UndefAttr>(valueAttr))
      return;
    // StringAttr on array types is handled by LLVM translation.
    if (isa<StringAttr>(valueAttr))
      return;
    // IntegerAttr is only valid when the global type is also integer.
    if (isa<IntegerAttr>(valueAttr) && !isa<IntegerType>(global.getType())) {
      global.setValueAttr(LLVM::ZeroAttr::get(global.getContext()));
      return;
    }
  });

  // Check globals with initializer regions for non-LLVM ops. If the region
  // contains ops from non-LLVM dialects, the convertOperation() call in
  // convertGlobalsAndAliases() will fail. Replace such regions with ZeroAttr.
  microModule.walk([&](LLVM::GlobalOp global) {
    Region &initRegion = global.getInitializerRegion();
    if (initRegion.empty())
      return; // No initializer region — OK.
    bool hasNonLLVM = false;
    initRegion.walk([&](Operation *op) {
      if (!isa<LLVM::LLVMDialect>(op->getDialect())) {
        hasNonLLVM = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasNonLLVM) {
      // Clear the initializer region and set a zero value instead.
      initRegion.getBlocks().clear();
      global.setValueAttr(LLVM::ZeroAttr::get(global.getContext()));
    }
  });

  // Erase any remaining func::FuncOp operations. These should have been
  // lowered to llvm.func by the lowering pipeline; any that remain have
  // non-LLVM types and cannot be translated to LLVM IR.
  llvm::SmallVector<func::FuncOp> funcOpsToErase;
  microModule.walk([&](func::FuncOp funcOp) {
    funcOpsToErase.push_back(funcOp);
  });
  for (auto funcOp : funcOpsToErase) {
    stripped.push_back(funcOp.getSymName().str());
    funcOp.erase();
  }

  return stripped;
}

//===----------------------------------------------------------------------===//
// Trampoline generation for uncompiled functions
//===----------------------------------------------------------------------===//

/// Count how many uint64_t slots a type needs when flattened for trampoline ABI.
/// Returns 0 if the type can't be handled.
static unsigned countTrampolineSlots(Type ty) {
  if (isa<IntegerType>(ty) || isa<LLVM::LLVMPointerType>(ty) ||
      isa<FloatType>(ty))
    return 1;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(ty)) {
    unsigned total = 0;
    for (Type fieldTy : structTy.getBody()) {
      unsigned fieldSlots = countTrampolineSlots(fieldTy);
      if (fieldSlots == 0)
        return 0; // nested unsupported type
      total += fieldSlots;
    }
    return total;
  }
  return 0; // unsupported
}

/// Recursively pack a value into consecutive uint64_t slots in argsArray.
/// Returns the next slot index after packing.
static unsigned emitPackValue(OpBuilder &builder, Location loc, Value val,
                              Type ty, Value argsArray, unsigned slotIdx,
                              Type i64Ty, Type ptrTy) {
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(ty)) {
    for (auto [fieldIdx, fieldTy] : llvm::enumerate(structTy.getBody())) {
      Value field = LLVM::ExtractValueOp::create(
          builder, loc, val, ArrayRef<int64_t>{(int64_t)fieldIdx});
      slotIdx = emitPackValue(builder, loc, field, fieldTy, argsArray, slotIdx,
                              i64Ty, ptrTy);
    }
    return slotIdx;
  }

  // Scalar: pack into one slot.
  Value packed;
  if (isa<LLVM::LLVMPointerType>(ty)) {
    packed = LLVM::PtrToIntOp::create(builder, loc, i64Ty, val);
  } else if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() < 64)
      packed = LLVM::ZExtOp::create(builder, loc, i64Ty, val);
    else if (intTy.getWidth() == 64)
      packed = val;
    else
      packed = LLVM::TruncOp::create(builder, loc, i64Ty, val);
  } else {
    // Float/double: bitcast to i64.
    packed = LLVM::BitcastOp::create(builder, loc, i64Ty, val);
  }

  Value slot;
  if (slotIdx == 0) {
    slot = argsArray;
  } else {
    auto idxVal = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                            builder.getI64IntegerAttr(slotIdx));
    slot = LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, argsArray,
                                ValueRange{idxVal});
  }
  LLVM::StoreOp::create(builder, loc, packed, slot);
  return slotIdx + 1;
}

/// Recursively unpack consecutive uint64_t slots from retsArray into a value of
/// the given type. Returns {value, next_slot_index}.
static std::pair<Value, unsigned>
emitUnpackValue(OpBuilder &builder, Location loc, Type ty, Value retsArray,
                unsigned slotIdx, Type i64Ty, Type ptrTy) {
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(ty)) {
    Value result = LLVM::UndefOp::create(builder, loc, structTy);
    for (auto [fieldIdx, fieldTy] : llvm::enumerate(structTy.getBody())) {
      auto [fieldVal, nextSlot] = emitUnpackValue(builder, loc, fieldTy,
                                                   retsArray, slotIdx, i64Ty,
                                                   ptrTy);
      result = LLVM::InsertValueOp::create(builder, loc, result, fieldVal,
                                            ArrayRef<int64_t>{(int64_t)fieldIdx});
      slotIdx = nextSlot;
    }
    return {result, slotIdx};
  }

  // Scalar: read one slot.
  Value slot;
  if (slotIdx == 0) {
    slot = retsArray;
  } else {
    auto idxVal = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                            builder.getI64IntegerAttr(slotIdx));
    slot = LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, retsArray,
                                ValueRange{idxVal});
  }
  Value raw = LLVM::LoadOp::create(builder, loc, i64Ty, slot);

  Value result;
  if (isa<LLVM::LLVMPointerType>(ty)) {
    result = LLVM::IntToPtrOp::create(builder, loc, ty, raw);
  } else if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() < 64)
      result = LLVM::TruncOp::create(builder, loc, ty, raw);
    else if (intTy.getWidth() == 64)
      result = raw;
    else
      result = LLVM::ZExtOp::create(builder, loc, ty, raw);
  } else {
    result = LLVM::BitcastOp::create(builder, loc, ty, raw);
  }
  return {result, slotIdx + 1};
}

/// Collect vtable FuncId assignments by walking globals with
/// `circt.vtable_entries` in the same order as the interpreter does in
/// LLHDProcessInterpreterGlobals.cpp. This ensures the compiler assigns the
/// same FuncIds as the interpreter (0xF0000000+N where N is sequential).
///
/// Returns the list of function names indexed by FuncId.
static llvm::SmallVector<std::string>
collectVtableFuncIds(ModuleOp module) {
  llvm::SmallVector<std::string> allFuncNames;

  // Walk globals in module iteration order — same as the interpreter does in
  // LLHDProcessInterpreterGlobals.cpp:274-310. Each vtable entry gets a
  // unique FuncId even if the same function name appears multiple times
  // (matching the interpreter's `addressToFunction.size()` counter).
  for (auto globalOp : module.getOps<LLVM::GlobalOp>()) {
    auto vtableEntriesAttr = globalOp->getAttr("circt.vtable_entries");
    if (!vtableEntriesAttr)
      continue;

    auto entriesArray = dyn_cast<ArrayAttr>(vtableEntriesAttr);
    if (!entriesArray)
      continue;

    for (auto entry : entriesArray) {
      auto entryArray = dyn_cast<ArrayAttr>(entry);
      if (!entryArray || entryArray.size() < 2)
        continue;

      auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
      auto funcSymbol = dyn_cast<FlatSymbolRefAttr>(entryArray[1]);
      if (!indexAttr || !funcSymbol)
        continue;

      StringRef funcName = funcSymbol.getValue();
      allFuncNames.push_back(funcName.str());
    }
  }

  return allFuncNames;
}

struct TaggedVtableEntry {
  uint32_t slot = 0;
  uint32_t fid = 0;
};

using TaggedVtableAssignments =
    llvm::StringMap<llvm::SmallVector<TaggedVtableEntry>>;

/// Collect per-vtable tagged FuncId assignments.
/// For each global with `circt.vtable_entries`, records:
///   slot -> (0xF0000000 + fid)
/// where fid is assigned in the same module-order walk used by
/// collectVtableFuncIds()/interpreter vtable initialization.
static TaggedVtableAssignments collectTaggedVtableAssignments(ModuleOp module) {
  TaggedVtableAssignments assignments;
  uint32_t nextFid = 0;

  for (auto globalOp : module.getOps<LLVM::GlobalOp>()) {
    auto entriesAttr =
        dyn_cast_or_null<ArrayAttr>(globalOp->getAttr("circt.vtable_entries"));
    if (!entriesAttr)
      continue;

    auto &entries = assignments[globalOp.getSymName()];
    for (Attribute entryAttr : entriesAttr) {
      auto entryArray = dyn_cast<ArrayAttr>(entryAttr);
      if (!entryArray || entryArray.size() < 2)
        continue;
      auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
      auto funcSymbol = dyn_cast<FlatSymbolRefAttr>(entryArray[1]);
      if (!indexAttr || !funcSymbol)
        continue;
      entries.push_back(
          TaggedVtableEntry{static_cast<uint32_t>(indexAttr.getInt()), nextFid});
      ++nextFid;
    }
  }

  return assignments;
}

/// Materialize tagged FuncId values (0xF0000000+fid) into vtable globals in
/// the LLVM module so compiled code loading from __vtable__ globals dispatches
/// through LowerTaggedIndirectCalls correctly.
static unsigned initializeTaggedVtableGlobals(
    llvm::Module &llvmModule,
    const TaggedVtableAssignments &taggedVtablesByGlobal) {
  const llvm::DataLayout &dl = llvmModule.getDataLayout();
  llvm::LLVMContext &ctx = llvmModule.getContext();
  unsigned ptrBytes = dl.getPointerSize();
  if (ptrBytes == 0)
    ptrBytes = 8;
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);

  unsigned initialized = 0;
  for (const auto &kv : taggedVtablesByGlobal) {
    auto *gv = llvmModule.getGlobalVariable(kv.getKey(),
                                            /*AllowInternal=*/true);
    if (!gv) {
      if (verbose)
        llvm::errs() << "[circt-sim-compile] tagged-vtable: missing global '"
                     << kv.getKey() << "' in LLVM module\n";
      continue;
    }
    auto *arrayTy = dyn_cast<llvm::ArrayType>(gv->getValueType());
    if (!arrayTy) {
      if (verbose)
        llvm::errs() << "[circt-sim-compile] tagged-vtable: global '"
                     << kv.getKey() << "' has non-array type\n";
      continue;
    }

    bool wroteInit = false;
    if (arrayTy->getElementType()->isIntegerTy(8)) {
      // Flattened form: [N x i8], write tagged pointers as little-endian bytes.
      uint64_t numBytes = arrayTy->getNumElements();
      llvm::SmallVector<uint8_t> bytes(numBytes, 0);
      for (const auto &entry : kv.getValue()) {
        uint64_t taggedAddr = 0xF0000000ULL + entry.fid;
        uint64_t byteOffset = static_cast<uint64_t>(entry.slot) * ptrBytes;
        if (byteOffset + ptrBytes > numBytes)
          continue;
        for (unsigned i = 0; i < ptrBytes; ++i)
          bytes[byteOffset + i] =
              static_cast<uint8_t>((taggedAddr >> (i * 8)) & 0xFF);
      }
      gv->setInitializer(llvm::ConstantDataArray::get(ctx, bytes));
      wroteInit = true;
    } else if (auto *elemPtrTy =
                   dyn_cast<llvm::PointerType>(arrayTy->getElementType())) {
      // Typed form: [N x ptr], write tagged addresses as inttoptr constants.
      uint64_t numElems = arrayTy->getNumElements();
      llvm::SmallVector<llvm::Constant *> elems(
          numElems, llvm::ConstantPointerNull::get(elemPtrTy));
      for (const auto &entry : kv.getValue()) {
        if (entry.slot >= numElems)
          continue;
        uint64_t taggedAddr = 0xF0000000ULL + entry.fid;
        auto *taggedInt = llvm::ConstantInt::get(i64Ty, taggedAddr);
        elems[entry.slot] = llvm::ConstantExpr::getIntToPtr(taggedInt, elemPtrTy);
      }
      gv->setInitializer(llvm::ConstantArray::get(arrayTy, elems));
      wroteInit = true;
    }

    if (wroteInit)
      ++initialized;
  }

  return initialized;
}

/// Generate trampoline bodies for external function declarations in the
/// micro-module. Each trampoline packs arguments into a uint64_t array,
/// calls __circt_sim_call_interpreted() to dispatch to the MLIR interpreter,
/// and unpacks the return value.
///
/// This allows compiled code to call functions that weren't compiled (e.g.,
/// functions containing unsupported ops) without crashing at runtime.
static llvm::SmallVector<std::string>
generateTrampolines(ModuleOp microModule) {
  llvm::SmallVector<std::string> trampolineNames;
  auto *mlirCtx = microModule.getContext();
  OpBuilder builder(mlirCtx);
  auto loc = microModule.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(mlirCtx);
  auto i32Ty = IntegerType::get(mlirCtx, 32);
  auto i64Ty = IntegerType::get(mlirCtx, 64);
  auto voidTy = LLVM::LLVMVoidType::get(mlirCtx);

  // Define the __circt_sim_ctx global (set by runtime before calling
  // compiled code). Single-threaded, so a plain global suffices.
  // Use an initializer region returning a null pointer so the symbol is a
  // DEFINITION (not merely an external declaration), allowing dlsym to find it.
  if (!microModule.lookupSymbol<LLVM::GlobalOp>("__circt_sim_ctx")) {
    builder.setInsertionPointToEnd(microModule.getBody());
    auto globalOp =
        LLVM::GlobalOp::create(builder, loc, ptrTy, /*isConstant=*/false,
                               LLVM::Linkage::External, "__circt_sim_ctx",
                               /*value=*/Attribute());
    // Add initializer body: { return null; }
    Block *initBlock = builder.createBlock(&globalOp.getInitializerRegion());
    builder.setInsertionPointToStart(initBlock);
    auto nullVal = LLVM::ZeroOp::create(builder, loc, ptrTy);
    LLVM::ReturnOp::create(builder, loc, nullVal.getResult());
    builder.setInsertionPointToEnd(microModule.getBody());
  }

  // Declare __circt_sim_call_interpreted().
  auto callInterpFuncTy = LLVM::LLVMFunctionType::get(
      voidTy, {ptrTy, i32Ty, ptrTy, i32Ty, ptrTy, i32Ty}, /*isVarArg=*/false);
  auto callInterpDecl = microModule.lookupSymbol<LLVM::LLVMFuncOp>(
      "__circt_sim_call_interpreted");
  if (!callInterpDecl) {
    builder.setInsertionPointToEnd(microModule.getBody());
    callInterpDecl = LLVM::LLVMFuncOp::create(
        builder, loc, "__circt_sim_call_interpreted", callInterpFuncTy);
  }

  // Collect external functions that need trampolines.
  llvm::SmallVector<LLVM::LLVMFuncOp> externals;
  for (auto funcOp : microModule.getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp.isExternal())
      continue;
    auto name = funcOp.getSymName();
    // Skip runtime/system functions (resolved by dynamic linker).
    if (name.starts_with("__circt_sim_") || name.starts_with("__moore_") ||
        name.starts_with("__arc_sched_") || name.starts_with("llvm."))
      continue;
    // Skip vararg functions (can't pack cleanly).
    if (funcOp.isVarArg())
      continue;
    // Skip functions with types we can't flatten into uint64_t slots.
    auto funcTy = funcOp.getFunctionType();
    bool hasUnsupported = false;
    unsigned totalArgSlots = 0;
    for (unsigned i = 0; i < funcTy.getNumParams(); ++i) {
      unsigned slots = countTrampolineSlots(funcTy.getParamType(i));
      if (slots == 0) {
        hasUnsupported = true;
        break;
      }
      totalArgSlots += slots;
    }
    unsigned totalRetSlots = 0;
    if (!hasUnsupported && !isa<LLVM::LLVMVoidType>(funcTy.getReturnType())) {
      totalRetSlots = countTrampolineSlots(funcTy.getReturnType());
      if (totalRetSlots == 0)
        hasUnsupported = true;
    }
    if (hasUnsupported)
      continue;
    externals.push_back(funcOp);
  }

  unsigned funcId = 0;
  for (auto funcOp : externals) {
    auto funcTy = funcOp.getFunctionType();
    unsigned numArgs = funcTy.getNumParams();
    auto retTy = funcTy.getReturnType();
    auto funcLoc = funcOp.getLoc();

    // Recompute flattened slot counts for this function.
    unsigned totalArgSlots = 0;
    for (unsigned i = 0; i < numArgs; ++i)
      totalArgSlots += countTrampolineSlots(funcTy.getParamType(i));
    unsigned totalRetSlots = 0;
    if (!isa<LLVM::LLVMVoidType>(retTy))
      totalRetSlots = countTrampolineSlots(retTy);

    // Add entry block with arguments matching the function signature.
    Region &body = funcOp.getBody();
    Block *entry = new Block();
    body.push_back(entry);
    for (unsigned i = 0; i < numArgs; ++i)
      entry->addArgument(funcTy.getParamType(i), funcLoc);
    funcOp.setLinkage(LLVM::Linkage::External);

    builder.setInsertionPointToStart(entry);

    // Load ctx from __circt_sim_ctx global.
    auto ctxAddr = LLVM::AddressOfOp::create(builder, funcLoc, ptrTy,
                                             "__circt_sim_ctx");
    auto ctx = LLVM::LoadOp::create(builder, funcLoc, ptrTy, ctxAddr);

    // Allocate uint64_t args[totalArgSlots].
    Value argsArray;
    if (totalArgSlots > 0) {
      auto countVal = LLVM::ConstantOp::create(
          builder, funcLoc, i64Ty,
          builder.getI64IntegerAttr(totalArgSlots));
      argsArray = LLVM::AllocaOp::create(builder, funcLoc, ptrTy, i64Ty,
                                          countVal);

      // Pack each argument into the array (recursively for structs).
      unsigned slotIdx = 0;
      for (unsigned i = 0; i < numArgs; ++i) {
        auto arg = entry->getArgument(i);
        slotIdx = emitPackValue(builder, funcLoc, arg,
                                funcTy.getParamType(i), argsArray, slotIdx,
                                i64Ty, ptrTy);
      }
    } else {
      Value zero = LLVM::ConstantOp::create(builder, funcLoc, i64Ty,
                                             builder.getI64IntegerAttr(0));
      argsArray = LLVM::IntToPtrOp::create(builder, funcLoc, ptrTy, zero);
    }

    // Allocate uint64_t rets[totalRetSlots].
    Value retsArray;
    if (totalRetSlots > 0) {
      Value retCountVal = LLVM::ConstantOp::create(
          builder, funcLoc, i64Ty,
          builder.getI64IntegerAttr(totalRetSlots));
      retsArray = LLVM::AllocaOp::create(builder, funcLoc, ptrTy, i64Ty,
                                          retCountVal);
    } else {
      Value zero = LLVM::ConstantOp::create(builder, funcLoc, i64Ty,
                                             builder.getI64IntegerAttr(0));
      retsArray = LLVM::IntToPtrOp::create(builder, funcLoc, ptrTy, zero);
    }

    // Call __circt_sim_call_interpreted(ctx, funcId, args, totalArgSlots,
    //                                   rets, totalRetSlots).
    auto funcIdVal = LLVM::ConstantOp::create(
        builder, funcLoc, i32Ty, builder.getI32IntegerAttr(funcId));
    auto numArgsVal = LLVM::ConstantOp::create(
        builder, funcLoc, i32Ty,
        builder.getI32IntegerAttr(totalArgSlots));
    auto numRetsVal = LLVM::ConstantOp::create(
        builder, funcLoc, i32Ty,
        builder.getI32IntegerAttr(totalRetSlots));
    LLVM::CallOp::create(builder, funcLoc, callInterpDecl,
                          ValueRange{ctx, funcIdVal, argsArray, numArgsVal,
                                     retsArray, numRetsVal});

    // Unpack return value.
    if (totalRetSlots > 0) {
      auto [result, _] = emitUnpackValue(builder, funcLoc, retTy,
                                          retsArray, 0, i64Ty, ptrTy);
      LLVM::ReturnOp::create(builder, funcLoc, result);
    } else {
      LLVM::ReturnOp::create(builder, funcLoc, ValueRange{});
    }

    trampolineNames.push_back(funcOp.getSymName().str());
    ++funcId;
  }

  return trampolineNames;
}

//===----------------------------------------------------------------------===//
// Aggregate ABI flattening: pass aggregates by pointer
//===----------------------------------------------------------------------===//

/// Check if a type is an aggregate that LLVM codegen cannot pass by value.
static bool isAggregateType(Type ty) {
  if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(ty))
    return true;
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth() > 64;
  return false;
}

/// Flatten aggregate function ABIs by converting aggregate arguments and return
/// values to pointer-based passing. Must be called after
/// lowerFuncArithCfToLLVM() so we operate on LLVM dialect ops.
///
/// For aggregate arguments: the parameter type becomes !llvm.ptr and a load is
/// inserted at the function entry. Call sites alloca+store the value and pass
/// the pointer.
///
/// For aggregate return values: an extra !llvm.ptr "sret" parameter is prepended
/// and the function becomes void-returning. The return value is stored through
/// the pointer. Call sites alloca a buffer, pass it, and load the result.
static void flattenAggregateFunctionABIs(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type voidTy = LLVM::LLVMVoidType::get(ctx);

  // Info needed to rewrite a function and its call sites.
  struct FuncRewriteInfo {
    LLVM::LLVMFuncOp funcOp;
    LLVM::LLVMFunctionType oldFuncType;
    LLVM::LLVMFunctionType newFuncType;
    bool hasAggregateReturn;
  };
  llvm::StringMap<FuncRewriteInfo> rewriteMap;

  // Phase 1: Identify non-external functions needing rewrite.
  moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (funcOp.isExternal())
      return;
    auto funcType = funcOp.getFunctionType();
    bool needsRewrite = false;
    for (auto argTy : funcType.getParams())
      if (isAggregateType(argTy)) {
        needsRewrite = true;
        break;
      }
    Type retType = funcType.getReturnType();
    bool hasAggRet = !isa<LLVM::LLVMVoidType>(retType) && isAggregateType(retType);
    if (hasAggRet)
      needsRewrite = true;
    if (!needsRewrite)
      return;

    // Build the new parameter list.
    SmallVector<Type> newParams;
    if (hasAggRet)
      newParams.push_back(ptrTy);
    for (auto argTy : funcType.getParams())
      newParams.push_back(isAggregateType(argTy) ? ptrTy : argTy);

    Type newRetType = hasAggRet ? voidTy : retType;
    auto newFuncType =
        LLVM::LLVMFunctionType::get(newRetType, newParams, funcType.isVarArg());
    rewriteMap[funcOp.getName()] = {funcOp, funcType, newFuncType, hasAggRet};
  });

  if (rewriteMap.empty())
    return;

  // Phase 2: Rewrite function bodies and signatures.
  for (auto &kv : rewriteMap) {
    auto &info = kv.second;
    auto &funcOp = info.funcOp;
    auto &oldFuncType = info.oldFuncType;
    Block &entry = funcOp.getBody().front();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&entry);

    // If the function had an aggregate return, prepend an sret pointer arg.
    Value retBuf;
    if (info.hasAggregateReturn)
      retBuf = entry.insertArgument(0u, ptrTy, funcOp.getLoc());

    // Replace each aggregate argument with a pointer + load.
    unsigned offset = info.hasAggregateReturn ? 1 : 0;
    for (unsigned i = 0; i < oldFuncType.getNumParams(); ++i) {
      Type oldTy = oldFuncType.getParams()[i];
      if (!isAggregateType(oldTy))
        continue;
      BlockArgument arg = entry.getArgument(i + offset);
      arg.setType(ptrTy);
      Value loaded = builder.create<LLVM::LoadOp>(funcOp.getLoc(), oldTy, arg);
      arg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
    }

    // Rewrite return ops for aggregate returns.
    if (info.hasAggregateReturn) {
      SmallVector<LLVM::ReturnOp> returns;
      funcOp.walk([&](LLVM::ReturnOp ret) { returns.push_back(ret); });
      for (auto ret : returns) {
        OpBuilder rb(ret);
        if (ret.getNumOperands() > 0)
          rb.create<LLVM::StoreOp>(ret.getLoc(), ret.getOperand(0), retBuf);
        rb.create<LLVM::ReturnOp>(ret.getLoc(), ValueRange{});
        ret.erase();
      }
    }

    // Update the function's type to the new signature.
    funcOp.setFunctionType(info.newFuncType);
  }

  // Phase 3: Rewrite call sites.
  SmallVector<LLVM::CallOp> callsToRewrite;
  moduleOp.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee)
      return; // Indirect call — skip.
    if (rewriteMap.count(*callee))
      callsToRewrite.push_back(callOp);
  });

  for (auto callOp : callsToRewrite) {
    auto callee = *callOp.getCallee();
    auto &info = rewriteMap[callee];
    auto &oldFuncType = info.oldFuncType;
    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    // Constant 1 for alloca sizes.
    Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                 builder.getI64IntegerAttr(1));

    SmallVector<Value> newOperands;

    // If the function had an aggregate return, alloca a result buffer.
    Value retBuf;
    if (info.hasAggregateReturn) {
      Type retTy = oldFuncType.getReturnType();
      retBuf = builder.create<LLVM::AllocaOp>(loc, ptrTy, retTy, one);
      newOperands.push_back(retBuf);
    }

    // Process each argument.
    for (unsigned i = 0; i < oldFuncType.getNumParams(); ++i) {
      Value operand = callOp.getOperand(i);
      Type oldArgTy = oldFuncType.getParams()[i];
      if (isAggregateType(oldArgTy)) {
        // Alloca + store + pass pointer.
        Value buf = builder.create<LLVM::AllocaOp>(loc, ptrTy, oldArgTy, one);
        builder.create<LLVM::StoreOp>(loc, operand, buf);
        newOperands.push_back(buf);
      } else {
        newOperands.push_back(operand);
      }
    }

    // Create the replacement call.
    if (info.hasAggregateReturn) {
      // Function now returns void; load result from buffer.
      builder.create<LLVM::CallOp>(loc, TypeRange{}, callee, newOperands);
      Value result =
          builder.create<LLVM::LoadOp>(loc, oldFuncType.getReturnType(), retBuf);
      callOp.getResult().replaceAllUsesWith(result);
    } else {
      // Return type unchanged — just update operands.
      auto newCall = builder.create<LLVM::CallOp>(loc, callOp.getResultTypes(),
                                                  callee, newOperands);
      // LLVM::CallOp has at most one result; use getResult() with no args.
      if (callOp.getNumResults() > 0)
        callOp.getResult().replaceAllUsesWith(newCall.getResult());
    }
    callOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Process body compilation (Phase A: callback processes)
//===----------------------------------------------------------------------===//

/// Check if an llhd.process op can be compiled as a callback function.
/// Phase A requirements:
///   - At most 1 llhd.wait (0 = one-shot/halt, 1 = callback loop)
///   - No yield operands on the wait (no process results)
///   - No wait dest operands (no loop-carried state through wait)
///   - All probed/driven signals are integer type ≤64 bits with known IDs
///   - Only supported ops (llhd, arith, cf, func, LLVM, comb, hw.constant)
static bool isProcessCallbackEligible(
    llhd::ProcessOp procOp,
    const DenseMap<Value, uint32_t> &signalIdMap,
    std::string *rejectionReason = nullptr) {
  unsigned waitCount = 0;
  bool eligible = true;

  procOp.walk([&](Operation *op) -> WalkResult {
    if (auto waitOp = dyn_cast<llhd::WaitOp>(op)) {
      ++waitCount;
      if (waitCount > 1) {
        if (rejectionReason)
          *rejectionReason = "multiple_waits";
        eligible = false;
        return WalkResult::interrupt();
      }
      if (!waitOp.getYieldOperands().empty() ||
          !waitOp.getDestOperands().empty()) {
        if (rejectionReason)
          *rejectionReason = "wait_with_operands";
        eligible = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto prbOp = dyn_cast<llhd::ProbeOp>(op)) {
      auto resultTy = prbOp.getResult().getType();
      bool ok = false;
      if (auto intTy = dyn_cast<IntegerType>(resultTy))
        ok = intTy.getWidth() <= 64;
      else if (auto w = getFourStateValueWidth(resultTy))
        ok = *w <= 64;
      if (!ok) {
        if (rejectionReason)
          *rejectionReason = "probe_wide_signal";
        eligible = false;
        return WalkResult::interrupt();
      }
      if (!signalIdMap.count(prbOp.getSignal())) {
        if (rejectionReason)
          *rejectionReason = "probe_unknown_signal";
        eligible = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto drvOp = dyn_cast<llhd::DriveOp>(op)) {
      auto valTy = drvOp.getValue().getType();
      bool ok = false;
      if (auto intTy = dyn_cast<IntegerType>(valTy))
        ok = intTy.getWidth() <= 64;
      else if (auto w = getFourStateValueWidth(valTy))
        ok = *w <= 64;
      if (!ok) {
        if (rejectionReason)
          *rejectionReason = "drive_wide_signal";
        eligible = false;
        return WalkResult::interrupt();
      }
      if (!signalIdMap.count(drvOp.getSignal())) {
        if (rejectionReason)
          *rejectionReason = "drive_unknown_signal";
        eligible = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (isa<llhd::HaltOp, llhd::IntToTimeOp, llhd::ConstantTimeOp,
            llhd::ProcessOp>(op))
      return WalkResult::advance();
    // Allow safe cast forms that the main lowering pipeline already resolves.
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      bool allResultsDead = true;
      for (Value result : castOp.getResults())
        allResultsDead &= result.use_empty();
      if (allResultsDead)
        return WalkResult::advance();
      if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
        Type inTy = castOp.getOperand(0).getType();
        Type outTy = castOp.getResult(0).getType();
        if (inTy == outTy)
          return WalkResult::advance();
        // func.call_indirect lowering uses ptr -> function casts.
        if (isa<LLVM::LLVMPointerType>(inTy) && isa<FunctionType>(outTy))
          return WalkResult::advance();
        if ((isa<LLVM::LLVMPointerType>(inTy) && isa<IntegerType>(outTy)) ||
            (isa<IntegerType>(inTy) && isa<LLVM::LLVMPointerType>(outTy)) ||
            (isa<LLVM::LLVMPointerType>(inTy) &&
             isa<LLVM::LLVMPointerType>(outTy)) ||
            (isa<IntegerType>(inTy) && isa<IntegerType>(outTy)))
          return WalkResult::advance();
      }
      if (rejectionReason)
        *rejectionReason = op->getName().getStringRef().str();
      eligible = false;
      return WalkResult::interrupt();
    }
    if (isa<arith::ArithDialect, cf::ControlFlowDialect, LLVM::LLVMDialect,
            func::FuncDialect>(op->getDialect()))
      return WalkResult::advance();
    if (isa<comb::CombDialect>(op->getDialect())) {
      if (isa<comb::XorOp, comb::AndOp, comb::OrOp, comb::AddOp,
              comb::SubOp, comb::MulOp, comb::ICmpOp, comb::MuxOp,
              // Phase K: shifts, extract, concat, replicate, div/mod, parity
              comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
              comb::ShlOp, comb::ShrUOp, comb::ShrSOp,
              comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp,
              comb::ParityOp>(op))
        return WalkResult::advance();
      if (rejectionReason)
        *rejectionReason = op->getName().getStringRef().str();
      eligible = false;
      return WalkResult::interrupt();
    }
    if (isa<hw::ConstantOp, hw::StructExtractOp, hw::StructCreateOp,
            hw::AggregateConstantOp,
            // Phase K: array and bitcast ops
            hw::ArrayGetOp, hw::ArrayCreateOp, hw::BitcastOp>(op))
      return WalkResult::advance();
    if (rejectionReason)
      *rejectionReason = op->getName().getStringRef().str();
    eligible = false;
    return WalkResult::interrupt();
  });

  return eligible;
}

/// Lower a variadic comb op to chained binary arith ops.
template <typename ArithOp>
static Value lowerVariadicToArith(OperandRange operands, OpBuilder &builder,
                                  Location loc, IRMapping &mapping) {
  if (operands.empty())
    return nullptr;
  Value result = mapping.lookupOrDefault(operands[0]);
  for (unsigned i = 1; i < operands.size(); ++i) {
    Value rhs = mapping.lookupOrDefault(operands[i]);
    result = ArithOp::create(builder, loc, result, rhs);
  }
  return result;
}

/// Convert a comb::ICmpPredicate to arith::CmpIPredicate.
/// Returns std::nullopt for predicates without an arith equivalent (ceq, etc.).
static std::optional<arith::CmpIPredicate>
convertCombPredicate(comb::ICmpPredicate pred) {
  switch (pred) {
  case comb::ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case comb::ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case comb::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case comb::ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case comb::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case comb::ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case comb::ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case comb::ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case comb::ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case comb::ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  default:
    return std::nullopt;
  }
}

/// Extract eligible llhd.process bodies into func.func ops with all
/// LLHD/comb/hw ops lowered to arith/cf/func ops. The resulting functions
/// are ready for the standard compilation pipeline.
///
/// Phase A: Only compiles the "body" portion of callback processes (the blocks
/// between the resume point and the next wait). The init path is left for the
/// interpreter. Each compiled function has signature void(ptr ctx, ptr frame).
static unsigned compileProcessBodies(
    ModuleOp module,
    llvm::SmallVectorImpl<std::string> &procNames,
    llvm::SmallVectorImpl<uint8_t> &procKinds,
    unsigned *totalProcessCount = nullptr,
    llvm::StringMap<unsigned> *procRejectionReasons = nullptr) {
  auto *mlirCtx = module.getContext();
  auto loc = module.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(mlirCtx);
  auto i32Ty = IntegerType::get(mlirCtx, 32);
  auto i64Ty = IntegerType::get(mlirCtx, 64);
  auto i8Ty = IntegerType::get(mlirCtx, 8);

  // Step 1: Assign signal IDs. Walk all llhd.sig ops in module order.
  // Runtime (ProcessScheduler) uses 1-based signal IDs (ID 0 is sentinel),
  // so we must start at 1 to match.
  DenseMap<Value, uint32_t> signalIdMap;
  uint32_t nextSigId = 1;
  module.walk([&](llhd::SignalOp sigOp) {
    signalIdMap[sigOp.getResult()] = nextSigId++;
  });

  if (nextSigId == 1)
    return 0;

  // Step 2: Declare runtime functions in the parent module.
  OpBuilder moduleBuilder(mlirCtx);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  auto getOrDeclareFunc = [&](StringRef name, FunctionType funcTy) {
    if (module.lookupSymbol<func::FuncOp>(name))
      return;
    auto decl = func::FuncOp::create(moduleBuilder, loc, name, funcTy);
    decl.setVisibility(SymbolTable::Visibility::Private);
  };

  getOrDeclareFunc("__circt_sim_signal_read_u64",
                   FunctionType::get(mlirCtx, {ptrTy, i32Ty}, {i64Ty}));
  getOrDeclareFunc("__circt_sim_signal_drive_u64",
                   FunctionType::get(mlirCtx, {ptrTy, i32Ty, i64Ty, i8Ty,
                                               i64Ty},
                                     {}));
  // 4-state narrow fast lane: read/drive {value, xz} as u64 pair.
  getOrDeclareFunc("__circt_sim_signal_read4_u64",
                   FunctionType::get(mlirCtx,
                                     {ptrTy, i32Ty, ptrTy, ptrTy}, {}));
  getOrDeclareFunc("__circt_sim_signal_drive4_u64",
                   FunctionType::get(mlirCtx,
                                     {ptrTy, i32Ty, i64Ty, i64Ty, i8Ty,
                                      i64Ty},
                                     {}));
  // Direct signal memory access: get hot data pointer.
  getOrDeclareFunc("__circt_sim_get_hot",
                   FunctionType::get(mlirCtx, {ptrTy}, {ptrTy}));
  // Specialized drive entry points (avoid delay_kind branching).
  getOrDeclareFunc("__circt_sim_drive_delta",
                   FunctionType::get(mlirCtx, {ptrTy, i32Ty, i64Ty}, {}));
  getOrDeclareFunc("__circt_sim_drive_nba",
                   FunctionType::get(mlirCtx, {ptrTy, i32Ty, i64Ty}, {}));
  getOrDeclareFunc("__circt_sim_drive_time",
                   FunctionType::get(mlirCtx, {ptrTy, i32Ty, i64Ty, i64Ty},
                                     {}));

  // Step 3: Walk processes and extract eligible ones.
  unsigned compiled = 0;
  llvm::SmallVector<llhd::ProcessOp> processes;
  module.walk([&](llhd::ProcessOp procOp) {
    processes.push_back(procOp);
  });

  if (totalProcessCount)
    *totalProcessCount = processes.size();

  // Pre-compute canonical names for all processes.
  // Format: "<hw_module_sym>.process_<local_index>" where local_index is
  // the 0-based position among all llhd.process ops in that hw.module.
  // This must match the naming in LLHDProcessInterpreter::registerProcess().
  DenseMap<Operation *, std::string> canonicalProcNames;
  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    std::string moduleSym = hwMod.getSymName().str();
    unsigned localIndex = 0;
    for (auto &op : hwMod.getBodyBlock()->getOperations()) {
      if (isa<llhd::ProcessOp>(op)) {
        canonicalProcNames[&op] =
            moduleSym + ".process_" + std::to_string(localIndex);
        localIndex++;
      }
    }
  }

  for (auto procOp : processes) {
    std::string reason;
    if (!isProcessCallbackEligible(procOp, signalIdMap,
                                   procRejectionReasons ? &reason : nullptr)) {
      if (procRejectionReasons && !reason.empty())
        ++(*procRejectionReasons)[reason];
      continue;
    }

    // Find the wait op (if any).
    llhd::WaitOp waitOp = nullptr;
    procOp.walk([&](llhd::WaitOp w) { waitOp = w; });

    Region &procRegion = procOp.getBody();
    Block *waitBlock = waitOp ? waitOp->getBlock() : nullptr;

    // Determine body blocks.
    llvm::SmallVector<Block *> bodyBlocks;
    if (!waitOp) {
      // One-shot process (no wait): all blocks are the body.
      for (Block &block : procRegion)
        bodyBlocks.push_back(&block);
    } else {
      // Callback loop: collect blocks reachable from resumeBlock,
      // stopping at back-edges to waitBlock.
      Block *resumeBlock = waitOp.getDest();
      llvm::SmallPtrSet<Block *, 8> visited;
      llvm::SmallVector<Block *, 8> worklist;
      worklist.push_back(resumeBlock);
      while (!worklist.empty()) {
        Block *b = worklist.pop_back_val();
        if (!visited.insert(b).second)
          continue;
        bodyBlocks.push_back(b);
        for (Block *succ : b->getSuccessors()) {
          if (succ != waitBlock)
            worklist.push_back(succ);
        }
      }
    }

    if (bodyBlocks.empty())
      continue;

    // Skip if any body block references values from non-body process blocks.
    llvm::SmallPtrSet<Block *, 8> bodyBlockSet(bodyBlocks.begin(),
                                               bodyBlocks.end());
    bool hasInvalidRef = false;
    for (Block *b : bodyBlocks) {
      for (Operation &op : *b) {
        for (Value operand : op.getOperands()) {
          if (auto *defBlock = operand.getParentBlock())
            if (bodyBlockSet.contains(defBlock))
              continue;
          if (!procRegion.isAncestor(operand.getParentRegion()))
            continue; // Module-level external — handled separately.
          hasInvalidRef = true;
          break;
        }
        if (hasInvalidRef)
          break;
      }
      if (hasInvalidRef)
        break;
    }
    if (hasInvalidRef)
      if (procRejectionReasons)
        ++(*procRejectionReasons)["process_body_external_value"];
    if (hasInvalidRef)
      continue;

    // Create function: void(ptr ctx, ptr frame).
    // Use canonical module-scoped name so the runtime can match this compiled
    // function to the corresponding interpreter process by name.
    auto it = canonicalProcNames.find(procOp.getOperation());
    std::string funcName = it != canonicalProcNames.end()
                               ? it->second
                               : "__circt_sim_proc_" + std::to_string(compiled);
    auto funcType = FunctionType::get(mlirCtx, {ptrTy, ptrTy}, {});
    auto funcOp = func::FuncOp::create(moduleBuilder, loc, funcName, funcType);
    funcOp.setVisibility(SymbolTable::Visibility::Private);

    Block *entry = funcOp.addEntryBlock();
    Value ctxArg = entry->getArgument(0);

    // Create cloned blocks in the function.
    IRMapping mapping;
    // Track 4-state struct values as {value, unknown} component pairs.
    // Keys are ORIGINAL (source) SSA values from the process body.
    DenseMap<Value, std::pair<Value, Value>> fourStateComponents;
    DenseMap<Block *, Block *> blockMap;
    for (Block *srcBlock : bodyBlocks) {
      Block *newBlock = new Block();
      funcOp.getBody().push_back(newBlock);
      blockMap[srcBlock] = newBlock;
      for (auto arg : srcBlock->getArguments())
        mapping.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));
    }

    // Entry: get hot data pointer for direct signal memory access,
    // then branch to first body block.
    OpBuilder builder(entry, entry->begin());

    // %hot = call @__circt_sim_get_hot(%ctx) → CirctSimHot*
    auto hotCall = func::CallOp::create(
        builder, loc, "__circt_sim_get_hot", TypeRange{ptrTy},
        ValueRange{ctxArg});
    Value hotPtr = hotCall.getResult(0);
    // sig2_base is the first field of CirctSimHot (offset 0).
    // load ptr from hotPtr gives us the uint64_t* signal memory base.
    Value sig2Base = LLVM::LoadOp::create(builder, loc, ptrTy, hotPtr);

    cf::BranchOp::create(builder, loc, blockMap[bodyBlocks[0]]);

    // Pre-scan: clone external constants into entry block.
    bool extractionFailed = false;
    std::string extractionFailKind;
    for (Block *srcBlock : bodyBlocks) {
      for (Operation &op : *srcBlock) {
        // Skip operands of ops that will be replaced entirely (wait/halt).
        if (isa<llhd::WaitOp, llhd::HaltOp>(&op))
          continue;
        for (Value operand : op.getOperands()) {
          if (procRegion.isAncestor(operand.getParentRegion()))
            continue;
          if (mapping.contains(operand))
            continue;
          if (isa_and_nonnull<llhd::SignalOp>(operand.getDefiningOp()) ||
              isa_and_nonnull<llhd::ConstantTimeOp>(operand.getDefiningOp()))
            continue; // Handled during op lowering.

          builder.setInsertionPoint(entry->getTerminator());
          auto failWith = [&](llvm::StringRef reason) {
            extractionFailed = true;
            extractionFailKind = reason.str();
          };
          if (auto hwConst = operand.getDefiningOp<hw::ConstantOp>()) {
            auto c = arith::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(hwConst.getType(),
                                       hwConst.getValue()));
            mapping.map(operand, c.getResult());
          } else if (auto arithConst =
                         operand.getDefiningOp<arith::ConstantOp>()) {
            auto *cloned = builder.clone(*arithConst);
            mapping.map(operand, cloned->getResult(0));
          } else if (auto llvmConst =
                         operand.getDefiningOp<LLVM::ConstantOp>()) {
            auto *cloned = builder.clone(*llvmConst);
            mapping.map(operand, cloned->getResult(0));
          } else if (auto addrOf = operand.getDefiningOp<LLVM::AddressOfOp>()) {
            auto *cloned = builder.clone(*addrOf);
            mapping.map(operand, cloned->getResult(0));
          } else if (auto zeroOp = operand.getDefiningOp<LLVM::ZeroOp>()) {
            auto *cloned = builder.clone(*zeroOp);
            mapping.map(operand, cloned->getResult(0));
          } else if (auto undefOp = operand.getDefiningOp<LLVM::UndefOp>()) {
            auto *cloned = builder.clone(*undefOp);
            mapping.map(operand, cloned->getResult(0));
          } else if (auto aggConst =
                         operand.getDefiningOp<hw::AggregateConstantOp>()) {
            // Decompose 4-state aggregate constant into {value, unknown} pair.
            if (auto w = getFourStateValueWidth(aggConst.getType())) {
              auto fields = aggConst.getFields();
              auto valAttr = cast<IntegerAttr>(fields[0]);
              auto unkAttr = cast<IntegerAttr>(fields[1]);
              auto valConst = arith::ConstantOp::create(builder, loc, valAttr);
              auto unkConst = arith::ConstantOp::create(builder, loc, unkAttr);
              fourStateComponents[operand] = {valConst, unkConst};
              // Map to a dummy — actual uses go through fourStateComponents.
              mapping.map(operand, valConst.getResult());
            } else {
              failWith("hw.aggregate_constant(non-fourstate)");
              break;
            }
          } else {
            if (Operation *def = operand.getDefiningOp())
              failWith(def->getName().getStringRef());
            else
              failWith("block_argument");
            break;
          }
        }
        if (extractionFailed)
          break;
      }
      if (extractionFailed)
        break;
    }
    if (extractionFailed) {
      if (procRejectionReasons) {
        std::string reason = "process_extract_external_operand";
        if (!extractionFailKind.empty())
          reason += ":" + extractionFailKind;
        ++(*procRejectionReasons)[reason];
      }
      funcOp.erase();
      continue;
    }

    // Clone and lower ops from body blocks.
    bool loweringFailed = false;
    std::string loweringFailOpName;
    for (Block *srcBlock : bodyBlocks) {
      Block *dstBlock = blockMap[srcBlock];
      builder.setInsertionPointToEnd(dstBlock);

      for (Operation &op : *srcBlock) {
        loweringFailOpName = op.getName().getStringRef().str();
        auto opLoc = op.getLoc();

        // === LLHD ops ===
        if (auto prbOp = dyn_cast<llhd::ProbeOp>(&op)) {
          uint32_t sigId = signalIdMap.lookup(prbOp.getSignal());
          auto sigIdVal = arith::ConstantOp::create(
              builder, opLoc, builder.getI32IntegerAttr(sigId));
          auto resultTy = prbOp.getResult().getType();

          if (auto w = getFourStateValueWidth(resultTy)) {
            // 4-state narrow fast lane: physical width is 2*N.
            unsigned logW = *w;
            if (2 * logW <= 64) {
              // Fits in one u64: direct memory load from sig2_base.
              auto fourStateSigI64 = LLVM::ConstantOp::create(
                  builder, opLoc, i64Ty,
                  builder.getI64IntegerAttr(sigId));
              auto fourStateElemPtr = LLVM::GEPOp::create(
                  builder, opLoc, ptrTy, i64Ty, sig2Base,
                  ValueRange{fourStateSigI64});
              Value raw = LLVM::LoadOp::create(builder, opLoc, i64Ty,
                                               fourStateElemPtr);
              // HW struct bit order: value=high N bits, unknown=low N bits.
              auto intLogTy = IntegerType::get(mlirCtx, logW);
              Value xz = raw;
              if (logW < 64)
                xz = arith::TruncIOp::create(builder, opLoc, intLogTy, raw);
              Value shifted = arith::ShRUIOp::create(
                  builder, opLoc, raw,
                  arith::ConstantOp::create(
                      builder, opLoc,
                      builder.getI64IntegerAttr(logW)));
              Value val = shifted;
              if (logW < 64)
                val = arith::TruncIOp::create(builder, opLoc, intLogTy,
                                              shifted);
              fourStateComponents[prbOp.getResult()] = {val, xz};
              mapping.map(prbOp.getResult(), val); // dummy
            } else {
              // 32 < N <= 64: use read4_u64 with alloca out-params.
              auto one = arith::ConstantOp::create(
                  builder, opLoc, builder.getI64IntegerAttr(1));
              auto valAlloca =
                  LLVM::AllocaOp::create(builder, opLoc, ptrTy, i64Ty, one);
              auto xzAlloca =
                  LLVM::AllocaOp::create(builder, opLoc, ptrTy, i64Ty, one);
              func::CallOp::create(
                  builder, opLoc, "__circt_sim_signal_read4_u64", TypeRange{},
                  ValueRange{ctxArg, sigIdVal, valAlloca, xzAlloca});
              Value valRaw =
                  LLVM::LoadOp::create(builder, opLoc, i64Ty, valAlloca);
              Value xzRaw =
                  LLVM::LoadOp::create(builder, opLoc, i64Ty, xzAlloca);
              auto intLogTy = IntegerType::get(mlirCtx, logW);
              Value val = (logW < 64)
                              ? (Value)arith::TruncIOp::create(
                                    builder, opLoc, intLogTy, valRaw)
                              : valRaw;
              Value xz = (logW < 64)
                             ? (Value)arith::TruncIOp::create(
                                   builder, opLoc, intLogTy, xzRaw)
                             : xzRaw;
              fourStateComponents[prbOp.getResult()] = {val, xz};
              mapping.map(prbOp.getResult(), val); // dummy
            }
          } else {
            // 2-state integer signal: direct memory load from sig2_base.
            // %ptr = gep i64, sig2_base, sigId
            // %raw = load i64, %ptr
            auto sigIdI64 = LLVM::ConstantOp::create(
                builder, opLoc, i64Ty,
                builder.getI64IntegerAttr(sigId));
            auto elemPtr = LLVM::GEPOp::create(
                builder, opLoc, ptrTy, i64Ty, sig2Base,
                ValueRange{sigIdI64});
            Value result = LLVM::LoadOp::create(builder, opLoc, i64Ty,
                                                elemPtr);
            if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
              if (intTy.getWidth() < 64)
                result =
                    arith::TruncIOp::create(builder, opLoc, intTy, result);
            }
            mapping.map(prbOp.getResult(), result);
          }
          continue;
        }

        if (auto drvOp = dyn_cast<llhd::DriveOp>(&op)) {
          uint32_t sigId = signalIdMap.lookup(drvOp.getSignal());
          auto sigIdVal = arith::ConstantOp::create(
              builder, opLoc, builder.getI32IntegerAttr(sigId));
          auto valTy = drvOp.getValue().getType();

          if (auto w = getFourStateValueWidth(valTy)) {
            // 4-state narrow drive.
            unsigned logW = *w;
            auto it = fourStateComponents.find(drvOp.getValue());
            if (it == fourStateComponents.end()) {
              loweringFailed = true;
              break;
            }
            Value valComp = it->second.first;
            Value xzComp = it->second.second;

            if (2 * logW <= 64) {
              // Pack into single u64: (val << N) | xz.
              Value valExt = valComp;
              Value xzExt = xzComp;
              if (logW < 64) {
                valExt =
                    arith::ExtUIOp::create(builder, opLoc, i64Ty, valComp);
                xzExt =
                    arith::ExtUIOp::create(builder, opLoc, i64Ty, xzComp);
              }
              Value shifted = arith::ShLIOp::create(
                  builder, opLoc, valExt,
                  arith::ConstantOp::create(
                      builder, opLoc,
                      builder.getI64IntegerAttr(logW)));
              Value packed =
                  arith::OrIOp::create(builder, opLoc, shifted, xzExt);
              // Specialized delta drive: zero-delay, no delay_kind branching.
              func::CallOp::create(
                  builder, opLoc, "__circt_sim_drive_delta", TypeRange{},
                  ValueRange{ctxArg, sigIdVal, packed});
            } else {
              // Use drive4_u64 for wider 4-state signals (separate val/xz).
              auto delayKind = arith::ConstantOp::create(
                  builder, opLoc, builder.getI8IntegerAttr(0));
              auto delayVal = arith::ConstantOp::create(
                  builder, opLoc, builder.getI64IntegerAttr(0));
              Value valExt = (logW < 64)
                                 ? (Value)arith::ExtUIOp::create(
                                       builder, opLoc, i64Ty, valComp)
                                 : valComp;
              Value xzExt = (logW < 64)
                                ? (Value)arith::ExtUIOp::create(
                                      builder, opLoc, i64Ty, xzComp)
                                : xzComp;
              func::CallOp::create(
                  builder, opLoc, "__circt_sim_signal_drive4_u64", TypeRange{},
                  ValueRange{ctxArg, sigIdVal, valExt, xzExt, delayKind,
                             delayVal});
            }
          } else {
            // 2-state integer drive: specialized delta drive.
            Value val = mapping.lookupOrDefault(drvOp.getValue());
            if (auto intTy = dyn_cast<IntegerType>(val.getType())) {
              if (intTy.getWidth() < 64)
                val = arith::ExtUIOp::create(builder, opLoc, i64Ty, val);
            }
            func::CallOp::create(
                builder, opLoc, "__circt_sim_drive_delta", TypeRange{},
                ValueRange{ctxArg, sigIdVal, val});
          }
          continue;
        }

        if (isa<llhd::WaitOp>(&op)) {
          func::ReturnOp::create(builder, opLoc);
          continue;
        }

        if (isa<llhd::HaltOp>(&op)) {
          func::ReturnOp::create(builder, opLoc);
          continue;
        }

        if (auto intToTimeOp = dyn_cast<llhd::IntToTimeOp>(&op)) {
          mapping.map(intToTimeOp.getResult(),
                      mapping.lookupOrDefault(intToTimeOp.getInput()));
          continue;
        }

        if (auto ctOp = dyn_cast<llhd::ConstantTimeOp>(&op)) {
          auto c = arith::ConstantOp::create(builder, opLoc,
                                             builder.getI64IntegerAttr(0));
          mapping.map(ctOp.getResult(), c.getResult());
          continue;
        }

        // === Comb ops ===
        if (auto xorOp = dyn_cast<comb::XorOp>(&op)) {
          auto r = lowerVariadicToArith<arith::XOrIOp>(
              xorOp.getOperands(), builder, opLoc, mapping);
          if (!r) {
            loweringFailed = true;
            break;
          }
          mapping.map(xorOp.getResult(), r);
          continue;
        }

        if (auto andOp = dyn_cast<comb::AndOp>(&op)) {
          auto r = lowerVariadicToArith<arith::AndIOp>(
              andOp.getOperands(), builder, opLoc, mapping);
          if (!r) {
            loweringFailed = true;
            break;
          }
          mapping.map(andOp.getResult(), r);
          continue;
        }

        if (auto orOp = dyn_cast<comb::OrOp>(&op)) {
          auto r = lowerVariadicToArith<arith::OrIOp>(
              orOp.getOperands(), builder, opLoc, mapping);
          if (!r) {
            loweringFailed = true;
            break;
          }
          mapping.map(orOp.getResult(), r);
          continue;
        }

        if (auto addOp = dyn_cast<comb::AddOp>(&op)) {
          auto r = lowerVariadicToArith<arith::AddIOp>(
              addOp.getOperands(), builder, opLoc, mapping);
          if (!r) {
            loweringFailed = true;
            break;
          }
          mapping.map(addOp.getResult(), r);
          continue;
        }

        if (auto mulOp = dyn_cast<comb::MulOp>(&op)) {
          auto r = lowerVariadicToArith<arith::MulIOp>(
              mulOp.getOperands(), builder, opLoc, mapping);
          if (!r) {
            loweringFailed = true;
            break;
          }
          mapping.map(mulOp.getResult(), r);
          continue;
        }

        if (auto subOp = dyn_cast<comb::SubOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(subOp.getLhs());
          auto rhs = mapping.lookupOrDefault(subOp.getRhs());
          auto r = arith::SubIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(subOp.getResult(), r);
          continue;
        }

        if (auto icmpOp = dyn_cast<comb::ICmpOp>(&op)) {
          auto pred = convertCombPredicate(icmpOp.getPredicate());
          if (!pred) {
            loweringFailed = true;
            break;
          }
          auto lhs = mapping.lookupOrDefault(icmpOp.getLhs());
          auto rhs = mapping.lookupOrDefault(icmpOp.getRhs());
          auto r = arith::CmpIOp::create(builder, opLoc, *pred, lhs, rhs);
          mapping.map(icmpOp.getResult(), r);
          continue;
        }

        if (auto muxOp = dyn_cast<comb::MuxOp>(&op)) {
          auto cond = mapping.lookupOrDefault(muxOp.getCond());
          // Check if this mux operates on 4-state values.
          auto trueIt = fourStateComponents.find(muxOp.getTrueValue());
          auto falseIt = fourStateComponents.find(muxOp.getFalseValue());
          if (trueIt != fourStateComponents.end() &&
              falseIt != fourStateComponents.end()) {
            // 4-state mux: select each component independently.
            Value valR = arith::SelectOp::create(
                builder, opLoc, cond, trueIt->second.first,
                falseIt->second.first);
            Value xzR = arith::SelectOp::create(
                builder, opLoc, cond, trueIt->second.second,
                falseIt->second.second);
            fourStateComponents[muxOp.getResult()] = {valR, xzR};
            mapping.map(muxOp.getResult(), valR); // dummy
          } else {
            auto tv = mapping.lookupOrDefault(muxOp.getTrueValue());
            auto fv = mapping.lookupOrDefault(muxOp.getFalseValue());
            auto r = arith::SelectOp::create(builder, opLoc, cond, tv, fv);
            mapping.map(muxOp.getResult(), r);
          }
          continue;
        }

        // === Phase K: additional comb ops ===

        // comb.extract → arith.shrui + arith.trunci
        if (auto extractOp = dyn_cast<comb::ExtractOp>(&op)) {
          auto input = mapping.lookupOrDefault(extractOp.getInput());
          unsigned lowBit = extractOp.getLowBit();
          unsigned resultWidth =
              cast<IntegerType>(extractOp.getType()).getWidth();
          Value result = input;
          if (lowBit > 0) {
            auto shamt = arith::ConstantOp::create(
                builder, opLoc,
                builder.getIntegerAttr(input.getType(), lowBit));
            result = arith::ShRUIOp::create(builder, opLoc, result, shamt);
          }
          if (resultWidth <
              cast<IntegerType>(input.getType()).getWidth()) {
            auto resTy = builder.getIntegerType(resultWidth);
            result = arith::TruncIOp::create(builder, opLoc, resTy, result);
          }
          mapping.map(extractOp.getResult(), result);
          continue;
        }

        // comb.concat → chain of arith.extui + arith.shli + arith.ori
        // Concat semantics: concat(a, b, c) = (a << (bw+cw)) | (b << cw) | c
        // i.e., first operand is MSB, last is LSB.
        if (auto concatOp = dyn_cast<comb::ConcatOp>(&op)) {
          auto inputs = concatOp.getInputs();
          unsigned resultWidth =
              cast<IntegerType>(concatOp.getType()).getWidth();
          auto resTy = builder.getIntegerType(resultWidth);
          if (inputs.empty()) {
            loweringFailed = true;
            break;
          }
          // Start from LSB (last operand) and shift+or each higher operand.
          Value result = mapping.lookupOrDefault(inputs.back());
          if (cast<IntegerType>(result.getType()).getWidth() < resultWidth)
            result = arith::ExtUIOp::create(builder, opLoc, resTy, result);
          unsigned accWidth =
              cast<IntegerType>(inputs.back().getType()).getWidth();
          for (int i = (int)inputs.size() - 2; i >= 0; --i) {
            Value operand = mapping.lookupOrDefault(inputs[i]);
            unsigned opWidth =
                cast<IntegerType>(inputs[i].getType()).getWidth();
            Value extended = operand;
            if (opWidth < resultWidth)
              extended =
                  arith::ExtUIOp::create(builder, opLoc, resTy, extended);
            auto shamt = arith::ConstantOp::create(
                builder, opLoc,
                builder.getIntegerAttr(resTy, accWidth));
            Value shifted =
                arith::ShLIOp::create(builder, opLoc, extended, shamt);
            result = arith::OrIOp::create(builder, opLoc, result, shifted);
            accWidth += opWidth;
          }
          mapping.map(concatOp.getResult(), result);
          continue;
        }

        // comb.replicate → repeated concat pattern
        if (auto repOp = dyn_cast<comb::ReplicateOp>(&op)) {
          auto input = mapping.lookupOrDefault(repOp.getInput());
          unsigned inputWidth =
              cast<IntegerType>(repOp.getInput().getType()).getWidth();
          unsigned resultWidth =
              cast<IntegerType>(repOp.getType()).getWidth();
          unsigned multiple = resultWidth / inputWidth;
          auto resTy = builder.getIntegerType(resultWidth);
          Value result = input;
          if (inputWidth < resultWidth)
            result = arith::ExtUIOp::create(builder, opLoc, resTy, result);
          Value acc = result;
          for (unsigned i = 1; i < multiple; ++i) {
            auto shamt = arith::ConstantOp::create(
                builder, opLoc,
                builder.getIntegerAttr(resTy, i * inputWidth));
            Value shifted =
                arith::ShLIOp::create(builder, opLoc, result, shamt);
            acc = arith::OrIOp::create(builder, opLoc, acc, shifted);
          }
          mapping.map(repOp.getResult(), acc);
          continue;
        }

        // comb.shl → arith.shli
        if (auto shlOp = dyn_cast<comb::ShlOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(shlOp.getLhs());
          auto rhs = mapping.lookupOrDefault(shlOp.getRhs());
          auto r = arith::ShLIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(shlOp.getResult(), r);
          continue;
        }

        // comb.shru → arith.shrui
        if (auto shruOp = dyn_cast<comb::ShrUOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(shruOp.getLhs());
          auto rhs = mapping.lookupOrDefault(shruOp.getRhs());
          auto r = arith::ShRUIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(shruOp.getResult(), r);
          continue;
        }

        // comb.shrs → arith.shrsi
        if (auto shrsOp = dyn_cast<comb::ShrSOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(shrsOp.getLhs());
          auto rhs = mapping.lookupOrDefault(shrsOp.getRhs());
          auto r = arith::ShRSIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(shrsOp.getResult(), r);
          continue;
        }

        // comb.divu → arith.divui
        if (auto divuOp = dyn_cast<comb::DivUOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(divuOp.getLhs());
          auto rhs = mapping.lookupOrDefault(divuOp.getRhs());
          auto r = arith::DivUIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(divuOp.getResult(), r);
          continue;
        }

        // comb.divs → arith.divsi
        if (auto divsOp = dyn_cast<comb::DivSOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(divsOp.getLhs());
          auto rhs = mapping.lookupOrDefault(divsOp.getRhs());
          auto r = arith::DivSIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(divsOp.getResult(), r);
          continue;
        }

        // comb.modu → arith.remui
        if (auto moduOp = dyn_cast<comb::ModUOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(moduOp.getLhs());
          auto rhs = mapping.lookupOrDefault(moduOp.getRhs());
          auto r = arith::RemUIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(moduOp.getResult(), r);
          continue;
        }

        // comb.mods → arith.remsi
        if (auto modsOp = dyn_cast<comb::ModSOp>(&op)) {
          auto lhs = mapping.lookupOrDefault(modsOp.getLhs());
          auto rhs = mapping.lookupOrDefault(modsOp.getRhs());
          auto r = arith::RemSIOp::create(builder, opLoc, lhs, rhs);
          mapping.map(modsOp.getResult(), r);
          continue;
        }

        // comb.parity → XOR reduction (fold-in-half)
        if (auto parityOp = dyn_cast<comb::ParityOp>(&op)) {
          auto input = mapping.lookupOrDefault(parityOp.getInput());
          unsigned width =
              cast<IntegerType>(parityOp.getInput().getType()).getWidth();
          auto i1Ty = builder.getI1Type();
          // XOR-reduce by folding in half repeatedly.
          Value val = input;
          for (unsigned half = width / 2; half > 0; half /= 2) {
            auto shamt = arith::ConstantOp::create(
                builder, opLoc,
                builder.getIntegerAttr(val.getType(), half));
            Value shifted =
                arith::ShRUIOp::create(builder, opLoc, val, shamt);
            val = arith::XOrIOp::create(builder, opLoc, val, shifted);
          }
          // Truncate to i1.
          if (width > 1)
            val = arith::TruncIOp::create(builder, opLoc, i1Ty, val);
          mapping.map(parityOp.getResult(), val);
          continue;
        }

        // === Phase K: additional hw ops ===

        // hw.bitcast → identity for same-width integers
        if (auto bitcastOp = dyn_cast<hw::BitcastOp>(&op)) {
          auto input = mapping.lookupOrDefault(bitcastOp.getInput());
          auto resultTy = bitcastOp.getResult().getType();
          if (auto intResultTy = dyn_cast<IntegerType>(resultTy)) {
            auto inputIntTy = dyn_cast<IntegerType>(input.getType());
            if (inputIntTy &&
                inputIntTy.getWidth() == intResultTy.getWidth()) {
              mapping.map(bitcastOp.getResult(), input);
            } else if (inputIntTy &&
                       inputIntTy.getWidth() > intResultTy.getWidth()) {
              auto r = arith::TruncIOp::create(builder, opLoc, intResultTy,
                                               input);
              mapping.map(bitcastOp.getResult(), r);
            } else if (inputIntTy) {
              auto r = arith::ExtUIOp::create(builder, opLoc, intResultTy,
                                              input);
              mapping.map(bitcastOp.getResult(), r);
            } else {
              loweringFailed = true;
              break;
            }
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // hw.array_get → shift + trunc (integer-element arrays)
        // Array packed as bitvector: element[0] at LSB.
        if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(&op)) {
          auto input = mapping.lookupOrDefault(arrayGetOp.getInput());
          auto index = mapping.lookupOrDefault(arrayGetOp.getIndex());
          auto elemTy = arrayGetOp.getResult().getType();
          if (auto intElemTy = dyn_cast<IntegerType>(elemTy)) {
            unsigned elemWidth = intElemTy.getWidth();
            auto inputIntTy = dyn_cast<IntegerType>(input.getType());
            if (!inputIntTy) {
              loweringFailed = true;
              break;
            }
            // shamt = index * elemWidth
            auto elemWidthConst = arith::ConstantOp::create(
                builder, opLoc,
                builder.getIntegerAttr(inputIntTy, elemWidth));
            Value idx = index;
            if (cast<IntegerType>(idx.getType()).getWidth() <
                inputIntTy.getWidth())
              idx = arith::ExtUIOp::create(builder, opLoc, inputIntTy, idx);
            Value shamt =
                arith::MulIOp::create(builder, opLoc, idx, elemWidthConst);
            Value shifted =
                arith::ShRUIOp::create(builder, opLoc, input, shamt);
            Value result = shifted;
            if (inputIntTy.getWidth() > elemWidth)
              result = arith::TruncIOp::create(builder, opLoc, intElemTy,
                                               shifted);
            mapping.map(arrayGetOp.getResult(), result);
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // hw.array_create → pack elements into bitvector via shift+or
        // Operands: [N-1] ... [0] (MSB first), element[0] at LSB.
        if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(&op)) {
          auto inputs = arrayCreateOp.getInputs();
          auto arrayTy = cast<hw::ArrayType>(arrayCreateOp.getType());
          auto elemTy = arrayTy.getElementType();
          if (auto intElemTy = dyn_cast<IntegerType>(elemTy)) {
            unsigned elemWidth = intElemTy.getWidth();
            unsigned totalWidth = elemWidth * arrayTy.getNumElements();
            auto packedTy = builder.getIntegerType(totalWidth);
            if (inputs.empty()) {
              loweringFailed = true;
              break;
            }
            // Last operand = element[0] (LSB).
            Value result = mapping.lookupOrDefault(inputs.back());
            if (elemWidth < totalWidth)
              result =
                  arith::ExtUIOp::create(builder, opLoc, packedTy, result);
            for (int i = (int)inputs.size() - 2; i >= 0; --i) {
              Value elem = mapping.lookupOrDefault(inputs[i]);
              if (elemWidth < totalWidth)
                elem =
                    arith::ExtUIOp::create(builder, opLoc, packedTy, elem);
              unsigned shift =
                  ((unsigned)inputs.size() - 1 - (unsigned)i) * elemWidth;
              auto shamt = arith::ConstantOp::create(
                  builder, opLoc,
                  builder.getIntegerAttr(packedTy, shift));
              Value shifted =
                  arith::ShLIOp::create(builder, opLoc, elem, shamt);
              result = arith::OrIOp::create(builder, opLoc, result, shifted);
            }
            mapping.map(arrayCreateOp.getResult(), result);
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // hw.constant → arith.constant
        if (auto hwConst = dyn_cast<hw::ConstantOp>(&op)) {
          auto c = arith::ConstantOp::create(
              builder, opLoc,
              builder.getIntegerAttr(hwConst.getType(), hwConst.getValue()));
          mapping.map(hwConst.getResult(), c.getResult());
          continue;
        }

        // hw.struct_extract → extract component from 4-state pair
        if (auto extractOp = dyn_cast<hw::StructExtractOp>(&op)) {
          auto it = fourStateComponents.find(extractOp.getInput());
          if (it != fourStateComponents.end()) {
            StringRef fieldName = extractOp.getFieldName();
            if (fieldName == "value")
              mapping.map(extractOp.getResult(), it->second.first);
            else if (fieldName == "unknown")
              mapping.map(extractOp.getResult(), it->second.second);
            else {
              loweringFailed = true;
              break;
            }
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // hw.struct_create → assemble 4-state pair from components
        if (auto createOp = dyn_cast<hw::StructCreateOp>(&op)) {
          if (getFourStateValueWidth(createOp.getType())) {
            Value val = mapping.lookupOrDefault(createOp.getOperand(0));
            Value xz = mapping.lookupOrDefault(createOp.getOperand(1));
            fourStateComponents[createOp.getResult()] = {val, xz};
            mapping.map(createOp.getResult(), val); // dummy
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // hw.aggregate_constant → decompose into {value, unknown} pair
        if (auto aggConst = dyn_cast<hw::AggregateConstantOp>(&op)) {
          if (auto w = getFourStateValueWidth(aggConst.getType())) {
            auto fields = aggConst.getFields();
            auto valAttr = cast<IntegerAttr>(fields[0]);
            auto unkAttr = cast<IntegerAttr>(fields[1]);
            auto valConst =
                arith::ConstantOp::create(builder, opLoc, valAttr);
            auto unkConst =
                arith::ConstantOp::create(builder, opLoc, unkAttr);
            fourStateComponents[aggConst.getResult()] = {valConst, unkConst};
            mapping.map(aggConst.getResult(), valConst.getResult());
          } else {
            loweringFailed = true;
            break;
          }
          continue;
        }

        // === CF terminators: replace back-edges to waitBlock with return ===
        if (auto brOp = dyn_cast<cf::BranchOp>(&op)) {
          Block *dest = brOp.getDest();
          if (dest == waitBlock) {
            func::ReturnOp::create(builder, opLoc);
          } else {
            auto it = blockMap.find(dest);
            if (it == blockMap.end()) {
              loweringFailed = true;
              break;
            }
            llvm::SmallVector<Value> args;
            for (auto arg : brOp.getDestOperands())
              args.push_back(mapping.lookupOrDefault(arg));
            cf::BranchOp::create(builder, opLoc, it->second, args);
          }
          continue;
        }

        if (auto condBrOp = dyn_cast<cf::CondBranchOp>(&op)) {
          Value cond = mapping.lookupOrDefault(condBrOp.getCondition());
          Block *trueDest = condBrOp.getTrueDest();
          Block *falseDest = condBrOp.getFalseDest();
          bool trueIsReturn = (trueDest == waitBlock);
          bool falseIsReturn = (falseDest == waitBlock);

          if (trueIsReturn && falseIsReturn) {
            func::ReturnOp::create(builder, opLoc);
          } else if (trueIsReturn) {
            Block *retBlock = new Block();
            funcOp.getBody().push_back(retBlock);
            OpBuilder(retBlock, retBlock->begin())
                .create<func::ReturnOp>(opLoc);
            Block *ft = blockMap.lookup(falseDest);
            if (!ft) {
              loweringFailed = true;
              break;
            }
            llvm::SmallVector<Value> falseArgs;
            for (auto arg : condBrOp.getFalseDestOperands())
              falseArgs.push_back(mapping.lookupOrDefault(arg));
            cf::CondBranchOp::create(builder, opLoc, cond, retBlock,
                                 ValueRange{}, ft, falseArgs);
          } else if (falseIsReturn) {
            Block *retBlock = new Block();
            funcOp.getBody().push_back(retBlock);
            OpBuilder(retBlock, retBlock->begin())
                .create<func::ReturnOp>(opLoc);
            Block *tt = blockMap.lookup(trueDest);
            if (!tt) {
              loweringFailed = true;
              break;
            }
            llvm::SmallVector<Value> trueArgs;
            for (auto arg : condBrOp.getTrueDestOperands())
              trueArgs.push_back(mapping.lookupOrDefault(arg));
            cf::CondBranchOp::create(builder, opLoc, cond, tt, trueArgs,
                                 retBlock, ValueRange{});
          } else {
            Block *tt = blockMap.lookup(trueDest);
            Block *ft = blockMap.lookup(falseDest);
            if (!tt || !ft) {
              loweringFailed = true;
              break;
            }
            llvm::SmallVector<Value> trueArgs, falseArgs;
            for (auto arg : condBrOp.getTrueDestOperands())
              trueArgs.push_back(mapping.lookupOrDefault(arg));
            for (auto arg : condBrOp.getFalseDestOperands())
              falseArgs.push_back(mapping.lookupOrDefault(arg));
            cf::CondBranchOp::create(builder, opLoc, cond, tt, trueArgs, ft,
                                 falseArgs);
          }
          continue;
        }

        // Default: clone with mapping.
        builder.clone(op, mapping);
      }
      if (loweringFailed)
        break;
    }
    if (loweringFailed) {
      if (procRejectionReasons) {
        std::string reason = "process_lowering_failed";
        if (!loweringFailOpName.empty())
          reason += ":" + loweringFailOpName;
        ++(*procRejectionReasons)[reason];
      }
      funcOp.erase();
      continue;
    }

    procNames.push_back(funcName);
    procKinds.push_back(CIRCT_PROC_CALLBACK);
    ++compiled;
  }

  return compiled;
}

//===----------------------------------------------------------------------===//
// Descriptor synthesis: emit CirctSimCompiledModule as LLVM IR globals
//===----------------------------------------------------------------------===//

/// Synthesize the descriptor tables and entrypoint functions into the LLVM IR
/// module. This creates:
///   - String constants for function names
///   - Arrays: func_names, func_entries
///   - The CirctSimCompiledModule struct
///   - circt_sim_get_compiled_module() returning a pointer to it
///   - circt_sim_get_build_id() returning a build ID string
static void synthesizeDescriptor(llvm::Module &llvmModule,
                                 const llvm::SmallVector<std::string> &funcNames,
                                 const llvm::SmallVector<std::string> &trampolineNames,
                                 const llvm::SmallVector<std::string> &procNames,
                                 const llvm::SmallVector<uint8_t> &procKinds,
                                 const std::string &buildId,
                                 const llvm::SmallVector<std::string> &globalPatchNames,
                                 const llvm::SmallVector<llvm::GlobalVariable *> &globalPatchVars,
                                 const llvm::SmallVector<uint32_t> &globalPatchSizes,
                                 const llvm::SmallVector<std::string> &allFuncEntryNames) {
  auto &ctx = llvmModule.getContext();
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  auto *ptrTy = llvm::PointerType::get(ctx, 0);
  unsigned numFuncs = funcNames.size();

  // Create string constants for function names.
  llvm::SmallVector<llvm::Constant *> nameGlobals;
  for (unsigned i = 0; i < numFuncs; ++i) {
    auto *strConst =
        llvm::ConstantDataArray::getString(ctx, funcNames[i], true);
    auto *strGlobal = new llvm::GlobalVariable(
        llvmModule, strConst->getType(), true,
        llvm::GlobalValue::PrivateLinkage, strConst,
        "__circt_sim_fname_" + std::to_string(i));
    nameGlobals.push_back(strGlobal);
  }

  // func_names array: const char* const[]
  auto *nameArrayTy = llvm::ArrayType::get(ptrTy, numFuncs);
  auto *nameArray = llvm::ConstantArray::get(
      nameArrayTy, llvm::ArrayRef<llvm::Constant *>(nameGlobals));
  auto *funcNamesGlobal = new llvm::GlobalVariable(
      llvmModule, nameArrayTy, true, llvm::GlobalValue::PrivateLinkage,
      nameArray, "__circt_sim_func_names");

  // func_entry array: const void* const[] — initially null, filled by
  // the linker via the actual symbol addresses.
  llvm::SmallVector<llvm::Constant *> entryPtrs;
  for (unsigned i = 0; i < numFuncs; ++i) {
    auto *func = llvmModule.getFunction(funcNames[i]);
    if (func) {
      entryPtrs.push_back(func);
    } else {
      entryPtrs.push_back(llvm::ConstantPointerNull::get(ptrTy));
    }
  }
  auto *entryArrayTy = llvm::ArrayType::get(ptrTy, numFuncs);
  auto *entryArray = llvm::ConstantArray::get(
      entryArrayTy, llvm::ArrayRef<llvm::Constant *>(entryPtrs));
  auto *funcEntryGlobal = new llvm::GlobalVariable(
      llvmModule, entryArrayTy, true, llvm::GlobalValue::PrivateLinkage,
      entryArray, "__circt_sim_func_entry");

  // Create process name string constants, kind array, and entry array.
  unsigned numProcs = procNames.size();
  llvm::Constant *procNamesGlobal = nullptr;
  llvm::Constant *procKindGlobal = nullptr;
  llvm::Constant *procEntryGlobal = nullptr;

  if (numProcs > 0) {
    // proc_names: string constants.
    llvm::SmallVector<llvm::Constant *> procNameGlobals;
    for (unsigned i = 0; i < numProcs; ++i) {
      auto *strConst =
          llvm::ConstantDataArray::getString(ctx, procNames[i], true);
      auto *strGlobal = new llvm::GlobalVariable(
          llvmModule, strConst->getType(), true,
          llvm::GlobalValue::PrivateLinkage, strConst,
          "__circt_sim_pname_" + std::to_string(i));
      procNameGlobals.push_back(strGlobal);
    }
    auto *procNameArrayTy = llvm::ArrayType::get(ptrTy, numProcs);
    auto *procNameArray = llvm::ConstantArray::get(
        procNameArrayTy, llvm::ArrayRef<llvm::Constant *>(procNameGlobals));
    procNamesGlobal = new llvm::GlobalVariable(
        llvmModule, procNameArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        procNameArray, "__circt_sim_proc_names");

    // proc_kind: uint8_t[].
    auto *i8LLTy = llvm::Type::getInt8Ty(ctx);
    llvm::SmallVector<llvm::Constant *> kindConstants;
    for (auto kind : procKinds)
      kindConstants.push_back(llvm::ConstantInt::get(i8LLTy, kind));
    auto *kindArrayTy = llvm::ArrayType::get(i8LLTy, numProcs);
    auto *kindArray = llvm::ConstantArray::get(
        kindArrayTy, llvm::ArrayRef<llvm::Constant *>(kindConstants));
    procKindGlobal = new llvm::GlobalVariable(
        llvmModule, kindArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        kindArray, "__circt_sim_proc_kind");

    // proc_entry: function pointers.
    llvm::SmallVector<llvm::Constant *> procEntryPtrs;
    for (unsigned i = 0; i < numProcs; ++i) {
      auto *func = llvmModule.getFunction(procNames[i]);
      if (func)
        procEntryPtrs.push_back(func);
      else
        procEntryPtrs.push_back(llvm::ConstantPointerNull::get(ptrTy));
    }
    auto *procEntryArrayTy = llvm::ArrayType::get(ptrTy, numProcs);
    auto *procEntryArray = llvm::ConstantArray::get(
        procEntryArrayTy, llvm::ArrayRef<llvm::Constant *>(procEntryPtrs));
    procEntryGlobal = new llvm::GlobalVariable(
        llvmModule, procEntryArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        procEntryArray, "__circt_sim_proc_entry");
  }

  // Create trampoline name string constants and array.
  unsigned numTrampolines = trampolineNames.size();
  llvm::SmallVector<llvm::Constant *> trampNameGlobals;
  for (unsigned i = 0; i < numTrampolines; ++i) {
    auto *strConst =
        llvm::ConstantDataArray::getString(ctx, trampolineNames[i], true);
    auto *strGlobal = new llvm::GlobalVariable(
        llvmModule, strConst->getType(), true,
        llvm::GlobalValue::PrivateLinkage, strConst,
        "__circt_sim_tname_" + std::to_string(i));
    trampNameGlobals.push_back(strGlobal);
  }

  llvm::Constant *trampNamesGlobal;
  if (numTrampolines > 0) {
    auto *trampNameArrayTy = llvm::ArrayType::get(ptrTy, numTrampolines);
    auto *trampNameArray = llvm::ConstantArray::get(
        trampNameArrayTy, llvm::ArrayRef<llvm::Constant *>(trampNameGlobals));
    trampNamesGlobal = new llvm::GlobalVariable(
        llvmModule, trampNameArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        trampNameArray, "__circt_sim_trampoline_names");
  } else {
    trampNamesGlobal = llvm::ConstantPointerNull::get(ptrTy);
  }

  // Build patch table: string constants, address array, and size array.
  unsigned numGlobalPatches = globalPatchNames.size();
  llvm::Constant *globalPatchNamesGlobal = nullptr;
  llvm::Constant *globalPatchAddrsGlobal = nullptr;
  llvm::Constant *globalPatchSizesGlobal = nullptr;

  if (numGlobalPatches > 0) {
    // global_patch_names: string constants.
    llvm::SmallVector<llvm::Constant *> patchNameGlobals;
    for (unsigned i = 0; i < numGlobalPatches; ++i) {
      auto *strConst =
          llvm::ConstantDataArray::getString(ctx, globalPatchNames[i], true);
      auto *strGlobal = new llvm::GlobalVariable(
          llvmModule, strConst->getType(), true,
          llvm::GlobalValue::PrivateLinkage, strConst,
          "__circt_sim_gpname_" + std::to_string(i));
      patchNameGlobals.push_back(strGlobal);
    }
    auto *patchNameArrayTy = llvm::ArrayType::get(ptrTy, numGlobalPatches);
    auto *patchNameArray = llvm::ConstantArray::get(
        patchNameArrayTy,
        llvm::ArrayRef<llvm::Constant *>(patchNameGlobals));
    globalPatchNamesGlobal = new llvm::GlobalVariable(
        llvmModule, patchNameArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        patchNameArray, "__circt_sim_global_patch_names");

    // global_patch_addrs: pointers to the actual globals in the .so.
    llvm::SmallVector<llvm::Constant *> patchAddrConstants;
    for (unsigned i = 0; i < numGlobalPatches; ++i)
      patchAddrConstants.push_back(globalPatchVars[i]);
    auto *patchAddrArrayTy = llvm::ArrayType::get(ptrTy, numGlobalPatches);
    auto *patchAddrArray = llvm::ConstantArray::get(
        patchAddrArrayTy,
        llvm::ArrayRef<llvm::Constant *>(patchAddrConstants));
    globalPatchAddrsGlobal = new llvm::GlobalVariable(
        llvmModule, patchAddrArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        patchAddrArray, "__circt_sim_global_patch_addrs");

    // global_patch_sizes: uint32_t[].
    auto *i32ArrayTy = llvm::ArrayType::get(i32Ty, numGlobalPatches);
    llvm::SmallVector<llvm::Constant *> sizeConstants;
    for (unsigned i = 0; i < numGlobalPatches; ++i)
      sizeConstants.push_back(
          llvm::ConstantInt::get(i32Ty, globalPatchSizes[i]));
    auto *sizeArray = llvm::ConstantArray::get(
        i32ArrayTy, llvm::ArrayRef<llvm::Constant *>(sizeConstants));
    globalPatchSizesGlobal = new llvm::GlobalVariable(
        llvmModule, i32ArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        sizeArray, "__circt_sim_global_patch_sizes");
  }

  // Build unified func_entries table: allFuncEntryNames[fid] → pointer.
  // For each FuncId, the entry is either a compiled function or a trampoline.
  unsigned numAllFuncs = allFuncEntryNames.size();
  llvm::Constant *allFuncEntriesGlobal = nullptr;
  llvm::Constant *allFuncEntryNamesGlobal = nullptr;

  if (numAllFuncs > 0) {
    // Build entry pointer array.
    llvm::SmallVector<llvm::Constant *> allEntryPtrs;
    for (unsigned i = 0; i < numAllFuncs; ++i) {
      auto *func = llvmModule.getFunction(allFuncEntryNames[i]);
      if (func) {
        allEntryPtrs.push_back(func);
      } else {
        allEntryPtrs.push_back(llvm::ConstantPointerNull::get(ptrTy));
      }
    }
    auto *allEntryArrayTy = llvm::ArrayType::get(ptrTy, numAllFuncs);
    auto *allEntryArray = llvm::ConstantArray::get(
        allEntryArrayTy, llvm::ArrayRef<llvm::Constant *>(allEntryPtrs));
    allFuncEntriesGlobal = new llvm::GlobalVariable(
        llvmModule, allEntryArrayTy, /*isConstant=*/true,
        llvm::GlobalValue::ExternalLinkage, allEntryArray,
        "__circt_sim_func_entries");

    // Build name string array.
    llvm::SmallVector<llvm::Constant *> allNameGlobals;
    for (unsigned i = 0; i < numAllFuncs; ++i) {
      auto *strConst =
          llvm::ConstantDataArray::getString(ctx, allFuncEntryNames[i], true);
      auto *strGlobal = new llvm::GlobalVariable(
          llvmModule, strConst->getType(), true,
          llvm::GlobalValue::PrivateLinkage, strConst,
          "__circt_sim_all_fname_" + std::to_string(i));
      allNameGlobals.push_back(strGlobal);
    }
    auto *allNameArrayTy = llvm::ArrayType::get(ptrTy, numAllFuncs);
    auto *allNameArray = llvm::ConstantArray::get(
        allNameArrayTy, llvm::ArrayRef<llvm::Constant *>(allNameGlobals));
    allFuncEntryNamesGlobal = new llvm::GlobalVariable(
        llvmModule, allNameArrayTy, true, llvm::GlobalValue::PrivateLinkage,
        allNameArray, "__circt_sim_all_func_entry_names");
  }

  // Build the CirctSimCompiledModule struct.
  // Layout: {i32, i32, ptr, ptr, ptr, i32, ptr, ptr, i32, ptr, i32, ptr, ptr, ptr, i32, ptr, ptr}
  auto *descriptorTy = llvm::StructType::create(
      ctx,
      {i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy, i32Ty, ptrTy,
       i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy},
      "CirctSimCompiledModule");

  auto *nullPtr = llvm::ConstantPointerNull::get(ptrTy);
  auto *descriptor = llvm::ConstantStruct::get(
      descriptorTy,
      {
          llvm::ConstantInt::get(i32Ty, CIRCT_SIM_ABI_VERSION), // abi_version
          llvm::ConstantInt::get(i32Ty, numProcs),              // num_procs
          numProcs > 0 ? procNamesGlobal : nullPtr,             // proc_names
          numProcs > 0 ? procKindGlobal : nullPtr,              // proc_kind
          numProcs > 0 ? procEntryGlobal : nullPtr,             // proc_entry
          llvm::ConstantInt::get(i32Ty, numFuncs),              // num_funcs
          funcNamesGlobal,                                       // func_names
          funcEntryGlobal,                                       // func_entry
          llvm::ConstantInt::get(i32Ty, numTrampolines),    // num_trampolines
          trampNamesGlobal,                                 // trampoline_names
          llvm::ConstantInt::get(i32Ty, numGlobalPatches),  // num_global_patches
          numGlobalPatches > 0 ? globalPatchNamesGlobal : nullPtr, // global_patch_names
          numGlobalPatches > 0 ? globalPatchAddrsGlobal : nullPtr, // global_patch_addrs
          numGlobalPatches > 0 ? globalPatchSizesGlobal : nullPtr, // global_patch_sizes
          llvm::ConstantInt::get(i32Ty, numAllFuncs),       // num_all_funcs
          numAllFuncs > 0 ? allFuncEntriesGlobal : nullPtr, // all_func_entries
          numAllFuncs > 0 ? allFuncEntryNamesGlobal : nullPtr, // all_func_entry_names
      });

  auto *descriptorGlobal = new llvm::GlobalVariable(
      llvmModule, descriptorTy, true, llvm::GlobalValue::PrivateLinkage,
      descriptor, "__circt_sim_descriptor");

  // circt_sim_get_compiled_module() — returns pointer to descriptor.
  auto *getModuleFnTy = llvm::FunctionType::get(ptrTy, false);
  auto *getModuleFn = llvm::Function::Create(
      getModuleFnTy, llvm::GlobalValue::ExternalLinkage,
      "circt_sim_get_compiled_module", &llvmModule);
  getModuleFn->setVisibility(llvm::GlobalValue::DefaultVisibility);

  auto *bb = llvm::BasicBlock::Create(ctx, "entry", getModuleFn);
  llvm::IRBuilder<> irBuilder(bb);
  irBuilder.CreateRet(descriptorGlobal);

  // Build ID string.
  auto *buildIdStr =
      llvm::ConstantDataArray::getString(ctx, buildId, true);
  auto *buildIdGlobal = new llvm::GlobalVariable(
      llvmModule, buildIdStr->getType(), true,
      llvm::GlobalValue::PrivateLinkage, buildIdStr, "__circt_sim_build_id");

  auto *getBuildIdFnTy = llvm::FunctionType::get(ptrTy, false);
  auto *getBuildIdFn = llvm::Function::Create(
      getBuildIdFnTy, llvm::GlobalValue::ExternalLinkage,
      "circt_sim_get_build_id", &llvmModule);
  getBuildIdFn->setVisibility(llvm::GlobalValue::DefaultVisibility);

  auto *bb2 = llvm::BasicBlock::Create(ctx, "entry", getBuildIdFn);
  llvm::IRBuilder<> irBuilder2(bb2);
  irBuilder2.CreateRet(buildIdGlobal);
}

/// Set all functions in the module to hidden visibility except the exported
/// entrypoints. This keeps the .so symbol table minimal.
static void setDefaultHiddenVisibility(llvm::Module &llvmModule) {
  for (auto &func : llvmModule.functions()) {
    if (func.getName() == "circt_sim_get_compiled_module" ||
        func.getName() == "circt_sim_get_build_id" ||
        func.getName().starts_with("__circt_sim_module_init__"))
      continue;
    // Don't hide external declarations (runtime API functions).
    if (func.isDeclaration())
      continue;
    // Local linkage (private/internal) requires DefaultVisibility.
    if (func.hasLocalLinkage())
      continue;
    func.setVisibility(llvm::GlobalValue::HiddenVisibility);
  }
  for (auto &global : llvmModule.globals()) {
    // Local linkage (private/internal) requires DefaultVisibility.
    if (global.hasLocalLinkage())
      continue;
    // Keep __circt_sim_ctx visible so dlsym() can find it at runtime.
    if (global.getName() == "__circt_sim_ctx")
      continue;
    global.setVisibility(llvm::GlobalValue::HiddenVisibility);
  }
}

//===----------------------------------------------------------------------===//
// Internalization and inlining preparation
//===----------------------------------------------------------------------===//

/// Mark all defined functions as internal linkage except the ABI entrypoints
/// and runtime API declarations. This enables whole-program inlining and DCE.
static void internalizeNonExported(llvm::Module &llvmModule) {
  for (auto &F : llvmModule) {
    if (F.isDeclaration())
      continue;
    llvm::StringRef name = F.getName();
    if (name == "circt_sim_get_compiled_module" ||
        name == "circt_sim_get_build_id" ||
        name.starts_with("__circt_sim_") ||
        name.starts_with("__moore_") ||
        name.starts_with("__arc_sched_"))
      continue;
    F.setLinkage(llvm::GlobalValue::InternalLinkage);
  }
}

/// Mark small defined functions (< 20 instructions) as alwaysinline to
/// encourage the optimizer to inline them aggressively even at O1.
static unsigned addAlwaysInlineToSmallFunctions(llvm::Module &llvmModule) {
  unsigned count = 0;
  for (auto &F : llvmModule) {
    if (F.isDeclaration() || F.hasFnAttribute(llvm::Attribute::NoInline))
      continue;
    unsigned instCount = 0;
    for (auto &BB : F)
      instCount += BB.size();
    if (instCount < 20) {
      F.addFnAttr(llvm::Attribute::AlwaysInline);
      ++count;
    }
  }
  return count;
}

//===----------------------------------------------------------------------===//
// Packed struct layout: rewrite GEPs for interpreter-compatible layout
//===----------------------------------------------------------------------===//

/// Compute the allocation size of a type using packed (no-padding) layout.
/// For struct types, this sums the packed sizes of all fields without any
/// alignment padding. For non-struct types, this uses the DataLayout.
static uint64_t computePackedTypeAllocSize(llvm::Type *ty,
                                           const llvm::DataLayout &DL) {
  if (auto *st = llvm::dyn_cast<llvm::StructType>(ty)) {
    uint64_t total = 0;
    for (unsigned i = 0; i < st->getNumElements(); ++i)
      total += computePackedTypeAllocSize(st->getElementType(i), DL);
    return total;
  }
  if (auto *at = llvm::dyn_cast<llvm::ArrayType>(ty))
    return at->getNumElements() *
           computePackedTypeAllocSize(at->getElementType(), DL);
  return DL.getTypeAllocSize(ty);
}

/// Compute the byte offset of field `fieldIdx` in a struct type using packed
/// (no-padding) layout: sum the packed sizes of all preceding fields.
static uint64_t computePackedFieldOffset(llvm::StructType *st,
                                         unsigned fieldIdx,
                                         const llvm::DataLayout &DL) {
  uint64_t offset = 0;
  for (unsigned i = 0; i < fieldIdx; ++i)
    offset += computePackedTypeAllocSize(st->getElementType(i), DL);
  return offset;
}

/// Rewrite GEP instructions that index into struct types so they use the
/// packed (no-padding) field offsets instead of the host DataLayout offsets.
///
/// The interpreter uses UNALIGNED struct layout (no padding between fields).
/// When the AOT compiler produces LLVM IR, struct GEPs use the host
/// DataLayout which inserts alignment padding. This function rewrites every
/// struct-indexed GEP to an i8-based byte-offset GEP with the correct
/// packed offset, eliminating the mismatch.
///
/// Returns the number of GEPs rewritten.
static unsigned rewriteGEPsForPackedLayout(llvm::Module &M) {
  const llvm::DataLayout &DL = M.getDataLayout();
  auto &ctx = M.getContext();
  auto *i8Ty = llvm::Type::getInt8Ty(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);

  unsigned rewritten = 0;

  for (auto &F : M) {
    // Collect GEPs first to avoid iterator invalidation.
    llvm::SmallVector<llvm::GetElementPtrInst *> geps;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          geps.push_back(GEP);

    for (auto *GEP : geps) {
      // Check if this GEP indexes into any struct fields. Struct field
      // indices are always constant in LLVM IR, but other indices (first
      // index, array indices) may be dynamic. We handle both cases:
      // - All-constant indices: compute a constant packed offset
      // - Mixed (dynamic first/array index + constant struct indices):
      //   build an i8 GEP with dynamic scaling + constant struct offsets

      // First pass: check if there are any struct indices and if we can
      // handle the GEP (all indices at struct levels must be constant).
      bool hasStructIndex = false;
      bool hasDynamicIndex = false;
      {
        llvm::Type *ty = GEP->getSourceElementType();
        auto it = GEP->idx_begin();
        // First index can be dynamic.
        if (it != GEP->idx_end()) {
          if (!llvm::isa<llvm::ConstantInt>(it->get()))
            hasDynamicIndex = true;
          ++it;
        }
        for (; it != GEP->idx_end(); ++it) {
          if (auto *st = llvm::dyn_cast<llvm::StructType>(ty)) {
            hasStructIndex = true;
            // Struct indices must be constant (LLVM rule).
            auto *ci = llvm::cast<llvm::ConstantInt>(it->get());
            ty = st->getElementType(ci->getZExtValue());
          } else if (auto *at = llvm::dyn_cast<llvm::ArrayType>(ty)) {
            if (!llvm::isa<llvm::ConstantInt>(it->get()))
              hasDynamicIndex = true;
            ty = at->getElementType();
          }
        }
      }

      if (!hasStructIndex)
        continue;

      // Build the replacement GEP. We compute the packed byte offset for
      // each index level. For dynamic indices, we emit a multiply + add.
      llvm::IRBuilder<> builder(GEP);
      auto *basePtr = GEP->getPointerOperand();

      if (!hasDynamicIndex) {
        // Fast path: all indices are constant — compute a single constant
        // packed byte offset.
        uint64_t packedOffset = 0;
        llvm::Type *curTy = GEP->getSourceElementType();
        auto idxIt = GEP->idx_begin();

        if (idxIt != GEP->idx_end()) {
          auto *firstIdx = llvm::cast<llvm::ConstantInt>(idxIt->get());
          int64_t idx = firstIdx->getSExtValue();
          packedOffset += idx * computePackedTypeAllocSize(curTy, DL);
          ++idxIt;
        }
        for (; idxIt != GEP->idx_end(); ++idxIt) {
          auto *idx = llvm::cast<llvm::ConstantInt>(idxIt->get());
          if (auto *st = llvm::dyn_cast<llvm::StructType>(curTy)) {
            unsigned fieldIdx = static_cast<unsigned>(idx->getZExtValue());
            packedOffset += computePackedFieldOffset(st, fieldIdx, DL);
            curTy = st->getElementType(fieldIdx);
          } else if (auto *at = llvm::dyn_cast<llvm::ArrayType>(curTy)) {
            int64_t arrIdx = idx->getSExtValue();
            packedOffset +=
                arrIdx * computePackedTypeAllocSize(at->getElementType(), DL);
            curTy = at->getElementType();
          }
        }

        auto *offsetVal = llvm::ConstantInt::get(i64Ty, packedOffset);
        llvm::Value *newGEP;
        if (GEP->isInBounds())
          newGEP = builder.CreateInBoundsGEP(i8Ty, basePtr, offsetVal,
                                             GEP->getName() + ".packed");
        else
          newGEP = builder.CreateGEP(i8Ty, basePtr, offsetVal,
                                     GEP->getName() + ".packed");
        GEP->replaceAllUsesWith(newGEP);
        GEP->eraseFromParent();
      } else {
        // Slow path: some indices are dynamic. Build a byte offset
        // expression: offset = sum of (index * packed_elem_size) for each
        // level.
        llvm::Value *totalOffset = llvm::ConstantInt::get(i64Ty, 0);
        llvm::Type *curTy = GEP->getSourceElementType();
        auto idxIt = GEP->idx_begin();

        if (idxIt != GEP->idx_end()) {
          llvm::Value *idx = idxIt->get();
          if (idx->getType() != i64Ty)
            idx = builder.CreateSExtOrTrunc(idx, i64Ty);
          uint64_t elemSize = computePackedTypeAllocSize(curTy, DL);
          auto *elemSizeVal = llvm::ConstantInt::get(i64Ty, elemSize);
          totalOffset = builder.CreateAdd(
              totalOffset, builder.CreateMul(idx, elemSizeVal));
          ++idxIt;
        }
        for (; idxIt != GEP->idx_end(); ++idxIt) {
          if (auto *st = llvm::dyn_cast<llvm::StructType>(curTy)) {
            auto *ci = llvm::cast<llvm::ConstantInt>(idxIt->get());
            unsigned fieldIdx = static_cast<unsigned>(ci->getZExtValue());
            uint64_t fieldOff = computePackedFieldOffset(st, fieldIdx, DL);
            totalOffset = builder.CreateAdd(
                totalOffset, llvm::ConstantInt::get(i64Ty, fieldOff));
            curTy = st->getElementType(fieldIdx);
          } else if (auto *at = llvm::dyn_cast<llvm::ArrayType>(curTy)) {
            llvm::Value *idx = idxIt->get();
            if (idx->getType() != i64Ty)
              idx = builder.CreateSExtOrTrunc(idx, i64Ty);
            uint64_t elemSize =
                computePackedTypeAllocSize(at->getElementType(), DL);
            auto *elemSizeVal = llvm::ConstantInt::get(i64Ty, elemSize);
            totalOffset = builder.CreateAdd(
                totalOffset, builder.CreateMul(idx, elemSizeVal));
            curTy = at->getElementType();
          }
        }

        llvm::Value *newGEP;
        if (GEP->isInBounds())
          newGEP = builder.CreateInBoundsGEP(i8Ty, basePtr, totalOffset,
                                             GEP->getName() + ".packed");
        else
          newGEP = builder.CreateGEP(i8Ty, basePtr, totalOffset,
                                     GEP->getName() + ".packed");
        GEP->replaceAllUsesWith(newGEP);
        GEP->eraseFromParent();
      }
      ++rewritten;
    }
  }

  return rewritten;
}

/// Replace each mutable global variable's type with a flat [N x i8] array
/// where N is the packed (unpadded) size computed by computePackedTypeAllocSize.
///
/// This ensures:
/// 1. LLVM allocates exactly N bytes for the global (no alignment padding)
/// 2. The optimizer cannot reconstruct typed struct accesses (the type is now
///    an i8 array)
/// 3. The GEP rewrites (which already use i8-based byte offsets) remain
///    consistent
///
/// Skips constant globals, declaration-only (external) globals, unnamed
/// globals, and internal descriptor globals (__circt_sim_*).
///
/// Returns the number of globals flattened.
static unsigned flattenGlobalTypesToByteArrays(llvm::Module &M) {
  const llvm::DataLayout &DL = M.getDataLayout();
  unsigned count = 0;

  // Collect globals to flatten first — we can't modify the global list while
  // iterating it.
  llvm::SmallVector<llvm::GlobalVariable *> toFlatten;
  for (auto &G : M.globals()) {
    if (G.isConstant() || G.isDeclaration())
      continue;
    if (!G.hasName() || G.getName().empty())
      continue;
    if (G.getName().starts_with("__circt_sim_"))
      continue;
    toFlatten.push_back(&G);
  }

  for (auto *oldGV : toFlatten) {
    llvm::Type *oldTy = oldGV->getValueType();
    uint64_t packedSize = computePackedTypeAllocSize(oldTy, DL);
    if (packedSize == 0)
      continue;

    // Create the flat [N x i8] type.
    auto *i8Ty = llvm::Type::getInt8Ty(M.getContext());
    auto *flatTy = llvm::ArrayType::get(i8Ty, packedSize);

    // If the global already has this type, skip it.
    if (oldTy == flatTy)
      continue;

    // Create a new global with the flat type and zero initializer.
    auto *newGV = new llvm::GlobalVariable(
        M, flatTy, /*isConstant=*/false, oldGV->getLinkage(),
        llvm::Constant::getNullValue(flatTy), oldGV->getName() + ".flat",
        /*InsertBefore=*/oldGV, oldGV->getThreadLocalMode(),
        oldGV->getAddressSpace());
    newGV->setAlignment(oldGV->getAlign());
    newGV->setSection(oldGV->getSection());
    newGV->setVisibility(oldGV->getVisibility());

    // With opaque pointers (LLVM 15+), all pointer types are just `ptr`,
    // so we can directly replace all uses — no bitcast needed.
    oldGV->replaceAllUsesWith(newGV);

    // Transfer the name from old to new.
    std::string name = oldGV->getName().str();
    // Remove ".flat" suffix that was appended to avoid name collision.
    oldGV->eraseFromParent();
    newGV->setName(name);

    ++count;
  }

  return count;
}

/// Statistics about a module used for pre/post-optimization comparison.
struct ModuleStats {
  unsigned definedFnCount = 0;
  unsigned internalFnCount = 0;
  unsigned callCount = 0;
};

static ModuleStats collectModuleStats(llvm::Module &llvmModule) {
  ModuleStats stats;
  for (auto &F : llvmModule) {
    if (!F.isDeclaration()) {
      ++stats.definedFnCount;
      if (F.hasLocalLinkage())
        ++stats.internalFnCount;
    }
    for (auto &BB : F)
      for (auto &I : BB)
        if (llvm::isa<llvm::CallInst>(I))
          ++stats.callCount;
  }
  return stats;
}

//===----------------------------------------------------------------------===//
// Object file emission via LLVM TargetMachine
//===----------------------------------------------------------------------===//

/// Run LLVM optimization passes on the module (new pass manager).
static void runOptimizationPasses(llvm::Module &llvmModule,
                                  llvm::TargetMachine *tm, int optLvl) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Use aggressive inlining threshold to encourage whole-program inlining of
  // internalized helper functions. Default is ~225; 500 inlines more freely.
  llvm::PipelineTuningOptions PTO;
  PTO.InlinerThreshold = 500;

  llvm::PassBuilder pb(tm, PTO);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel level;
  switch (optLvl) {
  case 0:
    level = llvm::OptimizationLevel::O0;
    break;
  case 1:
    level = llvm::OptimizationLevel::O1;
    break;
  case 2:
    level = llvm::OptimizationLevel::O2;
    break;
  default:
    level = llvm::OptimizationLevel::O3;
    break;
  }

  if (level == llvm::OptimizationLevel::O0) {
    auto mpm = pb.buildO0DefaultPipeline(level);
    mpm.run(llvmModule, mam);
  } else {
    auto mpm = pb.buildPerModuleDefaultPipeline(level);
    mpm.run(llvmModule, mam);
  }
}

/// Create a TargetMachine for the host, set the module's target triple and
/// data layout, and return the machine.  The caller takes ownership.
/// Returns nullptr on failure (error already printed to stderr).
static llvm::TargetMachine *
createHostTargetMachine(llvm::Module &llvmModule, int optLvl) {
  auto targetTripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple targetTriple(targetTripleStr);
  llvmModule.setTargetTriple(targetTriple);

  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Error looking up target: " << error << "\n";
    return nullptr;
  }

  auto cpu = llvm::sys::getHostCPUName();
  llvm::SubtargetFeatures features;
  auto hostFeatures = llvm::sys::getHostCPUFeatures();
  for (auto &feature : hostFeatures)
    if (feature.second)
      features.AddFeature(feature.first());

  llvm::CodeGenOptLevel cgOptLevel;
  switch (optLvl) {
  case 0:
    cgOptLevel = llvm::CodeGenOptLevel::None;
    break;
  case 1:
    cgOptLevel = llvm::CodeGenOptLevel::Less;
    break;
  case 2:
    cgOptLevel = llvm::CodeGenOptLevel::Default;
    break;
  default:
    cgOptLevel = llvm::CodeGenOptLevel::Aggressive;
    break;
  }

  llvm::TargetOptions targetOpts;
  auto rm = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
  auto *targetMachine = target->createTargetMachine(
      targetTriple, cpu, features.getString(), targetOpts, rm,
      std::nullopt, cgOptLevel);
  if (!targetMachine) {
    llvm::errs() << "Failed to create target machine\n";
    return nullptr;
  }

  llvmModule.setDataLayout(targetMachine->createDataLayout());
  return targetMachine;
}

/// Emit an already-optimized module to an object file.  The caller must have
/// already run optimization passes and any post-optimization transforms (e.g.
/// rewriteGEPsForPackedLayout).  Does NOT run optimization passes internally.
static LogicalResult emitObjectFileNoOpt(llvm::Module &llvmModule,
                                         llvm::TargetMachine *targetMachine,
                                         llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream dest(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Could not open output file: " << ec.message() << "\n";
    return failure();
  }

  llvm::legacy::PassManager pm;
  if (targetMachine->addPassesToEmitFile(pm, dest, nullptr,
                                          llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return failure();
  }

  pm.run(llvmModule);
  dest.flush();
  return success();
}

//===----------------------------------------------------------------------===//
// Link .o → .so via clang
//===----------------------------------------------------------------------===//

static LogicalResult linkToSharedObject(llvm::StringRef objectPath,
                                        llvm::StringRef outputPath) {
  // Find a C++ compiler in PATH. Prefer clang, fall back to gcc variants.
  auto clangPath = llvm::sys::findProgramByName("clang++");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("clang");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("g++");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("gcc");
  if (!clangPath) {
    llvm::errs()
        << "Error: cannot find 'clang', 'clang++', 'g++', or 'gcc' in PATH "
           "for linking\n";
    return failure();
  }

  llvm::SmallVector<llvm::StringRef> args;
  args.push_back(*clangPath);
  args.push_back("-shared");
  args.push_back("-fvisibility=hidden");
  args.push_back("-o");
  args.push_back(outputPath);
  args.push_back(objectPath);

  // On Linux, add -Wl,--no-undefined to catch missing symbols early.
  // Compiled code references __circt_sim_* runtime symbols which will be
  // resolved at dlopen time, so we use --allow-shlib-undefined instead.
#ifndef __APPLE__
  args.push_back("-Wl,--allow-shlib-undefined");
#endif

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(*clangPath, args, std::nullopt, {},
                                          /*SecondsToWait=*/60, 0, &errMsg);
  if (result != 0) {
    llvm::errs() << "Linking failed (exit code " << result << ")";
    if (!errMsg.empty())
      llvm::errs() << ": " << errMsg;
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Main compilation pipeline
//===----------------------------------------------------------------------===//

static LogicalResult compile(MLIRContext &mlirContext) {
  auto startTime = std::chrono::steady_clock::now();

  // Parse input MLIR.
  llvm::SourceMgr sourceMgr;
  std::string errorMessage;
  auto input = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    llvm::errs() << "Error: cannot open input file '" << inputFilename
                 << "'\n";
    return failure();
  }
  // Hash the input content before moving the buffer into sourceMgr, so we can
  // include it in the build ID for cache invalidation.
  std::string contentHash;
  {
    llvm::MD5 hasher;
    hasher.update((*input)->getBuffer());
    llvm::MD5::MD5Result result;
    hasher.final(result);
    llvm::SmallString<32> digest = result.digest();
    contentHash = std::string(digest);
  }

  sourceMgr.AddNewSourceBuffer(std::move(*input), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &mlirContext);

  auto module = parseSourceFile<ModuleOp>(sourceMgr, &mlirContext);
  if (!module) {
    llvm::errs() << "Error: failed to parse input\n";
    return failure();
  }

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] parse: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
  }

  // Compile process bodies (Phase A: callback processes).
  llvm::SmallVector<std::string> procNames;
  llvm::SmallVector<uint8_t> procKinds;
  unsigned totalProcesses = 0;
  llvm::StringMap<unsigned> procRejectionReasons;
  unsigned numProcsCompiled =
      compileProcessBodies(*module, procNames, procKinds,
                           &totalProcesses, &procRejectionReasons);
  if (numProcsCompiled > 0) {
    llvm::errs() << "[circt-sim-compile] Compiled " << numProcsCompiled
                 << " process bodies\n";
  }

  unsigned rejectedProcesses = totalProcesses - numProcsCompiled;
  llvm::errs() << "[circt-sim-compile] Processes: " << totalProcesses
               << " total, " << numProcsCompiled << " callback-eligible, "
               << rejectedProcesses << " rejected\n";

  if (verbose && !procRejectionReasons.empty()) {
    llvm::SmallVector<std::pair<llvm::StringRef, unsigned>> sorted;
    for (auto &kv : procRejectionReasons)
      sorted.push_back({kv.first(), kv.second});
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    llvm::errs() << "[circt-sim-compile] Top process rejection reasons:\n";
    for (unsigned i = 0; i < std::min<unsigned>(5, sorted.size()); ++i)
      llvm::errs() << "  " << sorted[i].second << "x " << sorted[i].first
                   << "\n";
  }

  // Track process function names to exclude from func_* descriptor arrays.
  llvm::StringSet<> procNameSet;
  for (const auto &name : procNames)
    procNameSet.insert(name);

  // Collect compilable func.func bodies.
  llvm::SmallVector<func::FuncOp> candidates;
  unsigned totalFuncs = 0, externalFuncs = 0, rejectedFuncs = 0;
  llvm::StringMap<unsigned> funcRejectionReasons;

  module->walk([&](func::FuncOp funcOp) {
    ++totalFuncs;
    if (funcOp.isExternal()) {
      ++externalFuncs;
      return;
    }
    std::string reason;
    if (!isFuncBodyCompilable(funcOp, &reason)) {
      ++rejectedFuncs;
      if (!reason.empty())
        ++funcRejectionReasons[reason];
      return;
    }
    candidates.push_back(funcOp);
  });

  llvm::errs() << "[circt-sim-compile] Functions: " << totalFuncs << " total, "
               << externalFuncs << " external, " << rejectedFuncs
               << " rejected, " << candidates.size() << " compilable\n";

  if (verbose && !funcRejectionReasons.empty()) {
    llvm::SmallVector<std::pair<llvm::StringRef, unsigned>> sorted;
    for (auto &kv : funcRejectionReasons)
      sorted.push_back({kv.first(), kv.second});
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    llvm::errs() << "[circt-sim-compile] Top function rejection reasons:\n";
    for (unsigned i = 0; i < std::min<unsigned>(5, sorted.size()); ++i)
      llvm::errs() << "  " << sorted[i].second << "x " << sorted[i].first
                   << "\n";
  }

  if (candidates.empty() && procNames.empty()) {
    llvm::errs() << "[circt-sim-compile] No compilable functions found\n";
    return failure();
  }

  // Build micro-module with only compilable functions.
  auto microModule = ModuleOp::create(UnknownLoc::get(&mlirContext));
  OpBuilder builder(&mlirContext);
  builder.setInsertionPointToEnd(microModule.getBody());
  IRMapping mapping;

  llvm::SmallVector<std::string> funcNames;
  funcNames.reserve(candidates.size());
  for (auto funcOp : candidates) {
    builder.clone(*funcOp, mapping);
    // Process functions go into proc_* arrays, not func_* arrays.
    if (!procNameSet.contains(funcOp.getSymName()))
      funcNames.push_back(funcOp.getSymName().str());
  }

  NativeModuleInitSynthesisStats nativeModuleInitStats =
      synthesizeNativeModuleInitFunctions(*module, microModule);
  if (nativeModuleInitStats.emittedModules > 0)
    llvm::errs() << "[circt-sim-compile] Native module init functions: "
                 << nativeModuleInitStats.emittedModules << "\n";
  if (verbose && nativeModuleInitStats.totalModules > 0) {
    llvm::errs() << "[circt-sim-compile] Native module init modules: "
                 << nativeModuleInitStats.emittedModules << " emitted / "
                 << nativeModuleInitStats.totalModules << " total\n";
    if (!nativeModuleInitStats.skipReasons.empty()) {
      llvm::SmallVector<std::pair<llvm::StringRef, unsigned>> sorted;
      for (auto &kv : nativeModuleInitStats.skipReasons)
        sorted.push_back({kv.first(), kv.second});
      llvm::sort(sorted, [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      llvm::errs()
          << "[circt-sim-compile] Top native module init skip reasons:\n";
      for (unsigned i = 0; i < std::min<unsigned>(5, sorted.size()); ++i)
        llvm::errs() << "  " << sorted[i].second << "x " << sorted[i].first
                     << "\n";
    }
  }

  cloneReferencedDeclarations(microModule, *module, mapping);

  // Normalize ref-like func argument ABIs in the micro-module to keep helper
  // call wrappers pointer-typed for LLVM lowering.
  canonicalizeLLHDRefArgumentABIs(microModule);

  // Strip bodies of intercepted functions so they become external
  // declarations → trampolines. This ensures compiled code calls back into
  // the interpreter for these functions, where interceptors can fire.
  // Without this, compiled→compiled direct calls bypass interceptors and
  // cause crashes (e.g., factory returning null, config_db not firing).
  //
  // The pattern matching here must be conservative: any function that MIGHT
  // have an interpreter interceptor should be demoted. False positives just
  // mean the function goes through the interpreter (slower but correct).
  // False negatives cause crashes.
  {
    // Legacy compatibility mode: keep blanket demotion of all uvm_ symbols.
    // Default is off so pure helpers with uvm_ prefixes can stay compiled.
    bool demoteAllUvmByPrefix =
        std::getenv("CIRCT_AOT_INTERCEPT_ALL_UVM") != nullptr;
    bool aggressiveNativeUvm =
        std::getenv("CIRCT_AOT_AGGRESSIVE_UVM") != nullptr;
    bool allowNativeUvmAlloc = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_ALLOC") != nullptr;
    bool allowNativeUvmTypeInfo = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_TYPEINFO") != nullptr;
    bool allowNativeUvmAccessors = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_ACCESSORS") != nullptr;
    bool allowNativeUvmHierarchy = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_HIERARCHY") != nullptr;
    bool allowNativeUvmPhaseGraph = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_PHASE_GRAPH") != nullptr;
    bool allowNativeUvmFactory =
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_FACTORY") != nullptr;
    bool allowNativeUvmReporting = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING") != nullptr;
    bool allowNativeUvmRandom = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_RANDOM") != nullptr;
    bool allowNativeMooreDelayCallers =
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_MOORE_DELAY_CALLERS") != nullptr;
    bool allowNativeUvmSingletonGetters = aggressiveNativeUvm ||
        std::getenv("CIRCT_AOT_ALLOW_NATIVE_UVM_SINGLETON_GETTERS") !=
            nullptr;
    auto isInterceptedFunc = [demoteAllUvmByPrefix, allowNativeUvmAlloc,
                              allowNativeUvmTypeInfo,
                              allowNativeUvmAccessors, allowNativeUvmHierarchy,
                              allowNativeUvmPhaseGraph,
                              allowNativeUvmFactory, allowNativeUvmReporting,
                              allowNativeUvmRandom,
                              allowNativeUvmSingletonGetters](
                                 llvm::StringRef name) -> bool {
      if (demoteAllUvmByPrefix && name.contains("uvm_"))
        return true;
      // Moore runtime support (__moore_delay, __moore_string_*, etc.)
      if (name.starts_with("__moore_"))
        return true;
      // Standalone intercepted functions (short names without uvm_ prefix)
      if (name == "self" || name == "malloc")
        return true;
      if (!allowNativeUvmSingletonGetters &&
          (name == "get" || name == "get_0" ||
           name == "get_common_domain" ||
           name == "get_global_hopper"))
        return true;
      // String/class conversion
      if (name.starts_with("to_string_") || name == "to_class" ||
          name.starts_with("to_class_"))
        return true;
      // Random state management
      if (name == "srandom" || name.starts_with("srandom_") ||
          name == "get_randstate" || name.starts_with("get_randstate_") ||
          name == "set_randstate" || name.starts_with("set_randstate_"))
        return true;
      // DB/port patterns (may exist outside uvm_ namespace)
      if (name.contains("config_db") || name.contains("resource_db") ||
          name.contains("analysis_port") || name.contains("seq_item_pull") ||
          name.contains("sqr_if_base"))
        return true;
      // BFM protocol interceptors (driver_bfm::, monitor_bfm::, i3c_*_bfm::, etc.)
      if (name.contains("_bfm::") || name.contains("Bfm::"))
        return true;
      // Die interceptor
      if (name.ends_with("::die"))
        return true;
      // Phase-graph/state mutators have interpreter-side cache/update logic.
      if (!allowNativeUvmPhaseGraph &&
          (name.contains("uvm_phase::") ||
           name.contains("uvm_component::set_domain")))
        return true;
      // Component hierarchy mutators update Moore-assoc-backed name/child maps.
      if (!allowNativeUvmHierarchy && name.contains("::m_add_child"))
        return true;
      // UVM-generated class methods with package-qualified names (no "uvm_"
      // prefix). Emitted by uvm_component_utils/uvm_object_utils macros.
      if (!allowNativeUvmAlloc &&
          (name == "create" || name.starts_with("create_") ||
           name.ends_with("::create")))
        return true;
      if (!allowNativeUvmTypeInfo &&
          (name == "m_initialize" || name.starts_with("m_initialize_") ||
           name.ends_with("::m_initialize")))
        return true;
      if (!allowNativeUvmTypeInfo && name.starts_with("m_register_cb"))
        return true;
      if (!allowNativeUvmTypeInfo &&
          (name.contains("get_object_type") || name.contains("get_type_name")))
        return true;
      // Type registry accessors (get_type_NNN mangled names)
      if (!allowNativeUvmTypeInfo &&
          (name.starts_with("get_type_") || name.contains("::get_type_")))
        return true;
      // TLM imp port accessors (get_imp_NNN mangled names)
      if (!allowNativeUvmAccessors &&
          (name.starts_with("get_imp_") || name.contains("::get_imp_")))
        return true;
      // Type name string helpers (type_name_NNN)
      if (!allowNativeUvmTypeInfo && name.starts_with("type_name_"))
        return true;
      // Factory override/creation paths are stateful and currently rely on
      // interpreter-side behavior; keep native-disabled by default.
      if (!allowNativeUvmFactory &&
          (name.contains("::uvm_factory::") ||
           name.contains("::uvm_default_factory::")))
        return true;
      // Reporting paths manage dynamic report handlers/messages and can crash
      // under native dispatch before interpreter-level handling runs.
      if (!allowNativeUvmReporting &&
          (name.contains("::uvm_report_handler::") ||
           name.contains("::uvm_report_object::") ||
           name.contains("::uvm_report_message::get_severity") ||
           name.contains("process_report_message")))
        return true;
      // Random-seeding paths build dynamic strings and rely on interpreter-side
      // pointer lifetime behavior.
      if (!allowNativeUvmRandom &&
          (name.contains("create_random_seed") ||
           name.contains("::reseed")))
        return true;
      // Constructors (may allocate objects the interpreter tracks)
      if (!allowNativeUvmAlloc && name.ends_with("::new"))
        return true;
      // UVM singleton/registry accessors
      if (!allowNativeUvmAccessors &&
          (name.contains("::is_auditing") ||
           name.contains("get_print_config_matches") ||
           name.contains("get_root_blocks") ||
           name == "get_inst" || name.starts_with("get_inst_")))
        return true;
      // UVM objection methods
      if (name.contains("raise_objection") || name.contains("drop_objection"))
        return true;

      // UVM execute_phase
      if (name.ends_with("::execute_phase") || name == "execute_phase")
        return true;

      // Sequence body methods
      if (name.ends_with("::body") && name.contains("_seq"))
        return true;
      return false;
    };

    unsigned demoted = 0;
    llvm::SmallVector<func::FuncOp> toDemote;
    auto callsMooreDelay = [](func::FuncOp funcOp) {
      bool found = false;
      funcOp.walk([&](Operation *op) {
        if (auto llvmCall = dyn_cast<LLVM::CallOp>(op)) {
          if (auto callee = llvmCall.getCallee();
              callee && *callee == "__moore_delay") {
            found = true;
            return WalkResult::interrupt();
          }
        }
        if (auto funcCall = dyn_cast<func::CallOp>(op)) {
          if (funcCall.getCallee() == "__moore_delay") {
            found = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      return found;
    };
    microModule.walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      if (isInterceptedFunc(funcOp.getSymName()) ||
          (!allowNativeMooreDelayCallers && callsMooreDelay(funcOp)))
        toDemote.push_back(funcOp);
    });
    for (auto funcOp : toDemote) {
      llvm::StringRef name = funcOp.getSymName();
      // Erase and re-create as external declaration (empty body).
      auto funcType = funcOp.getFunctionType();
      auto loc = funcOp.getLoc();
      auto symName = funcOp.getSymName().str();
      funcOp.erase();
      builder.setInsertionPointToEnd(microModule.getBody());
      auto newDecl = builder.create<func::FuncOp>(loc, symName, funcType);
      newDecl.setPrivate();
      ++demoted;
      for (auto &fn : funcNames) {
        if (fn == name)
          fn.clear();
      }
    }
    // Clean up empty names.
    llvm::erase_if(funcNames, [](const std::string &s) { return s.empty(); });
    if (demoted > 0)
      llvm::errs() << "[circt-sim-compile] Demoted " << demoted
                   << " intercepted functions to trampolines\n";
  }

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] clone: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
  }

  // Lower SCF structured control flow (if/for/while/...) to cf.* first.
  // This is best-effort: if conversion fails for some function body, keep
  // going and rely on later stripping of residual non-LLVM bodies.
  if (!lowerSCFToCF(microModule))
    llvm::errs() << "[circt-sim-compile] Warning: SCF->CF lowering failed; "
                    "falling back to residual-op stripping\n";

  // Lower arith/cf/func → LLVM dialect.
  if (!lowerFuncArithCfToLLVM(microModule, mlirContext)) {
    llvm::errs() << "[circt-sim-compile] Lowering failed\n";
    microModule.erase();
    return failure();
  }

  // Flatten aggregate ABIs: pass struct/array args by pointer.
  flattenAggregateFunctionABIs(microModule);

  // Strip functions with residual non-LLVM ops.
  auto stripped = stripNonLLVMFunctions(microModule);
  if (!stripped.empty()) {
    llvm::DenseSet<llvm::StringRef> strippedSet;
    for (const auto &name : stripped)
      strippedSet.insert(name);
    // Remove stripped names from the function and process lists.
    llvm::SmallVector<std::string> survivingNames;
    for (auto &name : funcNames) {
      if (!strippedSet.contains(name))
        survivingNames.push_back(std::move(name));
    }
    funcNames = std::move(survivingNames);
    // Also strip from process names/kinds.
    llvm::SmallVector<std::string> survivingProcs;
    llvm::SmallVector<uint8_t> survivingKinds;
    for (unsigned i = 0; i < procNames.size(); ++i) {
      if (!strippedSet.contains(procNames[i])) {
        survivingProcs.push_back(std::move(procNames[i]));
        survivingKinds.push_back(procKinds[i]);
      }
    }
    procNames = std::move(survivingProcs);
    procKinds = std::move(survivingKinds);
    llvm::errs() << "[circt-sim-compile] Stripped " << stripped.size()
                 << " functions with non-LLVM ops\n";
  }

  // Collect vtable FuncId assignments from the original module.
  // This must happen BEFORE trampoline generation so we can ensure all
  // vtable functions have either compiled bodies or trampolines.
  auto allFuncEntryNames = collectVtableFuncIds(*module);
  auto taggedVtableAssignments = collectTaggedVtableAssignments(*module);
  llvm::errs() << "[circt-sim-compile] Collected " << allFuncEntryNames.size()
               << " vtable FuncIds\n";
  if (verbose && !taggedVtableAssignments.empty())
    llvm::errs() << "[circt-sim-compile] Tagged vtable globals discovered: "
                 << taggedVtableAssignments.size() << "\n";

  // Ensure all vtable functions have declarations in the micro-module so
  // that generateTrampolines() will create trampolines for uncompiled ones.
  {
    auto *mlirCtx = microModule.getContext();
    OpBuilder declBuilder(mlirCtx);
    declBuilder.setInsertionPointToEnd(microModule.getBody());

    for (const auto &name : allFuncEntryNames) {
      // Skip if already present (compiled or already declared).
      if (microModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        continue;
      // Look up the function type from the original module.
      // Try LLVM::LLVMFuncOp first (already lowered), then func::FuncOp
      // (user functions in the original module are func::FuncOp, not LLVM).
      auto origFunc = module->lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (origFunc) {
        // Clone as external declaration.
        auto declOp = LLVM::LLVMFuncOp::create(
            declBuilder, origFunc.getLoc(), name, origFunc.getFunctionType());
        declOp.setLinkage(LLVM::Linkage::External);
      } else if (auto funcFunc =
                     module->lookupSymbol<func::FuncOp>(name)) {
        // Convert func::FuncOp's FunctionType to LLVM::LLVMFunctionType.
        auto funcType = funcFunc.getFunctionType();
        SmallVector<Type> argTypes;
        bool unsupported = false;
        for (auto ty : funcType.getInputs()) {
          if (isa<IntegerType>(ty) || isa<LLVM::LLVMPointerType>(ty) ||
              isa<FloatType>(ty)) {
            argTypes.push_back(ty);
          } else if (isa<IndexType>(ty)) {
            argTypes.push_back(IntegerType::get(mlirCtx, 64));
          } else {
            unsupported = true;
            break;
          }
        }
        if (!unsupported) {
          Type retType;
          if (funcType.getNumResults() == 0) {
            retType = LLVM::LLVMVoidType::get(mlirCtx);
          } else if (funcType.getNumResults() == 1) {
            auto rt = funcType.getResult(0);
            if (isa<IntegerType>(rt) || isa<LLVM::LLVMPointerType>(rt) ||
                isa<FloatType>(rt)) {
              retType = rt;
            } else if (isa<IndexType>(rt)) {
              retType = IntegerType::get(mlirCtx, 64);
            } else {
              unsupported = true;
            }
          } else {
            unsupported = true;
          }
          if (!unsupported) {
            auto llvmFuncType = LLVM::LLVMFunctionType::get(retType, argTypes);
            auto declOp = LLVM::LLVMFuncOp::create(declBuilder,
                                                     funcFunc.getLoc(), name,
                                                     llvmFuncType);
            declOp.setLinkage(LLVM::Linkage::External);
          }
        }
      }
      // If not found in original module (or unsupported type), skip.
      // The entry table will have a null entry which is fine (interpreter
      // handles it).
    }
  }

  // Generate trampolines for uncompiled functions referenced by compiled code.
  // This now includes vtable functions that aren't compiled.
  auto trampolineNames = generateTrampolines(microModule);
  if (!trampolineNames.empty()) {
    llvm::errs() << "[circt-sim-compile] Generated " << trampolineNames.size()
                 << " interpreter trampolines\n";
  }

  llvm::errs() << "[circt-sim-compile] " << funcNames.size()
               << " functions + " << procNames.size()
               << " processes ready for codegen\n";

  if (funcNames.empty() && procNames.empty()) {
    llvm::errs() << "[circt-sim-compile] No functions survived lowering\n";
    microModule.erase();
    return failure();
  }

  // Verify the module.
  if (failed(mlir::verify(microModule))) {
    llvm::errs() << "[circt-sim-compile] Module verification failed\n";
    microModule.erase();
    return failure();
  }

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] lower: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
  }

  // Translate MLIR LLVM dialect → LLVM IR.
  registerBuiltinDialectTranslation(mlirContext);
  registerLLVMDialectTranslation(mlirContext);

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(microModule, llvmContext);
  microModule.erase();

  if (!llvmModule) {
    llvm::errs() << "[circt-sim-compile] LLVM IR translation failed\n";
    return failure();
  }

  // Flatten mutable global variable types to [N x i8] byte arrays using the
  // packed (unpadded) size.  This prevents LLVM from re-introducing alignment
  // padding and keeps the layout consistent with the i8-based GEP rewrites
  // performed below.
  {
    unsigned flatCount = flattenGlobalTypesToByteArrays(*llvmModule);
    if (flatCount > 0)
      llvm::errs() << "[circt-sim-compile] Flattened " << flatCount
                   << " globals to byte arrays\n";
  }

  // Rewrite struct-indexed GEPs to use packed (unaligned) byte offsets matching
  // the interpreter's struct layout.  Must run BEFORE the optimizer so all 994+
  // GEPs are caught before LLVM converts them to raw pointer arithmetic.
  {
    unsigned gepCount = rewriteGEPsForPackedLayout(*llvmModule);
    if (gepCount > 0)
      llvm::errs() << "[circt-sim-compile] Rewrote " << gepCount
                   << " struct GEPs for packed layout\n";
  }

  // Compute build ID: encode ABI version, target triple, and a content hash of
  // the input file so that stale cached .so files can be detected at load time.
  std::string buildId = "circt-sim-abi-v" +
                        std::to_string(CIRCT_SIM_ABI_VERSION) + "-" +
                        llvm::sys::getDefaultTargetTriple() + "-" +
                        contentHash;

  // Collect mutable globals for the patch table (including vtable globals).
  // Vtable globals are included so their .so storage gets real addresses that
  // native code can dereference when reading vtable_ptr from objects.
  llvm::SmallVector<std::string> globalPatchNames;
  llvm::SmallVector<llvm::GlobalVariable *> globalPatchVars;
  llvm::SmallVector<uint32_t> globalPatchSizes;
  for (auto &global : llvmModule->globals()) {
    if (global.isConstant() || global.isDeclaration())
      continue;
    if (!global.hasName() || global.getName().empty())
      continue;
    if (global.getName().starts_with("__circt_sim_"))
      continue;
    globalPatchNames.push_back(global.getName().str());
    globalPatchVars.push_back(&global);
    auto *type = global.getValueType();
    // Use packed (no-padding) size to match the interpreter's unaligned layout.
    uint64_t size = computePackedTypeAllocSize(type, llvmModule->getDataLayout());
    globalPatchSizes.push_back(static_cast<uint32_t>(size));
  }
  llvm::errs() << "[circt-sim-compile] Global patches: "
               << globalPatchNames.size() << " mutable globals\n";

  // Ensure compiled loads from vtable globals observe the same tagged FuncId
  // addresses (0xF0000000+N) used by the interpreter.
  {
    unsigned initializedVtables = initializeTaggedVtableGlobals(
        *llvmModule, taggedVtableAssignments);
    if (initializedVtables > 0)
      llvm::errs() << "[circt-sim-compile] Initialized "
                   << initializedVtables
                   << " vtable globals with tagged FuncIds\n";
  }

  // Synthesize descriptor and entrypoints into the LLVM IR module.
  synthesizeDescriptor(*llvmModule, funcNames, trampolineNames,
                       procNames, procKinds, buildId,
                       globalPatchNames, globalPatchVars, globalPatchSizes,
                       allFuncEntryNames);

  // Rewrite indirect calls through tagged synthetic vtable addresses
  // (0xF0000000+N) to use the func_entries table. Must run after
  // synthesizeDescriptor (which creates @__circt_sim_func_entries) and before
  // optimization passes (so the optimizer can simplify the tagged checks).
  runLowerTaggedIndirectCalls(*llvmModule);

  // Set hidden visibility on everything except entrypoints.
  setDefaultHiddenVisibility(*llvmModule);

  // Internalize non-exported functions to enable whole-program optimization
  // (inlining across function boundaries, GlobalDCE of dead helpers).
  internalizeNonExported(*llvmModule);

  // Mark small functions as alwaysinline so the optimizer inlines them even
  // at O1. This is cheap and produces significantly better codegen for the
  // many small accessor/wrapper functions that appear in compiled MLIR.
  unsigned alwaysInlineCount = addAlwaysInlineToSmallFunctions(*llvmModule);

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] translate: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
    llvm::errs() << "[circt-sim-compile] internalize: marked "
                 << alwaysInlineCount
                 << " small functions as alwaysinline\n";
  }

  // Determine output path.
  // When --emit-snapshot is active, -o specifies the snapshot directory, not
  // the .so. We write the .so to a temporary path and then move it into the
  // snapshot directory later.
  std::string outPath;
  if (emitSnapshot) {
    // .so goes to a temp file; it will be moved into the snapshot dir below.
    llvm::SmallString<256> tmpSo;
    if (auto ec = llvm::sys::fs::createTemporaryFile("circt-sim-compile",
                                                      "so", tmpSo)) {
      llvm::errs() << "Error creating temp file: " << ec.message() << "\n";
      return failure();
    }
    outPath = std::string(tmpSo);
  } else {
    outPath = outputFilename;
    if (outPath.empty()) {
      if (inputFilename == "-") {
        outPath = "a.out.so";
      } else {
        llvm::SmallString<256> p(inputFilename);
        llvm::sys::path::replace_extension(p, ".so");
        outPath = std::string(p);
      }
    }
  }

  // Initialize LLVM targets (needed for createHostTargetMachine below).
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create the target machine and set the module's data layout.  This must
  // happen before optimization so the optimizer uses the correct layout.
  auto *targetMachine = createHostTargetMachine(*llvmModule, optLevel);
  if (!targetMachine) {
    llvm::errs() << "[circt-sim-compile] Target machine creation failed\n";
    return failure();
  }

  ModuleStats preStats;
  if (verbose) {
    preStats = collectModuleStats(*llvmModule);
    llvm::errs() << "[circt-sim-compile] pre-opt: " << preStats.definedFnCount
                 << " functions (" << preStats.internalFnCount
                 << " internal), " << preStats.callCount << " calls\n";
  }

  // Run LLVM optimization passes.
  runOptimizationPasses(*llvmModule, targetMachine, optLevel);

  if (verbose) {
    auto postStats = collectModuleStats(*llvmModule);
    unsigned dce = preStats.internalFnCount > postStats.internalFnCount
                       ? preStats.internalFnCount - postStats.internalFnCount
                       : 0;
    llvm::errs() << "[circt-sim-compile] post-opt: " << postStats.definedFnCount
                 << " functions (" << postStats.internalFnCount << " internal, "
                 << dce << " DCE'd), " << postStats.callCount << " calls\n";
  }

  // Emit LLVM IR if requested (after optimization and GEP rewrite so the
  // dumped IR reflects exactly what will be compiled to object code).
  if (emitLLVM) {
    llvm::SmallString<256> llPath(outPath);
    if (llPath.ends_with(".so"))
      llvm::sys::path::replace_extension(llPath, ".ll");
    std::error_code ec;
    llvm::raw_fd_ostream out(llPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Error opening output: " << ec.message() << "\n";
      return failure();
    }
    llvmModule->print(out, nullptr);
    llvm::errs() << "[circt-sim-compile] Wrote LLVM IR to " << llPath << "\n";
    return success();
  }

  // Emit object file.
  llvm::SmallString<256> objPath;
  if (emitObject) {
    // Direct .o output.
    objPath = outPath;
    if (!objPath.ends_with(".o"))
      llvm::sys::path::replace_extension(objPath, ".o");
  } else {
    // Temporary .o for linking.
    std::error_code ec =
        llvm::sys::fs::createTemporaryFile("circt-sim-compile", "o", objPath);
    if (ec) {
      llvm::errs() << "Error creating temp file: " << ec.message() << "\n";
      return failure();
    }
  }

  if (failed(emitObjectFileNoOpt(*llvmModule, targetMachine, objPath))) {
    llvm::errs() << "[circt-sim-compile] Object emission failed\n";
    return failure();
  }

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] codegen: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
  }

  if (emitObject) {
    llvm::errs() << "[circt-sim-compile] Wrote object to " << objPath << "\n";
    return success();
  }

  // Link .o → .so.
  if (failed(linkToSharedObject(objPath, outPath))) {
    llvm::errs() << "[circt-sim-compile] Linking failed\n";
    llvm::sys::fs::remove(objPath);
    return failure();
  }

  // Clean up temporary .o.
  llvm::sys::fs::remove(objPath);

  // Report output file size.
  uint64_t fileSize = 0;
  if (!llvm::sys::fs::file_size(outPath, fileSize))
    llvm::errs() << "[circt-sim-compile] Output size: " << (fileSize / 1024)
                 << " KB\n";

  auto elapsed = std::chrono::steady_clock::now() - startTime;
  llvm::errs() << "[circt-sim-compile] Wrote " << outPath << " ("
               << procNames.size() << " processes, "
               << funcNames.size() << " functions, "
               << trampolineNames.size() << " trampolines, "
               << std::chrono::duration<double>(elapsed).count() << "s)\n";

  // If --emit-snapshot, bundle the .so + MLIR bytecode + metadata into a
  // snapshot directory that circt-sim can load directly.
  if (emitSnapshot) {
    // Determine snapshot directory path: use -o value or derive from input.
    std::string snapDir = outputFilename;
    if (snapDir.empty()) {
      if (inputFilename == "-") {
        snapDir = "a.out.csnap";
      } else {
        llvm::SmallString<256> p(inputFilename);
        llvm::sys::path::replace_extension(p, ".csnap");
        snapDir = std::string(p);
      }
    }
    // Ensure the directory ends with .csnap for clarity (but don't force it).
    if (auto ec = llvm::sys::fs::create_directories(snapDir)) {
      llvm::errs() << "[circt-sim-compile] Error creating snapshot dir "
                   << snapDir << ": " << ec.message() << "\n";
      return failure();
    }

    // 1. Move the .so into the snapshot directory as native.so.
    llvm::SmallString<256> soDestPath(snapDir);
    llvm::sys::path::append(soDestPath, "native.so");
    if (auto ec = llvm::sys::fs::rename(outPath, soDestPath)) {
      // rename can fail across filesystems; fall back to copy + remove.
      if (auto ec2 = llvm::sys::fs::copy_file(outPath, soDestPath)) {
        llvm::errs() << "[circt-sim-compile] Error copying .so to snapshot: "
                     << ec2.message() << "\n";
        return failure();
      }
      llvm::sys::fs::remove(outPath);
    }

    // 2. Save the MLIR module as bytecode.
    llvm::SmallString<256> bcPath(snapDir);
    llvm::sys::path::append(bcPath, "design.mlirbc");
    {
      std::error_code ec;
      llvm::raw_fd_ostream os(bcPath, ec);
      if (ec) {
        llvm::errs() << "[circt-sim-compile] Error opening " << bcPath << ": "
                     << ec.message() << "\n";
        return failure();
      }
      mlir::BytecodeWriterConfig bcConfig;
      if (mlir::failed(
              mlir::writeBytecodeToFile(module->getOperation(), os, bcConfig))) {
        llvm::errs()
            << "[circt-sim-compile] Error writing bytecode to " << bcPath
            << "\n";
        return failure();
      }
      os.flush();
    }

    // 3. Write meta.json with snapshot metadata.
    llvm::SmallString<256> metaPath(snapDir);
    llvm::sys::path::append(metaPath, "meta.json");
    {
      std::error_code ec;
      llvm::raw_fd_ostream os(metaPath, ec);
      if (ec) {
        llvm::errs() << "[circt-sim-compile] Error opening " << metaPath
                     << ": " << ec.message() << "\n";
        return failure();
      }
      // Build a timestamp string.
      auto now = std::chrono::system_clock::now();
      auto nowTime = std::chrono::system_clock::to_time_t(now);
      char timeBuf[64];
      std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%SZ",
                    std::gmtime(&nowTime));

      os << "{\n";
      os << "  \"abi_version\": " << CIRCT_SIM_ABI_VERSION << ",\n";
      os << "  \"preprocessed\": true,\n";
      os << "  \"timestamp\": \"" << timeBuf << "\",\n";
      os << "  \"content_hash\": \"" << contentHash << "\"\n";
      os << "}\n";
      os.flush();
    }

    llvm::errs() << "[circt-sim-compile] Wrote snapshot to " << snapDir
                 << "/\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "CIRCT simulation AOT compiler\n");

  // Register all required MLIR dialects.
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
                          func::FuncDialect, scf::SCFDialect,
                          LLVM::LLVMDialect, hw::HWDialect, comb::CombDialect,
                          llhd::LLHDDialect, moore::MooreDialect,
                          seq::SeqDialect, sim::SimDialect,
                          verif::VerifDialect>();

  // Allow unregistered dialects so we can parse arbitrary MLIR.
  mlirContext.allowUnregisteredDialects();

  if (succeeded(compile(mlirContext)))
    return 0;
  return 1;
}
