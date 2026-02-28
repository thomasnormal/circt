//===- AOTProcessCompiler.cpp - AOT batch process compilation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AOTProcessCompiler.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include <chrono>
#include <functional>

#ifdef CIRCT_SIM_JIT_ENABLED
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#endif

#define DEBUG_TYPE "aot-process-jit"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

/// Register runtime symbols with the JIT engine. The MLIR ExecutionEngine
/// already resolves process-local symbols via dlsym(), so __moore_* and
/// __circt_sim_* functions linked into the binary are found automatically.
/// This hook exists as an extension point for any additional symbol overrides.
static void registerJITRuntimeSymbols(mlir::ExecutionEngine *) {
  // No-op: runtime symbols are resolved via dlsym() by the ExecutionEngine.
}

//===----------------------------------------------------------------------===//
// Type conversions for AOT compilation (same as FullProcessJIT)
//===----------------------------------------------------------------------===//

static void addProcessTypeConversions(LLVMTypeConverter &converter) {
  // llhd.time → i64 (packed delay encoding)
  converter.addConversion([](llhd::TimeType type) -> Type {
    return IntegerType::get(type.getContext(), 64);
  });

  // llhd.ref<T> → !llvm.ptr
  converter.addConversion([](llhd::RefType type) -> Type {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
}

//===----------------------------------------------------------------------===//
// Lowering patterns (reused from FullProcessJIT.cpp)
//===----------------------------------------------------------------------===//

namespace {

// Pack llhd.time into the 64-bit runtime delay encoding expected by
// __circt_sim_yield / scheduler bridge:
//   bits [63:32] real-time femtoseconds
//   bits [31:16] delta
//   bits [15:0]  epsilon
static int64_t encodeJITDelay(uint64_t realTimeFs, uint32_t delta,
                              uint32_t epsilon) {
  uint64_t packedRealTime = std::min<uint64_t>(realTimeFs, 0xFFFFFFFFULL);
  uint64_t packedDelta = std::min<uint32_t>(delta, 0xFFFFU);
  uint64_t packedEpsilon = std::min<uint32_t>(epsilon, 0xFFFFU);
  uint64_t bits = (packedRealTime << 32) | (packedDelta << 16) | packedEpsilon;
  return static_cast<int64_t>(bits);
}

/// Lower llhd.prb to signal read.
struct ProcessProbeOpLowering : public OpConversionPattern<llhd::ProbeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ProbeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto module = op->getParentOfType<ModuleOp>();

    if (auto intTy = dyn_cast<IntegerType>(resultType)) {
      unsigned width = intTy.getWidth();
      if (width > 0 && width <= 64) {
        auto i64Ty = rewriter.getI64Type();
        auto baseFuncOp = LLVM::lookupOrCreateFn(
            rewriter, module, "__circt_sim_signal_memory_base", {}, ptrTy);
        if (failed(baseFuncOp))
          return failure();
        auto baseCall =
            LLVM::CallOp::create(rewriter, loc, *baseFuncOp, ValueRange{});
        auto sigIndex = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                                  adaptor.getSignal());
        auto elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                            baseCall.getResult(),
                                            ValueRange{sigIndex});
        auto word = LLVM::LoadOp::create(rewriter, loc, i64Ty, elemPtr);
        if (width < 64) {
          rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, intTy, word);
        } else {
          rewriter.replaceOp(op, word.getResult());
        }
        return success();
      }
    }

    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__arc_sched_read_signal", {ptrTy},
                                          ptrTy);
    if (failed(funcOp))
      return failure();
    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                        ValueRange{adaptor.getSignal()});
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                               callOp.getResult());
    return success();
  }
};

/// Lower llhd.drv to signal drive.
struct ProcessDriveOpLowering : public OpConversionPattern<llhd::DriveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::DriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i8Ty = rewriter.getIntegerType(8);
    auto valueConverted = adaptor.getValue();
    auto valueType = valueConverted.getType();
    Value enableVal;
    if (op.getEnable())
      enableVal = LLVM::ZExtOp::create(rewriter, loc, i8Ty, adaptor.getEnable());
    else
      enableVal = LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           rewriter.getIntegerAttr(i8Ty, 1));
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    if (auto intTy = dyn_cast<IntegerType>(valueType)) {
      unsigned width = intTy.getWidth();
      if (width > 0 && width <= 64) {
        auto i32Ty = rewriter.getIntegerType(32);
        auto sigIdI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                                  adaptor.getSignal());
        auto sigIdI32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, sigIdI64);
        Value valueI64;
        if (width < 64)
          valueI64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, valueConverted);
        else
          valueI64 = valueConverted;
        auto fastDriveFn = LLVM::lookupOrCreateFn(
            rewriter, module, "__arc_sched_drive_signal_fast",
            {i32Ty, i64Ty, i64Ty, i8Ty}, voidTy);
        if (failed(fastDriveFn))
          return failure();
        LLVM::CallOp::create(rewriter, loc, *fastDriveFn,
                              ValueRange{sigIdI32, valueI64,
                                         adaptor.getTime(), enableVal});
        rewriter.eraseOp(op);
        return success();
      }
    }
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(1));
    auto allocaOp =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, valueType, one);
    LLVM::StoreOp::create(rewriter, loc, valueConverted, allocaOp);
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__arc_sched_drive_signal",
        {ptrTy, ptrTy, i64Ty, i8Ty}, voidTy);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp,
                          ValueRange{adaptor.getSignal(), allocaOp,
                                     adaptor.getTime(), enableVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower llhd.constant_time to packed i64 delay.
struct ProcessConstantTimeLowering
    : public OpConversionPattern<llhd::ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto timeAttr = op.getValueAttr();
    int64_t encoded = 0;
    if (auto time = llvm::dyn_cast<llhd::TimeAttr>(timeAttr)) {
      uint64_t realFs = time.getTime();
      llvm::StringRef unit = time.getTimeUnit();
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
      encoded = encodeJITDelay(realFs, time.getDelta(), time.getEpsilon());
    }
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(encoded));
    return success();
  }
};

/// Lower llhd.int_to_time: pack the raw femtosecond integer into the JIT
/// delay encoding format (realTimeFs in bits [63:32]).
struct ProcessIntToTimeLowering
    : public OpConversionPattern<llhd::IntToTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::IntToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto rawFs = adaptor.getOperands()[0];
    auto shift = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(32));
    rewriter.replaceOpWithNewOp<LLVM::ShlOp>(op, i64Ty, rawFs, shift);
    return success();
  }
};

/// Lower llhd.time_to_int: extract the raw femtosecond integer from the JIT
/// delay encoding format (realTimeFs in bits [63:32]).
struct ProcessTimeToIntLowering
    : public OpConversionPattern<llhd::TimeToIntOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::TimeToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto packed = adaptor.getOperands()[0];
    auto shift = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(32));
    rewriter.replaceOpWithNewOp<LLVM::LShrOp>(op, i64Ty, packed, shift);
    return success();
  }
};

/// Lower llhd.wait to a call to __circt_sim_yield + branch to dest block.
struct ProcessWaitOpLowering : public OpConversionPattern<llhd::WaitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getIntegerType(32);
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto module = op->getParentOfType<ModuleOp>();
    auto yieldFuncOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__circt_sim_yield",
        {i32Ty, i64Ty, ptrTy, ptrTy, i32Ty}, voidTy);
    if (failed(yieldFuncOp))
      return failure();
    Value yieldKind, yieldData;
    bool hasDelay = adaptor.getDelay() != nullptr;
    if (hasDelay) {
      yieldKind = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getIntegerAttr(i32Ty, static_cast<int32_t>(YieldKind::WaitDelay)));
      yieldData = adaptor.getDelay();
    } else {
      yieldKind = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getIntegerAttr(i32Ty, static_cast<int32_t>(YieldKind::WaitSignal)));
      yieldData = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    }
    int32_t numSignals = static_cast<int32_t>(adaptor.getObserved().size());
    Value signalIdsPtr, edgeTypesPtr;
    Value numSignalsVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, numSignals));
    if (numSignals > 0) {
      auto one = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(1));
      auto arrayTy = LLVM::LLVMArrayType::get(i32Ty, numSignals);
      auto sigAlloca = LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, arrayTy, one);
      auto edgeAlloca = LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, arrayTy, one);
      for (int32_t i = 0; i < numSignals; i++) {
        Value sigHandle = adaptor.getObserved()[i];
        auto ptrAsInt =
            LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, sigHandle);
        auto sigId = LLVM::TruncOp::create(rewriter, loc, i32Ty, ptrAsInt);
        auto idx = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(i));
        auto gepSig = LLVM::GEPOp::create(
            rewriter, loc, ptrTy, arrayTy, sigAlloca, ValueRange{idx});
        LLVM::StoreOp::create(rewriter, loc, sigId, gepSig);
        auto edgeVal = LLVM::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));
        auto gepEdge = LLVM::GEPOp::create(
            rewriter, loc, ptrTy, arrayTy, edgeAlloca, ValueRange{idx});
        LLVM::StoreOp::create(rewriter, loc, edgeVal, gepEdge);
      }
      signalIdsPtr = sigAlloca;
      edgeTypesPtr = edgeAlloca;
    } else {
      signalIdsPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      edgeTypesPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }
    LLVM::CallOp::create(
        rewriter, loc, *yieldFuncOp,
        ValueRange{yieldKind, yieldData, signalIdsPtr, edgeTypesPtr,
                   numSignalsVal});
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        op, adaptor.getDestOperands(), op.getDest());
    return success();
  }
};

/// Lower llhd.wait to `ret void` for run-to-completion callback processes.
/// The compiled function represents one iteration of the body — it returns
/// instead of yielding, and the scheduler re-suspends on the same signals.
struct ProcessWaitOpCallbackLowering
    : public OpConversionPattern<llhd::WaitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // For callback processes, the wait is the end of one body iteration.
    // Replace with `return void`.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
    return success();
  }
};

/// Lower llhd.halt to __circt_sim_yield(Halt) + unreachable.
struct ProcessHaltOpLowering : public OpConversionPattern<llhd::HaltOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::HaltOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getIntegerType(32);
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto module = op->getParentOfType<ModuleOp>();
    auto yieldFuncOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__circt_sim_yield",
        {i32Ty, i64Ty, ptrTy, ptrTy, i32Ty}, voidTy);
    if (failed(yieldFuncOp))
      return failure();
    auto haltKind = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getIntegerAttr(i32Ty, static_cast<int32_t>(YieldKind::Halt)));
    auto zero64 = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    auto zero32 = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getIntegerAttr(i32Ty, 0));
    LLVM::CallOp::create(
        rewriter, loc, *yieldFuncOp,
        ValueRange{haltKind, zero64, nullPtr, nullPtr, zero32});
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
    return success();
  }
};

/// Lower llhd.sig projection ops to passthrough (identity).
template <typename OpT>
struct ProcessSigProjectionPassthrough : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Lower sim.proc.print to a call to __circt_sim_proc_print(ptr).
struct ProcessPrintOpLowering
    : public OpConversionPattern<sim::PrintFormattedProcOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::PrintFormattedProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__circt_sim_proc_print", {ptrTy},
                                          voidTy);
    if (failed(funcOp))
      return failure();
    Value input = adaptor.getInput();
    if (input.getType() != ptrTy)
      input = LLVM::BitcastOp::create(rewriter, loc, ptrTy, input);
    LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange{input});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.terminate to a no-op (erase). In AOT-compiled processes,
/// termination is communicated through __circt_sim_yield(Halt). The
/// sim.terminate side effects (setting terminationRequested) are handled
/// by the scheduler when it observes the Halt yield kind.
struct TerminateOpErasure : public OpConversionPattern<sim::TerminateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::TerminateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.fmt.literal to a global constant string + runtime call.
struct FmtLiteralOpLowering
    : public OpConversionPattern<sim::FormatLiteralOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();

    llvm::StringRef literal = op.getLiteral();

    // Create a global constant for the string data.
    auto globalName =
        ("__circt_fmt_lit_" + llvm::Twine(llvm::hash_value(literal))).str();
    LLVM::GlobalOp global;
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
      global = existing;
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto strType =
          LLVM::LLVMArrayType::get(rewriter.getI8Type(), literal.size());
      global = LLVM::GlobalOp::create(
          rewriter, loc, strType, /*isConstant=*/true,
          LLVM::Linkage::Internal, globalName,
          rewriter.getStringAttr(literal));
    }

    // Get pointer to the global string.
    Value addr = LLVM::AddressOfOp::create(rewriter, loc, global);
    Value len =
        LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                 rewriter.getI64IntegerAttr(literal.size()));

    auto voidPtrTy = ptrTy;
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__circt_sim_fmt_literal", {ptrTy, i64Ty}, voidPtrTy);
    if (failed(funcOp))
      return failure();
    auto call =
        LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange{addr, len});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.dec to a runtime call.
struct FmtDecOpLowering : public OpConversionPattern<sim::FormatDecOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatDecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto i8Ty = rewriter.getI8Type();
    auto module = op->getParentOfType<ModuleOp>();

    Value input = adaptor.getValue();
    unsigned width = op.getValue().getType().getIntOrFloatBitWidth();
    bool isSigned = op.getIsSigned();

    // Extend/truncate to i64.
    Value val;
    if (width < 64) {
      if (isSigned)
        val = LLVM::SExtOp::create(rewriter, loc, i64Ty, input);
      else
        val = LLVM::ZExtOp::create(rewriter, loc, i64Ty, input);
    } else if (width == 64) {
      val = input;
    } else {
      val = LLVM::TruncOp::create(rewriter, loc, i64Ty, input);
    }

    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(width));
    Value signedVal = LLVM::ConstantOp::create(
        rewriter, loc, i8Ty, rewriter.getI8IntegerAttr(isSigned ? 1 : 0));

    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__circt_sim_fmt_dec", {i64Ty, i32Ty, i8Ty}, ptrTy);
    if (failed(funcOp))
      return failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                     ValueRange{val, widthVal, signedVal});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.hex to a runtime call.
struct FmtHexOpLowering : public OpConversionPattern<sim::FormatHexOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatHexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto module = op->getParentOfType<ModuleOp>();

    Value input = adaptor.getValue();
    unsigned width = op.getValue().getType().getIntOrFloatBitWidth();

    Value val;
    if (width < 64)
      val = LLVM::ZExtOp::create(rewriter, loc, i64Ty, input);
    else if (width == 64)
      val = input;
    else
      val = LLVM::TruncOp::create(rewriter, loc, i64Ty, input);

    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(width));

    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__circt_sim_fmt_hex",
                                          {i64Ty, i32Ty}, ptrTy);
    if (failed(funcOp))
      return failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                     ValueRange{val, widthVal});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.bin to a runtime call.
struct FmtBinOpLowering : public OpConversionPattern<sim::FormatBinOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatBinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto module = op->getParentOfType<ModuleOp>();

    Value input = adaptor.getValue();
    unsigned width = op.getValue().getType().getIntOrFloatBitWidth();

    Value val;
    if (width < 64)
      val = LLVM::ZExtOp::create(rewriter, loc, i64Ty, input);
    else if (width == 64)
      val = input;
    else
      val = LLVM::TruncOp::create(rewriter, loc, i64Ty, input);

    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(width));

    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__circt_sim_fmt_bin",
                                          {i64Ty, i32Ty}, ptrTy);
    if (failed(funcOp))
      return failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                     ValueRange{val, widthVal});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.char to a runtime call.
struct FmtCharOpLowering : public OpConversionPattern<sim::FormatCharOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatCharOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();

    Value input = adaptor.getValue();
    unsigned width = op.getValue().getType().getIntOrFloatBitWidth();

    Value val;
    if (width < 64)
      val = LLVM::ZExtOp::create(rewriter, loc, i64Ty, input);
    else if (width == 64)
      val = input;
    else
      val = LLVM::TruncOp::create(rewriter, loc, i64Ty, input);

    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__circt_sim_fmt_char",
                                          {i64Ty}, ptrTy);
    if (failed(funcOp))
      return failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange{val});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.concat to an alloca + runtime call.
struct FmtConcatOpLowering
    : public OpConversionPattern<sim::FormatStringConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatStringConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();

    auto inputs = adaptor.getInputs();
    int32_t count = static_cast<int32_t>(inputs.size());

    if (count == 0) {
      // Empty concat → null pointer (empty string).
      Value null = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      rewriter.replaceOp(op, null);
      return success();
    }

    // Alloca an array of ptr for the parts.
    auto arrTy = LLVM::LLVMArrayType::get(ptrTy, count);
    Value one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(1));
    Value arr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrTy, one,
                                       /*alignment=*/8);

    // Store each part pointer.
    for (int32_t i = 0; i < count; ++i) {
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(i));
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrTy, arrTy, arr,
                                       ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, inputs[i], slot);
    }

    Value countVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(count));

    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "__circt_sim_fmt_concat",
                                          {ptrTy, i32Ty}, ptrTy);
    if (failed(funcOp))
      return failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                     ValueRange{arr, countVal});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower sim.fmt.dyn_string — the input is already an !llvm.ptr in the
/// converted type system (pointing to a {ptr, i64} string struct). For now,
/// just pass it through as the fstring pointer.
struct FmtDynStringOpLowering
    : public OpConversionPattern<sim::FormatDynStringOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatDynStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // The input is already converted to !llvm.ptr. Pass it through as-is.
    // The print runtime will need to handle this differently, but for now
    // this prevents compilation failures.
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

/// Lower hw.bitcast to bit-level reinterpretation in LLVM IR.
/// hw.bitcast reinterprets the bit pattern of a value as a different type
/// of the same width (e.g., i2 → struct<i1, i1>).
struct HWBitcastOpLowering : public OpConversionPattern<hw::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type srcLLVMTy = input.getType();
    Type dstLLVMTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstLLVMTy)
      return failure();

    // If the converted types are identical, just forward.
    if (srcLLVMTy == dstLLVMTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    Type hwSrcTy = op.getInput().getType();
    Type hwDstTy = op.getResult().getType();

    // Integer → Struct: decompose integer bits into struct fields.
    if (isa<IntegerType>(srcLLVMTy) && isa<LLVM::LLVMStructType>(dstLLVMTy)) {
      auto structTy = cast<LLVM::LLVMStructType>(dstLLVMTy);
      auto hwStructTy = cast<hw::StructType>(hwDstTy);
      auto fields = structTy.getBody();

      Value result = LLVM::UndefOp::create(rewriter, loc, dstLLVMTy);
      unsigned bitOffset = 0;

      // Iterate LLVM struct fields in order [0..N-1].
      // LLVM field i corresponds to HW field (N-1-i) due to endianness
      // reversal. HW fields are ordered MSB-first, so HW field (N-1-i) is the
      // (N-1-i)-th from the MSB, i.e., it occupies the lowest bits among the
      // fields processed so far. We iterate in LLVM order, which processes
      // fields from the LSB upward.
      for (size_t i = 0, e = fields.size(); i < e; ++i) {
        // The HW field index that maps to LLVM field i.
        uint32_t hwIdx =
            HWToLLVMEndianessConverter::convertToLLVMEndianess(hwStructTy, i);
        (void)hwIdx; // The endianness mapping is already baked into the type
                     // converter; we just need to extract fields in LLVM order.

        auto fieldTy = cast<IntegerType>(fields[i]);
        unsigned fieldWidth = fieldTy.getWidth();

        // Extract bits [bitOffset .. bitOffset+fieldWidth-1] from the integer.
        Value bits = input;
        if (bitOffset > 0) {
          auto shiftAmt = LLVM::ConstantOp::create(
              rewriter, loc, srcLLVMTy,
              rewriter.getIntegerAttr(srcLLVMTy, bitOffset));
          bits = LLVM::LShrOp::create(rewriter, loc, bits, shiftAmt);
        }
        if (fieldWidth < cast<IntegerType>(srcLLVMTy).getWidth())
          bits = LLVM::TruncOp::create(rewriter, loc, fieldTy, bits);

        result =
            LLVM::InsertValueOp::create(rewriter, loc, result, bits, i);
        bitOffset += fieldWidth;
      }

      rewriter.replaceOp(op, result);
      return success();
    }

    // Struct → Integer: compose struct fields into an integer.
    if (isa<LLVM::LLVMStructType>(srcLLVMTy) && isa<IntegerType>(dstLLVMTy)) {
      auto structTy = cast<LLVM::LLVMStructType>(srcLLVMTy);
      auto hwStructTy = cast<hw::StructType>(hwSrcTy);
      auto fields = structTy.getBody();
      auto intTy = cast<IntegerType>(dstLLVMTy);

      Value result = LLVM::ConstantOp::create(rewriter, loc, intTy,
                                               rewriter.getIntegerAttr(intTy, 0));
      unsigned bitOffset = 0;

      for (size_t i = 0, e = fields.size(); i < e; ++i) {
        uint32_t hwIdx =
            HWToLLVMEndianessConverter::convertToLLVMEndianess(hwStructTy, i);
        (void)hwIdx;

        auto fieldTy = cast<IntegerType>(fields[i]);
        unsigned fieldWidth = fieldTy.getWidth();

        Value field =
            LLVM::ExtractValueOp::create(rewriter, loc, input, i);
        Value extended = LLVM::ZExtOp::create(rewriter, loc, intTy, field);
        if (bitOffset > 0) {
          auto shiftAmt = LLVM::ConstantOp::create(
              rewriter, loc, intTy,
              rewriter.getIntegerAttr(intTy, bitOffset));
          extended = LLVM::ShlOp::create(rewriter, loc, extended, shiftAmt);
        }
        result = LLVM::OrOp::create(rewriter, loc, result, extended);
        bitOffset += fieldWidth;
      }

      rewriter.replaceOp(op, result);
      return success();
    }

    // Both integers of the same width (shouldn't reach here, but handle).
    if (isa<IntegerType>(srcLLVMTy) && isa<IntegerType>(dstLLVMTy)) {
      rewriter.replaceOp(op, input);
      return success();
    }

    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Helper: Clone referenced declarations (same as FullProcessJIT)
//===----------------------------------------------------------------------===//

/// Check if a func.func body contains only ops that can be lowered by the
/// AOT pipeline. Returns true if the function can be inlined into the
/// micro-module with its body intact.
static bool isFuncBodyCompilable(func::FuncOp funcOp) {
  if (funcOp.isExternal())
    return false;

  // Reject functions with non-LLVM or aggregate types in their signature.
  // Aggregate types (structs, arrays) passed by value crash LLVM's ABI
  // lowering during JIT codegen (cast<IntegerType> assertion failure).
  // Only scalar types (integers ≤64 bits, pointers) are safe.
  auto isScalarType = [](Type ty) -> bool {
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return intTy.getWidth() <= 64;
    if (isa<LLVM::LLVMPointerType>(ty))
      return true;
    if (isa<Float32Type, Float64Type>(ty))
      return true;
    if (isa<LLVM::LLVMVoidType>(ty))
      return true;
    return false;
  };
  for (auto argType : funcOp.getArgumentTypes()) {
    if (!isScalarType(argType))
      return false;
  }
  for (auto resType : funcOp.getResultTypes()) {
    if (!isScalarType(resType))
      return false;
  }

  bool compilable = true;
  funcOp.walk([&](Operation *op) {
    // Allow ONLY pure arith/cf/LLVM/func dialects — these have clean
    // conversion to LLVM with no cross-dialect type interference.
    // Everything else (scf, hw, comb, sim, unrealized_conversion_cast)
    // is EXCLUDED to avoid type converter corruption of LLVM ops.
    if (isa<arith::ArithDialect, cf::ControlFlowDialect,
            LLVM::LLVMDialect, func::FuncDialect>(op->getDialect()))
      return WalkResult::advance();
    // Reject anything else.
    LLVM_DEBUG(llvm::dbgs() << "[AOT] Func body not compilable: "
                            << op->getName() << " in @"
                            << funcOp.getName() << "\n");
    compilable = false;
    return WalkResult::interrupt();
  });
  return compilable;
}

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
        if (auto callee = callOp.getCallee()) {
          if (!clonedSymbols.contains(*callee) &&
              !microModule.lookupSymbol(*callee))
            needed.push_back(*callee);
        }
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
      if (!srcOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[AOT] Referenced symbol not found: " << name << "\n");
        continue;
      }

      OpBuilder builder(microModule.getContext());
      builder.setInsertionPointToEnd(microModule.getBody());

      if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(srcOp)) {
        auto cloned = builder.clone(*llvmFunc, mapping);
        if (auto clonedFunc = dyn_cast<LLVM::LLVMFuncOp>(cloned)) {
          if (!clonedFunc.getBody().empty())
            clonedFunc.getBody().getBlocks().clear();
          // External declarations must have 'external' or 'extern_weak'
          // linkage. Reset if we stripped the body from an 'internal' func.
          auto linkage = clonedFunc.getLinkage();
          if (linkage != LLVM::Linkage::External &&
              linkage != LLVM::Linkage::ExternWeak)
            clonedFunc.setLinkage(LLVM::Linkage::External);
        }
        changed = true;
      } else if (auto funcFunc = dyn_cast<func::FuncOp>(srcOp)) {
        // Clone as external declaration only (no body).
        // Cloning the full body and then stripping it triggers "operation
        // destroyed but still has uses" because Block::clear() destroys
        // ops in forward order, hitting use-before-def.
        auto cloned = func::FuncOp::create(
            builder, funcFunc.getLoc(), funcFunc.getSymName(),
            funcFunc.getFunctionType());
        cloned.setVisibility(funcFunc.getVisibility());
        changed = true;
      } else if (auto globalOp = dyn_cast<LLVM::GlobalOp>(srcOp)) {
        builder.clone(*globalOp, mapping);
        changed = true;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AOTProcessCompiler
//===----------------------------------------------------------------------===//

AOTProcessCompiler::AOTProcessCompiler(MLIRContext &ctx)
    : mlirContext(ctx) {}

AOTProcessCompiler::~AOTProcessCompiler() = default;

bool AOTProcessCompiler::isProcessCompilable(llhd::ProcessOp processOp,
                                             std::string *reason) {
  bool compilable = true;
  auto &processRegion = processOp.getBody();

  processOp.walk([&](Operation *op) {
    if (isa<moore::WaitEventOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Process not compilable: contains moore.wait_event\n");
      if (reason)
        *reason = "moore.wait_event";
      compilable = false;
      return WalkResult::interrupt();
    }
    if (isa<sim::SimForkOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Process not compilable: contains sim.fork\n");
      if (reason)
        *reason = "sim.fork";
      compilable = false;
      return WalkResult::interrupt();
    }
    // Reject sim dialect ops that lack AOT lowering patterns.
    if (op->getName().getDialectNamespace() == "sim" &&
        !isa<sim::PrintFormattedProcOp, sim::TerminateOp,
             sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
             sim::FormatBinOp, sim::FormatCharOp, sim::FormatStringConcatOp,
             sim::FormatDynStringOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Process not compilable: unsupported sim op "
                 << op->getName() << "\n");
      if (reason)
        *reason = op->getName().getStringRef().str();
      compilable = false;
      return WalkResult::interrupt();
    }
    // Check operands defined outside the process (external values that
    // will be cloned into the micro-module entry block).
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || processRegion.isAncestor(defOp->getParentRegion()))
        continue;
      if (defOp->getName().getDialectNamespace() == "sim" &&
          !isa<sim::PrintFormattedProcOp, sim::TerminateOp,
               sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
               sim::FormatBinOp, sim::FormatCharOp,
               sim::FormatStringConcatOp, sim::FormatDynStringOp>(defOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[AOT] Process not compilable: uses external sim op "
                   << defOp->getName() << "\n");
        if (reason)
          *reason = "external:" + defOp->getName().getStringRef().str();
        compilable = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return compilable;
}

bool AOTProcessCompiler::isRunToCompletion(llhd::ProcessOp processOp) {
  // Count llhd.wait ops in the process.
  unsigned waitCount = 0;
  llhd::WaitOp singleWait = nullptr;

  processOp.walk([&](llhd::WaitOp waitOp) {
    ++waitCount;
    singleWait = waitOp;
  });

  // Must have exactly one wait.
  if (waitCount != 1)
    return false;

  // The wait must NOT have a delay (signal-only sensitivity).
  if (singleWait.getDelay())
    return false;

  // The wait must have sensitivity — either explicit observed signals or
  // derived sensitivity from llhd.prb ops in the body (the common pattern
  // for `always @(*)` where observed is empty but body reads signals).
  if (singleWait.getObserved().empty()) {
    // Check for derived sensitivity: any llhd.prb in the process body.
    bool hasProbes = false;
    processOp.walk([&](llhd::ProbeOp) { hasProbes = true; });
    if (!hasProbes)
      return false;
  }

  // The wait's destination block must be reachable from a back-edge:
  // the body must eventually branch back to the wait's parent block,
  // forming a loop. We verify this by checking that the wait's parent
  // block is a successor of some block in the body (via cf.br/cf.cond_br).
  Block *waitBlock = singleWait->getBlock();
  Block *bodyBlock = singleWait.getDest();
  if (!waitBlock || !bodyBlock)
    return false;

  // Verify the body eventually branches back to the wait block (loop).
  bool hasBackEdge = false;
  for (Block &block : processOp.getBody()) {
    if (auto *terminator = block.getTerminator()) {
      for (Block *succ : terminator->getSuccessors()) {
        if (succ == waitBlock && &block != waitBlock) {
          hasBackEdge = true;
          break;
        }
      }
    }
    if (hasBackEdge)
      break;
  }

  if (!hasBackEdge) {
    LLVM_DEBUG(llvm::dbgs()
               << "[AOT] Process not run-to-completion: no back-edge to wait "
                  "block\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "[AOT] Process is run-to-completion candidate "
                          << "(1 wait, no delay, "
                          << singleWait.getObserved().size()
                          << " observed signals)\n");
  return true;
}

//===----------------------------------------------------------------------===//
// classifyProcess — 6-step process classification algorithm
//===----------------------------------------------------------------------===//

CallbackPlan AOTProcessCompiler::classifyProcess(
    llhd::ProcessOp processOp,
    const llvm::DenseMap<Value, SignalId> &valueToSignal) {
  CallbackPlan plan;

  // Step A: Fast coroutine filters — bail early, no CFG walking.
  {
    bool forceCoroutine = false;
    processOp.walk([&](Operation *op) {
      if (isa<sim::SimForkOp>(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[classify] → Coroutine: contains sim.fork\n");
        forceCoroutine = true;
        return WalkResult::interrupt();
      }
      if (isa<moore::WaitEventOp>(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[classify] → Coroutine: contains moore.wait_event\n");
        forceCoroutine = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (forceCoroutine) {
      plan.model = ExecModel::Coroutine;
      return plan;
    }
  }

  // Step B: Collect suspension points — count llhd.wait ops.
  llvm::SmallVector<llhd::WaitOp, 2> waits;
  processOp.walk([&](llhd::WaitOp w) { waits.push_back(w); });

  if (waits.empty()) {
    // No wait ops can still hide suspension-sensitive behavior behind calls
    // (for example UVM startup helpers reached from the process body). Keep
    // these on interpreter/coroutine semantics until process-call native
    // policy is unified with func.call/call_indirect dispatch guards.
    bool hasCallOps = false;
    processOp.walk([&](Operation *op) {
      if (isa<func::CallOp, func::CallIndirectOp, LLVM::CallOp>(op)) {
        hasCallOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasCallOps) {
      plan.model = ExecModel::Coroutine;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[classify] → Coroutine: no waits but contains call op(s)\n");
      return plan;
    }

    // Pure no-wait process with no calls → one-shot callback.
    plan.model = ExecModel::OneShotCallback;
    LLVM_DEBUG(llvm::dbgs()
               << "[classify] → OneShotCallback: no waits, no calls\n");
    return plan;
  }

  if (waits.size() > 1) {
    // Multiple waits → must be a coroutine.
    plan.model = ExecModel::Coroutine;
    LLVM_DEBUG(llvm::dbgs()
               << "[classify] → Coroutine: " << waits.size() << " waits\n");
    return plan;
  }

  // Exactly one wait.
  llhd::WaitOp W = waits[0];
  plan.wait = W;
  plan.waitBlock = W->getBlock();
  plan.resumeBlock = W.getDest();

  if (!plan.waitBlock || !plan.resumeBlock) {
    plan.model = ExecModel::Coroutine;
    LLVM_DEBUG(llvm::dbgs()
               << "[classify] → Coroutine: null wait/resume block\n");
    return plan;
  }

  // Step C: SCC sink verification — ensure callback resumption cannot enter
  // an infinite loop that bypasses the wait block.
  //
  // Any block reachable from resumeBlock that can reach waitBlock is safe.
  // For blocks that do not reach waitBlock, allow only finite acyclic tails
  // (e.g. wait delay -> terminate). Reject cycles in that non-reentering tail.
  {
    // Reverse reachability from waitBlock.
    llvm::DenseSet<Block *> reverseReachable;
    llvm::SmallVector<Block *, 8> worklist;
    reverseReachable.insert(plan.waitBlock);
    worklist.push_back(plan.waitBlock);
    while (!worklist.empty()) {
      Block *curr = worklist.pop_back_val();
      for (Block *pred : curr->getPredecessors()) {
        if (reverseReachable.insert(pred).second)
          worklist.push_back(pred);
      }
    }

    // Forward reachability from resumeBlock.
    llvm::DenseSet<Block *> forwardReachable;
    forwardReachable.insert(plan.resumeBlock);
    worklist.push_back(plan.resumeBlock);
    while (!worklist.empty()) {
      Block *curr = worklist.pop_back_val();
      if (auto *terminator = curr->getTerminator()) {
        for (Block *succ : terminator->getSuccessors()) {
          if (forwardReachable.insert(succ).second)
            worklist.push_back(succ);
        }
      }
    }

    // Fast path: all resume-reachable blocks can re-enter wait.
    bool allReachWait = true;
    for (Block *fwd : forwardReachable) {
      if (!reverseReachable.count(fwd)) {
        allReachWait = false;
        break;
      }
    }

    if (!allReachWait) {
      llvm::DenseSet<Block *> nonReenterTail;
      for (Block *fwd : forwardReachable)
        if (!reverseReachable.count(fwd))
          nonReenterTail.insert(fwd);

      // Reject only if the non-reentering tail has a cycle.
      llvm::DenseMap<Block *, uint8_t> color;
      std::function<bool(Block *)> hasTailCycle = [&](Block *block) -> bool {
        color[block] = 1; // visiting
        if (auto *terminator = block->getTerminator()) {
          for (Block *succ : terminator->getSuccessors()) {
            if (!nonReenterTail.count(succ))
              continue;
            uint8_t succColor = color.lookup(succ);
            if (succColor == 1)
              return true;
            if (succColor == 0 && hasTailCycle(succ))
              return true;
          }
        }
        color[block] = 2; // done
        return false;
      };

      for (Block *tailBlock : nonReenterTail) {
        if (color.lookup(tailBlock) != 0)
          continue;
        if (hasTailCycle(tailBlock)) {
          plan.model = ExecModel::Coroutine;
          LLVM_DEBUG(llvm::dbgs()
                     << "[classify] → Coroutine: infinite loop reachable from "
                        "resume without re-entering wait\n");
          return plan;
        }
      }
    }
  }

  // Step D: Init run detection.
  Block &entryBlock = processOp.getBody().front();
  plan.needsInitRun = (&entryBlock != plan.resumeBlock);
  LLVM_DEBUG(llvm::dbgs() << "[classify] needsInitRun="
                          << plan.needsInitRun << "\n");

  // Collect loop-carried state from wait.destOperands.
  for (Value destOp : W.getDestOperands()) {
    // Reject alloca-backed pointers in the frame: the alloca's lifetime
    // doesn't survive across callback returns.
    if (isa<LLVM::LLVMPointerType>(destOp.getType())) {
      if (auto *defOp = destOp.getDefiningOp()) {
        if (isa<LLVM::AllocaOp>(defOp)) {
          plan.model = ExecModel::Coroutine;
          LLVM_DEBUG(
              llvm::dbgs()
              << "[classify] → Coroutine: alloca ptr in destOperands\n");
          return plan;
        }
      }
    }
    plan.frameSlotTypes.push_back(destOp.getType());
  }

  // Step E: Wait kind classification.
  if (!W.getObserved().empty()) {
    // Has observed signals. Check if all are statically resolvable.
    bool allStatic = true;
    for (Value observed : W.getObserved()) {
      auto it = valueToSignal.find(observed);
      if (it == valueToSignal.end() || it->second == 0) {
        allStatic = false;
        break;
      }
    }

    if (allStatic) {
      plan.model = ExecModel::CallbackStaticObserved;
    } else {
      plan.model = ExecModel::CallbackDynamicWait;
    }
  } else if (W.getDelay()) {
    // Delay-only wait (no observed signals).
    plan.model = ExecModel::CallbackTimeOnly;
    // Extract constant delay if available.
    if (auto constOp = W.getDelay().getDefiningOp<llhd::ConstantTimeOp>()) {
      // Encode as ns + delta*epsilon. For now, extract just the integer value.
      if (auto timeAttr =
              llvm::dyn_cast<llhd::TimeAttr>(constOp.getValueAttr()))
        plan.delayValue = timeAttr.getTime();
    }
  } else {
    // No observed signals and no delay. Check for derived sensitivity from
    // llhd.prb ops in the body (the `always @(*)` pattern).
    bool hasProbes = false;
    processOp.walk([&](llhd::ProbeOp) { hasProbes = true; });
    if (hasProbes) {
      llvm::DenseSet<SignalId> selfDrivenSigs;
      processOp.walk([&](llhd::DriveOp driveOp) {
        auto it = valueToSignal.find(driveOp.getSignal());
        if (it != valueToSignal.end() && it->second != 0)
          selfDrivenSigs.insert(it->second);
      });

      // Derive signals from probes — treat as static if all resolvable.
      llvm::DenseSet<SignalId> derivedSigs;
      bool allStatic = true;
      bool hasPointerProbeSignal = false;
      processOp.walk([&](llhd::ProbeOp probeOp) {
        if (auto refTy = dyn_cast<llhd::RefType>(probeOp.getSignal().getType()))
          hasPointerProbeSignal |= isa<LLVM::LLVMPointerType>(
              refTy.getNestedType());
        auto it = valueToSignal.find(probeOp.getSignal());
        if (it != valueToSignal.end() && it->second != 0) {
          // Ignore probes of signals driven by this process. Static
          // sensitivity on self-driven probes can deadlock real dependencies
          // (e.g., string always @(*) lowered through module-level mirrors).
          if (!selfDrivenSigs.count(it->second))
            derivedSigs.insert(it->second);
        }
        else
          allStatic = false;
      });

      // Pointer-backed probes (e.g. interface handles lowered as !llvm.ptr)
      // should not be treated as static sensitivity sources: the pointer value
      // is stable while pointee fields change. Keep these in dynamic mode so
      // runtime wait sensitivity derivation can track the real dependencies.
      if (!hasPointerProbeSignal && allStatic && !derivedSigs.empty()) {
        plan.model = ExecModel::CallbackStaticObserved;
        // Populate staticSignals from derived probes.
        for (SignalId sigId : derivedSigs)
          plan.staticSignals.push_back({sigId, EdgeType::AnyEdge});
        LLVM_DEBUG(llvm::dbgs()
                   << "[classify] → CallbackStaticObserved (derived, "
                   << derivedSigs.size() << " signals)\n");
        return plan;
      } else {
        plan.model = ExecModel::CallbackDynamicWait;
        if (hasPointerProbeSignal) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[classify] → CallbackDynamicWait: pointer probe "
                        "derived sensitivity\n");
        }
      }
    } else {
      // No observed, no delay, no probes — conservative fallback.
      plan.model = ExecModel::Coroutine;
      LLVM_DEBUG(llvm::dbgs()
                 << "[classify] → Coroutine: no observed/delay/probes\n");
      return plan;
    }
  }

  // Step F: Static sensitivity extraction (for CallbackStaticObserved).
  if (plan.model == ExecModel::CallbackStaticObserved &&
      plan.staticSignals.empty()) {
    for (Value observed : W.getObserved()) {
      auto it = valueToSignal.find(observed);
      SignalId sigId = (it != valueToSignal.end()) ? it->second : 0;
      // TODO: extract edge kind from wait attributes (posedge/negedge).
      // For now, default to AnyEdge.
      plan.staticSignals.push_back({sigId, EdgeType::AnyEdge});
    }
  }

  LLVM_DEBUG({
    const char *modelStr = "?";
    switch (plan.model) {
    case ExecModel::CallbackStaticObserved:
      modelStr = "CallbackStaticObserved";
      break;
    case ExecModel::CallbackDynamicWait:
      modelStr = "CallbackDynamicWait";
      break;
    case ExecModel::CallbackTimeOnly:
      modelStr = "CallbackTimeOnly";
      break;
    case ExecModel::OneShotCallback:
      modelStr = "OneShotCallback";
      break;
    case ExecModel::Coroutine:
      modelStr = "Coroutine";
      break;
    }
    llvm::dbgs() << "[classify] → " << modelStr;
    if (plan.model == ExecModel::CallbackStaticObserved)
      llvm::dbgs() << " (" << plan.staticSignals.size() << " signals)";
    if (plan.model == ExecModel::CallbackTimeOnly)
      llvm::dbgs() << " (delay=" << plan.delayValue << ")";
    if (plan.hasFrame())
      llvm::dbgs() << " (frame: " << plan.frameSlotTypes.size() << " slots)";
    if (plan.needsInitRun)
      llvm::dbgs() << " (init-run)";
    llvm::dbgs() << "\n";
  });

  return plan;
}

bool AOTProcessCompiler::compileAllProcesses(
    const llvm::SmallVector<std::pair<ProcessId, llhd::ProcessOp>>
        &processes,
    const llvm::DenseMap<Value, SignalId> &valueToSignal,
    ModuleOp parentModule, llvm::SmallVector<AOTCompiledProcess> &results) {
#ifndef CIRCT_SIM_JIT_ENABLED
  LLVM_DEBUG(llvm::dbgs() << "[AOT] JIT not enabled\n");
  return false;
#else
  auto startTime = std::chrono::steady_clock::now();

  if (processes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[AOT] No processes to compile\n");
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "[AOT] Batch compiling " << processes.size()
                          << " processes\n");

  // Store function names as strings since func::FuncOp references become
  // invalid after applyPartialConversion converts them to LLVM::LLVMFuncOp.
  struct ExtractedFuncInfo {
    ProcessId procId;
    llhd::ProcessOp originalOp;
    std::string funcName;
    bool isCallback = false;
    CallbackPlan plan; // Full classification from classifyProcess()
    llvm::SmallVector<std::pair<SignalId, EdgeType>> waitSignals;
  };

  // === Phase A: Classify all processes and collect metadata ===
  // We separate collection from compilation so we can chunk the compilation.
  llvm::SmallVector<ExtractedFuncInfo, 4> allFuncInfos;
  for (auto [procId, processOp] : processes) {
    std::string rejReason;
    if (!isProcessCompilable(processOp, &rejReason)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Skipping non-compilable process " << procId << "\n");
      if (!rejReason.empty())
        rejectionStats[rejReason]++;
      continue;
    }

    std::string funcName = "__aot_process_" + std::to_string(procId);

    CallbackPlan plan = classifyProcess(processOp, valueToSignal);
    if (plan.isCallback() && plan.hasFrame()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Demoting " << procId
                 << " to Coroutine: has frame ("
                 << plan.frameSlotTypes.size() << " slots) — not yet supported\n");
      plan.model = ExecModel::Coroutine;
    }

    bool isCallback = plan.isCallback();
    ExtractedFuncInfo info;
    info.procId = procId;
    info.originalOp = processOp;
    info.funcName = funcName;
    info.isCallback = isCallback;
    info.plan = plan;

    // For callbacks, extract the sensitivity list from the wait op.
    if (isCallback) {
      llhd::WaitOp originalWait = nullptr;
      processOp.walk([&](llhd::WaitOp w) { originalWait = w; });
      assert(originalWait && "classifyProcess callback but no wait found");

      if (!plan.staticSignals.empty()) {
        info.waitSignals = plan.staticSignals;
      } else {
        if (!originalWait.getObserved().empty()) {
          for (Value observed : originalWait.getObserved()) {
            auto sigIt = valueToSignal.find(observed);
            SignalId sigId = (sigIt != valueToSignal.end()) ? sigIt->second : 0;
            info.waitSignals.push_back({sigId, EdgeType::AnyEdge});
          }
        } else {
          llvm::DenseSet<SignalId> derivedSigs;
          processOp.walk([&](llhd::ProbeOp probeOp) {
            auto sigIt = valueToSignal.find(probeOp.getSignal());
            if (sigIt != valueToSignal.end() && sigIt->second != 0)
              derivedSigs.insert(sigIt->second);
          });
          for (SignalId sigId : derivedSigs)
            info.waitSignals.push_back({sigId, EdgeType::AnyEdge});
        }
      }
    }

    allFuncInfos.push_back(std::move(info));
  }

  if (allFuncInfos.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[AOT] No compilable processes found\n");
    return true;
  }

  // Initialize native target once (required for JIT compilation).
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // === Phase B: Chunked compilation ===
  // Split into chunks of 64 processes to avoid massive single-module lowering.
  constexpr size_t kChunkSize = 64;
  size_t totalProcs = allFuncInfos.size();
  size_t numChunks = (totalProcs + kChunkSize - 1) / kChunkSize;

  llvm::errs() << "[AOT] Compiling " << totalProcs << " processes in "
               << numChunks << " chunk(s) of up to " << kChunkSize << "\n";

  auto ptrTy = LLVM::LLVMPointerType::get(&mlirContext);

  for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    auto chunkStart = std::chrono::steady_clock::now();

    size_t chunkBegin = chunkIdx * kChunkSize;
    size_t chunkEnd = std::min(chunkBegin + kChunkSize, totalProcs);
    size_t chunkSize = chunkEnd - chunkBegin;

    // Step B1: Create a fresh micro-module for this chunk.
    auto microModule = ModuleOp::create(UnknownLoc::get(&mlirContext));
    OpBuilder builder(&mlirContext);

    llvm::DenseMap<std::pair<ProcessId, Value>, SignalId> procSignalIdMap;
    llvm::SmallVector<ExtractedFuncInfo *, 64> chunkFuncs;

    // Step B2: Extract this chunk's processes into functions.
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      auto &info = allFuncInfos[i];
      auto procId = info.procId;
      auto processOp = info.originalOp;

      LLVM_DEBUG(llvm::dbgs() << "[AOT] Extracting " << info.funcName << "\n");

      llvm::SmallVector<Value, 8> signalValues;
      llvm::DenseMap<Value, SignalId> localSignalIdMap;
      llvm::SetVector<Value> externalValues;

      auto &processRegion = processOp.getBody();
      processOp.walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          auto it = valueToSignal.find(operand);
          if (it != valueToSignal.end() &&
              !localSignalIdMap.count(operand)) {
            signalValues.push_back(operand);
            localSignalIdMap[operand] = it->second;
            procSignalIdMap[{procId, operand}] = it->second;
            continue;
          }
          if (auto *defOp = operand.getDefiningOp()) {
            if (!processRegion.isAncestor(defOp->getParentRegion()) &&
                !localSignalIdMap.count(operand)) {
              externalValues.insert(operand);
            }
          }
        }
      });

      auto funcTy = FunctionType::get(&mlirContext, {}, {});
      auto funcOp =
          func::FuncOp::create(processOp.getLoc(), info.funcName, funcTy);
      microModule.push_back(funcOp);

      IRMapping mapping;
      auto &entryBlock = funcOp.getBody().emplaceBlock();
      {
        OpBuilder entryBuilder(&mlirContext);
        entryBuilder.setInsertionPointToStart(&entryBlock);
        auto i64Ty = entryBuilder.getI64Type();

        for (auto sigVal : signalValues) {
          auto sigId = localSignalIdMap[sigVal];
          auto constOp = LLVM::ConstantOp::create(
              entryBuilder, processOp.getLoc(), i64Ty,
              entryBuilder.getI64IntegerAttr(static_cast<int64_t>(sigId)));
          auto ptrOp = LLVM::IntToPtrOp::create(
              entryBuilder, processOp.getLoc(), ptrTy, constOp.getResult());
          mapping.map(sigVal, ptrOp.getResult());
        }

        for (Value extVal : externalValues) {
          auto *defOp = extVal.getDefiningOp();
          assert(defOp && "external value without defining op");
          entryBuilder.clone(*defOp, mapping);
        }
      }

      for (auto &block : processRegion) {
        Block *newBlock;
        if (&block == &processRegion.front()) {
          newBlock = &entryBlock;
        } else {
          newBlock = new Block();
          funcOp.getBody().push_back(newBlock);
        }
        mapping.map(&block, newBlock);

        if (&block != &processRegion.front()) {
          for (auto arg : block.getArguments()) {
            auto newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
            mapping.map(arg, newArg);
          }
        }
      }

      for (auto &block : processRegion) {
        auto *newBlock = mapping.lookup(&block);
        builder.setInsertionPointToEnd(newBlock);
        for (auto &op : block) {
          builder.clone(op, mapping);
        }
      }

      // For callbacks, transform the function to body-only execution.
      if (info.isCallback) {
        llhd::WaitOp clonedWait = nullptr;
        funcOp.walk([&](llhd::WaitOp w) { clonedWait = w; });

        if (clonedWait) {
          Block *waitBlock = clonedWait->getBlock();
          Block *bodyBlock = clonedWait.getDest();

          if (waitBlock == bodyBlock) {
            {
              OpBuilder waitBuilder(clonedWait);
              LLVM::ReturnOp::create(waitBuilder, clonedWait->getLoc(),
                                     ValueRange{});
              clonedWait.erase();
            }
          } else {
            for (Block &block : funcOp.getBody()) {
              auto *terminator = block.getTerminator();
              if (!terminator)
                continue;
              for (unsigned j = 0; j < terminator->getNumSuccessors(); ++j) {
                if (terminator->getSuccessor(j) == waitBlock &&
                    &block != waitBlock) {
                  OpBuilder retBuilder(terminator);
                  LLVM::ReturnOp::create(retBuilder, terminator->getLoc(),
                                         ValueRange{});
                  terminator->erase();
                  break;
                }
              }
            }

            {
              OpBuilder waitBuilder(clonedWait);
              LLVM::ReturnOp::create(waitBuilder, clonedWait->getLoc(),
                                     ValueRange{});
              clonedWait.erase();
            }

            for (Block &block : funcOp.getBody()) {
              auto *terminator = block.getTerminator();
              if (!terminator)
                continue;
              for (unsigned j = 0; j < terminator->getNumSuccessors(); ++j) {
                if (terminator->getSuccessor(j) == waitBlock) {
                  terminator->setSuccessor(bodyBlock, j);
                }
              }
            }
          }

          {
            Block &funcEntry = funcOp.getBody().front();
            llvm::SmallVector<Block *> toErase;
            for (Block &block : funcOp.getBody()) {
              if (&block == &funcEntry)
                continue;
              if (block.hasNoPredecessors())
                toErase.push_back(&block);
            }
            for (Block *block : toErase) {
              block->dropAllDefinedValueUses();
              block->erase();
            }
          }

          LLVM_DEBUG(llvm::dbgs()
                     << "[AOT] Transformed " << info.funcName
                     << " to callback (body-only, "
                     << info.waitSignals.size() << " signals)\n");
        }
      }

      chunkFuncs.push_back(&info);
    }

    // Step B3: Clone referenced declarations into the micro-module.
    IRMapping globalMapping;
    cloneReferencedDeclarations(microModule, parentModule, globalMapping);

    LLVM_DEBUG({
      llvm::dbgs() << "[AOT] Chunk " << chunkIdx + 1 << "/" << numChunks
                   << ": extracted " << chunkSize << " functions\n";
      microModule.dump();
    });

    // Step B4: Run lowering pipeline on this chunk's micro-module.
    LLVMTypeConverter converter(&mlirContext);
    addProcessTypeConversions(converter);
    populateHWToLLVMTypeConversions(converter);

    converter.addConversion([](sim::FormatStringType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    LLVMConversionTarget target(mlirContext);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<hw::HWDialect, comb::CombDialect,
                             llhd::LLHDDialect, scf::SCFDialect>();
    target.addIllegalOp<sim::PrintFormattedProcOp>();
    target.addIllegalOp<sim::TerminateOp>();
    target.addIllegalOp<sim::FormatLiteralOp, sim::FormatDecOp,
                        sim::FormatHexOp, sim::FormatBinOp,
                        sim::FormatCharOp, sim::FormatStringConcatOp,
                        sim::FormatDynStringOp>();

    RewritePatternSet patterns(&mlirContext);

    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateSCFToControlFlowConversionPatterns(patterns);

    Namespace globals;
    DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggMap;
    std::optional<HWToLLVMArraySpillCache> spillCache =
        HWToLLVMArraySpillCache();
    {
      OpBuilder spillBuilder(microModule);
      spillCache->spillNonHWOps(spillBuilder, converter, microModule);
    }
    populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                       constAggMap, spillCache);
    populateCombToArithConversionPatterns(converter, patterns);
    populateCombToLLVMConversionPatterns(converter, patterns);

    patterns.add<ProcessProbeOpLowering, ProcessDriveOpLowering,
                 ProcessConstantTimeLowering, ProcessIntToTimeLowering,
                 ProcessTimeToIntLowering, ProcessWaitOpLowering,
                 ProcessHaltOpLowering>(converter, &mlirContext);

    patterns.add<ProcessSigProjectionPassthrough<llhd::SigExtractOp>,
                 ProcessSigProjectionPassthrough<llhd::SigArrayGetOp>,
                 ProcessSigProjectionPassthrough<llhd::SigArraySliceOp>,
                 ProcessSigProjectionPassthrough<llhd::SigStructExtractOp>>(
        converter, &mlirContext);

    patterns.add<ProcessPrintOpLowering>(converter, &mlirContext);
    patterns.add<TerminateOpErasure>(converter, &mlirContext);
    patterns.add<FmtLiteralOpLowering, FmtDecOpLowering, FmtHexOpLowering,
                 FmtBinOpLowering, FmtCharOpLowering, FmtConcatOpLowering,
                 FmtDynStringOpLowering>(converter, &mlirContext);

    patterns.add<HWBitcastOpLowering>(converter, &mlirContext);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(microModule, target, std::move(patterns),
                                      config))) {
      llvm::errs() << "[AOT] Chunk " << chunkIdx + 1 << "/" << numChunks
                   << ": conversion FAILED — skipping chunk\n";
      LLVM_DEBUG(microModule.dump());
      microModule.erase();
      continue; // Skip this chunk, try the rest
    }

    // Clean up unrealized_conversion_cast ops.
    {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallVector<UnrealizedConversionCastOp> casts;
        microModule.walk([&](UnrealizedConversionCastOp op) {
          casts.push_back(op);
        });
        for (auto castOp : casts) {
          if (castOp.use_empty()) {
            castOp.erase();
            changed = true;
            continue;
          }
          if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
            auto inputCast = castOp.getOperand(0).getDefiningOp<
                UnrealizedConversionCastOp>();
            if (inputCast && inputCast.getNumOperands() == 1 &&
                inputCast.getNumResults() == 1) {
              if (castOp.getResult(0).getType() ==
                  inputCast.getOperand(0).getType()) {
                castOp.getResult(0).replaceAllUsesWith(
                    inputCast.getOperand(0));
                castOp.erase();
                changed = true;
              }
            }
          }
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[AOT] Chunk " << chunkIdx + 1 << " lowered LLVM IR:\n";
      microModule.dump();
    });

    // Step B5: Create ExecutionEngine for this chunk (O1, not O2).
    registerBuiltinDialectTranslation(*microModule.getContext());
    registerLLVMDialectTranslation(*microModule.getContext());

    ExecutionEngineOptions engineOpts;
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
    engineOpts.transformer = makeOptimizingTransformer(
        /*optLevel=*/1, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto engineOrErr = ExecutionEngine::create(microModule, engineOpts);
    if (!engineOrErr) {
      llvm::errs() << "[AOT] Chunk " << chunkIdx + 1 << "/" << numChunks
                   << ": ExecutionEngine creation FAILED — skipping chunk\n";
      LLVM_DEBUG(llvm::dbgs()
                 << llvm::toString(engineOrErr.takeError()) << "\n");
      microModule.erase();
      continue; // Skip this chunk, try the rest
    }

    engines.push_back(std::move(*engineOrErr));
    auto *jitEngine = engines.back().get();

    registerJITRuntimeSymbols(jitEngine);

    // Step B6: Lookup function pointers for this chunk.
    bool chunkLookupOk = true;
    for (auto *infoPtr : chunkFuncs) {
      auto &info = *infoPtr;
      auto expectedFn = jitEngine->lookup(info.funcName);
      if (!expectedFn) {
        llvm::errs() << "[AOT] Function lookup failed: " << info.funcName
                     << "\n";
        LLVM_DEBUG(llvm::dbgs()
                   << llvm::toString(expectedFn.takeError()) << "\n");
        chunkLookupOk = false;
        continue;
      }

      AOTCompiledProcess result;
      result.procId = info.procId;
      result.entryFunc =
          reinterpret_cast<UcontextProcessState::EntryFuncTy>(*expectedFn);
      result.funcName = info.funcName;
      result.isCallback = info.isCallback;
      result.execModel = info.plan.model;
      result.needsInitRun = info.plan.needsInitRun;
      result.waitSignals = std::move(info.waitSignals);
      results.push_back(std::move(result));
    }

    microModule.erase();

    auto chunkEndTime = std::chrono::steady_clock::now();
    double chunkMs = std::chrono::duration<double, std::milli>(
                         chunkEndTime - chunkStart)
                         .count();
    llvm::errs() << "[AOT] Chunk " << chunkIdx + 1 << "/" << numChunks << ": "
                 << chunkSize << " processes, "
                 << llvm::format("%.1f", chunkMs) << " ms"
                 << (chunkLookupOk ? "" : " (some lookups failed)") << "\n";
  }

  auto endTime = std::chrono::steady_clock::now();
  double compileMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  llvm::errs() << "[AOT] Total: compiled " << results.size() << " processes in "
               << llvm::format("%.1f", compileMs) << " ms\n";

  return !results.empty();
#endif
}

//===----------------------------------------------------------------------===//
// Manual arith/cf/func → LLVM lowering (avoids applyPartialConversion)
//===----------------------------------------------------------------------===//

/// Convert arith CmpI predicate to LLVM ICmp predicate.
static LLVM::ICmpPredicate convertCmpPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:  return LLVM::ICmpPredicate::eq;
  case arith::CmpIPredicate::ne:  return LLVM::ICmpPredicate::ne;
  case arith::CmpIPredicate::slt: return LLVM::ICmpPredicate::slt;
  case arith::CmpIPredicate::sle: return LLVM::ICmpPredicate::sle;
  case arith::CmpIPredicate::sgt: return LLVM::ICmpPredicate::sgt;
  case arith::CmpIPredicate::sge: return LLVM::ICmpPredicate::sge;
  case arith::CmpIPredicate::ult: return LLVM::ICmpPredicate::ult;
  case arith::CmpIPredicate::ule: return LLVM::ICmpPredicate::ule;
  case arith::CmpIPredicate::ugt: return LLVM::ICmpPredicate::ugt;
  case arith::CmpIPredicate::uge: return LLVM::ICmpPredicate::uge;
  }
  llvm_unreachable("unhandled arith::CmpIPredicate");
}

/// In-place lowering of arith/cf/func ops to LLVM dialect equivalents.
/// Returns true on success. This avoids applyPartialConversion which
/// corrupts existing LLVM ops by rebuilding function regions.
static bool lowerFuncArithCfToLLVM(ModuleOp microModule,
                                    MLIRContext &mlirContext) {
  IRRewriter rewriter(&mlirContext);
  bool hadError = false;

  // Phase 1: Rewrite arith/cf/func ops inside function bodies.
  // Collect first, then rewrite (walk + modify is unsafe).
  llvm::SmallVector<Operation *> toRewrite;
  microModule.walk([&](Operation *op) {
    auto *dialect = op->getDialect();
    if (dialect && (isa<arith::ArithDialect>(dialect) ||
                    isa<cf::ControlFlowDialect>(dialect)))
      toRewrite.push_back(op);
    else if (isa<func::ReturnOp, func::CallOp>(op))
      toRewrite.push_back(op);
  });

  for (auto *op : toRewrite) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();

    // --- arith ops ---
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto r = rewriter.create<LLVM::ConstantOp>(loc, c.getType(), c.getValue());
      c.replaceAllUsesWith(r.getResult());
      rewriter.eraseOp(c);
    } else if (auto o = dyn_cast<arith::AddIOp>(op)) {
      auto r = rewriter.create<LLVM::AddOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::SubIOp>(op)) {
      auto r = rewriter.create<LLVM::SubOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::MulIOp>(op)) {
      auto r = rewriter.create<LLVM::MulOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::DivSIOp>(op)) {
      auto r = rewriter.create<LLVM::SDivOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::DivUIOp>(op)) {
      auto r = rewriter.create<LLVM::UDivOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::RemSIOp>(op)) {
      auto r = rewriter.create<LLVM::SRemOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::RemUIOp>(op)) {
      auto r = rewriter.create<LLVM::URemOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::AndIOp>(op)) {
      auto r = rewriter.create<LLVM::AndOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::OrIOp>(op)) {
      auto r = rewriter.create<LLVM::OrOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::XOrIOp>(op)) {
      auto r = rewriter.create<LLVM::XOrOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShLIOp>(op)) {
      auto r = rewriter.create<LLVM::ShlOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShRSIOp>(op)) {
      auto r = rewriter.create<LLVM::AShrOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ShRUIOp>(op)) {
      auto r = rewriter.create<LLVM::LShrOp>(loc, o.getType(), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ExtSIOp>(op)) {
      auto r = rewriter.create<LLVM::SExtOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::ExtUIOp>(op)) {
      auto r = rewriter.create<LLVM::ZExtOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::TruncIOp>(op)) {
      auto r = rewriter.create<LLVM::TruncOp>(loc, o.getType(), o.getIn());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::CmpIOp>(op)) {
      auto r = rewriter.create<LLVM::ICmpOp>(
          loc, convertCmpPredicate(o.getPredicate()), o.getLhs(), o.getRhs());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::SelectOp>(op)) {
      auto r = rewriter.create<LLVM::SelectOp>(
          loc, o.getType(), o.getCondition(), o.getTrueValue(), o.getFalseValue());
      o.replaceAllUsesWith(r.getResult()); rewriter.eraseOp(o);
    } else if (auto o = dyn_cast<arith::IndexCastOp>(op)) {
      // index → integer or integer → index: these types should already be i64.
      o.replaceAllUsesWith(o.getIn()); rewriter.eraseOp(o);
    }
    // --- cf ops ---
    else if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
      rewriter.create<LLVM::BrOp>(loc, brOp.getDestOperands(), brOp.getDest());
      rewriter.eraseOp(brOp);
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(op)) {
      rewriter.create<LLVM::CondBrOp>(
          loc, condBrOp.getCondition(), condBrOp.getTrueDest(),
          condBrOp.getTrueDestOperands(), condBrOp.getFalseDest(),
          condBrOp.getFalseDestOperands());
      rewriter.eraseOp(condBrOp);
    }
    // --- func ops ---
    else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
      rewriter.create<LLVM::ReturnOp>(loc, retOp.getOperands());
      rewriter.eraseOp(retOp);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto newCall = rewriter.create<LLVM::CallOp>(
          loc, callOp.getResultTypes(), callOp.getCallee(),
          callOp.getOperands());
      callOp.replaceAllUsesWith(newCall.getResults());
      rewriter.eraseOp(callOp);
    }
    // Unhandled ops remain (will cause ExecutionEngine failure → graceful skip).
  }

  // Phase 2: Convert func.func → llvm.func by manual body inlining.
  llvm::SmallVector<func::FuncOp> funcOps;
  microModule.walk([&](func::FuncOp op) { funcOps.push_back(op); });

  for (auto funcOp : funcOps) {
    auto funcType = funcOp.getFunctionType();
    std::string origName = funcOp.getSymName().str();
    auto loc = funcOp.getLoc();

    // Build LLVM function type.
    SmallVector<Type> argTypes(funcType.getInputs());
    Type retType = funcType.getNumResults() == 0
                       ? LLVM::LLVMVoidType::get(&mlirContext)
                       : funcType.getResult(0);
    auto llvmFuncType = LLVM::LLVMFunctionType::get(retType, argTypes);

    if (funcOp.isExternal()) {
      // External: erase func.func first, then create llvm.func to avoid
      // symbol conflict.
      rewriter.setInsertionPoint(funcOp);
      rewriter.eraseOp(funcOp);
      rewriter.setInsertionPointToEnd(microModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(loc, origName, llvmFuncType);
      continue;
    }

    // For functions with bodies: splice blocks from func.func to llvm.func.
    // First extract the body, then erase func.func, then create llvm.func
    // with the extracted body. This avoids symbol name conflicts.
    rewriter.setInsertionPoint(funcOp);

    // Create temp llvm.func to receive the body.
    std::string tmpName = "__tmp_f1_" + origName;
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc, tmpName, llvmFuncType);

    // Splice all blocks from func.func into llvm.func.
    auto &srcRegion = funcOp.getBody();
    auto &dstRegion = llvmFunc.getBody();
    dstRegion.getBlocks().splice(dstRegion.end(), srcRegion.getBlocks());

    // funcOp's body is now empty. Safe to erase.
    rewriter.eraseOp(funcOp);

    // Rename to original name.
    llvmFunc.setSymName(origName);
  }

  return !hadError;
}

/// Post-lowering validation: walk each llvm.func with a body and verify all
/// ops and types are pure LLVM dialect. Remove functions that still contain
/// non-LLVM types (e.g., index, unrealized_conversion_cast residuals) to
/// prevent assertion failures in LLVM's JIT backend.
/// Returns the names of stripped functions.
static llvm::SmallVector<std::string>
stripNonLLVMFunctions(ModuleOp microModule) {
  llvm::SmallVector<std::string> stripped;
  llvm::SmallVector<LLVM::LLVMFuncOp> toStrip;

  microModule.walk([&](LLVM::LLVMFuncOp llvmFunc) {
    if (llvmFunc.isExternal())
      return;

    bool hasNonLLVM = false;
    llvmFunc.walk([&](Operation *op) {
      // Check op dialect — must be LLVM.
      if (op->getDialect() &&
          !isa<LLVM::LLVMDialect>(op->getDialect())) {
        hasNonLLVM = true;
        return WalkResult::interrupt();
      }
      // Check all operand and result types.
      for (auto type : op->getOperandTypes()) {
        if (!LLVM::isCompatibleType(type)) {
          hasNonLLVM = true;
          return WalkResult::interrupt();
        }
      }
      for (auto type : op->getResultTypes()) {
        if (!LLVM::isCompatibleType(type)) {
          hasNonLLVM = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (hasNonLLVM)
      toStrip.push_back(llvmFunc);
  });

  for (auto func : toStrip) {
    stripped.push_back(func.getSymName().str());
    // Strip to external declaration: clear body, ensure external linkage.
    func.getBody().getBlocks().clear();
    auto linkage = func.getLinkage();
    if (linkage != LLVM::Linkage::External &&
        linkage != LLVM::Linkage::ExternWeak)
      func.setLinkage(LLVM::Linkage::External);
  }

  return stripped;
}

//===----------------------------------------------------------------------===//
// compileAllFuncBodies — Phase F1: bulk function compilation
//===----------------------------------------------------------------------===//

bool AOTProcessCompiler::compileAllFuncBodies(
    ModuleOp parentModule, llvm::SmallVector<AOTCompiledFunc> &results) {
#ifndef CIRCT_SIM_JIT_ENABLED
  return false;
#else
  auto startTime = std::chrono::steady_clock::now();

  // Collect compilable func.func ops.
  llvm::SmallVector<func::FuncOp, 64> candidates;
  unsigned totalFuncs = 0, externalFuncs = 0, rejectedFuncs = 0;

  parentModule.walk([&](func::FuncOp funcOp) {
    ++totalFuncs;
    if (funcOp.isExternal()) {
      ++externalFuncs;
      return;
    }
    if (!isFuncBodyCompilable(funcOp)) {
      ++rejectedFuncs;
      funcOp.walk([&](Operation *op) {
        if (isa<arith::ArithDialect, cf::ControlFlowDialect, scf::SCFDialect,
                LLVM::LLVMDialect, func::FuncDialect>(op->getDialect()))
          return WalkResult::advance();
        if (isa<hw::HWDialect, comb::CombDialect>(op->getDialect()))
          return WalkResult::advance();
        if (isa<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                sim::FormatBinOp, sim::FormatCharOp, sim::FormatStringConcatOp,
                sim::FormatDynStringOp, sim::PrintFormattedProcOp,
                sim::TerminateOp>(op))
          return WalkResult::advance();
        if (isa<UnrealizedConversionCastOp>(op))
          return WalkResult::advance();
        funcRejectionStats[op->getName().getStringRef()]++;
        return WalkResult::interrupt();
      });
      return;
    }
    candidates.push_back(funcOp);
  });

  llvm::errs() << "[AOT-F1] Func bodies: " << totalFuncs << " total, "
               << externalFuncs << " external, " << rejectedFuncs
               << " rejected, " << candidates.size() << " compilable\n";

  if (candidates.empty())
    return false;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Allow limiting candidates via env var for debugging.
  size_t totalCandidates = candidates.size();
  if (auto *limitStr = std::getenv("CIRCT_SIM_AOT_FUNC_LIMIT")) {
    size_t limit = std::atol(limitStr);
    if (limit > 0 && limit < totalCandidates) {
      llvm::errs() << "[AOT-F1] Limiting to " << limit << " of "
                   << totalCandidates << " candidates\n";
      candidates.resize(limit);
      totalCandidates = limit;
    }
  }
  llvm::errs() << "[AOT-F1] Compiling " << totalCandidates
               << " func bodies (single engine)\n";

  // Single micro-module for ALL eligible functions.
  auto microModule = ModuleOp::create(UnknownLoc::get(&mlirContext));
  OpBuilder builder(&mlirContext);
  builder.setInsertionPointToEnd(microModule.getBody());
  IRMapping mapping;

  llvm::SmallVector<std::string> allFuncNames;
  allFuncNames.reserve(totalCandidates);
  for (auto funcOp : candidates) {
    builder.clone(*funcOp, mapping);
    allFuncNames.push_back(funcOp.getSymName().str());
  }

  // Clone referenced declarations ONCE for the whole module.
  cloneReferencedDeclarations(microModule, parentModule, mapping);

  // Manual in-place lowering: rewrite arith/cf/func ops to LLVM equivalents.
  // We avoid applyPartialConversion entirely because the conversion framework
  // rebuilds function regions during type conversion, destroying SSA values
  // that existing LLVM ops (GEP, load, store) still reference.
  if (!lowerFuncArithCfToLLVM(microModule, mlirContext)) {
    llvm::errs() << "[AOT-F1] Manual lowering FAILED — aborting\n";
    microModule.erase();
    return false;
  }

  // Post-lowering: strip functions that still contain non-LLVM ops/types.
  // These would cause assertion failures in LLVM's JIT backend.
  {
    auto stripped = stripNonLLVMFunctions(microModule);
    if (!stripped.empty()) {
      // Remove stripped functions from allFuncNames so we don't look them up.
      llvm::DenseSet<llvm::StringRef> strippedSet;
      for (const auto &name : stripped)
        strippedSet.insert(name);
      // Mark stripped names so lookup skips them.
      for (auto &name : allFuncNames) {
        if (strippedSet.contains(name))
          name.clear(); // Empty name → skip during lookup
      }
      llvm::errs() << "[AOT-F1] Stripped " << stripped.size()
                   << " functions with non-LLVM ops/types\n";
    }
  }

  // Verify the module before attempting ExecutionEngine creation.
  if (failed(mlir::verify(microModule))) {
    llvm::errs() << "[AOT-F1] Module verification FAILED — aborting\n";
    microModule.erase();
    return false;
  }

  // Collect external function/global names that need stub resolution.
  // These are functions declared (no body) in the micro-module — they're
  // interpreted functions that compiled code might call.
  llvm::SmallVector<std::string> externalDeclNames;
  {
    llvm::DenseSet<llvm::StringRef> compiledNames;
    for (const auto &name : allFuncNames) {
      if (!name.empty())
        compiledNames.insert(name);
    }
    microModule.walk([&](LLVM::LLVMFuncOp func) {
      if (func.isExternal() && !compiledNames.contains(func.getSymName()))
        externalDeclNames.push_back(func.getSymName().str());
    });
    microModule.walk([&](LLVM::GlobalOp global) {
      if (!global.getInitializerRegion().empty())
        return; // Has an initializer — not external
      if (!global.getValue())
        externalDeclNames.push_back(global.getSymName().str());
    });
  }

  // Create ONE ExecutionEngine with O1.
  registerBuiltinDialectTranslation(*microModule.getContext());
  registerLLVMDialectTranslation(*microModule.getContext());

  ExecutionEngineOptions engineOpts;
  engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  engineOpts.transformer = makeOptimizingTransformer(
      /*optLevel=*/1, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  auto engineOrErr = ExecutionEngine::create(microModule, engineOpts);
  if (!engineOrErr) {
    llvm::errs() << "[AOT-F1] ExecutionEngine creation FAILED — aborting\n";
    llvm::consumeError(engineOrErr.takeError());
    microModule.erase();
    return false;
  }

  engines.push_back(std::move(*engineOrErr));
  auto *jitEngine = engines.back().get();
  registerJITRuntimeSymbols(jitEngine);

  // Register stub symbols for all external declarations so the JIT can
  // resolve cross-references to interpreted functions. Without these stubs,
  // the JIT fails to materialize even functions that never actually call
  // the missing symbols at runtime (because materialization is eager).
  if (!externalDeclNames.empty()) {
    // Universal abort stub: if a compiled function actually calls an
    // interpreted function at runtime, this will be called.
    static auto abortStub = +[]() -> void {
      llvm::errs() << "[AOT-F1] FATAL: compiled code called unresolved "
                      "interpreter function\n";
      std::abort();
    };
    jitEngine->registerSymbols(
        [&](llvm::orc::MangleAndInterner interner) {
          llvm::orc::SymbolMap symbolMap;
          for (const auto &name : externalDeclNames) {
            symbolMap[interner(name)] = {
                llvm::orc::ExecutorAddr::fromPtr(
                    reinterpret_cast<void *>(abortStub)),
                llvm::JITSymbolFlags::Exported};
          }
          return symbolMap;
        });
    llvm::errs() << "[AOT-F1] Registered " << externalDeclNames.size()
                 << " stub symbols for external declarations\n";
  }

  // Lookup ALL function pointers from the single engine.
  unsigned lookupFailed = 0;
  for (size_t i = 0; i < allFuncNames.size(); ++i) {
    const auto &name = allFuncNames[i];
    if (name.empty())
      continue; // Stripped during post-lowering validation
    auto expectedFn = jitEngine->lookup(name);
    if (!expectedFn) {
      llvm::consumeError(expectedFn.takeError());
      ++lookupFailed;
      continue;
    }

    auto funcOp = candidates[i];
    AOTCompiledFunc result;
    result.funcName = name;
    result.funcPtr = reinterpret_cast<void *>(*expectedFn);
    result.numArgs = funcOp.getNumArguments();
    result.numResults = funcOp.getNumResults();
    results.push_back(std::move(result));
  }

  microModule.erase();

  auto endTime = std::chrono::steady_clock::now();
  double totalMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  llvm::errs() << "[AOT-F1] Compiled " << results.size() << "/"
               << totalCandidates << " eligible functions in "
               << llvm::format("%.1f", totalMs) << " ms (single engine)";
  if (lookupFailed > 0)
    llvm::errs() << " (" << lookupFailed << " lookup failures)";
  llvm::errs() << "\n";

  return !results.empty();
#endif
}

//===----------------------------------------------------------------------===//
// compileFunctions — Phase F1 packed-wrapper compilation
//===----------------------------------------------------------------------===//

/// Check if an LLVM type is a scalar type suitable for packed wrapper slots.
/// Only integers (<=64 bits) and pointers are supported in v1.
static bool isPackedScalarType(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth() <= 64;
  if (isa<LLVM::LLVMPointerType>(ty))
    return true;
  return false;
}

/// Generate packed wrapper functions in the micro-module for all eligible
/// LLVM::LLVMFuncOp entries. Each wrapper has signature:
///   void __packed_<name>(ptr %args, ptr %results)
/// where args/results are arrays of 8-byte (i64-sized) slots.
///
/// Returns the number of wrappers generated. The wrapperNames output maps
/// original function name -> wrapper function name.
static unsigned generatePackedWrappers(
    ModuleOp microModule, ArrayRef<std::string> funcNames,
    llvm::SmallVector<std::pair<std::string, std::string>> &wrapperNames) {

  auto *ctx = microModule.getContext();
  OpBuilder builder(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  unsigned generated = 0;

  for (const auto &name : funcNames) {
    auto *symbol = microModule.lookupSymbol(name);
    auto llvmFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(symbol);
    if (!llvmFunc)
      continue;

    auto funcType = llvmFunc.getFunctionType();

    // Check all params are scalar.
    bool eligible = true;
    for (unsigned i = 0; i < funcType.getNumParams(); ++i) {
      if (!isPackedScalarType(funcType.getParamType(i))) {
        eligible = false;
        break;
      }
    }

    // Check return type is void or scalar.
    Type retTy = funcType.getReturnType();
    bool isVoid = isa<LLVM::LLVMVoidType>(retTy);
    if (!isVoid && !isPackedScalarType(retTy))
      eligible = false;

    if (!eligible)
      continue;

    std::string wrapperName = "__packed_" + name;

    // Create: void @__packed_<name>(ptr %args, ptr %results)
    auto wrapperFuncTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy});
    builder.setInsertionPointToEnd(microModule.getBody());
    auto wrapperFunc =
        LLVM::LLVMFuncOp::create(builder, llvmFunc.getLoc(), wrapperName,
                                  wrapperFuncTy);

    auto *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value argsPtr = wrapperFunc.getArgument(0);
    Value resultsPtr = wrapperFunc.getArgument(1);

    // Load arguments from 8-byte slots.
    SmallVector<Value> callArgs;
    for (unsigned i = 0; i < funcType.getNumParams(); ++i) {
      Type argTy = funcType.getParamType(i);

      // GEP: &args[i] (element type = i64, each slot = 8 bytes)
      auto slotPtr = LLVM::GEPOp::create(
          builder, llvmFunc.getLoc(), ptrTy, i64Ty, argsPtr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});

      if (isa<LLVM::LLVMPointerType>(argTy)) {
        // Load as ptr directly from the 8-byte slot.
        auto loaded =
            LLVM::LoadOp::create(builder, llvmFunc.getLoc(), ptrTy, slotPtr);
        callArgs.push_back(loaded);
      } else {
        // Load as i64, then trunc to the actual integer type if needed.
        auto loaded =
            LLVM::LoadOp::create(builder, llvmFunc.getLoc(), i64Ty, slotPtr);
        Value arg = loaded;
        if (argTy != i64Ty)
          arg = LLVM::TruncOp::create(builder, llvmFunc.getLoc(), argTy, arg);
        callArgs.push_back(arg);
      }
    }

    // Call the original function.
    if (isVoid) {
      LLVM::CallOp::create(builder, llvmFunc.getLoc(), llvmFunc, callArgs);
    } else {
      auto callResult =
          LLVM::CallOp::create(builder, llvmFunc.getLoc(), llvmFunc, callArgs);
      Value result = callResult.getResult();

      // Store result into results[0].
      auto resultSlotPtr = LLVM::GEPOp::create(
          builder, llvmFunc.getLoc(), ptrTy, i64Ty, resultsPtr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(0)});

      if (isa<LLVM::LLVMPointerType>(retTy)) {
        LLVM::StoreOp::create(builder, llvmFunc.getLoc(), result,
                               resultSlotPtr);
      } else {
        // Zero-extend to i64 then store into the 8-byte slot.
        Value toStore = result;
        if (retTy != i64Ty)
          toStore =
              LLVM::ZExtOp::create(builder, llvmFunc.getLoc(), i64Ty, result);
        LLVM::StoreOp::create(builder, llvmFunc.getLoc(), toStore,
                               resultSlotPtr);
      }
    }

    LLVM::ReturnOp::create(builder, llvmFunc.getLoc(), ValueRange{});

    wrapperNames.push_back({name, wrapperName});
    ++generated;
  }

  return generated;
}

LogicalResult AOTProcessCompiler::compileFunctions(
    ModuleOp parentModule,
    llvm::DenseMap<llvm::StringRef, void *> &compiled) {
#ifndef CIRCT_SIM_JIT_ENABLED
  return failure();
#else
  auto startTime = std::chrono::steady_clock::now();

  // Collect compilable func.func ops.
  llvm::SmallVector<func::FuncOp, 64> candidates;
  unsigned totalFuncs = 0, externalFuncs = 0, rejectedFuncs = 0;

  parentModule.walk([&](func::FuncOp funcOp) {
    ++totalFuncs;
    if (funcOp.isExternal()) {
      ++externalFuncs;
      return;
    }
    if (!isFuncBodyCompilable(funcOp)) {
      ++rejectedFuncs;
      funcOp.walk([&](Operation *op) {
        if (isa<arith::ArithDialect, cf::ControlFlowDialect, scf::SCFDialect,
                LLVM::LLVMDialect, func::FuncDialect>(op->getDialect()))
          return WalkResult::advance();
        if (isa<hw::HWDialect, comb::CombDialect>(op->getDialect()))
          return WalkResult::advance();
        if (isa<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                sim::FormatBinOp, sim::FormatCharOp, sim::FormatStringConcatOp,
                sim::FormatDynStringOp, sim::PrintFormattedProcOp,
                sim::TerminateOp>(op))
          return WalkResult::advance();
        if (isa<UnrealizedConversionCastOp>(op))
          return WalkResult::advance();
        funcRejectionStats[op->getName().getStringRef()]++;
        return WalkResult::interrupt();
      });
      return;
    }
    candidates.push_back(funcOp);
  });

  llvm::errs() << "[AOT-F1-packed] Func bodies: " << totalFuncs << " total, "
               << externalFuncs << " external, " << rejectedFuncs
               << " rejected, " << candidates.size() << " compilable\n";

  if (candidates.empty())
    return failure();

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  constexpr size_t kChunkSize = 64;
  size_t numChunks = (candidates.size() + kChunkSize - 1) / kChunkSize;

  llvm::errs() << "[AOT-F1-packed] Compiling " << candidates.size()
               << " func bodies in " << numChunks << " chunk(s)\n";

  for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    auto chunkStart = std::chrono::steady_clock::now();

    size_t chunkBegin = chunkIdx * kChunkSize;
    size_t chunkEnd = std::min(chunkBegin + kChunkSize, candidates.size());
    size_t chunkSize = chunkEnd - chunkBegin;

    auto microModule = ModuleOp::create(UnknownLoc::get(&mlirContext));
    OpBuilder builder(&mlirContext);
    builder.setInsertionPointToEnd(microModule.getBody());
    IRMapping mapping;

    llvm::SmallVector<std::string, 64> chunkFuncNames;
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      auto funcOp = candidates[i];
      builder.clone(*funcOp, mapping);
      chunkFuncNames.push_back(funcOp.getSymName().str());
    }

    cloneReferencedDeclarations(microModule, parentModule, mapping);

    // Manual in-place lowering (same as compileAllFuncBodies).
    if (!lowerFuncArithCfToLLVM(microModule, mlirContext)) {
      llvm::errs() << "[AOT-F1-packed] Chunk " << chunkIdx + 1 << "/"
                   << numChunks << ": manual lowering FAILED\n";
      microModule.erase();
      continue;
    }

    // === Generate packed wrappers (AFTER lowering, BEFORE engine) ===
    llvm::SmallVector<std::pair<std::string, std::string>> wrapperNames;
    unsigned numWrappers =
        generatePackedWrappers(microModule, chunkFuncNames, wrapperNames);

    LLVM_DEBUG(llvm::dbgs() << "[AOT-F1-packed] Chunk " << chunkIdx + 1
                            << ": generated " << numWrappers
                            << " packed wrappers\n");

    // Create ExecutionEngine with O1.
    registerBuiltinDialectTranslation(*microModule.getContext());
    registerLLVMDialectTranslation(*microModule.getContext());

    ExecutionEngineOptions engineOpts;
    engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
    engineOpts.transformer = makeOptimizingTransformer(
        /*optLevel=*/1, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto engineOrErr = ExecutionEngine::create(microModule, engineOpts);
    if (!engineOrErr) {
      llvm::errs() << "[AOT-F1-packed] Chunk " << chunkIdx + 1 << "/"
                   << numChunks << ": ExecutionEngine creation FAILED\n";
      llvm::consumeError(engineOrErr.takeError());
      microModule.erase();
      continue;
    }

    engines.push_back(std::move(*engineOrErr));
    auto *jitEngine = engines.back().get();
    registerJITRuntimeSymbols(jitEngine);

    // Lookup packed wrapper pointers, keyed by ORIGINAL function name.
    unsigned chunkCompiled = 0;
    for (auto &[origName, wrapperName] : wrapperNames) {
      auto expectedFn = jitEngine->lookup(wrapperName);
      if (!expectedFn) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[AOT-F1-packed] Lookup failed: " << wrapperName << "\n");
        llvm::consumeError(expectedFn.takeError());
        continue;
      }

      // Key: look up the original FuncOp's name from the parent module.
      // The StringRef from the parent module's symbol table is stable.
      auto *parentSymbol =
          mlir::SymbolTable::lookupSymbolIn(parentModule, origName);
      if (!parentSymbol)
        continue;

      auto parentFuncOp = dyn_cast<func::FuncOp>(parentSymbol);
      if (!parentFuncOp)
        continue;

      // Use the parent module's StringRef (stable lifetime).
      compiled[parentFuncOp.getSymName()] =
          reinterpret_cast<void *>(*expectedFn);
      ++chunkCompiled;
    }

    microModule.erase();

    auto chunkEndTime = std::chrono::steady_clock::now();
    double chunkMs = std::chrono::duration<double, std::milli>(
                         chunkEndTime - chunkStart)
                         .count();
    llvm::errs() << "[AOT-F1-packed] Chunk " << chunkIdx + 1 << "/"
                 << numChunks << ": " << chunkCompiled << "/" << chunkSize
                 << " compiled (" << numWrappers << " wrappers), "
                 << llvm::format("%.1f", chunkMs) << " ms\n";
  }

  auto endTime = std::chrono::steady_clock::now();
  double totalMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  llvm::errs() << "[AOT-F1-packed] Total: " << compiled.size()
               << " packed funcs in " << llvm::format("%.1f", totalMs)
               << " ms\n";

  return compiled.empty() ? failure() : success();
#endif
}
