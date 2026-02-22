//===- AOTProcessCompiler.cpp - AOT batch process compilation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AOTProcessCompiler.h"
#include "JITSchedulerRuntime.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

#ifdef CIRCT_SIM_JIT_ENABLED
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#endif

#define DEBUG_TYPE "aot-process-jit"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

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
  bool compilable = true;
  funcOp.walk([&](Operation *op) {
    // Allow standard dialects that have conversion patterns.
    if (isa<arith::ArithDialect, cf::ControlFlowDialect, scf::SCFDialect,
            LLVM::LLVMDialect, func::FuncDialect>(op->getDialect()))
      return WalkResult::advance();
    if (isa<hw::HWDialect, comb::CombDialect>(op->getDialect()))
      return WalkResult::advance();
    // Allow supported sim ops.
    if (isa<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
            sim::FormatBinOp, sim::FormatCharOp, sim::FormatStringConcatOp,
            sim::FormatDynStringOp, sim::PrintFormattedProcOp,
            sim::TerminateOp>(op))
      return WalkResult::advance();
    // Allow unrealized_conversion_cast (used in type conversion).
    if (isa<UnrealizedConversionCastOp>(op))
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
        }
        changed = true;
      } else if (auto funcFunc = dyn_cast<func::FuncOp>(srcOp)) {
        auto cloned = cast<func::FuncOp>(builder.clone(*funcFunc, mapping));
        // Keep the body if it's compilable; otherwise strip to declaration.
        if (!cloned.getBody().empty() && !isFuncBodyCompilable(funcFunc)) {
          cloned.getBody().getBlocks().clear();
        }
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

bool AOTProcessCompiler::isProcessCompilable(llhd::ProcessOp processOp) {
  bool compilable = true;
  auto &processRegion = processOp.getBody();

  processOp.walk([&](Operation *op) {
    if (isa<moore::WaitEventOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Process not compilable: contains moore.wait_event\n");
      compilable = false;
      return WalkResult::interrupt();
    }
    if (isa<sim::SimForkOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Process not compilable: contains sim.fork\n");
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
    // No wait ops → one-shot callback (seq.initial, llhd.combinational).
    plan.model = ExecModel::OneShotCallback;
    LLVM_DEBUG(llvm::dbgs() << "[classify] → OneShotCallback: no waits\n");
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

  // Step C: SCC sink verification — verify that waitBlock is reachable from
  // all paths starting at resumeBlock. Concretely: every block reachable from
  // resumeBlock must have a path to waitBlock.
  //
  // We verify a simpler sufficient condition: there exists at least one
  // back-edge from a body block to waitBlock, AND there are no infinite loops
  // (blocks unreachable from waitBlock in the reverse CFG).
  {
    // First check: at least one back-edge to waitBlock exists.
    bool hasBackEdge = false;
    for (Block &block : processOp.getBody()) {
      if (&block == plan.waitBlock)
        continue;
      if (auto *terminator = block.getTerminator()) {
        for (Block *succ : terminator->getSuccessors()) {
          if (succ == plan.waitBlock) {
            hasBackEdge = true;
            break;
          }
        }
      }
      if (hasBackEdge)
        break;
    }

    if (!hasBackEdge) {
      plan.model = ExecModel::Coroutine;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[classify] → Coroutine: no back-edge to wait block\n");
      return plan;
    }

    // Second check: verify no infinite loops that bypass the wait.
    // Do a reverse reachability analysis from waitBlock: all blocks reachable
    // from resumeBlock must be reverse-reachable from waitBlock.
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

    // Every block forward-reachable from resumeBlock must be reverse-reachable
    // from waitBlock. If not, there's a path that never reaches the wait.
    for (Block *fwd : forwardReachable) {
      if (!reverseReachable.count(fwd)) {
        plan.model = ExecModel::Coroutine;
        LLVM_DEBUG(
            llvm::dbgs()
            << "[classify] → Coroutine: block reachable from resume but "
               "not reaching wait (possible infinite loop)\n");
        return plan;
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
      // Derive signals from probes — treat as static if all resolvable.
      llvm::DenseSet<SignalId> derivedSigs;
      bool allStatic = true;
      processOp.walk([&](llhd::ProbeOp probeOp) {
        auto it = valueToSignal.find(probeOp.getSignal());
        if (it != valueToSignal.end() && it->second != 0)
          derivedSigs.insert(it->second);
        else
          allStatic = false;
      });

      if (allStatic && !derivedSigs.empty()) {
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

  // Create ONE micro-module for all processes
  auto microModule = ModuleOp::create(UnknownLoc::get(&mlirContext));
  OpBuilder builder(&mlirContext);

  // Process data collected across all processes
  llvm::DenseMap<std::pair<ProcessId, Value>, SignalId> procSignalIdMap;
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
  llvm::SmallVector<ExtractedFuncInfo, 4> extractedFuncs;

  auto ptrTy = LLVM::LLVMPointerType::get(&mlirContext);

  // === Step 1: Extract all processes into functions in the combined module ===
  for (auto [procId, processOp] : processes) {
    if (!isProcessCompilable(processOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Skipping non-compilable process " << procId << "\n");
      continue;
    }

    std::string funcName = "__aot_process_" + std::to_string(procId);
    LLVM_DEBUG(llvm::dbgs() << "[AOT] Extracting " << funcName << "\n");

    // Collect signal values and external values used by this process.
    // External values are those defined OUTSIDE the process region
    // (e.g. hw.constant, llhd.constant_time in the hw.module body).
    llvm::SmallVector<Value, 8> signalValues;
    llvm::DenseMap<Value, SignalId> localSignalIdMap;
    llvm::SetVector<Value> externalValues;

    auto &processRegion = processOp.getBody();
    processOp.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        // Check if this is a signal reference
        auto it = valueToSignal.find(operand);
        if (it != valueToSignal.end() &&
            !localSignalIdMap.count(operand)) {
          signalValues.push_back(operand);
          localSignalIdMap[operand] = it->second;
          procSignalIdMap[{procId, operand}] = it->second;
          continue;
        }
        // Check if defined outside the process region
        if (auto *defOp = operand.getDefiningOp()) {
          if (!processRegion.isAncestor(defOp->getParentRegion()) &&
              !localSignalIdMap.count(operand)) {
            externalValues.insert(operand);
          }
        }
      }
    });

    // Create function: void @__aot_process_<procId>()
    auto funcTy = FunctionType::get(&mlirContext, {}, {});
    auto funcOp = func::FuncOp::create(processOp.getLoc(), funcName, funcTy);
    microModule.push_back(funcOp);

    // Clone process body with signal ID constants baked and external values
    // materialized in the function entry block.
    IRMapping mapping;

    auto &entryBlock = funcOp.getBody().emplaceBlock();
    {
      OpBuilder entryBuilder(&mlirContext);
      entryBuilder.setInsertionPointToStart(&entryBlock);
      auto i64Ty = entryBuilder.getI64Type();

      // Map signal values → inttoptr(signalId)
      for (auto sigVal : signalValues) {
        auto sigId = localSignalIdMap[sigVal];
        auto constOp = LLVM::ConstantOp::create(
            entryBuilder, processOp.getLoc(), i64Ty,
            entryBuilder.getI64IntegerAttr(static_cast<int64_t>(sigId)));
        auto ptrOp = LLVM::IntToPtrOp::create(entryBuilder, processOp.getLoc(),
                                             ptrTy, constOp.getResult());
        mapping.map(sigVal, ptrOp.getResult());
      }

      // Clone external value definitions into entry block.
      // We iterate in topological order: if an external value's defining op
      // uses other external values, those were already inserted earlier
      // (since SetVector preserves insertion order, and operands are visited
      // before their users during the walk).
      for (Value extVal : externalValues) {
        auto *defOp = extVal.getDefiningOp();
        assert(defOp && "external value without defining op");
        entryBuilder.clone(*defOp, mapping);
      }
    }

    // Clone blocks from process body
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

    // Clone ops
    for (auto &block : processRegion) {
      auto *newBlock = mapping.lookup(&block);
      builder.setInsertionPointToEnd(newBlock);
      for (auto &op : block) {
        builder.clone(op, mapping);
      }
    }

    // Classify: run-to-completion callback vs coroutine.
    // Use both old isRunToCompletion (for backward compat) and new
    // classifyProcess (for the richer ExecModel).
    CallbackPlan plan = classifyProcess(processOp, valueToSignal);

    // Frame support not yet implemented: if the process has loop-carried
    // state (destOperands on the wait), fall back to coroutine. This is
    // safe and correct; frame-based callbacks will be enabled in Phase E2.
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

    // For callbacks, extract the sensitivity list from the wait op and
    // transform the function to be body-only (skip wait, return instead).
    if (isCallback) {
      // Find the single wait op in the original process to extract signals.
      llhd::WaitOp originalWait = nullptr;
      processOp.walk([&](llhd::WaitOp w) { originalWait = w; });
      assert(originalWait && "classifyProcess callback but no wait found");

      // Use the sensitivity list already extracted by classifyProcess.
      if (!plan.staticSignals.empty()) {
        info.waitSignals = plan.staticSignals;
      } else {
        // Fallback: extract from the wait op directly.
        if (!originalWait.getObserved().empty()) {
          for (Value observed : originalWait.getObserved()) {
            auto sigIt = valueToSignal.find(observed);
            SignalId sigId = (sigIt != valueToSignal.end()) ? sigIt->second : 0;
            info.waitSignals.push_back({sigId, EdgeType::AnyEdge});
          }
        } else {
          // Derived sensitivity: scan the body for llhd.prb ops.
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

      // Transform the extracted function: find the cloned wait and the
      // blocks to rewire for body-only execution.
      llhd::WaitOp clonedWait = nullptr;
      funcOp.walk([&](llhd::WaitOp w) { clonedWait = w; });

      if (clonedWait) {
        Block *waitBlock = clonedWait->getBlock();
        Block *bodyBlock = clonedWait.getDest();

        if (waitBlock == bodyBlock) {
          // Simple case: wait and body are in the same block.
          // Structure: ^body: <ops...> llhd.wait ^body
          // Transform: replace wait with return, keep entry→body branch.
          {
            OpBuilder waitBuilder(clonedWait);
            LLVM::ReturnOp::create(waitBuilder, clonedWait->getLoc(),
                                   ValueRange{});
            clonedWait.erase();
          }
        } else {
          // Complex case: wait and body are in different blocks.
          // Structure: ^wait: llhd.wait ^body; ^body: <ops...> cf.br ^wait
          // Transform:
          //   1. Replace body→wait back-edges with return
          //   2. Replace wait op with return (makes wait block a dead end)
          //   3. Redirect entry→wait branches to entry→body

          // Step 1: Cut back-edges from body blocks to wait block.
          for (Block &block : funcOp.getBody()) {
            auto *terminator = block.getTerminator();
            if (!terminator)
              continue;
            for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
              if (terminator->getSuccessor(i) == waitBlock &&
                  &block != waitBlock) {
                OpBuilder retBuilder(terminator);
                LLVM::ReturnOp::create(retBuilder, terminator->getLoc(),
                                       ValueRange{});
                terminator->erase();
                break;
              }
            }
          }

          // Step 2: Replace wait op with return.
          {
            OpBuilder waitBuilder(clonedWait);
            LLVM::ReturnOp::create(waitBuilder, clonedWait->getLoc(),
                                   ValueRange{});
            clonedWait.erase();
          }

          // Step 3: Redirect entry→wait to entry→body.
          for (Block &block : funcOp.getBody()) {
            auto *terminator = block.getTerminator();
            if (!terminator)
              continue;
            for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
              if (terminator->getSuccessor(i) == waitBlock) {
                terminator->setSuccessor(bodyBlock, i);
              }
            }
          }
        }

        // Erase unreachable blocks (blocks with no predecessors except entry).
        {
          Block &entryBlock = funcOp.getBody().front();
          llvm::SmallVector<Block *> toErase;
          for (Block &block : funcOp.getBody()) {
            if (&block == &entryBlock)
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
                   << "[AOT] Transformed " << funcName
                   << " to callback (body-only, "
                   << info.waitSignals.size() << " signals)\n");
      }
    }

    extractedFuncs.push_back(std::move(info));
  }

  if (extractedFuncs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[AOT] No compilable processes extracted\n");
    microModule.erase();
    return true;
  }

  // Clone referenced declarations
  IRMapping globalMapping; // Empty for now; can be extended if needed
  cloneReferencedDeclarations(microModule, parentModule, globalMapping);

  LLVM_DEBUG({
    llvm::dbgs() << "[AOT] Extracted module with " << extractedFuncs.size()
                 << " functions:\n";
    microModule.dump();
  });

  // === Step 2: Run lowering pipeline ONCE ===
  LLVMTypeConverter converter(&mlirContext);
  addProcessTypeConversions(converter);
  populateHWToLLVMTypeConversions(converter);

  // sim.fstring → !llvm.ptr
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
  populateHWToLLVMConversionPatterns(converter, patterns, globals, constAggMap,
                                     spillCache);
  populateCombToArithConversionPatterns(converter, patterns);
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Process LLHD patterns
  patterns.add<ProcessProbeOpLowering, ProcessDriveOpLowering,
               ProcessConstantTimeLowering, ProcessIntToTimeLowering,
               ProcessTimeToIntLowering, ProcessWaitOpLowering,
               ProcessHaltOpLowering>(converter, &mlirContext);

  // Signal projection passthroughs
  patterns.add<ProcessSigProjectionPassthrough<llhd::SigExtractOp>,
               ProcessSigProjectionPassthrough<llhd::SigArrayGetOp>,
               ProcessSigProjectionPassthrough<llhd::SigArraySliceOp>,
               ProcessSigProjectionPassthrough<llhd::SigStructExtractOp>>(
      converter, &mlirContext);

  // Sim dialect
  patterns.add<ProcessPrintOpLowering>(converter, &mlirContext);
  patterns.add<TerminateOpErasure>(converter, &mlirContext);
  patterns.add<FmtLiteralOpLowering, FmtDecOpLowering, FmtHexOpLowering,
               FmtBinOpLowering, FmtCharOpLowering, FmtConcatOpLowering,
               FmtDynStringOpLowering>(converter, &mlirContext);

  // HW bitcast (type reinterpretation, not covered by populateHWToLLVM)
  patterns.add<HWBitcastOpLowering>(converter, &mlirContext);

  // Apply conversion ONCE to the combined module
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(microModule, target, std::move(patterns),
                                    config))) {
    LLVM_DEBUG({
      llvm::dbgs() << "[AOT] Conversion failed\n";
      microModule.dump();
    });
    microModule.erase();
    return false;
  }

  // Clean up unrealized_conversion_cast ops left by partial conversion.
  // These arise from type materializations (e.g., !llhd.time → i64).
  // Strategy: repeatedly fold cast chains and erase dead casts until fixpoint.
  {
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<UnrealizedConversionCastOp> casts;
      microModule.walk([&](UnrealizedConversionCastOp op) {
        casts.push_back(op);
      });
      for (auto castOp : casts) {
        // Dead cast: no uses → erase.
        if (castOp.use_empty()) {
          castOp.erase();
          changed = true;
          continue;
        }
        // Round-trip fold: cast(B→A) where input is cast(A→B).
        // Replace cast(B→A)'s results with the original A values.
        if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
          auto inputCast = castOp.getOperand(0).getDefiningOp<
              UnrealizedConversionCastOp>();
          if (inputCast && inputCast.getNumOperands() == 1 &&
              inputCast.getNumResults() == 1) {
            // cast(cast(x : A→B) : B→A) → x
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
    llvm::dbgs() << "[AOT] Lowered LLVM IR:\n";
    microModule.dump();
  });

  // Register LLVM IR translation interfaces (required for ExecutionEngine).
  registerBuiltinDialectTranslation(*microModule.getContext());
  registerLLVMDialectTranslation(*microModule.getContext());

  // Initialize native target (required for JIT compilation).
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // === Step 3: Create ONE ExecutionEngine ===
  ExecutionEngineOptions engineOpts;
  engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  engineOpts.transformer = makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  auto engineOrErr = ExecutionEngine::create(microModule, engineOpts);
  if (!engineOrErr) {
    LLVM_DEBUG(llvm::dbgs()
               << "[AOT] ExecutionEngine creation failed: "
               << llvm::toString(engineOrErr.takeError()) << "\n");
    microModule.erase();
    return false;
  }

  engines.push_back(std::move(*engineOrErr));
  auto *jitEngine = engines.back().get();

  registerJITRuntimeSymbols(jitEngine);

  // === Step 4: Lookup all function pointers ===
  for (auto &info : extractedFuncs) {
    auto expectedFn = jitEngine->lookup(info.funcName);
    if (!expectedFn) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AOT] Function lookup failed: " << info.funcName << ": "
                 << llvm::toString(expectedFn.takeError()) << "\n");
      engines.pop_back();
      microModule.erase();
      return false;
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

  auto endTime = std::chrono::steady_clock::now();
  double compileMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  LLVM_DEBUG(llvm::dbgs() << "[AOT] Batch compiled " << results.size()
                          << " processes in "
                          << llvm::format("%.1f", compileMs) << " ms\n");

  microModule.erase();
  return true;
#endif
}
