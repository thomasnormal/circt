//===- BehavioralLowering.cpp - Lower behavioral LLHD to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers behavioral LLHD processes, Moore events, and Sim ops
// to LLVM IR with runtime scheduler calls. Combined with the existing
// CombToArith, HWToLLVM, ArithToLLVM, FuncToLLVM, and ControlFlowToLLVM
// conversion patterns, this enables arcilator to compile full UVM/testbench
// designs to native code.
//
//===----------------------------------------------------------------------===//

#include "BehavioralLowering.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/ConversionPatternSet.h"
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
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

static void addLLHDTypeConversions(LLVMTypeConverter &converter) {
  // !llhd.ref<T> → !llvm.ptr
  converter.addConversion([&](llhd::RefType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  // !llhd.time → i64
  converter.addConversion([&](llhd::TimeType type) {
    return IntegerType::get(type.getContext(), 64);
  });
}

static void addMooreTypeConversions(LLVMTypeConverter &converter) {
  // !moore.iN / !moore.lN → iN
  converter.addConversion([&](moore::IntType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });
  // !moore.event → i1
  converter.addConversion([&](moore::EventType type) {
    return IntegerType::get(type.getContext(), 1);
  });
}

static void addSimTypeConversions(LLVMTypeConverter &converter) {
  // !sim.fmtstr → !llvm.ptr
  converter.addConversion([&](sim::FormatStringType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
}

static void addSeqTypeConversions(LLVMTypeConverter &converter) {
  // !seq.clock -> i1
  converter.addConversion([&](seq::ClockType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  // !seq.immutable<T> -> T
  converter.addConversion([&](seq::ImmutableType type) {
    return converter.convertType(type.getInnerType());
  });
}

static FailureOr<Value>
materializeStructCastForUnrealizedCast(ConversionPatternRewriter &rewriter,
                                       Location loc, Value input,
                                       LLVM::LLVMStructType dstStructTy) {
  auto srcStructTy = llvm::dyn_cast<LLVM::LLVMStructType>(input.getType());
  if (!srcStructTy || srcStructTy.isOpaque() || dstStructTy.isOpaque())
    return failure();
  auto srcBody = srcStructTy.getBody();
  auto dstBody = dstStructTy.getBody();
  if (srcBody.size() != dstBody.size())
    return failure();

  // Identity cast.
  if (srcBody == dstBody)
    return input;

  // Behavioral imports frequently materialize the same struct with reverse
  // element order across dialect boundaries. Lower this explicitly instead of
  // leaving a builtin.unrealized_conversion_cast behind.
  const size_t count = srcBody.size();
  for (size_t i = 0; i < count; ++i)
    if (srcBody[count - 1 - i] != dstBody[i])
      return failure();

  Value result = LLVM::UndefOp::create(rewriter, loc, dstStructTy);
  for (size_t i = 0; i < count; ++i) {
    int64_t srcIdx = static_cast<int64_t>(count - 1 - i);
    int64_t dstIdx = static_cast<int64_t>(i);
    Value element = LLVM::ExtractValueOp::create(
        rewriter, loc, input, ArrayRef<int64_t>{srcIdx});
    result = LLVM::InsertValueOp::create(rewriter, loc, result, element,
                                         ArrayRef<int64_t>{dstIdx});
  }
  return result;
}

//===----------------------------------------------------------------------===//
// LLHD Operation Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower builtin.unrealized_conversion_cast to explicit LLVM operations.
struct UnrealizedConversionCastLowering
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    auto convertedResultTy =
        getTypeConverter()->convertType(op.getOutputs().front().getType());
    if (!convertedResultTy)
      return failure();
    Value input = adaptor.getInputs().front();
    auto loc = op.getLoc();

    if (input.getType() == convertedResultTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (llvm::isa<LLVM::LLVMPointerType>(input.getType())) {
      if (auto dstIntTy = llvm::dyn_cast<IntegerType>(convertedResultTy)) {
        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, dstIntTy, input);
        return success();
      }
    }

    if (auto srcIntTy = llvm::dyn_cast<IntegerType>(input.getType())) {
      if (llvm::isa<LLVM::LLVMPointerType>(convertedResultTy)) {
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, convertedResultTy,
                                                      input);
        return success();
      }
      if (auto dstIntTy = llvm::dyn_cast<IntegerType>(convertedResultTy)) {
        if (srcIntTy.getWidth() == dstIntTy.getWidth()) {
          rewriter.replaceOp(op, input);
          return success();
        }
        if (srcIntTy.getWidth() > dstIntTy.getWidth()) {
          rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, dstIntTy, input);
          return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstIntTy, input);
        return success();
      }
    }

    if (auto dstStructTy =
            llvm::dyn_cast<LLVM::LLVMStructType>(convertedResultTy)) {
      auto casted = materializeStructCastForUnrealizedCast(rewriter, loc, input,
                                                           dstStructTy);
      if (succeeded(casted)) {
        rewriter.replaceOp(op, *casted);
        return success();
      }
    }

    op.emitError()
        << "unsupported in arcilator BehavioralLowering: "
           "builtin.unrealized_conversion_cast from "
        << input.getType() << " to " << convertedResultTy;
    return failure();
  }
};

/// Canonicalize llvm.getelementptr element types through the active type
/// converter so no non-LLVM dialect types remain in `elem_type`.
struct GEPElemTypeLowering : public OpConversionPattern<LLVM::GEPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto convertedElemType = getTypeConverter()->convertType(op.getElemType());
    if (!convertedElemType)
      return failure();
    if (convertedElemType == op.getElemType())
      return failure();

    SmallVector<LLVM::GEPArg> gepArgs;
    gepArgs.reserve(op.getRawConstantIndices().size());
    unsigned dynamicIndex = 0;
    for (int32_t idx : op.getRawConstantIndices()) {
      if (idx == LLVM::GEPOp::kDynamicIndex) {
        if (dynamicIndex >= adaptor.getDynamicIndices().size())
          return failure();
        gepArgs.push_back(adaptor.getDynamicIndices()[dynamicIndex++]);
      } else {
        gepArgs.push_back(idx);
      }
    }

    auto rewritten = LLVM::GEPOp::create(rewriter, op.getLoc(), op.getType(),
                                         convertedElemType, adaptor.getBase(),
                                         gepArgs, op.getNoWrapFlags());
    rewriter.replaceOp(op, rewritten.getResult());
    return success();
  }
};

/// Lower llhd.constant_time to an i64 constant.
struct ConstantTimeLowering
    : public OpConversionPattern<llhd::ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto timeAttr = op.getValueAttr();
    int64_t encoded = 0;
    if (auto time = llvm::dyn_cast<llhd::TimeAttr>(timeAttr)) {
      auto real = time.getTime();
      auto delta = time.getDelta();
      auto eps = time.getEpsilon();
      encoded = real * 1000000 + delta * 1000 + eps;
    }
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(encoded));
    return success();
  }
};

/// Lower llhd.current_time to a runtime call.
struct CurrentTimeLowering
    : public OpConversionPattern<llhd::CurrentTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::CurrentTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__arc_sched_current_time", {}, i64Ty);
    if (failed(funcOp))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, *funcOp, ValueRange{});
    return success();
  }
};

/// Lower llhd.time_to_int — identity after type conversion.
struct TimeToIntLowering
    : public OpConversionPattern<llhd::TimeToIntOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::TimeToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

/// Lower llhd.int_to_time — identity after type conversion.
struct IntToTimeLowering
    : public OpConversionPattern<llhd::IntToTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::IntToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

/// Lower llhd.sig to a scheduler signal registration call.
struct SignalOpLowering : public OpConversionPattern<llhd::SignalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();

    auto innerType = op.getInit().getType();
    auto dataLayout = DataLayout::closest(op);
    unsigned sizeInBytes = 0;
    if (auto convertedType = getTypeConverter()->convertType(innerType))
      sizeInBytes = dataLayout.getTypeSize(convertedType);

    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__arc_sched_create_signal", {ptrTy, i64Ty}, ptrTy);
    if (failed(funcOp))
      return failure();

    // Store init value to a temporary alloca, pass pointer.
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(1));
    auto initConverted = adaptor.getInit();
    auto initType = initConverted.getType();
    auto allocaOp =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, initType, one);
    LLVM::StoreOp::create(rewriter, loc, initConverted, allocaOp);
    auto sizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(sizeInBytes));

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *funcOp, ValueRange{allocaOp, sizeVal});
    return success();
  }
};

/// Lower llhd.prb to a scheduler signal read call + load.
struct ProbeOpLowering : public OpConversionPattern<llhd::ProbeOp> {
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
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__arc_sched_read_signal", {ptrTy}, ptrTy);
    if (failed(funcOp))
      return failure();

    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                        ValueRange{adaptor.getSignal()});
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                               callOp.getResult());
    return success();
  }
};

/// Lower llhd.drv to a scheduler signal drive call.
struct DriveOpLowering : public OpConversionPattern<llhd::DriveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::DriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i1Ty = rewriter.getI1Type();

    // Store the value into a temporary.
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(1));
    auto valueConverted = adaptor.getValue();
    auto valueType = valueConverted.getType();
    auto allocaOp =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, valueType, one);
    LLVM::StoreOp::create(rewriter, loc, valueConverted, allocaOp);

    Value enableVal;
    if (op.getEnable())
      enableVal = adaptor.getEnable();
    else
      enableVal = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                           rewriter.getBoolAttr(true));

    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "__arc_sched_drive_signal",
        {ptrTy, ptrTy, i64Ty, i1Ty}, voidTy);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp,
                          ValueRange{adaptor.getSignal(), allocaOp,
                                     adaptor.getTime(), enableVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower LLHD signal projection ops as passthrough references.
///
/// This keeps behavioral lowering moving even when projection cleanup passes
/// left residual sig.extract/sig.array_get/sig.struct_extract ops.
template <typename OpT>
struct SigProjectionPassthroughLowering : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Lower llhd.process to zero-valued placeholders.
///
/// This is a temporary fallback to keep behavioral conversion progressing when
/// residual LLHD processes remain after front-end cleanup.
struct ProcessOpLowering : public OpConversionPattern<llhd::ProcessOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ProcessOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> replacements;
    replacements.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      auto convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType)
        return failure();
      replacements.push_back(
          LLVM::ZeroOp::create(rewriter, op.getLoc(), convertedType));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Lower llhd.combinational to zero-valued placeholders.
///
/// This preserves conversion progress for residual combinational regions that
/// survive LLHD cleanup in large imported UVM designs.
struct CombinationalOpLowering
    : public OpConversionPattern<llhd::CombinationalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::CombinationalOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> replacements;
    replacements.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      auto convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType)
        return failure();
      replacements.push_back(
          LLVM::ZeroOp::create(rewriter, op.getLoc(), convertedType));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Reject llhd.wait until process suspension semantics are implemented.
struct WaitOpLowering : public OpConversionPattern<llhd::WaitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::WaitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: llhd.wait "
           "(process suspension lowering not implemented)";
    return failure();
  }
};

/// Lower llhd.halt to unreachable.
struct HaltOpLowering : public OpConversionPattern<llhd::HaltOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::HaltOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Moore Operation Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower moore.constant to an LLVM integer constant.
///
/// Four-valued constants with X/Z bits remain unsupported in this path.
struct MooreConstantLowering
    : public OpConversionPattern<moore::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::ConstantOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    auto convertedType = getTypeConverter()->convertType(op.getType());
    auto intType = llvm::dyn_cast_or_null<IntegerType>(convertedType);
    if (!intType)
      return failure();

    FVInt value = op.getValue();
    if (value.hasUnknown()) {
      op.emitError()
          << "unsupported in arcilator BehavioralLowering: moore.constant "
             "with X/Z bits (four-valued constant lowering not implemented)";
      return failure();
    }

    auto apValue = value.toAPInt(false).zextOrTrunc(intType.getWidth());
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, intType, IntegerAttr::get(intType, apValue));
    return success();
  }
};

/// Lower moore.detect_event to no-op.
///
/// detect_event has no result; in behavioral fallback mode we conservatively
/// drop it and rely on higher-level wait handling.
struct DetectEventLowering
    : public OpConversionPattern<moore::DetectEventOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::DetectEventOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower moore.wait_event to no-op.
///
/// This unblocks compilation of behavioral helper functions that still contain
/// event controls after front-end lowering. Full event suspension/resume
/// semantics are not implemented in this pass yet.
struct WaitEventLowering
    : public OpConversionPattern<moore::WaitEventOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::WaitEventOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sim Operation Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower sim.fork as a stub handle.
struct SimForkLowering : public OpConversionPattern<sim::SimForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    Value forkHandle = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, forkHandle);
    return success();
  }
};

/// Lower sim.fork.terminator by erasing it.
struct SimForkTerminatorLowering
    : public OpConversionPattern<sim::SimForkTerminatorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimForkTerminatorOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.proc.print as no-op.
struct PrintFormattedProcLowering
    : public OpConversionPattern<sim::PrintFormattedProcOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::PrintFormattedProcOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.terminate to a call to exit().
struct SimTerminateLowering
    : public OpConversionPattern<sim::TerminateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::TerminateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, "exit", {i32Ty}, voidTy);
    if (failed(funcOp))
      return failure();
    auto exitCode = LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty,
                                              rewriter.getI32IntegerAttr(0));
    LLVM::CallOp::create(rewriter, op.getLoc(), *funcOp,
                          ValueRange{exitCode});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.disable_fork as no-op.
struct SimDisableForkLowering
    : public OpConversionPattern<sim::SimDisableForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimDisableForkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.wait_fork as no-op.
struct SimWaitForkLowering
    : public OpConversionPattern<sim::SimWaitForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimWaitForkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower sim.pause as no-op.
struct SimPauseLowering : public OpConversionPattern<sim::PauseOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::PauseOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Seq Operation Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower seq.from_immutable — identity after type conversion.
struct SeqFromImmutableLowering
    : public OpConversionPattern<seq::FromImmutableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::FromImmutableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Lower seq.from_clock — identity after type conversion.
struct SeqFromClockLowering : public OpConversionPattern<seq::FromClockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::FromClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Lower seq.initial to zero placeholders for produced immutable values.
struct SeqInitialLowering : public OpConversionPattern<seq::InitialOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::InitialOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> replacements;
    replacements.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      auto convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType)
        return failure();
      replacements.push_back(
          LLVM::ZeroOp::create(rewriter, op.getLoc(), convertedType));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Lower seq.yield by erasing it.
struct SeqYieldLowering : public OpConversionPattern<seq::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(seq::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HW Module Operation Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower func.call_indirect conservatively to zero placeholders.
struct FuncCallIndirectLowering
    : public OpConversionPattern<func::CallIndirectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> replacements;
    replacements.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      auto convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType)
        return failure();
      replacements.push_back(
          LLVM::ZeroOp::create(rewriter, op.getLoc(), convertedType));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Lower hw.module to an LLVM function.
struct HWModuleOpLowering : public OpConversionPattern<hw::HWModuleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto funcName = op.getName();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());

    SmallVector<Type> inputTypes;
    for (auto inputType : op.getInputTypes()) {
      auto converted = getTypeConverter()->convertType(inputType);
      if (!converted)
        return failure();
      inputTypes.push_back(converted);
    }

    auto funcType = LLVM::LLVMFunctionType::get(voidTy, inputTypes);
    auto funcOp =
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), funcName, funcType);

    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

    auto &entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      auto arg = entryBlock.getArgument(i);
      auto newType = getTypeConverter()->convertType(arg.getType());
      if (!newType)
        return failure();
      arg.setType(newType);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower hw.output to llvm.return void.
struct HWOutputOpLowering : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
    return success();
  }
};

/// Lower hw.bitcast for simple behavioral cases.
///
/// This currently supports:
/// 1. no-op casts where source and destination LLVM types already match
/// 2. zero-valued integer-to-aggregate casts used for aggregate zero init
struct HWBitcastOpLowering : public OpConversionPattern<hw::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getInput();
    if (input.getType() == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto srcStructTy =
            llvm::dyn_cast<LLVM::LLVMStructType>(input.getType())) {
      if (auto dstIntTy = llvm::dyn_cast<IntegerType>(convertedResultType)) {
        auto srcElems = srcStructTy.getBody();
        if (!srcElems.empty()) {
          uint64_t totalWidth = 0;
          bool allIntegerElems = true;
          for (Type elemTy : srcElems) {
            auto intElemTy = llvm::dyn_cast<IntegerType>(elemTy);
            if (!intElemTy) {
              allIntegerElems = false;
              break;
            }
            totalWidth += intElemTy.getWidth();
          }
          if (allIntegerElems && totalWidth == dstIntTy.getWidth()) {
            auto loc = op.getLoc();
            Value accum = LLVM::ConstantOp::create(
                rewriter, loc, dstIntTy, IntegerAttr::get(dstIntTy, 0));
            uint64_t bitOffset = 0;
            for (auto [idx, elemTy] : llvm::enumerate(srcElems)) {
              auto intElemTy = llvm::cast<IntegerType>(elemTy);
              Value elem = LLVM::ExtractValueOp::create(
                  rewriter, loc, input, ArrayRef<int64_t>{(int64_t)idx});
              Value ext = intElemTy.getWidth() == dstIntTy.getWidth()
                              ? elem
                              : Value(LLVM::ZExtOp::create(rewriter, loc, dstIntTy,
                                                           elem));
              if (bitOffset != 0) {
                auto shiftAttr = IntegerAttr::get(
                    dstIntTy, APInt(dstIntTy.getWidth(), bitOffset));
                Value shiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, dstIntTy, shiftAttr);
                ext = LLVM::ShlOp::create(rewriter, loc, dstIntTy, ext,
                                          shiftAmt);
              }
              accum = LLVM::OrOp::create(rewriter, loc, dstIntTy, accum, ext);
              bitOffset += intElemTy.getWidth();
            }
            rewriter.replaceOp(op, accum);
            return success();
          }
        }
      }
    }

    if (auto srcArrayTy =
            llvm::dyn_cast<LLVM::LLVMArrayType>(input.getType())) {
      if (auto dstIntTy = llvm::dyn_cast<IntegerType>(convertedResultType)) {
        if (auto srcElemIntTy =
                llvm::dyn_cast<IntegerType>(srcArrayTy.getElementType())) {
          uint64_t elemWidth = srcElemIntTy.getWidth();
          uint64_t numElems = srcArrayTy.getNumElements();
          if (elemWidth > 0 &&
              dstIntTy.getWidth() == static_cast<unsigned>(numElems * elemWidth)) {
            auto loc = op.getLoc();
            Value accum = LLVM::ConstantOp::create(
                rewriter, loc, dstIntTy, IntegerAttr::get(dstIntTy, 0));
            for (uint64_t i = 0; i < numElems; ++i) {
              Value elem = LLVM::ExtractValueOp::create(
                  rewriter, loc, input, ArrayRef<int64_t>{(int64_t)i});
              Value ext = LLVM::ZExtOp::create(rewriter, loc, dstIntTy, elem);
              auto shiftAttr = IntegerAttr::get(
                  dstIntTy, APInt(dstIntTy.getWidth(), i * elemWidth));
              Value shiftAmt = LLVM::ConstantOp::create(rewriter, loc, dstIntTy,
                                                        shiftAttr);
              Value shifted =
                  LLVM::ShlOp::create(rewriter, loc, dstIntTy, ext, shiftAmt);
              accum = LLVM::OrOp::create(rewriter, loc, dstIntTy, accum, shifted);
            }
            rewriter.replaceOp(op, accum);
            return success();
          }
        }
      }

      if (auto dstStructTy =
              llvm::dyn_cast<LLVM::LLVMStructType>(convertedResultType)) {
        auto srcElemStructTy =
            llvm::dyn_cast<LLVM::LLVMStructType>(srcArrayTy.getElementType());
        auto dstElems = dstStructTy.getBody();
        if (srcElemStructTy && dstElems.size() == 2) {
          auto srcElems = srcElemStructTy.getBody();
          auto srcLoTy = srcElems.size() > 0 ? llvm::dyn_cast<IntegerType>(srcElems[0])
                                             : IntegerType();
          auto srcHiTy = srcElems.size() > 1 ? llvm::dyn_cast<IntegerType>(srcElems[1])
                                             : IntegerType();
          auto dstLoTy = llvm::dyn_cast<IntegerType>(dstElems[0]);
          auto dstHiTy = llvm::dyn_cast<IntegerType>(dstElems[1]);
          uint64_t numElems = srcArrayTy.getNumElements();
          if (srcElems.size() == 2 && srcLoTy && srcHiTy && dstLoTy && dstHiTy) {
            uint64_t loWidth = srcLoTy.getWidth();
            uint64_t hiWidth = srcHiTy.getWidth();
            if (loWidth > 0 && hiWidth > 0 &&
                dstLoTy.getWidth() ==
                    static_cast<unsigned>(numElems * loWidth) &&
                dstHiTy.getWidth() ==
                    static_cast<unsigned>(numElems * hiWidth)) {
              auto loc = op.getLoc();
              Value loAccum = LLVM::ConstantOp::create(
                  rewriter, loc, dstLoTy, IntegerAttr::get(dstLoTy, 0));
              Value hiAccum = LLVM::ConstantOp::create(
                  rewriter, loc, dstHiTy, IntegerAttr::get(dstHiTy, 0));
              for (uint64_t i = 0; i < numElems; ++i) {
                Value elem = LLVM::ExtractValueOp::create(
                    rewriter, loc, input, ArrayRef<int64_t>{(int64_t)i});
                Value lo = LLVM::ExtractValueOp::create(
                    rewriter, loc, elem, ArrayRef<int64_t>{0});
                Value hi = LLVM::ExtractValueOp::create(
                    rewriter, loc, elem, ArrayRef<int64_t>{1});
                Value loExt =
                    LLVM::ZExtOp::create(rewriter, loc, dstLoTy, lo);
                Value hiExt =
                    LLVM::ZExtOp::create(rewriter, loc, dstHiTy, hi);
                auto loShiftAttr = IntegerAttr::get(
                    dstLoTy, APInt(dstLoTy.getWidth(), i * loWidth));
                auto hiShiftAttr = IntegerAttr::get(
                    dstHiTy, APInt(dstHiTy.getWidth(), i * hiWidth));
                Value loShiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, dstLoTy, loShiftAttr);
                Value hiShiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, dstHiTy, hiShiftAttr);
                Value loShifted = LLVM::ShlOp::create(rewriter, loc, dstLoTy,
                                                      loExt, loShiftAmt);
                Value hiShifted = LLVM::ShlOp::create(rewriter, loc, dstHiTy,
                                                      hiExt, hiShiftAmt);
                loAccum = LLVM::OrOp::create(rewriter, loc, dstLoTy, loAccum,
                                             loShifted);
                hiAccum = LLVM::OrOp::create(rewriter, loc, dstHiTy, hiAccum,
                                             hiShifted);
              }
              Value result =
                  LLVM::UndefOp::create(rewriter, op.getLoc(), dstStructTy);
              result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                                   loAccum, ArrayRef<int64_t>{0});
              result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                                   hiAccum, ArrayRef<int64_t>{1});
              rewriter.replaceOp(op, result);
              return success();
            }
          }
        }
      }
    }

    if (auto srcIntTy = llvm::dyn_cast<IntegerType>(input.getType())) {
      if (auto dstArrayTy =
              llvm::dyn_cast<LLVM::LLVMArrayType>(convertedResultType)) {
        if (auto dstElemIntTy =
                llvm::dyn_cast<IntegerType>(dstArrayTy.getElementType())) {
          uint64_t elemWidth = dstElemIntTy.getWidth();
          uint64_t numElems = dstArrayTy.getNumElements();
          if (elemWidth > 0 &&
              srcIntTy.getWidth() == static_cast<unsigned>(numElems * elemWidth)) {
            auto loc = op.getLoc();
            Value result =
                LLVM::UndefOp::create(rewriter, loc, convertedResultType);
            for (uint64_t i = 0; i < numElems; ++i) {
              auto shiftAttr = IntegerAttr::get(
                  srcIntTy, APInt(srcIntTy.getWidth(), i * elemWidth));
              Value shiftAmt =
                  LLVM::ConstantOp::create(rewriter, loc, srcIntTy, shiftAttr);
              Value shifted =
                  LLVM::LShrOp::create(rewriter, loc, srcIntTy, input, shiftAmt);
              Value chunk = LLVM::TruncOp::create(rewriter, loc, dstElemIntTy,
                                                  shifted);
              result = LLVM::InsertValueOp::create(
                  rewriter, loc, result, chunk, ArrayRef<int64_t>{(int64_t)i});
            }
            rewriter.replaceOp(op, result);
            return success();
          }
        }
      }

      if (auto dstStructTy =
              llvm::dyn_cast<LLVM::LLVMStructType>(convertedResultType)) {
        auto elements = dstStructTy.getBody();
        if (!elements.empty()) {
          uint64_t totalWidth = 0;
          bool allIntegerElems = true;
          for (Type elemTy : elements) {
            auto intElemTy = llvm::dyn_cast<IntegerType>(elemTy);
            if (!intElemTy) {
              allIntegerElems = false;
              break;
            }
            totalWidth += intElemTy.getWidth();
          }
          if (allIntegerElems && totalWidth == srcIntTy.getWidth()) {
            auto loc = op.getLoc();
            Value result =
                LLVM::UndefOp::create(rewriter, loc, convertedResultType);
            uint64_t bitOffset = 0;
            for (auto [idx, elemTy] : llvm::enumerate(elements)) {
              auto intElemTy = llvm::cast<IntegerType>(elemTy);
              Value value = input;
              if (bitOffset != 0) {
                auto shiftAttr = IntegerAttr::get(
                    srcIntTy, APInt(srcIntTy.getWidth(), bitOffset));
                Value shiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, srcIntTy, shiftAttr);
                value = LLVM::LShrOp::create(rewriter, loc, srcIntTy, input,
                                             shiftAmt);
              }
              Value chunk = LLVM::TruncOp::create(rewriter, loc, intElemTy,
                                                  value);
              result = LLVM::InsertValueOp::create(
                  rewriter, loc, result, chunk,
                  ArrayRef<int64_t>{(int64_t)idx});
              bitOffset += intElemTy.getWidth();
            }
            rewriter.replaceOp(op, result);
            return success();
          }
        }

        if (elements.size() == 2) {
          auto dstElem0Ty = llvm::dyn_cast<IntegerType>(elements[0]);
          auto dstElem1Ty = llvm::dyn_cast<IntegerType>(elements[1]);
          if (dstElem0Ty && dstElem1Ty &&
              srcIntTy.getWidth() ==
                  dstElem0Ty.getWidth() + dstElem1Ty.getWidth()) {
            Value low = LLVM::TruncOp::create(rewriter, op.getLoc(), dstElem0Ty,
                                              input);
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), srcIntTy,
                IntegerAttr::get(srcIntTy, dstElem0Ty.getWidth()));
            Value highShifted = LLVM::LShrOp::create(
                rewriter, op.getLoc(), srcIntTy, input, shiftAmt);
            Value high = LLVM::TruncOp::create(rewriter, op.getLoc(),
                                               dstElem1Ty, highShifted);

            Value result =
                LLVM::UndefOp::create(rewriter, op.getLoc(), dstStructTy);
            result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                                 low, ArrayRef<int64_t>{0});
            result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                                 high, ArrayRef<int64_t>{1});
            rewriter.replaceOp(op, result);
            return success();
          }
        }
      }
    }

    if (auto srcStructTy =
            llvm::dyn_cast<LLVM::LLVMStructType>(input.getType())) {
      if (auto dstArrayTy =
              llvm::dyn_cast<LLVM::LLVMArrayType>(convertedResultType)) {
        auto srcElems = srcStructTy.getBody();
        auto dstElemStructTy =
            llvm::dyn_cast<LLVM::LLVMStructType>(dstArrayTy.getElementType());
        if (srcElems.size() == 2 && dstElemStructTy) {
          auto dstElems = dstElemStructTy.getBody();
          auto srcLoTy = llvm::dyn_cast<IntegerType>(srcElems[0]);
          auto srcHiTy = llvm::dyn_cast<IntegerType>(srcElems[1]);
          auto dstLoTy = dstElems.size() > 0 ? llvm::dyn_cast<IntegerType>(dstElems[0])
                                             : IntegerType();
          auto dstHiTy = dstElems.size() > 1 ? llvm::dyn_cast<IntegerType>(dstElems[1])
                                             : IntegerType();
          uint64_t numElems = dstArrayTy.getNumElements();
          if (srcLoTy && srcHiTy && dstElems.size() == 2 && dstLoTy && dstHiTy) {
            uint64_t loWidth = dstLoTy.getWidth();
            uint64_t hiWidth = dstHiTy.getWidth();
            if (loWidth > 0 && hiWidth > 0 &&
                srcLoTy.getWidth() ==
                    static_cast<unsigned>(numElems * loWidth) &&
                srcHiTy.getWidth() ==
                    static_cast<unsigned>(numElems * hiWidth)) {
              auto loc = op.getLoc();
              Value srcLo = LLVM::ExtractValueOp::create(
                  rewriter, loc, input, ArrayRef<int64_t>{0});
              Value srcHi = LLVM::ExtractValueOp::create(
                  rewriter, loc, input, ArrayRef<int64_t>{1});
              Value result =
                  LLVM::UndefOp::create(rewriter, loc, convertedResultType);
              for (uint64_t i = 0; i < numElems; ++i) {
                auto loShiftAttr = IntegerAttr::get(
                    srcLoTy, APInt(srcLoTy.getWidth(), i * loWidth));
                auto hiShiftAttr = IntegerAttr::get(
                    srcHiTy, APInt(srcHiTy.getWidth(), i * hiWidth));
                Value loShiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, srcLoTy, loShiftAttr);
                Value hiShiftAmt = LLVM::ConstantOp::create(
                    rewriter, loc, srcHiTy, hiShiftAttr);
                Value loShifted = LLVM::LShrOp::create(rewriter, loc, srcLoTy,
                                                       srcLo, loShiftAmt);
                Value hiShifted = LLVM::LShrOp::create(rewriter, loc, srcHiTy,
                                                       srcHi, hiShiftAmt);
                Value loChunk = LLVM::TruncOp::create(rewriter, loc, dstLoTy,
                                                      loShifted);
                Value hiChunk = LLVM::TruncOp::create(rewriter, loc, dstHiTy,
                                                      hiShifted);
                Value elem =
                    LLVM::UndefOp::create(rewriter, loc, dstElemStructTy);
                elem = LLVM::InsertValueOp::create(rewriter, loc, elem, loChunk,
                                                   ArrayRef<int64_t>{0});
                elem = LLVM::InsertValueOp::create(rewriter, loc, elem, hiChunk,
                                                   ArrayRef<int64_t>{1});
                result = LLVM::InsertValueOp::create(
                    rewriter, loc, result, elem, ArrayRef<int64_t>{(int64_t)i});
              }
              rewriter.replaceOp(op, result);
              return success();
            }
          }
        }
      }
    }

    // Generic same-size reinterpret cast fallback for aggregate bitcasts.
    // This covers packed struct/array reshapes that appear in AVIP imports.
    if (LLVM::isCompatibleType(input.getType()) &&
        LLVM::isCompatibleType(convertedResultType)) {
      auto dataLayout = DataLayout::closest(op);
      uint64_t srcBits = dataLayout.getTypeSizeInBits(input.getType());
      uint64_t dstBits = dataLayout.getTypeSizeInBits(convertedResultType);
      if (srcBits == dstBits) {
        auto loc = op.getLoc();
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64Ty = rewriter.getI64Type();
        auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(1));
        auto alloca =
            LLVM::AllocaOp::create(rewriter, loc, ptrTy, input.getType(), one);
        LLVM::StoreOp::create(rewriter, loc, input, alloca);
        auto casted =
            LLVM::LoadOp::create(rewriter, loc, convertedResultType, alloca);
        rewriter.replaceOp(op, casted.getResult());
        return success();
      }
    }

    auto isZeroConstant = [&](Value value) -> bool {
      if (auto llvmConst = value.getDefiningOp<LLVM::ConstantOp>()) {
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(llvmConst.getValue()))
          return intAttr.getValue().isZero();
        return llvm::isa<LLVM::ZeroAttr>(llvmConst.getValue());
      }
      if (auto hwConst = op.getInput().getDefiningOp<hw::ConstantOp>())
        return hwConst.getValueAttr().getValue().isZero();
      return false;
    };

    if (isZeroConstant(input)) {
      rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, convertedResultType);
      return success();
    }

    op.emitError()
        << "unsupported in arcilator BehavioralLowering: hw.bitcast "
           "(only identity and zero-initializer casts are currently lowered)";
    return failure();
  }
};

/// Lower hw.instance to a function call.
struct HWInstanceOpLowering : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());

    SmallVector<Type> argTypes;
    for (auto arg : adaptor.getOperands())
      argTypes.push_back(arg.getType());

    auto funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, op.getModuleName(), argTypes, voidTy);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, op.getLoc(), *funcOp,
                          adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sim Format String Lowering
//===----------------------------------------------------------------------===//

/// Lower sim.fmt.* formatting metadata to null pointers.
template <typename OpT>
struct SimFmtNullLowering : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrTy);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerBehavioralToLLVMPass
    : public PassWrapper<LowerBehavioralToLLVMPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBehavioralToLLVMPass)

  StringRef getArgument() const override {
    return "lower-behavioral-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower behavioral LLHD/Moore/Sim operations to LLVM IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, mlir::arith::ArithDialect,
                    mlir::cf::ControlFlowDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void LowerBehavioralToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Set up the type converter.
  LLVMTypeConverter converter(&getContext());
  addLLHDTypeConversions(converter);
  addMooreTypeConversions(converter);
  addSimTypeConversions(converter);
  addSeqTypeConversions(converter);
  populateHWToLLVMTypeConversions(converter);

  // Set up the conversion target.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<LLVM::GEPOp>([](LLVM::GEPOp op) {
    return LLVM::isCompatibleType(op.getElemType());
  });
  target.addIllegalOp<UnrealizedConversionCastOp>();

  // Mark non-LLVM dialects as illegal.
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalDialect<func::FuncDialect>();
  target.addIllegalDialect<llhd::LLHDDialect>();
  target.addIllegalDialect<moore::MooreDialect>();
  target.addIllegalDialect<sim::SimDialect>();
  target.addIllegalDialect<seq::SeqDialect>();

  // Set up the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);

  // Standard MLIR lowering patterns.
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  patterns.add<FuncCallIndirectLowering>(converter, &getContext());
  patterns.add<UnrealizedConversionCastLowering>(converter, &getContext());
  patterns.add<GEPElemTypeLowering>(converter, &getContext());
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // CIRCT HW/Comb lowering patterns.
  Namespace globals;
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  std::optional<HWToLLVMArraySpillCache> spillCacheOpt =
      HWToLLVMArraySpillCache();
  {
    OpBuilder spillBuilder(module);
    spillCacheOpt->spillNonHWOps(spillBuilder, converter, module);
  }
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap, spillCacheOpt);
  populateCombToArithConversionPatterns(converter, patterns);
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Behavioral LLHD lowering patterns.
  patterns.add<ConstantTimeLowering, CurrentTimeLowering, TimeToIntLowering,
               IntToTimeLowering, SignalOpLowering, ProbeOpLowering,
               DriveOpLowering,
               SigProjectionPassthroughLowering<llhd::SigExtractOp>,
               SigProjectionPassthroughLowering<llhd::SigArrayGetOp>,
               SigProjectionPassthroughLowering<llhd::SigArraySliceOp>,
               SigProjectionPassthroughLowering<llhd::SigStructExtractOp>,
               ProcessOpLowering, CombinationalOpLowering, WaitOpLowering,
               HaltOpLowering>(converter, &getContext());

  // Moore lowering patterns.
  patterns.add<MooreConstantLowering, DetectEventLowering, WaitEventLowering>(
      converter, &getContext());

  // Sim lowering patterns.
  patterns.add<SimForkLowering, SimForkTerminatorLowering,
               PrintFormattedProcLowering,
               SimTerminateLowering, SimDisableForkLowering,
               SimWaitForkLowering, SimPauseLowering>(converter,
                                                       &getContext());

  // Seq lowering patterns.
  patterns.add<SeqFromImmutableLowering, SeqFromClockLowering,
               SeqInitialLowering, SeqYieldLowering>(converter, &getContext());

  // HW module lowering patterns.
  patterns.add<HWModuleOpLowering, HWOutputOpLowering, HWBitcastOpLowering,
               HWInstanceOpLowering>(converter, &getContext());

  // Format string lowering.
  patterns.add<SimFmtNullLowering<sim::FormatLiteralOp>,
               SimFmtNullLowering<sim::FormatHexOp>,
               SimFmtNullLowering<sim::FormatOctOp>,
               SimFmtNullLowering<sim::FormatBinOp>,
               SimFmtNullLowering<sim::FormatScientificOp>,
               SimFmtNullLowering<sim::FormatFloatOp>,
               SimFmtNullLowering<sim::FormatGeneralOp>,
               SimFmtNullLowering<sim::FormatDecOp>,
               SimFmtNullLowering<sim::FormatCharOp>,
               SimFmtNullLowering<sim::FormatDynStringOp>,
               SimFmtNullLowering<sim::FormatStringConcatOp>>(converter,
                                                               &getContext());

  // Apply the conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(module, target, std::move(patterns),
                                    config)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerBehavioralToLLVMPass() {
  return std::make_unique<LowerBehavioralToLLVMPass>();
}
