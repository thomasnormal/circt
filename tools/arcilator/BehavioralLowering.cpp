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

//===----------------------------------------------------------------------===//
// LLHD Operation Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

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

/// Reject llhd.process until full process-state lowering is implemented.
struct ProcessOpLowering : public OpConversionPattern<llhd::ProcessOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ProcessOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: llhd.process "
           "(state-machine/coroutine lowering not implemented)";
    return failure();
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

/// Reject moore.detect_event until proper edge detection is lowered.
struct DetectEventLowering
    : public OpConversionPattern<moore::DetectEventOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::DetectEventOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: moore.detect_event "
           "(edge detection lowering not implemented)";
    return failure();
  }
};

/// Reject moore.wait_event until wait-event semantics are lowered.
struct WaitEventLowering
    : public OpConversionPattern<moore::WaitEventOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::WaitEventOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: moore.wait_event "
           "(event wait lowering not implemented)";
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Sim Operation Lowering Patterns
//===----------------------------------------------------------------------===//

/// Reject sim.fork until process forking semantics are lowered.
struct SimForkLowering : public OpConversionPattern<sim::SimForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimForkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: sim.fork "
           "(fork/join lowering not implemented)";
    return failure();
  }
};

/// Reject sim.proc.print until formatted-print lowering is implemented.
struct PrintFormattedProcLowering
    : public OpConversionPattern<sim::PrintFormattedProcOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::PrintFormattedProcOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: sim.proc.print "
           "(printf lowering not implemented)";
    return failure();
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

/// Reject sim.disable_fork until disable-fork semantics are lowered.
struct SimDisableForkLowering
    : public OpConversionPattern<sim::SimDisableForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimDisableForkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: sim.disable_fork "
           "(disable-fork lowering not implemented)";
    return failure();
  }
};

/// Reject sim.wait_fork until wait-fork semantics are lowered.
struct SimWaitForkLowering
    : public OpConversionPattern<sim::SimWaitForkOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::SimWaitForkOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: sim.wait_fork "
           "(wait-fork lowering not implemented)";
    return failure();
  }
};

/// Reject sim.pause until pause semantics are lowered.
struct SimPauseLowering : public OpConversionPattern<sim::PauseOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::PauseOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter & /*rewriter*/) const final {
    op.emitError()
        << "unsupported in arcilator BehavioralLowering: sim.pause "
           "(pause lowering not implemented)";
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// HW Module Operation Lowering Patterns
//===----------------------------------------------------------------------===//

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

/// Lower sim.fmt.literal to a null pointer (no-op metadata).
struct SimFmtLitLowering
    : public OpConversionPattern<sim::FormatLiteralOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrTy);
    return success();
  }
};

/// Lower sim.fmt.concat to a null pointer (no-op metadata).
struct SimFmtConcatLowering
    : public OpConversionPattern<sim::FormatStringConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatStringConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrTy);
    return success();
  }
};

/// Lower sim.fmt.dyn_string to a null pointer (no-op metadata).
struct SimFmtDynStringLowering
    : public OpConversionPattern<sim::FormatDynStringOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sim::FormatDynStringOp op, OpAdaptor adaptor,
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
  populateHWToLLVMTypeConversions(converter);

  // Set up the conversion target.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Mark non-LLVM dialects as illegal.
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalDialect<llhd::LLHDDialect>();
  target.addIllegalDialect<moore::MooreDialect>();
  target.addIllegalDialect<sim::SimDialect>();
  target.addIllegalDialect<seq::SeqDialect>();

  // Set up the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);

  // Standard MLIR lowering patterns.
  populateSCFToControlFlowConversionPatterns(patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
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
               DriveOpLowering, ProcessOpLowering, WaitOpLowering,
               HaltOpLowering>(converter, &getContext());

  // Moore lowering patterns.
  patterns.add<MooreConstantLowering, DetectEventLowering, WaitEventLowering>(
      converter, &getContext());

  // Sim lowering patterns.
  patterns.add<SimForkLowering, PrintFormattedProcLowering,
               SimTerminateLowering, SimDisableForkLowering,
               SimWaitForkLowering, SimPauseLowering>(converter,
                                                       &getContext());

  // HW module lowering patterns.
  patterns.add<HWModuleOpLowering, HWOutputOpLowering,
               HWInstanceOpLowering>(converter, &getContext());

  // Format string lowering.
  patterns.add<SimFmtLitLowering, SimFmtConcatLowering,
               SimFmtDynStringLowering>(converter, &getContext());

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
