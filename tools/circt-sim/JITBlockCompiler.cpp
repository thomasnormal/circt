//===- JITBlockCompiler.cpp - Block-level JIT compilation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITBlockCompiler.h"
#include "JITSchedulerRuntime.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <chrono>

#ifdef CIRCT_SIM_JIT_ENABLED
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#endif

#define DEBUG_TYPE "jit-block-compiler"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Lowering patterns for hot-block ops (subset of BehavioralLowering)
//===----------------------------------------------------------------------===//

namespace {

static bool dependsOnProbeValue(Value value,
                                llvm::SmallPtrSetImpl<Value> &visited) {
  if (!value)
    return false;
  if (!visited.insert(value).second)
    return false;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<llhd::ProbeOp>(defOp))
    return true;

  for (Value operand : defOp->getOperands()) {
    if (dependsOnProbeValue(operand, visited))
      return true;
  }
  return false;
}

static bool isJITSupportedSignalPayloadType(Type type) {
  return isa<IntegerType>(type);
}

/// Type conversion: llhd.time → i64, llhd.ref → ptr.
static void addHotBlockTypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion([&](llhd::RefType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](llhd::TimeType type) {
    return IntegerType::get(type.getContext(), 64);
  });
}

/// Lower llhd.prb → call __arc_sched_read_signal + load.
struct JITProbeOpLowering : public OpConversionPattern<llhd::ProbeOp> {
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

/// Lower llhd.drv → alloca + store + call __arc_sched_drive_signal.
struct JITDriveOpLowering : public OpConversionPattern<llhd::DriveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::DriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i8Ty = rewriter.getIntegerType(8);

    // Store value to temporary.
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(1));
    auto valueConverted = adaptor.getValue();
    auto valueType = valueConverted.getType();
    auto allocaOp =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, valueType, one);
    LLVM::StoreOp::create(rewriter, loc, valueConverted, allocaOp);

    // Enable flag.
    Value enableVal;
    if (op.getEnable())
      enableVal = LLVM::ZExtOp::create(rewriter, loc, i8Ty, adaptor.getEnable());
    else
      enableVal = LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           rewriter.getIntegerAttr(i8Ty, 1));

    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
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

/// Lower llhd.constant_time → i64 constant (using our JIT delay encoding).
struct JITConstantTimeLowering
    : public OpConversionPattern<llhd::ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto timeAttr = op.getValueAttr();
    int64_t encoded = 0;
    if (auto time = llvm::dyn_cast<llhd::TimeAttr>(timeAttr)) {
      // Convert real-time component to femtoseconds.
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
      // "fs" or empty: already in fs

      encoded = encodeJITDelay(realFs, time.getDelta(), time.getEpsilon());
    }
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(encoded));
    return success();
  }
};

/// Lower llhd.int_to_time → identity (after type conversion, both are i64).
struct JITIntToTimeLowering
    : public OpConversionPattern<llhd::IntToTimeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::IntToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

/// Lower llhd.time_to_int → identity.
struct JITTimeToIntLowering
    : public OpConversionPattern<llhd::TimeToIntOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(llhd::TimeToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// JITBlockCompiler implementation
//===----------------------------------------------------------------------===//

JITBlockCompiler::JITBlockCompiler(MLIRContext &ctx) : mlirContext(ctx) {
  // Ensure all dialects needed for lowering are loaded.
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<cf::ControlFlowDialect>();
#ifdef CIRCT_SIM_JIT_ENABLED
  // Register LLVM IR translation interfaces for ExecutionEngine.
  registerBuiltinDialectTranslation(ctx);
  registerLLVMDialectTranslation(ctx);
  // Initialize native target so ORC JIT can generate code.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
#endif
}

JITBlockCompiler::~JITBlockCompiler() = default;

bool JITBlockCompiler::isBlockJITCompatible(Block *block) {
  for (auto &op : *block) {
    // Terminators (cf.br, cf.cond_br) are OK.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    // LLHD signal ops: prb and drv are JIT-compatible.
    if (isa<llhd::ProbeOp, llhd::DriveOp, llhd::ConstantTimeOp,
            llhd::IntToTimeOp, llhd::TimeToIntOp>(op))
      continue;
    // Comb ops.
    if (op.getDialect() &&
        op.getDialect()->getNamespace() == "comb")
      continue;
    // Arith ops.
    if (isa<arith::ArithDialect>(op.getDialect()))
      continue;
    // HW constants.
    if (isa<hw::ConstantOp>(op))
      continue;
    // LLVM memory ops — but reject stores through probed ptr signals
    // because probe values may be synthetic interpreter addresses.
    if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
      Value addr = storeOp.getAddr();
      llvm::SmallPtrSet<Value, 16> visited;
      if (dependsOnProbeValue(addr, visited)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[JIT] Incompatible: store address depends on probe\n");
          return false;
      }
      continue;
    }
    if (isa<LLVM::AllocaOp, LLVM::LoadOp, LLVM::GEPOp,
            LLVM::ConstantOp, LLVM::UndefOp, LLVM::AddressOfOp,
            LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp, LLVM::BitcastOp,
            LLVM::ICmpOp, LLVM::SelectOp>(op))
      continue;
    // CF control flow.
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      continue;
    // Anything else is NOT JIT-compatible.
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Incompatible op: " << op.getName()
                            << "\n");
    return false;
  }
  return true;
}

bool JITBlockCompiler::identifyHotBlock(
    llhd::ProcessOp processOp,
    const llvm::DenseMap<Value, SignalId> &signalIdMap,
    ProcessScheduler &scheduler, JITBlockSpec &spec) {
  ++stats.blocksAnalyzed;

  auto &region = processOp.getBody();
  if (region.empty())
    return false;

  // Pattern: entry → branch to loop header → wait → resume block (hot) → back.
  // Find the wait op that leads to the hot block.
  llhd::WaitOp candidateWait = nullptr;
  Block *hotBlock = nullptr;

  for (auto &block : region) {
    for (auto waitOp : block.getOps<llhd::WaitOp>()) {
      Block *dest = waitOp.getDest();
      if (!dest)
        continue;
      // The wait destination is the hot block candidate.
      // Check that it's a simple resume block (no phi args from wait).
      if (waitOp.getDestOperands().size() > 0)
        continue;
      candidateWait = waitOp;
      hotBlock = dest;
      break;
    }
    if (candidateWait)
      break;
  }

  if (!candidateWait || !hotBlock) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] No wait → resume pattern found\n");
    return false;
  }

  // Compile a linear resume block that loops back to a wait.
  // Two patterns are supported:
  //   1. Self-loop:  hotBlock: [ops] → llhd.wait → hotBlock
  //   2. Two-block:  waitBlock: [delay] → llhd.wait → hotBlock: [ops] → cf.br → waitBlock
  // The extracted native function drops the block terminator and re-enters
  // wait via interpreter logic.
  if (hotBlock->getNumArguments() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Hot block has block arguments\n");
    return false;
  }

  llhd::WaitOp loopWait = nullptr;
  if (auto hotWait = dyn_cast<llhd::WaitOp>(hotBlock->getTerminator())) {
    // Pattern 1: Self-loop — hot block ends with wait back to itself.
    if (hotWait.getDest() != hotBlock || !hotWait.getDestOperands().empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[JIT] Hot block wait is not a simple self-loop\n");
      return false;
    }
    loopWait = hotWait;
  } else if (auto hotBr =
                 dyn_cast<cf::BranchOp>(hotBlock->getTerminator())) {
    // Pattern 2: Two-block loop — hot block branches back to the wait block.
    Block *waitBlock = candidateWait->getBlock();
    if (hotBr.getDest() == waitBlock && hotBr.getDestOperands().empty()) {
      loopWait = candidateWait;
      LLVM_DEBUG(llvm::dbgs()
                 << "[JIT] Matched 2-block loop pattern\n");
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[JIT] Hot block branch does not loop back to wait block\n");
      return false;
    }
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "[JIT] Hot block terminator is not llhd.wait or cf.br\n");
    return false;
  }

  if (Value delayVal = loopWait.getDelay()) {
    bool supportedDelay = false;
    if (delayVal.getDefiningOp<llhd::ConstantTimeOp>()) {
      supportedDelay = true;
    } else if (auto intToTime = delayVal.getDefiningOp<llhd::IntToTimeOp>()) {
      Value input = intToTime.getOperand();
      supportedDelay = static_cast<bool>(
          input.getDefiningOp<arith::ConstantOp>()) ||
          static_cast<bool>(input.getDefiningOp<hw::ConstantOp>());
    }
    if (!supportedDelay) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[JIT] Hot block wait delay is not constant-shaped\n");
      return false;
    }
  }

  // Verify the hot block is JIT-compatible.
  if (!isBlockJITCompatible(hotBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Hot block not JIT-compatible\n");
    return false;
  }

  // Collect signal reads and drives.
  SmallVector<SignalId, 4> reads, drives;
  SmallVector<uint32_t, 4> readWidths, driveWidths;

  for (auto &op : *hotBlock) {
    if (auto probeOp = dyn_cast<llhd::ProbeOp>(op)) {
      if (!isJITSupportedSignalPayloadType(probeOp.getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[JIT] Unsupported probe payload type: "
                   << probeOp.getType() << "\n");
        return false;
      }
      auto it = signalIdMap.find(probeOp.getSignal());
      if (it == signalIdMap.end()) {
        LLVM_DEBUG(llvm::dbgs() << "[JIT] Signal not in map for probe\n");
        return false;
      }
      reads.push_back(it->second);
      readWidths.push_back(
          scheduler.getSignalValue(it->second).getWidth());
    } else if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
      if (!isJITSupportedSignalPayloadType(driveOp.getValue().getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[JIT] Unsupported drive payload type: "
                   << driveOp.getValue().getType() << "\n");
        return false;
      }
      auto it = signalIdMap.find(driveOp.getSignal());
      if (it == signalIdMap.end()) {
        LLVM_DEBUG(llvm::dbgs() << "[JIT] Signal not in map for drive\n");
        return false;
      }
      drives.push_back(it->second);
      driveWidths.push_back(
          scheduler.getSignalValue(it->second).getWidth());
    }
  }

  if (reads.empty() && drives.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Block has no signal I/O — skip\n");
    return false;
  }

  // Pre-resolve the drive delay (from the wait op's time or the drive's time).
  int64_t driveDelayEncoded = 0;
  for (auto &op : *hotBlock) {
    if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
      if (auto constTime =
              driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>()) {
        auto timeAttr = constTime.getValueAttr();
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
          driveDelayEncoded =
              encodeJITDelay(realFs, time.getDelta(), time.getEpsilon());
        }
      }
      break; // Use first drive's delay for now.
    }
  }

  spec.hotBlock = hotBlock;
  // Use the loop's wait op for post-JIT re-entry (either the hot block's own
  // self-loop wait or the wait in the loop header for 2-block patterns).
  spec.waitOp = loopWait;
  spec.signalReads = std::move(reads);
  spec.signalDrives = std::move(drives);
  spec.readWidths = std::move(readWidths);
  spec.driveWidths = std::move(driveWidths);
  spec.driveDelayEncoded = driveDelayEncoded;
  spec.funcName = "__jit_block_" + std::to_string(funcCounter++);

  ++stats.blocksEligible;
  LLVM_DEBUG(llvm::dbgs() << "[JIT] Hot block eligible: " << spec.funcName
                          << " reads=" << spec.signalReads.size()
                          << " drives=" << spec.signalDrives.size() << "\n");
  return true;
}

/// Build a micro-module with a single function extracted from the hot block.
///
/// The extracted function has signature:
///   func @__jit_block_N(%sig0: !llvm.ptr, %sig1: !llvm.ptr, ...) -> ()
///
/// Where each %sigI is a signal handle (SignalId encoded as ptr). The function
/// body contains the hot block's ops with signal refs replaced by arguments.
bool JITBlockCompiler::extractBlockFunction(
    JITBlockSpec &spec, ModuleOp microModule) {
  OpBuilder builder(&mlirContext);
  auto loc = spec.hotBlock->front().getLoc();

  // Build argument types: one ptr per signal (reads + drives), plus one i64
  // per drive (for delay).
  auto ptrTy = LLVM::LLVMPointerType::get(&mlirContext);
  SmallVector<Type> argTypes;
  // Read signal handles.
  for (size_t i = 0; i < spec.signalReads.size(); ++i)
    argTypes.push_back(ptrTy);
  // Drive signal handles.
  for (size_t i = 0; i < spec.signalDrives.size(); ++i)
    argTypes.push_back(ptrTy);

  auto funcType = builder.getFunctionType(argTypes, {});
  auto funcOp = func::FuncOp::create(loc, spec.funcName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Public);

  // Create the function body.
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Map original signal values to function arguments.
  IRMapping mapping;
  unsigned argIdx = 0;
  // Map read signals.
  llvm::DenseMap<Value, unsigned> readSignalArgIdx;
  for (auto &op : *spec.hotBlock) {
    if (auto probeOp = dyn_cast<llhd::ProbeOp>(op)) {
      if (argIdx < spec.signalReads.size()) {
        mapping.map(probeOp.getSignal(), entryBlock->getArgument(argIdx));
        readSignalArgIdx[probeOp.getSignal()] = argIdx;
        ++argIdx;
      }
    }
  }
  // Map drive signals.
  unsigned driveArgStart = spec.signalReads.size();
  unsigned driveIdx = 0;
  for (auto &op : *spec.hotBlock) {
    if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
      if (driveIdx < spec.signalDrives.size()) {
        mapping.map(driveOp.getSignal(),
                    entryBlock->getArgument(driveArgStart + driveIdx));
        ++driveIdx;
      }
    }
  }

  // Clone the ops from the hot block into the function, replacing signal refs.
  for (auto &op : *spec.hotBlock) {
    // Skip terminators — we'll add a return instead.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    // For constant_time ops defined OUTSIDE the hot block that are used by
    // drive ops inside it, we need to re-create them.
    for (auto operand : op.getOperands()) {
      if (!mapping.contains(operand)) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (isa<llhd::ConstantTimeOp>(defOp) ||
              isa<hw::ConstantOp>(defOp) ||
              isa<arith::ConstantOp>(defOp) ||
              isa<LLVM::ConstantOp>(defOp)) {
            builder.clone(*defOp, mapping);
          }
        }
      }
    }
    builder.clone(op, mapping);
  }

  // Add return terminator.
  func::ReturnOp::create(builder, loc);

  // Insert into micro-module.
  microModule.push_back(funcOp);
  return true;
}

bool JITBlockCompiler::lowerAndCompile(JITBlockSpec &spec,
                                       ModuleOp microModule) {
#ifndef CIRCT_SIM_JIT_ENABLED
  llvm::errs() << "[JIT] JIT not enabled — cannot compile block\n";
  return false;
#else
  auto startTime = std::chrono::steady_clock::now();

  // Set up the type converter and target.
  LLVMTypeConverter converter(&mlirContext);
  addHotBlockTypeConversions(converter);
  populateHWToLLVMTypeConversions(converter);

  LLVMConversionTarget target(mlirContext);
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<hw::HWDialect, comb::CombDialect,
                           llhd::LLHDDialect>();

  // Build the conversion patterns.
  RewritePatternSet patterns(&mlirContext);

  // Standard MLIR patterns.
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // CIRCT HW/Comb patterns.
  Namespace globals;
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggMap;
  std::optional<HWToLLVMArraySpillCache> spillCache = HWToLLVMArraySpillCache();
  {
    OpBuilder spillBuilder(microModule);
    spillCache->spillNonHWOps(spillBuilder, converter, microModule);
  }
  populateHWToLLVMConversionPatterns(converter, patterns, globals, constAggMap,
                                     spillCache);
  populateCombToArithConversionPatterns(converter, patterns);
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Hot-block LLHD patterns.
  patterns.add<JITProbeOpLowering, JITDriveOpLowering,
               JITConstantTimeLowering, JITIntToTimeLowering,
               JITTimeToIntLowering>(converter, &mlirContext);

  // Apply the conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(microModule, target, std::move(patterns),
                                    config))) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Conversion failed for "
                            << spec.funcName << "\n");
    LLVM_DEBUG(microModule.dump());
    ++stats.blocksFailed;
    return false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[JIT] Lowered LLVM IR module:\n";
    microModule.dump();
  });

  // JIT-compile via mlir::ExecutionEngine.
  ExecutionEngineOptions engineOpts;
  engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default; // O2
  engineOpts.transformer = makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  auto engineOrErr = ExecutionEngine::create(microModule, engineOpts);
  if (!engineOrErr) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] ExecutionEngine creation failed: "
                            << llvm::toString(engineOrErr.takeError()) << "\n");
    ++stats.blocksFailed;
    return false;
  }

  // Keep the engine alive before lookup so any returned function pointer stays
  // valid for the lifetime of this compiler instance.
  engines.push_back(std::move(*engineOrErr));
  auto *jitEngine = engines.back().get();

  // Register our runtime symbols.
  registerJITRuntimeSymbols(jitEngine);

  // Look up the compiled function.
  auto expectedFn = jitEngine->lookupPacked(spec.funcName);
  if (!expectedFn) {
    LLVM_DEBUG(llvm::dbgs() << "[JIT] Function lookup failed: "
                            << llvm::toString(expectedFn.takeError()) << "\n");
    engines.pop_back();
    ++stats.blocksFailed;
    return false;
  }

  spec.nativeFunc = reinterpret_cast<JITBlockSpec::NativeFuncTy>(*expectedFn);

  auto endTime = std::chrono::steady_clock::now();
  double compileMs = std::chrono::duration<double, std::milli>(
                         endTime - startTime)
                         .count();
  stats.totalCompileTimeMs += compileMs;
  ++stats.blocksCompiled;

  LLVM_DEBUG(llvm::dbgs() << "[JIT] Compiled " << spec.funcName << " in "
                          << llvm::format("%.1f", compileMs) << " ms\n");
  return true;
#endif
}

bool JITBlockCompiler::compileBlock(JITBlockSpec &spec,
                                    ModuleOp sourceModule) {
  // Create a micro-module containing just the extracted function.
  auto microModule = ModuleOp::create(
      UnknownLoc::get(&mlirContext));

  // Extract the hot block into a standalone function.
  if (!extractBlockFunction(spec, microModule)) {
    ++stats.blocksFailed;
    microModule.erase();
    return false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[JIT] Extracted micro-module:\n";
    microModule.dump();
  });

  // Lower and JIT-compile.
  bool ok = lowerAndCompile(spec, microModule);

  // Clean up the micro-module (we keep the compiled code via the engine).
  microModule.erase();
  return ok;
}
