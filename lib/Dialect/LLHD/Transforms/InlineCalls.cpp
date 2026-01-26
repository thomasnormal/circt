//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-inline-calls"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_INLINECALLSPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// UVM get_full_name() Runtime Support
//===----------------------------------------------------------------------===//
//
// UVM's get_full_name() is recursive and cannot be inlined. Instead of
// failing, we detect calls to get_full_name() on UVM component classes and
// replace them with calls to a runtime function that iteratively walks the
// parent chain.
//

/// Check if a function name matches the pattern for get_full_name on a
/// UVM-related class. The function name format from ImportVerilog is:
///   uvm_pkg::ClassName::get_full_name
static bool isUvmGetFullNameMethod(StringRef funcName) {
  // Check for common patterns of get_full_name in UVM classes
  // The function name typically contains the class name followed by
  // get_full_name
  if (!funcName.contains("get_full_name"))
    return false;

  // Check if it's from uvm_pkg (UVM classes)
  if (funcName.starts_with("uvm_pkg::"))
    return true;

  // Also check for common UVM component class names in the function name
  // These are the classes that typically have the recursive get_full_name
  static const char *uvmClasses[] = {"uvm_component",   "uvm_object",
                                     "uvm_report_object", "uvm_agent",
                                     "uvm_driver",      "uvm_monitor",
                                     "uvm_sequencer",   "uvm_env",
                                     "uvm_test",        "uvm_sequence",
                                     "uvm_reg",         "uvm_reg_block",
                                     "uvm_reg_map",     "uvm_reg_field",
                                     "uvm_mem"};

  for (const char *className : uvmClasses) {
    if (funcName.contains(className))
      return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// UVM Initialization Guarded Recursion Support
//===----------------------------------------------------------------------===//
//
// UVM's initialization code has a recursive call pattern that is guarded by
// state checks at runtime:
//   uvm_get_report_object -> uvm_coreservice_t::get -> uvm_init ->
//     (via uvm_fatal macro) -> uvm_get_report_object
//
// This recursion is safe at runtime because:
// 1. uvm_coreservice_t::get() checks if (inst==null) before calling uvm_init
// 2. uvm_init() checks get_core_state() and returns early if not uninitialized
// 3. The recursive path only happens in error/race conditions
//
// To handle this, we detect these UVM initialization functions and allow
// recursion through them, trusting the runtime guards to prevent infinite loops.
//

/// Check if a function is part of the UVM initialization recursive cycle.
/// These functions have guarded recursion that terminates at runtime.
///
/// The function name detection needs to be specific enough to avoid false
/// positives with non-UVM code, while still catching the mangled names
/// from ImportVerilog.
static bool isUvmInitializationFunction(StringRef funcName) {
  // The UVM initialization has several recursive call patterns:
  //
  // Pattern 1: Report object initialization
  //   uvm_get_report_object -> uvm_coreservice_t::get -> uvm_init
  //   -> (error path via uvm_fatal) -> uvm_get_report_object
  //
  // Pattern 2: Factory/Registry initialization
  //   uvm_coreservice_t::get -> uvm_init -> m_rh_init -> type_id::create
  //   -> create_by_type -> uvm_coreservice_t::get
  //
  // All these patterns are guarded by:
  // - uvm_coreservice_t::get() checks if (inst==null) before calling uvm_init
  // - uvm_init() checks get_core_state() and returns early if not uninitialized

  // Only match if the function is from uvm_pkg (UVM package)
  // This avoids false positives with non-UVM code
  if (!funcName.contains("uvm_pkg") && !funcName.contains("uvm_"))
    return false;

  // Check for uvm_coreservice_t::get
  if (funcName.contains("uvm_coreservice_t::get") ||
      funcName.contains("coreservice") ||
      // The mangled name may be "get_N" where N is a number, but only if
      // we're sure this is a UVM function (checked above)
      (funcName.contains("::get") && funcName.contains("uvm_")))
    return true;

  // List of UVM initialization functions involved in the recursive cycle
  static const char *initFunctions[] = {
      "uvm_get_report_object",  // Global function that calls coreservice::get
      "uvm_init",               // Main init function, guarded by state check
      "get_core_state",         // State check function
      // Report functions that can be called from uvm_init error paths
      "uvm_report_fatal",
      "uvm_report_error",
      "uvm_report_warning",
      "uvm_report_info",
      "uvm_report",
      "uvm_report_enabled",
      // Factory/Registry functions in the initialization chain
      "create_by_type",
      "m_rh_init",
      "type_id::create",
      "uvm_object_registry",
      "uvm_component_registry",
      "uvm_registry_object_creator",
      "uvm_registry_component_creator",
  };

  for (const char *initFunc : initFunctions) {
    if (funcName.contains(initFunc))
      return true;
  }

  return false;
}

/// Check if a function is a UVM constructor (`::new`), which can recurse
/// through package initialization and should not be inlined in --ir-hw.
static bool isUvmConstructorFunction(StringRef funcName) {
  if (!funcName.contains("uvm_pkg"))
    return false;
  return funcName.contains("::new");
}

/// Get or create the __moore_component_get_full_name runtime function
/// declaration in the module.
static LLVM::LLVMFuncOp
getOrCreateGetFullNameRuntimeFunc(ModuleOp mod, OpBuilder &builder) {
  const char *funcName = "__moore_component_get_full_name";

  // Check if the function already exists
  if (auto existingFn = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return existingFn;

  MLIRContext *ctx = mod.getContext();

  // Function signature:
  // MooreString __moore_component_get_full_name(void *component,
  //                                              int64_t parentOffset,
  //                                              int64_t nameOffset)
  // MooreString is {ptr, i64}
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  auto fnTy =
      LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, i64Ty, i64Ty});

  // Create the function declaration at the start of the module
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mod.getBody());

  auto fn = LLVM::LLVMFuncOp::create(builder, mod.getLoc(), funcName, fnTy);
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

/// Replace a call to get_full_name() with a call to the runtime function.
/// Returns true if the replacement was successful.
static bool replaceGetFullNameWithRuntimeCall(func::CallOp callOp,
                                               func::FuncOp funcOp,
                                               OpBuilder &builder) {
  // Get the module containing this call
  auto mod = callOp->getParentOfType<ModuleOp>();
  if (!mod)
    return false;

  // The call should have one operand (self/this pointer) and return a string
  if (callOp.getNumOperands() != 1 || callOp.getNumResults() != 1)
    return false;

  MLIRContext *ctx = builder.getContext();
  Location loc = callOp.getLoc();

  // Get or create the runtime function
  auto runtimeFn = getOrCreateGetFullNameRuntimeFunc(mod, builder);

  // Get types
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

  // The operand is the 'self' pointer (component instance)
  Value selfPtr = callOp.getOperand(0);

  // For now, we use hardcoded offsets based on the UVM component layout.
  // In a more sophisticated implementation, we would look up the class
  // definition to determine the actual field offsets.
  //
  // Typical UVM component layout (after vtable pointer):
  //   - vtable pointer (8 bytes on 64-bit)
  //   - m_name field (MooreString = {ptr, i64} = 16 bytes)
  //   - m_parent field (ptr = 8 bytes)
  //
  // However, the actual offsets depend on the specific class hierarchy.
  // We'll use reasonable defaults:
  //   - m_parent offset: 24 bytes (after vtable + m_name)
  //   - m_name offset: 8 bytes (after vtable)
  //
  // TODO: Compute actual offsets from class metadata or pass them as
  // additional call operands from the frontend.
  int64_t parentOffsetValue = 24; // Offset of m_parent field
  int64_t nameOffsetValue = 8;    // Offset of m_name field

  // Create constant offset values
  builder.setInsertionPoint(callOp);
  Value parentOffset =
      LLVM::ConstantOp::create(builder, loc, i64Ty, parentOffsetValue);
  Value nameOffset =
      LLVM::ConstantOp::create(builder, loc, i64Ty, nameOffsetValue);

  // Create the call to the runtime function
  auto runtimeCall = LLVM::CallOp::create(
      builder, loc, TypeRange{stringStructTy}, SymbolRefAttr::get(runtimeFn),
      ValueRange{selfPtr, parentOffset, nameOffset});

  // The result type might need conversion if the original call returned
  // a different string type representation
  Value result = runtimeCall.getResult();

  // Replace the original call with the runtime call result
  // Note: We need to handle potential type mismatches between the LLVM struct
  // and any hw::StructType or other representation the caller expects
  if (callOp.getResult(0).getType() != stringStructTy) {
    // Create an unrealized conversion cast if types don't match
    auto castOp = UnrealizedConversionCastOp::create(
        builder, loc, callOp.getResult(0).getType(), result);
    result = castOp.getResult(0);
  }

  callOp.getResult(0).replaceAllUsesWith(result);

  LLVM_DEBUG(llvm::dbgs() << "Replaced recursive get_full_name() call with "
                             "runtime function call\n");

  return true;
}

namespace {
/// Implementation of the `InlinerInterface` that allows calls in SSACFG regions
/// nested within `llhd.process`, `llhd.final`, and `llhd.combinational` ops to
/// be inlined.
struct FunctionInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    // Only inline `func.func` ops.
    if (!isa<func::FuncOp>(callable))
      return false;

    // Only inline into SSACFG regions embedded within procedural regions.
    if (!mayHaveSSADominance(*call->getParentRegion()))
      return false;
    if (call->getParentWithTrait<ProceduralRegion>())
      return true;
    return false;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }

  bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const override {
    return false;
  }
};

static LogicalResult inlineSingleBlockCall(func::CallOp callOp,
                                           func::FuncOp funcOp) {
  if (!funcOp || funcOp.empty())
    return failure();
  if (!funcOp.getBody().hasOneBlock())
    return failure();

  auto &calleeBlock = funcOp.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(calleeBlock.getTerminator());
  if (!returnOp)
    return failure();

  IRMapping mapping;
  for (auto [arg, operand] :
       llvm::zip(calleeBlock.getArguments(), callOp.getOperands()))
    mapping.map(arg, operand);

  OpBuilder builder(callOp);
  for (auto &op : calleeBlock.without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> results;
  results.reserve(returnOp.getNumOperands());
  for (auto value : returnOp.getOperands())
    results.push_back(mapping.lookup(value));

  if (!results.empty())
    callOp.replaceAllUsesWith(results);
  return success();
}

/// Pass implementation.
struct InlineCallsPass
    : public llhd::impl::InlineCallsPassBase<InlineCallsPass> {
  using CallStack = SmallSetVector<func::FuncOp, 8>;
  void runOnOperation() override;
  LogicalResult runOnRegion(Region &region, const SymbolTable &symbolTable,
                            CallStack &callStack);
};
} // namespace

void InlineCallsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
    CallStack callStack;
    if (failed(runOnRegion(module.getBody(), symbolTable, callStack))) {
      signalPassFailure();
      return;
    }
  }
}

LogicalResult InlineCallsPass::runOnRegion(Region &region,
                                           const SymbolTable &symbolTable,
                                           CallStack &callStack) {
  FunctionInliner inliner(&getContext());
  InlinerConfig config;
  SmallVector<Operation *> callsToErase;
  SmallVector<std::pair<Operation *, func::FuncOp>> inlineEndMarkers;

  // Walk all calls in the HW module and inline each. Emit a diagnostic if a
  // call does not target a `func.func` op or the inliner fails for some reason.
  // We use a custom version of `Operation::walk` here to ensure that we visit
  // the inlined operations immediately after visiting the call.
  for (auto &block : region) {
    for (auto &op : block) {
      // Pop all calls that are followed by this op off the call stack.
      while (!inlineEndMarkers.empty() &&
             inlineEndMarkers.back().first == &op) {
        assert(inlineEndMarkers.back().second == callStack.back());
        LLVM_DEBUG(llvm::dbgs()
                   << "- Finished @"
                   << inlineEndMarkers.back().second.getSymName() << "\n");
        inlineEndMarkers.pop_back();
        callStack.pop_back();
      }

      // Handle nested regions.
      for (auto &nestedRegion : op.getRegions())
        if (failed(runOnRegion(nestedRegion, symbolTable, callStack)))
          return failure();

      // We only care about calls.
      auto callOp = dyn_cast<func::CallOp>(op);
      if (!callOp)
        continue;

      // Make sure we're calling a `func.func`.
      auto symbol = callOp.getCalleeAttr();
      auto calledOp = symbolTable.lookup(symbol.getAttr());
      auto funcOp = dyn_cast<func::FuncOp>(calledOp);
      if (!funcOp) {
        auto d = callOp.emitError("function call cannot be inlined: call "
                                  "target is not a regular function");
        d.attachNote(calledOp->getLoc()) << "call target defined here";
        return failure();
      }
      if (funcOp.isExternal()) {
        LLVM_DEBUG(llvm::dbgs() << "- Skipping external function "
                                << funcOp.getSymName() << "\n");
        continue;
      }

      // Check if this is a UVM initialization function with guarded recursion.
      // These functions are part of UVM's initialization cycle and have runtime
      // guards to prevent infinite recursion. We skip inlining them entirely
      // to avoid infinite expansion during compilation.
      StringRef funcName = funcOp.getSymName();
      if (isUvmInitializationFunction(funcName)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping UVM init function " << funcName
                   << " (not inlining to avoid recursion)\n");
        // Don't inline - leave as function call for runtime resolution
        continue;
      }
      if (isUvmConstructorFunction(funcName)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping UVM constructor " << funcName
                   << " (not inlining to avoid recursion)\n");
        continue;
      }

      if (!callOp->getParentWithTrait<ProceduralRegion>() &&
          callOp->getParentRegion()->hasOneBlock()) {
        if (!callStack.insert(funcOp)) {
          // Check if this is a UVM get_full_name() call that we can handle
          // with the runtime function instead of failing
          if (isUvmGetFullNameMethod(funcName)) {
            OpBuilder builder(callOp);
            if (replaceGetFullNameWithRuntimeCall(callOp, funcOp, builder)) {
              callsToErase.push_back(callOp);
              continue;
            }
          }
          // Fall back to error for other recursive functions
          auto diag = callOp.emitError(
              "recursive function call cannot be inlined (unsupported in --ir-hw)");
          diag.attachNote(funcOp.getLoc())
              << "callee is " << funcOp.getSymName();
          return failure();
        }
        if (failed(inlineSingleBlockCall(callOp, funcOp)))
          return callOp.emitError(
              "function call cannot be inlined in this region");
        callStack.remove(funcOp);
        ++numInlined;
        callsToErase.push_back(callOp);
        continue;
      }

      // Ensure that we are not recursively inlining a function, which would
      // just expand infinitely in the IR.
      if (!callStack.insert(funcOp)) {
        // Check if this is a UVM get_full_name() call that we can handle
        // with the runtime function instead of failing
        if (isUvmGetFullNameMethod(funcName)) {
          OpBuilder builder(callOp);
          if (replaceGetFullNameWithRuntimeCall(callOp, funcOp, builder)) {
            callsToErase.push_back(callOp);
            continue;
          }
        }
        // Fall back to error for other recursive functions
        auto diag = callOp.emitError(
            "recursive function call cannot be inlined (unsupported in --ir-hw)");
        diag.attachNote(funcOp.getLoc())
            << "callee is " << funcOp.getSymName();
        return failure();
      }
      inlineEndMarkers.push_back({op.getNextNode(), funcOp});

      // Inline the function body and remember the call for later removal. The
      // `inlineCall` function will inline the function body *after* the call
      // op, which allows the loop to immediately visit the inlined ops and
      // handling nested calls.
      LLVM_DEBUG(llvm::dbgs() << "- Inlining " << callOp << "\n");
      if (failed(inlineCall(inliner, config.getCloneCallback(), callOp, funcOp,
                            funcOp.getCallableRegion())))
        return callOp.emitError("function call cannot be inlined");
      callsToErase.push_back(callOp);
      ++numInlined;
    }
  }

  // Erase all call ops that were successfully inlined.
  for (auto *callOp : callsToErase)
    callOp->erase();

  return success();
}
