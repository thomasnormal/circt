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

#include "circt/Runtime/CirctSimABI.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/FourStateUtils.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
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
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
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

//===----------------------------------------------------------------------===//
// Lowering: func compilability check
//===----------------------------------------------------------------------===//

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
            func::FuncDialect>(op->getDialect()))
      return WalkResult::advance();
    if (rejectionReason)
      *rejectionReason = op->getName().getStringRef().str();
    compilable = false;
    return WalkResult::interrupt();
  });
  return compilable;
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
          if (!clonedFunc.getBody().empty())
            clonedFunc.getBody().getBlocks().clear();
          auto linkage = clonedFunc.getLinkage();
          if (linkage != LLVM::Linkage::External &&
              linkage != LLVM::Linkage::ExternWeak)
            clonedFunc.setLinkage(LLVM::Linkage::External);
        }
        changed = true;
      } else if (auto funcFunc = dyn_cast<func::FuncOp>(srcOp)) {
        func::FuncOp::create(builder, funcFunc.getLoc(), funcFunc.getSymName(),
                             funcFunc.getFunctionType());
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

  // Phase 1: Rewrite arith/cf/func ops to LLVM equivalents.
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
              extractionFailed = true;
              break;
            }
          } else {
            extractionFailed = true;
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
      funcOp.erase();
      continue;
    }

    // Clone and lower ops from body blocks.
    bool loweringFailed = false;
    for (Block *srcBlock : bodyBlocks) {
      Block *dstBlock = blockMap[srcBlock];
      builder.setInsertionPointToEnd(dstBlock);

      for (Operation &op : *srcBlock) {
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
                                 const std::string &buildId) {
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

  // Build the CirctSimCompiledModule struct.
  // Layout: {i32, i32, ptr, ptr, ptr, i32, ptr, ptr, i32, ptr}
  auto *descriptorTy = llvm::StructType::create(
      ctx,
      {i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy, i32Ty, ptrTy},
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
        func.getName() == "circt_sim_get_build_id")
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

static LogicalResult emitObjectFile(llvm::Module &llvmModule,
                                    llvm::StringRef outputPath, int optLvl) {
  auto targetTripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple targetTriple(targetTripleStr);
  llvmModule.setTargetTriple(targetTriple);

  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Error looking up target: " << error << "\n";
    return failure();
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
    return failure();
  }

  llvmModule.setDataLayout(targetMachine->createDataLayout());

  // Run LLVM optimization passes.
  runOptimizationPasses(llvmModule, targetMachine, optLvl);

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

  cloneReferencedDeclarations(microModule, *module, mapping);

  // Strip bodies of UVM-intercepted functions so they become external
  // declarations → trampolines. This ensures compiled code calls back into
  // the interpreter for these functions, where UVM interceptors can fire.
  // Without this, compiled→compiled direct calls bypass interceptors and
  // cause crashes (e.g., factory returning null, config_db not firing).
  //
  // The pattern matching here must be conservative: any function that MIGHT
  // have an interpreter interceptor should be demoted. False positives just
  // mean the function goes through the interpreter (slower but correct).
  // False negatives cause crashes.
  {
    auto isInterceptedFunc = [](llvm::StringRef name) -> bool {
      // All uvm_ functions (factory, config_db, phasing, sequencer, etc.)
      if (name.contains("uvm_"))
        return true;
      // Moore runtime support (__moore_delay, __moore_string_*, etc.)
      if (name.starts_with("__moore_"))
        return true;
      // Standalone intercepted functions (short names without uvm_ prefix)
      if (name == "get_0" || name == "self" || name == "malloc" ||
          name == "get_common_domain" || name == "get_global_hopper")
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
      // UVM-generated class methods with package-qualified names (no "uvm_"
      // prefix). Emitted by uvm_component_utils/uvm_object_utils macros.
      if (name == "create" || name.starts_with("create_") ||
          name.ends_with("::create"))
        return true;
      if (name == "m_initialize" || name.starts_with("m_initialize_") ||
          name.ends_with("::m_initialize"))
        return true;
      if (name.starts_with("m_register_cb"))
        return true;
      if (name.contains("get_object_type") || name.contains("get_type_name"))
        return true;
      // Type registry accessors (get_type_NNN mangled names)
      if (name.starts_with("get_type_") || name.contains("::get_type_"))
        return true;
      // TLM imp port accessors (get_imp_NNN mangled names)
      if (name.starts_with("get_imp_") || name.contains("::get_imp_"))
        return true;
      // Type name string helpers (type_name_NNN)
      if (name.starts_with("type_name_"))
        return true;
      // Constructors (may allocate objects the interpreter tracks)
      if (name.ends_with("::new"))
        return true;
      // UVM singleton/registry accessors
      if (name.contains("::is_auditing") ||
          name.contains("get_print_config_matches") ||
          name.contains("get_root_blocks") ||
          name == "get_inst" || name.starts_with("get_inst_"))
        return true;
      return false;
    };

    unsigned demoted = 0;
    llvm::SmallVector<func::FuncOp> toDemote;
    microModule.walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      if (isInterceptedFunc(funcOp.getSymName()))
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

  // Generate trampolines for uncompiled functions referenced by compiled code.
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

  // Compute build ID: encode ABI version, target triple, and a content hash of
  // the input file so that stale cached .so files can be detected at load time.
  std::string buildId = "circt-sim-abi-v" +
                        std::to_string(CIRCT_SIM_ABI_VERSION) + "-" +
                        llvm::sys::getDefaultTargetTriple() + "-" +
                        contentHash;

  // Synthesize descriptor and entrypoints into the LLVM IR module.
  synthesizeDescriptor(*llvmModule, funcNames, trampolineNames,
                       procNames, procKinds, buildId);

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
  std::string outPath = outputFilename;
  if (outPath.empty()) {
    if (inputFilename == "-") {
      outPath = "a.out.so";
    } else {
      llvm::SmallString<256> p(inputFilename);
      llvm::sys::path::replace_extension(p, ".so");
      outPath = std::string(p);
    }
  }

  // Emit LLVM IR if requested.
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

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

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

  ModuleStats preStats;
  if (verbose) {
    preStats = collectModuleStats(*llvmModule);
    llvm::errs() << "[circt-sim-compile] pre-opt: " << preStats.definedFnCount
                 << " functions (" << preStats.internalFnCount
                 << " internal), " << preStats.callCount << " calls\n";
  }

  if (failed(emitObjectFile(*llvmModule, objPath, optLevel))) {
    llvm::errs() << "[circt-sim-compile] Object emission failed\n";
    return failure();
  }

  if (verbose) {
    auto postStats = collectModuleStats(*llvmModule);
    unsigned dce = preStats.internalFnCount > postStats.internalFnCount
                       ? preStats.internalFnCount - postStats.internalFnCount
                       : 0;
    llvm::errs() << "[circt-sim-compile] post-opt: " << postStats.definedFnCount
                 << " functions (" << postStats.internalFnCount << " internal, "
                 << dce << " DCE'd), " << postStats.callCount << " calls\n";
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
