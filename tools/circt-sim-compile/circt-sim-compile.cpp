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
/// LLVM dialect. Only arith/cf/func/LLVM ops with scalar types are accepted.
static bool isFuncBodyCompilable(func::FuncOp funcOp) {
  if (funcOp.isExternal())
    return false;

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

  for (auto argType : funcOp.getArgumentTypes())
    if (!isScalarType(argType))
      return false;
  for (auto resType : funcOp.getResultTypes())
    if (!isScalarType(resType))
      return false;

  bool compilable = true;
  funcOp.walk([&](Operation *op) {
    if (isa<arith::ArithDialect, cf::ControlFlowDialect, LLVM::LLVMDialect,
            func::FuncDialect>(op->getDialect()))
      return WalkResult::advance();
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

static llvm::SmallVector<std::string>
stripNonLLVMFunctions(ModuleOp microModule) {
  llvm::SmallVector<std::string> stripped;
  llvm::SmallVector<LLVM::LLVMFuncOp> toStrip;

  microModule.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      return;
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
    func.getBody().getBlocks().clear();
    func.setLinkage(LLVM::Linkage::External);
  }
  return stripped;
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

  // proc_kind array: all CIRCT_PROC_CALLBACK for now (func.func bodies).
  // Processes will be added in a later phase.
  // For now num_procs = 0.

  // Build the CirctSimCompiledModule struct.
  // Layout: {i32, i32, ptr, ptr, ptr, i32, ptr, ptr}
  auto *descriptorTy = llvm::StructType::create(
      ctx, {i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy},
      "CirctSimCompiledModule");

  auto *nullPtr = llvm::ConstantPointerNull::get(ptrTy);
  auto *descriptor = llvm::ConstantStruct::get(
      descriptorTy,
      {
          llvm::ConstantInt::get(i32Ty, CIRCT_SIM_ABI_VERSION), // abi_version
          llvm::ConstantInt::get(i32Ty, 0),                     // num_procs
          nullPtr,                                               // proc_names
          nullPtr,                                               // proc_kind
          nullPtr,                                               // proc_entry
          llvm::ConstantInt::get(i32Ty, numFuncs),              // num_funcs
          funcNamesGlobal,                                       // func_names
          funcEntryGlobal,                                       // func_entry
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
    global.setVisibility(llvm::GlobalValue::HiddenVisibility);
  }
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

  llvm::PassBuilder pb(tm);
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

  // Collect compilable func.func bodies.
  llvm::SmallVector<func::FuncOp> candidates;
  unsigned totalFuncs = 0, externalFuncs = 0, rejectedFuncs = 0;

  module->walk([&](func::FuncOp funcOp) {
    ++totalFuncs;
    if (funcOp.isExternal()) {
      ++externalFuncs;
      return;
    }
    if (!isFuncBodyCompilable(funcOp)) {
      ++rejectedFuncs;
      return;
    }
    candidates.push_back(funcOp);
  });

  llvm::errs() << "[circt-sim-compile] Functions: " << totalFuncs << " total, "
               << externalFuncs << " external, " << rejectedFuncs
               << " rejected, " << candidates.size() << " compilable\n";

  if (candidates.empty()) {
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
    funcNames.push_back(funcOp.getSymName().str());
  }

  cloneReferencedDeclarations(microModule, *module, mapping);

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

  // Strip functions with residual non-LLVM ops.
  auto stripped = stripNonLLVMFunctions(microModule);
  if (!stripped.empty()) {
    llvm::DenseSet<llvm::StringRef> strippedSet;
    for (const auto &name : stripped)
      strippedSet.insert(name);
    // Remove stripped names from the function list.
    llvm::SmallVector<std::string> survivingNames;
    for (auto &name : funcNames) {
      if (!strippedSet.contains(name))
        survivingNames.push_back(std::move(name));
    }
    funcNames = std::move(survivingNames);
    llvm::errs() << "[circt-sim-compile] Stripped " << stripped.size()
                 << " functions with non-LLVM ops\n";
  }

  llvm::errs() << "[circt-sim-compile] " << funcNames.size()
               << " functions ready for codegen\n";

  if (funcNames.empty()) {
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

  // Compute build ID.
  std::string buildId = "circt-sim-abi-v" +
                        std::to_string(CIRCT_SIM_ABI_VERSION) + "-" +
                        llvm::sys::getDefaultTargetTriple();

  // Synthesize descriptor and entrypoints into the LLVM IR module.
  synthesizeDescriptor(*llvmModule, funcNames, buildId);

  // Set hidden visibility on everything except entrypoints.
  setDefaultHiddenVisibility(*llvmModule);

  if (verbose) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    llvm::errs() << "[circt-sim-compile] translate: "
                 << std::chrono::duration<double>(elapsed).count() << "s\n";
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

  if (failed(emitObjectFile(*llvmModule, objPath, optLevel))) {
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

  auto elapsed = std::chrono::steady_clock::now() - startTime;
  llvm::errs() << "[circt-sim-compile] Wrote " << outPath << " ("
               << funcNames.size() << " functions, "
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
