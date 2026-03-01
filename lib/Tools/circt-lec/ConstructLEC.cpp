//===- ConstructLEC.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_CONSTRUCTLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// ConstructLEC pass
//===----------------------------------------------------------------------===//

namespace {
static constexpr llvm::StringLiteral kBMCAbstractedLLHDInterfaceInputsAttr =
    "circt.bmc_abstracted_llhd_interface_inputs";
static constexpr llvm::StringLiteral
    kLECSelectedAbstractedLLHDInterfaceInputsAttr =
        "circt.lec_selected_abstracted_llhd_interface_inputs";

struct ConstructLECPass
    : public circt::impl::ConstructLECBase<ConstructLECPass> {
  using circt::impl::ConstructLECBase<ConstructLECPass>::ConstructLECBase;
  void runOnOperation() override;
  hw::HWModuleOp lookupModule(StringRef name);
  FailureOr<Value> constructMiter(OpBuilder builder, Location loc,
                                  hw::HWModuleOp moduleA,
                                  hw::HWModuleOp moduleB,
                                  ArrayRef<StringAttr> inputNames,
                                  ArrayRef<Type> inputTypes, bool withResult);
};
} // namespace

static Value lookupOrCreateStringGlobal(OpBuilder &builder, ModuleOp moduleOp,
                                        StringRef str) {
  Location loc = moduleOp.getLoc();
  auto global = moduleOp.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        str, StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
  }

  // FIXME: sanity check the fetched global: do all the attributes match what
  // we expect?

  return LLVM::AddressOfOp::create(builder, loc, global);
}

hw::HWModuleOp ConstructLECPass::lookupModule(StringRef name) {
  Operation *expectedModule = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), name));
  if (!expectedModule || !isa<hw::HWModuleOp>(expectedModule)) {
    getOperation().emitError("module named '") << name << "' not found";
    return {};
  }
  return cast<hw::HWModuleOp>(expectedModule);
}

static bool portsMatch(ArrayRef<Attribute> aNames, ArrayRef<Type> aTypes,
                       ArrayRef<Attribute> bNames, ArrayRef<Type> bTypes) {
  if (aNames.size() != bNames.size() || aTypes.size() != bTypes.size())
    return false;
  if (aNames.size() != aTypes.size())
    return false;
  for (auto [aName, bName] : llvm::zip(aNames, bNames))
    if (aName != bName)
      return false;
  for (auto [aType, bType] : llvm::zip(aTypes, bTypes))
    if (aType != bType)
      return false;
  return true;
}

static LogicalResult collectInputPorts(hw::HWModuleOp module,
                                       SmallVectorImpl<StringAttr> &names,
                                       SmallVectorImpl<Type> &types,
                                       llvm::StringMap<Type> &typeByName) {
  auto inputNames = module.getInputNames();
  auto inputTypes = module.getInputTypes();
  if (inputNames.size() != inputTypes.size()) {
    module.emitError("input names/types size mismatch in module type");
    return failure();
  }
  llvm::StringSet<> seenInModule;
  for (auto [nameAttr, type] : llvm::zip(inputNames, inputTypes)) {
    auto name = dyn_cast<StringAttr>(nameAttr);
    if (!name) {
      module.emitError("expected string input port name, got ") << nameAttr;
      return failure();
    }
    StringRef nameStr = name.getValue();
    if (!seenInModule.insert(nameStr).second) {
      module.emitError("duplicate input port name in module type: ") << name;
      return failure();
    }
    names.push_back(name);
    types.push_back(type);
    typeByName.insert({nameStr, type});
  }
  return success();
}

static bool isInputSubset(const llvm::StringMap<Type> &subset,
                          const llvm::StringMap<Type> &superset) {
  for (const auto &it : subset) {
    auto found = superset.find(it.getKey());
    if (found == superset.end() || found->second != it.getValue())
      return false;
  }
  return true;
}

static LogicalResult cloneModuleToLECCircuit(OpBuilder &builder, Location loc,
                                             hw::HWModuleOp module,
                                             Region &circuit,
                                             ArrayRef<StringAttr> inputNames,
                                             ArrayRef<Type> inputTypes) {
  if (inputNames.size() != inputTypes.size()) {
    module.emitError("aligned LEC input names/types size mismatch");
    return failure();
  }
  auto *sourceBlock = module.getBodyBlock();
  if (!sourceBlock) {
    module.emitError("expected module body block");
    return failure();
  }

  auto *targetBlock = new Block();
  for (Type type : inputTypes)
    targetBlock->addArgument(type, loc);
  circuit.push_back(targetBlock);

  llvm::StringMap<BlockArgument> alignedArgsByName;
  for (auto [index, name] : llvm::enumerate(inputNames)) {
    if (!alignedArgsByName.insert({name.getValue(), targetBlock->getArgument(index)})
             .second) {
      module.emitError("duplicate aligned input name in LEC miter: ") << name;
      return failure();
    }
  }

  auto sourceInputNames = module.getInputNames();
  auto sourceInputTypes = module.getInputTypes();
  if (sourceInputNames.size() != sourceInputTypes.size() ||
      sourceInputNames.size() != sourceBlock->getNumArguments()) {
    module.emitError("module body arguments do not match module input ports");
    return failure();
  }

  IRMapping mapper;
  mapper.map(sourceBlock, targetBlock);
  for (auto [index, input] :
       llvm::enumerate(llvm::zip(sourceInputNames, sourceInputTypes))) {
    auto [nameAttr, type] = input;
    auto name = dyn_cast<StringAttr>(nameAttr);
    if (!name) {
      module.emitError("expected string input port name, got ") << nameAttr;
      return failure();
    }
    auto found = alignedArgsByName.find(name.getValue());
    if (found == alignedArgsByName.end()) {
      module.emitError("missing aligned input for module input: ") << name;
      return failure();
    }
    if (found->second.getType() != type) {
      module.emitError("aligned input type mismatch for port ")
          << name << ": " << found->second.getType() << " vs " << type;
      return failure();
    }
    mapper.map(sourceBlock->getArgument(index), found->second);
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(targetBlock);
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto &op : sourceBlock->without_terminator()) {
    Operation *newOp = op.clone(mapper, cloneOptions);
    targetBlock->push_back(newOp);
    clonedOps.emplace_back(&op, newOp);
  }

  SmallVector<Value> remappedOperands;
  for (auto [oldOp, newOp] : clonedOps) {
    remappedOperands.resize(oldOp->getNumOperands());
    llvm::transform(oldOp->getOperands(), remappedOperands.begin(),
                    [&](Value operand) { return mapper.lookupOrDefault(operand); });
    newOp->setOperands(remappedOperands);
    for (auto [oldRegion, newRegion] :
         llvm::zip(oldOp->getRegions(), newOp->getRegions()))
      oldRegion.cloneInto(&newRegion, mapper);
  }

  auto *sourceTerminator = sourceBlock->getTerminator();
  SmallVector<Value> yieldedValues;
  yieldedValues.reserve(sourceTerminator->getNumOperands());
  for (Value operand : sourceTerminator->getOperands())
    yieldedValues.push_back(mapper.lookupOrDefault(operand));
  verif::YieldOp::create(builder, loc, yieldedValues);

  return success();
}

FailureOr<Value> ConstructLECPass::constructMiter(
    OpBuilder builder, Location loc, hw::HWModuleOp moduleA,
    hw::HWModuleOp moduleB, ArrayRef<StringAttr> inputNames,
    ArrayRef<Type> inputTypes, bool withResult) {

  // Create the miter circuit that return equivalence result.
  auto lecOp =
      verif::LogicEquivalenceCheckingOp::create(builder, loc, withResult);

  if (failed(cloneModuleToLECCircuit(builder, loc, moduleA,
                                     lecOp.getFirstCircuit(), inputNames,
                                     inputTypes)) ||
      failed(cloneModuleToLECCircuit(builder, loc, moduleB,
                                     lecOp.getSecondCircuit(), inputNames,
                                     inputTypes))) {
    lecOp->erase();
    return failure();
  }

  if (!inputNames.empty()) {
    SmallVector<Attribute> nameAttrs(inputNames.begin(), inputNames.end());
    lecOp->setAttr("lec.input_names", builder.getArrayAttr(nameAttrs));
  }
  if (!inputTypes.empty()) {
    SmallVector<Attribute> typeAttrs;
    typeAttrs.reserve(inputTypes.size());
    for (Type type : inputTypes)
      typeAttrs.push_back(TypeAttr::get(type));
    lecOp->setAttr("lec.input_types", builder.getArrayAttr(typeAttrs));
  }

  moduleA->erase();
  if (moduleA != moduleB)
    moduleB->erase();

  return withResult ? FailureOr<Value>(lecOp.getIsProven())
                    : FailureOr<Value>(Value{});
}

void ConstructLECPass::runOnOperation() {
  // Create necessary function declarations and globals
  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
  Location loc = getOperation()->getLoc();

  // Lookup the modules.
  auto moduleA = lookupModule(firstModule);
  if (!moduleA)
    return signalPassFailure();
  auto moduleB = lookupModule(secondModule);
  if (!moduleB)
    return signalPassFailure();

  // Track LLHD abstraction only for the compared circuits. The strip pass also
  // records whole-module totals; using selected-module totals avoids
  // misclassifying unrelated SAT mismatches as LLHD abstraction inconclusive.
  auto getAbstractedInterfaceInputCount = [&](hw::HWModuleOp module) -> int64_t {
    if (!module)
      return 0;
    if (auto countAttr = module->getAttrOfType<IntegerAttr>(
            kBMCAbstractedLLHDInterfaceInputsAttr))
      return countAttr.getInt();
    return 0;
  };
  int64_t selectedAbstractedInputCount =
      getAbstractedInterfaceInputCount(moduleA) +
      getAbstractedInterfaceInputCount(moduleB);
  getOperation()->setAttr(
      kLECSelectedAbstractedLLHDInterfaceInputsAttr,
      IntegerAttr::get(IntegerType::get(&getContext(), 32),
                       selectedAbstractedInputCount));

  SmallVector<StringAttr> moduleAInputNames, moduleBInputNames;
  SmallVector<Type> moduleAInputTypes, moduleBInputTypes;
  llvm::StringMap<Type> moduleAInputTypeByName, moduleBInputTypeByName;
  if (failed(collectInputPorts(moduleA, moduleAInputNames, moduleAInputTypes,
                               moduleAInputTypeByName)) ||
      failed(collectInputPorts(moduleB, moduleBInputNames, moduleBInputTypes,
                               moduleBInputTypeByName))) {
    return signalPassFailure();
  }

  SmallVector<StringAttr> alignedInputNames = moduleAInputNames;
  SmallVector<Type> alignedInputTypes = moduleAInputTypes;

  if (moduleA.getModuleType() != moduleB.getModuleType()) {
    if (allowIOAlignment) {
      auto outputsA = moduleA.getModuleType().getOutputNames();
      auto outputsATypes = moduleA.getModuleType().getOutputTypes();
      auto outputsB = moduleB.getModuleType().getOutputNames();
      auto outputsBTypes = moduleB.getModuleType().getOutputTypes();
      bool aSubsetB = isInputSubset(moduleAInputTypeByName, moduleBInputTypeByName);
      bool bSubsetA = isInputSubset(moduleBInputTypeByName, moduleAInputTypeByName);
      if (!portsMatch(outputsA, outputsATypes, outputsB, outputsBTypes) ||
          (!aSubsetB && !bSubsetA)) {
        moduleA.emitError("module's IO types don't match second modules: ")
            << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
        return signalPassFailure();
      }

      llvm::StringSet<> seenAligned;
      for (auto name : alignedInputNames)
        seenAligned.insert(name.getValue());
      for (auto [name, type] : llvm::zip(moduleBInputNames, moduleBInputTypes)) {
        if (seenAligned.insert(name.getValue()).second) {
          alignedInputNames.push_back(name);
          alignedInputTypes.push_back(type);
        }
      }
    } else {
      moduleA.emitError("module's IO types don't match second modules: ")
          << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
      return signalPassFailure();
    }
  }

  // Only construct the miter with no additional insertions.
  if (insertMode == lec::InsertAdditionalModeEnum::None) {
    if (failed(constructMiter(builder, loc, moduleA, moduleB, alignedInputNames,
                              alignedInputTypes, /*withResult*/ false)))
      return signalPassFailure();
    return;
  }

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> printfFunc;
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  // Lookup or declare printf function.
  printfFunc = LLVM::lookupOrCreateFn(builder, getOperation(), "printf", ptrTy,
                                      voidTy, true);
  if (failed(printfFunc)) {
    getOperation()->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Reuse the name of the first module for the entry function, so we don't
  // have to do any uniquing and the LEC driver also already knows this name.
  FunctionType functionType = FunctionType::get(&getContext(), {}, {});
  func::FuncOp entryFunc =
      func::FuncOp::create(builder, loc, firstModule, functionType);

  if (insertMode == lec::InsertAdditionalModeEnum::Main) {
    OpBuilder::InsertionGuard guard(builder);
    auto i32Ty = builder.getI32Type();
    auto mainFunc = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    func::CallOp::create(builder, loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
    func::ReturnOp::create(builder, loc, constZero);
  }

  builder.createBlock(&entryFunc.getBody());

  // Create the miter circuit that returns equivalence result.
  auto areEquivalent =
      constructMiter(builder, loc, moduleA, moduleB, alignedInputNames,
                     alignedInputTypes, /*withResult*/ true);
  if (failed(areEquivalent))
    return signalPassFailure();
  assert(*areEquivalent && "Expected LEC operation with result.");

  // TODO: we should find a more elegant way of reporting the result than
  // already inserting some LLVM here
  Value eqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 == c2\n");
  Value neqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 != c2\n");
  Value formatString = LLVM::SelectOp::create(builder, loc, *areEquivalent,
                                              eqFormatString, neqFormatString);
  LLVM::CallOp::create(builder, loc, printfFunc.value(),
                       ValueRange{formatString});

  func::ReturnOp::create(builder, loc, ValueRange{});
}
