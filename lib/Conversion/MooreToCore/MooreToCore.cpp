//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTMOORETOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

using comb::ICmpPredicate;
using llvm::SmallDenseSet;

/// Helper function to convert any type to a pure LLVM type for use in class
/// struct bodies. The regular type converter produces hw::StructType and
/// hw::ArrayType for some Moore types, but these don't have LLVM DataLayout
/// information, which causes crashes when computing class sizes.
/// This function recursively converts hw::StructType -> LLVM::LLVMStructType
/// and hw::ArrayType -> LLVM::LLVMArrayType.
static Type convertToLLVMType(Type type) {
  MLIRContext *ctx = type.getContext();

  // Handle hw::StructType -> LLVM::LLVMStructType
  if (auto hwStructTy = dyn_cast<hw::StructType>(type)) {
    SmallVector<Type> elementTypes;
    for (auto field : hwStructTy.getElements()) {
      Type convertedField = convertToLLVMType(field.type);
      elementTypes.push_back(convertedField);
    }
    return LLVM::LLVMStructType::getLiteral(ctx, elementTypes);
  }

  // Handle hw::ArrayType -> LLVM::LLVMArrayType
  if (auto hwArrayTy = dyn_cast<hw::ArrayType>(type)) {
    Type elementType = convertToLLVMType(hwArrayTy.getElementType());
    return LLVM::LLVMArrayType::get(elementType, hwArrayTy.getNumElements());
  }

  // Handle hw::UnionType -> LLVM array of bytes (largest element size)
  if (auto hwUnionTy = dyn_cast<hw::UnionType>(type)) {
    // For unions, we use the bit width to create an array of i8
    int64_t bitWidth = hw::getBitWidth(hwUnionTy);
    if (bitWidth > 0) {
      int64_t byteWidth = (bitWidth + 7) / 8;
      auto i8Ty = IntegerType::get(ctx, 8);
      return LLVM::LLVMArrayType::get(i8Ty, byteWidth);
    }
    // Fallback to i8 for empty unions
    return IntegerType::get(ctx, 8);
  }

  // Handle nested LLVM struct types (may contain hw types in fields)
  if (auto llvmStructTy = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (llvmStructTy.isIdentified()) {
      // For identified structs, return as-is (they should already be pure LLVM)
      return type;
    }
    // For literal structs, convert each element
    SmallVector<Type> elementTypes;
    for (Type elemTy : llvmStructTy.getBody()) {
      elementTypes.push_back(convertToLLVMType(elemTy));
    }
    return LLVM::LLVMStructType::getLiteral(ctx, elementTypes);
  }

  // Handle LLVM array types (may contain hw types as elements)
  if (auto llvmArrayTy = dyn_cast<LLVM::LLVMArrayType>(type)) {
    Type elementType = convertToLLVMType(llvmArrayTy.getElementType());
    return LLVM::LLVMArrayType::get(elementType, llvmArrayTy.getNumElements());
  }

  // Handle llhd::TimeType -> LLVM struct {i64 realTime, i32 delta, i32 epsilon}
  // This is needed because llhd::TimeType doesn't have DataLayout info.
  if (isa<llhd::TimeType>(type)) {
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    return LLVM::LLVMStructType::getLiteral(ctx, {i64Ty, i32Ty, i32Ty});
  }

  // All other types (IntegerType, LLVM::LLVMPointerType, etc.) pass through
  return type;
}

namespace {

/// Cache for identified structs and field GEP paths keyed by class symbol.
struct ClassTypeCache {
  struct ClassStructInfo {
    LLVM::LLVMStructType classBody;

    // field name -> GEP path inside ident (excluding the leading pointer index)
    DenseMap<StringRef, SmallVector<unsigned, 2>> propertyPath;

    // Type ID for RTTI support (used by $cast for dynamic type checking).
    // Each class gets a unique type ID assigned in topological order (base
    // classes get lower IDs than derived classes). The type ID is stored
    // as the first field in root class structs; derived classes access it
    // through their base class prefix.
    int32_t typeId = 0;

    // Inheritance depth (0 for root classes, 1 for direct derived, etc.)
    int32_t inheritanceDepth = 0;

    // TODO: Add classVTable in here.
    /// Record/overwrite the field path to a single property for a class.
    void setFieldPath(StringRef propertyName, ArrayRef<unsigned> path) {
      this->propertyPath[propertyName] =
          SmallVector<unsigned, 2>(path.begin(), path.end());
    }

    /// Lookup the full GEP path for a (class, field).
    std::optional<ArrayRef<unsigned>>
    getFieldPath(StringRef propertySym) const {
      if (auto prop = this->propertyPath.find(propertySym);
          prop != this->propertyPath.end())
        return ArrayRef<unsigned>(prop->second);
      return std::nullopt;
    }
  };

  // Counter for assigning unique type IDs to classes.
  // Type IDs are assigned in resolution order (base classes first).
  int32_t nextTypeId = 1;

  /// Allocate and return the next unique type ID for a class.
  int32_t allocateTypeId() { return nextTypeId++; }

  /// Record the identified struct body for a class.
  /// Implicitly finalizes the class to struct conversion.
  void setClassInfo(SymbolRefAttr classSym, const ClassStructInfo &info) {
    auto &dst = classToStructMap[classSym];
    dst = info;
  }

  /// Lookup the identified struct body for a class.
  std::optional<ClassStructInfo> getStructInfo(SymbolRefAttr classSym) const {
    if (auto it = classToStructMap.find(classSym); it != classToStructMap.end())
      return it->second;
    return std::nullopt;
  }

private:
  // Keyed by the SymbolRefAttr of the class.
  // Kept private so all accesses are done with helpers which preserve
  // invariants
  DenseMap<Attribute, ClassStructInfo> classToStructMap;
};

/// Cache for identified structs and signal GEP paths keyed by interface symbol.
struct InterfaceTypeCache {
  struct InterfaceStructInfo {
    LLVM::LLVMStructType interfaceBody;

    // signal name -> GEP index inside the struct
    DenseMap<StringRef, unsigned> signalIndex;

    /// Record the index for a signal.
    void setSignalIndex(StringRef signalName, unsigned index) {
      this->signalIndex[signalName] = index;
    }

    /// Lookup the GEP index for a signal.
    std::optional<unsigned> getSignalIndex(StringRef signalName) const {
      if (auto it = this->signalIndex.find(signalName);
          it != this->signalIndex.end())
        return it->second;
      return std::nullopt;
    }
  };

  /// Record the identified struct body for an interface.
  void setInterfaceInfo(SymbolRefAttr ifaceSym,
                        const InterfaceStructInfo &info) {
    auto &dst = interfaceToStructMap[ifaceSym];
    dst = info;
  }

  /// Lookup the identified struct body for an interface.
  std::optional<InterfaceStructInfo>
  getStructInfo(SymbolRefAttr ifaceSym) const {
    if (auto it = interfaceToStructMap.find(ifaceSym);
        it != interfaceToStructMap.end())
      return it->second;
    return std::nullopt;
  }

private:
  DenseMap<Attribute, InterfaceStructInfo> interfaceToStructMap;
};

/// Ensure we have `declare i8* @malloc(i64)` (opaque ptr prints as !llvm.ptr).
static LLVM::LLVMFuncOp getOrCreateMalloc(ModuleOp mod, OpBuilder &b) {
  if (auto f = mod.lookupSymbol<LLVM::LLVMFuncOp>("malloc"))
    return f;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(mod.getBody());

  auto i64Ty = IntegerType::get(mod.getContext(), 64);
  auto ptrTy = LLVM::LLVMPointerType::get(mod.getContext()); // opaque pointer
  auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {i64Ty}, false);

  auto fn = LLVM::LLVMFuncOp::create(b, mod.getLoc(), "malloc", fnTy);
  // Link this in from somewhere else.
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

/// Helper to get or create an external runtime function declaration.
static LLVM::LLVMFuncOp getOrCreateRuntimeFunc(ModuleOp mod, OpBuilder &b,
                                                StringRef name,
                                                LLVM::LLVMFunctionType fnTy) {
  if (auto f = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return f;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(mod.getBody());

  auto fn = LLVM::LLVMFuncOp::create(b, mod.getLoc(), name, fnTy);
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

/// Helper function to create an opaque LLVM Struct Type which corresponds
/// to the sym
static LLVM::LLVMStructType getOrCreateOpaqueStruct(MLIRContext *ctx,
                                                    SymbolRefAttr className) {
  return LLVM::LLVMStructType::getIdentified(ctx, className.getRootReference());
}

static LogicalResult resolveClassStructBody(ClassDeclOp op,
                                            TypeConverter const &typeConverter,
                                            ClassTypeCache &cache) {

  auto classSym = SymbolRefAttr::get(op.getSymNameAttr());
  auto structInfo = cache.getStructInfo(classSym);
  if (structInfo)
    // We already have a resolved class struct body.
    return success();

  // Otherwise we need to resolve.
  ClassTypeCache::ClassStructInfo structBody;
  SmallVector<Type> structBodyMembers;

  // Base-first (prefix) layout for single inheritance.
  unsigned derivedStartIdx = 0;
  MLIRContext *ctx = op.getContext();

  if (auto baseClass = op.getBaseAttr()) {

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto *opSym = mod.lookupSymbol(baseClass);
    auto classDeclOp = cast<ClassDeclOp>(opSym);

    if (failed(resolveClassStructBody(classDeclOp, typeConverter, cache)))
      return failure();

    // Process base class' struct layout first
    auto baseClassStruct = cache.getStructInfo(baseClass);
    structBodyMembers.push_back(baseClassStruct->classBody);
    derivedStartIdx = 1;

    // Inherit base field paths with a leading 0.
    for (auto &kv : baseClassStruct->propertyPath) {
      SmallVector<unsigned, 2> path;
      path.push_back(0); // into base subobject
      path.append(kv.second.begin(), kv.second.end());
      structBody.setFieldPath(kv.first, path);
    }

    // Derived class inherits from base, so increment depth
    structBody.inheritanceDepth = baseClassStruct->inheritanceDepth + 1;
  } else {
    // Root class: add type ID field as the first member.
    // This field stores the runtime type ID for RTTI support ($cast).
    // Layout: { i32 typeId, ... properties ... }
    auto i32Ty = IntegerType::get(ctx, 32);
    structBodyMembers.push_back(i32Ty);
    derivedStartIdx = 1; // Properties start after typeId field

    // Root class has inheritance depth 0
    structBody.inheritanceDepth = 0;
  }

  // Assign a unique type ID to this class
  structBody.typeId = cache.allocateTypeId();

  // Properties in source order.
  unsigned iterator = derivedStartIdx;
  auto &block = op.getBody().front();
  for (Operation &child : block) {
    if (auto prop = dyn_cast<ClassPropertyDeclOp>(child)) {
      Type mooreTy = prop.getPropertyType();
      Type convertedTy = typeConverter.convertType(mooreTy);
      if (!convertedTy)
        return prop.emitOpError()
               << "failed to convert property type " << mooreTy;

      // Convert to pure LLVM type for DataLayout compatibility.
      // The type converter may produce hw::StructType or hw::ArrayType,
      // which don't have DataLayout info, causing crashes when computing
      // class sizes for malloc.
      Type llvmTy = convertToLLVMType(convertedTy);
      structBodyMembers.push_back(llvmTy);

      // Derived field path: either {i} or {1+i} if base is present.
      SmallVector<unsigned, 2> path{iterator};
      structBody.setFieldPath(prop.getSymName(), path);
      ++iterator;
    }
  }

  // TODO: Handle vtable generation over ClassMethodDeclOp here.
  auto llvmStructTy = getOrCreateOpaqueStruct(ctx, classSym);
  // Empty structs may be kept opaque
  if (!structBodyMembers.empty() &&
      failed(llvmStructTy.setBody(structBodyMembers, false)))
    return op.emitOpError() << "Failed to set LLVM Struct body";

  structBody.classBody = llvmStructTy;
  cache.setClassInfo(classSym, structBody);

  return success();
}

/// Convenience overload that looks up ClassDeclOp
static LogicalResult resolveClassStructBody(ModuleOp mod, SymbolRefAttr op,
                                            TypeConverter const &typeConverter,
                                            ClassTypeCache &cache) {
  auto classDeclOp = cast<ClassDeclOp>(*mod.lookupSymbol(op));
  return resolveClassStructBody(classDeclOp, typeConverter, cache);
}

/// Resolve the struct body for an interface declaration.
/// This creates an LLVM struct type with fields for each signal in the
/// interface and caches the signal name to GEP index mapping.
static LogicalResult resolveInterfaceStructBody(InterfaceDeclOp op,
                                                TypeConverter const &typeConverter,
                                                InterfaceTypeCache &cache) {
  auto ifaceSym = SymbolRefAttr::get(op.getSymNameAttr());
  auto structInfo = cache.getStructInfo(ifaceSym);
  if (structInfo)
    // We already have a resolved interface struct body.
    return success();

  // Otherwise we need to resolve.
  InterfaceTypeCache::InterfaceStructInfo structBody;
  SmallVector<Type> structBodyMembers;

  // Iterate over all signals in the interface.
  unsigned idx = 0;
  auto &block = op.getBody().front();
  for (Operation &child : block) {
    if (auto signal = dyn_cast<InterfaceSignalDeclOp>(child)) {
      Type mooreTy = signal.getSignalType();
      Type convertedTy = typeConverter.convertType(mooreTy);
      if (!convertedTy)
        return signal.emitOpError()
               << "failed to convert signal type " << mooreTy;

      // Convert to pure LLVM type for DataLayout compatibility.
      Type llvmTy = convertToLLVMType(convertedTy);
      structBodyMembers.push_back(llvmTy);
      structBody.setSignalIndex(signal.getSymName(), idx);
      ++idx;
    }
  }

  // Create the identified struct type for this interface.
  auto llvmStructTy = LLVM::LLVMStructType::getIdentified(
      op.getContext(), ("interface." + op.getSymName()).str());

  // Empty structs may be kept opaque.
  if (!structBodyMembers.empty() &&
      failed(llvmStructTy.setBody(structBodyMembers, false)))
    return op.emitOpError() << "Failed to set LLVM Struct body for interface";

  structBody.interfaceBody = llvmStructTy;
  cache.setInterfaceInfo(ifaceSym, structBody);

  return success();
}

/// Convenience overload that looks up InterfaceDeclOp.
static LogicalResult resolveInterfaceStructBody(ModuleOp mod,
                                                SymbolRefAttr ifaceSym,
                                                TypeConverter const &typeConverter,
                                                InterfaceTypeCache &cache) {
  auto *opSym = mod.lookupSymbol(ifaceSym.getRootReference());
  if (!opSym)
    return failure();
  auto ifaceDeclOp = dyn_cast<InterfaceDeclOp>(opSym);
  if (!ifaceDeclOp)
    return failure();
  return resolveInterfaceStructBody(ifaceDeclOp, typeConverter, cache);
}

/// Returns the passed value if the integer width is already correct.
/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if it
/// is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return comb::ConcatOp::create(builder, loc, ValueRange{zeroExt, value});
  }

  Value hi = comb::ExtractOp::create(builder, loc, value, targetWidth,
                                     intWidth - targetWidth);
  Value zero = hw::ConstantOp::create(
      builder, loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, hi,
                                      zero, false);
  Value lo = comb::ExtractOp::create(builder, loc, value, 0, targetWidth);
  Value max = hw::ConstantOp::create(builder, loc,
                                     builder.getIntegerType(targetWidth), -1);
  return comb::MuxOp::create(builder, loc, isZero, lo, max, false);
}

/// Get the ModulePortInfo from a SVModuleOp.
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      // FIXME: Once we support net<...>, ref<...> type to represent type of
      // special port like inout or ref port which is not a input or output
      // port. It can change to generate corresponding types for direction of
      // port or do specified operation to it. Now inout and ref port is treated
      // as input port.
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }

  return hw::ModulePortInfo(ports);
}

//===----------------------------------------------------------------------===//
// Structural Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConversion : public OpConversionPattern<SVModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    // Create the hw.module to replace moore.module
    auto hwModuleOp =
        hw::HWModuleOp::create(rewriter, op.getLoc(), op.getSymNameAttr(),
                               getModulePortInfo(*typeConverter, op));
    // Make hw.module have the same visibility as the moore.module.
    // The entry/top level module is public, otherwise is private.
    SymbolTable::setSymbolVisibility(hwModuleOp,
                                     SymbolTable::getSymbolVisibility(op));
    rewriter.eraseBlock(hwModuleOp.getBodyBlock());
    if (failed(
            rewriter.convertRegionTypes(&op.getBodyRegion(), *typeConverter)))
      return failure();
    rewriter.inlineRegionBefore(op.getBodyRegion(), hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpConversion : public OpConversionPattern<OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct InstanceOpConversion : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = op.getInstanceNameAttr();
    auto moduleName = op.getModuleNameAttr();

    // Create the new hw instanceOp to replace the original one.
    rewriter.setInsertionPoint(op);
    auto instOp = hw::InstanceOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), instName, moduleName,
        op.getInputs(), op.getInputNamesAttr(), op.getOutputNamesAttr(),
        /*Parameter*/ rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr,
        /*doNotPrint*/ nullptr);

    // Replace uses chain and erase the original op.
    op.replaceAllUsesWith(instOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

static void getValuesToObserve(Region *region,
                               function_ref<void(Value)> setInsertionPoint,
                               const TypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value> &observeValues) {
  SmallDenseSet<Value> alreadyObserved;
  Location loc = region->getLoc();

  auto probeIfSignal = [&](Value value) -> Value {
    if (!isa<llhd::RefType>(value.getType()))
      return value;
    return llhd::ProbeOp::create(rewriter, loc, value);
  };

  region->getParentOp()->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](Operation *operation) {
        for (auto value : operation->getOperands()) {
          if (isa<BlockArgument>(value))
            value = rewriter.getRemappedValue(value);

          if (region->isAncestor(value.getParentRegion()))
            continue;
          if (auto *defOp = value.getDefiningOp();
              defOp && defOp->hasTrait<OpTrait::ConstantLike>())
            continue;
          if (!alreadyObserved.insert(value).second)
            continue;

          OpBuilder::InsertionGuard g(rewriter);
          if (auto remapped = rewriter.getRemappedValue(value)) {
            setInsertionPoint(remapped);
            observeValues.push_back(probeIfSignal(remapped));
          } else {
            setInsertionPoint(value);
            auto type = typeConverter->convertType(value.getType());
            auto converted = typeConverter->materializeTargetConversion(
                rewriter, loc, type, value);
            observeValues.push_back(probeIfSignal(converted));
          }
        }
      });
}

struct ProcedureOpConversion : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Collect values to observe before we do any modifications to the region.
    SmallVector<Value> observedValues;
    if (op.getKind() == ProcedureKind::AlwaysComb ||
        op.getKind() == ProcedureKind::AlwaysLatch) {
      auto setInsertionPoint = [&](Value value) {
        rewriter.setInsertionPoint(op);
      };
      getValuesToObserve(&op.getBody(), setInsertionPoint, typeConverter,
                         rewriter, observedValues);
    }

    auto loc = op.getLoc();
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter)))
      return failure();

    // Handle initial procedures. Simple initial blocks (without wait_event
    // or wait_delay) lower to `seq.initial` which is supported by arcilator
    // for simulation. Initial blocks with wait ops need `llhd.process` because
    // they require multiple basic blocks for the wait/check/resume pattern.
    // Unreachable ops (from $finish) are converted to sim.terminate + seq.yield
    // in the seq.initial path.
    if (op.getKind() == ProcedureKind::Initial) {
      bool hasWaitEvent = !op.getBody().getOps<WaitEventOp>().empty();
      bool hasWaitDelay = !op.getBody().getOps<WaitDelayOp>().empty();

      // Check if all captured values are constant-like. If not, we can't
      // use seq.initial due to IsolatedFromAbove constraint.
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(op.getBody(), op.getBody(), captures);
      bool allCapturesConstant = true;
      for (Value capture : captures) {
        Operation *defOp = capture.getDefiningOp();
        if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>()) {
          allCapturesConstant = false;
          break;
        }
      }

      // Simple initial blocks can use seq.initial for arcilator support.
      // This includes blocks with $finish (unreachable ops) which get
      // converted to sim.terminate.
      bool hasSingleBlock = op.getBody().hasOneBlock();
      if (!hasWaitEvent && !hasWaitDelay && allCapturesConstant &&
          hasSingleBlock) {
        auto initialOp =
            seq::InitialOp::create(rewriter, loc, TypeRange{}, std::function<void()>{});
        auto &body = initialOp.getBody();
        // The builder creates an empty block, erase it before inlining
        rewriter.eraseBlock(&body.front());
        rewriter.inlineRegionBefore(op.getBody(), body, body.end());

        // Clone constant-like operations that are used inside the initial block
        // but defined outside, to satisfy IsolatedFromAbove constraint.
        rewriter.setInsertionPointToStart(&body.front());
        for (Value capture : captures) {
          Operation *defOp = capture.getDefiningOp();
          Operation *cloned = rewriter.clone(*defOp);
          for (auto [orig, replacement] :
               llvm::zip(defOp->getResults(), cloned->getResults()))
            replaceAllUsesInRegionWith(orig, replacement, body);
        }

        for (auto returnOp :
             llvm::make_early_inc_range(body.getOps<ReturnOp>())) {
          rewriter.setInsertionPoint(returnOp);
          seq::YieldOp::create(rewriter, returnOp.getLoc());
          rewriter.eraseOp(returnOp);
        }

        // Convert unreachable ops (from $finish) to sim.terminate + seq.yield.
        // The sim.terminate is already generated by FinishBIOp conversion,
        // so we just need to add seq.yield to properly terminate the block.
        for (auto unreachableOp :
             llvm::make_early_inc_range(body.getOps<UnreachableOp>())) {
          rewriter.setInsertionPoint(unreachableOp);
          seq::YieldOp::create(rewriter, unreachableOp.getLoc());
          rewriter.eraseOp(unreachableOp);
        }

        rewriter.eraseOp(op);
        return success();
      }
      // Complex initial blocks (with wait ops or non-constant captures)
      // still need llhd.process with halt
      auto newOp = llhd::ProcessOp::create(rewriter, loc, TypeRange{});
      auto &body = newOp->getRegion(0);
      rewriter.inlineRegionBefore(op.getBody(), body, body.end());
      for (auto returnOp :
           llvm::make_early_inc_range(body.getOps<ReturnOp>())) {
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<llhd::HaltOp>(returnOp, ValueRange{});
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Handle final procedures. These lower to `llhd.final` op that executes
    // the body and then halts.
    if (op.getKind() == ProcedureKind::Final) {
      auto newOp = llhd::FinalOp::create(rewriter, loc);
      auto &body = newOp->getRegion(0);
      rewriter.inlineRegionBefore(op.getBody(), body, body.end());
      for (auto returnOp :
           llvm::make_early_inc_range(body.getOps<ReturnOp>())) {
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<llhd::HaltOp>(returnOp, ValueRange{});
      }
      rewriter.eraseOp(op);
      return success();
    }

    // All other procedures lower to a an `llhd.process`.
    auto newOp = llhd::ProcessOp::create(rewriter, loc, TypeRange{});

    // We need to add an empty entry block because it is not allowed in MLIR to
    // branch back to the entry block. Instead we put the logic in the second
    // block and branch to that.
    rewriter.createBlock(&newOp.getBody());
    auto *block = &op.getBody().front();
    cf::BranchOp::create(rewriter, loc, block);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());

    // Add special handling for `always_comb` and `always_latch` procedures.
    // These run once at simulation startup and then implicitly wait for any of
    // the values they access to change before running again. To implement this,
    // we create another basic block that contains the implicit wait, and make
    // all `moore.return` ops branch to that wait block instead of immediately
    // jumping back up to the body.
    if (op.getKind() == ProcedureKind::AlwaysComb ||
        op.getKind() == ProcedureKind::AlwaysLatch) {
      Block *waitBlock = rewriter.createBlock(&newOp.getBody());
      llhd::WaitOp::create(rewriter, loc, ValueRange{}, Value(), observedValues,
                           ValueRange{}, block);
      block = waitBlock;
    }

    // Make all `moore.return` ops branch back up to the beginning of the
    // process, or the wait block created above for `always_comb` and
    // `always_latch` procedures.
    for (auto returnOp : llvm::make_early_inc_range(newOp.getOps<ReturnOp>())) {
      rewriter.setInsertionPoint(returnOp);
      cf::BranchOp::create(rewriter, loc, block);
      rewriter.eraseOp(returnOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitEventOpConversion : public OpConversionPattern<WaitEventOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitEventOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In order to convert the `wait_event` op we need to create three separate
    // blocks at the location of the op:
    //
    // - A "wait" block that reads the current state of any values used to
    //   detect events and then waits until any of those values change. When a
    //   change occurs, control transfers to the "check" block.
    // - A "check" block which is executed after any interesting signal has
    //   changed. This is where any `detect_event` ops read the current state of
    //   interesting values and compare them against their state before the wait
    //   in order to detect an event. If any events were detected, control
    //   transfers to the "resume" block; otherwise control goes back to the
    //   "wait" block.
    // - A "resume" block which holds any ops after the `wait_event` op. This is
    //   where control is expected to resume after an event has happened.
    //
    // Block structure before:
    //     opA
    //     moore.wait_event { ... }
    //     opB
    //
    // Block structure after:
    //     opA
    //     cf.br ^wait
    // ^wait:
    //     <read "before" values>
    //     llhd.wait ^check, ...
    // ^check:
    //     <read "after" values>
    //     <detect edges>
    //     cf.cond_br %event, ^resume, ^wait
    // ^resume:
    //     opB
    auto *resumeBlock =
        rewriter.splitBlock(op->getBlock(), ++Block::iterator(op));

    // If the 'wait_event' op is empty, we can lower it to a 'llhd.wait' op
    // without any observed values, but since the process will never wake up
    // from suspension anyway, we can also just terminate it using the
    // 'llhd.halt' op.
    if (op.getBody().front().empty()) {
      // Let the cleanup iteration after the dialect conversion clean up all
      // remaining unreachable blocks.
      rewriter.replaceOpWithNewOp<llhd::HaltOp>(op, ValueRange{});
      return success();
    }

    auto *waitBlock = rewriter.createBlock(resumeBlock);
    auto *checkBlock = rewriter.createBlock(resumeBlock);

    auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, loc, waitBlock);

    // We need to inline two copies of the `wait_event`'s body region: one is
    // used to determine the values going into `detect_event` ops before the
    // `llhd.wait`, and one will do the actual event detection after the
    // `llhd.wait`.
    //
    // Create a copy of the entire `wait_event` op in the wait block, which also
    // creates a copy of its region. Take note of all inputs to `detect_event`
    // ops and delete the `detect_event` ops in this copy.
    SmallVector<Value> valuesBefore;
    rewriter.setInsertionPointToEnd(waitBlock);
    auto clonedOp = cast<WaitEventOp>(rewriter.clone(*op));
    bool allDetectsAreAnyChange = true;
    for (auto detectOp :
         llvm::make_early_inc_range(clonedOp.getOps<DetectEventOp>())) {
      if (detectOp.getEdge() != Edge::AnyChange || detectOp.getCondition())
        allDetectsAreAnyChange = false;
      valuesBefore.push_back(detectOp.getInput());
      rewriter.eraseOp(detectOp);
    }

    // Determine the values used during event detection that are defined outside
    // the `wait_event`'s body region. We want to wait for a change on these
    // signals before we check if any interesting event happened.
    SmallVector<Value> observeValues;
    auto setInsertionPointAfterDef = [&](Value value) {
      if (auto *op = value.getDefiningOp())
        rewriter.setInsertionPointAfter(op);
      if (auto arg = dyn_cast<BlockArgument>(value))
        rewriter.setInsertionPointToStart(value.getParentBlock());
    };

    getValuesToObserve(&clonedOp.getBody(), setInsertionPointAfterDef,
                       typeConverter, rewriter, observeValues);

    // Create the `llhd.wait` op that suspends the current process and waits for
    // a change in the interesting values listed in `observeValues`. When a
    // change is detected, execution resumes in the "check" block.
    auto waitOp = llhd::WaitOp::create(rewriter, loc, ValueRange{}, Value(),
                                       observeValues, ValueRange{}, checkBlock);
    rewriter.inlineBlockBefore(&clonedOp.getBody().front(), waitOp);
    rewriter.eraseOp(clonedOp);

    // Collect a list of all detect ops and inline the `wait_event` body into
    // the check block.
    SmallVector<DetectEventOp> detectOps(op.getBody().getOps<DetectEventOp>());
    rewriter.inlineBlockBefore(&op.getBody().front(), checkBlock,
                               checkBlock->end());
    rewriter.eraseOp(op);

    // Helper function to detect if a certain change occurred between a value
    // before the `llhd.wait` and after.
    auto computeTrigger = [&](Value before, Value after, Edge edge) -> Value {
      assert(before.getType() == after.getType() &&
             "mismatched types after clone op");
      auto beforeType = cast<IntType>(before.getType());

      // 9.4.2 IEEE 1800-2017: An edge event shall be detected only on the LSB
      // of the expression
      if (beforeType.getWidth() != 1 && edge != Edge::AnyChange) {
        constexpr int LSB = 0;
        beforeType =
            IntType::get(rewriter.getContext(), 1, beforeType.getDomain());
        before =
            moore::ExtractOp::create(rewriter, loc, beforeType, before, LSB);
        after = moore::ExtractOp::create(rewriter, loc, beforeType, after, LSB);
      }

      auto intType = rewriter.getIntegerType(beforeType.getWidth());
      before = typeConverter->materializeTargetConversion(rewriter, loc,
                                                          intType, before);
      after = typeConverter->materializeTargetConversion(rewriter, loc, intType,
                                                         after);

      if (edge == Edge::AnyChange)
        return comb::ICmpOp::create(rewriter, loc, ICmpPredicate::ne, before,
                                    after, true);

      SmallVector<Value> disjuncts;
      Value trueVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 1));

      if (edge == Edge::PosEdge || edge == Edge::BothEdges) {
        Value notOldVal =
            comb::XorOp::create(rewriter, loc, before, trueVal, true);
        Value posedge =
            comb::AndOp::create(rewriter, loc, notOldVal, after, true);
        disjuncts.push_back(posedge);
      }

      if (edge == Edge::NegEdge || edge == Edge::BothEdges) {
        Value notCurrVal =
            comb::XorOp::create(rewriter, loc, after, trueVal, true);
        Value posedge =
            comb::AndOp::create(rewriter, loc, before, notCurrVal, true);
        disjuncts.push_back(posedge);
      }

      return rewriter.createOrFold<comb::OrOp>(loc, disjuncts, true);
    };

    // Convert all `detect_event` ops into a check for the corresponding event
    // between the value before and after the `llhd.wait`. The "before" value
    // has been collected into `valuesBefore` in the "wait" block; the "after"
    // value corresponds to the detect op's input.
    SmallVector<Value> triggers;
    for (auto [detectOp, before] : llvm::zip(detectOps, valuesBefore)) {
      if (!allDetectsAreAnyChange) {
        if (!isa<IntType>(before.getType()))
          return detectOp->emitError() << "requires int operand";

        rewriter.setInsertionPoint(detectOp);
        auto trigger =
            computeTrigger(before, detectOp.getInput(), detectOp.getEdge());
        if (detectOp.getCondition()) {
          auto condition = typeConverter->materializeTargetConversion(
              rewriter, loc, rewriter.getI1Type(), detectOp.getCondition());
          trigger =
              comb::AndOp::create(rewriter, loc, trigger, condition, true);
        }
        triggers.push_back(trigger);
      }

      rewriter.eraseOp(detectOp);
    }

    rewriter.setInsertionPointToEnd(checkBlock);
    if (triggers.empty()) {
      // If there are no triggers to check, we always branch to the resume
      // block. If there are no detect_event operations in the wait event, the
      // 'llhd.wait' operation will not have any observed values and thus the
      // process will hang there forever.
      cf::BranchOp::create(rewriter, loc, resumeBlock);
    } else {
      // If any `detect_event` op detected an event, branch to the "resume"
      // block which contains any code after the `wait_event` op. If no events
      // were detected, branch back to the "wait" block to wait for the next
      // change on the interesting signals.
      auto triggered = rewriter.createOrFold<comb::OrOp>(loc, triggers, true);
      cf::CondBranchOp::create(rewriter, loc, triggered, resumeBlock,
                               waitBlock);
    }

    return success();
  }
};

// moore.wait_delay -> llhd.wait
static LogicalResult convert(WaitDelayOp op, WaitDelayOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto *resumeBlock =
      rewriter.splitBlock(op->getBlock(), ++Block::iterator(op));
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<llhd::WaitOp>(op, ValueRange{},
                                            adaptor.getDelay(), ValueRange{},
                                            ValueRange{}, resumeBlock);
  rewriter.setInsertionPointToStart(resumeBlock);
  return success();
}

// moore.unreachable -> llhd.halt
static LogicalResult convert(UnreachableOp op, UnreachableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::HaltOp>(op, ValueRange{});
  return success();
}

/// Conversion for moore.event_triggered -> runtime function call.
/// This calls __moore_event_triggered to check if the event was triggered.
struct EventTriggeredOpConversion
    : public OpConversionPattern<EventTriggeredOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EventTriggeredOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);

    // __moore_event_triggered takes a pointer to the event and returns i1.
    auto fnTy = LLVM::LLVMFunctionType::get(i1Ty, {ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_event_triggered",
                                     fnTy);

    // Store the event value to an alloca and pass a pointer to it.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto eventAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i1Ty, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getEvent(), eventAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i1Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{eventAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.event_trigger -> runtime function call.
/// This calls __moore_event_trigger to trigger the event.
struct EventTriggerOpConversion : public OpConversionPattern<EventTriggerOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EventTriggerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);

    // __moore_event_trigger takes a pointer to the event and returns void.
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_event_trigger", fnTy);

    // Store the event value to an alloca and pass a pointer to it.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto eventAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i1Ty, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getEvent(), eventAlloca);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{eventAlloca});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.wait_condition -> runtime function call.
/// Implements the SystemVerilog `wait(condition)` statement by calling
/// the __moore_wait_condition runtime function.
struct WaitConditionOpConversion : public OpConversionPattern<WaitConditionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    // __moore_wait_condition takes an i32 condition value.
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_wait_condition",
                                     fnTy);

    // The condition is i1, we need to zero-extend to i32.
    auto condition = adaptor.getCondition();
    auto conditionI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, condition);

    // Call the runtime function.
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{conditionI32});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Process Control Conversion (Fork/Join)
//===----------------------------------------------------------------------===//

// moore.fork -> sim.fork
struct ForkOpConversion : public OpConversionPattern<ForkOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Map Moore JoinType to Sim join_type string
    StringRef simJoinType;
    switch (op.getJoinType()) {
    case JoinType::JoinAll:
      simJoinType = "join";
      break;
    case JoinType::JoinAny:
      simJoinType = "join_any";
      break;
    case JoinType::JoinNone:
      simJoinType = "join_none";
      break;
    }

    // Get optional name attribute
    StringAttr nameAttr = nullptr;
    if (op.getName())
      nameAttr = rewriter.getStringAttr(op.getName().value());

    // Create sim.fork operation with correct number of branches
    unsigned numBranches = op.getBranches().size();
    auto simFork = sim::SimForkOp::create(rewriter, loc, rewriter.getI64Type(),
                                          simJoinType, nameAttr, numBranches);

    // Move regions from Moore fork to Sim fork
    for (auto [idx, branch] : llvm::enumerate(op.getBranches())) {
      Region &simRegion = simFork.getBranches()[idx];
      rewriter.inlineRegionBefore(branch, simRegion, simRegion.end());

      // Convert ForkTerminatorOp to SimForkTerminatorOp
      for (Block &block : simRegion) {
        if (auto terminator = dyn_cast<ForkTerminatorOp>(block.getTerminator())) {
          rewriter.setInsertionPoint(terminator);
          sim::SimForkTerminatorOp::create(rewriter, terminator.getLoc());
          rewriter.eraseOp(terminator);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.wait_fork -> sim.wait_fork
static LogicalResult convert(WaitForkOp op, WaitForkOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::SimWaitForkOp>(op);
  return success();
}

// moore.disable_fork -> sim.disable_fork
static LogicalResult convert(DisableForkOp op, DisableForkOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::SimDisableForkOp>(op);
  return success();
}

// moore.disable -> sim.disable
static LogicalResult convert(DisableOp op, DisableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::SimDisableOp>(op, op.getTarget());
  return success();
}

// moore.named_block -> sim.named_block
struct NamedBlockOpConversion : public OpConversionPattern<NamedBlockOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NamedBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Create sim.named_block operation with the block name as StringRef
    auto simNamedBlock =
        sim::SimNamedBlockOp::create(rewriter, loc, op.getBlockName().str());

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), simNamedBlock.getBody(),
                                simNamedBlock.getBody().end());

    // Convert NamedBlockTerminatorOp to SimNamedBlockTerminatorOp
    for (Block &block : simNamedBlock.getBody()) {
      if (auto terminator =
              dyn_cast<NamedBlockTerminatorOp>(block.getTerminator())) {
        rewriter.setInsertionPoint(terminator);
        sim::SimNamedBlockTerminatorOp::create(rewriter, terminator.getLoc());
        rewriter.eraseOp(terminator);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

/// Compute the size of a type in bytes, handling types that DataLayout
/// doesn't support (like llhd::TimeType).
static uint64_t getTypeSizeSafe(Type type, ModuleOp mod) {
  // Handle llhd::TimeType specially - DataLayout doesn't support it.
  // Time is represented as {i64 realTime, i32 delta, i32 epsilon} = 16 bytes.
  if (isa<llhd::TimeType>(type))
    return 16;

  // Handle LLVM struct types recursively to catch nested llhd::TimeType.
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type)) {
    uint64_t size = 0;
    for (Type fieldTy : structTy.getBody())
      size += getTypeSizeSafe(fieldTy, mod);
    return size;
  }

  // Handle LLVM array types.
  if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(type))
    return arrayTy.getNumElements() * getTypeSizeSafe(arrayTy.getElementType(), mod);

  // For other types, use DataLayout.
  DataLayout dl(mod);
  return dl.getTypeSize(type);
}

static Value createZeroValue(Type type, Location loc,
                             ConversionPatternRewriter &rewriter) {
  // Handle pointers.
  if (isa<mlir::LLVM::LLVMPointerType>(type))
    return mlir::LLVM::ZeroOp::create(rewriter, loc, type);

  // Handle time values.
  if (isa<llhd::TimeType>(type)) {
    auto timeAttr =
        llhd::TimeAttr::get(type.getContext(), 0U, llvm::StringRef("ns"), 0, 0);
    return llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
  }

  // Handle real values.
  if (auto floatType = dyn_cast<FloatType>(type)) {
    auto floatAttr = rewriter.getFloatAttr(floatType, 0.0);
    return mlir::arith::ConstantOp::create(rewriter, loc, floatAttr);
  }

  // Otherwise try to create a zero integer and bitcast it to the result type.
  int64_t width = hw::getBitWidth(type);
  if (width == -1)
    return {};

  // TODO: Once the core dialects support four-valued integers, this code
  // will additionally need to generate an all-X value for four-valued
  // variables.
  Value constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
  return rewriter.createOrFold<hw::BitcastOp>(loc, type, constZero);
}

struct ClassPropertyRefOpConversion
    : public OpConversionPattern<circt::moore::ClassPropertyRefOp> {
  ClassPropertyRefOpConversion(TypeConverter &tc, MLIRContext *ctx,
                               ClassTypeCache &cache)
      : OpConversionPattern(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(circt::moore::ClassPropertyRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Convert result type; we expect !llhd.ref<someT>.
    Type dstTy = getTypeConverter()->convertType(op.getPropertyRef().getType());
    // Operand is a !llvm.ptr
    Value instRef = adaptor.getInstance();

    // Resolve identified struct from cache.
    auto classRefTy =
        cast<circt::moore::ClassHandleType>(op.getInstance().getType());
    SymbolRefAttr classSym = classRefTy.getClassSym();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (failed(resolveClassStructBody(mod, classSym, *typeConverter, cache)))
      return rewriter.notifyMatchFailure(op,
                                         "Could not resolve class struct for " +
                                             classSym.getRootReference().str());

    auto structInfo = cache.getStructInfo(classSym);
    assert(structInfo && "class struct info must exist");
    auto structTy = structInfo->classBody;

    // Look up cached GEP path for the property.
    auto propSym = op.getProperty();
    auto pathOpt = structInfo->getFieldPath(propSym);
    if (!pathOpt)
      return rewriter.notifyMatchFailure(op,
                                         "no GEP path for property " + propSym);

    auto i32Ty = IntegerType::get(ctx, 32);
    SmallVector<Value> idxVals;
    for (unsigned idx : *pathOpt)
      idxVals.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(idx)));

    // GEP to the field (opaque ptr mode requires element type).
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto gep =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, instRef, idxVals);

    // Wrap pointer back to !llhd.ref<someT>.
    Value fieldRef = UnrealizedConversionCastOp::create(rewriter, loc, dstTy,
                                                        gep.getResult())
                         .getResult(0);

    rewriter.replaceOp(op, fieldRef);
    return success();
  }

private:
  ClassTypeCache &cache;
};

/// Lowering for VirtualInterfaceSignalRefOp.
/// Converts a virtual interface signal reference to a GEP operation that
/// extracts the signal field from the interface struct.
struct VirtualInterfaceSignalRefOpConversion
    : public OpConversionPattern<VirtualInterfaceSignalRefOp> {
  VirtualInterfaceSignalRefOpConversion(TypeConverter &tc, MLIRContext *ctx,
                                        InterfaceTypeCache &cache)
      : OpConversionPattern(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(VirtualInterfaceSignalRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Convert result type; we expect !llhd.ref<someT>.
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Operand is a !llvm.ptr (the virtual interface).
    Value vifRef = adaptor.getVif();

    // Get the interface symbol from the virtual interface type.
    auto vifType = op.getVif().getType();
    auto ifaceRef = vifType.getInterface();
    auto ifaceSym = SymbolRefAttr::get(
        rewriter.getContext(),
        ifaceRef.getRootReference());

    // Resolve the interface struct body.
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (failed(resolveInterfaceStructBody(mod, ifaceSym, *typeConverter, cache)))
      return rewriter.notifyMatchFailure(
          op, "Could not resolve interface struct for " +
                  ifaceSym.getRootReference().str());

    auto structInfo = cache.getStructInfo(ifaceSym);
    assert(structInfo && "interface struct info must exist");
    auto structTy = structInfo->interfaceBody;

    // Look up the signal index.
    auto signalName = op.getSignal();
    auto idxOpt = structInfo->getSignalIndex(signalName);
    if (!idxOpt)
      return rewriter.notifyMatchFailure(
          op, "no GEP index for signal " + signalName);

    // Build GEP indices: [0, signalIndex] to access the signal field.
    auto i32Ty = IntegerType::get(ctx, 32);
    SmallVector<Value> idxVals;
    idxVals.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    idxVals.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(*idxOpt)));

    // GEP to the signal field (opaque ptr mode requires element type).
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto gep =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, vifRef, idxVals);

    // Wrap pointer back to !llhd.ref<someT>.
    Value signalRef = UnrealizedConversionCastOp::create(rewriter, loc, dstTy,
                                                         gep.getResult())
                          .getResult(0);

    rewriter.replaceOp(op, signalRef);
    return success();
  }

private:
  InterfaceTypeCache &cache;
};

/// Lowering for InterfaceSignalDeclOp.
/// Signal declarations within interfaces are processed by resolveInterfaceStructBody
/// and then erased - they become part of the interface struct type.
struct InterfaceSignalDeclOpConversion
    : public OpConversionPattern<InterfaceSignalDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InterfaceSignalDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Signal declarations are metadata used to build the interface struct.
    // They have already been processed by resolveInterfaceStructBody.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ModportDeclOp.
/// Modport declarations are directional views that are resolved at compile time.
/// They don't produce runtime code; their information is used during signal access.
struct ModportDeclOpConversion : public OpConversionPattern<ModportDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ModportDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Modports are metadata describing signal directions.
    // They don't generate runtime code.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ClockingBlockDeclOp.
/// Clocking blocks are compile-time constructs that define synchronization.
/// They don't produce runtime code; their timing information is used during
/// signal sampling and driving.
struct ClockingBlockDeclOpConversion
    : public OpConversionPattern<ClockingBlockDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockingBlockDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Clocking blocks are compile-time constructs for specifying timing.
    // They don't generate runtime code.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ClockingSignalOp.
/// Clocking signals are part of the clocking block metadata and don't
/// produce runtime code.
struct ClockingSignalOpConversion
    : public OpConversionPattern<ClockingSignalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockingSignalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Clocking signals are metadata within clocking blocks.
    // They don't generate runtime code.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for InterfaceInstanceOp.
/// Creates an interface instance by allocating memory for the interface struct.
struct InterfaceInstanceOpConversion
    : public OpConversionPattern<InterfaceInstanceOp> {
  InterfaceInstanceOpConversion(TypeConverter &tc, MLIRContext *ctx,
                                InterfaceTypeCache &cache)
      : OpConversionPattern(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(InterfaceInstanceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Get the interface symbol.
    auto ifaceSym = op.getInterfaceName();
    auto ifaceSymRef = SymbolRefAttr::get(ctx, ifaceSym);

    // Resolve the interface struct body.
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (failed(resolveInterfaceStructBody(mod, ifaceSymRef, *typeConverter,
                                          cache)))
      return op.emitError()
             << "Could not resolve interface struct for " << ifaceSym;

    auto structInfo = cache.getStructInfo(ifaceSymRef);
    assert(structInfo && "interface struct info must exist");
    auto structTy = structInfo->interfaceBody;

    // Calculate the size of the interface struct.
    uint64_t byteSize = getTypeSizeSafe(structTy, mod);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto cSize = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(byteSize));

    // Get or declare malloc and call it.
    auto mallocFn = getOrCreateMalloc(mod, rewriter);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto call =
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                             SymbolRefAttr::get(mallocFn), ValueRange{cSize});

    Value ifacePtr = call.getResult();

    // Replace the instance op with the allocated pointer.
    rewriter.replaceOp(op, ifacePtr);
    return success();
  }

private:
  InterfaceTypeCache &cache;
};

/// Lowering for VirtualInterfaceGetOp.
/// Extracts a modport view from a virtual interface. Since modports are just
/// directional metadata, this is a pass-through - the underlying pointer is the same.
struct VirtualInterfaceGetOpConversion
    : public OpConversionPattern<VirtualInterfaceGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VirtualInterfaceGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Modport views share the same underlying interface pointer.
    // The modport just restricts which signals can be accessed and their directions.
    // At runtime, we just pass through the same pointer.
    rewriter.replaceOp(op, adaptor.getVif());
    return success();
  }
};

struct ClassUpcastOpConversion : public OpConversionPattern<ClassUpcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassUpcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect lowered types like !llvm.ptr
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    Type srcTy = adaptor.getInstance().getType();

    if (!dstTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // If the types are already identical (opaque pointer mode), just forward.
    if (dstTy == srcTy && isa<LLVM::LLVMPointerType>(srcTy)) {
      rewriter.replaceOp(op, adaptor.getInstance());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "Upcast applied to non-opaque pointers!");
  }
};

/// moore.class.dyn_cast lowering: runtime type check and downcast.
/// Performs proper RTTI check by loading the source object's type ID and
/// comparing it against the target type ID using the runtime function.
struct ClassDynCastOpConversion : public OpConversionPattern<ClassDynCastOp> {
  ClassDynCastOpConversion(TypeConverter &tc, MLIRContext *ctx,
                           ClassTypeCache &cache)
      : OpConversionPattern<ClassDynCastOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassDynCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Convert the result type (should be an LLVM pointer)
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    Type srcTy = adaptor.getSource().getType();

    if (!dstTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (!(dstTy == srcTy && isa<LLVM::LLVMPointerType>(srcTy)))
      return rewriter.notifyMatchFailure(
          op, "DynCast applied to non-opaque pointers!");

    // Get the target class type info to obtain its type ID
    auto targetHandleTy = cast<ClassHandleType>(op.getResult().getType());
    auto targetSym = targetHandleTy.getClassSym();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    if (failed(resolveClassStructBody(mod, targetSym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve target class struct for "
                            << targetSym;

    auto targetInfo = cache.getStructInfo(targetSym);
    if (!targetInfo)
      return op.emitError() << "No struct info for target class " << targetSym;

    // Get the source class type info to determine inheritance depth
    auto srcHandleTy = cast<ClassHandleType>(op.getSource().getType());
    auto srcSym = srcHandleTy.getClassSym();

    if (failed(resolveClassStructBody(mod, srcSym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve source class struct for "
                            << srcSym;

    auto srcInfo = cache.getStructInfo(srcSym);
    if (!srcInfo)
      return op.emitError() << "No struct info for source class " << srcSym;

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    Value srcPtr = adaptor.getSource();

    // Create a null pointer for comparison
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);

    // Load the type ID from the source object.
    // The type ID is stored at the beginning of the root class struct.
    // Build GEP path: one 0 per inheritance level + 0 for typeId field.
    SmallVector<Value> gepIndices;
    for (int32_t i = 0; i <= srcInfo->inheritanceDepth; ++i) {
      gepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    }

    auto typeIdPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                         srcInfo->classBody, srcPtr, gepIndices);
    Value srcTypeId = LLVM::LoadOp::create(rewriter, loc, i32Ty, typeIdPtr);

    // Get or declare the __moore_dyn_cast_check runtime function
    auto fnTy = LLVM::LLVMFunctionType::get(
        i1Ty, {i32Ty, i32Ty, i32Ty}, /*isVarArg=*/false);
    auto dynCastFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_dyn_cast_check", fnTy);

    // Call the runtime function to check type compatibility
    Value targetTypeId = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(targetInfo->typeId));
    Value inheritanceDepth = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(targetInfo->inheritanceDepth));

    auto checkCall = LLVM::CallOp::create(
        rewriter, loc, TypeRange{i1Ty}, SymbolRefAttr::get(dynCastFn),
        ValueRange{srcTypeId, targetTypeId, inheritanceDepth});

    // Final success flag: cast succeeds if pointer is not null AND type check passes
    Value typeCheckResult = checkCall.getResult();
    Value notNull = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
                                         LLVM::ICmpPredicate::ne, srcPtr, nullPtr);
    Value successFlag =
        LLVM::AndOp::create(rewriter, loc, notNull, typeCheckResult);

    // The casted pointer is the same as the input (pointer bit-cast)
    // The success flag indicates whether the cast was valid
    rewriter.replaceOp(op, {srcPtr, successFlag});
    return success();
  }

private:
  ClassTypeCache &cache;
};

/// moore.class.new lowering: heap-allocate storage for the class object.
/// After allocation, initializes the type ID field for RTTI support.
struct ClassNewOpConversion : public OpConversionPattern<ClassNewOp> {
  ClassNewOpConversion(TypeConverter &tc, MLIRContext *ctx,
                       ClassTypeCache &cache)
      : OpConversionPattern<ClassNewOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto handleTy = cast<ClassHandleType>(op.getResult().getType());
    auto sym = handleTy.getClassSym();

    ModuleOp mod = op->getParentOfType<ModuleOp>();

    if (failed(resolveClassStructBody(mod, sym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve class struct for " << sym;

    auto structInfo = cache.getStructInfo(sym);
    auto structTy = structInfo->classBody;

    // Compute struct size (handles llhd::TimeType which DataLayout doesn't support).
    uint64_t byteSize = getTypeSizeSafe(structTy, mod);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto cSize = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(byteSize));

    // Get or declare malloc and call it.
    auto mallocFn = getOrCreateMalloc(mod, rewriter);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx); // opaque pointer result
    auto call =
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                             SymbolRefAttr::get(mallocFn), ValueRange{cSize});

    Value objPtr = call.getResult();

    // Initialize the type ID field for RTTI support.
    // The type ID is stored at the beginning of the object (either directly
    // for root classes, or nested in the base class prefix for derived classes).
    // We need to drill down through the base class chain to find the root's
    // typeId field at offset 0.
    //
    // For a class hierarchy A -> B -> C, the layout is:
    //   C: { B: { A: { i32 typeId, ...A fields }, ...B fields }, ...C fields }
    // So we GEP through indices [0, 0, ...0, 0] to reach the typeId.

    // Build the GEP path to the typeId field: one 0 per inheritance level,
    // then 0 for the typeId field itself.
    SmallVector<Value> gepIndices;
    for (int32_t i = 0; i <= structInfo->inheritanceDepth; ++i) {
      gepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    }

    // GEP to the typeId field
    auto typeIdPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, objPtr, gepIndices);

    // Store the type ID value
    auto typeIdConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(structInfo->typeId));
    LLVM::StoreOp::create(rewriter, loc, typeIdConst, typeIdPtr);

    // Replace the new op with the malloc pointer
    rewriter.replaceOp(op, objPtr);
    return success();
  }

private:
  ClassTypeCache &cache; // shared, owned by the pass
};

struct ClassDeclOpConversion : public OpConversionPattern<ClassDeclOp> {
  ClassDeclOpConversion(TypeConverter &tc, MLIRContext *ctx,
                        ClassTypeCache &cache)
      : OpConversionPattern<ClassDeclOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(resolveClassStructBody(op, *typeConverter, cache)))
      return failure();
    // The declaration itself is a no-op
    rewriter.eraseOp(op);
    return success();
  }

private:
  ClassTypeCache &cache; // shared, owned by the pass
};

/// moore.class.null lowering: create a null pointer constant.
struct ClassNullOpConversion : public OpConversionPattern<ClassNullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ClassHandleType converts to !llvm.ptr, so we just need a null pointer.
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrTy);
    return success();
  }
};

/// Lowering for InterfaceDeclOp.
/// The interface declaration is resolved to an LLVM struct and then erased.
struct InterfaceDeclOpConversion : public OpConversionPattern<InterfaceDeclOp> {
  InterfaceDeclOpConversion(TypeConverter &tc, MLIRContext *ctx,
                            InterfaceTypeCache &cache)
      : OpConversionPattern<InterfaceDeclOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(InterfaceDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(resolveInterfaceStructBody(op, *typeConverter, cache)))
      return failure();
    // The declaration itself is a no-op; erase it.
    rewriter.eraseOp(op);
    return success();
  }

private:
  InterfaceTypeCache &cache;
};

/// Helper to create a global string constant for covergroup/coverpoint names.
/// Creates the global if it doesn't exist and returns the address.
static Value createGlobalStringConstant(Location loc, ModuleOp mod,
                                        ConversionPatternRewriter &rewriter,
                                        StringRef name, StringRef globalName) {
  auto *ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  // Check if we already created this global
  if (!mod.lookupSymbol<LLVM::GlobalOp>(globalName)) {
    // Create array type for the string (including null terminator)
    auto i8Ty = IntegerType::get(ctx, 8);
    auto strTy = LLVM::LLVMArrayType::get(i8Ty, name.size() + 1);

    // Create null-terminated string value
    std::string strWithNull = (name + StringRef("\0", 1)).str();
    auto strAttr = rewriter.getStringAttr(strWithNull);

    // Insert global at module level
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());

    auto global = LLVM::GlobalOp::create(rewriter, loc, strTy,
                                         /*isConstant=*/true,
                                         LLVM::Linkage::Private, globalName,
                                         strAttr);
    global.setUnnamedAddr(LLVM::UnnamedAddr::Global);
  }

  // Return the address of the global
  return LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalName);
}

/// Lowering for CovergroupDeclOp.
/// Generates runtime calls to register the covergroup and its coverpoints:
/// 1. Creates a global variable to store the covergroup handle
/// 2. Generates an initialization function that:
///    - Calls __moore_covergroup_create(name, num_coverpoints) -> void*
///    - Calls __moore_coverpoint_init(cg, index, name) for each coverpoint
///    - Stores the handle to the global variable
///
/// Runtime functions used:
/// - __moore_covergroup_create(name, num_coverpoints) -> void*
/// - __moore_coverpoint_init(cg, index, name)
/// - __moore_coverpoint_sample(cg, index, value) (called from sample sites)
/// - __moore_covergroup_destroy(cg) (future: cleanup at end of simulation)
/// - __moore_coverage_report() (future: called at $finish)
struct CovergroupDeclOpConversion
    : public OpConversionPattern<CovergroupDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CovergroupDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get covergroup name
    StringRef cgName = op.getSymName();

    // Count coverpoints in this covergroup
    SmallVector<CoverpointDeclOp> coverpoints;
    for (auto &bodyOp : op.getBody().front()) {
      if (auto cp = dyn_cast<CoverpointDeclOp>(&bodyOp))
        coverpoints.push_back(cp);
    }
    int32_t numCoverpoints = coverpoints.size();

    // Create a global variable to hold the covergroup handle
    std::string globalHandleName = ("__cg_handle_" + cgName).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());

      // Check if global already exists
      if (!mod.lookupSymbol<LLVM::GlobalOp>(globalHandleName)) {
        auto nullAttr = rewriter.getZeroAttr(
            IntegerType::get(ctx, 64)); // Pointer-sized null
        LLVM::GlobalOp::create(rewriter, loc, ptrTy,
                               /*isConstant=*/false, LLVM::Linkage::Internal,
                               globalHandleName, nullAttr);
      }
    }

    // Create an initialization function for this covergroup
    std::string initFuncName = ("__cg_init_" + cgName).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());

      // Check if init function already exists
      if (!mod.lookupSymbol<LLVM::LLVMFuncOp>(initFuncName)) {
        auto initFnTy = LLVM::LLVMFunctionType::get(voidTy, {});
        auto initFn =
            LLVM::LLVMFuncOp::create(rewriter, loc, initFuncName, initFnTy);
        initFn.setLinkage(LLVM::Linkage::Internal);

        // Create the function body
        Block *entryBlock = rewriter.createBlock(&initFn.getBody());
        rewriter.setInsertionPointToStart(entryBlock);

        // Get or create __moore_covergroup_create function
        auto createFnTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, i32Ty});
        auto createFn = getOrCreateRuntimeFunc(mod, rewriter,
                                               "__moore_covergroup_create",
                                               createFnTy);

        // Get or create __moore_coverpoint_init function
        auto initCpFnTy =
            LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, ptrTy});
        auto initCpFn = getOrCreateRuntimeFunc(
            mod, rewriter, "__moore_coverpoint_init", initCpFnTy);

        // Create string constant for covergroup name
        std::string cgNameGlobal = ("__cg_name_" + cgName).str();
        Value cgNamePtr = createGlobalStringConstant(loc, mod, rewriter, cgName,
                                                     cgNameGlobal);

        // Call __moore_covergroup_create(name, num_coverpoints)
        auto numCpConst = LLVM::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numCoverpoints));
        auto createCall = LLVM::CallOp::create(
            rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(createFn),
            ValueRange{cgNamePtr, numCpConst});
        Value cgHandle = createCall.getResult();

        // Initialize each coverpoint
        int32_t cpIndex = 0;
        for (auto cp : coverpoints) {
          StringRef cpName = cp.getSymName();
          std::string cpNameGlobal =
              ("__cp_name_" + cgName + "_" + cpName).str();
          Value cpNamePtr = createGlobalStringConstant(loc, mod, rewriter,
                                                       cpName, cpNameGlobal);

          auto idxConst = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(cpIndex));

          LLVM::CallOp::create(rewriter, loc, TypeRange{},
                               SymbolRefAttr::get(initCpFn),
                               ValueRange{cgHandle, idxConst, cpNamePtr});
          ++cpIndex;
        }

        // Store the handle to the global variable
        auto handleGlobalPtr =
            LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalHandleName);
        LLVM::StoreOp::create(rewriter, loc, cgHandle, handleGlobalPtr);

        // Return from init function
        LLVM::ReturnOp::create(rewriter, loc, ValueRange{});
      }
    }

    // Erase the original covergroup declaration
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CoverpointDeclOp.
/// Coverpoints are processed by the parent CovergroupDeclOp during its
/// conversion. This pattern just erases the declaration after the parent has
/// processed it. The coverpoint initialization calls are generated by
/// CovergroupDeclOpConversion.
struct CoverpointDeclOpConversion
    : public OpConversionPattern<CoverpointDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoverpointDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Coverpoints are handled by the parent CovergroupDeclOp conversion.
    // Just erase this op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CoverCrossDeclOp.
/// Currently erases the declaration. Cross coverage support is future work.
struct CoverCrossDeclOpConversion
    : public OpConversionPattern<CoverCrossDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoverCrossDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Cross coverage requires additional runtime support.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CovergroupInstOp.
/// Calls the covergroup initialization function generated by CovergroupDeclOp
/// lowering and returns the handle to the covergroup instance.
struct CovergroupInstOpConversion
    : public OpConversionPattern<CovergroupInstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CovergroupInstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get the covergroup name from the symbol reference.
    StringRef cgName = op.getCovergroup();

    // Get the global handle for this covergroup.
    std::string globalHandleName = ("__cg_handle_" + cgName).str();

    // Get or create the init function for this covergroup.
    std::string initFuncName = ("__cg_init_" + cgName).str();
    auto initFnTy = LLVM::LLVMFunctionType::get(voidTy, {});
    auto initFn = getOrCreateRuntimeFunc(mod, rewriter, initFuncName, initFnTy);

    // Call the init function to ensure the covergroup is initialized.
    LLVM::CallOp::create(rewriter, loc, TypeRange{},
                         SymbolRefAttr::get(initFn), ValueRange{});

    // Load the covergroup handle from the global variable.
    auto handleGlobalPtr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalHandleName);
    auto cgHandle = LLVM::LoadOp::create(rewriter, loc, ptrTy, handleGlobalPtr);

    // Replace the op with the loaded handle.
    rewriter.replaceOp(op, cgHandle.getResult());
    return success();
  }
};

/// Lowering for CovergroupSampleOp.
/// Calls __moore_coverpoint_sample for each coverpoint in the covergroup.
struct CovergroupSampleOpConversion
    : public OpConversionPattern<CovergroupSampleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CovergroupSampleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get or create __moore_coverpoint_sample function.
    auto sampleFnTy =
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, i64Ty});
    auto sampleFn = getOrCreateRuntimeFunc(mod, rewriter,
                                           "__moore_coverpoint_sample",
                                           sampleFnTy);

    // Get the covergroup handle.
    Value cgHandle = adaptor.getCovergroup();

    // Sample each value.
    int32_t cpIndex = 0;
    for (Value val : adaptor.getValues()) {
      // Convert value to i64 for the runtime call.
      Value i64Val;
      if (val.getType().isInteger(64)) {
        i64Val = val;
      } else if (val.getType().isIntOrIndex()) {
        // Zero-extend or truncate to i64.
        unsigned width = val.getType().getIntOrFloatBitWidth();
        if (width < 64) {
          i64Val = arith::ExtUIOp::create(rewriter, loc, i64Ty, val);
        } else if (width > 64) {
          i64Val = arith::TruncIOp::create(rewriter, loc, i64Ty, val);
        } else {
          i64Val = val;
        }
      } else {
        // For non-integer types, bitcast to integer first if possible.
        // For now, skip non-integer values.
        ++cpIndex;
        continue;
      }

      auto idxConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(cpIndex));

      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(sampleFn),
                           ValueRange{cgHandle, idxConst, i64Val});
      ++cpIndex;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CovergroupGetCoverageOp.
/// Calls __moore_covergroup_get_coverage and returns the coverage percentage.
struct CovergroupGetCoverageOpConversion
    : public OpConversionPattern<CovergroupGetCoverageOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CovergroupGetCoverageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto f64Ty = Float64Type::get(ctx);

    // Get or create __moore_covergroup_get_coverage function.
    auto getCovFnTy = LLVM::LLVMFunctionType::get(f64Ty, {ptrTy});
    auto getCovFn = getOrCreateRuntimeFunc(mod, rewriter,
                                           "__moore_covergroup_get_coverage",
                                           getCovFnTy);

    // Get the covergroup handle.
    Value cgHandle = adaptor.getCovergroup();

    // Call the runtime function.
    auto callOp = LLVM::CallOp::create(rewriter, loc, TypeRange{f64Ty},
                                       SymbolRefAttr::get(getCovFn),
                                       ValueRange{cgHandle});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constraint Expression Operations Lowering
//===----------------------------------------------------------------------===//
//
// These patterns lower constraint ops to runtime calls for constraint checking
// during randomization. The constraint ops are processed in two phases:
// 1. During RandomizeOp lowering, constraints are extracted and used to guide
//    constrained random value generation.
// 2. After randomization, ConstraintExprOp conditions are evaluated to verify
//    that all constraints are satisfied.
//
// The lowering approach:
// - ConstraintBlockOp: Erased after constraints are extracted by RandomizeOp
// - ConstraintExprOp: Lowered to runtime constraint check call
// - ConstraintImplicationOp: Lowered to conditional constraint check
// - ConstraintIfElseOp: Lowered to conditional constraint selection
// - ConstraintUniqueOp: Lowered to uniqueness check runtime call
// - ConstraintInsideOp: Processed during range constraint extraction
// - ConstraintDistOp: Distribution constraint (erased, processed by solver)
// - ConstraintForeachOp: Loop constraint (erased, processed by solver)
// - ConstraintSolveBeforeOp: Ordering hint (erased, processed by solver)
// - ConstraintDisableOp: Disable soft constraint (erased, processed by solver)
//
//===----------------------------------------------------------------------===//

/// Lowering for ConstraintBlockOp.
/// The constraint block is erased after its contents are processed by
/// RandomizeOp lowering. The block arguments (random variables) are resolved
/// during constraint extraction.
struct ConstraintBlockOpConversion
    : public OpConversionPattern<ConstraintBlockOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintBlockOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Constraint blocks are processed by RandomizeOp conversion.
    // The block itself is a declaration and can be erased.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintExprOp.
/// Constraint expressions are boolean conditions that must hold during
/// randomization. They are evaluated after random values are generated
/// and the result is used to determine randomization success.
///
/// For now, constraint expressions are erased since they are processed
/// during the RandomizeOp lowering's constraint extraction phase.
/// In the future, this could generate runtime validation calls.
struct ConstraintExprOpConversion
    : public OpConversionPattern<ConstraintExprOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintExprOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Constraint expressions are evaluated during RandomizeOp processing.
    // The expression itself is erased as part of the parent constraint block.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintImplicationOp.
/// Implication constraints (expr -> constraint) are conditional: if the
/// antecedent is true, the consequent constraints must be satisfied.
/// These are processed during RandomizeOp lowering and then erased.
struct ConstraintImplicationOpConversion
    : public OpConversionPattern<ConstraintImplicationOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintImplicationOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implication constraints are processed by RandomizeOp conversion.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintIfElseOp.
/// If-else constraints select between two sets of constraints based on a
/// condition. These are processed during RandomizeOp lowering and then erased.
struct ConstraintIfElseOpConversion
    : public OpConversionPattern<ConstraintIfElseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintIfElseOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If-else constraints are processed by RandomizeOp conversion.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintForeachOp.
/// Foreach constraints iterate over array elements and apply constraints.
/// Currently erased; full support requires constraint solver integration.
struct ConstraintForeachOpConversion
    : public OpConversionPattern<ConstraintForeachOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintForeachOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Foreach constraints require constraint solver support.
    // TODO: Generate loop-based constraint validation.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintDistOp.
/// Distribution constraints specify weighted probability distributions.
/// Currently erased; full support requires weighted random generation.
struct ConstraintDistOpConversion : public OpConversionPattern<ConstraintDistOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintDistOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Distribution constraints are processed during RandomizeOp conversion.
    // TODO: Generate weighted random number generation calls.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintInsideOp.
/// Inside constraints are range membership constraints processed during
/// RandomizeOp lowering for constrained random generation.
struct ConstraintInsideOpConversion
    : public OpConversionPattern<ConstraintInsideOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintInsideOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inside constraints are processed by extractRangeConstraints().
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintSolveBeforeOp.
/// Solve-before constraints specify variable ordering for the solver.
/// Currently erased as ordering hints are not yet implemented.
struct ConstraintSolveBeforeOpConversion
    : public OpConversionPattern<ConstraintSolveBeforeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintSolveBeforeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Solve ordering is a hint to the constraint solver.
    // Currently not implemented; erase the op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintDisableOp.
/// Disables a soft constraint by name. Currently erased.
struct ConstraintDisableOpConversion
    : public OpConversionPattern<ConstraintDisableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintDisableOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Soft constraint disabling is handled during constraint extraction.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintUniqueOp.
/// Uniqueness constraints require all elements to have distinct values.
/// Currently erased; full support requires post-randomization validation.
struct ConstraintUniqueOpConversion
    : public OpConversionPattern<ConstraintUniqueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintUniqueOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Unique constraints require post-randomization validation.
    // TODO: Generate uniqueness check runtime call.
    rewriter.eraseOp(op);
    return success();
  }
};

/// moore.vtable lowering: erase the vtable declaration.
/// The vtable entries are resolved at load time via symbol lookup.
/// Note: This pattern has a lower benefit than VTableLoadMethodOpConversion
/// to ensure vtables are not erased before load_method ops are converted.
struct VTableOpConversion : public OpConversionPattern<VTableOp> {
  VTableOpConversion(TypeConverter &tc, MLIRContext *ctx)
      : OpConversionPattern<VTableOp>(tc, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(VTableOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The vtable declaration itself is a no-op in lowering.
    // Method dispatch is resolved via VTableLoadMethodOp.
    rewriter.eraseOp(op);
    return success();
  }
};

/// moore.vtable_entry lowering: erase the vtable entry declaration.
/// The entries are resolved at load time via symbol lookup.
/// Note: This pattern has a lower benefit than VTableLoadMethodOpConversion.
struct VTableEntryOpConversion : public OpConversionPattern<VTableEntryOp> {
  VTableEntryOpConversion(TypeConverter &tc, MLIRContext *ctx)
      : OpConversionPattern<VTableEntryOp>(tc, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(VTableEntryOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The vtable entry is a no-op; method lookup happens at dispatch time.
    rewriter.eraseOp(op);
    return success();
  }
};

/// moore.vtable.load_method lowering: resolve the virtual method and return
/// a function pointer.
/// Note: Higher benefit ensures this runs before VTableOp/VTableEntryOp erasure.
struct VTableLoadMethodOpConversion
    : public OpConversionPattern<VTableLoadMethodOp> {
  VTableLoadMethodOpConversion(TypeConverter &tc, MLIRContext *ctx)
      : OpConversionPattern<VTableLoadMethodOp>(tc, ctx, /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(VTableLoadMethodOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Get the class type from the object operand.
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    auto classSym = handleTy.getClassSym();

    // Look up the vtable for this class by searching all VTableOps in the
    // module. The vtable symbol is `@ClassName::@vtable` where ClassName
    // is the root reference.
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    VTableOp vtable = nullptr;
    StringRef className = classSym.getRootReference();

    // Helper to recursively find a vtable with matching class name.
    std::function<VTableOp(VTableOp)> findNestedVTable;
    findNestedVTable = [&](VTableOp vt) -> VTableOp {
      if (vt.getSymName().getRootReference() == className)
        return vt;
      for (Operation &child : vt.getBody().front()) {
        if (auto nestedVt = dyn_cast<VTableOp>(child)) {
          if (auto found = findNestedVTable(nestedVt))
            return found;
        }
      }
      return nullptr;
    };

    // First try to find a top-level vtable for this class.
    for (auto vt : mod.getOps<VTableOp>()) {
      // The vtable's sym_name is @ClassName::@vtable, so check if root matches.
      if (vt.getSymName().getRootReference() == className) {
        vtable = vt;
        break;
      }
    }

    // If no top-level vtable found (abstract class), search for a nested vtable
    // with matching class name inside any top-level vtable. Abstract classes
    // don't have their own top-level vtables but their vtable segments appear
    // nested inside concrete derived class vtables.
    if (!vtable) {
      for (auto vt : mod.getOps<VTableOp>()) {
        if (auto found = findNestedVTable(vt)) {
          vtable = found;
          break;
        }
      }
    }

    // Search for the method in the vtable (including nested vtables for base
    // classes).
    auto methodSym = op.getMethodSym();
    StringRef methodName = methodSym.getLeafReference();

    // Recursive helper to find entry in vtable hierarchy.
    std::function<VTableEntryOp(VTableOp)> findEntry =
        [&](VTableOp vt) -> VTableEntryOp {
      for (Operation &child : vt.getBody().front()) {
        if (auto entry = dyn_cast<VTableEntryOp>(child)) {
          if (entry.getName() == methodName)
            return entry;
        } else if (auto nestedVt = dyn_cast<VTableOp>(child)) {
          if (auto found = findEntry(nestedVt))
            return found;
        }
      }
      return nullptr;
    };

    VTableEntryOp entry = nullptr;
    if (vtable) {
      entry = findEntry(vtable);
    }

    // If no vtable was found for the class or the method wasn't in the found
    // vtable, search ALL vtables in the module for the method. This handles
    // cases where a class doesn't have its own vtable segment (e.g., a
    // non-abstract class that extends another class but has no concrete derived
    // classes with vtable segments for it).
    if (!entry) {
      for (auto vt : mod.getOps<VTableOp>()) {
        if (auto found = findEntry(vt)) {
          entry = found;
          break;
        }
      }
    }

    if (!entry)
      return rewriter.notifyMatchFailure(
          op, "could not find method " + methodName.str() + " in any vtable");

    // Get the target function symbol.
    auto targetSym = entry.getTarget();

    // Convert SymbolRefAttr to FlatSymbolRefAttr (the target should be a flat
    // symbol for a top-level function).
    auto flatTargetSym =
        FlatSymbolRefAttr::get(ctx, targetSym.getRootReference());

    // Convert the result function type from Moore types to the lowered types.
    auto resultTy = typeConverter->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Create a func.constant operation to get the function pointer.
    auto constOp =
        func::ConstantOp::create(rewriter, loc, resultTy, flatTargetSym);

    rewriter.replaceOp(op, constOp.getResult());
    return success();
  }
};

/// moore.class_handle_cmp lowering: compare two class handles using icmp.
struct ClassHandleCmpOpConversion
    : public OpConversionPattern<ClassHandleCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassHandleCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Map the moore predicate to LLVM icmp predicate.
    LLVM::ICmpPredicate pred;
    switch (op.getPredicate()) {
    case ClassHandleCmpPredicate::eq:
      pred = LLVM::ICmpPredicate::eq;
      break;
    case ClassHandleCmpPredicate::ne:
      pred = LLVM::ICmpPredicate::ne;
      break;
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, resultType, pred,
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// moore.virtual_interface.null lowering: create a null pointer.
struct VirtualInterfaceNullOpConversion
    : public OpConversionPattern<VirtualInterfaceNullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VirtualInterfaceNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Virtual interfaces are represented as pointers - null is a null pointer.
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrType);
    return success();
  }
};

/// moore.virtual_interface_cmp lowering: compare two virtual interfaces using icmp.
struct VirtualInterfaceCmpOpConversion
    : public OpConversionPattern<VirtualInterfaceCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VirtualInterfaceCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Map the moore predicate to LLVM icmp predicate.
    LLVM::ICmpPredicate pred;
    switch (op.getPredicate()) {
    case VirtualInterfaceCmpPredicate::eq:
      pred = LLVM::ICmpPredicate::eq;
      break;
    case VirtualInterfaceCmpPredicate::ne:
      pred = LLVM::ICmpPredicate::ne;
      break;
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, resultType, pred,
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct VariableOpConversion : public OpConversionPattern<VariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op.getLoc(), "invalid variable type");

    // Get the original Moore type to detect associative arrays
    auto mooreRefType = cast<moore::RefType>(op.getResult().getType());
    auto nestedMooreType = mooreRefType.getNestedType();

    // Handle associative array variables - these need runtime allocation
    if (auto assocType = dyn_cast<AssocArrayType>(nestedMooreType)) {
      auto *ctx = rewriter.getContext();
      ModuleOp mod = op->getParentOfType<ModuleOp>();

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get or create __moore_assoc_create function
      auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {i32Ty, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_create", fnTy);

      // Determine key size (0 for string keys)
      int32_t keySize = 0;
      auto keyType = assocType.getIndexType();
      if (!isa<StringType>(keyType)) {
        // For non-string keys, get the bit width
        auto convertedKeyType = typeConverter->convertType(keyType);
        if (auto intTy = dyn_cast<IntegerType>(convertedKeyType))
          keySize = intTy.getWidth() / 8;
        else
          keySize = 8; // Default to 64-bit for unknown types
      }

      // Determine value size
      auto valueType = assocType.getElementType();
      auto convertedValueType = typeConverter->convertType(valueType);
      int32_t valueSize = 4; // Default
      if (auto intTy = dyn_cast<IntegerType>(convertedValueType))
        valueSize = (intTy.getWidth() + 7) / 8;

      // Create constants for key_size and value_size
      auto keySizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(keySize));
      auto valueSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

      // Call __moore_assoc_create(key_size, value_size)
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
          ValueRange{keySizeConst, valueSizeConst});

      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // Handle string variables - these need stack allocation with empty init
    if (isa<StringType>(nestedMooreType)) {
      auto *ctx = rewriter.getContext();
      // Get the struct type by converting the nested string type directly
      auto structTy = cast<LLVM::LLVMStructType>(
          typeConverter->convertType(nestedMooreType));

      // Create an alloca for the string struct
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto alloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one);

      // Initialize with empty string {nullptr, 0}
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto zeroLen = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(0));

      // Build the struct value
      Value structVal = LLVM::UndefOp::create(rewriter, loc, structTy);
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, nullPtr,
                                              ArrayRef<int64_t>{0});
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, zeroLen,
                                              ArrayRef<int64_t>{1});

      // Store to alloca
      LLVM::StoreOp::create(rewriter, loc, structVal, alloca);

      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

    // Handle queue variables - these need stack allocation with empty init
    if (isa<QueueType>(nestedMooreType)) {
      auto *ctx = rewriter.getContext();
      // Get the struct type by converting the nested queue type directly
      auto structTy = cast<LLVM::LLVMStructType>(
          typeConverter->convertType(nestedMooreType));

      // Create an alloca for the queue struct
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto alloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one);

      // Initialize with empty queue {nullptr, 0}
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto zeroLen = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(0));

      // Build the struct value
      Value structVal = LLVM::UndefOp::create(rewriter, loc, structTy);
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, nullPtr,
                                              ArrayRef<int64_t>{0});
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, zeroLen,
                                              ArrayRef<int64_t>{1});

      // Store to alloca
      LLVM::StoreOp::create(rewriter, loc, structVal, alloca);

      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

    // Handle dynamic array variables - these need stack allocation with empty init
    if (isa<OpenUnpackedArrayType>(nestedMooreType)) {
      auto *ctx = rewriter.getContext();
      // Get the struct type by converting the nested array type directly
      auto structTy = cast<LLVM::LLVMStructType>(
          typeConverter->convertType(nestedMooreType));

      // Create an alloca for the array struct
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto alloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one);

      // Initialize with empty array {nullptr, 0}
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto zeroLen = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(0));

      // Build the struct value
      Value structVal = LLVM::UndefOp::create(rewriter, loc, structTy);
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, nullPtr,
                                              ArrayRef<int64_t>{0});
      structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal, zeroLen,
                                              ArrayRef<int64_t>{1});

      // Store to alloca
      LLVM::StoreOp::create(rewriter, loc, structVal, alloca);

      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

    // Handle unpacked struct variables containing dynamic types (strings, etc.)
    // These get converted to LLVM struct types, not hw::StructType
    if (auto unpackedStructType = dyn_cast<UnpackedStructType>(nestedMooreType)) {
      auto convertedType = typeConverter->convertType(nestedMooreType);
      if (auto structTy = dyn_cast<LLVM::LLVMStructType>(convertedType)) {
        auto *ctx = rewriter.getContext();
        auto ptrTy = LLVM::LLVMPointerType::get(ctx);

        // Create an alloca for the struct
        auto one = LLVM::ConstantOp::create(rewriter, loc,
                                            rewriter.getI64IntegerAttr(1));
        auto alloca =
            LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one);

        // Initialize with zero values - use LLVM's ZeroOp to create a zeroed
        // struct
        auto zeroVal = LLVM::ZeroOp::create(rewriter, loc, structTy);

        // Store the zero-initialized value to alloca
        LLVM::StoreOp::create(rewriter, loc, zeroVal, alloca);

        rewriter.replaceOp(op, alloca.getResult());
        return success();
      }
    }

    // For dynamic container types (queues, dynamic arrays, associative arrays),
    // the converted type is an LLVM pointer, not llhd.ref. These are handled
    // differently since they don't fit the llhd signal model.
    auto refType = dyn_cast<llhd::RefType>(resultType);
    if (!refType)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "variable type not supported for conversion");

    // Determine the initial value of the signal.
    Value init = adaptor.getInitial();
    if (!init) {
      auto elementType = refType.getNestedType();
      init = createZeroValue(elementType, loc, rewriter);
      if (!init)
        return failure();
    }

    rewriter.replaceOpWithNewOp<llhd::SignalOp>(op, resultType,
                                                op.getNameAttr(), init);
    return success();
  }
};

struct NetOpConversion : public OpConversionPattern<NetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getKind() != NetKind::Wire)
      return rewriter.notifyMatchFailure(loc, "only wire nets supported");

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(loc, "invalid net type");

    // TODO: Once the core dialects support four-valued integers, this code
    // will additionally need to generate an all-X value for four-valued nets.
    auto refType = dyn_cast<llhd::RefType>(resultType);
    if (!refType)
      return rewriter.notifyMatchFailure(loc,
                                         "net type must convert to llhd.ref");
    auto elementType = refType.getNestedType();
    int64_t width = hw::getBitWidth(elementType);
    if (width == -1)
      return failure();
    auto constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
    auto init =
        rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);

    auto signal = rewriter.replaceOpWithNewOp<llhd::SignalOp>(
        op, resultType, op.getNameAttr(), init);

    if (auto assignedValue = adaptor.getAssignment()) {
      auto timeAttr = llhd::TimeAttr::get(resultType.getContext(), 0U,
                                          llvm::StringRef("ns"), 0, 1);
      auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
      llhd::DriveOp::create(rewriter, loc, signal, assignedValue, time,
                            Value{});
    }

    return success();
  }
};


//===----------------------------------------------------------------------===//
// Global Variable Conversion
//===----------------------------------------------------------------------===//

/// Helper to create an LLVM zero initializer for a given type.
static Attribute createLLVMZeroAttr(Type type, MLIRContext *ctx) {
  // Handle LLVM struct and pointer types with ZeroAttr.
  if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType>(type))
    return LLVM::ZeroAttr::get(ctx);

  // Handle integer types.
  if (auto intTy = dyn_cast<IntegerType>(type))
    return IntegerAttr::get(intTy, 0);

  // Handle float types.
  if (auto floatTy = dyn_cast<FloatType>(type))
    return FloatAttr::get(floatTy, 0.0);

  // Handle hw::ArrayType.
  if (isa<hw::ArrayType>(type)) {
    int64_t width = hw::getBitWidth(type);
    if (width > 0) {
      auto intTy = IntegerType::get(ctx, width);
      return IntegerAttr::get(intTy, 0);
    }
  }

  // Handle hw::StructType by creating a zero-initialized integer.
  if (isa<hw::StructType>(type)) {
    int64_t width = hw::getBitWidth(type);
    if (width > 0) {
      auto intTy = IntegerType::get(ctx, width);
      return IntegerAttr::get(intTy, 0);
    }
  }

  // Default: return zero attr for types with known bit width.
  int64_t width = hw::getBitWidth(type);
  if (width > 0) {
    auto intTy = IntegerType::get(ctx, width);
    return IntegerAttr::get(intTy, 0);
  }

  return {};
}

struct GlobalVariableOpConversion
    : public OpConversionPattern<GlobalVariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GlobalVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type convertedType = typeConverter->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(loc,
                                         "failed to convert global var type");

    // Create an LLVM global with zero initialization.
    Attribute initAttr = createLLVMZeroAttr(convertedType, op.getContext());

    LLVM::GlobalOp::create(rewriter, loc, convertedType, /*isConstant=*/false,
                           LLVM::Linkage::Internal, op.getSymName(), initAttr);

    rewriter.eraseOp(op);
    return success();
  }
};

struct GetGlobalVariableOpConversion
    : public OpConversionPattern<GetGlobalVariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetGlobalVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(loc,
                                         "failed to convert result type");

    auto refType = dyn_cast<llhd::RefType>(resultType);

    auto ptrTy = LLVM::LLVMPointerType::get(op.getContext());
    auto addressOf =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, op.getGlobalName());

    if (refType) {
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
          op, resultType, ValueRange{addressOf.getResult()});
    } else {
      rewriter.replaceOp(op, addressOf.getResult());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FIXME: Discard unknown bits and map them to 0 for now.
    auto value = op.getValue().toAPInt(false);
    auto type = rewriter.getIntegerType(value.getBitWidth());
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, type, rewriter.getIntegerAttr(type, value));
    return success();
  }
};

struct ConstantRealOpConv : public OpConversionPattern<ConstantRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct ConstantTimeOpConv : public OpConversionPattern<ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::ConstantTimeOp>(
        op, llhd::TimeAttr::get(op->getContext(), op.getValue(),
                                StringRef("fs"), 0, 0));
    return success();
  }
};

struct ConstantStringOpConv : public OpConversionPattern<ConstantStringOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::ConstantStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto resultType =
        typeConverter->convertType(op.getResult().getType());
    const auto intType = mlir::cast<IntegerType>(resultType);

    const auto str = op.getValue();
    const unsigned byteWidth = intType.getWidth();
    APInt value(byteWidth, 0);

    // Pack ascii chars from the end of the string, until it fits.
    const size_t maxChars =
        std::min(str.size(), static_cast<size_t>(byteWidth / 8));
    for (size_t i = 0; i < maxChars; i++) {
      const size_t pos = str.size() - 1 - i;
      const auto asciiChar = static_cast<uint8_t>(str[pos]);
      value |= APInt(byteWidth, asciiChar) << (8 * i);
    }

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, resultType, rewriter.getIntegerAttr(resultType, value));
    return success();
  }
};

struct ConcatOpConversion : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType,
                                                   adaptor.getValue());
    return success();
  }
};

struct ExtractOpConversion : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: return X if the domain is four-valued for out-of-bounds accesses
    // once we support four-valued lowering
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType = adaptor.getInput().getType();
    int32_t low = adaptor.getLowBit();

    if (isa<IntegerType>(inputType)) {
      int32_t inputWidth = inputType.getIntOrFloatBitWidth();
      int32_t resultWidth = hw::getBitWidth(resultType);
      int32_t high = low + resultWidth;

      SmallVector<Value> toConcat;
      if (low < 0)
        toConcat.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(std::min(-low, resultWidth), 0)));

      if (low < inputWidth && high > 0) {
        int32_t lowIdx = std::max(low, 0);
        Value middle = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(),
            rewriter.getIntegerType(
                std::min(resultWidth, std::min(high, inputWidth) - lowIdx)),
            adaptor.getInput(), lowIdx);
        toConcat.push_back(middle);
      }

      int32_t diff = high - inputWidth;
      if (diff > 0) {
        Value val =
            hw::ConstantOp::create(rewriter, op.getLoc(), APInt(diff, 0));
        toConcat.push_back(val);
      }

      Value concat =
          rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), toConcat);
      rewriter.replaceOp(op, concat);
      return success();
    }

    if (auto arrTy = dyn_cast<hw::ArrayType>(inputType)) {
      int32_t width = llvm::Log2_64_Ceil(arrTy.getNumElements());
      int32_t inputWidth = arrTy.getNumElements();

      if (auto resArrTy = dyn_cast<hw::ArrayType>(resultType);
          resArrTy && resArrTy != arrTy.getElementType()) {
        int32_t elementWidth = hw::getBitWidth(arrTy.getElementType());
        if (elementWidth < 0)
          return failure();

        int32_t high = low + resArrTy.getNumElements();
        int32_t resWidth = resArrTy.getNumElements();

        SmallVector<Value> toConcat;
        if (low < 0) {
          Value val = hw::ConstantOp::create(
              rewriter, op.getLoc(),
              APInt(std::min((-low) * elementWidth, resWidth * elementWidth),
                    0));
          Value res = rewriter.createOrFold<hw::BitcastOp>(
              op.getLoc(), hw::ArrayType::get(arrTy.getElementType(), -low),
              val);
          toConcat.push_back(res);
        }

        if (low < inputWidth && high > 0) {
          int32_t lowIdx = std::max(0, low);
          Value lowIdxVal = hw::ConstantOp::create(
              rewriter, op.getLoc(), rewriter.getIntegerType(width), lowIdx);
          Value middle = rewriter.createOrFold<hw::ArraySliceOp>(
              op.getLoc(),
              hw::ArrayType::get(
                  arrTy.getElementType(),
                  std::min(resWidth, std::min(inputWidth, high) - lowIdx)),
              adaptor.getInput(), lowIdxVal);
          toConcat.push_back(middle);
        }

        int32_t diff = high - inputWidth;
        if (diff > 0) {
          Value constZero = hw::ConstantOp::create(
              rewriter, op.getLoc(), APInt(diff * elementWidth, 0));
          Value val = hw::BitcastOp::create(
              rewriter, op.getLoc(),
              hw::ArrayType::get(arrTy.getElementType(), diff), constZero);
          toConcat.push_back(val);
        }

        Value concat =
            rewriter.createOrFold<hw::ArrayConcatOp>(op.getLoc(), toConcat);
        rewriter.replaceOp(op, concat);
        return success();
      }

      // Otherwise, it has to be the array's element type
      if (low < 0 || low >= inputWidth) {
        int32_t bw = hw::getBitWidth(resultType);
        if (bw < 0)
          return failure();

        Value val = hw::ConstantOp::create(rewriter, op.getLoc(), APInt(bw, 0));
        Value bitcast =
            rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), resultType, val);
        rewriter.replaceOp(op, bitcast);
        return success();
      }

      Value idx = hw::ConstantOp::create(rewriter, op.getLoc(),
                                         rewriter.getIntegerType(width),
                                         adaptor.getLowBit());
      rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(), idx);
      return success();
    }

    return failure();
  }
};

struct ExtractRefOpConversion : public OpConversionPattern<ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // The input must be an llhd.ref type (not LLVM pointer which is used for
    // dynamic containers like strings, queues, etc.)
    auto inputRefType = dyn_cast<llhd::RefType>(adaptor.getInput().getType());
    if (!inputRefType)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "input type must be llhd.ref, not LLVM pointer");
    Type inputType = inputRefType.getNestedType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      int64_t width = hw::getBitWidth(inputType);
      if (width == -1)
        return failure();

      Value lowBit = hw::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerType(llvm::Log2_64_Ceil(width)),
          adaptor.getLowBit());
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultType, adaptor.getInput(), lowBit);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      Value lowBit = hw::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerType(llvm::Log2_64_Ceil(arrType.getNumElements())),
          adaptor.getLowBit());

      // If the result type is not the same as the array's element type, then
      // it has to be a slice.
      auto resultRefType = dyn_cast<llhd::RefType>(resultType);
      if (!resultRefType)
        return rewriter.notifyMatchFailure(
            op.getLoc(), "result type must be llhd.ref for array extraction");
      if (arrType.getElementType() != resultRefType.getNestedType()) {
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, adaptor.getInput(), lowBit);
        return success();
      }

      rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, adaptor.getInput(),
                                                       lowBit);
      return success();
    }

    return failure();
  }
};

struct DynExtractOpConversion : public OpConversionPattern<DynExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType = adaptor.getInput().getType();

    // Handle associative array element access (by value)
    // The input is an LLVM pointer (assoc array handle)
    if (isa<LLVM::LLVMPointerType>(inputType)) {
      auto *ctx = rewriter.getContext();
      ModuleOp mod = op->getParentOfType<ModuleOp>();

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Get or create __moore_assoc_get_ref function
      auto i32Ty = IntegerType::get(ctx, 32);
      auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_get_ref", fnTy);

      // Get the value size
      int32_t valueSize = 4; // Default
      if (auto intTy = dyn_cast<IntegerType>(resultType))
        valueSize = (intTy.getWidth() + 7) / 8;

      // Store the key to an alloca and pass its pointer
      auto keyType = adaptor.getLowBit().getType();
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getLowBit(), keyAlloca);

      auto valueSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

      // Call __moore_assoc_get_ref(array, key_ptr, value_size) -> value_ptr
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
          ValueRange{adaptor.getInput(), keyAlloca, valueSizeConst});

      // Load the value from the returned pointer
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, call.getResult());
      return success();
    }

    // Handle queue and dynamic array element access (by value)
    // Both are lowered to {ptr, length} structs
    if (isa<QueueType, OpenUnpackedArrayType>(op.getInput().getType())) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Extract the data pointer from field 0 of the struct
      Value dataPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                   adaptor.getInput(),
                                                   ArrayRef<int64_t>{0});

      // Convert index to i64 for GEP
      auto i64Ty = IntegerType::get(ctx, 64);
      Value idx = adaptor.getLowBit();
      if (idx.getType() != i64Ty) {
        if (cast<IntegerType>(idx.getType()).getWidth() < 64)
          idx = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
        else
          idx = arith::TruncIOp::create(rewriter, loc, i64Ty, idx);
      }

      // GEP to the i-th element
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, resultType,
                                          dataPtr, ValueRange{idx});

      // Load and return the element value
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, elemPtr);
      return success();
    }

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      Value amount = adjustIntegerWidth(rewriter, adaptor.getLowBit(),
                                        intType.getWidth(), loc);
      Value value = comb::ShrUOp::create(rewriter, loc,
                                         adaptor.getInput(), amount);

      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
      Value idx = adjustIntegerWidth(rewriter, adaptor.getLowBit(), idxWidth,
                                     loc);

      bool isSingleElementExtract = arrType.getElementType() == resultType;

      if (isSingleElementExtract)
        rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(),
                                                    idx);
      else
        rewriter.replaceOpWithNewOp<hw::ArraySliceOp>(op, resultType,
                                                      adaptor.getInput(), idx);

      return success();
    }

    return failure();
  }
};

struct DynExtractRefOpConversion : public OpConversionPattern<DynExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Handle dynamic container element access
    // When input is LLVM pointer, check the original Moore type to determine
    // the container type (associative array, queue, dynamic array, or string)
    if (isa<LLVM::LLVMPointerType>(adaptor.getInput().getType())) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Get the nested type from the input ref type
      auto mooreInputRefType = cast<moore::RefType>(op.getInput().getType());
      auto nestedType = mooreInputRefType.getNestedType();

      // Handle queue and dynamic array element access (ref version)
      // ref<queue<T>> or ref<open_uarray<T>> -> ptr to {ptr, i64} struct
      Type elemMooreType;
      if (auto queueType = dyn_cast<QueueType>(nestedType))
        elemMooreType = queueType.getElementType();
      else if (auto dynArrayType = dyn_cast<OpenUnpackedArrayType>(nestedType))
        elemMooreType = dynArrayType.getElementType();

      if (elemMooreType) {
        auto i64Ty = IntegerType::get(ctx, 64);
        auto containerStructTy =
            LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

        // Load the container struct from the ref
        Value containerStruct = LLVM::LoadOp::create(
            rewriter, loc, containerStructTy, adaptor.getInput());

        // Extract the data pointer from field 0
        Value dataPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                     containerStruct,
                                                     ArrayRef<int64_t>{0});

        // Convert index to i64 for GEP
        Value idx = adaptor.getLowBit();
        if (idx.getType() != i64Ty) {
          if (cast<IntegerType>(idx.getType()).getWidth() < 64)
            idx = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
          else
            idx = arith::TruncIOp::create(rewriter, loc, i64Ty, idx);
        }

        // Get the element type
        auto elemType = typeConverter->convertType(elemMooreType);

        // GEP to the i-th element and return the pointer
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                            dataPtr, ValueRange{idx});

        rewriter.replaceOp(op, elemPtr);
        return success();
      }

      // Handle associative array element access
      ModuleOp mod = op->getParentOfType<ModuleOp>();

      // Get or create __moore_assoc_get_ref function
      // Signature: (array_ptr, key_ptr, value_size) -> element_ptr
      auto i32Ty = IntegerType::get(ctx, 32);
      auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_get_ref", fnTy);

      // Get the value size from the original Moore type
      auto mooreRefType = cast<moore::RefType>(op.getResult().getType());
      auto valueType = mooreRefType.getNestedType();
      auto convertedValueType = typeConverter->convertType(valueType);
      int32_t valueSize = 4; // Default
      if (auto intTy = dyn_cast<IntegerType>(convertedValueType))
        valueSize = (intTy.getWidth() + 7) / 8;

      // Store the key to an alloca and pass its pointer
      auto keyType = adaptor.getLowBit().getType();
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getLowBit(), keyAlloca);

      auto valueSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

      // Call __moore_assoc_get_ref(array, key_ptr, value_size)
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
          ValueRange{adaptor.getInput(), keyAlloca, valueSizeConst});

      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // The input must be an llhd.ref type at this point (not LLVM pointer)
    auto inputRefType = dyn_cast<llhd::RefType>(adaptor.getInput().getType());
    if (!inputRefType)
      return rewriter.notifyMatchFailure(
          loc, "input type must be llhd.ref, not LLVM pointer");
    Type inputType = inputRefType.getNestedType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      int64_t width = hw::getBitWidth(inputType);
      if (width == -1)
        return failure();

      Value amount =
          adjustIntegerWidth(rewriter, adaptor.getLowBit(),
                             llvm::Log2_64_Ceil(width), loc);
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultType, adaptor.getInput(), amount);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      Value idx = adjustIntegerWidth(
          rewriter, adaptor.getLowBit(),
          llvm::Log2_64_Ceil(arrType.getNumElements()), loc);

      auto resultRefType = dyn_cast<llhd::RefType>(resultType);
      if (!resultRefType)
        return rewriter.notifyMatchFailure(
            loc, "result type must be llhd.ref for array extraction");
      auto resultNestedType = resultRefType.getNestedType();
      bool isSingleElementExtract =
          arrType.getElementType() == resultNestedType;

      if (isSingleElementExtract)
        rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, adaptor.getInput(),
                                                         idx);
      else
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, adaptor.getInput(), idx);

      return success();
    }

    return failure();
  }
};

struct ArrayCreateOpConversion : public OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<hw::ArrayCreateOp>(op, resultType,
                                                   adaptor.getElements());
    return success();
  }
};

struct StructCreateOpConversion : public OpConversionPattern<StructCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Check if this converts to an LLVM struct type (has dynamic fields).
    // If so, we need to use LLVM::UndefOp + LLVM::InsertValueOp instead of
    // hw::StructCreateOp.
    if (isa<LLVM::LLVMStructType>(resultType)) {
      auto loc = op.getLoc();
      // Create an undef struct and insert each field
      Value result = LLVM::UndefOp::create(rewriter, loc, resultType);
      for (auto [idx, field] : llvm::enumerate(adaptor.getFields())) {
        result = LLVM::InsertValueOp::create(rewriter, loc, result, field,
                                             ArrayRef<int64_t>{(int64_t)idx});
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    // Default: use hw::StructCreateOp for hw::StructType
    rewriter.replaceOpWithNewOp<hw::StructCreateOp>(op, resultType,
                                                    adaptor.getFields());
    return success();
  }
};

struct StructExtractOpConversion : public OpConversionPattern<StructExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = op.getInput().getType();

    // Check if this is an unpacked struct with dynamic fields (strings, classes,
    // etc.). These convert to LLVM struct types instead of hw::StructType, so we
    // need to use LLVM::ExtractValueOp instead of hw::StructExtractOp.
    bool needsLLVMHandling = false;
    SmallVector<StringAttr> fieldNames;

    if (auto unpackedStruct = dyn_cast<moore::UnpackedStructType>(inputType)) {
      auto convertedType = typeConverter->convertType(unpackedStruct);
      if (isa<LLVM::LLVMStructType>(convertedType)) {
        needsLLVMHandling = true;
      }
      for (auto &member : unpackedStruct.getMembers())
        fieldNames.push_back(member.name);
    } else if (auto packedStruct = dyn_cast<moore::StructType>(inputType)) {
      auto convertedType = typeConverter->convertType(packedStruct);
      if (isa<LLVM::LLVMStructType>(convertedType)) {
        needsLLVMHandling = true;
      }
      for (auto &member : packedStruct.getMembers())
        fieldNames.push_back(member.name);
    }

    if (needsLLVMHandling) {
      // Find the field index by name
      auto fieldName = op.getFieldNameAttr();
      int64_t fieldIndex = -1;
      for (size_t i = 0; i < fieldNames.size(); i++) {
        if (fieldNames[i] == fieldName) {
          fieldIndex = i;
          break;
        }
      }
      if (fieldIndex < 0) {
        return op.emitError("field '")
               << fieldName.getValue() << "' not found in struct";
      }

      // Convert the result type
      auto resultType = typeConverter->convertType(op.getResult().getType());
      if (!resultType)
        return failure();

      // Use LLVM::ExtractValueOp for LLVM struct types
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
          op, resultType, adaptor.getInput(), ArrayRef<int64_t>{fieldIndex});
      return success();
    }

    // Default: use hw::StructExtractOp for hw::StructType
    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct StructExtractRefOpConversion
    : public OpConversionPattern<StructExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get the original Moore type to determine if LLVM handling is needed.
    // Check if the struct type converts to LLVM struct (has dynamic fields).
    // This is more reliable than checking the adaptor type which may not
    // be converted yet for function block arguments.
    auto mooreRefType = cast<moore::RefType>(op.getInput().getType());
    auto nestedType = mooreRefType.getNestedType();

    bool needsLLVMHandling = false;
    LLVM::LLVMStructType llvmStructType;
    SmallVector<StringAttr> fieldNames;

    if (auto unpackedStruct = dyn_cast<moore::UnpackedStructType>(nestedType)) {
      auto convertedType = typeConverter->convertType(unpackedStruct);
      if (isa<LLVM::LLVMStructType>(convertedType)) {
        needsLLVMHandling = true;
        llvmStructType = cast<LLVM::LLVMStructType>(convertedType);
      }
      for (auto &member : unpackedStruct.getMembers())
        fieldNames.push_back(member.name);
    } else if (auto packedStruct = dyn_cast<moore::StructType>(nestedType)) {
      auto convertedType = typeConverter->convertType(packedStruct);
      if (isa<LLVM::LLVMStructType>(convertedType)) {
        needsLLVMHandling = true;
        llvmStructType = cast<LLVM::LLVMStructType>(convertedType);
      }
      for (auto &member : packedStruct.getMembers())
        fieldNames.push_back(member.name);
    } else {
      return rewriter.notifyMatchFailure(
          loc, "expected struct type for struct_extract_ref");
    }

    // If struct has dynamic fields (converts to LLVM struct), use GEP
    if (needsLLVMHandling) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Find the field index
      unsigned fieldIndex = 0;
      for (auto name : fieldNames) {
        if (name == op.getFieldNameAttr())
          break;
        ++fieldIndex;
      }

      // GEP to the field
      auto zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
      auto fieldIdx = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(fieldIndex));
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, ptrTy, llvmStructType, adaptor.getInput(),
          ValueRange{zero, fieldIdx});
      return success();
    }

    // Otherwise use LLHD sig struct extract for signal types.
    rewriter.replaceOpWithNewOp<llhd::SigStructExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct UnionCreateOpConversion : public OpConversionPattern<UnionCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<hw::UnionCreateOp>(
        op, resultType, adaptor.getFieldNameAttr(), adaptor.getInput());
    return success();
  }
};

struct UnionExtractOpConversion : public OpConversionPattern<UnionExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::UnionExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct UnionExtractRefOpConversion
    : public OpConversionPattern<UnionExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For union references, all members share the same memory location.
    // We use an UnrealizedConversionCastOp to change the reference type,
    // since LLHD doesn't have a native union extract ref operation.
    // This is semantically correct as unions are overlaid in memory.
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, resultType, adaptor.getInput());
    return success();
  }
};

struct ReduceAndOpConversion : public OpConversionPattern<ReduceAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value max = hw::ConstantOp::create(rewriter, op->getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              adaptor.getInput(), max);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value zero = hw::ConstantOp::create(rewriter, op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              adaptor.getInput(), zero);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    if (isa_and_nonnull<IntegerType>(resultType)) {
      Value zero =
          hw::ConstantOp::create(rewriter, op->getLoc(), resultType, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                adaptor.getInput(), zero);
      return success();
    }
    // Handle pointer types (virtual interfaces, class handles) - compare to null.
    if (isa_and_nonnull<LLVM::LLVMPointerType>(resultType)) {
      auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto nullPtr = LLVM::ZeroOp::create(rewriter, op->getLoc(), ptrTy);
      auto i1Ty = rewriter.getI1Type();
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, i1Ty, LLVM::ICmpPredicate::ne, adaptor.getInput(), nullPtr);
      return success();
    }
    return failure();
  }
};

struct NotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = hw::ConstantOp::create(rewriter, op.getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value zero = hw::ConstantOp::create(rewriter, op.getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, adaptor.getInput());
    return success();
  }
};

struct NegRealOpConversion : public OpConversionPattern<NegRealOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::NegFOp>(op, adaptor.getInput());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryRealOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct UnaryRealOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getValue());
    return success();
  }
};

/// Conversion pattern for $clog2(n) - ceiling of log base 2.
/// Implements IEEE 1800-2017 Section 20.8.1:
/// - $clog2(0) = 0
/// - $clog2(1) = 0
/// - $clog2(n) = ceil(log2(n)) for n > 1
///
/// Algorithm: clog2(n) = n <= 1 ? 0 : bitwidth - ctlz(n - 1)
struct Clog2BIOpConversion : public OpConversionPattern<Clog2BIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Clog2BIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto inputType = cast<IntegerType>(input.getType());
    unsigned bitWidth = inputType.getWidth();

    // Create constants
    Value zero = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, 0));
    Value one = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, 1));
    Value bitWidthVal =
        hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, bitWidth));

    // Compute n - 1
    Value nMinus1 = comb::SubOp::create(rewriter, loc, input, one);

    // Compute ctlz(n - 1) using LLVM intrinsic.
    // is_zero_poison = false because we handle the n <= 1 case with a mux.
    Value ctlz = LLVM::CountLeadingZerosOp::create(
        rewriter, loc, inputType, nMinus1, rewriter.getBoolAttr(false));

    // Compute bitwidth - ctlz(n - 1)
    Value result = comb::SubOp::create(rewriter, loc, bitWidthVal, ctlz);

    // Check if n <= 1 (i.e., n == 0 or n == 1)
    Value isZeroOrOne = comb::ICmpOp::create(
        rewriter, loc, comb::ICmpPredicate::ule, input, one, false);

    // Select: n <= 1 ? 0 : (bitwidth - ctlz(n - 1))
    rewriter.replaceOpWithNewOp<comb::MuxOp>(op, isZeroOrOne, zero, result,
                                             false);
    return success();
  }
};

/// Conversion pattern for $atan2(y, x) - two-argument arc-tangent.
/// Maps directly to math::Atan2Op.
struct Atan2BIOpConversion : public OpConversionPattern<Atan2BIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Atan2BIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<math::Atan2Op>(op, adaptor.getY(),
                                               adaptor.getX());
    return success();
  }
};

/// Conversion pattern for $hypot(x, y) = sqrt(x^2 + y^2).
/// MLIR's math dialect doesn't have a native hypot op, so we lower it manually.
struct HypotBIOpConversion : public OpConversionPattern<HypotBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HypotBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value x = adaptor.getX();
    Value y = adaptor.getY();

    // Compute x^2 and y^2
    Value x2 = arith::MulFOp::create(rewriter, loc, x, x);
    Value y2 = arith::MulFOp::create(rewriter, loc, y, y);

    // Compute x^2 + y^2
    Value sum = arith::AddFOp::create(rewriter, loc, x2, y2);

    // Compute sqrt(x^2 + y^2)
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, sum);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct ICmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename SourceOp, arith::CmpFPredicate pred>
struct FCmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(
        op, resultType, pred, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename SourceOp, bool withoutX>
struct CaseXZEqOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check each operand if it is a known constant and extract the X and/or Z
    // bits to be ignored.
    // TODO: Once the core dialects support four-valued integers, we will have
    // to create ops that extract X and Z bits from the operands, since we also
    // have to do the right casez/casex comparison on non-constant inputs.
    unsigned bitWidth = op.getLhs().getType().getWidth();
    auto ignoredBits = APInt::getZero(bitWidth);
    auto detectIgnoredBits = [&](Value value) {
      auto constOp = value.getDefiningOp<ConstantOp>();
      if (!constOp)
        return;
      auto constValue = constOp.getValue();
      if (withoutX)
        ignoredBits |= constValue.getZBits();
      else
        ignoredBits |= constValue.getUnknownBits();
    };
    detectIgnoredBits(op.getLhs());
    detectIgnoredBits(op.getRhs());

    // If we have detected any bits to be ignored, mask them in the operands for
    // the comparison.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!ignoredBits.isZero()) {
      ignoredBits.flipAllBits();
      auto maskOp = hw::ConstantOp::create(rewriter, op.getLoc(), ignoredBits);
      lhs = rewriter.createOrFold<comb::AndOp>(op.getLoc(), lhs, maskOp);
      rhs = rewriter.createOrFold<comb::AndOp>(op.getLoc(), rhs, maskOp);
    }

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, ICmpPredicate::ceq, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

struct ConversionOpConversion : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) {
      op.emitError("conversion result type is not currently supported");
      return failure();
    }

    // Handle class handle conversions (e.g., null class to specific class type)
    // Both the input and result are LLVM pointers, so we can directly use the
    // converted input value.
    if (isa<LLVM::LLVMPointerType>(resultType) &&
        isa<LLVM::LLVMPointerType>(adaptor.getInput().getType())) {
      // Class handles are all pointers, so the conversion is a no-op
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Handle ref<virtual_interface> to virtual_interface conversions.
    // This is a dereference operation that reads the pointer from the reference.
    // Check the original Moore type rather than the adaptor type, as the adaptor
    // type may not be converted yet for function block arguments.
    if (auto mooreRefType = dyn_cast<moore::RefType>(op.getInput().getType())) {
      if (isa<VirtualInterfaceType>(mooreRefType.getNestedType())) {
        // The converted result type is !llvm.ptr, and the converted input type
        // is !llhd.ref<!llvm.ptr>. We need to probe the reference to get the
        // pointer value.
        auto probeOp =
            llhd::ProbeOp::create(rewriter, loc, adaptor.getInput());
        rewriter.replaceOp(op, probeOp.getResult());
        return success();
      }
    }

    int64_t inputBw = hw::getBitWidth(adaptor.getInput().getType());
    int64_t resultBw = hw::getBitWidth(resultType);
    if (inputBw == -1 || resultBw == -1)
      return failure();

    Value input = rewriter.createOrFold<hw::BitcastOp>(
        loc, rewriter.getIntegerType(inputBw), adaptor.getInput());
    Value amount = adjustIntegerWidth(rewriter, input, resultBw, loc);

    Value result =
        rewriter.createOrFold<hw::BitcastOp>(loc, resultType, amount);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp>
struct BitcastConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConversionPattern::typeConverter;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (type == adaptor.getInput().getType())
      rewriter.replaceOp(op, adaptor.getInput());
    else
      rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, type, adaptor.getInput());
    return success();
  }
};

struct TruncOpConversion : public OpConversionPattern<TruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getInput(), 0,
                                                 op.getType().getWidth());
    return success();
  }
};

struct ZExtOpConversion : public OpConversionPattern<ZExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetWidth = op.getType().getWidth();
    auto inputWidth = op.getInput().getType().getWidth();

    auto zeroExt = hw::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerType(targetWidth - inputWidth), 0);

    rewriter.replaceOpWithNewOp<comb::ConcatOp>(
        op, ValueRange{zeroExt, adaptor.getInput()});
    return success();
  }
};

struct SExtOpConversion : public OpConversionPattern<SExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getType());
    auto value =
        comb::createOrFoldSExt(op.getLoc(), adaptor.getInput(), type, rewriter);
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct SIntToRealOpConversion : public OpConversionPattern<SIntToRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SIntToRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

struct UIntToRealOpConversion : public OpConversionPattern<UIntToRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UIntToRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

struct RealToIntOpConversion : public OpConversionPattern<RealToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RealToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::FPToSIOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

struct RealtobitsBIOpConversion : public OpConversionPattern<RealtobitsBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RealtobitsBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // $realtobits: Reinterpret f64 bits as i64
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getValue());
    return success();
  }
};

struct BitstorealBIOpConversion : public OpConversionPattern<BitstorealBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitstorealBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // $bitstoreal: Reinterpret i64 bits as f64
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getValue());
    return success();
  }
};

struct ShortrealtobitsBIOpConversion
    : public OpConversionPattern<ShortrealtobitsBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShortrealtobitsBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // $shortrealtobits: Reinterpret f32 bits as i32
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getValue());
    return success();
  }
};

struct BitstoshortrealBIOpConversion
    : public OpConversionPattern<BitstoshortrealBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitstoshortrealBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // $bitstoshortreal: Reinterpret i32 bits as f32
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

struct HWInstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, convResTypes, op.getInstanceName(), op.getModuleName(),
        adaptor.getOperands(), op.getArgNames(),
        op.getResultNames(), /*Parameter*/
        rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());
    return success();
  }
};

/// Conversion for func.call_indirect to handle type conversion.
/// The callee function type must be converted to use the converted argument
/// and result types.
struct CallIndirectOpConversion
    : public OpConversionPattern<func::CallIndirectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Get the converted callee operands (excluding the callee function itself)
    auto calleeOperands = adaptor.getCalleeOperands();

    // Build the new function type from converted input/result types
    SmallVector<Type> inputTypes;
    for (auto operand : calleeOperands)
      inputTypes.push_back(operand.getType());
    auto newFuncType =
        FunctionType::get(op.getContext(), inputTypes, convResTypes);

    // Get the adapted callee value
    Value callee = adaptor.getCallee();

    // Check if the callee needs type conversion
    // The callee may be an UnrealizedConversionCast if the type converter
    // inserted a materialization. We need to check if the callee's function
    // type matches our expected type.
    auto calleeType = dyn_cast<FunctionType>(callee.getType());

    if (calleeType && calleeType == newFuncType) {
      // Types match, use the callee directly
      rewriter.replaceOpWithNewOp<func::CallIndirectOp>(op, callee,
                                                        calleeOperands);
      return success();
    }

    // If the callee type doesn't match (which happens when the callee is from
    // a vtable.load_method that hasn't been converted yet), we need to create
    // a new function type and cast the callee.
    //
    // This can happen because:
    // 1. The callee comes from a materialization cast
    // 2. The original function type had Moore types that were converted

    // Create a cast to the correct function type
    auto castOp = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), TypeRange{newFuncType}, ValueRange{callee});

    rewriter.replaceOpWithNewOp<func::CallIndirectOp>(op, castOp.getResult(0),
                                                      calleeOperands);
    return success();
  }
};

struct UnrealizedConversionCastConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

struct ShlOpConversion : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct PowUOpConversion : public OpConversionPattern<PowUOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    Location loc = op->getLoc();

    Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    // zero extend both LHS & RHS to ensure the unsigned integers are
    // interpreted correctly when calculating power
    auto lhs = comb::ConcatOp::create(rewriter, loc, zeroVal, adaptor.getLhs());
    auto rhs = comb::ConcatOp::create(rewriter, loc, zeroVal, adaptor.getRhs());

    // lower the exponentiation via MLIR's math dialect
    auto pow = mlir::math::IPowIOp::create(rewriter, loc, lhs, rhs);

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, pow, 0);
    return success();
  }
};

struct PowSOpConversion : public OpConversionPattern<PowSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // utilize MLIR math dialect's math.ipowi to handle the exponentiation of
    // expression
    rewriter.replaceOpWithNewOp<mlir::math::IPowIOp>(
        op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the input was converted to an LLVM pointer (for queues, dynamic
    // arrays, etc.), use LLVM load instead of llhd.probe.
    if (isa<LLVM::LLVMPointerType>(adaptor.getInput().getType())) {
      auto resultType = typeConverter->convertType(op.getResult().getType());
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                                 adaptor.getInput());
    } else {
      rewriter.replaceOpWithNewOp<llhd::ProbeOp>(op, adaptor.getInput());
    }
    return success();
  }
};

struct AssignedVariableOpConversion
    : public OpConversionPattern<AssignedVariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssignedVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, adaptor.getInput(),
                                            adaptor.getNameAttr());
    return success();
  }
};

template <typename OpTy>
struct AssignOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dst = adaptor.getDst();

    // Check if destination is LLVM pointer (possibly through unrealized cast)
    // This handles strings, queues, dynamic arrays, and associative arrays.
    Value llvmPtrDst;
    if (isa<LLVM::LLVMPointerType>(dst.getType())) {
      llvmPtrDst = dst;
    } else if (auto castOp =
                   dst.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Look through unrealized_conversion_cast to find LLVM pointer
      if (castOp.getInputs().size() == 1 &&
          isa<LLVM::LLVMPointerType>(castOp.getInputs()[0].getType())) {
        llvmPtrDst = castOp.getInputs()[0];
      }
    }

    if (llvmPtrDst) {
      LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getSrc(),
                            llvmPtrDst);
      rewriter.eraseOp(op);
      return success();
    }

    // Determine the delay for the assignment.
    Value delay;
    if constexpr (std::is_same_v<OpTy, ContinuousAssignOp> ||
                  std::is_same_v<OpTy, BlockingAssignOp>) {
      // Blocking and continuous assignments get a 0ns 0d 1e delay.
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          llhd::TimeAttr::get(op->getContext(), 0U, "ns", 0, 1));
    } else if constexpr (std::is_same_v<OpTy, NonBlockingAssignOp>) {
      // Non-blocking assignments get a 0ns 1d 0e delay.
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          llhd::TimeAttr::get(op->getContext(), 0U, "ns", 1, 0));
    } else {
      // Delayed assignments have a delay operand.
      delay = adaptor.getDelay();
    }

    rewriter.replaceOpWithNewOp<llhd::DriveOp>(
        op, dst, adaptor.getSrc(), delay, Value{});
    return success();
  }
};

struct ConditionalOpConversion : public OpConversionPattern<ConditionalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: This lowering is only correct if the condition is two-valued. If
    // the condition is X or Z, both branches of the conditional must be
    // evaluated and merged with the appropriate lookup table. See documentation
    // for `ConditionalOp`.
    auto type = typeConverter->convertType(op.getType());

    auto hasNoWriteEffect = [](Region &region) {
      auto result = region.walk([](Operation *operation) {
        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(operation))
          if (!memOp.hasEffect<MemoryEffects::Write>() &&
              !memOp.hasEffect<MemoryEffects::Free>())
            return WalkResult::advance();

        if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return WalkResult::advance();

        return WalkResult::interrupt();
      });
      return !result.wasInterrupted();
    };

    if (hasNoWriteEffect(op.getTrueRegion()) &&
        hasNoWriteEffect(op.getFalseRegion())) {
      Operation *trueTerm = op.getTrueRegion().front().getTerminator();
      Operation *falseTerm = op.getFalseRegion().front().getTerminator();

      rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
      rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);

      Value convTrueVal = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), type, trueTerm->getOperand(0));
      Value convFalseVal = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), type, falseTerm->getOperand(0));

      rewriter.eraseOp(trueTerm);
      rewriter.eraseOp(falseTerm);

      rewriter.replaceOpWithNewOp<comb::MuxOp>(op, adaptor.getCondition(),
                                               convTrueVal, convFalseVal);
      return success();
    }

    auto ifOp =
        scf::IfOp::create(rewriter, op.getLoc(), type, adaptor.getCondition());
    rewriter.inlineRegionBefore(op.getTrueRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getFalseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());
    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResult());
    return success();
  }
};

template <typename SourceOp>
struct InPlaceOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

template <typename MooreOpTy, typename VerifOpTy>
struct AssertLikeOpConversion : public OpConversionPattern<MooreOpTy> {
  using OpConversionPattern<MooreOpTy>::OpConversionPattern;
  using OpAdaptor = typename MooreOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(MooreOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr label =
        op.getLabel().has_value()
            ? StringAttr::get(op->getContext(), op.getLabel().value())
            : StringAttr::get(op->getContext());
    rewriter.replaceOpWithNewOp<VerifOpTy>(op, adaptor.getCond(), mlir::Value(),
                                           label);
    return success();
  }
};

/// Lowering for moore.past operation.
/// The moore.past op captures a value from a previous clock cycle. In the
/// MooreToCore lowering, we convert it to an ltl.past operation for boolean
/// values (used in assertion conditions), or pass through the value directly
/// for non-boolean types (used in comparisons where the verification
/// infrastructure handles the past semantics).
struct PastOpConversion : public OpConversionPattern<PastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    int64_t delay = op.getDelay();

    // If the input is i1 (boolean), convert to ltl.past for proper LTL semantics
    if (input.getType().isInteger(1)) {
      rewriter.replaceOpWithNewOp<ltl::PastOp>(op, input, delay);
      return success();
    }

    // For non-boolean types (used in comparisons like `$past(val) == 0`),
    // we need to create a register chain to capture past values.
    // However, this requires a clock signal which isn't available in this
    // context. For now, we pass through the input value and rely on the
    // verification infrastructure to handle the past semantics correctly.
    //
    // TODO: Once clock information is propagated through assertions, implement
    // proper register-based delay using seq::CompRegOp.
    rewriter.replaceOp(op, input);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Format String Conversion
//===----------------------------------------------------------------------===//

struct FormatLiteralOpConversion : public OpConversionPattern<FormatLiteralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::FormatLiteralOp>(op, adaptor.getLiteral());
    return success();
  }
};

struct FormatConcatOpConversion : public OpConversionPattern<FormatConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::FormatStringConcatOp>(op,
                                                           adaptor.getInputs());
    return success();
  }
};

struct FormatIntOpConversion : public OpConversionPattern<FormatIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    char padChar = adaptor.getPadding() == IntPadding::Space ? 32 : 48;
    IntegerAttr padCharAttr = rewriter.getI8IntegerAttr(padChar);
    auto widthAttr = adaptor.getSpecifierWidthAttr();

    bool isLeftAligned = adaptor.getAlignment() == IntAlign::Left;
    BoolAttr isLeftAlignedAttr = rewriter.getBoolAttr(isLeftAligned);

    switch (op.getFormat()) {
    case IntFormat::Decimal:
      rewriter.replaceOpWithNewOp<sim::FormatDecOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, padCharAttr, widthAttr,
          adaptor.getIsSignedAttr());
      return success();
    case IntFormat::Binary:
      rewriter.replaceOpWithNewOp<sim::FormatBinOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::Octal:
      rewriter.replaceOpWithNewOp<sim::FormatOctOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::HexLower:
      rewriter.replaceOpWithNewOp<sim::FormatHexOp>(
          op, adaptor.getValue(), rewriter.getBoolAttr(false),
          isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::HexUpper:
      rewriter.replaceOpWithNewOp<sim::FormatHexOp>(
          op, adaptor.getValue(), rewriter.getBoolAttr(true), isLeftAlignedAttr,
          padCharAttr, widthAttr);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported int format");
  }
};

struct FormatClassOpConversion : public OpConversionPattern<FormatClassOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Emit a placeholder since actual object addresses are runtime-specific
    rewriter.replaceOpWithNewOp<sim::FormatLiteralOp>(op, "<object>");
    return success();
  }
};

struct FormatRealOpConversion : public OpConversionPattern<FormatRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fracDigitsAttr = adaptor.getFracDigitsAttr();

    auto fieldWidthAttr = adaptor.getFieldWidthAttr();
    bool isLeftAligned = adaptor.getAlignment() == IntAlign::Left;
    mlir::BoolAttr isLeftAlignedAttr = rewriter.getBoolAttr(isLeftAligned);

    switch (op.getFormat()) {
    case RealFormat::General:
      rewriter.replaceOpWithNewOp<sim::FormatGeneralOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    case RealFormat::Float:
      rewriter.replaceOpWithNewOp<sim::FormatFloatOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    case RealFormat::Exponential:
      rewriter.replaceOpWithNewOp<sim::FormatScientificOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    }
    llvm_unreachable("unhandled RealFormat");
  }
};

struct FormatStringOpConversion : public OpConversionPattern<FormatStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The input is a dynamic string (converted from moore::StringType to
    // LLVM struct {ptr, i64}). We convert it to sim::FormatDynStringOp
    // which produces a format string that can be used in display operations.
    rewriter.replaceOpWithNewOp<sim::FormatDynStringOp>(op, adaptor.getString());
    return success();
  }
};

struct DisplayBIOpConversion : public OpConversionPattern<DisplayBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DisplayBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

/// Conversion for moore.fstring_to_string -> runtime evaluation of format
/// string. This operation converts a format string (FormatStringType) to a
/// dynamic string (StringType). The conversion handles different cases:
/// - Literals: directly create a string from the literal value
/// - Dynamic strings (from FormatDynStringOp): pass through the input
/// - Formatted integers: call __moore_int_to_string runtime function
/// - Concatenations: recursively convert each input and concatenate
struct FormatStringToStringOpConversion
    : public OpConversionPattern<FormatStringToStringOp> {
  using OpConversionPattern::OpConversionPattern;

  // Helper to create an empty string
  static Value createEmptyString(ConversionPatternRewriter &rewriter,
                                 Location loc, MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    Value zero = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
    Value result = LLVM::UndefOp::create(rewriter, loc, stringStructTy);
    result = LLVM::InsertValueOp::create(rewriter, loc, result, nullPtr,
                                         ArrayRef<int64_t>{0});
    result = LLVM::InsertValueOp::create(rewriter, loc, result, zero,
                                         ArrayRef<int64_t>{1});
    return result;
  }

  // Helper to convert a format string value to a dynamic string
  // Returns the dynamic string value, or nullptr if conversion failed
  Value convertFormatStringToString(Value fmtValue, Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    ModuleOp mod) const {
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

    auto *defOp = fmtValue.getDefiningOp();
    if (!defOp)
      return createEmptyString(rewriter, loc, ctx);

    // Case 1: Format literal - create string from the literal value
    if (auto literalOp = dyn_cast<sim::FormatLiteralOp>(defOp)) {
      StringRef literal = literalOp.getLiteral();
      int64_t len = literal.size();

      if (len == 0)
        return createEmptyString(rewriter, loc, ctx);

      // Create global string constant
      auto globalName =
          "__moore_str_" +
          std::to_string(reinterpret_cast<uintptr_t>(literalOp.getOperation()));
      auto i8Ty = IntegerType::get(ctx, 8);
      auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, len);

      // Check if global already exists
      if (!mod.lookupSymbol<LLVM::GlobalOp>(globalName)) {
        // Create global string constant
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        LLVM::GlobalOp::create(rewriter, loc, arrayTy, /*isConstant=*/true,
                               LLVM::Linkage::Internal, globalName,
                               rewriter.getStringAttr(literal));
      }

      // Get address of global
      Value globalAddr =
          LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalName);
      Value lenVal = arith::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(len));

      // Create result struct
      Value result = LLVM::UndefOp::create(rewriter, loc, stringStructTy);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, globalAddr,
                                           ArrayRef<int64_t>{0});
      result = LLVM::InsertValueOp::create(rewriter, loc, result, lenVal,
                                           ArrayRef<int64_t>{1});
      return result;
    }

    // Case 2: Format dynamic string - the input is already a dynamic string
    if (auto dynStringOp = dyn_cast<sim::FormatDynStringOp>(defOp))
      return dynStringOp.getValue();

    // Case 3: Formatted integer - call __moore_int_to_string
    if (auto decOp = dyn_cast<sim::FormatDecOp>(defOp)) {
      Value intVal = decOp.getValue();
      auto intWidth = intVal.getType().getIntOrFloatBitWidth();

      // Extend or truncate to i64
      Value intI64;
      if (intWidth < 64)
        intI64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, intVal);
      else if (intWidth > 64)
        intI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, intVal);
      else
        intI64 = intVal;

      // Call __moore_int_to_string
      auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
      auto runtimeFn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_int_to_string", fnTy);
      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                       SymbolRefAttr::get(runtimeFn),
                                       ValueRange{intI64});
      return call.getResult();
    }

    // Case 4: Formatted hex integer
    if (auto hexOp = dyn_cast<sim::FormatHexOp>(defOp)) {
      // For now, use int_to_string as a fallback (TODO: implement hex version)
      Value intVal = hexOp.getValue();
      auto intWidth = intVal.getType().getIntOrFloatBitWidth();

      Value intI64;
      if (intWidth < 64)
        intI64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, intVal);
      else if (intWidth > 64)
        intI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, intVal);
      else
        intI64 = intVal;

      auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
      auto runtimeFn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_int_to_string", fnTy);
      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                       SymbolRefAttr::get(runtimeFn),
                                       ValueRange{intI64});
      return call.getResult();
    }

    // Case 5: Format concatenation - recursively convert each input
    if (auto concatOp = dyn_cast<sim::FormatStringConcatOp>(defOp)) {
      auto inputs = concatOp.getInputs();
      if (inputs.empty())
        return createEmptyString(rewriter, loc, ctx);

      // Convert the first input
      Value result = convertFormatStringToString(inputs[0], loc, rewriter, mod);
      if (!result)
        result = createEmptyString(rewriter, loc, ctx);

      // Concatenate the rest
      if (inputs.size() > 1) {
        // Get the string concat runtime function
        auto fnTy =
            LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, ptrTy});
        auto concatFn = getOrCreateRuntimeFunc(mod, rewriter,
                                               "__moore_string_concat", fnTy);

        auto one = LLVM::ConstantOp::create(rewriter, loc,
                                            rewriter.getI64IntegerAttr(1));

        for (size_t i = 1; i < inputs.size(); ++i) {
          Value nextStr =
              convertFormatStringToString(inputs[i], loc, rewriter, mod);
          if (!nextStr)
            nextStr = createEmptyString(rewriter, loc, ctx);

          // Allocate stack space for both strings
          auto lhsAlloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
          LLVM::StoreOp::create(rewriter, loc, result, lhsAlloca);

          auto rhsAlloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
          LLVM::StoreOp::create(rewriter, loc, nextStr, rhsAlloca);

          // Call __moore_string_concat
          auto call = LLVM::CallOp::create(
              rewriter, loc, TypeRange{stringStructTy},
              SymbolRefAttr::get(concatFn), ValueRange{lhsAlloca, rhsAlloca});
          result = call.getResult();
        }
      }
      return result;
    }

    // Unsupported format string type - return empty string
    return createEmptyString(rewriter, loc, ctx);
  }

  LogicalResult
  matchAndRewrite(FormatStringToStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    Value fmtString = adaptor.getFmtstring();

    Value result = convertFormatStringToString(fmtString, loc, rewriter, mod);
    if (!result) {
      auto *ctx = rewriter.getContext();
      result = createEmptyString(rewriter, loc, ctx);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// File I/O Operation Conversions
//===----------------------------------------------------------------------===//

/// Helper to get LLVM struct type for a string: {ptr, i64}.
static LLVM::LLVMStructType getFileIOStringStructType(MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
}

/// Conversion for moore.builtin.fopen -> runtime function call.
/// Lowers $fopen(filename, mode) to __moore_fopen(filename_ptr, mode_ptr).
struct FOpenBIOpConversion : public OpConversionPattern<FOpenBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FOpenBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringTy = getFileIOStringStructType(ctx);

    // Get or create __moore_fopen function: int32_t(ptr, ptr)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fopen", fnTy);

    // Allocate stack space for filename string and store it
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto filenameAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getFilename(), filenameAlloca);

    // Handle optional mode argument
    Value modePtr;
    if (adaptor.getMode()) {
      // Mode is provided - allocate and store it
      auto modeAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getMode(), modeAlloca);
      modePtr = modeAlloca;
    } else {
      // Mode not provided - pass null pointer (runtime will default to "r")
      modePtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Call __moore_fopen(filename_ptr, mode_ptr)
    auto call =
        LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                             SymbolRefAttr::get(fn),
                             ValueRange{filenameAlloca, modePtr});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.builtin.fwrite -> runtime function call.
/// Lowers $fwrite(fd, message) to __moore_fwrite(fd, message_ptr).
struct FWriteBIOpConversion : public OpConversionPattern<FWriteBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FWriteBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get or create __moore_fwrite function: void(i32, ptr)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fwrite", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Create alloca for the message string and store a placeholder
    auto stringTy = getFileIOStringStructType(ctx);
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto msgAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringTy, one);

    // Create a placeholder string: the actual message conversion requires
    // more complex handling of the format string type.
    auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    auto zeroLen =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));

    Value stringVal = LLVM::UndefOp::create(rewriter, loc, stringTy);
    stringVal = LLVM::InsertValueOp::create(rewriter, loc, stringVal, nullPtr,
                                            ArrayRef<int64_t>{0});
    stringVal = LLVM::InsertValueOp::create(rewriter, loc, stringVal, zeroLen,
                                            ArrayRef<int64_t>{1});
    LLVM::StoreOp::create(rewriter, loc, stringVal, msgAlloca);

    // Call __moore_fwrite(fd, message_ptr)
    LLVM::CallOp::create(rewriter, loc, TypeRange{},
                         SymbolRefAttr::get(fn),
                         ValueRange{fd, msgAlloca});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.fclose -> runtime function call.
/// Lowers $fclose(fd) to __moore_fclose(fd).
struct FCloseBIOpConversion : public OpConversionPattern<FCloseBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FCloseBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get or create __moore_fclose function: void(i32)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fclose", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_fclose(fd)
    LLVM::CallOp::create(rewriter, loc, TypeRange{},
                         SymbolRefAttr::get(fn),
                         ValueRange{fd});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.fgetc -> runtime function call.
/// Lowers $fgetc(fd) to __moore_fgetc(fd).
struct FGetCBIOpConversion : public OpConversionPattern<FGetCBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FGetCBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    // Get or create __moore_fgetc function: i32(i32)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fgetc", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_fgetc(fd)
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{fd});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.builtin.fgets -> runtime function call.
/// Lowers $fgets(str, fd) to __moore_fgets(str_ptr, fd).
struct FGetSBIOpConversion : public OpConversionPattern<FGetSBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FGetSBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get or create __moore_fgets function: i32(ptr, i32)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fgets", fnTy);

    // Get the string reference (already a ptr) and file descriptor
    Value strRef = adaptor.getStr();
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_fgets(str_ptr, fd)
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{strRef, fd});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.builtin.feof -> runtime function call.
/// Lowers $feof(fd) to __moore_feof(fd).
struct FEofBIOpConversion : public OpConversionPattern<FEofBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FEofBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    // Get or create __moore_feof function: i32(i32)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_feof", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_feof(fd)
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{fd});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.builtin.fflush -> runtime function call.
/// Lowers $fflush(fd) to __moore_fflush(fd).
struct FFlushBIOpConversion : public OpConversionPattern<FFlushBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FFlushBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get or create __moore_fflush function: void(i32)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_fflush", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_fflush(fd)
    LLVM::CallOp::create(rewriter, loc, TypeRange{},
                         SymbolRefAttr::get(fn),
                         ValueRange{fd});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.ftell -> runtime function call.
/// Lowers $ftell(fd) to __moore_ftell(fd).
struct FTellBIOpConversion : public OpConversionPattern<FTellBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FTellBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    // Get or create __moore_ftell function: i32(i32)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_ftell", fnTy);

    // Get the file descriptor (convert to i32 if needed)
    Value fd = adaptor.getFd();
    if (fd.getType() != i32Ty) {
      fd = LLVM::TruncOp::create(rewriter, loc, i32Ty, fd);
    }

    // Call __moore_ftell(fd)
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{fd});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct UArrayCmpOpConversion : public OpConversionPattern<UArrayCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UArrayCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    auto lhsType = adaptor.getLhs().getType();
    auto rhsType = adaptor.getRhs().getType();
    int64_t bitWidth = hw::getBitWidth(lhsType);
    if (bitWidth <= 0 || lhsType != rhsType) {
      bool sameValue = adaptor.getLhs() == adaptor.getRhs();
      bool result = op.getPredicate() == moore::UArrayCmpPredicate::eq
                        ? sameValue
                        : !sameValue;
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, resultType,
          rewriter.getIntegerAttr(resultType, result ? 1 : 0));
      return success();
    }

    auto intTy = rewriter.getIntegerType(bitWidth);
    Value lhsBits = adaptor.getLhs();
    Value rhsBits = adaptor.getRhs();
    if (lhsType != intTy)
      lhsBits = hw::BitcastOp::create(rewriter, loc, intTy, lhsBits);
    if (rhsType != intTy)
      rhsBits = hw::BitcastOp::create(rewriter, loc, intTy, rhsBits);

    ICmpPredicate pred = ICmpPredicate::eq;
    if (op.getPredicate() == moore::UArrayCmpPredicate::ne)
      pred = ICmpPredicate::ne;

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, lhsBits,
                                              rhsBits);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Queue and Array Operation Conversions
//===----------------------------------------------------------------------===//

/// Base class for queue/array operations that lower to runtime function calls.
template <typename SourceOp>
struct RuntimeCallConversionBase : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

protected:
  /// Get the LLVM struct type for a queue: {ptr, i64}.
  static LLVM::LLVMStructType getQueueStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }

  /// Get the LLVM struct type for a dynamic array: {ptr, i64}.
  static LLVM::LLVMStructType getDynArrayStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }

  /// Get the size in bytes for a type. Uses hw::getBitWidth which handles
  /// all types including structs. Returns at least 1 byte.
  static int64_t getTypeSizeInBytes(Type type) {
    int64_t bitWidth = hw::getBitWidth(type);
    if (bitWidth <= 0)
      return 1; // Default to 1 byte for unknown/opaque types
    int64_t byteSize = bitWidth / 8;
    return byteSize > 0 ? byteSize : 1;
  }
};

/// Conversion for moore.queue.max -> runtime function call.
struct QueueMaxOpConversion : public RuntimeCallConversionBase<QueueMaxOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_max", fnTy);

    // Store input to alloca and pass pointer.
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.queue.min -> runtime function call.
struct QueueMinOpConversion : public RuntimeCallConversionBase<QueueMinOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueMinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_min", fnTy);

    // Store input to alloca and pass pointer.
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.queue.delete -> runtime function call or erase op.
struct QueueDeleteOpConversion
    : public RuntimeCallConversionBase<QueueDeleteOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueDeleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    if (adaptor.getIndex()) {
      // delete(index) - remove element at specific index
      // Function signature: void delete_index(queue_ptr, index, element_size)
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_queue_delete_index", fnTy);

      // Get the element type from the queue reference type
      auto refType = cast<moore::RefType>(op.getQueue().getType());
      auto mooreQueueTy = cast<moore::QueueType>(refType.getNestedType());
      auto elemType = typeConverter->convertType(mooreQueueTy.getElementType());

      // Queue is passed by pointer, index by value, element_size by value
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto queueAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

      // Calculate element size
      auto elemSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

      // Truncate index to i32 if necessary
      Value indexVal = adaptor.getIndex();
      if (indexVal.getType() != i32Ty)
        indexVal = LLVM::TruncOp::create(rewriter, loc, i32Ty, indexVal);

      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(fn),
                           ValueRange{queueAlloca, indexVal, elemSize});
    } else {
      // delete() - clear all elements
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_clear", fnTy);

      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto queueAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                           ValueRange{queueAlloca});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.push_back -> runtime function call.
struct QueuePushBackOpConversion
    : public RuntimeCallConversionBase<QueuePushBackOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueuePushBackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void push_back(queue_ptr, element_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_push_back", fnTy);

    // Store queue to alloca and pass pointer
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto queueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

    // Store element to alloca and pass pointer
    auto elemType = typeConverter->convertType(op.getElement().getType());
    auto elemAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemType, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getElement(), elemAlloca);

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{queueAlloca, elemAlloca, elemSize});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.push_front -> runtime function call.
struct QueuePushFrontOpConversion
    : public RuntimeCallConversionBase<QueuePushFrontOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueuePushFrontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void push_front(queue_ptr, element_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_push_front", fnTy);

    // Store queue to alloca and pass pointer
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto queueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

    // Store element to alloca and pass pointer
    auto elemType = typeConverter->convertType(op.getElement().getType());
    auto elemAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemType, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getElement(), elemAlloca);

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{queueAlloca, elemAlloca, elemSize});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.pop_back -> runtime function call.
struct QueuePopBackOpConversion
    : public RuntimeCallConversionBase<QueuePopBackOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueuePopBackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Function signature: i64 pop_back(queue_ptr, element_size)
    // Returns the element as i64 (caller truncates/extends as needed)
    auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_pop_back", fnTy);

    // Store queue to alloca and pass pointer
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto queueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(resultType)));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{queueAlloca, elemSize});

    // Convert result to the expected type
    Value result = call.getResult();
    if (resultType.isIntOrFloat()) {
      auto resultWidth = resultType.getIntOrFloatBitWidth();
      if (resultWidth < 64) {
        result = arith::TruncIOp::create(rewriter, loc, resultType, result);
      } else if (resultWidth > 64) {
        result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
      }
    } else {
      // For non-integer types (structs, etc.), bitcast from i64
      result = LLVM::BitcastOp::create(rewriter, loc, resultType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.queue.pop_front -> runtime function call.
struct QueuePopFrontOpConversion
    : public RuntimeCallConversionBase<QueuePopFrontOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueuePopFrontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Function signature: i64 pop_front(queue_ptr, element_size)
    // Returns the element as i64 (caller truncates/extends as needed)
    auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_pop_front", fnTy);

    // Store queue to alloca and pass pointer
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto queueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(resultType)));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{queueAlloca, elemSize});

    // Convert result to the expected type
    Value result = call.getResult();
    if (resultType.isIntOrFloat()) {
      auto resultWidth = resultType.getIntOrFloatBitWidth();
      if (resultWidth < 64) {
        result = arith::TruncIOp::create(rewriter, loc, resultType, result);
      } else if (resultWidth > 64) {
        result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
      }
    } else {
      // For non-integer types (structs, etc.), bitcast from i64
      result = LLVM::BitcastOp::create(rewriter, loc, resultType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.queue.unique -> runtime function call.
struct QueueUniqueOpConversion
    : public RuntimeCallConversionBase<QueueUniqueOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueUniqueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_unique", fnTy);

    // Store input to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.queue.unique_index -> runtime function call.
struct QueueUniqueIndexOpConversion
    : public RuntimeCallConversionBase<QueueUniqueIndexOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueUniqueIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto inputType = op.getArray().getType();
    Type elementType;
    if (auto queueType = dyn_cast<moore::QueueType>(inputType))
      elementType = queueType.getElementType();
    else if (auto dynArrayType = dyn_cast<moore::OpenUnpackedArrayType>(inputType))
      elementType = dynArrayType.getElementType();
    else
      return failure();

    // Function signature: queue unique_index(queue_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_array_unique_index", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    auto elemType = typeConverter->convertType(elementType);
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca, elemSize});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.queue.reduce -> runtime function call.
struct QueueReduceOpConversion
    : public RuntimeCallConversionBase<QueueReduceOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto inputType = op.getArray().getType();
    Type elementType;
    if (auto queueType = dyn_cast<moore::QueueType>(inputType))
      elementType = queueType.getElementType();
    else if (auto dynArrayType = dyn_cast<moore::OpenUnpackedArrayType>(inputType))
      elementType = dynArrayType.getElementType();
    else
      return failure();

    // Function signature: i64 reduce(array_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
    StringRef fnName;
    switch (op.getKind()) {
    case moore::QueueReduceKind::Sum:
      fnName = "__moore_array_reduce_sum";
      break;
    case moore::QueueReduceKind::Product:
      fnName = "__moore_array_reduce_product";
      break;
    case moore::QueueReduceKind::And:
      fnName = "__moore_array_reduce_and";
      break;
    case moore::QueueReduceKind::Or:
      fnName = "__moore_array_reduce_or";
      break;
    case moore::QueueReduceKind::Xor:
      fnName = "__moore_array_reduce_xor";
      break;
    }

    auto fn = getOrCreateRuntimeFunc(mod, rewriter, fnName, fnTy);

    auto queueTy = getQueueStructType(ctx);
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    auto elemType = typeConverter->convertType(elementType);
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca, elemSize});

    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value result = call.getResult();
    if (resultType.isIntOrFloat()) {
      auto width = resultType.getIntOrFloatBitWidth();
      if (width < 64)
        result = arith::TruncIOp::create(rewriter, loc, resultType, result);
      else if (width > 64)
        result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
    } else {
      result = LLVM::BitcastOp::create(rewriter, loc, resultType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.queue.sort -> runtime function call.
/// Sorts the queue in-place.
struct QueueSortOpConversion : public RuntimeCallConversionBase<QueueSortOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueSortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void sort(queue_ptr)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_sort", fnTy);

    // Pass queue reference pointer directly
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getQueue()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.rsort -> runtime function call.
struct QueueRSortOpConversion : public RuntimeCallConversionBase<QueueRSortOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueRSortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_rsort", fnTy);

    auto refType = cast<moore::RefType>(op.getQueue().getType());
    auto nestedType = refType.getNestedType();
    Type elementType;
    if (auto queueType = dyn_cast<moore::QueueType>(nestedType))
      elementType = queueType.getElementType();
    else if (auto dynArrayType = dyn_cast<moore::OpenUnpackedArrayType>(nestedType))
      elementType = dynArrayType.getElementType();
    else
      return failure();

    auto elemType = typeConverter->convertType(elementType);
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getQueue(), elemSize});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.shuffle -> runtime function call.
struct QueueShuffleOpConversion
    : public RuntimeCallConversionBase<QueueShuffleOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_shuffle", fnTy);

    auto refType = cast<moore::RefType>(op.getQueue().getType());
    auto nestedType = refType.getNestedType();
    Type elementType;
    if (auto queueType = dyn_cast<moore::QueueType>(nestedType))
      elementType = queueType.getElementType();
    else if (auto dynArrayType = dyn_cast<moore::OpenUnpackedArrayType>(nestedType))
      elementType = dynArrayType.getElementType();
    else
      return failure();

    auto elemType = typeConverter->convertType(elementType);
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getQueue(), elemSize});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.concat -> runtime function call.
/// Concatenates multiple queues into a single queue.
struct QueueConcatOpConversion
    : public RuntimeCallConversionBase<QueueConcatOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    // Function signature: queue concat(ptr to array of queues, count)
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_concat", fnTy);

    auto inputs = adaptor.getInputs();
    auto numInputs = inputs.size();

    // Create an array to hold all input queues.
    auto count = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(numInputs));
    auto arrayAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, count);

    // Store each input queue into the array.
    for (size_t i = 0; i < numInputs; ++i) {
      auto idx = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(i));
      auto elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrTy, queueTy, arrayAlloca,
                              ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, inputs[i], elemPtr);
    }

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{arrayAlloca, count});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.queue.slice -> runtime function call.
struct QueueSliceOpConversion : public RuntimeCallConversionBase<QueueSliceOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    // Function signature: queue slice(queue_ptr, start, end, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i64Ty, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_slice", fnTy);

    // Store input queue to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto queueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getQueue(), queueAlloca);

    // Extend bounds to i64.
    auto startI64 =
        arith::ExtSIOp::create(rewriter, loc, i64Ty, adaptor.getStart());
    auto endI64 =
        arith::ExtSIOp::create(rewriter, loc, i64Ty, adaptor.getEnd());

    // Calculate element size in bytes.
    auto mooreQueueTy = cast<moore::QueueType>(op.getQueue().getType());
    auto elemType = typeConverter->convertType(mooreQueueTy.getElementType());
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
        ValueRange{queueAlloca, startI64, endI64, elemSize});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.array.size -> extract length from struct or runtime
/// call. For queues and dynamic arrays, the size is stored as field 1 of the
/// {ptr, i64} struct. For associative arrays, we call a runtime function.
struct ArraySizeOpConversion : public RuntimeCallConversionBase<ArraySizeOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(ArraySizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto inputType = op.getArray().getType();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // For queues and dynamic arrays, the size is field 1 of the struct
    if (isa<QueueType, OpenUnpackedArrayType>(inputType)) {
      // Extract field 1 (length) from the {ptr, i64} struct
      Value length = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty,
                                                  adaptor.getArray(),
                                                  ArrayRef<int64_t>{1});
      // Truncate to i32 (result type is TwoValuedI32)
      Value result = arith::TruncIOp::create(rewriter, loc, i32Ty, length);
      rewriter.replaceOp(op, result);
      return success();
    }

    // For associative arrays, call the runtime function
    if (isa<AssocArrayType>(inputType)) {
      ModuleOp mod = op->getParentOfType<ModuleOp>();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Function signature: i64 __moore_assoc_size(void* array)
      auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_size", fnTy);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{adaptor.getArray()});
      // Truncate to i32
      Value result =
          arith::TruncIOp::create(rewriter, loc, i32Ty, call.getResult());
      rewriter.replaceOp(op, result);
      return success();
    }

    return failure();
  }
};

/// Conversion for moore.stream_concat -> runtime function call.
/// This handles streaming concatenation of queues/dynamic arrays.
/// For string queues, concatenates all strings into a single string.
/// For other types, packs element bits into an integer.
struct StreamConcatOpConversion
    : public RuntimeCallConversionBase<StreamConcatOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(StreamConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto inputType = op.getInput().getType();
    auto resultType = op.getResult().getType();
    bool isRightToLeft = op.getIsRightToLeft();

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // Store input queue to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getInput(), inputAlloca);

    // Create boolean constant for direction
    auto directionConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i1Ty, isRightToLeft ? 1 : 0));

    // Check if input is a queue/array of strings
    Type elementType;
    if (auto queueType = dyn_cast<QueueType>(inputType)) {
      elementType = queueType.getElementType();
    } else if (auto dynArrayType = dyn_cast<OpenUnpackedArrayType>(inputType)) {
      elementType = dynArrayType.getElementType();
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported input type");
    }

    if (isa<StringType>(elementType)) {
      // String queue: call __moore_stream_concat_strings
      auto stringStructTy = getStringStructType(ctx);
      auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, i1Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_stream_concat_strings", fnTy);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{inputAlloca, directionConst});
      rewriter.replaceOp(op, call.getResult());
    } else {
      // Non-string type: call __moore_stream_concat_bits
      // Determine element bit width
      int64_t elementBitWidth = 32; // default
      if (auto intType = dyn_cast<IntType>(elementType)) {
        elementBitWidth = intType.getWidth();
      } else if (auto packedType = dyn_cast<PackedType>(elementType)) {
        elementBitWidth = packedType.getBitSize().value_or(32);
      }

      auto bitWidthConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(elementBitWidth));

      auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i32Ty, i1Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_stream_concat_bits", fnTy);

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i64Ty}, SymbolRefAttr::get(fn),
          ValueRange{inputAlloca, bitWidthConst, directionConst});

      // Convert result to the expected type
      auto convertedResultType = typeConverter->convertType(resultType);
      if (!convertedResultType)
        return rewriter.notifyMatchFailure(op, "failed to convert result type");

      Value result = call.getResult();
      if (convertedResultType != i64Ty) {
        if (convertedResultType.isIntOrFloat()) {
          auto resultWidth = convertedResultType.getIntOrFloatBitWidth();
          if (resultWidth < 64) {
            result = arith::TruncIOp::create(rewriter, loc, convertedResultType,
                                             result);
          } else if (resultWidth > 64) {
            result = arith::ExtUIOp::create(rewriter, loc, convertedResultType,
                                            result);
          }
        } else {
          // For non-integer types (structs, etc.), bitcast from i64
          result = LLVM::BitcastOp::create(rewriter, loc, convertedResultType,
                                           result);
        }
      }
      rewriter.replaceOp(op, result);
    }

    return success();
  }

private:
  /// Get the LLVM struct type for a string: {ptr, i64}.
  static LLVM::LLVMStructType getStringStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }
};

/// Conversion for moore.stream_unpack -> runtime function call.
/// This handles streaming unpacking from a bit vector into a dynamic array or
/// queue. The inverse of StreamConcatOp.
struct StreamUnpackOpConversion
    : public RuntimeCallConversionBase<StreamUnpackOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(StreamUnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto dstType = op.getDst().getType();
    bool isRightToLeft = op.getIsRightToLeft();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Get the destination array reference (already converted to ptr)
    Value dstPtr = adaptor.getDst();

    // Get the source value
    Value srcValue = adaptor.getSrc();

    // Extend or truncate source to i64 for the runtime call
    Type srcType = srcValue.getType();
    if (srcType != i64Ty) {
      if (srcType.isIntOrFloat()) {
        auto srcWidth = srcType.getIntOrFloatBitWidth();
        if (srcWidth < 64) {
          srcValue = arith::ExtUIOp::create(rewriter, loc, i64Ty, srcValue);
        } else if (srcWidth > 64) {
          srcValue = arith::TruncIOp::create(rewriter, loc, i64Ty, srcValue);
        }
      }
    }

    // Create boolean constant for direction
    auto directionConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i1Ty, isRightToLeft ? 1 : 0));

    // Determine element type from the ref type
    Type elementType;
    if (auto refType = dyn_cast<RefType>(dstType)) {
      auto nestedType = refType.getNestedType();
      if (auto queueType = dyn_cast<QueueType>(nestedType)) {
        elementType = queueType.getElementType();
      } else if (auto dynArrayType =
                     dyn_cast<OpenUnpackedArrayType>(nestedType)) {
        elementType = dynArrayType.getElementType();
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported destination ref type");
      }
    } else {
      return rewriter.notifyMatchFailure(op, "expected RefType for destination");
    }

    // Determine element bit width
    int64_t elementBitWidth = 1; // default for bit[]
    if (auto intType = dyn_cast<IntType>(elementType)) {
      elementBitWidth = intType.getWidth();
    } else if (auto packedType = dyn_cast<PackedType>(elementType)) {
      elementBitWidth = packedType.getBitSize().value_or(1);
    }

    auto bitWidthConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(elementBitWidth));

    // Call __moore_stream_unpack_bits(dst_ptr, src_bits, element_width,
    // direction)
    auto fnTy =
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i64Ty, i32Ty, i1Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_stream_unpack_bits",
                                     fnTy);

    LLVM::CallOp::create(
        rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
        ValueRange{dstPtr, srcValue, bitWidthConst, directionConst});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.array.locator -> runtime function call.
/// This pattern-matches the predicate region to detect common patterns
/// (comparison operations) and uses specialized runtime functions.
/// Supports:
/// - Simple predicates: `item == constant`, `item > constant`, etc.
/// - Field access predicates: `item.field == val`, `item.field > val`, etc.
/// For simple equality predicates (`item == constant`), we use
/// `__moore_array_find_eq`. For other comparison predicates, we use
/// `__moore_array_find_cmp` with the appropriate comparison mode.
/// For field access predicates, we use `__moore_array_find_field_cmp`.
struct ArrayLocatorOpConversion : public OpConversionPattern<ArrayLocatorOp> {
  ArrayLocatorOpConversion(TypeConverter &tc, MLIRContext *ctx,
                           ClassTypeCache &cache)
      : OpConversionPattern(tc, ctx), classCache(cache) {}

  using OpAdaptor = typename ArrayLocatorOp::Adaptor;

  /// Get the LLVM struct type for a queue: {ptr, i64}.
  static LLVM::LLVMStructType getQueueStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }

  /// Comparison mode enum matching MooreCmpMode in MooreRuntime.h.
  enum CmpMode {
    CMP_EQ = 0,  // Equal (==)
    CMP_NE = 1,  // Not equal (!=)
    CMP_SGT = 2, // Signed greater than (>)
    CMP_SGE = 3, // Signed greater than or equal (>=)
    CMP_SLT = 4, // Signed less than (<)
    CMP_SLE = 5  // Signed less than or equal (<=)
  };

  /// Try to extract comparison info from a binary comparison operation.
  /// Returns true if successful, setting constValue and cmpMode.
  /// The blockArg must be one of the operands.
  template <typename OpTy>
  static bool tryExtractComparisonInfo(OpTy cmpOp, Value blockArg,
                                       CmpMode mode, Value &constValue,
                                       CmpMode &cmpMode) {
    if (cmpOp.getLhs() == blockArg) {
      constValue = cmpOp.getRhs();
      cmpMode = mode;
      return true;
    } else if (cmpOp.getRhs() == blockArg) {
      constValue = cmpOp.getLhs();
      // For relational ops, swap the comparison direction when operands swap
      // e.g., if we have `const > item`, that's equivalent to `item < const`
      switch (mode) {
      case CMP_SGT:
        cmpMode = CMP_SLT;
        break;
      case CMP_SGE:
        cmpMode = CMP_SLE;
        break;
      case CMP_SLT:
        cmpMode = CMP_SGT;
        break;
      case CMP_SLE:
        cmpMode = CMP_SGE;
        break;
      default:
        cmpMode = mode; // eq and ne are symmetric
        break;
      }
      return true;
    }
    return false;
  }

  /// Information about a field access pattern.
  struct FieldAccessInfo {
    ClassPropertyRefOp propRefOp; // The property ref operation
    SymbolRefAttr classSym;       // The class symbol
    StringRef fieldName;          // The field name
    Type fieldType;               // The Moore type of the field
  };

  /// Try to detect if a value comes from reading a field of the block argument.
  /// Pattern: read(class.property_ref(blockArg))
  static std::optional<FieldAccessInfo> tryMatchFieldAccess(Value value,
                                                            Value blockArg) {
    // The value should come from a read operation
    auto readOp = value.getDefiningOp<ReadOp>();
    if (!readOp)
      return std::nullopt;

    // The read operand should be a class.property_ref
    auto propRefOp = readOp.getInput().getDefiningOp<ClassPropertyRefOp>();
    if (!propRefOp)
      return std::nullopt;

    // The property ref instance should be the block argument
    if (propRefOp.getInstance() != blockArg)
      return std::nullopt;

    // Extract class and field information
    auto classHandleType =
        dyn_cast<ClassHandleType>(propRefOp.getInstance().getType());
    if (!classHandleType)
      return std::nullopt;

    FieldAccessInfo info;
    info.propRefOp = propRefOp;
    info.classSym = classHandleType.getClassSym();
    info.fieldName = propRefOp.getProperty();
    // Get the field type from the property ref result (which is a RefType)
    if (auto refType = dyn_cast<RefType>(propRefOp.getPropertyRef().getType()))
      info.fieldType = refType.getNestedType();
    else
      return std::nullopt;

    return info;
  }

  /// Try to extract comparison info for field access patterns.
  /// Returns true if one operand is a field access and sets fieldInfo and
  /// cmpValue.
  template <typename OpTy>
  static bool tryExtractFieldComparisonInfo(OpTy cmpOp, Value blockArg,
                                            CmpMode mode,
                                            std::optional<FieldAccessInfo> &fieldInfo,
                                            Value &cmpValue, CmpMode &cmpMode) {
    // Try lhs as field access
    if (auto info = tryMatchFieldAccess(cmpOp.getLhs(), blockArg)) {
      fieldInfo = info;
      cmpValue = cmpOp.getRhs();
      cmpMode = mode;
      return true;
    }
    // Try rhs as field access (swap comparison direction for relational ops)
    if (auto info = tryMatchFieldAccess(cmpOp.getRhs(), blockArg)) {
      fieldInfo = info;
      cmpValue = cmpOp.getLhs();
      // Swap comparison direction for relational ops
      switch (mode) {
      case CMP_SGT:
        cmpMode = CMP_SLT;
        break;
      case CMP_SGE:
        cmpMode = CMP_SLE;
        break;
      case CMP_SLT:
        cmpMode = CMP_SGT;
        break;
      case CMP_SLE:
        cmpMode = CMP_SGE;
        break;
      default:
        cmpMode = mode;
        break;
      }
      return true;
    }
    return false;
  }

  /// Compute field offset in bytes from GEP indices using the LLVM struct type.
  /// This is a simplified calculation that works for common cases.
  static int64_t computeFieldOffset(LLVM::LLVMStructType structTy,
                                    ArrayRef<unsigned> gepPath) {
    int64_t offset = 0;
    Type currentType = structTy;

    for (unsigned idx : gepPath) {
      if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        // Sum sizes of all fields before this index
        for (unsigned i = 0; i < idx; ++i) {
          Type fieldType = structType.getBody()[i];
          offset += getTypeSize(fieldType);
        }
        currentType = structType.getBody()[idx];
      } else {
        // Unsupported type in path
        return -1;
      }
    }
    return offset;
  }

  /// Get the size in bytes of an LLVM type (simplified).
  static int64_t getTypeSize(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type))
      return (intType.getWidth() + 7) / 8;
    if (isa<LLVM::LLVMPointerType>(type))
      return 8; // Assume 64-bit pointers
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
      int64_t size = 0;
      for (Type fieldType : structType.getBody())
        size += getTypeSize(fieldType);
      return size;
    }
    // Default to 8 bytes for unknown types
    return 8;
  }

  /// Lower array.locator with an inline loop for complex predicates.
  /// This works for arbitrary predicate expressions because the predicate
  /// region is cloned into the loop body and converted by existing patterns.
  LogicalResult
  lowerWithInlineLoop(ArrayLocatorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      MLIRContext *ctx, ModuleOp mod) const {
    Block &body = op.getBody().front();
    Value blockArg = body.getArgument(0);

    auto yieldOp = dyn_cast<ArrayLocatorYieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(op, "expected array.locator.yield");

    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);

    // Get element type from the input array
    Type mooreElemType;
    if (auto queueType = dyn_cast<QueueType>(op.getArray().getType())) {
      mooreElemType = queueType.getElementType();
    } else if (auto arrayType =
                   dyn_cast<UnpackedArrayType>(op.getArray().getType())) {
      mooreElemType = arrayType.getElementType();
    } else if (auto dynArrayType =
                   dyn_cast<OpenUnpackedArrayType>(op.getArray().getType())) {
      mooreElemType = dynArrayType.getElementType();
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported array type");
    }

    Type elemType = typeConverter->convertType(mooreElemType);
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "failed to convert element type");

    int64_t elemSizeBytes = getTypeSize(elemType);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto resultAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);

    // Initialize queue to {nullptr, 0}
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    Value zeroLen = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
    Value emptyQueue = LLVM::UndefOp::create(rewriter, loc, queueTy);
    emptyQueue =
        LLVM::InsertValueOp::create(rewriter, loc, emptyQueue, nullPtr,
                                    ArrayRef<int64_t>{0});
    emptyQueue =
        LLVM::InsertValueOp::create(rewriter, loc, emptyQueue, zeroLen,
                                    ArrayRef<int64_t>{1});
    LLVM::StoreOp::create(rewriter, loc, emptyQueue, resultAlloca);

    // Extract array length and data pointer
    Value arrayLen = LLVM::ExtractValueOp::create(
        rewriter, loc, i64Ty, adaptor.getArray(), ArrayRef<int64_t>{1});
    Value dataPtr = LLVM::ExtractValueOp::create(
        rewriter, loc, ptrTy, adaptor.getArray(), ArrayRef<int64_t>{0});

    Value lb = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(0));
    Value step = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    auto forOp = scf::ForOp::create(rewriter, loc, lb, arrayLen, step);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();

    Value elemPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType, dataPtr, iv);
    Value currentElem = LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);

    // Map the predicate block argument to the current element (with a cast
    // back to Moore type for the cloned predicate ops).
    IRMapping mapper;
    Value currentElemMoore =
        UnrealizedConversionCastOp::create(rewriter, loc, blockArg.getType(),
                                           currentElem)
            .getResult(0);
    mapper.map(blockArg, currentElemMoore);

    for (Operation &innerOp : body.without_terminator()) {
      Operation *clonedOp = rewriter.clone(innerOp, mapper);
      for (auto [oldResult, newResult] :
           llvm::zip(innerOp.getResults(), clonedOp->getResults()))
        mapper.map(oldResult, newResult);
    }

    Value condValue = mapper.lookupOrDefault(yieldOp.getCondition());
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, loc, i1Ty, condValue);
    if (!cond)
      return rewriter.notifyMatchFailure(op, "failed to convert predicate");

    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, cond,
                                  /*withElseRegion=*/false);

    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto pushFnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto pushFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_push_back",
                               pushFnTy);

    bool returnIndices = op.getIndexed();
    if (returnIndices) {
      auto indexAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, one);
      LLVM::StoreOp::create(rewriter, loc, iv, indexAlloca);
      auto indexSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(8));
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(pushFn),
                           ValueRange{resultAlloca, indexAlloca, indexSize});
    } else {
      auto elemAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemType, one);
      LLVM::StoreOp::create(rewriter, loc, currentElem, elemAlloca);
      auto elemSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(elemSizeBytes));
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(pushFn),
                           ValueRange{resultAlloca, elemAlloca, elemSize});
    }

    rewriter.setInsertionPointAfter(forOp);

    Value result = LLVM::LoadOp::create(rewriter, loc, queueTy, resultAlloca);

    auto mode = op.getMode();
    if (mode != LocatorMode::All) {
      Value resultLen = LLVM::ExtractValueOp::create(
          rewriter, loc, i64Ty, result, ArrayRef<int64_t>{1});
      Value resultDataPtr = LLVM::ExtractValueOp::create(
          rewriter, loc, ptrTy, result, ArrayRef<int64_t>{0});

      Value isNotEmpty = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, resultLen, zeroLen);
      auto modeIfOp =
          scf::IfOp::create(rewriter, loc, TypeRange{queueTy}, isNotEmpty,
                            /*withElseRegion=*/true);

      rewriter.setInsertionPointToStart(&modeIfOp.getThenRegion().front());
      Value extractIdx;
      if (mode == LocatorMode::First) {
        extractIdx = zeroLen;
      } else {
        extractIdx = arith::SubIOp::create(rewriter, loc, resultLen, step);
      }

      Type resultElemType = returnIndices ? i64Ty : elemType;
      int64_t resultElemSize = returnIndices ? 8 : elemSizeBytes;

      Value singleElemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrTy, resultElemType,
                              resultDataPtr, extractIdx);
      Value singleElem = LLVM::LoadOp::create(rewriter, loc, resultElemType,
                                              singleElemPtr);

      auto singleResultAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
      LLVM::StoreOp::create(rewriter, loc, emptyQueue, singleResultAlloca);

      auto singleElemAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, resultElemType, one);
      LLVM::StoreOp::create(rewriter, loc, singleElem, singleElemAlloca);
      auto singleElemSizeVal = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(resultElemSize));
      LLVM::CallOp::create(
          rewriter, loc, TypeRange{}, SymbolRefAttr::get(pushFn),
          ValueRange{singleResultAlloca, singleElemAlloca, singleElemSizeVal});

      Value singleResult =
          LLVM::LoadOp::create(rewriter, loc, queueTy, singleResultAlloca);
      scf::YieldOp::create(rewriter, loc, ValueRange{singleResult});

      rewriter.setInsertionPointToStart(&modeIfOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, ValueRange{emptyQueue});

      rewriter.setInsertionPointAfter(modeIfOp);
      result = modeIfOp.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  matchAndRewrite(ArrayLocatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // Check if predicate is a simple comparison: `item <op> constant`
    // or a field comparison: `item.field <op> value`
    // The region has format:
    //   ^bb0(%item):
    //     %const = moore.constant ...
    //     %cond = moore.<cmpop> %item, %const (or %const, %item)
    //     moore.array.locator.yield %cond
    // Or for field access:
    //   ^bb0(%item):
    //     %ref = moore.class.property_ref %item[@field]
    //     %fieldVal = moore.read %ref
    //     %cmpVal = moore.read %someVar  (or moore.constant)
    //     %cond = moore.<cmpop> %fieldVal, %cmpVal
    //     moore.array.locator.yield %cond
    Block &body = op.getBody().front();
    if (body.getNumArguments() != 1)
      return rewriter.notifyMatchFailure(op, "expected single block argument");

    Value blockArg = body.getArgument(0);

    // Find the yield op
    auto yieldOp = dyn_cast<ArrayLocatorYieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(op, "expected array.locator.yield");

    // Try to match various comparison operations
    Value constValue = nullptr;
    CmpMode cmpMode = CMP_EQ;
    bool foundComparison = false;
    bool isFieldAccess = false;
    std::optional<FieldAccessInfo> fieldInfo;
    Value fieldCmpValue = nullptr;

    Value condValue = yieldOp.getCondition();

    // First try to match field access patterns (item.field <op> value)
    // Try EqOp with field access
    if (auto cmpOp = condValue.getDefiningOp<EqOp>()) {
      if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_EQ, fieldInfo,
                                        fieldCmpValue, cmpMode)) {
        isFieldAccess = true;
        foundComparison = true;
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<NeOp>()) {
        if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_NE, fieldInfo,
                                          fieldCmpValue, cmpMode)) {
          isFieldAccess = true;
          foundComparison = true;
        }
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SgtOp>()) {
        if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_SGT, fieldInfo,
                                          fieldCmpValue, cmpMode)) {
          isFieldAccess = true;
          foundComparison = true;
        }
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SgeOp>()) {
        if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_SGE, fieldInfo,
                                          fieldCmpValue, cmpMode)) {
          isFieldAccess = true;
          foundComparison = true;
        }
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SltOp>()) {
        if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_SLT, fieldInfo,
                                          fieldCmpValue, cmpMode)) {
          isFieldAccess = true;
          foundComparison = true;
        }
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SleOp>()) {
        if (tryExtractFieldComparisonInfo(cmpOp, blockArg, CMP_SLE, fieldInfo,
                                          fieldCmpValue, cmpMode)) {
          isFieldAccess = true;
          foundComparison = true;
        }
      }
    }

    // If not a field access pattern, try simple item comparison
    if (!foundComparison) {
      // Try EqOp
      if (auto cmpOp = condValue.getDefiningOp<EqOp>()) {
        foundComparison =
            tryExtractComparisonInfo(cmpOp, blockArg, CMP_EQ, constValue, cmpMode);
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<NeOp>()) {
        foundComparison = tryExtractComparisonInfo(cmpOp, blockArg, CMP_NE,
                                                   constValue, cmpMode);
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SgtOp>()) {
        foundComparison = tryExtractComparisonInfo(cmpOp, blockArg, CMP_SGT,
                                                   constValue, cmpMode);
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SgeOp>()) {
        foundComparison = tryExtractComparisonInfo(cmpOp, blockArg, CMP_SGE,
                                                   constValue, cmpMode);
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SltOp>()) {
        foundComparison = tryExtractComparisonInfo(cmpOp, blockArg, CMP_SLT,
                                                   constValue, cmpMode);
      }
    }
    if (!foundComparison) {
      if (auto cmpOp = condValue.getDefiningOp<SleOp>()) {
        foundComparison = tryExtractComparisonInfo(cmpOp, blockArg, CMP_SLE,
                                                   constValue, cmpMode);
      }
    }

    // If simple pattern matching failed, use inline loop approach for complex
    // predicates (string comparisons, class handle comparisons, AND/OR, etc.)
    if (!foundComparison) {
      return lowerWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod);
    }

    // Handle field access predicates
    if (isFieldAccess) {
      if (!fieldInfo)
        return rewriter.notifyMatchFailure(op, "failed to extract field info");

      // Resolve class struct info
      if (failed(resolveClassStructBody(mod, fieldInfo->classSym,
                                        *getTypeConverter(), classCache)))
        return rewriter.notifyMatchFailure(
            op, "failed to resolve class struct for field access");

      auto structInfo = classCache.getStructInfo(fieldInfo->classSym);
      if (!structInfo)
        return rewriter.notifyMatchFailure(op, "class struct info not found");

      auto gepPath = structInfo->getFieldPath(fieldInfo->fieldName);
      if (!gepPath)
        return rewriter.notifyMatchFailure(op, "field path not found");

      // Compute field offset
      int64_t fieldOffset =
          computeFieldOffset(structInfo->classBody, *gepPath);
      if (fieldOffset < 0)
        return rewriter.notifyMatchFailure(op,
                                           "failed to compute field offset");

      // Get field size
      int64_t fieldSizeBytes = 4; // Default to 4 bytes
      if (auto intType = dyn_cast<IntType>(fieldInfo->fieldType))
        fieldSizeBytes = (intType.getWidth() + 7) / 8;

      // Set up types
      auto queueTy = getQueueStructType(ctx);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto i32Ty = IntegerType::get(ctx, 32);
      auto i1Ty = IntegerType::get(ctx, 1);

      // Store input array to alloca and pass pointer
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto inputAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

      // Convert and store comparison value
      auto convertedFieldType =
          getTypeConverter()->convertType(fieldInfo->fieldType);
      if (!convertedFieldType)
        return rewriter.notifyMatchFailure(op,
                                           "failed to convert field type");

      auto cmpValueAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, convertedFieldType, one);

      // Get the comparison value - it may come from a read op or constant
      Value llvmCmpValue;
      if (auto readOp = fieldCmpValue.getDefiningOp<ReadOp>()) {
        // The value comes from reading a variable - we need to handle this
        // For now, we'll inline the read and get the converted value
        // The conversion infrastructure should have already converted reads
        // Look for already-converted value in the adaptor
        // Since the read is inside the region, we need to handle it specially
        return rewriter.notifyMatchFailure(
            op, "field comparison with variable read not yet supported - "
                "use constants");
      } else if (auto constOp = fieldCmpValue.getDefiningOp<ConstantOp>()) {
        APInt constAPInt = constOp.getValue().toAPInt(false);
        llvmCmpValue = LLVM::ConstantOp::create(rewriter, loc, convertedFieldType,
                                                constAPInt.getSExtValue());
      } else {
        return rewriter.notifyMatchFailure(
            op, "field comparison value must be a constant");
      }

      LLVM::StoreOp::create(rewriter, loc, llvmCmpValue, cmpValueAlloca);

      // For class handles, element size is pointer size (8 bytes)
      auto elementSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(8));

      // Field offset and size constants
      auto fieldOffsetConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(fieldOffset));
      auto fieldSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(fieldSizeBytes));

      // Locator mode and indexed flag
      int32_t locatorModeValue = static_cast<int32_t>(op.getMode());
      auto locatorModeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(locatorModeValue));
      bool returnIndices = op.getIndexed();
      auto indicesConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i1Ty, returnIndices ? 1 : 0));

      // Comparison mode
      auto cmpModeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(cmpMode)));

      // Call __moore_array_find_field_cmp
      // Signature: MooreQueue(MooreQueue*, i64, i64, i64, void*, i32, i32, i1)
      auto fnTy = LLVM::LLVMFunctionType::get(
          queueTy, {ptrTy, i64Ty, i64Ty, i64Ty, ptrTy, i32Ty, i32Ty, i1Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_array_find_field_cmp", fnTy);

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
          ValueRange{inputAlloca, elementSizeConst, fieldOffsetConst,
                     fieldSizeConst, cmpValueAlloca, cmpModeConst,
                     locatorModeConst, indicesConst});
      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // Handle simple item comparison (existing code path)
    if (!constValue)
      return rewriter.notifyMatchFailure(
          op, "comparison must compare block argument with a value");

    // The constant value must be defined by a moore.constant op
    auto constOp = constValue.getDefiningOp<ConstantOp>();
    if (!constOp)
      return rewriter.notifyMatchFailure(
          op, "comparison value must be a constant");

    // Get element type and size
    Type elementType;
    if (auto queueType = dyn_cast<QueueType>(op.getArray().getType())) {
      elementType = queueType.getElementType();
    } else if (auto arrayType =
                   dyn_cast<UnpackedArrayType>(op.getArray().getType())) {
      elementType = arrayType.getElementType();
    } else if (auto dynArrayType =
                   dyn_cast<OpenUnpackedArrayType>(op.getArray().getType())) {
      elementType = dynArrayType.getElementType();
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported array type");
    }

    // Get element size in bytes
    int64_t elementSizeBytes = 4; // default to 4 bytes (32 bits)
    if (auto intType = dyn_cast<IntType>(elementType)) {
      elementSizeBytes = (intType.getWidth() + 7) / 8;
    }

    // Set up types
    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    // Store input array to alloca and pass pointer
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    // Store constant value to alloca and pass pointer
    auto convertedConstType = typeConverter->convertType(constValue.getType());
    if (!convertedConstType)
      return rewriter.notifyMatchFailure(op, "failed to convert constant type");

    auto constAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, convertedConstType, one);

    // Get the converted constant value
    // The constant op is still in Moore dialect; we need to convert it
    // Use toAPInt(false) to convert FVInt to APInt, mapping X/Z bits to 0
    APInt constAPInt = constOp.getValue().toAPInt(false);
    auto llvmConstValue = LLVM::ConstantOp::create(
        rewriter, loc, convertedConstType, constAPInt.getSExtValue());
    LLVM::StoreOp::create(rewriter, loc, llvmConstValue, constAlloca);

    // Create element size constant
    auto elementSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Create locator mode constant: All=0, First=1, Last=2
    int32_t locatorModeValue = static_cast<int32_t>(op.getMode());
    auto locatorModeConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(locatorModeValue));

    // Create indexed flag constant
    bool returnIndices = op.getIndexed();
    auto indicesConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i1Ty, returnIndices ? 1 : 0));

    // Choose runtime function based on comparison mode
    // For simple equality, use the optimized __moore_array_find_eq
    // For other comparisons, use __moore_array_find_cmp
    if (cmpMode == CMP_EQ) {
      // Signature: MooreQueue __moore_array_find_eq(MooreQueue*, i64, void*,
      // i32, i1)
      auto fnTy = LLVM::LLVMFunctionType::get(
          queueTy, {ptrTy, i64Ty, ptrTy, i32Ty, i1Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_find_eq", fnTy);

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
          ValueRange{inputAlloca, elementSizeConst, constAlloca,
                     locatorModeConst, indicesConst});
      rewriter.replaceOp(op, call.getResult());
    } else {
      // Signature: MooreQueue __moore_array_find_cmp(MooreQueue*, i64, void*,
      // i32, i32, i1)
      auto fnTy = LLVM::LLVMFunctionType::get(
          queueTy, {ptrTy, i64Ty, ptrTy, i32Ty, i32Ty, i1Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_find_cmp", fnTy);

      // Create comparison mode constant
      auto cmpModeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(cmpMode)));

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
          ValueRange{inputAlloca, elementSizeConst, constAlloca, cmpModeConst,
                     locatorModeConst, indicesConst});
      rewriter.replaceOp(op, call.getResult());
    }

    return success();
  }

private:
  ClassTypeCache &classCache;
};

/// Conversion for moore.dyn_array.new -> runtime function call.
struct DynArrayNewOpConversion
    : public RuntimeCallConversionBase<DynArrayNewOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(DynArrayNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto dynArrayTy = getDynArrayStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    // Two variants: with and without init array.
    if (adaptor.getInit()) {
      // __moore_dyn_array_new_copy(size, init_ptr) -> {ptr, i64}
      auto fnTy = LLVM::LLVMFunctionType::get(dynArrayTy, {i32Ty, ptrTy});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_dyn_array_new_copy", fnTy);

      // Store init array to alloca and pass pointer.
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto initAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, dynArrayTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getInit(), initAlloca);

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{dynArrayTy}, SymbolRefAttr::get(fn),
          ValueRange{adaptor.getSize(), initAlloca});
      rewriter.replaceOp(op, call.getResult());
    } else {
      // __moore_dyn_array_new(size) -> {ptr, i64}
      auto fnTy = LLVM::LLVMFunctionType::get(dynArrayTy, {i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_dyn_array_new", fnTy);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{dynArrayTy},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{adaptor.getSize()});
      rewriter.replaceOp(op, call.getResult());
    }
    return success();
  }
};

/// Conversion for moore.assoc.delete -> runtime function call.
struct AssocArrayDeleteOpConversion
    : public OpConversionPattern<AssocArrayDeleteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssocArrayDeleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_delete", fnTy);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getArray()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.assoc.delete_key -> runtime function call.
struct AssocArrayDeleteKeyOpConversion
    : public OpConversionPattern<AssocArrayDeleteKeyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssocArrayDeleteKeyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    // The key is passed by pointer for genericity.
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_delete_key",
                                     fnTy);

    // Store key to alloca and pass pointer.
    auto keyType = adaptor.getKey().getType();
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto keyAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getKey(), keyAlloca);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getArray(), keyAlloca});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Base class for associative array iterator operations (first, next, last,
/// prev).
template <typename SourceOp>
struct AssocArrayIteratorOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  AssocArrayIteratorOpConversion(TypeConverter &typeConverter,
                                 MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        funcName(funcName) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->template getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);
    // All iterator functions: (array_ptr, key_ref_ptr) -> i1
    auto fnTy = LLVM::LLVMFunctionType::get(i1Ty, {ptrTy, ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, funcName, fnTy);

    // Get the key pointer - either directly if LLVM pointer, or create an
    // alloca for llhd.ref types
    Value keyArg = adaptor.getKey();
    Value keyAlloca;
    Type keyValueType;  // For writeback of llhd.ref keys
    if (isa<LLVM::LLVMPointerType>(keyArg.getType())) {
      // String keys are already LLVM pointers (to struct {ptr, i64})
      keyAlloca = keyArg;
    } else if (auto refType = dyn_cast<llhd::RefType>(keyArg.getType())) {
      // Integer keys are llhd.ref - we need to create an alloca and copy the
      // value to it (the runtime will update it in place for first/next/etc)
      keyValueType = refType.getNestedType();
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyValueType, one);

      // Read current key value and store to alloca
      auto currentKey = llhd::ProbeOp::create(rewriter, loc, keyArg);
      LLVM::StoreOp::create(rewriter, loc, currentKey, keyAlloca);
    } else {
      return rewriter.notifyMatchFailure(loc, "unsupported key type");
    }

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{i1Ty}, SymbolRefAttr::get(fn),
        ValueRange{adaptor.getArray(), keyAlloca});

    // For non-string keys (llhd.ref), write back the updated key value
    if (keyValueType) {
      auto updatedKey =
          LLVM::LoadOp::create(rewriter, loc, keyValueType, keyAlloca);
      auto delay = llhd::ConstantTimeOp::create(
          rewriter, loc, llhd::TimeAttr::get(ctx, 0U, "ns", 0, 1));
      llhd::DriveOp::create(rewriter, loc, keyArg, updatedKey, delay, Value{});
    }

    rewriter.replaceOp(op, call.getResult());
    return success();
  }

private:
  StringRef funcName;
};

/// Conversion for moore.assoc.first -> runtime function call.
struct AssocArrayFirstOpConversion
    : public AssocArrayIteratorOpConversion<AssocArrayFirstOp> {
  AssocArrayFirstOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : AssocArrayIteratorOpConversion(typeConverter, context,
                                       "__moore_assoc_first") {}
};

/// Conversion for moore.assoc.next -> runtime function call.
struct AssocArrayNextOpConversion
    : public AssocArrayIteratorOpConversion<AssocArrayNextOp> {
  AssocArrayNextOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : AssocArrayIteratorOpConversion(typeConverter, context,
                                       "__moore_assoc_next") {}
};

/// Conversion for moore.assoc.last -> runtime function call.
struct AssocArrayLastOpConversion
    : public AssocArrayIteratorOpConversion<AssocArrayLastOp> {
  AssocArrayLastOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : AssocArrayIteratorOpConversion(typeConverter, context,
                                       "__moore_assoc_last") {}
};

/// Conversion for moore.assoc.prev -> runtime function call.
struct AssocArrayPrevOpConversion
    : public AssocArrayIteratorOpConversion<AssocArrayPrevOp> {
  AssocArrayPrevOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : AssocArrayIteratorOpConversion(typeConverter, context,
                                       "__moore_assoc_prev") {}
};

/// Conversion for moore.assoc.exists -> runtime function call.
struct AssocArrayExistsOpConversion
    : public OpConversionPattern<AssocArrayExistsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssocArrayExistsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    // The key is passed by pointer for genericity.
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_exists", fnTy);

    // Store key to alloca and pass pointer.
    auto keyType = adaptor.getKey().getType();
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto keyAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getKey(), keyAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{adaptor.getArray(), keyAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// String Operations Conversion
//===----------------------------------------------------------------------===//

/// Get the LLVM struct type used to represent strings: {ptr, i64}
static LLVM::LLVMStructType getStringStructType(MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);
  return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
}

// moore.string.len -> call to __moore_string_len runtime function
struct StringLenOpConversion : public OpConversionPattern<StringLenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringLenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_len", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.toupper -> call to __moore_string_toupper runtime function
struct StringToUpperOpConversion : public OpConversionPattern<StringToUpperOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringToUpperOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_toupper", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.tolower -> call to __moore_string_tolower runtime function
struct StringToLowerOpConversion : public OpConversionPattern<StringToLowerOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringToLowerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_tolower", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.getc -> call to __moore_string_getc runtime function
struct StringGetCOpConversion : public OpConversionPattern<StringGetCOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringGetCOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i8Ty, {ptrTy, i32Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_getc", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{i8Ty}, SymbolRefAttr::get(runtimeFn),
        ValueRange{strAlloca, adaptor.getIndex()});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.putc -> call to __moore_string_putc runtime function
// The runtime function takes a string by-pointer (for reading), index, character,
// and returns a new string with the modified character.
struct StringPutCOpConversion : public OpConversionPattern<StringPutCOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringPutCOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // __moore_string_putc(str_ptr, index, char) -> new_string
    // The function returns a new string with the character at index replaced.
    auto fnTy =
        LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, i32Ty, i8Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_putc", fnTy);

    // The str operand is a pointer to the string struct (LLVM pointer type)
    // since moore.ref<string> is converted to LLVM pointer for dynamic types.
    Value strPtr = adaptor.getStr();

    // Call the runtime function directly with the pointer.
    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{stringStructTy}, SymbolRefAttr::get(runtimeFn),
        ValueRange{strPtr, adaptor.getIndex(), adaptor.getCharacter()});

    // Store the new string back to the pointer.
    LLVM::StoreOp::create(rewriter, loc, call.getResult(), strPtr);

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.string.substr -> call to __moore_string_substr runtime function
struct StringSubstrOpConversion : public OpConversionPattern<StringSubstrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringSubstrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy =
        LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, i32Ty, i32Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_substr", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{stringStructTy}, SymbolRefAttr::get(runtimeFn),
        ValueRange{strAlloca, adaptor.getStart(), adaptor.getLen()});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.itoa -> call to __moore_string_itoa runtime function
struct StringItoaOpConversion : public OpConversionPattern<StringItoaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringItoaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto value = adaptor.getValue();
    auto valueWidth = value.getType().getIntOrFloatBitWidth();

    Value valueI64;
    if (valueWidth < 64) {
      valueI64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
    } else if (valueWidth > 64) {
      valueI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, value);
    } else {
      valueI64 = value;
    }

    // __moore_string_itoa returns a new string (ptr, len) struct
    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_itoa", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{valueI64});
    Value resultString = call.getResult();

    // Store the result to the destination pointer.
    // moore.ref<string> is converted to LLVM pointer for dynamic types.
    LLVM::StoreOp::create(rewriter, loc, resultString, adaptor.getDest());

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.string.atoi -> call to __moore_string_atoi runtime function
struct StringAtoIOpConversion : public OpConversionPattern<StringAtoIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringAtoIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_atoi", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.atohex -> call to __moore_string_atohex runtime function
struct StringAtoHexOpConversion : public OpConversionPattern<StringAtoHexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringAtoHexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_atohex", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.atooct -> call to __moore_string_atooct runtime function
struct StringAtoOctOpConversion : public OpConversionPattern<StringAtoOctOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringAtoOctOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_atooct", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.atobin -> call to __moore_string_atobin runtime function
struct StringAtoBinOpConversion : public OpConversionPattern<StringAtoBinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringAtoBinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_atobin", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string_concat -> call to __moore_string_concat runtime function
struct StringConcatOpConversion : public OpConversionPattern<StringConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto inputs = adaptor.getInputs();

    if (inputs.empty()) {
      Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      Value zero = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
      Value result = LLVM::UndefOp::create(rewriter, loc, stringStructTy);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, nullPtr,
                                           ArrayRef<int64_t>{0});
      result = LLVM::InsertValueOp::create(rewriter, loc, result, zero,
                                           ArrayRef<int64_t>{1});
      rewriter.replaceOp(op, result);
      return success();
    }

    if (inputs.size() == 1) {
      rewriter.replaceOp(op, inputs[0]);
      return success();
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_concat", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));

    Value result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto lhsAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, result, lhsAlloca);

      auto rhsAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, inputs[i], rhsAlloca);

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{stringStructTy},
          SymbolRefAttr::get(runtimeFn), ValueRange{lhsAlloca, rhsAlloca});
      result = call.getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// moore.string_replicate -> call to __moore_string_replicate runtime function
struct StringReplicateOpConversion
    : public OpConversionPattern<StringReplicateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    // __moore_string_replicate(str_ptr, count) -> string
    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy, i32Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_replicate", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getString(), strAlloca);

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{stringStructTy}, SymbolRefAttr::get(runtimeFn),
        ValueRange{strAlloca, adaptor.getCount()});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string_cmp -> call to __moore_string_cmp runtime function
struct StringCmpOpConversion : public OpConversionPattern<StringCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_cmp", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto lhsAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getLhs(), lhsAlloca);

    auto rhsAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getRhs(), rhsAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{lhsAlloca, rhsAlloca});
    Value cmpResult = call.getResult();

    Value zero = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

    Value result;
    switch (op.getPredicate()) {
    case StringCmpPredicate::eq:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     cmpResult, zero);
      break;
    case StringCmpPredicate::ne:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                     cmpResult, zero);
      break;
    case StringCmpPredicate::lt:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                     cmpResult, zero);
      break;
    case StringCmpPredicate::le:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle,
                                     cmpResult, zero);
      break;
    case StringCmpPredicate::gt:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                     cmpResult, zero);
      break;
    case StringCmpPredicate::ge:
      result = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                     cmpResult, zero);
      break;
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// moore.int_to_string -> call to __moore_int_to_string runtime function
struct IntToStringOpConversion : public OpConversionPattern<IntToStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntToStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto input = adaptor.getInput();
    auto inputWidth = input.getType().getIntOrFloatBitWidth();

    Value inputI64;
    if (inputWidth < 64) {
      inputI64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, input);
    } else if (inputWidth > 64) {
      inputI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, input);
    } else {
      inputI64 = input;
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_int_to_string", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{inputI64});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string_to_int -> call to __moore_string_to_int runtime function
struct StringToIntOpConversion : public OpConversionPattern<StringToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto resultType = typeConverter->convertType(op.getResult().getType());
    auto resultWidth = resultType.getIntOrFloatBitWidth();

    auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_to_int", fnTy);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getInput(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    Value result = call.getResult();

    if (resultWidth < 64) {
      result = arith::TruncIOp::create(rewriter, loc, resultType, result);
    } else if (resultWidth > 64) {
      result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SScanfBIOp Conversion
//===----------------------------------------------------------------------===//

/// Conversion for moore.builtin.sscanf -> runtime function call.
/// Lowers $sscanf(input, format, args...) to __moore_sscanf_* calls based on
/// the format specifier.
struct SScanfBIOpConversion : public OpConversionPattern<SScanfBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SScanfBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto stringStructTy =
        LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

    // Get format string to determine which runtime function to call
    StringRef format = op.getFormat();

    // Store input string to stack
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getInput(), strAlloca);

    // Handle common single-specifier formats
    // For now, support %d, %h, %x, %o, %b for integers
    StringRef runtimeFnName;
    if (format == "%d")
      runtimeFnName = "__moore_string_atoi";
    else if (format == "%h" || format == "%x")
      runtimeFnName = "__moore_string_atohex";
    else if (format == "%o")
      runtimeFnName = "__moore_string_atooct";
    else if (format == "%b")
      runtimeFnName = "__moore_string_atobin";
    else {
      // For unsupported formats, return 0 (no items parsed)
      // TODO: Implement full sscanf with multiple format specifiers
      auto zero = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(0));
      rewriter.replaceOp(op, zero);
      return success();
    }

    // Get the runtime function
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
    auto runtimeFn = getOrCreateRuntimeFunc(mod, rewriter, runtimeFnName, fnTy);

    // Call the runtime function
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    Value parsedValue = call.getResult();

    // Store the result in the first argument if present
    if (!adaptor.getArgs().empty()) {
      auto destRef = adaptor.getArgs()[0];
      // The destination is an llhd.ref (converted from moore.ref)
      // Get the target type to potentially extend/truncate the result
      if (auto sigTy = dyn_cast<llhd::RefType>(destRef.getType())) {
        auto targetTy = sigTy.getNestedType();
        unsigned targetWidth = targetTy.getIntOrFloatBitWidth();

        // Extend or truncate to match the target type
        Value valueToStore;
        if (targetWidth < 32) {
          valueToStore =
              arith::TruncIOp::create(rewriter, loc, targetTy, parsedValue);
        } else if (targetWidth > 32) {
          // Sign extend for larger types
          valueToStore =
              arith::ExtSIOp::create(rewriter, loc, targetTy, parsedValue);
        } else {
          valueToStore = parsedValue;
        }

        // Create llhd.drive to store the value
        auto timeAttr = llhd::TimeAttr::get(ctx, 0U, llvm::StringRef("ns"), 0, 1);
        auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
        llhd::DriveOp::create(rewriter, loc, destRef, valueToStore, time,
                              Value{});
      }
    }

    // Return 1 (one item successfully parsed)
    auto resultVal = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
    rewriter.replaceOp(op, resultVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IsUnknownBIOp Conversion
//===----------------------------------------------------------------------===//

// moore.builtin.isunknown -> constant false (since we only support two-valued)
struct IsUnknownBIOpConversion : public OpConversionPattern<IsUnknownBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsUnknownBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In the current two-valued lowering, there are no X or Z bits,
    // so $isunknown always returns 0.
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, 0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CountOnesBIOp Conversion
//===----------------------------------------------------------------------===//

/// $countones(x) -> llvm.ctpop(x)
/// Returns the number of 1 bits in the input.
struct CountOnesBIOpConversion : public OpConversionPattern<CountOnesBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CountOnesBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto inputType = cast<IntegerType>(input.getType());

    // Use LLVM's ctpop (count population) intrinsic to count 1 bits.
    Value ctpop = LLVM::CtPopOp::create(rewriter, loc, inputType, input);
    rewriter.replaceOp(op, ctpop);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OneHotBIOp Conversion
//===----------------------------------------------------------------------===//

/// $onehot(x) -> ctpop(x) == 1
/// Returns 1 if exactly one bit is set in the input.
struct OneHotBIOpConversion : public OpConversionPattern<OneHotBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OneHotBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto inputType = cast<IntegerType>(input.getType());
    unsigned bitWidth = inputType.getWidth();

    // Use LLVM's ctpop (count population) intrinsic to count 1 bits.
    Value ctpop = LLVM::CtPopOp::create(rewriter, loc, inputType, input);

    // Compare count to 1: exactly one bit set
    Value one = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, 1));
    Value isOneHot =
        comb::ICmpOp::create(rewriter, loc, ICmpPredicate::eq, ctpop, one);

    rewriter.replaceOp(op, isOneHot);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OneHot0BIOp Conversion
//===----------------------------------------------------------------------===//

/// $onehot0(x) -> ctpop(x) <= 1
/// Returns 1 if at most one bit is set in the input (zero or one).
struct OneHot0BIOpConversion : public OpConversionPattern<OneHot0BIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OneHot0BIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto inputType = cast<IntegerType>(input.getType());
    unsigned bitWidth = inputType.getWidth();

    // Use LLVM's ctpop (count population) intrinsic to count 1 bits.
    Value ctpop = LLVM::CtPopOp::create(rewriter, loc, inputType, input);

    // Compare count to 1: at most one bit set (count <= 1)
    Value one = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, 1));
    Value isOneHot0 =
        comb::ICmpOp::create(rewriter, loc, ICmpPredicate::ule, ctpop, one);

    rewriter.replaceOp(op, isOneHot0);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CountBitsBIOp Conversion
//===----------------------------------------------------------------------===//

/// $countbits(x, control_bits) lowering for two-valued types.
/// control_bits mask: 0b0001=count zeros, 0b0010=count ones,
///                    0b0100=count X (always 0), 0b1000=count Z (always 0)
///
/// For two-valued lowering:
/// - $countbits(x, 1) = ctpop(x)
/// - $countbits(x, 0) = bitwidth - ctpop(x)
/// - $countbits(x, 0, 1) = bitwidth (all bits are either 0 or 1)
struct CountBitsBIOpConversion : public OpConversionPattern<CountBitsBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CountBitsBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getValue();
    auto inputType = cast<IntegerType>(input.getType());
    unsigned bitWidth = inputType.getWidth();

    int32_t controlBits = op.getControlBits();
    bool countZeros = (controlBits & 1) != 0;  // 0b0001
    bool countOnes = (controlBits & 2) != 0;   // 0b0010
    // X and Z are always 0 in two-valued lowering (bits 0b0100 and 0b1000)

    Value result;
    if (countOnes && countZeros) {
      // Counting both 0s and 1s = all bits = bitwidth
      result = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, bitWidth));
    } else if (countOnes) {
      // Count 1 bits using ctpop
      result = LLVM::CtPopOp::create(rewriter, loc, inputType, input);
    } else if (countZeros) {
      // Count 0 bits = bitwidth - ctpop(x)
      Value ctpop = LLVM::CtPopOp::create(rewriter, loc, inputType, input);
      Value width =
          hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, bitWidth));
      result = comb::SubOp::create(rewriter, loc, width, ctpop);
    } else {
      // No 0s or 1s to count (only X/Z which are 0 in two-valued)
      result = hw::ConstantOp::create(rewriter, loc, APInt(bitWidth, 0));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Simulation Control Conversion
//===----------------------------------------------------------------------===//

// moore.builtin.stop -> sim.pause
static LogicalResult convert(StopBIOp op, StopBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::PauseOp>(op, /*verbose=*/false);
  return success();
}

// moore.builtin.finish -> sim.terminate
static LogicalResult convert(FinishBIOp op, FinishBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::TerminateOp>(op, op.getExitCode() == 0,
                                                /*verbose=*/false);
  return success();
}

// moore.builtin.severity -> sim.proc.print
static LogicalResult convert(SeverityBIOp op, SeverityBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {

  std::string severityString;

  switch (op.getSeverity()) {
  case (Severity::Fatal):
    severityString = "Fatal: ";
    break;
  case (Severity::Error):
    severityString = "Error: ";
    break;
  case (Severity::Warning):
    severityString = "Warning: ";
    break;
  default:
    return failure();
  }

  auto prefix =
      sim::FormatLiteralOp::create(rewriter, op.getLoc(), severityString);
  auto message = sim::FormatStringConcatOp::create(
      rewriter, op.getLoc(), ValueRange{prefix, adaptor.getMessage()});
  rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(op, message);
  return success();
}

// moore.builtin.finish_message
static LogicalResult convert(FinishMessageBIOp op,
                             FinishMessageBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // We don't support printing termination/pause messages yet.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Timing Control Conversion
//===----------------------------------------------------------------------===//

// moore.builtin.time
static LogicalResult convert(TimeBIOp op, TimeBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::CurrentTimeOp>(op);
  return success();
}

// moore.logic_to_time
static LogicalResult convert(LogicToTimeOp op, LogicToTimeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::IntToTimeOp>(op, adaptor.getInput());
  return success();
}

// moore.time_to_logic
static LogicalResult convert(TimeToLogicOp op, TimeToLogicOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::TimeToIntOp>(op, adaptor.getInput());
  return success();
}

//===----------------------------------------------------------------------===//
// Random Number Generation Conversion
//===----------------------------------------------------------------------===//

namespace {

/// Conversion for moore.builtin.urandom -> runtime function call.
/// Implements the SystemVerilog $urandom system function.
struct UrandomBIOpConversion : public OpConversionPattern<UrandomBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UrandomBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    if (adaptor.getSeed()) {
      // Seeded version: __moore_urandom_seeded(seed) -> u32
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_urandom_seeded", fnTy);
      auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                         SymbolRefAttr::get(fn),
                                         ValueRange{adaptor.getSeed()});
      rewriter.replaceOp(op, result.getResult());
    } else {
      // Non-seeded version: __moore_urandom() -> u32
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_urandom", fnTy);
      auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                         SymbolRefAttr::get(fn), ValueRange{});
      rewriter.replaceOp(op, result.getResult());
    }

    return success();
  }
};

/// Conversion for moore.builtin.urandom_range -> runtime function call.
/// Implements the SystemVerilog $urandom_range system function.
struct UrandomRangeBIOpConversion
    : public OpConversionPattern<UrandomRangeBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UrandomRangeBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    // __moore_urandom_range(max, min) -> u32
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_urandom_range",
                                     fnTy);

    Value maxval = adaptor.getMaxval();
    Value minval = adaptor.getMinval();

    // If minval is not provided, use 0
    if (!minval) {
      minval = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(0));
    }

    auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{maxval, minval});
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Conversion for moore.builtin.random -> runtime function call.
/// Implements the SystemVerilog $random system function.
struct RandomBIOpConversion : public OpConversionPattern<RandomBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RandomBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    if (adaptor.getSeed()) {
      // Seeded version: __moore_random_seeded(seed) -> i32
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_random_seeded", fnTy);
      auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                         SymbolRefAttr::get(fn),
                                         ValueRange{adaptor.getSeed()});
      rewriter.replaceOp(op, result.getResult());
    } else {
      // Non-seeded version: __moore_random() -> i32
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_random", fnTy);
      auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                         SymbolRefAttr::get(fn), ValueRange{});
      rewriter.replaceOp(op, result.getResult());
    }

    return success();
  }
};

/// Helper structure to hold extracted range constraint information.
/// Supports both single-range and multi-range constraints.
struct RangeConstraintInfo {
  StringRef propertyName; // Name of the constrained property
  int64_t minValue;       // Minimum value (inclusive) - for single range
  int64_t maxValue;       // Maximum value (inclusive) - for single range
  SmallVector<std::pair<int64_t, int64_t>> ranges; // All ranges for multi-range
  unsigned fieldIndex;    // Index of the field in the struct
  unsigned bitWidth;      // Bit width of the property
  bool isSoft = false;    // Whether this is a soft constraint
  bool isMultiRange = false; // Whether this has multiple ranges
};

/// Helper structure to hold extracted soft constraint information.
/// Soft constraints provide default values that can be overridden by hard
/// constraints. Pattern: `constraint soft_c { soft value == 42; }`
struct SoftConstraintInfo {
  StringRef propertyName; // Name of the constrained property
  int64_t defaultValue;   // Default value when no hard constraint applies
  unsigned fieldIndex;    // Index of the field in the struct
  unsigned bitWidth;      // Bit width of the property
};

/// Extract simple range constraints from a class declaration.
/// Returns a list of range constraints found in ConstraintInsideOp operations.
/// Only extracts constraints that can be expressed as simple [min, max] ranges.
static SmallVector<RangeConstraintInfo>
extractRangeConstraints(ClassDeclOp classDecl, ClassTypeCache &cache,
                        SymbolRefAttr classSym) {
  SmallVector<RangeConstraintInfo> constraints;

  // Build a map from property names to their indices and types
  DenseMap<StringRef, std::pair<unsigned, Type>> propertyMap;
  unsigned propIdx = 0;

  // Account for base class if present (base class is at index 0)
  if (classDecl.getBaseAttr())
    propIdx = 1;

  // Also account for type ID field (index 0 in root classes)
  if (!classDecl.getBaseAttr())
    propIdx = 1; // Type ID is at index 0

  for (auto &op : classDecl.getBody().getOps()) {
    if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(op)) {
      propertyMap[propDecl.getSymName()] = {propIdx, propDecl.getPropertyType()};
      propIdx++;
    }
  }

  // Walk through constraint blocks looking for ConstraintInsideOp
  for (auto &op : classDecl.getBody().getOps()) {
    auto constraintBlock = dyn_cast<ConstraintBlockOp>(op);
    if (!constraintBlock)
      continue;

    // Walk the constraint block body
    for (auto &constraintOp : constraintBlock.getBody().getOps()) {
      // Look for ConstraintInsideOp which represents range constraints
      if (auto insideOp = dyn_cast<ConstraintInsideOp>(constraintOp)) {
        // Get the variable being constrained
        Value variable = insideOp.getVariable();

        // The variable should be a block argument referencing a property
        // In the constraint block, arguments correspond to random properties
        auto blockArg = dyn_cast<BlockArgument>(variable);
        if (!blockArg)
          continue;

        // Get the argument index - this corresponds to the order of rand properties
        unsigned argIdx = blockArg.getArgNumber();

        // Find the corresponding property by counting rand properties
        unsigned randPropCount = 0;
        StringRef propName;
        Type propType;
        for (auto &innerOp : classDecl.getBody().getOps()) {
          if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(innerOp)) {
            if (propDecl.isRandomizable()) {
              if (randPropCount == argIdx) {
                propName = propDecl.getSymName();
                propType = propDecl.getPropertyType();
                break;
              }
              randPropCount++;
            }
          }
        }

        if (propName.empty())
          continue;

        // Get the ranges array - pairs of [low, high] values
        ArrayRef<int64_t> rangesArr = insideOp.getRanges();

        // Ranges must come in pairs (even number of values, at least 2)
        if (rangesArr.size() < 2 || rangesArr.size() % 2 != 0)
          continue;

        // Find the property index and bit width
        auto it = propertyMap.find(propName);
        if (it == propertyMap.end())
          continue;

        unsigned fieldIdx = it->second.first;
        Type fieldType = it->second.second;

        // Get bit width from the type
        unsigned bitWidth = 32; // Default
        if (auto intType = dyn_cast<IntType>(fieldType))
          bitWidth = intType.getWidth();

        RangeConstraintInfo info;
        info.propertyName = propName;
        info.fieldIndex = fieldIdx;
        info.bitWidth = bitWidth;
        info.isSoft = insideOp.getIsSoft();

        // Check if this is a single range or multiple ranges
        if (rangesArr.size() == 2) {
          // Single range - use the simple path
          info.minValue = rangesArr[0];
          info.maxValue = rangesArr[1];
          info.isMultiRange = false;
        } else {
          // Multiple ranges - store all range pairs
          info.isMultiRange = true;
          // Also set minValue/maxValue to the first range for compatibility
          info.minValue = rangesArr[0];
          info.maxValue = rangesArr[1];
          // Store all ranges as pairs
          for (size_t i = 0; i < rangesArr.size(); i += 2) {
            info.ranges.push_back({rangesArr[i], rangesArr[i + 1]});
          }
        }
        constraints.push_back(info);
      }
    }
  }

  return constraints;
}

/// Extract soft constraints from a class declaration.
/// Soft constraints provide default values that are applied when no hard
/// constraint specifies a value for the property.
/// Pattern: `constraint soft_c { soft value == 42; }` -> value defaults to 42
/// Also supports soft inside: `soft value inside {[0:10]}` -> defaults to first
/// value.
static SmallVector<SoftConstraintInfo>
extractSoftConstraints(ClassDeclOp classDecl, ClassTypeCache &cache,
                       SymbolRefAttr classSym) {
  SmallVector<SoftConstraintInfo> softConstraints;

  // Build a map from property names to their indices and types
  DenseMap<StringRef, std::pair<unsigned, Type>> propertyMap;
  unsigned propIdx = 0;

  // Account for type ID field (index 0 in root classes) or base class
  if (classDecl.getBaseAttr())
    propIdx = 1;
  else
    propIdx = 1; // Type ID is at index 0

  for (auto &op : classDecl.getBody().getOps()) {
    if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(op)) {
      propertyMap[propDecl.getSymName()] = {propIdx, propDecl.getPropertyType()};
      propIdx++;
    }
  }

  // Walk through constraint blocks looking for soft constraints
  for (auto &op : classDecl.getBody().getOps()) {
    auto constraintBlock = dyn_cast<ConstraintBlockOp>(op);
    if (!constraintBlock)
      continue;

    // Walk the constraint block body looking for soft ConstraintInsideOp
    // with single-value ranges (representing equality constraints)
    for (auto &constraintOp : constraintBlock.getBody().getOps()) {
      // Look for soft ConstraintInsideOp with a single value range
      if (auto insideOp = dyn_cast<ConstraintInsideOp>(constraintOp)) {
        // Only process soft constraints
        if (!insideOp.getIsSoft())
          continue;

        // Get the variable being constrained
        Value variable = insideOp.getVariable();

        // The variable should be a block argument referencing a property
        auto blockArg = dyn_cast<BlockArgument>(variable);
        if (!blockArg)
          continue;

        // Get the argument index - this corresponds to the order of rand
        // properties
        unsigned argIdx = blockArg.getArgNumber();

        // Find the corresponding property by counting rand properties
        unsigned randPropCount = 0;
        StringRef propName;
        Type propType;
        for (auto &innerOp : classDecl.getBody().getOps()) {
          if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(innerOp)) {
            if (propDecl.isRandomizable()) {
              if (randPropCount == argIdx) {
                propName = propDecl.getSymName();
                propType = propDecl.getPropertyType();
                break;
              }
              randPropCount++;
            }
          }
        }

        if (propName.empty())
          continue;

        // Get the ranges array
        ArrayRef<int64_t> ranges = insideOp.getRanges();

        // For soft constraints, extract the default value
        // If it's a single value (low == high), use that as default
        // If it's a range, use the low value as default
        if (ranges.size() < 2)
          continue;

        int64_t defaultValue = ranges[0];
        // If low == high, it's an equality constraint (soft value == X)
        // If low != high, it's a range, and we use low as default

        // Find the property index and bit width
        auto it = propertyMap.find(propName);
        if (it == propertyMap.end())
          continue;

        unsigned fieldIdx = it->second.first;
        Type fieldType = it->second.second;

        // Get bit width from the type
        unsigned bitWidth = 32; // Default
        if (auto intType = dyn_cast<IntType>(fieldType))
          bitWidth = intType.getWidth();

        SoftConstraintInfo info;
        info.propertyName = propName;
        info.defaultValue = defaultValue;
        info.fieldIndex = fieldIdx;
        info.bitWidth = bitWidth;
        softConstraints.push_back(info);
      }
    }
  }

  return softConstraints;
}

static Type resolveStructFieldType(LLVM::LLVMStructType structTy,
                                   ArrayRef<unsigned> path) {
  Type current = structTy;
  for (unsigned index : path) {
    auto currentStruct = dyn_cast<LLVM::LLVMStructType>(current);
    if (!currentStruct)
      return {};
    auto body = currentStruct.getBody();
    if (index >= body.size())
      return {};
    current = body[index];
  }
  return current;
}

/// Conversion for moore.randomize -> runtime function call.
/// Implements the SystemVerilog randomize() method on class objects.
/// Supports constraint-aware randomization for simple range constraints.
struct RandomizeOpConversion : public OpConversionPattern<RandomizeOp> {
  RandomizeOpConversion(TypeConverter &tc, MLIRContext *ctx,
                        ClassTypeCache &cache)
      : OpConversionPattern<RandomizeOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(RandomizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // Get the class handle type and resolve the class struct info
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    auto classSym = handleTy.getClassSym();

    if (failed(resolveClassStructBody(mod, classSym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve class struct for " << classSym;

    auto structInfo = cache.getStructInfo(classSym);
    if (!structInfo)
      return op.emitError() << "No struct info for class " << classSym;

    auto structTy = structInfo->classBody;

    // Calculate the class size (handles llhd::TimeType which DataLayout doesn't support).
    uint64_t byteSize = getTypeSizeSafe(structTy, mod);

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);

    // Get the class instance pointer from the adaptor
    Value classPtr = adaptor.getObject();

    // Look up the class declaration to extract constraints
    auto *classDeclSym = mod.lookupSymbol(classSym);
    auto classDecl = dyn_cast_or_null<ClassDeclOp>(classDeclSym);

    SmallVector<std::pair<Value, Value>> preservedFields;
    if (classDecl) {
      for (auto propDecl : classDecl.getBody().getOps<ClassPropertyDeclOp>()) {
        if (propDecl.isRandomizable())
          continue;
        auto pathOpt = structInfo->getFieldPath(propDecl.getSymName());
        if (!pathOpt)
          continue;
        Type fieldTy = resolveStructFieldType(structTy, *pathOpt);
        if (!fieldTy)
          continue;

        SmallVector<LLVM::GEPArg> gepIndices;
        gepIndices.push_back(0);
        for (unsigned idx : *pathOpt)
          gepIndices.push_back(static_cast<int32_t>(idx));

        auto fieldPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy,
                                            classPtr, gepIndices);
        auto fieldVal =
            LLVM::LoadOp::create(rewriter, loc, fieldTy, fieldPtr);
        preservedFields.push_back({fieldPtr, fieldVal});
      }
    }

    auto restorePreservedFields = [&]() {
      for (auto &entry : preservedFields)
        LLVM::StoreOp::create(rewriter, loc, entry.second, entry.first);
    };

    auto applyRandcFields = [&](const llvm::DenseSet<StringRef> *hardConstrained,
                                const llvm::DenseSet<StringRef> *softConstrained) {
      if (!classDecl)
        return;

      auto randcFnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
      auto randcFn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_randc_next",
                                            randcFnTy);

      for (auto propDecl : classDecl.getBody().getOps<ClassPropertyDeclOp>()) {
        if (propDecl.getRandMode() != RandMode::RandC)
          continue;
        if ((hardConstrained &&
             hardConstrained->contains(propDecl.getSymName())) ||
            (softConstrained &&
             softConstrained->contains(propDecl.getSymName())))
          continue;

        auto pathOpt = structInfo->getFieldPath(propDecl.getSymName());
        if (!pathOpt)
          continue;
        Type fieldTy = resolveStructFieldType(structTy, *pathOpt);
        auto intTy = dyn_cast_or_null<IntegerType>(fieldTy);
        if (!intTy)
          continue;

        unsigned bitWidth = intTy.getWidth();
        if (bitWidth == 0)
          continue;

        SmallVector<LLVM::GEPArg> gepIndices;
        gepIndices.push_back(0);
        for (unsigned idx : *pathOpt)
          gepIndices.push_back(static_cast<int32_t>(idx));
        auto fieldPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy,
                                            classPtr, gepIndices);

        auto widthConst = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bitWidth));
        auto randcValue = LLVM::CallOp::create(
            rewriter, loc, TypeRange{i64Ty}, SymbolRefAttr::get(randcFn),
            ValueRange{fieldPtr, widthConst});

        Value converted = randcValue.getResult();
        if (bitWidth < 64) {
          converted =
              arith::TruncIOp::create(rewriter, loc, intTy, converted);
        } else if (bitWidth > 64) {
          converted =
              arith::ExtSIOp::create(rewriter, loc, intTy, converted);
        }
        LLVM::StoreOp::create(rewriter, loc, converted, fieldPtr);
      }
    };

    // Extract range constraints from the class (includes both hard and soft)
    SmallVector<RangeConstraintInfo> rangeConstraints;
    SmallVector<SoftConstraintInfo> softConstraints;
    if (classDecl) {
      rangeConstraints = extractRangeConstraints(classDecl, cache, classSym);
      softConstraints = extractSoftConstraints(classDecl, cache, classSym);
    }

    // Separate hard constraints from soft range constraints
    SmallVector<RangeConstraintInfo> hardConstraints;
    llvm::DenseSet<StringRef> hardConstrainedProps;
    for (const auto &constraint : rangeConstraints) {
      if (!constraint.isSoft) {
        hardConstraints.push_back(constraint);
        hardConstrainedProps.insert(constraint.propertyName);
      }
    }

    // Filter soft constraints to only those without hard constraints
    SmallVector<SoftConstraintInfo> effectiveSoftConstraints;
    for (const auto &soft : softConstraints) {
      if (!hardConstrainedProps.contains(soft.propertyName)) {
        effectiveSoftConstraints.push_back(soft);
      }
    }

    // If we have any constraints, use constraint-aware randomization
    if (!hardConstraints.empty() || !effectiveSoftConstraints.empty()) {
      llvm::DenseSet<StringRef> softConstrainedProps;
      for (const auto &soft : effectiveSoftConstraints)
        softConstrainedProps.insert(soft.propertyName);

      // First, do basic randomization for the whole class
      auto classSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(byteSize));
      auto basicFnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty});
      auto basicFn = getOrCreateRuntimeFunc(mod, rewriter,
                                            "__moore_randomize_basic", basicFnTy);
      LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                           SymbolRefAttr::get(basicFn),
                           ValueRange{classPtr, classSizeConst});

      // Apply hard range constraints - generate constrained random values
      if (!hardConstraints.empty()) {
        // Function type for single-range: __moore_randomize_with_range(i64, i64) -> i64
        auto rangeFnTy = LLVM::LLVMFunctionType::get(i64Ty, {i64Ty, i64Ty});
        auto rangeFn = getOrCreateRuntimeFunc(mod, rewriter,
                                              "__moore_randomize_with_range",
                                              rangeFnTy);

        // Function type for multi-range: __moore_randomize_with_ranges(ptr, i64) -> i64
        auto rangesFnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
        auto rangesFn = getOrCreateRuntimeFunc(mod, rewriter,
                                               "__moore_randomize_with_ranges",
                                               rangesFnTy);

        for (const auto &constraint : hardConstraints) {
          Value rangeResultVal;

          if (constraint.isMultiRange) {
            // Multi-range constraint - create an array of range pairs and call
            // __moore_randomize_with_ranges

            // Create array type for the ranges: [numRanges * 2 x i64]
            size_t numRanges = constraint.ranges.size();
            auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, numRanges * 2);

            // Allocate stack space for the ranges array
            auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(1));
            auto rangesAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy,
                                                       arrayTy, one);

            // Store each range pair into the array
            for (size_t i = 0; i < numRanges; ++i) {
              // Store min value at index i*2
              auto minIdx = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty,
                  rewriter.getI64IntegerAttr(static_cast<int64_t>(i * 2)));
              auto minPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                                rangesAlloca,
                                                ValueRange{minIdx});
              auto minVal = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty,
                  rewriter.getI64IntegerAttr(constraint.ranges[i].first));
              LLVM::StoreOp::create(rewriter, loc, minVal, minPtr);

              // Store max value at index i*2+1
              auto maxIdx = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty,
                  rewriter.getI64IntegerAttr(static_cast<int64_t>(i * 2 + 1)));
              auto maxPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                                rangesAlloca,
                                                ValueRange{maxIdx});
              auto maxVal = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty,
                  rewriter.getI64IntegerAttr(constraint.ranges[i].second));
              LLVM::StoreOp::create(rewriter, loc, maxVal, maxPtr);
            }

            // Call __moore_randomize_with_ranges(ranges_ptr, num_ranges)
            auto numRangesConst = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(static_cast<int64_t>(numRanges)));
            auto rangeResult = LLVM::CallOp::create(
                rewriter, loc, TypeRange{i64Ty}, SymbolRefAttr::get(rangesFn),
                ValueRange{rangesAlloca, numRangesConst});
            rangeResultVal = rangeResult.getResult();
          } else {
            // Single range constraint - use existing __moore_randomize_with_range
            auto minConst = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(constraint.minValue));
            auto maxConst = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(constraint.maxValue));

            // Call __moore_randomize_with_range(min, max)
            auto rangeResult = LLVM::CallOp::create(
                rewriter, loc, TypeRange{i64Ty}, SymbolRefAttr::get(rangeFn),
                ValueRange{minConst, maxConst});
            rangeResultVal = rangeResult.getResult();
          }

          // Truncate to the field's bit width if needed
          Type fieldIntTy = IntegerType::get(ctx, constraint.bitWidth);
          Value truncatedVal = rangeResultVal;
          if (constraint.bitWidth < 64) {
            truncatedVal = arith::TruncIOp::create(rewriter, loc, fieldIntTy,
                                                   rangeResultVal);
          }

          // Get GEP to the field using the property path
          auto it = structInfo->propertyPath.find(constraint.propertyName);
          if (it != structInfo->propertyPath.end()) {
            // Build GEP indices
            SmallVector<LLVM::GEPArg> gepIndices;
            gepIndices.push_back(0); // Initial pointer dereference
            for (unsigned idx : it->second) {
              gepIndices.push_back(static_cast<int32_t>(idx));
            }

            // Create GEP to the field
            auto fieldPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy,
                                                classPtr, gepIndices);

            // Store the constrained random value
            LLVM::StoreOp::create(rewriter, loc, truncatedVal, fieldPtr);
          }
        }
      }

      // Apply soft constraints - set default values for properties without hard
      // constraints. Soft constraints provide fallback values that can be
      // overridden.
      for (const auto &soft : effectiveSoftConstraints) {
        // Create the default value constant
        Type fieldIntTy = IntegerType::get(ctx, soft.bitWidth);
        Value defaultVal;
        if (soft.bitWidth <= 64) {
          auto i64Val = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty,
              rewriter.getI64IntegerAttr(soft.defaultValue));
          if (soft.bitWidth < 64) {
            defaultVal =
                arith::TruncIOp::create(rewriter, loc, fieldIntTy, i64Val);
          } else {
            defaultVal = i64Val;
          }
        } else {
          // For larger bit widths, just use the low 64 bits
          auto i64Val = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty,
              rewriter.getI64IntegerAttr(soft.defaultValue));
          defaultVal =
              arith::ExtSIOp::create(rewriter, loc, fieldIntTy, i64Val);
        }

        // Get GEP to the field using the property path
        auto it = structInfo->propertyPath.find(soft.propertyName);
        if (it != structInfo->propertyPath.end()) {
          // Build GEP indices
          SmallVector<LLVM::GEPArg> gepIndices;
          gepIndices.push_back(0); // Initial pointer dereference
          for (unsigned idx : it->second) {
            gepIndices.push_back(static_cast<int32_t>(idx));
          }

          // Create GEP to the field
          auto fieldPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy,
                                              classPtr, gepIndices);

          // Store the soft constraint default value
          LLVM::StoreOp::create(rewriter, loc, defaultVal, fieldPtr);
        }
      }

      applyRandcFields(&hardConstrainedProps, &softConstrainedProps);
      // Return success
      restorePreservedFields();
      auto successVal = hw::ConstantOp::create(rewriter, loc, i1Ty, 1);
      rewriter.replaceOp(op, successVal);
      return success();
    }

    // No constraints - use basic randomization
    auto classSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(byteSize));

    // __moore_randomize_basic(void *classPtr, int64_t classSize) -> i32
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_randomize_basic", fnTy);

    auto result = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{classPtr, classSizeConst});

    // The runtime returns i32 (1 for success, 0 for failure), but the op
    // returns i1. Truncate the result to i1.
    auto truncResult =
        arith::TruncIOp::create(rewriter, loc, i1Ty, result.getResult());

    applyRandcFields(nullptr, nullptr);
    restorePreservedFields();
    rewriter.replaceOp(op, truncResult);
    return success();
  }

private:
  ClassTypeCache &cache;
};

/// Conversion for moore.std_randomize -> runtime function call.
/// Implements the SystemVerilog std::randomize() function for standalone
/// variable randomization (IEEE 1800-2017 Section 18.12).
struct StdRandomizeOpConversion : public OpConversionPattern<StdRandomizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdRandomizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);

    // Get or create runtime function: int __moore_random()
    // Returns a random 32-bit integer.
    auto randomFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
    auto randomFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_random", randomFnTy);

    // For each variable, generate a random value and drive it to the signal.
    for (auto [origVar, convertedVar] :
         llvm::zip(op.getVariables(), adaptor.getVariables())) {
      // Get the element type of the ref
      auto refTy = dyn_cast<llhd::RefType>(convertedVar.getType());
      if (!refTy) {
        return op.emitError() << "expected llhd.ref type, got "
                              << convertedVar.getType();
      }
      auto elemTy = refTy.getNestedType();
      auto intTy = dyn_cast<IntegerType>(elemTy);
      if (!intTy) {
        return op.emitError()
               << "std::randomize only supports integer types, got " << elemTy;
      }

      // Generate random value(s) to fill the integer
      unsigned bitWidth = intTy.getWidth();
      Value randomVal;

      if (bitWidth <= 32) {
        // Single call is sufficient
        auto callResult =
            LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                 SymbolRefAttr::get(randomFn), ValueRange{});
        randomVal = callResult.getResult();
        if (bitWidth < 32) {
          randomVal = arith::TruncIOp::create(rewriter, loc, intTy, randomVal);
        }
      } else {
        // Need multiple calls to fill larger integers
        unsigned numCalls = (bitWidth + 31) / 32;
        randomVal = hw::ConstantOp::create(rewriter, loc, intTy, 0);

        for (unsigned i = 0; i < numCalls; ++i) {
          auto callResult =
              LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                   SymbolRefAttr::get(randomFn), ValueRange{});
          Value chunk = arith::ExtUIOp::create(rewriter, loc, intTy,
                                               callResult.getResult());
          if (i > 0) {
            auto shiftAmt =
                hw::ConstantOp::create(rewriter, loc, intTy, i * 32);
            chunk = comb::ShlOp::create(rewriter, loc, chunk, shiftAmt);
          }
          randomVal = comb::OrOp::create(rewriter, loc, randomVal, chunk);
        }
      }

      // Create a time delay (0 ns, 0 delta, 1 epsilon) for the drive
      auto timeAttr =
          llhd::TimeAttr::get(ctx, 0, "ns", 0, 1);
      auto timeConst = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);

      // Drive the random value to the signal (no enable condition)
      llhd::DriveOp::create(rewriter, loc, convertedVar, randomVal, timeConst,
                            Value{});
    }

    // Return success (1)
    auto i1Ty = IntegerType::get(ctx, 1);
    auto successVal = hw::ConstantOp::create(rewriter, loc, i1Ty, 1);
    rewriter.replaceOp(op, successVal);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target,
                             const TypeConverter &converter) {
  target.addIllegalDialect<MooreDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<sim::SimDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  target.addLegalOp<debug::ScopeOp>();

  target.addDynamicallyLegalOp<scf::YieldOp, func::CallOp, func::CallIndirectOp,
                               func::ReturnOp, func::ConstantOp,
                               UnrealizedConversionCastOp, hw::OutputOp,
                               hw::InstanceOp, debug::ArrayOp, debug::StructOp,
                               debug::VariableOp>(
      [&](Operation *op) { return converter.isLegal(op); });

  target.addDynamicallyLegalOp<scf::IfOp, scf::ForOp, scf::ExecuteRegionOp,
                               scf::WhileOp, scf::ForallOp>([&](Operation *op) {
    return converter.isLegal(op) && !op->getParentOfType<llhd::ProcessOp>();
  });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
    return converter.isSignatureLegal(op.getModuleType().getFuncType()) &&
           converter.isLegal(&op.getBody());
  });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](IntType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });

  typeConverter.addConversion([&](RealType type) -> mlir::Type {
    MLIRContext *ctx = type.getContext();
    switch (type.getWidth()) {
    case moore::RealWidth::f32:
      return mlir::Float32Type::get(ctx);
    case moore::RealWidth::f64:
      return mlir::Float64Type::get(ctx);
    }
    llvm_unreachable("unhandled RealWidth");
  });

  typeConverter.addConversion(
      [&](TimeType type) { return llhd::TimeType::get(type.getContext()); });

  // EventType -> i1 (tracks whether the event has been triggered).
  // In simulation, events are tracked as boolean flags indicating their
  // triggered state within the current time slot.
  typeConverter.addConversion([&](EventType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });

  typeConverter.addConversion([&](FormatStringType type) {
    return sim::FormatStringType::get(type.getContext());
  });

  typeConverter.addConversion([&](ArrayType type) -> std::optional<Type> {
    if (auto elementType = typeConverter.convertType(type.getElementType()))
      return hw::ArrayType::get(elementType, type.getSize());
    return {};
  });

  // FIXME: Unpacked arrays support more element types than their packed
  // variants, and as such, mapping them to hw::Array is somewhat naive. See
  // also the analogous note below concerning unpacked struct type conversion.
  typeConverter.addConversion(
      [&](UnpackedArrayType type) -> std::optional<Type> {
        if (auto elementType = typeConverter.convertType(type.getElementType()))
          return hw::ArrayType::get(elementType, type.getSize());
        return {};
      });

  typeConverter.addConversion([&](StructType type) -> std::optional<Type> {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::StructType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      fields.push_back(info);
    }
    return hw::StructType::get(type.getContext(), fields);
  });

  // FIXME: Mapping unpacked struct type to struct type in hw dialect may be a
  // plain solution. The packed and unpacked data structures have some
  // differences though they look similarily. The packed data structure is
  // contiguous in memory but another is opposite. The differences will affect
  // data layout and granularity of event tracking in simulation.
  typeConverter.addConversion(
      [&](UnpackedStructType type) -> std::optional<Type> {
        auto *ctx = type.getContext();
        SmallVector<Type> fieldTypes;
        bool hasLLVMType = false;

        // First pass: convert all field types and check for LLVM types
        for (auto field : type.getMembers()) {
          auto convertedType = typeConverter.convertType(field.type);
          if (!convertedType)
            return {};
          fieldTypes.push_back(convertedType);
          // Check if any field converts to an LLVM type (strings, queues,
          // dynamic arrays, assoc arrays, time, or nested structs with these)
          if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType, llhd::TimeType>(
                  convertedType))
            hasLLVMType = true;
        }

        // If any field is an LLVM type, use LLVM struct for the whole struct
        // This is necessary because hw::StructType cannot contain LLVM types
        if (hasLLVMType) {
          // Convert hw types to LLVM types for the struct
          SmallVector<Type> llvmFieldTypes;
          for (auto fieldType : fieldTypes) {
            auto llvmType = convertToLLVMType(fieldType);
            if (!llvmType)
              return {};
            llvmFieldTypes.push_back(llvmType);
          }
          return LLVM::LLVMStructType::getLiteral(ctx, llvmFieldTypes);
        }

        // Otherwise use hw::StructType
        SmallVector<hw::StructType::FieldInfo> fields;
        auto members = type.getMembers();
        for (size_t i = 0; i < members.size(); ++i) {
          hw::StructType::FieldInfo info;
          info.type = fieldTypes[i];
          info.name = members[i].name;
          fields.push_back(info);
        }
        return hw::StructType::get(ctx, fields);
      });

  // QueueType -> LLVM struct {ptr, i64} representing a dynamic queue.
  typeConverter.addConversion([&](QueueType type) -> std::optional<Type> {
    auto *ctx = type.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  });

  // StringType -> LLVM struct {ptr, i64} representing a dynamic string.
  typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
    auto *ctx = type.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  });

  // OpenUnpackedArrayType (dynamic array) -> LLVM struct {ptr, i64}.
  typeConverter.addConversion(
      [&](OpenUnpackedArrayType type) -> std::optional<Type> {
        auto *ctx = type.getContext();
        auto ptrTy = LLVM::LLVMPointerType::get(ctx);
        auto i64Ty = IntegerType::get(ctx, 64);
        return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
      });

  // AssocArrayType (associative array) -> LLVM pointer (opaque map handle).
  typeConverter.addConversion([&](AssocArrayType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Convert packed union type to hw::UnionType
  typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
    SmallVector<hw::UnionType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::UnionType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      info.offset = 0; // All union members share the same offset
      fields.push_back(info);
    }
    return hw::UnionType::get(type.getContext(), fields);
  });

  // Convert unpacked union type to hw::UnionType
  typeConverter.addConversion(
      [&](UnpackedUnionType type) -> std::optional<Type> {
        SmallVector<hw::UnionType::FieldInfo> fields;
        for (auto field : type.getMembers()) {
          hw::UnionType::FieldInfo info;
          info.type = typeConverter.convertType(field.type);
          if (!info.type)
            return {};
          info.name = field.name;
          info.offset = 0; // All union members share the same offset
          fields.push_back(info);
        }
        return hw::UnionType::get(type.getContext(), fields);
      });

  // Conversion of CHandle to LLVMPointerType
  typeConverter.addConversion([&](ChandleType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Explicitly mark LLVM types as legal targets
  typeConverter.addConversion(
      [](LLVM::LLVMPointerType t) -> std::optional<Type> { return t; });
  typeConverter.addConversion(
      [](LLVM::LLVMStructType t) -> std::optional<Type> { return t; });

  // ClassHandleType  ->  !llvm.ptr
  typeConverter.addConversion([&](ClassHandleType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // CovergroupHandleType -> !llvm.ptr
  // Covergroup instances are pointers to runtime coverage data structures.
  typeConverter.addConversion(
      [&](CovergroupHandleType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });

  // VirtualInterfaceType -> !llvm.ptr
  // Virtual interfaces are pointers to interface struct instances.
  typeConverter.addConversion(
      [&](VirtualInterfaceType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });

  // EventType -> i1 (tracks whether the event has been triggered).
  typeConverter.addConversion([&](EventType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });

  typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
    auto nestedType = type.getNestedType();
    if (auto innerType = typeConverter.convertType(nestedType)) {
      // For dynamic container types (queues, dynamic arrays, associative
      // arrays, and strings), return an LLVM pointer instead of llhd.ref.
      // These are dynamic types that don't fit the llhd signal/probe model.
      // Check the original Moore type to distinguish from other pointer types.
      if (isa<QueueType, OpenUnpackedArrayType, AssocArrayType, StringType>(
              nestedType))
        return LLVM::LLVMPointerType::get(type.getContext());
      // If the inner type converted to an LLVM struct (e.g., unpacked struct
      // containing dynamic types like strings), also use LLVM pointer.
      if (isa<LLVM::LLVMStructType>(innerType))
        return LLVM::LLVMPointerType::get(type.getContext());
      return llhd::RefType::get(innerType);
    }
    return {};
  });

  // Valid target types.
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });
  typeConverter.addConversion([](llhd::TimeType type) { return type; });
  typeConverter.addConversion([](debug::ArrayType type) { return type; });
  typeConverter.addConversion([](debug::ScopeType type) { return type; });
  typeConverter.addConversion([](debug::StructType type) { return type; });

  // Function types (used for function pointers from vtable.load_method).
  typeConverter.addConversion(
      [&](mlir::FunctionType type) -> std::optional<Type> {
        SmallVector<Type> inputs;
        for (auto input : type.getInputs()) {
          auto converted = typeConverter.convertType(input);
          if (!converted)
            return {};
          inputs.push_back(converted);
        }
        SmallVector<Type> results;
        for (auto result : type.getResults()) {
          auto converted = typeConverter.convertType(result);
          if (!converted)
            return {};
          results.push_back(converted);
        }
        return mlir::FunctionType::get(type.getContext(), inputs, results);
      });

  typeConverter.addConversion([&](llhd::RefType type) -> std::optional<Type> {
    if (auto innerType = typeConverter.convertType(type.getNestedType()))
      return llhd::RefType::get(innerType);
    return {};
  });

  typeConverter.addConversion([&](hw::ArrayType type) -> std::optional<Type> {
    if (auto elementType = typeConverter.convertType(type.getElementType()))
      return hw::ArrayType::get(elementType, type.getNumElements());
    return {};
  });

  typeConverter.addConversion([&](hw::StructType type) -> std::optional<Type> {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto field : type.getElements()) {
      hw::StructType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      fields.push_back(info);
    }
    return hw::StructType::get(type.getContext(), fields);
  });

  // hw::UnionType is a legal target type - recursively convert nested types.
  typeConverter.addConversion([&](hw::UnionType type) -> std::optional<Type> {
    SmallVector<hw::UnionType::FieldInfo> fields;
    for (auto field : type.getElements()) {
      hw::UnionType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      info.offset = field.offset;
      fields.push_back(info);
    }
    return hw::UnionType::get(type.getContext(), fields);
  });

  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1 || !inputs[0])
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            .getResult(0);
      });

  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });
}

static void populateOpConversion(ConversionPatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 ClassTypeCache &classCache,
                                 InterfaceTypeCache &interfaceCache) {

  patterns.add<ClassDeclOpConversion>(typeConverter, patterns.getContext(),
                                      classCache);
  patterns.add<ClassNewOpConversion>(typeConverter, patterns.getContext(),
                                     classCache);
  patterns.add<ClassPropertyRefOpConversion>(typeConverter,
                                             patterns.getContext(), classCache);
  // ClassDynCastOpConversion needs cache for RTTI type ID lookup
  patterns.add<ClassDynCastOpConversion>(typeConverter, patterns.getContext(),
                                         classCache);

  // Virtual interface patterns.
  patterns.add<InterfaceDeclOpConversion>(typeConverter, patterns.getContext(),
                                          interfaceCache);
  patterns.add<VirtualInterfaceSignalRefOpConversion>(
      typeConverter, patterns.getContext(), interfaceCache);
  patterns.add<InterfaceInstanceOpConversion>(typeConverter,
                                              patterns.getContext(),
                                              interfaceCache);
  patterns.add<InterfaceSignalDeclOpConversion>(typeConverter,
                                                patterns.getContext());
  patterns.add<ModportDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<VirtualInterfaceGetOpConversion>(typeConverter,
                                                patterns.getContext());

  // Clocking block patterns (erased during lowering).
  patterns.add<ClockingBlockDeclOpConversion>(typeConverter,
                                              patterns.getContext());
  patterns.add<ClockingSignalOpConversion>(typeConverter, patterns.getContext());

  // Coverage patterns (erased during lowering).
  patterns.add<CovergroupDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CoverpointDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CoverCrossDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CovergroupInstOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CovergroupSampleOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CovergroupGetCoverageOpConversion>(typeConverter, patterns.getContext());

  // Constraint patterns (processed during RandomizeOp lowering, then erased).
  patterns.add<ConstraintBlockOpConversion>(typeConverter, patterns.getContext());
  patterns.add<ConstraintExprOpConversion>(typeConverter, patterns.getContext());
  patterns.add<ConstraintImplicationOpConversion>(typeConverter,
                                                  patterns.getContext());
  patterns.add<ConstraintIfElseOpConversion>(typeConverter,
                                             patterns.getContext());
  patterns.add<ConstraintForeachOpConversion>(typeConverter,
                                              patterns.getContext());
  patterns.add<ConstraintDistOpConversion>(typeConverter, patterns.getContext());
  patterns.add<ConstraintInsideOpConversion>(typeConverter,
                                             patterns.getContext());
  patterns.add<ConstraintSolveBeforeOpConversion>(typeConverter,
                                                  patterns.getContext());
  patterns.add<ConstraintDisableOpConversion>(typeConverter,
                                              patterns.getContext());
  patterns.add<ConstraintUniqueOpConversion>(typeConverter,
                                             patterns.getContext());

  // Patterns of vtable operations (with explicit benefits for ordering).
  patterns.add<VTableOpConversion>(typeConverter, patterns.getContext());
  patterns.add<VTableEntryOpConversion>(typeConverter, patterns.getContext());
  patterns.add<VTableLoadMethodOpConversion>(typeConverter, patterns.getContext());

  // clang-format off
  patterns.add<
    ClassUpcastOpConversion,
    ClassNullOpConversion,
    ClassHandleCmpOpConversion,
    // Virtual interface comparison patterns.
    VirtualInterfaceNullOpConversion,
    VirtualInterfaceCmpOpConversion,
    // Patterns of declaration operations.
    VariableOpConversion,
    NetOpConversion,
    GlobalVariableOpConversion,
    GetGlobalVariableOpConversion,

    // Patterns for conversion operations.
    ConversionOpConversion,
    BitcastConversion<PackedToSBVOp>,
    BitcastConversion<SBVToPackedOp>,
    BitcastConversion<LogicToIntOp>,
    BitcastConversion<IntToLogicOp>,
    BitcastConversion<ToBuiltinBoolOp>,
    TruncOpConversion,
    ZExtOpConversion,
    SExtOpConversion,
    SIntToRealOpConversion,
    UIntToRealOpConversion,
    RealToIntOpConversion,

    // Patterns of miscellaneous operations.
    ConstantOpConv,
    ConstantRealOpConv,
    ConcatOpConversion,
    ReplicateOpConversion,
    ConstantTimeOpConv,
    ExtractOpConversion,
    DynExtractOpConversion,
    DynExtractRefOpConversion,
    ReadOpConversion,
    StructExtractOpConversion,
    StructExtractRefOpConversion,
    ExtractRefOpConversion,
    StructCreateOpConversion,
    UnionCreateOpConversion,
    UnionExtractOpConversion,
    UnionExtractRefOpConversion,
    ConditionalOpConversion,
    ArrayCreateOpConversion,
    YieldOpConversion,
    OutputOpConversion,
    ConstantStringOpConv,

    // Patterns of unary operations.
    ReduceAndOpConversion,
    ReduceOrOpConversion,
    ReduceXorOpConversion,
    BoolCastOpConversion,
    NotOpConversion,
    NegOpConversion,
    NegRealOpConversion,

    // Patterns of binary operations.
    BinaryOpConversion<AddOp, comb::AddOp>,
    BinaryOpConversion<SubOp, comb::SubOp>,
    BinaryOpConversion<MulOp, comb::MulOp>,
    BinaryOpConversion<DivUOp, comb::DivUOp>,
    BinaryOpConversion<DivSOp, comb::DivSOp>,
    BinaryOpConversion<ModUOp, comb::ModUOp>,
    BinaryOpConversion<ModSOp, comb::ModSOp>,
    BinaryOpConversion<AndOp, comb::AndOp>,
    BinaryOpConversion<OrOp, comb::OrOp>,
    BinaryOpConversion<XorOp, comb::XorOp>,

    // Patterns for binary real operations.
    BinaryRealOpConversion<AddRealOp, arith::AddFOp>,
    BinaryRealOpConversion<SubRealOp, arith::SubFOp>,
    BinaryRealOpConversion<DivRealOp, arith::DivFOp>,
    BinaryRealOpConversion<MulRealOp, arith::MulFOp>,
    BinaryRealOpConversion<PowRealOp, math::PowFOp>,

    // Patterns for unary real math operations.
    UnaryRealOpConversion<SinBIOp, math::SinOp>,
    UnaryRealOpConversion<CosBIOp, math::CosOp>,
    UnaryRealOpConversion<TanBIOp, math::TanOp>,
    UnaryRealOpConversion<AsinBIOp, math::AsinOp>,
    UnaryRealOpConversion<AcosBIOp, math::AcosOp>,
    UnaryRealOpConversion<AtanBIOp, math::AtanOp>,
    UnaryRealOpConversion<SinhBIOp, math::SinhOp>,
    UnaryRealOpConversion<CoshBIOp, math::CoshOp>,
    UnaryRealOpConversion<TanhBIOp, math::TanhOp>,
    UnaryRealOpConversion<AsinhBIOp, math::AsinhOp>,
    UnaryRealOpConversion<AcoshBIOp, math::AcoshOp>,
    UnaryRealOpConversion<AtanhBIOp, math::AtanhOp>,
    UnaryRealOpConversion<ExpBIOp, math::ExpOp>,
    UnaryRealOpConversion<LnBIOp, math::LogOp>,
    UnaryRealOpConversion<Log10BIOp, math::Log10Op>,
    UnaryRealOpConversion<SqrtBIOp, math::SqrtOp>,
    UnaryRealOpConversion<FloorBIOp, math::FloorOp>,
    UnaryRealOpConversion<CeilBIOp, math::CeilOp>,

    // Patterns for integer math functions.
    Clog2BIOpConversion,

    // Patterns for binary real math functions.
    Atan2BIOpConversion,
    HypotBIOpConversion,

    // Patterns for real/bits conversion functions.
    RealtobitsBIOpConversion,
    BitstorealBIOpConversion,
    ShortrealtobitsBIOpConversion,
    BitstoshortrealBIOpConversion,

    // Patterns of power operations.
    PowUOpConversion, PowSOpConversion,

    // Patterns of relational operations.
    ICmpOpConversion<UltOp, ICmpPredicate::ult>,
    ICmpOpConversion<SltOp, ICmpPredicate::slt>,
    ICmpOpConversion<UleOp, ICmpPredicate::ule>,
    ICmpOpConversion<SleOp, ICmpPredicate::sle>,
    ICmpOpConversion<UgtOp, ICmpPredicate::ugt>,
    ICmpOpConversion<SgtOp, ICmpPredicate::sgt>,
    ICmpOpConversion<UgeOp, ICmpPredicate::uge>,
    ICmpOpConversion<SgeOp, ICmpPredicate::sge>,
    ICmpOpConversion<EqOp, ICmpPredicate::eq>,
    ICmpOpConversion<NeOp, ICmpPredicate::ne>,
    ICmpOpConversion<CaseEqOp, ICmpPredicate::ceq>,
    ICmpOpConversion<CaseNeOp, ICmpPredicate::cne>,
    ICmpOpConversion<WildcardEqOp, ICmpPredicate::weq>,
    ICmpOpConversion<WildcardNeOp, ICmpPredicate::wne>,
    FCmpOpConversion<NeRealOp, arith::CmpFPredicate::ONE>,
    FCmpOpConversion<FltOp, arith::CmpFPredicate::OLT>,
    FCmpOpConversion<FleOp, arith::CmpFPredicate::OLE>,
    FCmpOpConversion<FgtOp, arith::CmpFPredicate::OGT>,
    FCmpOpConversion<FgeOp, arith::CmpFPredicate::OGE>,
    FCmpOpConversion<EqRealOp, arith::CmpFPredicate::OEQ>,
    CaseXZEqOpConversion<CaseZEqOp, true>,
    CaseXZEqOpConversion<CaseXZEqOp, false>,

    // Patterns of structural operations.
    SVModuleOpConversion,
    InstanceOpConversion,
    ProcedureOpConversion,
    WaitEventOpConversion,

    // Patterns of shifting operations.
    ShrOpConversion,
    ShlOpConversion,
    AShrOpConversion,

    // Patterns of assignment operations.
    AssignOpConversion<ContinuousAssignOp>,
    AssignOpConversion<DelayedContinuousAssignOp>,
    AssignOpConversion<BlockingAssignOp>,
    AssignOpConversion<NonBlockingAssignOp>,
    AssignOpConversion<DelayedNonBlockingAssignOp>,
    AssignedVariableOpConversion,

    // Patterns of other operations outside Moore dialect.
    HWInstanceOpConversion,
    ReturnOpConversion,
    CallOpConversion,
    CallIndirectOpConversion,
    UnrealizedConversionCastConversion,
    InPlaceOpConversion<debug::ArrayOp>,
    InPlaceOpConversion<debug::StructOp>,
    InPlaceOpConversion<debug::VariableOp>,

    // Patterns of assert-like operations
    AssertLikeOpConversion<AssertOp, verif::AssertOp>,
    AssertLikeOpConversion<AssumeOp, verif::AssumeOp>,
    AssertLikeOpConversion<CoverOp, verif::CoverOp>,
    PastOpConversion,

    // Format strings.
    FormatLiteralOpConversion,
    FormatConcatOpConversion,
    FormatIntOpConversion,
    FormatClassOpConversion,
    FormatRealOpConversion,
    FormatStringOpConversion,
    FormatStringToStringOpConversion,
    DisplayBIOpConversion,

    // Patterns for file I/O operations.
    FOpenBIOpConversion,
    FWriteBIOpConversion,
    FCloseBIOpConversion,
    FGetCBIOpConversion,
    FGetSBIOpConversion,
    FEofBIOpConversion,
    FFlushBIOpConversion,
    FTellBIOpConversion,

    // Patterns for queue and dynamic array operations.
    QueueMaxOpConversion,
    QueueMinOpConversion,
    QueueUniqueOpConversion,
    QueueUniqueIndexOpConversion,
    QueueReduceOpConversion,
    QueueSortOpConversion,
    QueueRSortOpConversion,
    QueueShuffleOpConversion,
    QueueConcatOpConversion,
    QueueSliceOpConversion,
    QueueDeleteOpConversion,
    QueuePushBackOpConversion,
    QueuePushFrontOpConversion,
    QueuePopBackOpConversion,
    QueuePopFrontOpConversion,
    StreamConcatOpConversion,
    StreamUnpackOpConversion,
    DynArrayNewOpConversion,
    ArraySizeOpConversion,
    AssocArrayDeleteOpConversion,
    AssocArrayDeleteKeyOpConversion,
    AssocArrayExistsOpConversion,

    // Patterns for string operations.
    StringLenOpConversion,
    StringToUpperOpConversion,
    StringToLowerOpConversion,
    StringGetCOpConversion,
    StringPutCOpConversion,
    StringSubstrOpConversion,
    StringItoaOpConversion,
    StringAtoIOpConversion,
    StringAtoHexOpConversion,
    StringAtoOctOpConversion,
    StringAtoBinOpConversion,
    StringConcatOpConversion,
    StringReplicateOpConversion,
    StringCmpOpConversion,
    UArrayCmpOpConversion,
    IntToStringOpConversion,
    StringToIntOpConversion,
    SScanfBIOpConversion,
    IsUnknownBIOpConversion,
    CountOnesBIOpConversion,
    OneHotBIOpConversion,
    OneHot0BIOpConversion,
    CountBitsBIOpConversion
  >(typeConverter, patterns.getContext());
  // clang-format on

  // Array locator operations (need class cache for field access support).
  patterns.add<ArrayLocatorOpConversion>(typeConverter, patterns.getContext(),
                                         classCache);

  // Associative array iterator operations (need explicit constructor).
  patterns.add<AssocArrayFirstOpConversion>(typeConverter, patterns.getContext());
  patterns.add<AssocArrayNextOpConversion>(typeConverter, patterns.getContext());
  patterns.add<AssocArrayLastOpConversion>(typeConverter, patterns.getContext());
  patterns.add<AssocArrayPrevOpConversion>(typeConverter, patterns.getContext());

  // Structural operations
  patterns.add<WaitDelayOp>(convert);
  patterns.add<UnreachableOp>(convert);
  patterns.add<EventTriggeredOpConversion>(typeConverter, patterns.getContext());
  patterns.add<EventTriggerOpConversion>(typeConverter, patterns.getContext());
  patterns.add<WaitConditionOpConversion>(typeConverter, patterns.getContext());

  // Process control (fork/join)
  patterns.add<ForkOpConversion>(typeConverter, patterns.getContext());
  patterns.add<NamedBlockOpConversion>(typeConverter, patterns.getContext());
  patterns.add<WaitForkOp>(convert);
  patterns.add<DisableForkOp>(convert);
  patterns.add<DisableOp>(convert);

  // Simulation control
  patterns.add<StopBIOp>(convert);
  patterns.add<SeverityBIOp>(convert);
  patterns.add<FinishBIOp>(convert);
  patterns.add<FinishMessageBIOp>(convert);

  // Timing control
  patterns.add<TimeBIOp>(convert);
  patterns.add<LogicToTimeOp>(convert);
  patterns.add<TimeToLogicOp>(convert);

  // Random number generation
  patterns.add<UrandomBIOpConversion>(typeConverter, patterns.getContext());
  patterns.add<UrandomRangeBIOpConversion>(typeConverter, patterns.getContext());
  patterns.add<RandomBIOpConversion>(typeConverter, patterns.getContext());

  // Randomization (needs class cache for struct info)
  patterns.add<RandomizeOpConversion>(typeConverter, patterns.getContext(),
                                      classCache);
  patterns.add<StdRandomizeOpConversion>(typeConverter, patterns.getContext());

  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                            typeConverter);
  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
  populateSCFToControlFlowConversionPatterns(patterns);
  populateArithToCombPatterns(patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass
    : public circt::impl::ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();
  ClassTypeCache classCache;
  InterfaceTypeCache interfaceCache;

  IRRewriter rewriter(module);
  (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());

  TypeConverter typeConverter;
  populateTypeConversion(typeConverter);

  ConversionTarget target(context);
  populateLegality(target, typeConverter);

  ConversionPatternSet patterns(&context, typeConverter);
  populateOpConversion(patterns, typeConverter, classCache, interfaceCache);
  mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                           patterns, target);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
