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
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include <queue>

namespace circt {
#define GEN_PASS_DEF_CONVERTMOORETOCORE
#define GEN_PASS_DEF_INITVTABLES
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

/// Helper function to convert an SSA value from hw dialect types (hw::StructType,
/// hw::ArrayType) to equivalent LLVM types. This is needed when storing values
/// to LLVM alloca, since llvm.store requires LLVM-typed operands.
/// Returns the original value if no conversion is needed.
static Value convertValueToLLVMType(Value value, Location loc,
                                    OpBuilder &builder) {
  Type origType = value.getType();
  Type llvmType = convertToLLVMType(origType);

  // If the type didn't change, return the original value
  if (origType == llvmType)
    return value;

  // Handle hw::StructType -> LLVM::LLVMStructType
  if (auto hwStructTy = dyn_cast<hw::StructType>(origType)) {
    auto llvmStructTy = cast<LLVM::LLVMStructType>(llvmType);

    // Create an undef of the target LLVM struct type
    Value result = LLVM::UndefOp::create(builder, loc, llvmStructTy);

    // Extract each field from the hw struct and insert into LLVM struct
    auto fields = hwStructTy.getElements();
    for (size_t i = 0; i < fields.size(); ++i) {
      // Extract the field from the hw struct
      Value fieldVal = hw::StructExtractOp::create(builder, loc, value,
                                                   fields[i].name);
      // Recursively convert if the field is also a composite type
      fieldVal = convertValueToLLVMType(fieldVal, loc, builder);
      // Insert into the LLVM struct
      result = LLVM::InsertValueOp::create(builder, loc, result, fieldVal,
                                           ArrayRef<int64_t>{(int64_t)i});
    }
    return result;
  }

  // Handle hw::ArrayType -> LLVM::LLVMArrayType
  if (auto hwArrayTy = dyn_cast<hw::ArrayType>(origType)) {
    auto llvmArrayTy = cast<LLVM::LLVMArrayType>(llvmType);

    // Create an undef of the target LLVM array type
    Value result = LLVM::UndefOp::create(builder, loc, llvmArrayTy);

    // Extract each element from the hw array and insert into LLVM array
    int64_t numElements = hwArrayTy.getNumElements();
    // hw.array_get requires index width to be ceil(log2(numElements)), or 1 if
    // numElements <= 1
    unsigned idxWidth = llvm::Log2_64_Ceil(numElements);
    if (idxWidth == 0)
      idxWidth = 1;
    for (int64_t i = 0; i < numElements; ++i) {
      // Create constant index with correct width for hw.array_get
      Value idx = hw::ConstantOp::create(builder, loc,
                                         APInt(idxWidth, numElements - 1 - i));
      // Extract the element from the hw array (hw.array uses reverse indexing)
      Value elemVal = hw::ArrayGetOp::create(builder, loc, value, idx);
      // Recursively convert if the element is also a composite type
      elemVal = convertValueToLLVMType(elemVal, loc, builder);
      // Insert into the LLVM array
      result = LLVM::InsertValueOp::create(builder, loc, result, elemVal,
                                           ArrayRef<int64_t>{i});
    }
    return result;
  }

  // For other types that don't need deep conversion, return as-is
  // (integers, pointers, etc. are already LLVM-compatible)
  return value;
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

    // Vtable pointer field index (1 for root classes - after typeId).
    // The vtable pointer is stored right after the type ID in root classes.
    // Derived classes access it through their base class prefix at [0, 1].
    static constexpr unsigned vtablePtrFieldIndex = 1;

    // Method name to vtable index mapping for this class.
    // This maps virtual method names to their index in the vtable array.
    DenseMap<StringRef, unsigned> methodToVtableIndex;

    // Name of the global vtable symbol for this class (e.g., "@MyClass::vtable")
    std::string vtableGlobalName;
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

/// Create a struct type for 4-state values: {value: iN, unknown: iN}
/// where unknown[i]=1 means bit i is X or Z, and when unknown[i]=1,
/// value[i]=0 means X, value[i]=1 means Z.
static hw::StructType getFourStateStructType(MLIRContext *ctx, unsigned width) {
  auto intType = IntegerType::get(ctx, width);
  return hw::StructType::get(ctx, {{StringAttr::get(ctx, "value"), intType},
                                   {StringAttr::get(ctx, "unknown"), intType}});
}

/// Check if a type is a 4-state struct type (has value and unknown fields).
static bool isFourStateStructType(Type type) {
  auto structType = dyn_cast<hw::StructType>(type);
  if (!structType)
    return false;
  auto fields = structType.getElements();
  if (fields.size() != 2)
    return false;
  return fields[0].name.getValue() == "value" &&
         fields[1].name.getValue() == "unknown";
}

/// Extract the value component from a 4-state struct.
static Value extractFourStateValue(OpBuilder &builder, Location loc,
                                   Value fourState) {
  return hw::StructExtractOp::create(builder, loc, fourState, "value");
}

/// Extract the unknown mask from a 4-state struct.
static Value extractFourStateUnknown(OpBuilder &builder, Location loc,
                                     Value fourState) {
  return hw::StructExtractOp::create(builder, loc, fourState, "unknown");
}

/// Create a 4-state struct from value and unknown components.
static Value createFourStateStruct(OpBuilder &builder, Location loc,
                                   Value value, Value unknown) {
  auto structType = getFourStateStructType(
      builder.getContext(), value.getType().getIntOrFloatBitWidth());
  return hw::StructCreateOp::create(builder, loc, structType,
                                    ValueRange{value, unknown});
}

/// Convert a Moore type to its packed representation (plain integers for
/// 4-state types). This is used for packed unions where all members share
/// the same bit storage and we cannot expand 4-state types.
static Type convertTypeToPacked(const TypeConverter &typeConverter,
                                Type mooreType) {
  MLIRContext *ctx = mooreType.getContext();

  // For IntType, always return plain integer regardless of domain
  if (auto intType = dyn_cast<IntType>(mooreType)) {
    return IntegerType::get(ctx, intType.getWidth());
  }

  // For struct types, recursively convert members
  if (auto structType = dyn_cast<StructType>(mooreType)) {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto member : structType.getMembers()) {
      Type convertedType = convertTypeToPacked(typeConverter, member.type);
      if (!convertedType)
        return {};
      fields.push_back({member.name, convertedType});
    }
    return hw::StructType::get(ctx, fields);
  }

  // For array types, convert element type
  if (auto arrayType = dyn_cast<ArrayType>(mooreType)) {
    Type elemType = convertTypeToPacked(typeConverter, arrayType.getElementType());
    if (!elemType)
      return {};
    return hw::ArrayType::get(elemType, arrayType.getSize());
  }

  // For other types, use the standard converter
  return typeConverter.convertType(mooreType);
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
    // Root class: add type ID field and vtable pointer as the first members.
    // This field stores the runtime type ID for RTTI support ($cast).
    // Layout: { i32 typeId, ptr vtablePtr, ... properties ... }
    auto i32Ty = IntegerType::get(ctx, 32);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    structBodyMembers.push_back(i32Ty);   // typeId at index 0
    structBodyMembers.push_back(ptrTy);   // vtablePtr at index 1
    derivedStartIdx = 2; // Properties start after typeId and vtablePtr fields

    // Root class has inheritance depth 0
    structBody.inheritanceDepth = 0;
  }

  // Assign a unique type ID to this class
  structBody.typeId = cache.allocateTypeId();

  // Set the vtable global name for this class
  structBody.vtableGlobalName =
      classSym.getRootReference().str() + "::__vtable__";

  // Collect virtual method indices for the vtable.
  // The vtable contains function pointers for all virtual methods,
  // indexed by method name. We need to establish consistent indices
  // across the inheritance hierarchy.
  unsigned vtableIdx = 0;
  auto &block = op.getBody().front();

  // If this class has a base, inherit its method indices
  if (auto baseAttr = op.getBaseAttr()) {
    auto baseClassStruct = cache.getStructInfo(baseAttr);
    if (baseClassStruct) {
      for (const auto &kv : baseClassStruct->methodToVtableIndex) {
        structBody.methodToVtableIndex[kv.first] = kv.second;
        if (kv.second >= vtableIdx)
          vtableIdx = kv.second + 1;
      }
    }
  }

  // Add new virtual methods defined in this class
  for (Operation &child : block) {
    if (auto methodDecl = dyn_cast<ClassMethodDeclOp>(child)) {
      StringRef methodName = methodDecl.getSymName();
      // Only add if not already inherited from base
      if (structBody.methodToVtableIndex.find(methodName) ==
          structBody.methodToVtableIndex.end()) {
        structBody.methodToVtableIndex[methodName] = vtableIdx++;
      }
    }
  }

  // Properties in source order.
  unsigned iterator = derivedStartIdx;
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
  auto *symbol = mod.lookupSymbol(op);
  if (!symbol)
    return failure();
  auto classDeclOp = dyn_cast<ClassDeclOp>(*symbol);
  if (!classDeclOp)
    return failure();
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
                               SmallVector<Value> &observeValues,
                               const SmallDenseSet<Value> *ignoreValues) {
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
          Value rawValue = value;
          if (isa<BlockArgument>(value))
            value = rewriter.getRemappedValue(value);

          if (ignoreValues) {
            if (ignoreValues->contains(rawValue) ||
                (value && ignoreValues->contains(value)))
              continue;
          }

          if (region->isAncestor(value.getParentRegion()))
            continue;
          if (auto *defOp = value.getDefiningOp();
              defOp && defOp->hasTrait<OpTrait::ConstantLike>())
            continue;
          if (!alreadyObserved.insert(value).second)
            continue;

          OpBuilder::InsertionGuard g(rewriter);
          Value observeValue;
          if (auto remapped = rewriter.getRemappedValue(value)) {
            setInsertionPoint(remapped);
            observeValue = probeIfSignal(remapped);
          } else {
            setInsertionPoint(value);
            auto type = typeConverter->convertType(value.getType());
            auto converted = typeConverter->materializeTargetConversion(
                rewriter, loc, type, value);
            observeValue = probeIfSignal(converted);
          }
          if (isFourStateStructType(observeValue.getType())) {
            auto structType = cast<hw::StructType>(observeValue.getType());
            auto valueType = structType.getElements()[0].type;
            if (valueType.isInteger(1)) {
              Value valueField =
                  extractFourStateValue(rewriter, loc, observeValue);
              Value unknownField =
                  extractFourStateUnknown(rewriter, loc, observeValue);
              Value trueConst =
                  hw::ConstantOp::create(rewriter, loc, valueType, 1);
              Value notUnknown =
                  comb::XorOp::create(rewriter, loc, unknownField, trueConst);
              observeValue = comb::AndOp::create(
                  rewriter, loc, ValueRange{valueField, notUnknown}, true);
            }
          }
          if (hw::isHWValueType(observeValue.getType()))
            observeValues.push_back(observeValue);
        }
      });
}

static void collectAssignedValues(Region &region,
                                  SmallDenseSet<Value> &assignedValues) {
  region.walk([&](Operation *op) {
    if (auto assign = dyn_cast<ContinuousAssignOp>(op))
      assignedValues.insert(assign.getDst());
    else if (auto assign = dyn_cast<DelayedContinuousAssignOp>(op))
      assignedValues.insert(assign.getDst());
    else if (auto assign = dyn_cast<BlockingAssignOp>(op))
      assignedValues.insert(assign.getDst());
    else if (auto assign = dyn_cast<NonBlockingAssignOp>(op))
      assignedValues.insert(assign.getDst());
    else if (auto assign = dyn_cast<DelayedNonBlockingAssignOp>(op))
      assignedValues.insert(assign.getDst());
  });
}

/// Check if any function called from within the given region (transitively)
/// has multiple basic blocks. This is used to determine if we can lower an
/// initial block to seq.initial (single-block only) or need llhd.process
/// (supports multiple blocks for inlined control flow).
static bool hasMultiBlockFunctionCalls(Region &region, ModuleOp moduleOp) {
  llvm::SmallDenseSet<StringRef> visited;
  llvm::SmallVector<StringRef> worklist;
  bool hasIndirectCall = false;

  // Collect initial function calls from the region
  region.walk([&](func::CallOp callOp) {
    worklist.push_back(callOp.getCallee());
  });

  // Check for indirect calls (e.g., virtual method calls via call_indirect).
  // These cannot be statically analyzed, so we conservatively assume they
  // may call multi-block functions.
  region.walk([&](func::CallIndirectOp) { hasIndirectCall = true; });
  if (hasIndirectCall)
    return true;

  while (!worklist.empty()) {
    StringRef callee = worklist.pop_back_val();
    if (!visited.insert(callee).second)
      continue;

    // Look up the function in the module
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
    if (!funcOp || funcOp.empty())
      continue;

    // If this function has multiple blocks, we need llhd.process
    if (!funcOp.getBody().hasOneBlock())
      return true;

    // Check for transitive calls (both direct and indirect)
    funcOp.getBody().walk([&](func::CallOp nestedCall) {
      worklist.push_back(nestedCall.getCallee());
    });
    funcOp.getBody().walk([&](func::CallIndirectOp) { hasIndirectCall = true; });
    if (hasIndirectCall)
      return true;
  }

  return false;
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
      SmallDenseSet<Value> assignedValues;
      collectAssignedValues(op.getBody(), assignedValues);
      auto setInsertionPoint = [&](Value value) {
        rewriter.setInsertionPoint(op);
      };
      getValuesToObserve(&op.getBody(), setInsertionPoint, typeConverter,
                         rewriter, observedValues, &assignedValues);
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
      // Note: We can't use getUsedValuesDefinedAbove() here because during
      // dialect conversion, block arguments that are being converted may have
      // their parent region set to null, causing the check to miss them.
      // Instead, we manually check for:
      // 1. Block arguments (which are never constant-like)
      // 2. Non-constant operations defined outside the procedure body
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

      // During dialect conversion, block arguments may have null parent regions,
      // so getUsedValuesDefinedAbove misses them. Check for BlockArguments
      // explicitly - if any operand is a BlockArgument from outside the
      // procedure body, we can't use seq.initial.
      if (allCapturesConstant) {
        for (auto &block : op.getBody()) {
          for (auto &bodyOp : block) {
            for (Value operand : bodyOp.getOperands()) {
              if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                // Check if this block argument belongs to a block in the
                // procedure body
                Block *ownerBlock = blockArg.getOwner();
                if (!ownerBlock || !ownerBlock->getParent()) {
                  // Null owner or parent region - happens during conversion
                  allCapturesConstant = false;
                  break;
                }
                if (ownerBlock->getParent() != &op.getBody()) {
                  allCapturesConstant = false;
                  break;
                }
              }
            }
            if (!allCapturesConstant)
              break;
          }
          if (!allCapturesConstant)
            break;
        }
      }

      // Simple initial blocks can use seq.initial for arcilator support.
      // This includes blocks with $finish (unreachable ops) which get
      // converted to sim.terminate.
      // We also need to check if any called functions have multiple blocks,
      // because seq.initial only supports single-block regions and the
      // InlineCallsPass cannot inline multi-block functions into it.
      bool hasSingleBlock = op.getBody().hasOneBlock();
      auto moduleOp = op->getParentOfType<ModuleOp>();
      bool hasMultiBlockCalls = moduleOp &&
          hasMultiBlockFunctionCalls(op.getBody(), moduleOp);

      // Track E: Force llhd.process for hvl_top modules to preserve UVM
      // runtime calls. The seq.initial lowering inlines function bodies,
      // which breaks UVM run_test() that needs to call runtime functions.
      // By detecting "hvl" in the module name, we force these modules to
      // use llhd.process which preserves func.call operations.
      // Note: By the time ProcedureOp is converted, the parent SVModuleOp
      // has already been converted to hw::HWModuleOp, so we check that.
      bool isHvlModule = false;
      if (auto parentModule = op->getParentOfType<hw::HWModuleOp>()) {
        StringRef moduleName = parentModule.getSymName();
        isHvlModule = moduleName.contains_insensitive("hvl");
      }

      if (!hasWaitEvent && !hasWaitDelay && allCapturesConstant &&
          hasSingleBlock && !hasMultiBlockCalls && !isHvlModule) {
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
      rewriter.setInsertionPoint(op);
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
      rewriter.setInsertionPoint(op);
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
    rewriter.setInsertionPoint(op);
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
                       typeConverter, rewriter, observeValues, nullptr);

    // Create the `llhd.wait` op that suspends the current process and waits for
    // a change in the interesting values listed in `observeValues`. When a
    // change is detected, execution resumes in the "check" block.
    auto waitOp = llhd::WaitOp::create(rewriter, loc, ValueRange{}, Value(),
                                       observeValues, ValueRange{}, checkBlock);
    rewriter.inlineBlockBefore(&clonedOp.getBody().front(), waitOp);
    rewriter.eraseOp(clonedOp);
    auto convertToBoolIfFourState = [&](Value value) -> Value {
      if (auto mooreInt = dyn_cast<IntType>(value.getType())) {
        if (mooreInt.getBitSize() == 1)
          return moore::ToBuiltinBoolOp::create(rewriter, loc, value);
        return value;
      }
      if (!isFourStateStructType(value.getType()))
        return value;
      auto structType = cast<hw::StructType>(value.getType());
      auto valueType = structType.getElements()[0].type;
      if (!valueType.isInteger(1))
        return value;
      Value valueField = extractFourStateValue(rewriter, loc, value);
      Value unknownField = extractFourStateUnknown(rewriter, loc, value);
      Value trueConst = hw::ConstantOp::create(rewriter, loc, valueType, 1);
      Value notUnknown =
          comb::XorOp::create(rewriter, loc, unknownField, trueConst);
      return comb::AndOp::create(rewriter, loc,
                                 ValueRange{valueField, notUnknown}, true);
    };
    for (auto &value : valuesBefore) {
      OpBuilder::InsertionGuard g(rewriter);
      setInsertionPointAfterDef(value);
      value = convertToBoolIfFourState(value);
    }

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

      if (auto mooreInt = dyn_cast<IntType>(before.getType())) {
        // 9.4.2 IEEE 1800-2017: An edge event shall be detected only on the LSB
        // of the expression
        if (mooreInt.getWidth() != 1 && edge != Edge::AnyChange) {
          constexpr int LSB = 0;
          mooreInt =
              IntType::get(rewriter.getContext(), 1, mooreInt.getDomain());
          before =
              moore::ExtractOp::create(rewriter, loc, mooreInt, before, LSB);
          after =
              moore::ExtractOp::create(rewriter, loc, mooreInt, after, LSB);
        }

        auto intType = rewriter.getIntegerType(mooreInt.getWidth());
        before = typeConverter->materializeTargetConversion(rewriter, loc,
                                                            intType, before);
        after = typeConverter->materializeTargetConversion(
            rewriter, loc, intType, after);
      } else if (auto intType = dyn_cast<IntegerType>(before.getType())) {
        if (intType.getWidth() != 1 && edge != Edge::AnyChange) {
          constexpr int LSB = 0;
          before =
              comb::ExtractOp::create(rewriter, loc, before, LSB, 1).getResult();
          after =
              comb::ExtractOp::create(rewriter, loc, after, LSB, 1).getResult();
          intType = rewriter.getIntegerType(1);
        }
      } else {
        return Value();
      }

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
        if (!isa<IntType>(before.getType()) &&
            !before.getType().isSignlessInteger())
          return detectOp->emitError() << "requires int operand";

        rewriter.setInsertionPoint(detectOp);
        auto after = convertToBoolIfFourState(detectOp.getInput());
        auto trigger =
            computeTrigger(before, after, detectOp.getEdge());
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

/// Conversion for moore.wait_delay -> either llhd.wait (in llhd.process context)
/// or __moore_delay runtime call (in func.func context for class methods).
///
/// In SystemVerilog, delay statements (#delay) can appear in both:
/// 1. Module procedures (always, initial blocks) - these lower to llhd.process
/// 2. Class tasks/methods - these lower to func.func
///
/// The llhd.wait operation requires an llhd.process parent, so we need a
/// different lowering strategy for class method contexts. For class methods,
/// we call a runtime function that suspends the current coroutine/task.
struct WaitDelayOpConversion : public OpConversionPattern<WaitDelayOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitDelayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get the delay value. The adaptor gives us the converted operand.
    // Since TimeType now converts to i64, the delay should be i64.
    Value delay = adaptor.getDelay();

    // Check if we're inside a moore.procedure (which will become llhd.process)
    // or already inside an llhd.process. In these cases, use llhd.wait.
    // Note: During conversion, we check for ProcedureOp because the conversion
    // of ProcedureOp to llhd.process may happen after this conversion.
    // However, if we're inside a fork branch (moore.fork or sim.fork), we need
    // to use the runtime call instead, because fork branches become sim.fork
    // regions, not llhd.process, and llhd.wait requires llhd.process parent.
    bool insideFork = op->getParentOfType<ForkOp>() ||
                      op->getParentOfType<sim::SimForkOp>();
    if (!insideFork && (op->getParentOfType<ProcedureOp>() ||
                        op->getParentOfType<llhd::ProcessOp>())) {
      // llhd.wait expects llhd::TimeType, so we need to convert from i64.
      // Use llhd.int_to_time to convert i64 (femtoseconds) to llhd.time.
      Value llhdTime = llhd::IntToTimeOp::create(rewriter, loc, delay);
      auto *resumeBlock =
          rewriter.splitBlock(op->getBlock(), ++Block::iterator(op));
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<llhd::WaitOp>(op, ValueRange{},
                                                llhdTime, ValueRange{},
                                                ValueRange{}, resumeBlock);
      rewriter.setInsertionPointToStart(resumeBlock);
      return success();
    }

    // We're inside a func.func (class method context) - use runtime call.
    // We pass the time (i64 in femtoseconds) to the runtime.
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    // __moore_delay takes the delay time in time units as i64.
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_delay", fnTy);

    // The delay value should already be i64 (time in femtoseconds).
    // Call the runtime function.
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{delay});

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.unreachable -> llhd.halt (in process/final) or llvm.unreachable (in func)
static LogicalResult convert(UnreachableOp op, UnreachableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Check if we're inside an llhd.process or llhd.final
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<llhd::ProcessOp, llhd::FinalOp>(parent)) {
      // Inside a process or final block - use llhd.halt
      rewriter.replaceOpWithNewOp<llhd::HaltOp>(op, ValueRange{});
      return success();
    }
    if (isa<func::FuncOp, LLVM::LLVMFuncOp>(parent)) {
      // Inside a function (class method) - use llvm.unreachable
      rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
      return success();
    }
    parent = parent->getParentOp();
  }
  // Default to llhd.halt for hardware constructs
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
// UVM Configuration Database Conversion
//===----------------------------------------------------------------------===//

/// Helper to create a MooreString struct from a string attribute.
/// Creates global string constant and returns struct {ptr, i64}.
static Value createMooreStringFromAttr(Location loc, ModuleOp mod,
                                       ConversionPatternRewriter &rewriter,
                                       StringRef str, StringRef globalPrefix) {
  auto *ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i64Ty = IntegerType::get(ctx, 64);

  // Create the string struct type: {ptr, i64}
  auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

  int64_t len = str.size();

  // Create global string constant
  std::string globalName = (globalPrefix + "_" +
      std::to_string(std::hash<std::string>{}(str.str()))).str();
  auto i8Ty = IntegerType::get(ctx, 8);
  auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, len);

  // Check if global already exists
  if (!mod.lookupSymbol<LLVM::GlobalOp>(globalName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    LLVM::GlobalOp::create(rewriter, loc, arrayTy, /*isConstant=*/true,
                           LLVM::Linkage::Internal, globalName,
                           rewriter.getStringAttr(str));
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

/// Conversion for moore.uvm.config_db.set -> runtime function call.
/// Stores a value in the UVM configuration database.
struct UVMConfigDbSetOpConversion
    : public OpConversionPattern<UVMConfigDbSetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UVMConfigDbSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // __moore_config_db_set(context, instName, instLen, fieldName, fieldLen,
    //                       value, valueSize, typeId)
    auto fnTy = LLVM::LLVMFunctionType::get(
        voidTy, {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, ptrTy, i64Ty, i32Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_config_db_set", fnTy);

    // Get the context pointer (null if not provided)
    Value contextPtr;
    if (adaptor.getContext()) {
      contextPtr = adaptor.getContext();
    } else {
      contextPtr =
          LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Get inst_name and field_name from attributes
    StringRef instName = op.getInstName();
    StringRef fieldName = op.getFieldName();

    // Create global string constants for inst_name and field_name
    Value instNameStr = createMooreStringFromAttr(loc, mod, rewriter, instName,
                                                  "__config_db_inst");
    Value fieldNameStr = createMooreStringFromAttr(loc, mod, rewriter, fieldName,
                                                   "__config_db_field");

    // Extract pointer and length from string structs
    Value instNamePtr = LLVM::ExtractValueOp::create(rewriter, loc, instNameStr,
                                                     ArrayRef<int64_t>{0});
    Value instNameLen = LLVM::ExtractValueOp::create(rewriter, loc, instNameStr,
                                                     ArrayRef<int64_t>{1});
    Value fieldNamePtr = LLVM::ExtractValueOp::create(rewriter, loc, fieldNameStr,
                                                      ArrayRef<int64_t>{0});
    Value fieldNameLen = LLVM::ExtractValueOp::create(rewriter, loc, fieldNameStr,
                                                      ArrayRef<int64_t>{1});

    // Get the value to store - we need to allocate it and pass a pointer
    Value value = adaptor.getValue();
    Type valueType = value.getType();

    // Allocate space for the value and store it
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto valueAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, valueType, one);
    LLVM::StoreOp::create(rewriter, loc, value, valueAlloca);

    // Compute value size using DataLayout
    // For simplicity, we use a type ID based on a hash of the original Moore type
    Type origType = op.getValue().getType();
    int32_t typeId = static_cast<int32_t>(
        llvm::hash_value(origType.getAsOpaquePointer()));

    // Estimate value size (in bytes) - simplified heuristic
    int64_t valueSize = 8;  // Default to 8 bytes
    if (auto intTy = dyn_cast<IntegerType>(valueType)) {
      valueSize = (intTy.getWidth() + 7) / 8;
    } else if (isa<LLVM::LLVMPointerType>(valueType)) {
      valueSize = 8;  // Pointer size
    } else if (auto structTy = dyn_cast<LLVM::LLVMStructType>(valueType)) {
      // For structs, estimate based on number of elements
      valueSize = structTy.getBody().size() * 8;
    }

    Value valueSizeVal = arith::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(valueSize));
    Value typeIdVal = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(typeId));

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{contextPtr, instNamePtr, instNameLen,
                                    fieldNamePtr, fieldNameLen, valueAlloca,
                                    valueSizeVal, typeIdVal});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.uvm.config_db.get -> runtime function call.
/// Retrieves a value from the UVM configuration database.
struct UVMConfigDbGetOpConversion
    : public OpConversionPattern<UVMConfigDbGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UVMConfigDbGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // __moore_config_db_get(context, instName, instLen, fieldName, fieldLen,
    //                       typeId, outValue, valueSize) -> i32
    auto fnTy = LLVM::LLVMFunctionType::get(
        i32Ty, {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i32Ty, ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_config_db_get", fnTy);

    // Get the context pointer
    Value contextPtr = adaptor.getContext();

    // Get inst_name and field_name from attributes
    StringRef instName = op.getInstName();
    StringRef fieldName = op.getFieldName();

    // Create global string constants for inst_name and field_name
    Value instNameStr = createMooreStringFromAttr(loc, mod, rewriter, instName,
                                                  "__config_db_inst");
    Value fieldNameStr = createMooreStringFromAttr(loc, mod, rewriter, fieldName,
                                                   "__config_db_field");

    // Extract pointer and length from string structs
    Value instNamePtr = LLVM::ExtractValueOp::create(rewriter, loc, instNameStr,
                                                     ArrayRef<int64_t>{0});
    Value instNameLen = LLVM::ExtractValueOp::create(rewriter, loc, instNameStr,
                                                     ArrayRef<int64_t>{1});
    Value fieldNamePtr = LLVM::ExtractValueOp::create(rewriter, loc, fieldNameStr,
                                                      ArrayRef<int64_t>{0});
    Value fieldNameLen = LLVM::ExtractValueOp::create(rewriter, loc, fieldNameStr,
                                                      ArrayRef<int64_t>{1});

    // Determine the result type and allocate space for the output value
    Type resultType = getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType)
      return failure();

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto outValueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, resultType, one);

    // Compute type ID and value size
    Type origType = op.getValue().getType();
    int32_t typeId = static_cast<int32_t>(
        llvm::hash_value(origType.getAsOpaquePointer()));

    // Estimate value size (in bytes) - simplified heuristic
    int64_t valueSize = 8;  // Default to 8 bytes
    if (auto intTy = dyn_cast<IntegerType>(resultType)) {
      valueSize = (intTy.getWidth() + 7) / 8;
    } else if (isa<LLVM::LLVMPointerType>(resultType)) {
      valueSize = 8;  // Pointer size
    } else if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultType)) {
      valueSize = structTy.getBody().size() * 8;
    }

    Value typeIdVal = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(typeId));
    Value valueSizeVal = arith::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(valueSize));

    // Call the runtime function
    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(fn),
        ValueRange{contextPtr, instNamePtr, instNameLen, fieldNamePtr,
                   fieldNameLen, typeIdVal, outValueAlloca, valueSizeVal});

    // Convert i32 result to i1 for the 'found' result
    Value foundI32 = call.getResult();
    Value zero = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
    Value found = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        foundI32, zero);

    // Load the retrieved value from the alloca
    Value retrievedValue =
        LLVM::LoadOp::create(rewriter, loc, resultType, outValueAlloca);

    // Replace the op with the found flag and retrieved value
    rewriter.replaceOp(op, {found, retrievedValue});
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

      // If the entry block has predecessors (e.g., from forever loops),
      // we need to create a new "loop header" block and redirect back-edges.
      // Region entry blocks must not have predecessors.
      Block *entryBlock = &simRegion.front();
      if (!entryBlock->hasNoPredecessors()) {
        // Create a new loop header block after the entry block.
        // We'll redirect all back-edges from the entry block to this header.
        auto *loopHeader = new Block();
        simRegion.getBlocks().insertAfter(entryBlock->getIterator(), loopHeader);

        // Move all operations from entry to the loop header.
        // We need to keep the structure such that:
        // 1. Entry block has no predecessors
        // 2. Entry block content can be printed (not elided)
        // 3. Back-edges go to loop header, not entry
        loopHeader->getOperations().splice(loopHeader->end(),
                                           entryBlock->getOperations());

        // After the splice, the loopHeader now has the original operations,
        // including any back-edge branches that were targeting entryBlock.
        // We need to update those branches to target loopHeader instead.
        // This must happen BEFORE we add the new branch from entry to loopHeader.

        // Update all branches in the region that target entryBlock.
        // After splice, these branches are now in loopHeader (or other blocks
        // if there were multiple blocks inlined).
        for (Block &block : simRegion) {
          Operation *terminator = block.getTerminator();
          if (!terminator)
            continue;
          if (auto brOp = dyn_cast<cf::BranchOp>(terminator)) {
            if (brOp.getDest() == entryBlock) {
              OpBuilder::InsertionGuard g(rewriter);
              rewriter.setInsertionPoint(brOp);
              cf::BranchOp::create(rewriter, brOp.getLoc(), loopHeader);
              rewriter.eraseOp(brOp);
            }
          } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(terminator)) {
            bool updateTrue = condBrOp.getTrueDest() == entryBlock;
            bool updateFalse = condBrOp.getFalseDest() == entryBlock;
            if (updateTrue || updateFalse) {
              OpBuilder::InsertionGuard g(rewriter);
              rewriter.setInsertionPoint(condBrOp);
              cf::CondBranchOp::create(
                  rewriter, condBrOp.getLoc(), condBrOp.getCondition(),
                  updateTrue ? loopHeader : condBrOp.getTrueDest(),
                  condBrOp.getTrueDestOperands(),
                  updateFalse ? loopHeader : condBrOp.getFalseDest(),
                  condBrOp.getFalseDestOperands());
              rewriter.eraseOp(condBrOp);
            }
          }
        }

        // Now add a branch from entry to loop header.
        // The entry block is now empty, but we'll add an op with side effects
        // to prevent the printer from eliding it completely.
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(entryBlock);
          // Add a non-empty print with side effects to prevent block elision.
          // We use a space character which won't print visibly but won't be
          // optimized away (unlike empty strings which get canonicalized out).
          // Note: "\0" doesn't work because C string literals treat null as
          // the terminator, so StringRef("\0") has length 0.
          auto spaceFmt = sim::FormatLiteralOp::create(rewriter, loc, " ");
          sim::PrintFormattedProcOp::create(rewriter, loc, spaceFmt);
          cf::BranchOp::create(rewriter, loc, loopHeader);
        }

        // After the transformation, verify the entry has no predecessors.
        assert(entryBlock->hasNoPredecessors() &&
               "entry block should have no predecessors after restructuring");
      }

      // Ensure the entry block won't be elided by the printer.
      // If the entry block only contains a branch (or nothing), the MLIR printer
      // may elide it, causing the next block (which might have predecessors from
      // loops) to be treated as the entry during re-parsing, which fails verification.
      // We add an op with side effects to ensure the entry block is preserved.
      entryBlock = &simRegion.front();
      if (entryBlock->hasNoPredecessors()) {
        // Check if entry block might be elided: only has a branch terminator
        // or no operations at all.
        Operation *termOp = entryBlock->getTerminator();
        bool hasOnlyBranch = termOp && isa<cf::BranchOp>(termOp) &&
                             entryBlock->without_terminator().empty();
        bool mightBeElided = entryBlock->empty() || hasOnlyBranch;

        if (mightBeElided) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(entryBlock);
          // Add a non-empty print with side effects to prevent block elision.
          // We use a space character which won't print visibly but won't be
          // optimized away (unlike empty strings which get canonicalized out).
          // Note: "\0" doesn't work because C string literals treat null as
          // the terminator, so StringRef("\0") has length 0.
          auto spaceFmt = sim::FormatLiteralOp::create(rewriter, loc, " ");
          sim::PrintFormattedProcOp::create(rewriter, loc, spaceFmt);
        }
      }

      // Convert ForkTerminatorOp to SimForkTerminatorOp, or add terminator
      // if the block doesn't have one
      for (Block &block : simRegion) {
        Operation *lastOp = block.empty() ? nullptr : &block.back();
        bool hasTerminator =
            lastOp && lastOp->hasTrait<mlir::OpTrait::IsTerminator>();
        if (auto forkTerm = dyn_cast_or_null<ForkTerminatorOp>(lastOp)) {
          rewriter.setInsertionPoint(forkTerm);
          sim::SimForkTerminatorOp::create(rewriter, forkTerm.getLoc());
          rewriter.eraseOp(forkTerm);
        } else if (!hasTerminator) {
          // Block has no terminator or the last op isn't a terminator,
          // add sim.fork.terminator
          rewriter.setInsertionPointToEnd(&block);
          sim::SimForkTerminatorOp::create(rewriter, loc);
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

  // Handle llhd::TimeType (for LLHD-native code, not MooreToCore conversion).
  // Note: moore::TimeType now converts to i64, not llhd::TimeType.
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

    // Handle block argument remapping for class method contexts.
    // When a class method's signature is converted, the 'this' pointer (block
    // argument 0) gets remapped. The adaptor may return the old/invalidated
    // block argument. We need to find the new block argument at the
    // corresponding position in the converted function's entry block.
    //
    // This is critical for class member access from methods other than the
    // constructor, where operations reference the 'this' pointer which is a
    // block argument that gets remapped during function signature conversion.
    if (auto blockArg = dyn_cast<BlockArgument>(op.getInstance())) {
      // Get the parent function's entry block to find the new block argument
      auto funcOp = op->getParentOfType<func::FuncOp>();
      if (funcOp) {
        Block *funcEntryBlock = &funcOp.getBody().front();
        unsigned argIndex = blockArg.getArgNumber();
        if (argIndex < funcEntryBlock->getNumArguments()) {
          // Use the converted block argument from the function's entry block
          instRef = funcEntryBlock->getArgument(argIndex);
        }
      }
    }

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
    // First index 0 is needed to dereference the pointer to the struct.
    // Subsequent indices navigate into the struct fields.
    idxVals.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
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

    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto mallocFn = getOrCreateMalloc(mod, rewriter);

    auto allocateInterfacePtr =
        [&](LLVM::LLVMStructType ifaceStructTy) -> Value {
      uint64_t ifaceSize = getTypeSizeSafe(ifaceStructTy, mod);
      auto sizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(ifaceSize));
      auto call =
          LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                               SymbolRefAttr::get(mallocFn),
                               ValueRange{sizeConst});
      return call.getResult();
    };

    Value ifacePtr = allocateInterfacePtr(structTy);

    // Initialize nested interface instances if this interface declares any.
    if (auto *ifaceDeclSym = mod.lookupSymbol(ifaceSymRef.getRootReference())) {
      if (auto ifaceDeclOp = dyn_cast<InterfaceDeclOp>(*ifaceDeclSym)) {
        auto &body = ifaceDeclOp.getBody().front();
        for (Operation &child : body) {
          auto signal = dyn_cast<InterfaceSignalDeclOp>(child);
          if (!signal || !signal->hasAttr("interface_instance"))
            continue;

          auto nestedType =
              dyn_cast<VirtualInterfaceType>(signal.getSignalType());
          if (!nestedType)
            continue;

          auto nestedIfaceRef = nestedType.getInterface();
          auto nestedIfaceSym =
              SymbolRefAttr::get(ctx, nestedIfaceRef.getRootReference());
          if (failed(resolveInterfaceStructBody(mod, nestedIfaceSym,
                                                *typeConverter, cache)))
            return op.emitError()
                   << "Could not resolve interface struct for "
                   << nestedIfaceSym;

          auto nestedInfo = cache.getStructInfo(nestedIfaceSym);
          if (!nestedInfo)
            return op.emitError()
                   << "Missing interface struct info for "
                   << nestedIfaceSym;

          Value nestedPtr = allocateInterfacePtr(nestedInfo->interfaceBody);

          auto idxOpt = structInfo->getSignalIndex(signal.getSymName());
          if (!idxOpt)
            return op.emitError()
                   << "Missing interface field index for "
                   << signal.getSymName();

          auto i32Ty = IntegerType::get(ctx, 32);
          Value zero = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
          Value idx = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(*idxOpt));
          auto fieldPtr = LLVM::GEPOp::create(
              rewriter, loc, ptrTy, structTy, ifacePtr, ValueRange{zero, idx});

          LLVM::StoreOp::create(rewriter, loc, nestedPtr, fieldPtr);
        }
      }
    }

    // The InterfaceInstanceOp returns a ref<virtual_interface> which should be
    // an llhd.ref that holds a pointer to the interface struct. Create a signal
    // to hold this pointer so that it can be read via probe/drive operations.
    // This enables virtual interface binding (vif = interface_instance) to work
    // correctly by reading the pointer from the signal.
    auto sigRefTy = llhd::RefType::get(ptrTy);
    auto sigOp = llhd::SignalOp::create(rewriter, loc, sigRefTy, StringAttr{},
                                        ifacePtr);

    // Replace the instance op with the signal reference.
    rewriter.replaceOp(op, sigOp.getResult());
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

    // Initialize the vtable pointer field for virtual method dispatch.
    // The vtable pointer is at index 1 in the root class (after typeId).
    // For a class hierarchy A -> B -> C, the layout is:
    //   C: { B: { A: { i32 typeId, ptr vtablePtr, ...A fields }, ...B fields }, ...C fields }
    // So we GEP through indices [0, 0, ...0, 1] to reach the vtablePtr.
    // The first 0 is the pointer dereference, subsequent 0s navigate through
    // the base class chain, and the final index accesses the vtable pointer.
    SmallVector<Value> vtableGepIndices;
    // First index is always 0 (pointer dereference)
    vtableGepIndices.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    // Navigate through base class chain
    for (int32_t i = 0; i < structInfo->inheritanceDepth; ++i) {
      vtableGepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    }
    // Index into the vtable pointer field (index 1 in root class)
    vtableGepIndices.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(ClassTypeCache::ClassStructInfo::vtablePtrFieldIndex)));

    // GEP to the vtable pointer field
    auto vtablePtrPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, objPtr, vtableGepIndices);

    // Ensure the vtable global exists before referencing it.
    // VTableOpConversion creates the actual vtable content, but it may run
    // after ClassNewOpConversion. Create a placeholder global if needed.
    std::string vtableGlobalName = structInfo->vtableGlobalName;
    if (!mod.lookupSymbol<LLVM::GlobalOp>(vtableGlobalName)) {
      // Determine vtable size from methodToVtableIndex
      unsigned vtableSize = 0;
      for (const auto &kv : structInfo->methodToVtableIndex) {
        if (kv.second >= vtableSize)
          vtableSize = kv.second + 1;
      }
      // Create at least a single-element array (min size 1)
      if (vtableSize == 0)
        vtableSize = 1;
      auto vtableArrayTy = LLVM::LLVMArrayType::get(ptrTy, vtableSize);
      auto zeroAttr = LLVM::ZeroAttr::get(ctx);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      LLVM::GlobalOp::create(rewriter, loc, vtableArrayTy, /*isConstant=*/false,
                             LLVM::Linkage::Internal, vtableGlobalName, zeroAttr);
    }

    // Get the address of the class's vtable global and store it
    auto vtableGlobalAddr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, vtableGlobalName);
    LLVM::StoreOp::create(rewriter, loc, vtableGlobalAddr, vtablePtrPtr);

    // Initialize queue-type properties to {nullptr, 0}.
    // Malloc returns uninitialized memory, so we need to zero-initialize
    // queue members to avoid reading garbage values.
    // Use the cached field paths from structInfo to get correct GEP indices.
    auto *classDeclOp = mod.lookupSymbol(sym);
    if (auto classDecl = dyn_cast_or_null<ClassDeclOp>(classDeclOp)) {
      // Walk through class hierarchy to initialize all queue members
      std::function<void(ClassDeclOp)> initQueueMembers =
          [&](ClassDeclOp decl) {
            // First, process base class if any
            if (auto baseAttr = decl.getBaseAttr()) {
              if (auto *baseOp = mod.lookupSymbol(baseAttr)) {
                if (auto baseDecl = dyn_cast<ClassDeclOp>(baseOp)) {
                  initQueueMembers(baseDecl);
                }
              }
            }

            // Process properties at this level
            for (auto &childOp : decl.getBody().getOps()) {
              if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(childOp)) {
                Type propType = propDecl.getPropertyType();
                if (isa<QueueType>(propType)) {
                  // Look up the cached GEP path for this property.
                  // This handles inheritance correctly - the path already
                  // accounts for the full class hierarchy.
                  auto pathOpt = structInfo->getFieldPath(propDecl.getSymName());
                  if (!pathOpt)
                    continue; // Skip if path not found (shouldn't happen)

                  // Build GEP indices from the cached path
                  SmallVector<Value> propGepIndices;
                  // First index is always 0 (pointer dereference)
                  propGepIndices.push_back(LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
                  // Add all indices from the cached path
                  for (unsigned idx : *pathOpt) {
                    propGepIndices.push_back(LLVM::ConstantOp::create(
                        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(idx)));
                  }

                  // GEP to the queue field
                  auto propPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                                     structTy, objPtr,
                                                     propGepIndices);

                  // Create the zero-initialized queue struct {nullptr, 0}
                  auto queueStructTy = LLVM::LLVMStructType::getLiteral(
                      ctx, {ptrTy, i64Ty});
                  Value zeroQueue =
                      LLVM::UndefOp::create(rewriter, loc, queueStructTy);
                  Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
                  Value zeroLen = LLVM::ConstantOp::create(
                      rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
                  zeroQueue = LLVM::InsertValueOp::create(rewriter, loc,
                                                          zeroQueue, nullPtr,
                                                          ArrayRef<int64_t>{0});
                  zeroQueue = LLVM::InsertValueOp::create(rewriter, loc,
                                                          zeroQueue, zeroLen,
                                                          ArrayRef<int64_t>{1});

                  // Store the zero-initialized queue
                  LLVM::StoreOp::create(rewriter, loc, zeroQueue, propPtr);
                }

                // Initialize associative array properties by calling __moore_assoc_create
                if (auto assocType = dyn_cast<AssocArrayType>(propType)) {
                  auto pathOpt = structInfo->getFieldPath(propDecl.getSymName());
                  if (!pathOpt)
                    continue;

                  // Build GEP indices from the cached path
                  SmallVector<Value> propGepIndices;
                  propGepIndices.push_back(LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
                  for (unsigned idx : *pathOpt) {
                    propGepIndices.push_back(LLVM::ConstantOp::create(
                        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(idx)));
                  }

                  // GEP to the associative array field
                  auto propPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                                     structTy, objPtr,
                                                     propGepIndices);

                  // Determine key size (0 for string keys)
                  int32_t keySize = 0;
                  auto keyType = assocType.getIndexType();
                  if (!isa<StringType>(keyType)) {
                    auto convertedKeyType = typeConverter->convertType(keyType);
                    if (auto intTy = dyn_cast<IntegerType>(convertedKeyType))
                      keySize = intTy.getWidth() / 8;
                    else if (isa<LLVM::LLVMPointerType>(convertedKeyType))
                      keySize = 8; // Pointer size
                    else
                      keySize = 4; // Default
                  }

                  // Determine value size
                  auto valueType = assocType.getElementType();
                  auto convertedValueType = typeConverter->convertType(valueType);
                  int32_t valueSize = 4; // Default
                  if (auto intTy = dyn_cast<IntegerType>(convertedValueType))
                    valueSize = (intTy.getWidth() + 7) / 8;
                  else if (isa<LLVM::LLVMPointerType>(convertedValueType))
                    valueSize = 8; // Pointer size for class handles

                  // Get or create __moore_assoc_create function
                  auto assocFnTy = LLVM::LLVMFunctionType::get(ptrTy, {i32Ty, i32Ty});
                  auto assocFn =
                      getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_create", assocFnTy);

                  // Create constants for key and value sizes
                  auto keySizeConst = LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(keySize));
                  auto valueSizeConst = LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

                  // Call __moore_assoc_create(key_size, value_size)
                  auto assocCall = LLVM::CallOp::create(
                      rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(assocFn),
                      ValueRange{keySizeConst, valueSizeConst});

                  // Store the created associative array handle in the property field
                  LLVM::StoreOp::create(rewriter, loc, assocCall.getResult(), propPtr);
                }
              }
            }
          };
      initQueueMembers(classDecl);
    }

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

/// moore.class.copy lowering: create a shallow copy of a class instance.
/// This allocates new memory and copies all bytes from the source object.
struct ClassCopyOpConversion : public OpConversionPattern<ClassCopyOp> {
  ClassCopyOpConversion(TypeConverter &tc, MLIRContext *ctx,
                        ClassTypeCache &cache)
      : OpConversionPattern<ClassCopyOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassCopyOp op, OpAdaptor adaptor,
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

    // Compute struct size.
    uint64_t byteSize = getTypeSizeSafe(structTy, mod);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto cSize = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(byteSize));

    // Get or declare malloc and call it to allocate memory for the new object.
    auto mallocFn = getOrCreateMalloc(mod, rewriter);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto mallocCall =
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                             SymbolRefAttr::get(mallocFn), ValueRange{cSize});

    Value destPtr = mallocCall.getResult();
    Value srcPtr = adaptor.getSource();

    // Copy all bytes from source to destination using memcpy.
    // This performs a shallow copy - nested class handles (pointers) are
    // copied as-is, so both original and copy reference the same nested objects.
    LLVM::MemcpyOp::create(rewriter, loc, destPtr, srcPtr, cSize,
                           /*isVolatile=*/false);

    // Replace the copy op with the new object pointer.
    rewriter.replaceOp(op, destPtr);
    return success();
  }

private:
  ClassTypeCache &cache;
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

        // Get or create illegal/ignore bin functions
        // __moore_coverpoint_add_illegal_bin(cg, cp_index, name, low, high)
        auto i64Ty = IntegerType::get(ctx, 64);
        auto addIllegalBinFnTy =
            LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, ptrTy, i64Ty, i64Ty});
        auto addIllegalBinFn = getOrCreateRuntimeFunc(
            mod, rewriter, "__moore_coverpoint_add_illegal_bin", addIllegalBinFnTy);

        // __moore_coverpoint_add_ignore_bin(cg, cp_index, name, low, high)
        auto addIgnoreBinFnTy =
            LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, ptrTy, i64Ty, i64Ty});
        auto addIgnoreBinFn = getOrCreateRuntimeFunc(
            mod, rewriter, "__moore_coverpoint_add_ignore_bin", addIgnoreBinFnTy);

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

        // Initialize each coverpoint and its illegal/ignore bins
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

          // Process illegal_bins and ignore_bins for this coverpoint
          for (auto &binOp : cp.getBody().front()) {
            auto bin = dyn_cast<CoverageBinDeclOp>(&binOp);
            if (!bin)
              continue;

            auto kind = bin.getKind();
            if (kind != CoverageBinKind::IllegalBins &&
                kind != CoverageBinKind::IgnoreBins)
              continue;

            // Get the bin name
            StringRef binName = bin.getSymName();
            std::string binNameGlobal =
                ("__bin_name_" + cgName + "_" + cpName + "_" + binName).str();
            Value binNamePtr = createGlobalStringConstant(loc, mod, rewriter,
                                                          binName, binNameGlobal);

            // Get the values array
            auto valuesAttr = bin.getValues();
            if (!valuesAttr)
              continue;

            // Process each value/range in the bin
            for (auto valAttr : *valuesAttr) {
              int64_t low = 0, high = 0;
              if (auto intAttr = dyn_cast<IntegerAttr>(valAttr)) {
                // Single value: low = high = value
                low = high = intAttr.getInt();
              } else if (auto arrAttr = dyn_cast<ArrayAttr>(valAttr)) {
                // Range: [low, high]
                if (arrAttr.size() >= 2) {
                  if (auto lowAttr = dyn_cast<IntegerAttr>(arrAttr[0]))
                    low = lowAttr.getInt();
                  if (auto highAttr = dyn_cast<IntegerAttr>(arrAttr[1]))
                    high = highAttr.getInt();
                }
              }

              auto lowConst = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(low));
              auto highConst = LLVM::ConstantOp::create(
                  rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(high));

              if (kind == CoverageBinKind::IllegalBins) {
                LLVM::CallOp::create(rewriter, loc, TypeRange{},
                                     SymbolRefAttr::get(addIllegalBinFn),
                                     ValueRange{cgHandle, idxConst, binNamePtr,
                                                lowConst, highConst});
              } else {
                LLVM::CallOp::create(rewriter, loc, TypeRange{},
                                     SymbolRefAttr::get(addIgnoreBinFn),
                                     ValueRange{cgHandle, idxConst, binNamePtr,
                                                lowConst, highConst});
              }
            }
          }

          ++cpIndex;
        }

        // Build a map from coverpoint symbol name to index for cross coverage
        DenseMap<StringAttr, int32_t> cpNameToIndex;
        cpIndex = 0;
        for (auto cp : coverpoints) {
          cpNameToIndex[cp.getSymNameAttr()] = cpIndex++;
        }

        // Collect cross coverage declarations from the covergroup body
        SmallVector<CoverCrossDeclOp> crosses;
        for (auto &bodyOp : op.getBody().front()) {
          if (auto cross = dyn_cast<CoverCrossDeclOp>(&bodyOp))
            crosses.push_back(cross);
        }

        // Initialize cross coverage items
        if (!crosses.empty()) {
          // Get or create __moore_cross_create function
          // int32_t __moore_cross_create(void *cg, const char *name,
          //                              int32_t *cp_indices, int32_t num_cps)
          auto crossCreateFnTy = LLVM::LLVMFunctionType::get(
              i32Ty, {ptrTy, ptrTy, ptrTy, i32Ty});
          auto crossCreateFn = getOrCreateRuntimeFunc(
              mod, rewriter, "__moore_cross_create", crossCreateFnTy);

          // Get or create __moore_cross_add_named_bin function
          // int32_t __moore_cross_add_named_bin(void *cg, int32_t cross_index,
          //     const char *name, int32_t kind,
          //     MooreCrossBinsofFilter *filters, int32_t num_filters)
          auto crossAddNamedBinFnTy = LLVM::LLVMFunctionType::get(
              i32Ty, {ptrTy, i32Ty, ptrTy, i32Ty, ptrTy, i32Ty});
          auto crossAddNamedBinFn = getOrCreateRuntimeFunc(
              mod, rewriter, "__moore_cross_add_named_bin", crossAddNamedBinFnTy);

          // Define the MooreCrossBinsofFilter struct type:
          // { i32 cp_index, ptr bin_indices, i32 num_bins,
          //   ptr values, i32 num_values, i1 negate }
          auto i1Ty = IntegerType::get(ctx, 1);
          auto filterStructTy = LLVM::LLVMStructType::getLiteral(
              ctx, {i32Ty, ptrTy, i32Ty, ptrTy, i32Ty, i1Ty});

          for (auto cross : crosses) {
            StringRef crossName = cross.getSymName();
            std::string crossNameGlobal =
                ("__cross_name_" + cgName + "_" + crossName).str();
            Value crossNamePtr = createGlobalStringConstant(
                loc, mod, rewriter, crossName, crossNameGlobal);

            // Build the array of coverpoint indices for this cross
            ArrayAttr targets = cross.getTargets();
            int32_t numCps = targets.size();

            // Allocate stack space for the indices array
            auto indicesArrayTy = LLVM::LLVMArrayType::get(i32Ty, numCps);
            auto indicesAlloca = LLVM::AllocaOp::create(
                rewriter, loc, ptrTy, indicesArrayTy,
                LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(1)));

            // Store the coverpoint indices
            for (int32_t i = 0; i < numCps; ++i) {
              auto targetRef = cast<FlatSymbolRefAttr>(targets[i]);
              StringAttr targetName = targetRef.getAttr();
              int32_t targetIdx = 0;
              auto it = cpNameToIndex.find(targetName);
              if (it != cpNameToIndex.end())
                targetIdx = it->second;

              auto idxVal = LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(targetIdx));
              SmallVector<LLVM::GEPArg> gepIndices = {0, i};
              auto elemPtr = LLVM::GEPOp::create(
                  rewriter, loc, ptrTy, indicesArrayTy, indicesAlloca,
                  gepIndices);
              LLVM::StoreOp::create(rewriter, loc, idxVal, elemPtr);
            }

            // Call __moore_cross_create
            auto numCpsConst = LLVM::ConstantOp::create(
                rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numCps));
            auto crossCreateCall = LLVM::CallOp::create(
                rewriter, loc, TypeRange{i32Ty},
                SymbolRefAttr::get(crossCreateFn),
                ValueRange{cgHandle, crossNamePtr, indicesAlloca, numCpsConst});
            Value crossIdx = crossCreateCall.getResult();

            // Process named bins in this cross (CrossBinDeclOp)
            for (auto &crossBodyOp : cross.getBody().front()) {
              auto crossBin = dyn_cast<CrossBinDeclOp>(&crossBodyOp);
              if (!crossBin)
                continue;

              StringRef binName = crossBin.getSymName();
              std::string binNameGlobal =
                  ("__crossbin_name_" + cgName + "_" + crossName + "_" + binName)
                      .str();
              Value binNamePtr = createGlobalStringConstant(
                  loc, mod, rewriter, binName, binNameGlobal);

              // Get the bin kind
              auto binKind = crossBin.getKind();
              int32_t kindVal = 0; // MOORE_CROSS_BIN_NORMAL
              if (binKind == CoverageBinKind::IgnoreBins)
                kindVal = 1; // MOORE_CROSS_BIN_IGNORE
              else if (binKind == CoverageBinKind::IllegalBins)
                kindVal = 2; // MOORE_CROSS_BIN_ILLEGAL

              auto kindConst = LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(kindVal));

              // Collect binsof filters from the cross bin body
              SmallVector<BinsOfOp> binsofOps;
              for (auto &binBodyOp : crossBin.getBody().front()) {
                if (auto binsof = dyn_cast<BinsOfOp>(&binBodyOp))
                  binsofOps.push_back(binsof);
              }

              int32_t numFilters = binsofOps.size();
              Value filtersPtr;
              Value numFiltersConst = LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numFilters));

              if (numFilters > 0) {
                // Allocate array of filter structs
                auto filtersArrayTy =
                    LLVM::LLVMArrayType::get(filterStructTy, numFilters);
                auto filtersAlloca = LLVM::AllocaOp::create(
                    rewriter, loc, ptrTy, filtersArrayTy,
                    LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(1)));

                // Fill in each filter
                for (int32_t fi = 0; fi < numFilters; ++fi) {
                  auto binsof = binsofOps[fi];
                  auto targetRef = binsof.getTarget();

                  // Get the coverpoint index from the target symbol
                  // The target is either a coverpoint (@cp) or a bin (@cp::@bin)
                  StringRef cpSymName = targetRef.getRootReference().getValue();
                  int32_t cpIdxVal = 0;
                  auto cpIt = cpNameToIndex.find(
                      StringAttr::get(ctx, cpSymName));
                  if (cpIt != cpNameToIndex.end())
                    cpIdxVal = cpIt->second;

                  // GEP to the filter struct at index fi
                  SmallVector<LLVM::GEPArg> filterGepIndices = {0, fi};
                  auto filterPtr = LLVM::GEPOp::create(
                      rewriter, loc, ptrTy, filtersArrayTy, filtersAlloca,
                      filterGepIndices);

                  // Store cp_index (field 0)
                  auto cpIdxConst = LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty,
                      rewriter.getI32IntegerAttr(cpIdxVal));
                  SmallVector<LLVM::GEPArg> field0Indices = {0, 0};
                  auto field0Ptr = LLVM::GEPOp::create(
                      rewriter, loc, ptrTy, filterStructTy, filterPtr,
                      field0Indices);
                  LLVM::StoreOp::create(rewriter, loc, cpIdxConst, field0Ptr);

                  // Store bin_indices = nullptr (field 1)
                  auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
                  SmallVector<LLVM::GEPArg> field1Indices = {0, 1};
                  auto field1Ptr = LLVM::GEPOp::create(
                      rewriter, loc, ptrTy, filterStructTy, filterPtr,
                      field1Indices);
                  LLVM::StoreOp::create(rewriter, loc, nullPtr, field1Ptr);

                  // Store num_bins = 0 (field 2)
                  auto zeroI32 = LLVM::ConstantOp::create(
                      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
                  SmallVector<LLVM::GEPArg> field2Indices = {0, 2};
                  auto field2Ptr = LLVM::GEPOp::create(
                      rewriter, loc, ptrTy, filterStructTy, filterPtr,
                      field2Indices);
                  LLVM::StoreOp::create(rewriter, loc, zeroI32, field2Ptr);

                  // Handle intersect values (field 3 and 4)
                  auto intersectValues = binsof.getIntersectValues();
                  if (intersectValues && !intersectValues->empty()) {
                    int32_t numValues = intersectValues->size();

                    // Allocate array for intersect values
                    auto valuesArrayTy =
                        LLVM::LLVMArrayType::get(i64Ty, numValues);
                    auto valuesAlloca = LLVM::AllocaOp::create(
                        rewriter, loc, ptrTy, valuesArrayTy,
                        LLVM::ConstantOp::create(
                            rewriter, loc, i32Ty,
                            rewriter.getI32IntegerAttr(1)));

                    // Store each value
                    for (int32_t vi = 0; vi < numValues; ++vi) {
                      int64_t val = 0;
                      if (auto intAttr =
                              dyn_cast<IntegerAttr>((*intersectValues)[vi])) {
                        val = intAttr.getInt();
                      }
                      auto valConst = LLVM::ConstantOp::create(
                          rewriter, loc, i64Ty,
                          rewriter.getI64IntegerAttr(val));
                      SmallVector<LLVM::GEPArg> valGepIndices = {0, vi};
                      auto valPtr = LLVM::GEPOp::create(
                          rewriter, loc, ptrTy, valuesArrayTy, valuesAlloca,
                          valGepIndices);
                      LLVM::StoreOp::create(rewriter, loc, valConst, valPtr);
                    }

                    // Store values pointer (field 3)
                    SmallVector<LLVM::GEPArg> field3Indices = {0, 3};
                    auto field3Ptr = LLVM::GEPOp::create(
                        rewriter, loc, ptrTy, filterStructTy, filterPtr,
                        field3Indices);
                    LLVM::StoreOp::create(rewriter, loc, valuesAlloca,
                                          field3Ptr);

                    // Store num_values (field 4)
                    auto numValuesConst = LLVM::ConstantOp::create(
                        rewriter, loc, i32Ty,
                        rewriter.getI32IntegerAttr(numValues));
                    SmallVector<LLVM::GEPArg> field4Indices = {0, 4};
                    auto field4Ptr = LLVM::GEPOp::create(
                        rewriter, loc, ptrTy, filterStructTy, filterPtr,
                        field4Indices);
                    LLVM::StoreOp::create(rewriter, loc, numValuesConst,
                                          field4Ptr);
                  } else {
                    // No intersect values: null pointer and 0 count
                    SmallVector<LLVM::GEPArg> field3Indices = {0, 3};
                    auto field3Ptr = LLVM::GEPOp::create(
                        rewriter, loc, ptrTy, filterStructTy, filterPtr,
                        field3Indices);
                    LLVM::StoreOp::create(rewriter, loc, nullPtr, field3Ptr);

                    SmallVector<LLVM::GEPArg> field4Indices = {0, 4};
                    auto field4Ptr = LLVM::GEPOp::create(
                        rewriter, loc, ptrTy, filterStructTy, filterPtr,
                        field4Indices);
                    LLVM::StoreOp::create(rewriter, loc, zeroI32, field4Ptr);
                  }

                  // Store negate field (field 5) - get from BinsOfOp attribute
                  bool isNegated = binsof.getNegate();
                  auto negateBool = LLVM::ConstantOp::create(
                      rewriter, loc, i1Ty, rewriter.getBoolAttr(isNegated));
                  SmallVector<LLVM::GEPArg> field5Indices = {0, 5};
                  auto field5Ptr = LLVM::GEPOp::create(
                      rewriter, loc, ptrTy, filterStructTy, filterPtr,
                      field5Indices);
                  LLVM::StoreOp::create(rewriter, loc, negateBool, field5Ptr);
                }

                filtersPtr = filtersAlloca;
              } else {
                // No filters: pass null
                filtersPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
              }

              // Call __moore_cross_add_named_bin
              LLVM::CallOp::create(
                  rewriter, loc, TypeRange{i32Ty},
                  SymbolRefAttr::get(crossAddNamedBinFn),
                  ValueRange{cgHandle, crossIdx, binNamePtr, kindConst,
                             filtersPtr, numFiltersConst});
            }
          }
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

/// Lowering for CoverageBinDeclOp.
/// Bins are processed by the parent CovergroupDeclOp during its conversion.
/// Illegal and ignore bins are registered with the runtime. Normal bins
/// are used for coverage calculation. This pattern just erases the declaration.
struct CoverageBinDeclOpConversion
    : public OpConversionPattern<CoverageBinDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoverageBinDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Bins are handled by the parent CovergroupDeclOp conversion.
    // Just erase this op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CoverCrossDeclOp.
/// Cross coverage declarations are processed by the parent CovergroupDeclOp
/// during its conversion. This pattern just erases the declaration.
struct CoverCrossDeclOpConversion
    : public OpConversionPattern<CoverCrossDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoverCrossDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Cross coverage is handled by the parent CovergroupDeclOp conversion.
    // Just erase this op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for CrossBinDeclOp.
/// Cross bins are processed by the parent CovergroupDeclOp during its
/// conversion. This pattern just erases the declaration.
struct CrossBinDeclOpConversion : public OpConversionPattern<CrossBinDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CrossBinDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Cross bins are handled by the parent CovergroupDeclOp conversion.
    // Just erase this op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for BinsOfOp.
/// BinsOf operations are processed by the parent CovergroupDeclOp during
/// cross bin lowering. This pattern just erases the operation.
struct BinsOfOpConversion : public OpConversionPattern<BinsOfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinsOfOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // BinsOf operations are handled by the parent CovergroupDeclOp conversion.
    // Just erase this op.
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
/// Calls __moore_coverpoint_sample for each coverpoint in the covergroup,
/// then calls __moore_cross_sample to sample all cross coverage items.
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

    // Get or create __moore_cross_sample function.
    // void __moore_cross_sample(void *cg, int64_t *cp_values, int32_t num_values)
    auto crossSampleFnTy =
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i32Ty});
    auto crossSampleFn = getOrCreateRuntimeFunc(mod, rewriter,
                                                "__moore_cross_sample",
                                                crossSampleFnTy);

    // Get the covergroup handle.
    Value cgHandle = adaptor.getCovergroup();

    // Collect i64 values for cross sampling
    SmallVector<Value> i64Values;
    int32_t numValues = adaptor.getValues().size();

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
        // For now, use 0 as placeholder.
        i64Val = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(0));
      }

      i64Values.push_back(i64Val);

      auto idxConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(cpIndex));

      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(sampleFn),
                           ValueRange{cgHandle, idxConst, i64Val});
      ++cpIndex;
    }

    // Call __moore_cross_sample with an array of all sampled values.
    // This samples all cross coverage items in the covergroup.
    if (numValues > 0) {
      // Allocate stack space for the values array
      auto valuesArrayTy = LLVM::LLVMArrayType::get(i64Ty, numValues);
      auto valuesAlloca = LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, valuesArrayTy,
          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                   rewriter.getI32IntegerAttr(1)));

      // Store each i64 value in the array
      for (int32_t i = 0; i < numValues; ++i) {
        SmallVector<LLVM::GEPArg> gepIndices = {0, i};
        auto elemPtr = LLVM::GEPOp::create(
            rewriter, loc, ptrTy, valuesArrayTy, valuesAlloca, gepIndices);
        LLVM::StoreOp::create(rewriter, loc, i64Values[i], elemPtr);
      }

      // Call __moore_cross_sample
      auto numValuesConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numValues));
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(crossSampleFn),
                           ValueRange{cgHandle, valuesAlloca, numValuesConst});
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
// - ConstraintDisableSoftOp: Disable soft constraint on variable (erased, processed by solver)
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
///
/// In the constraint solving model, foreach constraints are evaluated during
/// the RandomizeOp lowering phase. The foreach structure is used to extract
/// element-wise constraints which are then applied during post-randomization
/// validation.
///
/// Example SystemVerilog:
///   foreach (arr[i]) { arr[i] inside {[0:100]}; }
///
/// The constraint body operations are processed during constraint extraction
/// and the foreach op itself is erased since the actual validation happens
/// at randomization time via runtime calls like __moore_constraint_foreach_validate.
struct ConstraintForeachOpConversion
    : public OpConversionPattern<ConstraintForeachOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintForeachOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Foreach constraints are processed during RandomizeOp conversion.
    // The constraint body is analyzed to extract element-wise constraints
    // which are then validated via runtime functions.
    //
    // For now, we erase the foreach op. Full validation support would involve:
    // 1. Extracting the constraint predicate from the body
    // 2. Generating a call to __moore_constraint_foreach_validate() with
    //    a function pointer to the predicate
    // 3. Handling complex constraints (implications, nested foreach, etc.)
    //
    // TODO: Generate loop-based validation for complex element constraints.
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
    // Skip ops inside RandomizeOp's inline_constraints region - they are
    // processed by RandomizeOpConversion which has higher benefit.
    if (auto randomizeOp = op->getParentOfType<RandomizeOp>())
      return failure();
    if (auto stdRandomizeOp = op->getParentOfType<StdRandomizeOp>())
      return failure();

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
    // Skip ops inside RandomizeOp's inline_constraints region - they are
    // processed by RandomizeOpConversion which has higher benefit.
    if (auto randomizeOp = op->getParentOfType<RandomizeOp>())
      return failure();
    if (auto stdRandomizeOp = op->getParentOfType<StdRandomizeOp>())
      return failure();

    // Inside constraints are processed by extractRangeConstraints().
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintSolveBeforeOp.
/// Solve-before constraints specify variable ordering for the solver.
/// These are processed by extractSolveBeforeOrdering() during RandomizeOp
/// lowering to determine the order in which constraints are applied.
/// The op is erased after constraint ordering has been computed.
struct ConstraintSolveBeforeOpConversion
    : public OpConversionPattern<ConstraintSolveBeforeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintSolveBeforeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Solve-before ordering is processed during RandomizeOp conversion.
    // The ordering information has already been extracted and used to sort
    // constraints, so this op can be safely erased.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintDisableSoftOp.
/// Disables soft constraints on a variable. Currently erased.
struct ConstraintDisableSoftOpConversion
    : public OpConversionPattern<ConstraintDisableSoftOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintDisableSoftOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Soft constraint disabling is handled during constraint extraction.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering for ConstraintUniqueOp.
/// Uniqueness constraints require all elements to have distinct values.
/// Generates runtime calls to validate uniqueness:
/// - For arrays: calls __moore_constraint_unique_check
/// - For scalar variables: calls __moore_constraint_unique_scalars
struct ConstraintUniqueOpConversion
    : public OpConversionPattern<ConstraintUniqueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintUniqueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // If this unique constraint is inside a constraint block, just erase it.
    // Constraint blocks are declarations that are processed during RandomizeOp
    // lowering. The unique constraint info is extracted there.
    if (op->getParentOfType<ConstraintBlockOp>()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto variables = op.getVariables();
    if (variables.empty()) {
      // No variables to check - just erase
      rewriter.eraseOp(op);
      return success();
    }

    // Check if we have a single array variable or multiple scalar variables
    if (variables.size() == 1) {
      Value var = adaptor.getVariables()[0];
      Type mooreType = op.getVariables()[0].getType();

      // Check if it's an array type
      if (auto arrayType = dyn_cast<UnpackedArrayType>(mooreType)) {
        // Single array - check all elements are unique
        // Get array size and element size
        int64_t numElements = arrayType.getSize();
        Type elementType = arrayType.getElementType();
        Type convertedElemType = typeConverter->convertType(elementType);

        // Calculate element size in bytes
        int64_t elementSize = 1;
        if (auto intTy = dyn_cast<IntegerType>(convertedElemType))
          elementSize = (intTy.getWidth() + 7) / 8;
        else
          elementSize = getTypeSizeSafe(convertedElemType, mod);

        // Get pointer to the array data
        // The converted value should be an LLVM array or pointer to it
        Value arrayPtr = var;

        // If the value is an array (not a pointer), we need to store it
        // to get a pointer
        Type convertedType = var.getType();
        if (auto llvmArrayTy = dyn_cast<LLVM::LLVMArrayType>(convertedType)) {
          // LLVM array type - can store directly
          auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(1));
          auto alloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayTy, one);
          LLVM::StoreOp::create(rewriter, loc, var, alloca);
          arrayPtr = alloca;
        } else if (auto hwArrayTy = dyn_cast<hw::ArrayType>(convertedType)) {
          // hw::ArrayType - convert to LLVM array type first
          Type elemTy = hwArrayTy.getElementType();
          auto llvmArrayType =
              LLVM::LLVMArrayType::get(elemTy, hwArrayTy.getNumElements());
          auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(1));
          auto alloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayType, one);
          // Cast hw.array to LLVM array for storage
          auto casted = UnrealizedConversionCastOp::create(
                            rewriter, loc, llvmArrayType, var)
                            .getResult(0);
          LLVM::StoreOp::create(rewriter, loc, casted, alloca);
          arrayPtr = alloca;
        }

        // Create constants for runtime call
        auto numElemsConst = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(numElements));
        auto elemSizeConst = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elementSize));

        // Call __moore_constraint_unique_check(array, numElements, elementSize)
        auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty, i64Ty});
        auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                         "__moore_constraint_unique_check",
                                         fnTy);
        LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                             SymbolRefAttr::get(fn),
                             ValueRange{arrayPtr, numElemsConst, elemSizeConst});

        rewriter.eraseOp(op);
        return success();
      }

      // Check for queue type
      if (auto queueType = dyn_cast<QueueType>(mooreType)) {
        // Queue is stored as {ptr, length} struct
        // Extract the data pointer and length
        Type elementType = queueType.getElementType();
        Type convertedElemType = typeConverter->convertType(elementType);

        int64_t elementSize = 1;
        if (auto intTy = dyn_cast<IntegerType>(convertedElemType))
          elementSize = (intTy.getWidth() + 7) / 8;
        else
          elementSize = getTypeSizeSafe(convertedElemType, mod);

        // Extract data pointer (field 0) and length (field 1)
        Value dataPtr =
            LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, var, ArrayRef<int64_t>{0});
        Value length =
            LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, var, ArrayRef<int64_t>{1});

        auto elemSizeConst = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elementSize));

        auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty, i64Ty});
        auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                         "__moore_constraint_unique_check",
                                         fnTy);
        LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                             SymbolRefAttr::get(fn),
                             ValueRange{dataPtr, length, elemSizeConst});

        rewriter.eraseOp(op);
        return success();
      }

      // Check for dynamic array type
      if (auto dynArrayType = dyn_cast<OpenUnpackedArrayType>(mooreType)) {
        // Dynamic array is stored as {ptr, length} struct
        Type elementType = dynArrayType.getElementType();
        Type convertedElemType = typeConverter->convertType(elementType);

        int64_t elementSize = 1;
        if (auto intTy = dyn_cast<IntegerType>(convertedElemType))
          elementSize = (intTy.getWidth() + 7) / 8;
        else
          elementSize = getTypeSizeSafe(convertedElemType, mod);

        Value dataPtr =
            LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, var, ArrayRef<int64_t>{0});
        Value length =
            LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, var, ArrayRef<int64_t>{1});

        auto elemSizeConst = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elementSize));

        auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty, i64Ty});
        auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                         "__moore_constraint_unique_check",
                                         fnTy);
        LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                             SymbolRefAttr::get(fn),
                             ValueRange{dataPtr, length, elemSizeConst});

        rewriter.eraseOp(op);
        return success();
      }

      // Single scalar - always unique, just erase
      rewriter.eraseOp(op);
      return success();
    }

    // Multiple scalar variables - collect them into an array and check
    // All variables should have the same type for uniqueness constraint
    Type firstMooreType = op.getVariables()[0].getType();
    Type convertedType = typeConverter->convertType(firstMooreType);

    int64_t valueSize = 1;
    if (auto intTy = dyn_cast<IntegerType>(convertedType))
      valueSize = (intTy.getWidth() + 7) / 8;
    else
      valueSize = getTypeSizeSafe(convertedType, mod);

    int64_t numValues = variables.size();

    // Allocate stack space for the values array
    auto arrayTy = LLVM::LLVMArrayType::get(convertedType, numValues);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(1));
    auto valuesAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrayTy, one);

    // Store each value into the array
    for (int64_t i = 0; i < numValues; ++i) {
      Value val = adaptor.getVariables()[i];
      auto idx = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(i));
      auto elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, convertedType,
                                         valuesAlloca, ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, val, elemPtr);
    }

    // Create constants for runtime call
    auto numValsConst = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(numValues));
    auto valSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(valueSize));

    // Call __moore_constraint_unique_scalars(values, numValues, valueSize)
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, i64Ty, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_constraint_unique_scalars", fnTy);
    LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                         SymbolRefAttr::get(fn),
                         ValueRange{valuesAlloca, numValsConst, valSizeConst});

    rewriter.eraseOp(op);
    return success();
  }
};

/// moore.vtable lowering: generate LLVM global array of function pointers.
/// For dynamic dispatch, each class has a vtable global containing function
/// pointers indexed by the method's vtable index.
/// Note: This pattern has a lower benefit than VTableLoadMethodOpConversion
/// to ensure vtables are processed after load_method ops establish method usage.
struct VTableOpConversion : public OpConversionPattern<VTableOp> {
  VTableOpConversion(TypeConverter &tc, MLIRContext *ctx, ClassTypeCache &cache)
      : OpConversionPattern<VTableOp>(tc, ctx, /*benefit=*/1), cache(cache) {}

  LogicalResult
  matchAndRewrite(VTableOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only process top-level vtables (nested vtables are handled recursively)
    if (op->getParentOfType<VTableOp>()) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get the class symbol from the vtable's sym_name.
    // Vtable sym_name is @ClassName::@vtable, so root reference is the class.
    auto vtableSymName = op.getSymName();
    StringRef className = vtableSymName.getRootReference();
    auto classSym = SymbolRefAttr::get(ctx, className);

    // Resolve class struct info to get method-to-index mapping
    if (failed(resolveClassStructBody(mod, classSym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve class struct for " << classSym;

    auto structInfoOpt = cache.getStructInfo(classSym);
    if (!structInfoOpt)
      return op.emitError() << "No struct info for class " << classSym;

    auto &structInfo = *structInfoOpt;
    auto &methodToIndex = structInfo.methodToVtableIndex;

    // If no methods, just erase the vtable
    if (methodToIndex.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Collect vtable entries from this vtable and nested vtables.
    // We need to find the most-derived implementation for each method.
    DenseMap<StringRef, SymbolRefAttr> methodToTarget;

    std::function<void(VTableOp)> collectEntries = [&](VTableOp vt) {
      for (Operation &child : vt.getBody().front()) {
        if (auto entry = dyn_cast<VTableEntryOp>(child)) {
          // Overwrite with more derived implementation
          methodToTarget[entry.getName()] = entry.getTarget();
        } else if (auto nestedVt = dyn_cast<VTableOp>(child)) {
          // Process nested vtable (base class) first
          collectEntries(nestedVt);
        }
      }
    };
    collectEntries(op);

    // Determine vtable size (max index + 1)
    unsigned vtableSize = 0;
    for (const auto &kv : methodToIndex) {
      if (kv.second >= vtableSize)
        vtableSize = kv.second + 1;
    }

    if (vtableSize == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    // Create vtable array type: array of pointers
    auto vtableArrayTy = LLVM::LLVMArrayType::get(ptrTy, vtableSize);

    // Create the global vtable with an initializer region.
    // We use LLVM::AddressOfOp to reference functions, but since functions
    // are still func.func at this point (not llvm.func), we need to reference
    // them as llvm.func symbols. The func-to-llvm pass will create matching
    // llvm.func declarations.
    std::string globalName = structInfo.vtableGlobalName;

    // Check if vtable global already exists (may have been created as a
    // placeholder by ClassNewOpConversion).
    LLVM::GlobalOp existingGlobal = mod.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (existingGlobal) {
      // Global exists as placeholder - we still need to add vtable entries
      // Build vtable entries attribute for the existing global
      SmallVector<Attribute> vtableEntries;
      for (const auto &[methodName, targetSym] : methodToTarget) {
        auto indexIt = methodToIndex.find(methodName);
        if (indexIt == methodToIndex.end())
          continue;
        unsigned index = indexIt->second;
        SmallVector<Attribute> entry;
        entry.push_back(rewriter.getI64IntegerAttr(index));
        entry.push_back(targetSym);
        vtableEntries.push_back(rewriter.getArrayAttr(entry));
      }
      if (!vtableEntries.empty()) {
        existingGlobal->setAttr("circt.vtable_entries",
                                rewriter.getArrayAttr(vtableEntries));
      }
      rewriter.eraseOp(op);
      return success();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());

    // Create a zero-initialized vtable global.
    // The vtable will be populated with actual function pointers during a
    // later lowering pass (InitVtablesPass) when functions are converted
    // to LLVM and we can use AddressOfOp properly.
    //
    // For now, the vtable structure is:
    // - An array of ptr, size = max vtable index + 1
    // - Zero-initialized (all null pointers)
    //
    // The dynamic dispatch mechanism (load vtable ptr from object, index into
    // array) is fully functional. Once the vtable is populated, polymorphism
    // will work correctly.
    auto zeroAttr = LLVM::ZeroAttr::get(ctx);
    auto vtableGlobal = LLVM::GlobalOp::create(rewriter, loc, vtableArrayTy,
                                               /*isConstant=*/false,
                                               LLVM::Linkage::Internal,
                                               globalName, zeroAttr);

    // Store vtable metadata as an attribute for InitVtablesPass.
    // Create an array of [index, funcSymbol] pairs for each vtable entry.
    SmallVector<Attribute> vtableEntries;
    for (const auto &[methodName, targetSym] : methodToTarget) {
      auto indexIt = methodToIndex.find(methodName);
      if (indexIt == methodToIndex.end())
        continue;
      unsigned index = indexIt->second;
      // Store as [index, funcSymbol] tuple
      SmallVector<Attribute> entry;
      entry.push_back(rewriter.getI64IntegerAttr(index));
      entry.push_back(targetSym);
      vtableEntries.push_back(rewriter.getArrayAttr(entry));
    }
    if (!vtableEntries.empty()) {
      vtableGlobal->setAttr("circt.vtable_entries",
                            rewriter.getArrayAttr(vtableEntries));
    }

    // Erase the original vtable op
    rewriter.setInsertionPoint(op);
    rewriter.eraseOp(op);
    return success();
  }

private:
  ClassTypeCache &cache;
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

/// moore.vtable.load_method lowering: perform dynamic dispatch via vtable pointer.
/// This loads the vtable pointer from the object, then indexes into it at runtime
/// to get the function pointer for the requested method.
/// Note: Higher benefit ensures this runs before VTableOp/VTableEntryOp erasure.
struct VTableLoadMethodOpConversion
    : public OpConversionPattern<VTableLoadMethodOp> {
  VTableLoadMethodOpConversion(TypeConverter &tc, MLIRContext *ctx,
                               ClassTypeCache &cache)
      : OpConversionPattern<VTableLoadMethodOp>(tc, ctx, /*benefit=*/10),
        cache(cache) {}

  LogicalResult
  matchAndRewrite(VTableLoadMethodOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Get the class type from the object operand.
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    auto classSym = handleTy.getClassSym();

    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // Resolve the class struct info to get method-to-vtable-index mapping
    if (failed(resolveClassStructBody(mod, classSym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve class struct for " << classSym;

    auto structInfoOpt = cache.getStructInfo(classSym);
    if (!structInfoOpt)
      return op.emitError() << "No struct info for class " << classSym;

    auto &structInfo = *structInfoOpt;

    // Get the method name and find its vtable index
    auto methodSym = op.getMethodSym();
    StringRef methodName = methodSym.getLeafReference();

    auto indexIt = structInfo.methodToVtableIndex.find(methodName);
    if (indexIt == structInfo.methodToVtableIndex.end())
      return rewriter.notifyMatchFailure(
          op, "method " + methodName.str() + " not found in vtable index map");

    unsigned vtableIndex = indexIt->second;

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    // Get the object pointer (already converted by adaptor)
    Value objPtr = adaptor.getObject();

    // Get the class's struct type for GEP
    auto structTy = structInfo.classBody;

    // Build GEP path to the vtable pointer field.
    // The vtable pointer is at index 1 in the root class (after typeId).
    // For inheritance hierarchy A -> B -> C:
    //   C: { B: { A: { i32 typeId, ptr vtablePtr, ...}, ...}, ...}
    // We need to GEP through [0, 0, ..., 0, 1] to reach vtablePtr.
    SmallVector<Value> gepIndices;
    // First index is always 0 (pointer dereference)
    gepIndices.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    // Navigate through base class chain
    for (int32_t i = 0; i < structInfo.inheritanceDepth; ++i) {
      gepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
    }
    // Index to vtable pointer field (index 1 in root class)
    gepIndices.push_back(LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(ClassTypeCache::ClassStructInfo::vtablePtrFieldIndex)));

    // GEP to vtable pointer field
    Value vtablePtrPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, objPtr, gepIndices);

    // Load the vtable pointer
    Value vtablePtr = LLVM::LoadOp::create(rewriter, loc, ptrTy, vtablePtrPtr);

    // GEP into the vtable array at the method's index.
    // The vtable is an array of pointers, so we create a GEP that indexes
    // into the array. For LLVM GEP on an array type:
    // - First index (0) dereferences the pointer to the array
    // - Second index (vtableIndex) selects the element
    //
    // vtable[vtableIndex] = &((*vtablePtr)[vtableIndex])
    unsigned vtableSize = 0;
    for (const auto &kv : structInfo.methodToVtableIndex) {
      if (kv.second >= vtableSize)
        vtableSize = kv.second + 1;
    }
    auto vtableArrayTy = LLVM::LLVMArrayType::get(ptrTy, vtableSize > 0 ? vtableSize : 1);

    SmallVector<LLVM::GEPArg> vtableGepIndices;
    vtableGepIndices.push_back(static_cast<int64_t>(0));  // Dereference pointer
    vtableGepIndices.push_back(static_cast<int64_t>(vtableIndex));  // Array index

    Value funcPtrPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, vtableArrayTy, vtablePtr, vtableGepIndices);

    // Load the function pointer from the vtable
    Value funcPtr = LLVM::LoadOp::create(rewriter, loc, ptrTy, funcPtrPtr);

    // The result type should be a function pointer type
    rewriter.replaceOp(op, funcPtr);
    return success();
  }

private:
  ClassTypeCache &cache;
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

/// Lowering for VirtualInterfaceBindOp.
/// Binds an interface instance to a virtual interface variable by storing
/// the interface pointer into the destination reference.
struct VirtualInterfaceBindOpConversion
    : public OpConversionPattern<VirtualInterfaceBindOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VirtualInterfaceBindOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // The destination is a reference to a virtual interface (converted to
    // !llhd.ref<!llvm.ptr>). We need to store the source pointer into it.
    Value dest = adaptor.getDest();
    Value source = adaptor.getSource();

    // Check if the source is a reference type (e.g., from interface.instance).
    // If so, we need to probe it to get the actual pointer value.
    auto sourceType = op.getSource().getType();
    if (isa<moore::RefType>(sourceType)) {
      // The source is a reference to virtual interface - probe to get the
      // pointer value.
      source = llhd::ProbeOp::create(rewriter, loc, source);
    }

    // Get the destination as an LLVM pointer for the store.
    // The dest should be !llhd.ref<!llvm.ptr>, we need the underlying pointer.
    auto destRefType = dyn_cast<llhd::RefType>(dest.getType());
    if (!destRefType) {
      // If it's already an LLVM pointer (from class property), use llvm.store
      if (isa<LLVM::LLVMPointerType>(dest.getType())) {
        // Convert source to LLVM-compatible type if needed
        // For 4-state hw.struct<value, unknown>, convert to llvm.struct<(i, i)>
        Value storeVal = source;
        Type storeType = source.getType();
        Type llvmStoreType = convertToLLVMType(storeType);

        if (storeType != llvmStoreType) {
          storeVal = UnrealizedConversionCastOp::create(
                         rewriter, loc, llvmStoreType, storeVal)
                         .getResult(0);
        }
        LLVM::StoreOp::create(rewriter, loc, storeVal, dest);
        rewriter.eraseOp(op);
        return success();
      }
      return rewriter.notifyMatchFailure(op,
                                         "expected ref type for destination");
    }

    // Use llhd.drv to drive the signal with the new pointer value.
    auto timeZero = llhd::ConstantTimeOp::create(
        rewriter, loc,
        llhd::TimeAttr::get(rewriter.getContext(), 0, "ns", 0, 1));
    llhd::DriveOp::create(rewriter, loc, dest, source, timeZero.getResult(),
                          Value{});

    rewriter.eraseOp(op);
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

    // Handle wildcard associative array variables [*] - same as regular assoc
    // arrays but with default key/value sizes since index type is unspecified.
    if (auto wildcardType = dyn_cast<WildcardAssocArrayType>(nestedMooreType)) {
      auto *ctx = rewriter.getContext();
      ModuleOp mod = op->getParentOfType<ModuleOp>();

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get or create __moore_assoc_create function
      auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {i32Ty, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_create", fnTy);

      // Wildcard associative arrays can use any index type, so we use
      // string key (keySize=0) as a sensible default that works with any type.
      int32_t keySize = 0;

      // Determine value size
      auto valueType = wildcardType.getElementType();
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

    // Handle string variables - these need stack allocation
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

      // Check if there is an initial value
      Value initVal = adaptor.getInitial();
      if (initVal) {
        // Use the provided initial value
        LLVM::StoreOp::create(rewriter, loc, initVal, alloca);
      } else {
        // Initialize with empty string {nullptr, 0}
        auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
        auto zeroLen = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));

        // Build the struct value
        Value structVal = LLVM::UndefOp::create(rewriter, loc, structTy);
        structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal,
                                                nullPtr, ArrayRef<int64_t>{0});
        structVal = LLVM::InsertValueOp::create(rewriter, loc, structVal,
                                                zeroLen, ArrayRef<int64_t>{1});

        // Store to alloca
        LLVM::StoreOp::create(rewriter, loc, structVal, alloca);
      }

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

    // Handle fixed-size unpacked arrays containing LLVM types (e.g., string[N])
    // These get converted to LLVM array types, not hw::ArrayType
    if (auto unpackedArrayType = dyn_cast<UnpackedArrayType>(nestedMooreType)) {
      auto convertedType = typeConverter->convertType(nestedMooreType);
      if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(convertedType)) {
        auto *ctx = rewriter.getContext();
        auto ptrTy = LLVM::LLVMPointerType::get(ctx);

        // Create an alloca for the array
        auto one = LLVM::ConstantOp::create(rewriter, loc,
                                            rewriter.getI64IntegerAttr(1));
        auto alloca =
            LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrayTy, one);

        // Initialize with zero values - use LLVM's ZeroOp to create a zeroed
        // array (all elements zero-initialized, e.g., empty strings)
        auto zeroVal = LLVM::ZeroOp::create(rewriter, loc, arrayTy);

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

    // For local variables inside procedural blocks (llhd.process) or inside
    // functions (func.func), use LLVM alloca instead of llhd.sig. This gives
    // immediate memory semantics, which is required when:
    // 1. Variables are passed as ref parameters to functions
    // 2. Functions are inlined (they expect immediate memory access)
    // 3. Functions are called from global constructors (llvm.global_ctors),
    //    where no LLHD simulation context exists
    // Signal semantics (with delta-cycle delays) only work correctly within
    // the LLHD simulation runtime.
    if (op->getParentOfType<llhd::ProcessOp>() ||
        op->getParentOfType<func::FuncOp>()) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto elementType = refType.getNestedType();

      // Convert the element type to LLVM type if it's an hw type (e.g.,
      // hw::StructType for 4-state values gets converted to LLVM::LLVMStructType)
      Type llvmElementType = convertToLLVMType(elementType);

      // Create an alloca for the variable
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto alloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElementType, one);

      // Convert the init value to LLVM type if needed (e.g., hw.struct -> llvm.struct)
      Value storeVal = convertValueToLLVMType(init, loc, rewriter);

      // Store the initial value
      LLVM::StoreOp::create(rewriter, loc, storeVal, alloca);

      rewriter.replaceOp(op, alloca.getResult());
      return success();
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
    auto kind = op.getKind();

    // Support wire, interconnect, supply0, and supply1 nets.
    // Interconnect nets are used for connecting signals with potentially
    // different types but can be treated as wires for lowering purposes.
    // Supply0/supply1 are constant-driven nets (ground/power).
    if (kind != NetKind::Wire && kind != NetKind::Interconnect &&
        kind != NetKind::Supply0 && kind != NetKind::Supply1)
      return rewriter.notifyMatchFailure(
          loc, "only wire/interconnect/supply0/supply1 nets supported");

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

    bool isFourState = isFourStateStructType(elementType);
    int64_t valueWidth = 0;
    if (isFourState) {
      auto structTy = cast<hw::StructType>(elementType);
      auto valueTy = cast<IntegerType>(structTy.getElements()[0].type);
      valueWidth = valueTy.getWidth();
    }

    auto makeFourStateConst = [&](APInt valueBits,
                                  APInt unknownBits) -> Value {
      auto structTy = cast<hw::StructType>(elementType);
      auto valueTy = cast<IntegerType>(structTy.getElements()[0].type);
      auto unknownTy = cast<IntegerType>(structTy.getElements()[1].type);
      auto valueConst = hw::ConstantOp::create(
          rewriter, loc, IntegerAttr::get(valueTy, valueBits));
      auto unknownConst = hw::ConstantOp::create(
          rewriter, loc, IntegerAttr::get(unknownTy, unknownBits));
      return createFourStateStruct(rewriter, loc, valueConst, unknownConst);
    };

    // For supply0/supply1 nets, initialize to the appropriate constant value.
    // supply0 = always 0 (ground), supply1 = always 1 (all ones).
    // For nets with an assignment (e.g., from module inputs), use the assigned
    // value as the initial value to avoid initialization order issues where
    // probes read zero before drives take effect.
    Value init;
    Value assignedValue = adaptor.getAssignment();
    if (kind == NetKind::Supply0) {
      if (isFourState) {
        init = makeFourStateConst(APInt(valueWidth, 0), APInt(valueWidth, 0));
      } else {
        auto constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
        init =
            rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);
      }
    } else if (kind == NetKind::Supply1) {
      if (isFourState) {
        init = makeFourStateConst(APInt::getAllOnes(valueWidth),
                                  APInt(valueWidth, 0));
      } else {
        auto constOne =
            hw::ConstantOp::create(rewriter, loc, APInt::getAllOnes(width));
        init =
            rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constOne);
      }
    } else if (assignedValue) {
      // Use the assigned value as initial value to fix initialization order.
      // This ensures signals driven from module inputs have the correct value
      // from t=0, rather than zero that would be overwritten with epsilon delay.
      init = assignedValue;
    } else {
      if (isFourState) {
        init = makeFourStateConst(APInt(valueWidth, 0),
                                  APInt::getAllOnes(valueWidth));
      } else {
        auto constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
        init =
            rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);
      }
    }

    auto signal = rewriter.replaceOpWithNewOp<llhd::SignalOp>(
        op, resultType, op.getNameAttr(), init);

    // For nets with assigned values, also emit a continuous drive to handle
    // dynamic updates. The initial value handles t=0, the drive handles
    // subsequent changes.
    if (assignedValue) {
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

    auto globalOp = LLVM::GlobalOp::create(rewriter, loc, convertedType,
                                           /*isConstant=*/false,
                                           LLVM::Linkage::Internal,
                                           op.getSymName(), initAttr);

    // If there's an init region with a YieldOp, create a global constructor
    // function to initialize the variable at program startup.
    Block *initBlock = op.getInitBlock();
    if (initBlock && !initBlock->empty()) {
      // Find the YieldOp that provides the initial value.
      auto yieldOp = dyn_cast<YieldOp>(initBlock->getTerminator());
      if (yieldOp) {
        // Create an initializer function.
        std::string initFuncName =
            ("__moore_global_init_" + op.getSymName()).str();
        auto voidTy = LLVM::LLVMVoidType::get(op.getContext());
        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {});

        auto initFunc = LLVM::LLVMFuncOp::create(rewriter, loc, initFuncName,
                                                 funcTy);
        initFunc.setLinkage(LLVM::Linkage::Internal);

        // Create the function body.
        Block *funcBlock = rewriter.createBlock(&initFunc.getBody());
        rewriter.setInsertionPointToStart(funcBlock);

        // Clone all operations from the init region except the yield.
        IRMapping mapping;
        for (Operation &initOp : *initBlock) {
          if (!isa<YieldOp>(initOp)) {
            rewriter.clone(initOp, mapping);
          }
        }

        // Get the yielded value (mapped through the cloning).
        Value initValue = mapping.lookupOrDefault(yieldOp.getResult());

        // Convert the init value type if needed.
        Type valueType = initValue.getType();
        if (valueType != convertedType) {
          // Try unrealized conversion cast for type mismatch.
          initValue = rewriter.create<UnrealizedConversionCastOp>(
                                 loc, convertedType, initValue)
                          .getResult(0);
        }

        // Store the value to the global variable.
        auto ptrTy = LLVM::LLVMPointerType::get(op.getContext());
        auto addressOf =
            LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalOp.getSymName());
        LLVM::StoreOp::create(rewriter, loc, initValue, addressOf.getResult());

        // Return from the function.
        LLVM::ReturnOp::create(rewriter, loc, ValueRange{});

        // Register the initializer function with llvm.global_ctors.
        // Insert at module level after the global.
        rewriter.setInsertionPointAfter(globalOp);
        auto ctorAttr = rewriter.getArrayAttr(
            {FlatSymbolRefAttr::get(op.getContext(), initFuncName)});
        auto priorityAttr = rewriter.getI32ArrayAttr({65535}); // Default priority
        auto dataAttr = rewriter.getArrayAttr({LLVM::ZeroAttr::get(op.getContext())});
        LLVM::GlobalCtorsOp::create(rewriter, loc, ctorAttr, priorityAttr, dataAttr);
      }
    }

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
    auto fvint = op.getValue();
    auto width = fvint.getBitWidth();
    auto loc = op.getLoc();

    // Get the converted type (either IntegerType or 4-state struct)
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Check if this is a 4-state type
    if (isFourStateStructType(resultType)) {
      // 4-state constant: create a struct {value, unknown}
      auto intType = rewriter.getIntegerType(width);

      // Get the value component (0/X=0, 1/Z=1)
      auto valueAPInt = fvint.getRawValue();
      Value valueConst = hw::ConstantOp::create(
          rewriter, loc, intType, rewriter.getIntegerAttr(intType, valueAPInt));

      // Get the unknown mask (0/1=0, X/Z=1)
      auto unknownAPInt = fvint.getRawUnknown();
      Value unknownConst = hw::ConstantOp::create(
          rewriter, loc, intType,
          rewriter.getIntegerAttr(intType, unknownAPInt));

      // Create the 4-state struct
      auto result =
          createFourStateStruct(rewriter, loc, valueConst, unknownConst);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Two-valued constant: create a simple integer
    auto intType = rewriter.getIntegerType(width);
    // Convert FVInt to APInt - X and Z bits are mapped to 0
    auto value = fvint.toAPInt(false);
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, intType, rewriter.getIntegerAttr(intType, value));
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

// moore.constant_time -> i64 (femtoseconds)
// Creates an llhd.constant_time and converts it to i64 using an unrealized cast.
// The conversion pass will materialize this properly.
struct ConstantTimeOpConv : public OpConversionPattern<ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the time constant value directly as i64 (femtoseconds).
    auto i64Ty = rewriter.getIntegerType(64);
    auto valueAttr = rewriter.getIntegerAttr(i64Ty, op.getValue());
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, valueAttr);
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
    Location loc = op.getLoc();
    auto values = adaptor.getValues();

    // Check if any of the input types are 4-state struct types
    bool hasFourState = false;
    for (auto value : values) {
      if (isFourStateStructType(value.getType())) {
        hasFourState = true;
        break;
      }
    }

    if (!hasFourState) {
      // All inputs are 2-state integers - use simple concat
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, values);
      return success();
    }

    // Handle 4-state types: concatenate value and unknown components separately
    SmallVector<Value> valueComponents;
    SmallVector<Value> unknownComponents;

    for (auto value : values) {
      if (isFourStateStructType(value.getType())) {
        valueComponents.push_back(
            extractFourStateValue(rewriter, loc, value));
        unknownComponents.push_back(
            extractFourStateUnknown(rewriter, loc, value));
      } else {
        // 2-state value - unknown bits are all 0
        valueComponents.push_back(value);
        auto intType = cast<IntegerType>(value.getType());
        auto zero = hw::ConstantOp::create(rewriter, loc, intType, 0);
        unknownComponents.push_back(zero);
      }
    }

    Value concatValue = comb::ConcatOp::create(rewriter, loc, valueComponents);
    Value concatUnknown =
        comb::ConcatOp::create(rewriter, loc, unknownComponents);
    Value result =
        createFourStateStruct(rewriter, loc, concatValue, concatUnknown);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getValue();

    // Check if the input is a 4-state struct type
    if (isFourStateStructType(input.getType())) {
      // Handle 4-state types: replicate value and unknown components separately
      Value valueComp = extractFourStateValue(rewriter, loc, input);
      Value unknownComp = extractFourStateUnknown(rewriter, loc, input);

      auto structType = cast<hw::StructType>(resultType);
      auto valueType = structType.getElements()[0].type;

      Value replicatedValue =
          comb::ReplicateOp::create(rewriter, loc, valueType, valueComp);
      Value replicatedUnknown =
          comb::ReplicateOp::create(rewriter, loc, valueType, unknownComp);
      Value result =
          createFourStateStruct(rewriter, loc, replicatedValue, replicatedUnknown);
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType, input);
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

    // Handle LLVM array types (arrays of LLVM types like strings)
    if (auto llvmArrTy = dyn_cast<LLVM::LLVMArrayType>(inputType)) {
      auto loc = op.getLoc();
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto elemType = llvmArrTy.getElementType();

      // Store the array to get a pointer
      auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(1));
      auto alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getInput(), alloca);

      // GEP to the element
      auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
      auto idx = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(low));
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmArrTy,
                                          alloca, ValueRange{zero, idx});

      // Load the element
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);
      rewriter.replaceOp(op, elem);
      return success();
    }

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(inputType)) {
      auto loc = op.getLoc();
      auto structType = cast<hw::StructType>(inputType);
      auto valueType = structType.getElements()[0].type;
      auto intType = cast<IntegerType>(valueType);
      int32_t inputWidth = intType.getWidth();

      // Extract the value and unknown components
      Value inputValue = extractFourStateValue(rewriter, loc, adaptor.getInput());
      Value inputUnknown = extractFourStateUnknown(rewriter, loc, adaptor.getInput());

      // Check if result is also 4-state
      if (isFourStateStructType(resultType)) {
        auto resultStructType = cast<hw::StructType>(resultType);
        auto resultValueType = resultStructType.getElements()[0].type;
        int32_t resultWidth = cast<IntegerType>(resultValueType).getWidth();
        int32_t high = low + resultWidth;

        // Helper to extract bits from a component with proper bounds handling.
        // For unknown masks, out-of-bounds bits should become unknown (=1).
        auto extractBits = [&](Value component, bool fillUnknown) -> Value {
          SmallVector<Value> toConcat;

          // Handle negative low index - prepend zeros
          if (low < 0) {
            int32_t fillWidth = std::min(-low, resultWidth);
            APInt fillBits = fillUnknown ? APInt::getAllOnes(fillWidth)
                                         : APInt(fillWidth, 0);
            toConcat.push_back(hw::ConstantOp::create(
                rewriter, loc, fillBits));
          }

          // Extract the middle portion that's within bounds
          if (low < inputWidth && high > 0) {
            int32_t lowIdx = std::max(low, 0);
            Value middle = rewriter.createOrFold<comb::ExtractOp>(
                loc,
                rewriter.getIntegerType(
                    std::min(resultWidth, std::min(high, inputWidth) - lowIdx)),
                component, lowIdx);
            toConcat.push_back(middle);
          }

          // Handle out-of-bounds high - append zeros
          int32_t diff = high - inputWidth;
          if (diff > 0) {
            APInt fillBits = fillUnknown ? APInt::getAllOnes(diff)
                                         : APInt(diff, 0);
            toConcat.push_back(
                hw::ConstantOp::create(rewriter, loc, fillBits));
          }

          return rewriter.createOrFold<comb::ConcatOp>(loc, toConcat);
        };

        Value extractedValue = extractBits(inputValue, /*fillUnknown=*/false);
        Value extractedUnknown = extractBits(inputUnknown, /*fillUnknown=*/true);

        // Create the result 4-state struct
        auto result = createFourStateStruct(rewriter, loc, extractedValue,
                                            extractedUnknown);
        rewriter.replaceOp(op, result);
      } else {
        // Result is 2-state - just extract the value bits
        int32_t resultWidth = hw::getBitWidth(resultType);
        int32_t high = low + resultWidth;

        SmallVector<Value> toConcat;
        if (low < 0)
          toConcat.push_back(hw::ConstantOp::create(
              rewriter, loc, APInt(std::min(-low, resultWidth), 0)));

        if (low < inputWidth && high > 0) {
          int32_t lowIdx = std::max(low, 0);
          Value middle = rewriter.createOrFold<comb::ExtractOp>(
              loc,
              rewriter.getIntegerType(
                  std::min(resultWidth, std::min(high, inputWidth) - lowIdx)),
              inputValue, lowIdx);
          toConcat.push_back(middle);
        }

        int32_t diff = high - inputWidth;
        if (diff > 0)
          toConcat.push_back(
              hw::ConstantOp::create(rewriter, loc, APInt(diff, 0)));

        Value concat = rewriter.createOrFold<comb::ConcatOp>(loc, toConcat);
        rewriter.replaceOp(op, concat);
      }
      return success();
    }

    // Handle hw::StructType (packed structs) - bitcast to integer and extract.
    // This handles SystemVerilog packed struct bit-slicing like pack1[15:8].
    if (auto structType = dyn_cast<hw::StructType>(inputType)) {
      int64_t structWidth = hw::getBitWidth(structType);
      if (structWidth < 0)
        return failure();

      // Bitcast the struct to an integer of the same total width
      auto intType = rewriter.getIntegerType(structWidth);
      Value intValue = rewriter.createOrFold<hw::BitcastOp>(
          op.getLoc(), intType, adaptor.getInput());

      int32_t inputWidth = structWidth;
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
            intValue, lowIdx);
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

    return failure();
  }
};

struct ExtractRefOpConversion : public OpConversionPattern<ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Handle LLVM pointer inputs (for fixed-size arrays containing LLVM types,
    // like string arrays)
    if (isa<LLVM::LLVMPointerType>(adaptor.getInput().getType())) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);

      // Get the original Moore array type to determine element type
      auto mooreInputRefType = cast<moore::RefType>(op.getInput().getType());
      auto nestedType = mooreInputRefType.getNestedType();

      if (auto arrayType = dyn_cast<UnpackedArrayType>(nestedType)) {
        auto elemMooreType = arrayType.getElementType();
        auto elemType = typeConverter->convertType(elemMooreType);

        // Create GEP with constant index to access the element
        auto i64Ty = IntegerType::get(ctx, 64);
        auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
        auto idx = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty,
            rewriter.getI64IntegerAttr(adaptor.getLowBit()));

        // Create the converted array type to use as GEP base type
        auto llvmArrayType = LLVM::LLVMArrayType::get(elemType, arrayType.getSize());

        // GEP to the element - use two indices: [0][idx] to deref ptr and index
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmArrayType,
                                            adaptor.getInput(), ValueRange{zero, idx});

        rewriter.replaceOp(op, elemPtr);
        return success();
      }
    }

    // The input must be an llhd.ref type (not LLVM pointer which is used for
    // dynamic containers like strings, queues, etc.)
    auto inputRefType = dyn_cast<llhd::RefType>(adaptor.getInput().getType());
    if (!inputRefType)
      return rewriter.notifyMatchFailure(
          loc, "input type must be llhd.ref, not LLVM pointer");
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

    // Handle 4-state struct types ({value, unknown} structs in llhd.ref)
    if (isFourStateStructType(inputType)) {
      auto loc = op.getLoc();
      // Extract from both value and unknown fields via value reads.
      Value baseValue = llhd::ProbeOp::create(rewriter, loc, adaptor.getInput());
      Value valueField = extractFourStateValue(rewriter, loc, baseValue);
      Value unknownField = extractFourStateUnknown(rewriter, loc, baseValue);

      // Determine the result width from the converted result type.
      auto resultRefType = cast<llhd::RefType>(resultType);
      auto resultStructType =
          cast<hw::StructType>(resultRefType.getNestedType());
      auto resultIntType =
          cast<IntegerType>(resultStructType.getElements()[0].type);
      auto resultWidth = resultIntType.getWidth();
      int32_t inputWidth =
          static_cast<int32_t>(cast<IntegerType>(valueField.getType()).getWidth());
      int32_t low = adaptor.getLowBit();
      int32_t high = low + static_cast<int32_t>(resultWidth);

      // Helper to extract bits with bounds handling. For unknown masks,
      // out-of-bounds bits should become unknown (=1).
      auto extractBits = [&](Value component, bool fillUnknown) -> Value {
        SmallVector<Value> toConcat;
        if (low < 0) {
          int32_t fillWidth = std::min(-low, static_cast<int32_t>(resultWidth));
          APInt fillBits = fillUnknown ? APInt::getAllOnes(fillWidth)
                                       : APInt(fillWidth, 0);
          toConcat.push_back(
              hw::ConstantOp::create(rewriter, loc, fillBits));
        }
        if (low < inputWidth && high > 0) {
          int32_t lowIdx = std::max(low, 0);
          Value middle = rewriter.createOrFold<comb::ExtractOp>(
              loc,
              rewriter.getIntegerType(std::min(
                  static_cast<int32_t>(resultWidth),
                  std::min(high, inputWidth) - lowIdx)),
              component, lowIdx);
          toConcat.push_back(middle);
        }
        int32_t diff = high - inputWidth;
        if (diff > 0) {
          APInt fillBits = fillUnknown ? APInt::getAllOnes(diff)
                                       : APInt(diff, 0);
          toConcat.push_back(
              hw::ConstantOp::create(rewriter, loc, fillBits));
        }
        return rewriter.createOrFold<comb::ConcatOp>(loc, toConcat);
      };

      Value extractedValue = extractBits(valueField, /*fillUnknown=*/false);
      Value extractedUnknown = extractBits(unknownField, /*fillUnknown=*/true);

      // Create the 4-state struct result.
      auto fourStateStruct = hw::StructCreateOp::create(
          rewriter, loc, resultStructType,
          ValueRange{extractedValue, extractedUnknown});

      // Create a new signal to hold the extracted value and return ref to it.
      // Note: This creates a read-only view; writes won't propagate back.
      // For full read-write support, a more complex mechanism is needed.
      auto signal = llhd::SignalOp::create(rewriter, loc, resultType,
                                           StringAttr{}, fourStateStruct);
      rewriter.replaceOp(op, signal.getResult());
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
      else {
        // For struct/array types, compute size from hw::getBitWidth
        int64_t bitWidth = hw::getBitWidth(resultType);
        if (bitWidth > 0)
          valueSize = (bitWidth + 7) / 8;
      }

      // Store the key to an alloca and pass its pointer
      auto keyType = adaptor.getLowBit().getType();
      // Convert key type to pure LLVM type if needed (for hw.struct keys)
      Type llvmKeyType = convertToLLVMType(keyType);
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmKeyType, one);
      // Cast the key value to LLVM type if needed (hw.struct -> llvm.struct)
      Value keyToStore = adaptor.getLowBit();
      if (llvmKeyType != keyType) {
        keyToStore = UnrealizedConversionCastOp::create(
                         rewriter, loc, llvmKeyType, ValueRange{keyToStore})
                         .getResult(0);
      }
      LLVM::StoreOp::create(rewriter, loc, keyToStore, keyAlloca);

      auto valueSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

      // Call __moore_assoc_get_ref(array, key_ptr, value_size) -> value_ptr
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
          ValueRange{adaptor.getInput(), keyAlloca, valueSizeConst});

      // Load the value from the returned pointer
      // Convert result type to pure LLVM type for the load operation
      Type llvmResultType = convertToLLVMType(resultType);
      auto loaded =
          LLVM::LoadOp::create(rewriter, loc, llvmResultType, call.getResult());

      // If result types differ (hw.struct vs llvm.struct), use unrealized cast
      if (llvmResultType != resultType) {
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, resultType, loaded.getResult());
      } else {
        rewriter.replaceOp(op, loaded.getResult());
      }
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

      // Handle 4-state index types which are lowered to {value, unknown} structs
      if (isFourStateStructType(idx.getType()))
        idx = extractFourStateValue(rewriter, loc, idx);

      if (idx.getType() != i64Ty) {
        if (cast<IntegerType>(idx.getType()).getWidth() < 64)
          idx = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
        else
          idx = arith::TruncIOp::create(rewriter, loc, i64Ty, idx);
      }

      // Convert result type to pure LLVM type for GEP and load operations
      Type llvmResultType = convertToLLVMType(resultType);

      // GEP to the i-th element
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmResultType,
                                          dataPtr, ValueRange{idx});

      // Load and return the element value
      auto loaded =
          LLVM::LoadOp::create(rewriter, loc, llvmResultType, elemPtr);

      // If result types differ (hw.struct vs llvm.struct), use unrealized cast
      if (llvmResultType != resultType) {
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, resultType, loaded.getResult());
      } else {
        rewriter.replaceOp(op, loaded.getResult());
      }
      return success();
    }

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      // Handle 4-state index - extract just the value part
      Value amount = adaptor.getLowBit();
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);
      amount = adjustIntegerWidth(rewriter, amount, intType.getWidth(), loc);
      Value value = comb::ShrUOp::create(rewriter, loc,
                                         adaptor.getInput(), amount);

      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
      // Handle 4-state index - extract just the value part
      Value idx = adaptor.getLowBit();
      if (isFourStateStructType(idx.getType()))
        idx = extractFourStateValue(rewriter, loc, idx);
      idx = adjustIntegerWidth(rewriter, idx, idxWidth, loc);

      bool isSingleElementExtract = arrType.getElementType() == resultType;

      if (isSingleElementExtract)
        rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(),
                                                    idx);
      else
        rewriter.replaceOpWithNewOp<hw::ArraySliceOp>(op, resultType,
                                                      adaptor.getInput(), idx);

      return success();
    }

    // Handle LLVM array types (arrays of LLVM types like strings)
    if (auto llvmArrTy = dyn_cast<LLVM::LLVMArrayType>(inputType)) {
      auto *ctx = rewriter.getContext();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto elemType = llvmArrTy.getElementType();

      // Store the array to get a pointer
      auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(1));
      auto alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrTy, one);
      LLVM::StoreOp::create(rewriter, loc, adaptor.getInput(), alloca);

      // Convert index to i64 for GEP
      Value idx = adaptor.getLowBit();

      // Handle 4-state index types which are lowered to {value, unknown} structs
      if (isFourStateStructType(idx.getType()))
        idx = extractFourStateValue(rewriter, loc, idx);

      if (idx.getType() != i64Ty) {
        if (cast<IntegerType>(idx.getType()).getWidth() < 64)
          idx = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
        else
          idx = arith::TruncIOp::create(rewriter, loc, i64Ty, idx);
      }

      // GEP to the element
      auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmArrTy,
                                          alloca, ValueRange{zero, idx});

      // Load the element
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);
      rewriter.replaceOp(op, elem);
      return success();
    }

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(inputType)) {
      auto structType = cast<hw::StructType>(inputType);
      auto valueType = structType.getElements()[0].type;
      auto intType = cast<IntegerType>(valueType);

      // Extract the value and unknown components
      Value inputValue = extractFourStateValue(rewriter, loc, adaptor.getInput());
      Value inputUnknown = extractFourStateUnknown(rewriter, loc, adaptor.getInput());

      // Handle 4-state index - extract just the value part
      Value amount = adaptor.getLowBit();
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);
      amount = adjustIntegerWidth(rewriter, amount, intType.getWidth(), loc);

      // Shift both value and unknown components by the same amount
      Value shiftedValue = comb::ShrUOp::create(rewriter, loc, inputValue, amount);
      Value shiftedUnknown = comb::ShrUOp::create(rewriter, loc, inputUnknown, amount);

      // Check if result is also 4-state
      if (isFourStateStructType(resultType)) {
        auto resultStructType = cast<hw::StructType>(resultType);
        auto resultValueType = resultStructType.getElements()[0].type;

        // Extract the low bits from both components
        Value extractedValue = comb::ExtractOp::create(rewriter, loc, resultValueType,
                                                        shiftedValue, 0);
        Value extractedUnknown = comb::ExtractOp::create(rewriter, loc, resultValueType,
                                                          shiftedUnknown, 0);

        // Create the result 4-state struct
        auto result = createFourStateStruct(rewriter, loc, extractedValue, extractedUnknown);
        rewriter.replaceOp(op, result);
      } else {
        // Result is 2-state - just extract the value bits
        rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, shiftedValue, 0);
      }
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

      // Handle fixed-size unpacked arrays containing LLVM types (e.g., string[N])
      // These get converted to LLVM array types, and the ref is an LLVM pointer
      if (auto arrayType = dyn_cast<UnpackedArrayType>(nestedType)) {
        auto elemMooreType = arrayType.getElementType();
        auto elemType = typeConverter->convertType(elemMooreType);

        // Convert index to i64 for GEP
        auto i64Ty = IntegerType::get(ctx, 64);
        Value idx = adaptor.getLowBit();

        // Handle 4-state index types which are lowered to {value, unknown} structs
        if (isFourStateStructType(idx.getType()))
          idx = extractFourStateValue(rewriter, loc, idx);

        if (idx.getType() != i64Ty) {
          if (cast<IntegerType>(idx.getType()).getWidth() < 64)
            idx = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
          else
            idx = arith::TruncIOp::create(rewriter, loc, i64Ty, idx);
        }

        // Create the converted array type to use as GEP base type
        auto llvmArrayType = LLVM::LLVMArrayType::get(elemType, arrayType.getSize());

        // GEP to the element - use two indices: [0][idx] to deref ptr and index
        auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmArrayType,
                                            adaptor.getInput(), ValueRange{zero, idx});

        rewriter.replaceOp(op, elemPtr);
        return success();
      }

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

        // Handle 4-state index types which are lowered to {value, unknown} structs
        if (isFourStateStructType(idx.getType()))
          idx = extractFourStateValue(rewriter, loc, idx);

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
      else if (isa<LLVM::LLVMPointerType>(convertedValueType))
        valueSize = 8; // Class handles are pointers

      // Determine the actual array handle.
      // For local variables, adaptor.getInput() IS the handle (from __moore_assoc_create).
      // For class property refs (from GEP), adaptor.getInput() is a pointer TO the
      // field that contains the handle, so we need to load first.
      // For global refs (from AddressOfOp), we also need to load.
      Value arrayHandle = adaptor.getInput();
      {
        Value source = arrayHandle;
        // Unwrap unrealized conversion casts
        while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getInputs().size() == 1)
            source = castOp.getInputs()[0];
          else
            break;
        }
        // For GEP (class property) or AddressOfOp (global), load the handle
        if (source.getDefiningOp<LLVM::GEPOp>() ||
            source.getDefiningOp<LLVM::AddressOfOp>()) {
          arrayHandle = LLVM::LoadOp::create(rewriter, loc, ptrTy, arrayHandle);
        }
      }

      // Store the key to an alloca and pass its pointer
      auto keyType = adaptor.getLowBit().getType();
      // Convert key type to pure LLVM type for LLVM operations
      Type llvmKeyType = convertToLLVMType(keyType);
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      auto keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmKeyType, one);
      // Cast the key value to LLVM type if needed (hw.struct -> llvm.struct)
      Value keyToStore = adaptor.getLowBit();
      if (llvmKeyType != keyType) {
        keyToStore = UnrealizedConversionCastOp::create(
                         rewriter, loc, llvmKeyType, ValueRange{keyToStore})
                         .getResult(0);
      }
      LLVM::StoreOp::create(rewriter, loc, keyToStore, keyAlloca);

      auto valueSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

      // Call __moore_assoc_get_ref(array, key_ptr, value_size)
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
          ValueRange{arrayHandle, keyAlloca, valueSizeConst});

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

      // Handle 4-state index - extract just the value part
      Value amount = adaptor.getLowBit();
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);
      amount = adjustIntegerWidth(rewriter, amount,
                                  llvm::Log2_64_Ceil(width), loc);
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultType, adaptor.getInput(), amount);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      // Handle 4-state index - extract just the value part
      Value idx = adaptor.getLowBit();
      if (isFourStateStructType(idx.getType()))
        idx = extractFourStateValue(rewriter, loc, idx);
      idx = adjustIntegerWidth(rewriter, idx,
                               llvm::Log2_64_Ceil(arrType.getNumElements()),
                               loc);

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

    // Handle 4-state struct types ({value, unknown} structs in llhd.ref)
    if (isFourStateStructType(inputType)) {
      auto structType = cast<hw::StructType>(inputType);
      auto valueType = structType.getElements()[0].type;
      auto intType = cast<IntegerType>(valueType);
      int64_t width = intType.getWidth();

      // Get the index, handling 4-state index types
      Value idx = adaptor.getLowBit();
      Value idxUnknownCond;
      if (isFourStateStructType(idx.getType())) {
        Value idxUnknown = extractFourStateUnknown(rewriter, loc, idx);
        Value idxUnknownZero = hw::ConstantOp::create(
            rewriter, loc, idxUnknown.getType(), 0);
        idxUnknownCond = comb::ICmpOp::create(
            rewriter, loc, comb::ICmpPredicate::ne, idxUnknown,
            idxUnknownZero);
        idx = extractFourStateValue(rewriter, loc, idx);
      }
      idx = adjustIntegerWidth(rewriter, idx, llvm::Log2_64_Ceil(width), loc);

      // Extract from both value and unknown fields via value reads.
      Value baseValue = llhd::ProbeOp::create(rewriter, loc, adaptor.getInput());
      Value valueField = extractFourStateValue(rewriter, loc, baseValue);
      Value unknownField = extractFourStateUnknown(rewriter, loc, baseValue);

      auto resultRefType = cast<llhd::RefType>(resultType);
      auto resultStructType =
          cast<hw::StructType>(resultRefType.getNestedType());
      auto resultIntType =
          cast<IntegerType>(resultStructType.getElements()[0].type);
      auto resultWidth = resultIntType.getWidth();

      Value idxShift = adjustIntegerWidth(rewriter, idx, width, loc);
      Value valueShifted =
          comb::ShrUOp::create(rewriter, loc, valueField, idxShift);
      Value unknownShifted =
          comb::ShrUOp::create(rewriter, loc, unknownField, idxShift);
      Value extractedValue = comb::ExtractOp::create(
          rewriter, loc, valueShifted, 0, resultWidth);
      Value extractedUnknown = comb::ExtractOp::create(
          rewriter, loc, unknownShifted, 0, resultWidth);

      Value outOfBoundsCond;
      int64_t maxIdx = static_cast<int64_t>(width) -
                       static_cast<int64_t>(resultWidth);
      Value constTrue =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
      if (maxIdx < 0) {
        outOfBoundsCond = constTrue;
      } else {
        Value maxIdxConst = hw::ConstantOp::create(
            rewriter, loc, idxShift.getType(), maxIdx);
        outOfBoundsCond = comb::ICmpOp::create(
            rewriter, loc, comb::ICmpPredicate::ugt, idxShift, maxIdxConst);
      }
      Value cond = outOfBoundsCond;
      if (idxUnknownCond)
        cond = comb::OrOp::create(rewriter, loc, outOfBoundsCond, idxUnknownCond);

      Value zero =
          hw::ConstantOp::create(rewriter, loc, resultIntType, 0);
      Value allOnes =
          hw::ConstantOp::create(rewriter, loc, resultIntType, -1);
      Value finalValue = comb::MuxOp::create(rewriter, loc, cond, zero,
                                             extractedValue);
      Value finalUnknown = comb::MuxOp::create(rewriter, loc, cond, allOnes,
                                               extractedUnknown);

      // Create the 4-state struct result
      auto fourStateStruct = hw::StructCreateOp::create(
          rewriter, loc, resultStructType,
          ValueRange{finalValue, finalUnknown});

      // Create a new signal to hold the extracted value and return ref to it.
      // Note: This creates a read-only view; writes won't propagate back.
      // For full read-write support, a more complex mechanism is needed.
      auto signal = llhd::SignalOp::create(rewriter, loc, resultType,
                                           StringAttr{}, fourStateStruct);
      rewriter.replaceOp(op, signal.getResult());
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

      // Convert the result type - first to HW type, then to LLVM-compatible type
      // This is needed because the LLVM struct stores LLVM-converted types
      // (e.g., 4-state hw.struct<value, unknown> becomes LLVM::LLVMStructType{i, i})
      auto hwResultType = typeConverter->convertType(op.getResult().getType());
      if (!hwResultType)
        return failure();

      // Convert HW types to LLVM types for the extraction
      auto llvmResultType = convertToLLVMType(hwResultType);

      // Use LLVM::ExtractValueOp for LLVM struct types
      Value extracted = LLVM::ExtractValueOp::create(
          rewriter, op.getLoc(), llvmResultType, adaptor.getInput(),
          ArrayRef<int64_t>{fieldIndex});

      // If the HW result type differs from LLVM type (e.g., 4-state struct),
      // cast back to HW type for downstream consumers
      if (hwResultType != llvmResultType) {
        extracted = UnrealizedConversionCastOp::create(
                        rewriter, op.getLoc(), hwResultType, extracted)
                        .getResult(0);
      }

      rewriter.replaceOp(op, extracted);
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
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value value = extractFourStateValue(rewriter, loc, input);
      Value unknown = extractFourStateUnknown(rewriter, loc, input);

      Value allOnes = hw::ConstantOp::create(rewriter, loc, value.getType(), -1);
      Value zero = hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);

      Value valueOrUnknown =
          comb::OrOp::create(rewriter, loc, value, unknown, false);
      Value anyZero = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, valueOrUnknown, allOnes);
      Value anyUnknown = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, unknown, zero);

      Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
      Value noZero = comb::XorOp::create(rewriter, loc, anyZero, one, false);
      Value noUnknown =
          comb::XorOp::create(rewriter, loc, anyUnknown, one, false);
      Value resultVal =
          comb::AndOp::create(rewriter, loc, noZero, noUnknown, false);
      Value resultUnknown =
          comb::AndOp::create(rewriter, loc, noZero, anyUnknown, false);

      rewriter.replaceOp(
          op, createFourStateStruct(rewriter, loc, resultVal, resultUnknown));
      return success();
    }

    Value max =
        hw::ConstantOp::create(rewriter, loc, input.getType(), -1);
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType,
                                              comb::ICmpPredicate::eq, input,
                                              max);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value value = extractFourStateValue(rewriter, loc, input);
      Value unknown = extractFourStateUnknown(rewriter, loc, input);

      Value zeroVal = hw::ConstantOp::create(rewriter, loc, value.getType(), 0);
      Value zeroUnk =
          hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);
      Value anyOne = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, value, zeroVal);
      Value anyUnknown = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, unknown, zeroUnk);
      Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
      Value notAnyOne = comb::XorOp::create(rewriter, loc, anyOne, one, false);
      Value resultUnknown =
          comb::AndOp::create(rewriter, loc, notAnyOne, anyUnknown, false);
      rewriter.replaceOp(
          op, createFourStateStruct(rewriter, loc, anyOne, resultUnknown));
      return success();
    }

    Value zero =
        hw::ConstantOp::create(rewriter, loc, input.getType(), 0);
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType,
                                              comb::ICmpPredicate::ne, input,
                                              zero);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value value = extractFourStateValue(rewriter, loc, input);
      Value unknown = extractFourStateUnknown(rewriter, loc, input);
      Value parity = comb::ParityOp::create(rewriter, loc, value);
      Value zero = hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);
      Value anyUnknown = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, unknown, zero);
      rewriter.replaceOp(
          op, createFourStateStruct(rewriter, loc, parity, anyUnknown));
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, input);
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value value = extractFourStateValue(rewriter, loc, input);
      Value unknown = extractFourStateUnknown(rewriter, loc, input);
      Value zeroVal = hw::ConstantOp::create(rewriter, loc, value.getType(), 0);
      Value zeroUnk =
          hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);
      Value anyOne = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, value, zeroVal);
      Value anyUnknown = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, unknown, zeroUnk);
      Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
      Value noUnknown =
          comb::XorOp::create(rewriter, loc, anyUnknown, one, false);
      Value resultVal =
          comb::AndOp::create(rewriter, loc, anyOne, noUnknown, false);
      rewriter.replaceOp(
          op, createFourStateStruct(rewriter, loc, resultVal, anyUnknown));
      return success();
    }
    if (isa_and_nonnull<IntegerType>(resultType)) {
      // Compare input to zero of the input's type, not result type.
      // Handle both integer and float input types.
      if (auto floatTy = dyn_cast<FloatType>(input.getType())) {
        // For float types, use arith::CmpFOp to compare to 0.0
        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getFloatAttr(floatTy, 0.0));
        rewriter.replaceOpWithNewOp<arith::CmpFOp>(
            op, arith::CmpFPredicate::UNE, input, zero);
        return success();
      }
      // For integer types, use comb::ICmpOp
      Value zero =
          hw::ConstantOp::create(rewriter, loc, input.getType(), 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                input, zero);
      return success();
    }
    // Handle pointer types (virtual interfaces, class handles) - compare to null.
    if (isa_and_nonnull<LLVM::LLVMPointerType>(resultType)) {
      auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      auto i1Ty = rewriter.getI1Type();
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, i1Ty, LLVM::ICmpPredicate::ne, input, nullPtr);
      return success();
    }
    if (isFourStateStructType(resultType) && isa<IntegerType>(input.getType())) {
      Value zero = hw::ConstantOp::create(rewriter, loc, input.getType(), 0);
      Value anyOne = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, input, zero);
      Value unknown =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
      rewriter.replaceOp(
          op, createFourStateStruct(rewriter, loc, anyOne, unknown));
      return success();
    }
    return failure();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple negation (0 - x)
      Value zero = hw::ConstantOp::create(rewriter, loc, resultType, 0);
      rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, adaptor.getInput());
      return success();
    }

    // 4-state negation: if any bit is unknown, entire result is X
    Value inputVal = extractFourStateValue(rewriter, loc, adaptor.getInput());
    Value inputUnk = extractFourStateUnknown(rewriter, loc, adaptor.getInput());

    // Perform negation on the value component (0 - inputVal)
    Value zero = hw::ConstantOp::create(rewriter, loc, inputVal.getType(), 0);
    Value resultVal = comb::SubOp::create(rewriter, loc, zero, inputVal, false);
    auto width = resultVal.getType().getIntOrFloatBitWidth();

    // Check if input has any unknown bits
    // hasUnknown = (inputUnk != 0)
    Value allOnes =
        hw::ConstantOp::create(rewriter, loc, rewriter.getIntegerType(width), -1);
    Value hasUnknown = comb::ICmpOp::create(rewriter, loc,
                                            comb::ICmpPredicate::ne, inputUnk, zero);

    // If any unknown: result unknown = all ones, else = 0
    Value resultUnk =
        comb::MuxOp::create(rewriter, loc, hasUnknown, allOnes, zero);

    // Create the 4-state struct
    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
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

/// 4-state aware binary arithmetic operation conversion.
/// For arithmetic operations, if any bit in either operand is unknown (X/Z),
/// the entire result becomes unknown (X).
template <typename SourceOp, typename TargetOp>
struct FourStateArithOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple binary op
      rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs(), false);
      return success();
    }

    // 4-state arithmetic: if any bit is unknown, entire result is X
    Value lhsVal = extractFourStateValue(rewriter, loc, adaptor.getLhs());
    Value lhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getLhs());
    Value rhsVal = extractFourStateValue(rewriter, loc, adaptor.getRhs());
    Value rhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getRhs());

    // Perform the arithmetic operation on the value components
    Value resultVal = TargetOp::create(rewriter, loc, lhsVal, rhsVal, false);
    auto width = resultVal.getType().getIntOrFloatBitWidth();

    // Check if either operand has any unknown bits
    // hasUnknown = (lhsUnk != 0) || (rhsUnk != 0)
    Value zero = hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
    Value allOnes =
        hw::ConstantOp::create(rewriter, loc, rewriter.getIntegerType(width), -1);
    Value lhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                           comb::ICmpPredicate::ne, lhsUnk, zero);
    Value rhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                           comb::ICmpPredicate::ne, rhsUnk, zero);
    Value hasUnknown = comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);

    // If any unknown: result unknown = all ones, else = 0
    // Using mux: resultUnk = hasUnknown ? allOnes : 0
    Value resultUnk =
        comb::MuxOp::create(rewriter, loc, hasUnknown, allOnes, zero);

    // Create the 4-state struct
    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 4-State Logic Operation Conversions
//===----------------------------------------------------------------------===//

/// Conversion for moore.and with 4-state X-propagation.
/// Implements the truth table:
///     0 1 X Z
///   +--------
/// 0 | 0 0 0 0
/// 1 | 0 1 X X
/// X | 0 X X X
/// Z | 0 X X X
struct FourStateAndOpConversion : public OpConversionPattern<AndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple AND
      rewriter.replaceOpWithNewOp<comb::AndOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs(), false);
      return success();
    }

    // 4-state AND with X-propagation
    Value lhsVal = extractFourStateValue(rewriter, loc, adaptor.getLhs());
    Value lhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getLhs());
    Value rhsVal = extractFourStateValue(rewriter, loc, adaptor.getRhs());
    Value rhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getRhs());

    // Compute definite zeros: bits that are definitely 0 in either operand
    // A bit is definitely 0 if unknown=0 and value=0
    // ~lhsVal & ~lhsUnk gives bits that are definitely 0 in lhs
    Value lhsDefZero = comb::AndOp::create(
        rewriter, loc,
        comb::XorOp::create(
            rewriter, loc, lhsVal,
            hw::ConstantOp::create(rewriter, loc, lhsVal.getType(), -1), false),
        comb::XorOp::create(
            rewriter, loc, lhsUnk,
            hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), -1), false),
        false);
    Value rhsDefZero = comb::AndOp::create(
        rewriter, loc,
        comb::XorOp::create(
            rewriter, loc, rhsVal,
            hw::ConstantOp::create(rewriter, loc, rhsVal.getType(), -1), false),
        comb::XorOp::create(
            rewriter, loc, rhsUnk,
            hw::ConstantOp::create(rewriter, loc, rhsUnk.getType(), -1), false),
        false);
    Value defZeros = comb::OrOp::create(rewriter, loc, lhsDefZero, rhsDefZero, false);

    // Result value = lhsVal & rhsVal (with Z bits treated as X, i.e., value=0)
    Value lhsValMasked =
        comb::AndOp::create(rewriter, loc, lhsVal,
                            comb::XorOp::create(rewriter, loc, lhsUnk,
                                                hw::ConstantOp::create(
                                                    rewriter, loc, lhsUnk.getType(), -1),
                                                false),
                            false);
    Value rhsValMasked =
        comb::AndOp::create(rewriter, loc, rhsVal,
                            comb::XorOp::create(rewriter, loc, rhsUnk,
                                                hw::ConstantOp::create(
                                                    rewriter, loc, rhsUnk.getType(), -1),
                                                false),
                            false);
    Value resultVal = comb::AndOp::create(rewriter, loc, lhsValMasked, rhsValMasked, false);

    // Result unknown = (lhsUnk | rhsUnk) & ~defZeros
    // Unknown bits propagate unless the result is a definite zero
    Value allUnk = comb::OrOp::create(rewriter, loc, lhsUnk, rhsUnk, false);
    Value resultUnk =
        comb::AndOp::create(rewriter, loc, allUnk,
                            comb::XorOp::create(rewriter, loc, defZeros,
                                                hw::ConstantOp::create(
                                                    rewriter, loc, defZeros.getType(), -1),
                                                false),
                            false);

    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.or with 4-state X-propagation.
/// Implements the truth table:
///     0 1 X Z
///   +--------
/// 0 | 0 1 X X
/// 1 | 1 1 1 1
/// X | X 1 X X
/// Z | X 1 X X
struct FourStateOrOpConversion : public OpConversionPattern<OrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple OR
      rewriter.replaceOpWithNewOp<comb::OrOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs(), false);
      return success();
    }

    // 4-state OR with X-propagation
    Value lhsVal = extractFourStateValue(rewriter, loc, adaptor.getLhs());
    Value lhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getLhs());
    Value rhsVal = extractFourStateValue(rewriter, loc, adaptor.getRhs());
    Value rhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getRhs());

    // Compute definite ones: bits that are definitely 1 in either operand
    // A bit is definitely 1 if unknown=0 and value=1
    Value lhsDefOne = comb::AndOp::create(
        rewriter, loc, lhsVal,
        comb::XorOp::create(
            rewriter, loc, lhsUnk,
            hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), -1), false),
        false);
    Value rhsDefOne = comb::AndOp::create(
        rewriter, loc, rhsVal,
        comb::XorOp::create(
            rewriter, loc, rhsUnk,
            hw::ConstantOp::create(rewriter, loc, rhsUnk.getType(), -1), false),
        false);
    Value defOnes = comb::OrOp::create(rewriter, loc, lhsDefOne, rhsDefOne, false);

    // Result value = lhsVal | rhsVal
    Value resultVal = comb::OrOp::create(rewriter, loc, lhsVal, rhsVal, false);

    // Result unknown = (lhsUnk | rhsUnk) & ~defOnes
    // Unknown bits propagate unless the result is a definite one
    Value allUnk = comb::OrOp::create(rewriter, loc, lhsUnk, rhsUnk, false);
    Value resultUnk =
        comb::AndOp::create(rewriter, loc, allUnk,
                            comb::XorOp::create(rewriter, loc, defOnes,
                                                hw::ConstantOp::create(
                                                    rewriter, loc, defOnes.getType(), -1),
                                                false),
                            false);

    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.xor with 4-state X-propagation.
/// Implements the truth table:
///     0 1 X Z
///   +--------
/// 0 | 0 1 X X
/// 1 | 1 0 X X
/// X | X X X X
/// Z | X X X X
struct FourStateXorOpConversion : public OpConversionPattern<XorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple XOR
      rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs(), false);
      return success();
    }

    // 4-state XOR with X-propagation
    Value lhsVal = extractFourStateValue(rewriter, loc, adaptor.getLhs());
    Value lhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getLhs());
    Value rhsVal = extractFourStateValue(rewriter, loc, adaptor.getRhs());
    Value rhsUnk = extractFourStateUnknown(rewriter, loc, adaptor.getRhs());

    // Result value = lhsVal ^ rhsVal
    Value resultVal = comb::XorOp::create(rewriter, loc, lhsVal, rhsVal, false);

    // Result unknown = lhsUnk | rhsUnk
    // Any unknown bit in either operand makes the result unknown
    Value resultUnk = comb::OrOp::create(rewriter, loc, lhsUnk, rhsUnk, false);

    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Conversion for moore.not with 4-state X-propagation.
/// Implements the truth table:
/// 0 | 1
/// 1 | 0
/// X | X
/// Z | X
struct FourStateNotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Check if we're dealing with 4-state types
    if (!isFourStateStructType(resultType)) {
      // Two-valued: simple NOT (XOR with all ones)
      Value max = hw::ConstantOp::create(rewriter, loc, resultType, -1);
      rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
      return success();
    }

    // 4-state NOT with X-propagation
    Value inputVal = extractFourStateValue(rewriter, loc, adaptor.getInput());
    Value inputUnk = extractFourStateUnknown(rewriter, loc, adaptor.getInput());

    // Result value = ~inputVal (flip all bits)
    Value resultVal = comb::XorOp::create(
        rewriter, loc, inputVal,
        hw::ConstantOp::create(rewriter, loc, inputVal.getType(), -1), false);

    // Result unknown = inputUnk (unknown bits stay unknown)
    // Note: Z bits become X (unknown stays, but value component is flipped)
    Value resultUnk = inputUnk;

    auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
    rewriter.replaceOp(op, result);
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

    // Handle 4-state types which are lowered to {value, unknown} structs
    if (isFourStateStructType(input.getType()))
      input = extractFourStateValue(rewriter, loc, input);

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

template <typename SourceOp, ICmpPredicate pred>
struct LogicalICmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isFourStateStructType(lhs.getType())) {
      Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
      Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
      Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
      Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);

      Value cmpVal =
          comb::ICmpOp::create(rewriter, loc, pred, lhsVal, rhsVal);
      Value zeroUnk =
          hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
      Value lhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, lhsUnk, zeroUnk);
      Value rhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, rhsUnk, zeroUnk);
      Value hasUnk =
          comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);
      Value zero = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                          0);
      Value resultVal = comb::MuxOp::create(rewriter, loc, hasUnk, zero, cmpVal);
      Value result =
          createFourStateStruct(rewriter, loc, resultVal, hasUnk);
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, lhs, rhs);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct CaseEqOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isFourStateStructType(lhs.getType())) {
      Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
      Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
      Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
      Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);

      Value valEq =
          comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq, lhsVal,
                               rhsVal);
      Value unkEq =
          comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq, lhsUnk,
                               rhsUnk);
      Value eqAll =
          comb::AndOp::create(rewriter, loc, valEq, unkEq, false);
      if (pred == ICmpPredicate::cne) {
        Value one = hw::ConstantOp::create(rewriter, loc,
                                           rewriter.getI1Type(), 1);
        eqAll = comb::XorOp::create(rewriter, loc, eqAll, one, false);
      }
      rewriter.replaceOp(op, eqAll);
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, lhs, rhs);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct WildcardEqOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isFourStateStructType(lhs.getType())) {
      Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
      Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
      Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
      Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);

      Value allOnes = hw::ConstantOp::create(
          rewriter, loc, rhsUnk.getType(), -1);
      Value rhsMask =
          comb::XorOp::create(rewriter, loc, rhsUnk, allOnes, false);
      Value maskedLhs =
          comb::AndOp::create(rewriter, loc, lhsVal, rhsMask, false);
      Value maskedRhs =
          comb::AndOp::create(rewriter, loc, rhsVal, rhsMask, false);

      auto cmpPred =
          pred == ICmpPredicate::wne ? ICmpPredicate::ne : ICmpPredicate::eq;
      Value cmpVal =
          comb::ICmpOp::create(rewriter, loc, cmpPred, maskedLhs, maskedRhs);

      Value zeroUnk =
          hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
      Value lhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, lhsUnk, zeroUnk);
      Value zero = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                          0);
      Value resultVal =
          comb::MuxOp::create(rewriter, loc, lhsHasUnk, zero, cmpVal);
      Value result =
          createFourStateStruct(rewriter, loc, resultVal, lhsHasUnk);
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, lhs, rhs);
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

    // Get the adapted operands - may be 4-state structs
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Handle 4-state struct types by extracting the value component
    if (isFourStateStructType(lhs.getType()))
      lhs = extractFourStateValue(rewriter, loc, lhs);
    if (isFourStateStructType(rhs.getType()))
      rhs = extractFourStateValue(rewriter, loc, rhs);

    // If we have detected any bits to be ignored, mask them in the operands for
    // the comparison.
    if (!ignoredBits.isZero()) {
      ignoredBits.flipAllBits();
      auto maskOp = hw::ConstantOp::create(rewriter, loc, ignoredBits);
      lhs = rewriter.createOrFold<comb::AndOp>(loc, lhs, maskOp);
      rhs = rewriter.createOrFold<comb::AndOp>(loc, rhs, maskOp);
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

    // Handle chandle (pointer) to integer conversions.
    // This is used in UVM for DPI-C handle conversions.
    // Check the original Moore types since the adaptor type may not be converted
    // yet for function block arguments.
    if (isa<ChandleType>(op.getInput().getType()) &&
        isa<moore::IntType>(op.getResult().getType())) {
      // The input is chandle which converts to !llvm.ptr.
      // The result is an integer type (may be 2-state plain integer or 4-state struct).
      Value ptrValue = adaptor.getInput();
      if (!isa<LLVM::LLVMPointerType>(ptrValue.getType())) {
        ptrValue = rewriter.create<UnrealizedConversionCastOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
            ptrValue).getResult(0);
      }

      // Get the integer width from the Moore result type
      auto mooreResultType = cast<moore::IntType>(op.getResult().getType());
      int64_t width = mooreResultType.getWidth();
      Type intType = rewriter.getIntegerType(width);

      // Convert pointer to integer
      Value intResult = LLVM::PtrToIntOp::create(rewriter, loc, intType, ptrValue);

      // If the result is a 4-state type, wrap it in a struct
      if (isFourStateStructType(resultType)) {
        Value zero = hw::ConstantOp::create(rewriter, loc, intType, 0);
        intResult = createFourStateStruct(rewriter, loc, intResult, zero);
      }

      rewriter.replaceOp(op, intResult);
      return success();
    }

    // Handle integer to chandle (pointer) conversions.
    if (isa<ChandleType>(op.getResult().getType()) &&
        isa<moore::IntType>(op.getInput().getType())) {
      Value intValue = adaptor.getInput();

      // If input is a 4-state struct, extract the value part
      if (isFourStateStructType(intValue.getType())) {
        intValue = extractFourStateValue(rewriter, loc, intValue);
      }

      // Ensure we have a plain integer type
      if (!isa<IntegerType>(intValue.getType())) {
        int64_t width = cast<moore::IntType>(op.getInput().getType()).getWidth();
        intValue = rewriter.create<UnrealizedConversionCastOp>(
            loc, rewriter.getIntegerType(width), intValue).getResult(0);
      }

      Value result = LLVM::IntToPtrOp::create(rewriter, loc, resultType, intValue);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Handle class handle (including null) to integer conversions.
    // This is used when comparing chandle with null (null is class<@__null__>).
    if (isa<ClassHandleType>(op.getInput().getType()) &&
        isa<moore::IntType>(op.getResult().getType())) {
      // Class handles convert to !llvm.ptr.
      Value ptrValue = adaptor.getInput();
      if (!isa<LLVM::LLVMPointerType>(ptrValue.getType())) {
        ptrValue = rewriter.create<UnrealizedConversionCastOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
            ptrValue).getResult(0);
      }

      // Get the integer width from the Moore result type
      auto mooreResultType = cast<moore::IntType>(op.getResult().getType());
      int64_t width = mooreResultType.getWidth();
      Type intType = rewriter.getIntegerType(width);

      // Convert pointer to integer
      Value intResult = LLVM::PtrToIntOp::create(rewriter, loc, intType, ptrValue);

      // If the result is a 4-state type, wrap it in a struct
      if (isFourStateStructType(resultType)) {
        Value zero = hw::ConstantOp::create(rewriter, loc, intType, 0);
        intResult = createFourStateStruct(rewriter, loc, intResult, zero);
      }

      rewriter.replaceOp(op, intResult);
      return success();
    }

    // Handle open_uarray <-> queue conversions.
    // Both types convert to the same LLVM struct {ptr, i64}, so this is a no-op.
    // This is used in UVM for dynamic array operations.
    if ((isa<OpenUnpackedArrayType>(op.getInput().getType()) &&
         isa<QueueType>(op.getResult().getType())) ||
        (isa<QueueType>(op.getInput().getType()) &&
         isa<OpenUnpackedArrayType>(op.getResult().getType()))) {
      // Both types lower to the same struct, so just pass through
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Handle queue to queue conversions with different bounds.
    // Both convert to the same LLVM struct {ptr, i64}, so this is a no-op.
    // This is used when slicing queues (bounded to unbounded, or different bounds).
    if (isa<QueueType>(op.getInput().getType()) &&
        isa<QueueType>(op.getResult().getType())) {
      // Both types lower to the same struct, so just pass through
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Handle open_uarray to open_uarray conversions (different element types
    // but same representation). Both convert to the same LLVM struct {ptr, i64}.
    if (isa<OpenUnpackedArrayType>(op.getInput().getType()) &&
        isa<OpenUnpackedArrayType>(op.getResult().getType())) {
      // Both types lower to the same struct, so just pass through
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Handle fixed-size array to open_uarray conversion.
    // This converts a fixed-size unpacked array (uarray<N x T>) to a dynamic
    // array (open_uarray<T>). This is a legal SystemVerilog conversion.
    if (isa<UnpackedArrayType>(op.getInput().getType()) &&
        isa<OpenUnpackedArrayType>(op.getResult().getType())) {
      auto *ctx = rewriter.getContext();
      auto inputArrayType = cast<UnpackedArrayType>(op.getInput().getType());
      auto resultDynArrayType =
          cast<OpenUnpackedArrayType>(op.getResult().getType());

      // Get array size and element types
      int64_t numElements = inputArrayType.getSize();
      Type mooreElemType = resultDynArrayType.getElementType();
      Type llvmElemType = typeConverter->convertType(mooreElemType);
      if (!llvmElemType)
        return failure();

      // Setup LLVM types
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto dynArrayTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

      // Calculate element size in bytes
      int64_t elemBitWidth = 1;
      if (auto elemIntType = dyn_cast<moore::IntType>(mooreElemType))
        elemBitWidth = elemIntType.getWidth();
      else
        elemBitWidth = hw::getBitWidth(llvmElemType);
      if (elemBitWidth <= 0)
        elemBitWidth = 8; // Default to 1 byte
      int64_t elemByteSize = (elemBitWidth + 7) / 8;
      if (elemByteSize < 1)
        elemByteSize = 1;

      // Allocate memory for the array elements
      auto totalSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(numElements * elemByteSize));
      auto mallocFnTy = LLVM::LLVMFunctionType::get(ptrTy, {i64Ty});
      ModuleOp mod = op->getParentOfType<ModuleOp>();
      auto mallocFn =
          getOrCreateRuntimeFunc(mod, rewriter, "malloc", mallocFnTy);
      auto mallocCall = LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                                             SymbolRefAttr::get(mallocFn),
                                             ValueRange{totalSize});
      Value arrayPtr = mallocCall.getResult();

      // Copy elements from the fixed-size array to the allocated memory
      Value inputArray = adaptor.getInput();

      // Check if the input is valid (operand may not be remapped yet)
      if (!inputArray)
        return failure();

      // Get the input array type (could be hw::ArrayType or LLVM::LLVMArrayType)
      Type inputLLVMType = inputArray.getType();

      // Determine the index width needed for hw::ArrayGetOp
      unsigned idxWidth = llvm::Log2_64_Ceil(numElements);
      if (idxWidth == 0)
        idxWidth = 1;

      for (int64_t i = 0; i < numElements; ++i) {
        // Extract element from fixed-size array
        Value elemValue;
        if (isa<hw::ArrayType>(inputLLVMType)) {
          // Create index value for hw::ArrayGetOp
          Value idxValue = hw::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerType(idxWidth), i);
          elemValue =
              hw::ArrayGetOp::create(rewriter, loc, inputArray, idxValue);
        } else if (isa<LLVM::LLVMArrayType>(inputLLVMType)) {
          elemValue = LLVM::ExtractValueOp::create(rewriter, loc, inputArray,
                                                   ArrayRef<int64_t>{i});
        } else {
          return failure();
        }

        // Compute pointer to destination element
        auto idxConst = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(i));
        Value destPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmElemType, arrayPtr,
                                ValueRange{idxConst});

        // Store element
        LLVM::StoreOp::create(rewriter, loc, elemValue, destPtr);
      }

      // Create the result struct {ptr, i64}
      auto sizeConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(numElements));
      Value result = LLVM::UndefOp::create(rewriter, loc, dynArrayTy);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, arrayPtr,
                                           ArrayRef<int64_t>{0});
      result = LLVM::InsertValueOp::create(rewriter, loc, result, sizeConst,
                                           ArrayRef<int64_t>{1});

      rewriter.replaceOp(op, result);
      return success();
    }

    // Handle integer to queue/open_uarray conversion (bit unpacking).
    // This converts an N-bit integer to a queue or dynamic array of M-bit elements.
    // Used in streaming concatenation for unpacking bits into queue/array elements.
    if (isa<moore::IntType, moore::UnpackedStructType>(op.getInput().getType()) &&
        isa<QueueType, OpenUnpackedArrayType>(op.getResult().getType())) {
      auto *ctx = rewriter.getContext();
      ModuleOp mod = op->getParentOfType<ModuleOp>();
      Type elemType;
      if (auto queueType = dyn_cast<QueueType>(op.getResult().getType()))
        elemType = queueType.getElementType();
      else
        elemType = cast<OpenUnpackedArrayType>(op.getResult().getType())
                       .getElementType();

      // Get the bit widths
      int64_t inputBitWidth = -1;
      if (auto intType = dyn_cast<moore::IntType>(op.getInput().getType()))
        inputBitWidth = intType.getWidth();
      else
        inputBitWidth = hw::getBitWidth(adaptor.getInput().getType());

      int64_t elemBitWidth = 1;
      if (auto elemIntType = dyn_cast<moore::IntType>(elemType))
        elemBitWidth = elemIntType.getWidth();

      if (inputBitWidth <= 0 || elemBitWidth <= 0)
        return failure();

      // Calculate number of elements
      int64_t numElements = inputBitWidth / elemBitWidth;
      if (numElements <= 0)
        numElements = 1;

      // Convert the element type for LLVM
      Type llvmElemType = typeConverter->convertType(elemType);
      if (!llvmElemType)
        return failure();

      // Setup LLVM types
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto queueTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
      auto voidTy = LLVM::LLVMVoidType::get(ctx);

      // Create function type for push_back: void(queue_ptr, element_ptr, element_size)
      auto pushBackFnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
      auto pushBackFn = getOrCreateRuntimeFunc(mod, rewriter,
                                               "__moore_queue_push_back", pushBackFnTy);

      // Constants
      auto one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));

      // Create an alloca for the queue struct
      auto queueAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);

      // Initialize queue to {nullptr, 0}
      Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      Value zeroLen = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));
      Value emptyQueue = LLVM::UndefOp::create(rewriter, loc, queueTy);
      emptyQueue = LLVM::InsertValueOp::create(rewriter, loc, emptyQueue, nullPtr,
                                               ArrayRef<int64_t>{0});
      emptyQueue = LLVM::InsertValueOp::create(rewriter, loc, emptyQueue, zeroLen,
                                               ArrayRef<int64_t>{1});
      LLVM::StoreOp::create(rewriter, loc, emptyQueue, queueAlloca);

      // Get the input value - extract from 4-state struct if needed
      Value inputValue = adaptor.getInput();
      if (isFourStateStructType(inputValue.getType())) {
        inputValue = extractFourStateValue(rewriter, loc, inputValue);
      }

      // Ensure input is an integer type
      if (!isa<IntegerType>(inputValue.getType())) {
        inputValue = rewriter.createOrFold<hw::BitcastOp>(
            loc, rewriter.getIntegerType(inputBitWidth), inputValue);
      }

      // Calculate element size in bytes
      int64_t elemByteSize = (elemBitWidth + 7) / 8;
      if (elemByteSize < 1)
        elemByteSize = 1;
      auto elemSize = LLVM::ConstantOp::create(rewriter, loc,
                                               rewriter.getI64IntegerAttr(elemByteSize));

      // Create an alloca for the element
      auto elemAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);

      // For each element, extract bits and push to queue
      // We extract from LSB to MSB to maintain bit order
      for (int64_t i = 0; i < numElements; ++i) {
        Value elemValue;
        if (elemBitWidth == inputBitWidth) {
          // Single element case - use the whole input
          elemValue = inputValue;
        } else if (inputBitWidth < elemBitWidth) {
          // Input is narrower than element width - zero extend
          // E.g., i1 -> queue<i8>: the single i1 bit becomes an i8 element
          if (i == 0) {
            elemValue = comb::ConcatOp::create(
                rewriter, loc,
                hw::ConstantOp::create(rewriter, loc,
                    rewriter.getIntegerType(elemBitWidth - inputBitWidth), 0),
                inputValue);
          } else {
            // All elements beyond the first are zero
            elemValue = hw::ConstantOp::create(rewriter, loc,
                rewriter.getIntegerType(elemBitWidth), 0);
          }
        } else {
          // Extract bits [i*elemBitWidth, (i+1)*elemBitWidth)
          int64_t lowBit = i * elemBitWidth;
          auto lowBitConst = hw::ConstantOp::create(rewriter, loc,
              rewriter.getIntegerType(inputBitWidth), lowBit);
          Value shifted = comb::ShrUOp::create(rewriter, loc, inputValue, lowBitConst);
          elemValue = comb::ExtractOp::create(rewriter, loc,
              rewriter.getIntegerType(elemBitWidth), shifted, 0);
        }

        // If the element type is a 4-state struct, wrap it
        if (isFourStateStructType(llvmElemType)) {
          Value zero = hw::ConstantOp::create(rewriter, loc,
              rewriter.getIntegerType(elemBitWidth), 0);
          elemValue = createFourStateStruct(rewriter, loc, elemValue, zero);
        } else if (elemValue.getType() != llvmElemType) {
          // Bitcast to the correct element type
          elemValue = rewriter.createOrFold<hw::BitcastOp>(loc, llvmElemType, elemValue);
        }

        // Store element and call push_back
        LLVM::StoreOp::create(rewriter, loc, elemValue, elemAlloca);
        LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(pushBackFn),
                             ValueRange{queueAlloca, elemAlloca, elemSize});
      }

      // Load and return the final queue
      Value result = LLVM::LoadOp::create(rewriter, loc, queueTy, queueAlloca);
      rewriter.replaceOp(op, result);
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

    // Handle ref-to-ref type conversions (e.g., ref<uarray<16 x l1>> to
    // ref<l16>). This is used in streaming concatenation operations where
    // arrays are reinterpreted as integers of the same bit width.
    // Check the original Moore types since both input and result should be
    // RefType for this case.
    auto mooreInputRefType = dyn_cast<moore::RefType>(op.getInput().getType());
    auto mooreResultRefType = dyn_cast<moore::RefType>(op.getResult().getType());
    if (mooreInputRefType && mooreResultRefType) {
      // Convert the input type through the type converter
      Type inputType = typeConverter->convertType(op.getInput().getType());
      if (!inputType) {
        op.emitError("conversion input type is not currently supported");
        return failure();
      }
      auto inputRefType = dyn_cast<llhd::RefType>(inputType);
      auto resultRefType = dyn_cast<llhd::RefType>(resultType);
      if (inputRefType && resultRefType) {
        Type inputNestedType = inputRefType.getNestedType();
        Type resultNestedType = resultRefType.getNestedType();

        // Get bit width, supporting both hw types and float types
        auto getBitWidthForType = [](Type type) -> int64_t {
          // Try hw::getBitWidth first for hw types
          int64_t bw = hw::getBitWidth(type);
          if (bw != -1)
            return bw;
          // Handle float types
          if (auto floatTy = dyn_cast<FloatType>(type))
            return floatTy.getWidth();
          return -1;
        };

        int64_t inputBw = getBitWidthForType(inputNestedType);
        int64_t resultBw = getBitWidthForType(resultNestedType);

        // Both nested types must have known bit widths
        if (inputBw == -1 || resultBw == -1)
          return failure();

        // Probe the input reference to get the value
        Value probedValue = llhd::ProbeOp::create(rewriter, loc, adaptor.getInput());

        // For float input types, first bitcast to integer
        if (isa<FloatType>(inputNestedType)) {
          probedValue = rewriter.createOrFold<LLVM::BitcastOp>(
              loc, rewriter.getIntegerType(inputBw), probedValue);
        } else {
          probedValue = rewriter.createOrFold<hw::BitcastOp>(
              loc, rewriter.getIntegerType(inputBw), probedValue);
        }

        Value adjustedValue = adjustIntegerWidth(rewriter, probedValue, resultBw, loc);

        // For float result types, bitcast from integer
        Value resultValue;
        if (isa<FloatType>(resultNestedType)) {
          resultValue = rewriter.createOrFold<LLVM::BitcastOp>(
              loc, resultNestedType, adjustedValue);
        } else {
          resultValue = rewriter.createOrFold<hw::BitcastOp>(
              loc, resultNestedType, adjustedValue);
        }

        // Create a new signal with the converted value
        auto signal =
            llhd::SignalOp::create(rewriter, loc, resultRefType, StringAttr{}, resultValue);
        rewriter.replaceOp(op, signal.getResult());
        return success();
      }
    }

    // Handle value-to-ref conversions by creating a new signal with the input
    // value as the initializer.
    if (auto resultRefType = dyn_cast<llhd::RefType>(resultType)) {
      if (!isa<llhd::RefType>(adaptor.getInput().getType())) {
        Value initValue = adaptor.getInput();
        Type nestedType = resultRefType.getNestedType();
        if (initValue.getType() != nestedType) {
          auto getBitWidthForType = [](Type type) -> int64_t {
            int64_t bw = hw::getBitWidth(type);
            if (bw != -1)
              return bw;
            if (auto floatTy = dyn_cast<FloatType>(type))
              return floatTy.getWidth();
            return -1;
          };

          int64_t inputBw = getBitWidthForType(initValue.getType());
          int64_t resultBw = getBitWidthForType(nestedType);
          if (inputBw == -1 || resultBw == -1)
            return failure();

          if (isa<FloatType>(initValue.getType())) {
            initValue = rewriter.createOrFold<LLVM::BitcastOp>(
                loc, rewriter.getIntegerType(inputBw), initValue);
          } else {
            initValue = rewriter.createOrFold<hw::BitcastOp>(
                loc, rewriter.getIntegerType(inputBw), initValue);
          }

          Value adjusted =
              adjustIntegerWidth(rewriter, initValue, resultBw, loc);

          if (isa<FloatType>(nestedType)) {
            initValue = rewriter.createOrFold<LLVM::BitcastOp>(
                loc, nestedType, adjusted);
          } else {
            initValue = rewriter.createOrFold<hw::BitcastOp>(
                loc, nestedType, adjusted);
          }
        }

        auto signal = llhd::SignalOp::create(rewriter, loc, resultRefType,
                                             StringAttr{}, initValue);
        rewriter.replaceOp(op, signal.getResult());
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

struct ConcatRefOpConversion : public OpConversionPattern<ConcatRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return op.emitError("moore.concat_ref must be lowered before MooreToCore");
    rewriter.eraseOp(op);
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
    Value input = adaptor.getInput();
    Type inputType = input.getType();

    // If types already match, just forward the value.
    if (type == inputType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // Special handling for conversions between 4-state types and integers.
    // Time is 64-bit (i64) but l64 (4-state) is a struct with 128 bits total.
    // We need to extract/wrap the value component instead of bitcasting.
    if (isFourStateStructType(inputType) && isa<IntegerType>(type)) {
      // 4-state struct -> integer: extract the value component
      Value result = extractFourStateValue(rewriter, op.getLoc(), input);
      rewriter.replaceOp(op, result);
      return success();
    }
    if (isa<IntegerType>(inputType) && isFourStateStructType(type)) {
      // integer -> 4-state struct: wrap in struct with unknown=0
      Location loc = op.getLoc();
      auto intType = cast<IntegerType>(inputType);
      Value zero = hw::ConstantOp::create(rewriter, loc, intType, 0);
      auto structType = cast<hw::StructType>(type);
      Value result = hw::StructCreateOp::create(rewriter, loc, structType,
                                                ValueRange{input, zero});
      rewriter.replaceOp(op, result);
      return success();
    }

    // Handle 4-state struct to union bitcast.
    // When the source is a 4-state struct and destination is a union with
    // 4-state members, we need to extract the value component first to avoid
    // bitwidth mismatch (the union has double bitwidth due to 4-state members).
    if (isFourStateStructType(inputType) && isa<hw::UnionType>(type)) {
      Value valueComponent = extractFourStateValue(rewriter, op.getLoc(), input);
      rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, type, valueComponent);
      return success();
    }

    // Otherwise use bitcast.
    rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, type, input);
    return success();
  }
};

/// Conversion for moore.logic_to_int: converts 4-state logic to 2-state int.
/// When the input is a 4-state struct {value, unknown}, extract the value field.
struct LogicToIntOpConversion : public OpConversionPattern<LogicToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LogicToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getInput();

    // If input is a 4-state struct, extract the value field
    if (isFourStateStructType(input.getType())) {
      input = extractFourStateValue(rewriter, op.getLoc(), input);
    }

    // If types now match, just replace; otherwise bitcast
    if (resultType == input.getType())
      rewriter.replaceOp(op, input);
    else
      rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, resultType, input);
    return success();
  }
};

/// Conversion for moore.int_to_logic: converts 2-state int to 4-state logic.
/// When the result should be a 4-state struct, wrap in struct with unknown=0.
struct IntToLogicOpConversion : public OpConversionPattern<IntToLogicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntToLogicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());
    Value input = adaptor.getInput();

    // If result should be a 4-state struct, create it
    if (isFourStateStructType(resultType)) {
      // Ensure input is the right integer type for the struct
      auto structType = cast<hw::StructType>(resultType);
      auto valueType = structType.getElements()[0].type;
      if (input.getType() != valueType) {
        input = rewriter.createOrFold<hw::BitcastOp>(loc, valueType, input);
      }
      // Create struct with unknown=0 (2-state values have no X/Z)
      Value zero = hw::ConstantOp::create(rewriter, loc, valueType, 0);
      auto result = createFourStateStruct(rewriter, loc, input, zero);
      rewriter.replaceOp(op, result);
    } else {
      // No 4-state struct needed; just bitcast if types differ
      if (resultType == input.getType())
        rewriter.replaceOp(op, input);
      else
        rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, resultType, input);
    }
    return success();
  }
};

struct UnrealizedCastToBoolConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 1 || op.getNumResults() != 1)
      return failure();
    auto input = adaptor.getOperands()[0];
    auto resultType = op.getResult(0).getType();
    if (!resultType.isInteger(1))
      return failure();
    if (!isFourStateStructType(input.getType()))
      return failure();
    auto structType = cast<hw::StructType>(input.getType());
    auto valueType = structType.getElements()[0].type;
    if (!valueType.isInteger(1))
      return failure();
    Value value = extractFourStateValue(rewriter, op.getLoc(), input);
    Value unknown = extractFourStateUnknown(rewriter, op.getLoc(), input);
    Value trueConst =
        hw::ConstantOp::create(rewriter, op.getLoc(), valueType, 1);
    Value notUnknown =
        comb::XorOp::create(rewriter, op.getLoc(), unknown, trueConst);
    Value result = comb::AndOp::create(
        rewriter, op.getLoc(), ValueRange{value, notUnknown}, true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct UnrealizedCastToBoolRewrite
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();
    auto input = op.getOperand(0);
    auto resultType = op.getResult(0).getType();
    if (!resultType.isInteger(1))
      return failure();
    if (!isFourStateStructType(input.getType()))
      return failure();
    auto structType = cast<hw::StructType>(input.getType());
    auto valueType = structType.getElements()[0].type;
    if (!valueType.isInteger(1))
      return failure();
    Value value = extractFourStateValue(rewriter, op.getLoc(), input);
    Value unknown = extractFourStateUnknown(rewriter, op.getLoc(), input);
    Value trueConst =
        hw::ConstantOp::create(rewriter, op.getLoc(), valueType, 1);
    Value notUnknown =
        comb::XorOp::create(rewriter, op.getLoc(), unknown, trueConst);
    Value result = comb::AndOp::create(
        rewriter, op.getLoc(), ValueRange{value, notUnknown}, true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ToBuiltinBoolOpConversion : public OpConversionPattern<ToBuiltinBoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToBuiltinBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto inputType = input.getType();
    if (inputType.isInteger(1)) {
      rewriter.replaceOp(op, input);
      return success();
    }
    if (isFourStateStructType(inputType)) {
      auto loc = op.getLoc();
      Value value = extractFourStateValue(rewriter, loc, input);
      Value unknown = extractFourStateUnknown(rewriter, loc, input);
      Value zero =
          hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);
      Value unknownIsZero = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::eq, unknown, zero);
      Value result = comb::AndOp::create(
          rewriter, loc, ValueRange{value, unknownIsZero}, true);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct TruncOpConversion : public OpConversionPattern<TruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = typeConverter->convertType(op.getType());
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value inputValue = extractFourStateValue(rewriter, loc, input);
      Value inputUnknown = extractFourStateUnknown(rewriter, loc, input);

      if (isFourStateStructType(type)) {
        auto resultStructType = cast<hw::StructType>(type);
        auto resultValueType = resultStructType.getElements()[0].type;
        auto resultWidth = cast<IntegerType>(resultValueType).getWidth();

        Value truncValue = comb::ExtractOp::create(rewriter, loc, inputValue, 0,
                                                   resultWidth);
        Value truncUnknown = comb::ExtractOp::create(
            rewriter, loc, inputUnknown, 0, resultWidth);
        auto result =
            createFourStateStruct(rewriter, loc, truncValue, truncUnknown);
        rewriter.replaceOp(op, result);
      } else {
        auto resultWidth = cast<IntegerType>(type).getWidth();
        Value truncValue = comb::ExtractOp::create(rewriter, loc, inputValue, 0,
                                                   resultWidth);
        rewriter.replaceOp(op, truncValue);
      }
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(
        op, input, 0, cast<IntegerType>(type).getWidth());
    return success();
  }
};

struct ZExtOpConversion : public OpConversionPattern<ZExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = typeConverter->convertType(op.getType());
    Value input = adaptor.getInput();

    if (isFourStateStructType(input.getType())) {
      Value inputValue = extractFourStateValue(rewriter, loc, input);
      Value inputUnknown = extractFourStateUnknown(rewriter, loc, input);

      if (isFourStateStructType(type)) {
        auto resultStructType = cast<hw::StructType>(type);
        auto resultValueType = resultStructType.getElements()[0].type;
        auto resultWidth = cast<IntegerType>(resultValueType).getWidth();

        Value extValue = comb::createZExt(rewriter, loc, inputValue,
                                          resultWidth);
        Value extUnknown = comb::createZExt(rewriter, loc, inputUnknown,
                                            resultWidth);
        auto result =
            createFourStateStruct(rewriter, loc, extValue, extUnknown);
        rewriter.replaceOp(op, result);
      } else {
        auto resultWidth = cast<IntegerType>(type).getWidth();
        Value extValue =
            comb::createZExt(rewriter, loc, inputValue, resultWidth);
        rewriter.replaceOp(op, extValue);
      }
      return success();
    }

    auto resultWidth = cast<IntegerType>(type).getWidth();
    Value extValue = comb::createZExt(rewriter, loc, input, resultWidth);
    rewriter.replaceOp(op, extValue);
    return success();
  }
};

struct SExtOpConversion : public OpConversionPattern<SExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = typeConverter->convertType(op.getType());
    Value input = adaptor.getInput();

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(input.getType())) {
      // Extract value and unknown components
      Value inputValue = extractFourStateValue(rewriter, loc, input);
      Value inputUnknown = extractFourStateUnknown(rewriter, loc, input);

      if (isFourStateStructType(type)) {
        auto resultStructType = cast<hw::StructType>(type);
        auto resultValueType = resultStructType.getElements()[0].type;

        // Sign extend the value component
        Value extValue = comb::createOrFoldSExt(loc, inputValue, resultValueType, rewriter);

        // Zero extend the unknown mask (sign extension doesn't propagate unknown bits)
        Value extUnknown;
        auto unknownWidth = cast<IntegerType>(inputUnknown.getType()).getWidth();
        auto resultWidth = cast<IntegerType>(resultValueType).getWidth();
        if (unknownWidth < resultWidth) {
          Value zero = hw::ConstantOp::create(rewriter, loc,
              rewriter.getIntegerType(resultWidth - unknownWidth), 0);
          extUnknown = comb::ConcatOp::create(rewriter, loc, ValueRange{zero, inputUnknown});
        } else {
          extUnknown = inputUnknown;
        }

        auto result = createFourStateStruct(rewriter, loc, extValue, extUnknown);
        rewriter.replaceOp(op, result);
      } else {
        // Result is 2-state, just sign extend the value
        auto value = comb::createOrFoldSExt(loc, inputValue, type, rewriter);
        rewriter.replaceOp(op, value);
      }
      return success();
    }

    auto value = comb::createOrFoldSExt(loc, input, type, rewriter);
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

struct ConvertRealOpConversion : public OpConversionPattern<ConvertRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConvertRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Type inputType = adaptor.getInput().getType();
    if (inputType == resultType) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    auto inputFloat = dyn_cast<FloatType>(inputType);
    auto resultFloat = dyn_cast<FloatType>(resultType);
    if (!inputFloat || !resultFloat)
      return failure();

    if (inputFloat.getWidth() < resultFloat.getWidth()) {
      rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, resultType,
                                                 adaptor.getInput());
      return success();
    }

    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultType,
                                                 adaptor.getInput());
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
    auto loc = op.getLoc();
    StringRef callee = op.getCallee();

    // Intercept UVM report function calls and redirect to runtime functions.
    // There are two variants:
    // 1. Free functions (5 args): uvm_pkg::uvm_report_xxx(id, msg, verbosity, filename, line)
    // 2. Class methods (6 args): uvm_pkg::uvm_report_object::uvm_report_xxx(self, id, msg, verbosity, filename, line)
    // The runtime functions have this signature:
    //   __moore_uvm_report_xxx(id_ptr, id_len, msg_ptr, msg_len, verbosity,
    //                          filename_ptr, filename_len, line,
    //                          context_ptr, context_len)
    if (callee == "uvm_pkg::uvm_report_error" ||
        callee == "uvm_pkg::uvm_report_warning" ||
        callee == "uvm_pkg::uvm_report_info" ||
        callee == "uvm_pkg::uvm_report_fatal") {
      if (succeeded(convertUvmReportCall(op, adaptor, rewriter, callee, /*isMethod=*/false)))
        return success();
      // Fall through to default handling if interception fails
    }
    // Handle class method versions
    if (callee == "uvm_pkg::uvm_report_object::uvm_report_error" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_warning" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_info" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_fatal") {
      if (succeeded(convertUvmReportCall(op, adaptor, rewriter, callee, /*isMethod=*/true)))
        return success();
      // Fall through to default handling if interception fails
    }

    // Default handling for other calls
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());
    return success();
  }

private:
  /// Convert a UVM report function call to a runtime function call.
  /// UVM signature: (id, message, verbosity, filename, line, context_name, report_enabled_checked)
  /// For class methods, there's an additional 'self' parameter at the start.
  /// where strings are !llvm.struct<(ptr, i64)>
  LogicalResult convertUvmReportCall(func::CallOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      StringRef callee, bool isMethod) const {
    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    auto operands = adaptor.getOperands();

    // Verify we have the expected number of operands
    // Full UVM signature: (id, message, verbosity, filename, line, context_name, report_enabled_checked)
    // Minimal UVM signature: (id, message, verbosity, filename, line)
    // For methods, there's an additional 'self' parameter at the start.
    // Our stubs use the minimal 5-arg signature, while real UVM may use 7 args.
    size_t minOperands = isMethod ? 6 : 5;
    size_t maxOperands = isMethod ? 8 : 7;
    if (operands.size() < minOperands || operands.size() > maxOperands)
      return failure();

    auto ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Determine the runtime function name based on the callee
    std::string runtimeFuncName;
    if (callee.contains("report_error"))
      runtimeFuncName = "__moore_uvm_report_error";
    else if (callee.contains("report_warning"))
      runtimeFuncName = "__moore_uvm_report_warning";
    else if (callee.contains("report_info"))
      runtimeFuncName = "__moore_uvm_report_info";
    else if (callee.contains("report_fatal"))
      runtimeFuncName = "__moore_uvm_report_fatal";
    else
      return failure();

    // Runtime function signature:
    // void __moore_uvm_report_xxx(
    //   const char *id, int64_t idLen,
    //   const char *message, int64_t messageLen,
    //   int32_t verbosity,
    //   const char *filename, int64_t filenameLen,
    //   int32_t line,
    //   const char *context, int64_t contextLen)
    auto fnTy = LLVM::LLVMFunctionType::get(
        voidTy, {ptrTy, i64Ty, ptrTy, i64Ty, i32Ty, ptrTy, i64Ty, i32Ty, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, runtimeFuncName, fnTy);

    // Extract ptr and len from each string struct
    // For free functions (5-7 args):
    //   operands[0] = id (!llvm.struct<(ptr, i64)>)
    //   operands[1] = msg (!llvm.struct<(ptr, i64)>)
    //   operands[2] = verbosity (i32)
    //   operands[3] = filename (!llvm.struct<(ptr, i64)>)
    //   operands[4] = line (i32)
    //   operands[5] = context_name (!llvm.struct<(ptr, i64)>) - optional
    //   operands[6] = report_enabled_checked (i1) - optional, ignored
    // For class methods (6-8 args):
    //   operands[0] = self (ptr) - ignored
    //   operands[1..] = same as above
    size_t offset = isMethod ? 1 : 0;
    Value idStruct = operands[offset + 0];
    Value msgStruct = operands[offset + 1];
    Value verbosity = operands[offset + 2];
    Value filenameStruct = operands[offset + 3];
    Value line = operands[offset + 4];

    // Check if context_name is provided (6+ args for free funcs, 7+ for methods)
    bool hasContext = operands.size() > offset + 5;

    // Extract id ptr and len
    Value idPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, idStruct,
                                                ArrayRef<int64_t>{0});
    Value idLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, idStruct,
                                                ArrayRef<int64_t>{1});

    // Extract msg ptr and len
    Value msgPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, msgStruct,
                                                 ArrayRef<int64_t>{0});
    Value msgLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, msgStruct,
                                                 ArrayRef<int64_t>{1});

    // Extract filename ptr and len
    Value filenamePtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, filenameStruct,
                                                      ArrayRef<int64_t>{0});
    Value filenameLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, filenameStruct,
                                                      ArrayRef<int64_t>{1});

    // Extract context ptr and len, or use null/zero defaults if not provided
    Value contextPtr, contextLen;
    if (hasContext) {
      Value contextStruct = operands[offset + 5];
      contextPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, contextStruct,
                                                 ArrayRef<int64_t>{0});
      contextLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, contextStruct,
                                                 ArrayRef<int64_t>{1});
    } else {
      // Provide default empty context (null pointer, zero length)
      contextPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      contextLen = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    }

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{idPtr, idLen, msgPtr, msgLen, verbosity,
                                    filenamePtr, filenameLen, line,
                                    contextPtr, contextLen});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for func.call_indirect to handle type conversion.
/// The callee function type must be converted to use the converted argument
/// and result types. Also intercepts UVM report method calls via vtable dispatch.
struct CallIndirectOpConversion
    : public OpConversionPattern<func::CallIndirectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if this is a UVM report method call via vtable dispatch.
    // Trace back through the def-use chain to find the VTableLoadMethodOp.
    Value origCallee = op.getCallee();

    // Look through UnrealizedConversionCast ops to find the source
    while (auto castOp = origCallee.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1)
        origCallee = castOp.getInputs()[0];
      else
        break;
    }

    // Check if the callee comes from a VTableLoadMethodOp
    if (auto vtableLoadOp = origCallee.getDefiningOp<VTableLoadMethodOp>()) {
      auto methodSym = vtableLoadOp.getMethodSym();
      StringRef methodName = methodSym.getLeafReference();

      // Check if this is a UVM report method
      if (methodName == "uvm_report_info" || methodName == "uvm_report_warning" ||
          methodName == "uvm_report_error" || methodName == "uvm_report_fatal") {
        // Intercept this call and convert to runtime function
        if (succeeded(convertUvmReportVtableCall(op, adaptor, rewriter, methodName)))
          return success();
        // Fall through to default handling if interception fails
      }
    }

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

private:
  /// Convert a UVM report method call via vtable dispatch to a runtime function.
  /// The call_indirect has signature: (self, id, message, verbosity, filename, line)
  /// where strings are !llvm.struct<(ptr, i64)>
  LogicalResult convertUvmReportVtableCall(func::CallIndirectOp op,
                                            OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef methodName) const {
    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    auto calleeOperands = adaptor.getCalleeOperands();

    // UVM method signature: (self, id, message, verbosity, filename, line)
    // We expect 6 operands for class methods
    if (calleeOperands.size() != 6)
      return failure();

    auto ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Determine the runtime function name based on the method
    std::string runtimeFuncName;
    if (methodName == "uvm_report_error")
      runtimeFuncName = "__moore_uvm_report_error";
    else if (methodName == "uvm_report_warning")
      runtimeFuncName = "__moore_uvm_report_warning";
    else if (methodName == "uvm_report_info")
      runtimeFuncName = "__moore_uvm_report_info";
    else if (methodName == "uvm_report_fatal")
      runtimeFuncName = "__moore_uvm_report_fatal";
    else
      return failure();

    // Runtime function signature:
    // void __moore_uvm_report_xxx(
    //   const char *id, int64_t idLen,
    //   const char *message, int64_t messageLen,
    //   int32_t verbosity,
    //   const char *filename, int64_t filenameLen,
    //   int32_t line,
    //   const char *context, int64_t contextLen)
    auto fnTy = LLVM::LLVMFunctionType::get(
        voidTy, {ptrTy, i64Ty, ptrTy, i64Ty, i32Ty, ptrTy, i64Ty, i32Ty, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, runtimeFuncName, fnTy);

    // Extract operands:
    //   calleeOperands[0] = self (ptr) - ignored for now
    //   calleeOperands[1] = id (!llvm.struct<(ptr, i64)>)
    //   calleeOperands[2] = msg (!llvm.struct<(ptr, i64)>)
    //   calleeOperands[3] = verbosity (i32)
    //   calleeOperands[4] = filename (!llvm.struct<(ptr, i64)>)
    //   calleeOperands[5] = line (i32)
    Value idStruct = calleeOperands[1];
    Value msgStruct = calleeOperands[2];
    Value verbosity = calleeOperands[3];
    Value filenameStruct = calleeOperands[4];
    Value line = calleeOperands[5];

    // Extract id ptr and len
    Value idPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, idStruct,
                                                ArrayRef<int64_t>{0});
    Value idLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, idStruct,
                                                ArrayRef<int64_t>{1});

    // Extract msg ptr and len
    Value msgPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, msgStruct,
                                                 ArrayRef<int64_t>{0});
    Value msgLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, msgStruct,
                                                 ArrayRef<int64_t>{1});

    // Extract filename ptr and len
    Value filenamePtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, filenameStruct,
                                                      ArrayRef<int64_t>{0});
    Value filenameLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, filenameStruct,
                                                      ArrayRef<int64_t>{1});

    // Provide default empty context (null pointer, zero length)
    Value contextPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    Value contextLen = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{idPtr, idLen, msgPtr, msgLen, verbosity,
                                    filenamePtr, filenameLen, line,
                                    contextPtr, contextLen});

    rewriter.eraseOp(op);
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
    auto loc = op->getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value inputValue = adaptor.getValue();
    Value amount = adaptor.getAmount();

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(inputValue.getType())) {
      Value valueComp = extractFourStateValue(rewriter, loc, inputValue);
      Value unknownComp = extractFourStateUnknown(rewriter, loc, inputValue);

      // Handle 4-state amount - extract just the value
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);

      auto width = cast<IntegerType>(valueComp.getType()).getWidth();
      amount = adjustIntegerWidth(rewriter, amount, width, loc);

      // Shift both value and unknown components
      Value shiftedValue = comb::ShlOp::create(rewriter, loc, valueComp, amount, false);
      Value shiftedUnknown = comb::ShlOp::create(rewriter, loc, unknownComp, amount, false);

      auto result = createFourStateStruct(rewriter, loc, shiftedValue, shiftedUnknown);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (!resultType || !resultType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "shift operations require integer/float result type");

    // Handle 4-state amount - extract just the value
    if (isFourStateStructType(amount.getType()))
      amount = extractFourStateValue(rewriter, loc, amount);

    // Comb shift operations require the same bit-width for value and amount
    amount = adjustIntegerWidth(rewriter, amount,
                                resultType.getIntOrFloatBitWidth(), loc);
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, inputValue,
                                             amount, false);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value inputValue = adaptor.getValue();
    Value amount = adaptor.getAmount();

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(inputValue.getType())) {
      Value valueComp = extractFourStateValue(rewriter, loc, inputValue);
      Value unknownComp = extractFourStateUnknown(rewriter, loc, inputValue);

      // Handle 4-state amount - extract just the value
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);

      auto width = cast<IntegerType>(valueComp.getType()).getWidth();
      amount = adjustIntegerWidth(rewriter, amount, width, loc);

      // Shift both value and unknown components (zero fill from left)
      Value shiftedValue = comb::ShrUOp::create(rewriter, loc, valueComp, amount, false);
      Value shiftedUnknown = comb::ShrUOp::create(rewriter, loc, unknownComp, amount, false);

      auto result = createFourStateStruct(rewriter, loc, shiftedValue, shiftedUnknown);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (!resultType || !resultType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "shift operations require integer/float result type");

    // Handle 4-state amount - extract just the value
    if (isFourStateStructType(amount.getType()))
      amount = extractFourStateValue(rewriter, loc, amount);

    // Comb shift operations require the same bit-width for value and amount
    amount = adjustIntegerWidth(rewriter, amount,
                                resultType.getIntOrFloatBitWidth(), loc);
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, inputValue, amount, false);
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

    Value lhsInput = adaptor.getLhs();
    Value rhsInput = adaptor.getRhs();

    // Handle 4-state types
    if (isFourStateStructType(resultType)) {
      // Extract value and unknown components
      Value lhsVal = extractFourStateValue(rewriter, loc, lhsInput);
      Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhsInput);
      Value rhsVal = extractFourStateValue(rewriter, loc, rhsInput);
      Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhsInput);

      // Zero extend for unsigned power operation
      Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
      auto lhsExt = comb::ConcatOp::create(rewriter, loc, zeroVal, lhsVal);
      auto rhsExt = comb::ConcatOp::create(rewriter, loc, zeroVal, rhsVal);

      // Perform the power operation on value components
      auto pow = mlir::math::IPowIOp::create(rewriter, loc, lhsExt, rhsExt);
      auto width = lhsVal.getType().getIntOrFloatBitWidth();
      Value resultVal =
          comb::ExtractOp::create(rewriter, loc, rewriter.getIntegerType(width),
                                  pow, 0);

      // Compute unknown propagation: if any bit is unknown, entire result is X
      Value zero = hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
      Value allOnes =
          hw::ConstantOp::create(rewriter, loc, rewriter.getIntegerType(width), -1);
      Value lhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, lhsUnk, zero);
      Value rhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, rhsUnk, zero);
      Value hasUnknown =
          comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);

      // If any unknown: result unknown = all ones, else = 0
      Value resultUnk =
          comb::MuxOp::create(rewriter, loc, hasUnknown, allOnes, zero);

      auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Two-state: original implementation
    Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    // zero extend both LHS & RHS to ensure the unsigned integers are
    // interpreted correctly when calculating power
    auto lhs = comb::ConcatOp::create(rewriter, loc, zeroVal, lhsInput);
    auto rhs = comb::ConcatOp::create(rewriter, loc, zeroVal, rhsInput);

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
    Location loc = op->getLoc();

    Value lhsInput = adaptor.getLhs();
    Value rhsInput = adaptor.getRhs();

    // Handle 4-state types
    if (isFourStateStructType(resultType)) {
      // Extract value and unknown components
      Value lhsVal = extractFourStateValue(rewriter, loc, lhsInput);
      Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhsInput);
      Value rhsVal = extractFourStateValue(rewriter, loc, rhsInput);
      Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhsInput);

      // Perform the signed power operation on value components
      auto width = lhsVal.getType().getIntOrFloatBitWidth();
      Value resultVal =
          mlir::math::IPowIOp::create(rewriter, loc, lhsVal, rhsVal);

      // Compute unknown propagation: if any bit is unknown, entire result is X
      Value zero = hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
      Value allOnes =
          hw::ConstantOp::create(rewriter, loc, rewriter.getIntegerType(width), -1);
      Value lhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, lhsUnk, zero);
      Value rhsHasUnk = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, rhsUnk, zero);
      Value hasUnknown =
          comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);

      // If any unknown: result unknown = all ones, else = 0
      Value resultUnk =
          comb::MuxOp::create(rewriter, loc, hasUnknown, allOnes, zero);

      auto result = createFourStateStruct(rewriter, loc, resultVal, resultUnk);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Two-state: utilize MLIR math dialect's math.ipowi to handle the
    // exponentiation of expression
    rewriter.replaceOpWithNewOp<mlir::math::IPowIOp>(op, resultType, lhsInput,
                                                     rhsInput);
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value inputValue = adaptor.getValue();
    Value amount = adaptor.getAmount();

    // Handle 4-state struct types ({value, unknown} structs)
    if (isFourStateStructType(inputValue.getType())) {
      Value valueComp = extractFourStateValue(rewriter, loc, inputValue);
      Value unknownComp = extractFourStateUnknown(rewriter, loc, inputValue);

      // Handle 4-state amount - extract just the value
      if (isFourStateStructType(amount.getType()))
        amount = extractFourStateValue(rewriter, loc, amount);

      auto width = cast<IntegerType>(valueComp.getType()).getWidth();
      amount = adjustIntegerWidth(rewriter, amount, width, loc);

      // Arithmetic shift value (sign extension), logical shift unknown (zero fill)
      Value shiftedValue = comb::ShrSOp::create(rewriter, loc, valueComp, amount, false);
      Value shiftedUnknown = comb::ShrUOp::create(rewriter, loc, unknownComp, amount, false);

      auto result = createFourStateStruct(rewriter, loc, shiftedValue, shiftedUnknown);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (!resultType || !resultType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "shift operations require integer/float result type");

    // Handle 4-state amount - extract just the value
    if (isFourStateStructType(amount.getType()))
      amount = extractFourStateValue(rewriter, loc, amount);

    // Comb shift operations require the same bit-width for value and amount
    amount = adjustIntegerWidth(rewriter, amount,
                                resultType.getIntOrFloatBitWidth(), loc);
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, inputValue, amount, false);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    // Check if the input is an LLVM pointer (possibly through unrealized cast).
    // This handles class member access (from GEP), queues, dynamic arrays, etc.
    Value llvmPtrInput;

    if (isa<LLVM::LLVMPointerType>(input.getType())) {
      llvmPtrInput = input;
    } else if (auto castOp = input.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Look through unrealized_conversion_cast to find LLVM pointer.
      // This handles:
      // 1. Class member access where GEP returns !llvm.ptr cast to !llhd.ref<T>
      // 2. Local variables in func/process where alloca !llvm.ptr is cast to
      //    !llhd.ref<T> for type compatibility during conversion.
      if (castOp.getInputs().size() == 1 &&
          isa<LLVM::LLVMPointerType>(castOp.getInputs()[0].getType())) {
        llvmPtrInput = castOp.getInputs()[0];
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(input)) {
      // Check if this is a function parameter with !llhd.ref<T> type.
      // Function parameters of ref type are memory pointers, not signals,
      // so we need to use llvm.load instead of llhd.prb.
      // The simulator cannot track signal references through function call
      // boundaries for ref parameters.
      auto *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parentOp) &&
          isa<llhd::RefType>(input.getType())) {
        // Cast the !llhd.ref<T> to !llvm.ptr for memory access
        llvmPtrInput = UnrealizedConversionCastOp::create(
                           rewriter, op.getLoc(),
                           LLVM::LLVMPointerType::get(op.getContext()), input)
                           .getResult(0);
      }
    }

    // If the input was converted to an LLVM pointer (for queues, dynamic
    // arrays, class member access, etc.), use LLVM load instead of llhd.probe.
    if (llvmPtrInput) {
      // Get the original Moore type to check for associative arrays.
      auto mooreRefType = cast<moore::RefType>(op.getInput().getType());
      auto mooreNestedType = mooreRefType.getNestedType();

      // For local associative array variables, the pointer IS the value
      // (the handle), not a pointer to the value. Local variables are created
      // by VariableOp which calls __moore_assoc_create and returns the handle
      // directly.
      //
      // However, for global variable references (from GetGlobalVariableOp),
      // we DO need to load because the global stores the handle.
      //
      // Similarly, for class property references (from ClassPropertyRefOp),
      // we need to load because the property field stores the handle.
      //
      // We detect these by checking if the source is an AddressOfOp (global)
      // or a GEPOp (class property), vs a local variable handle.
      if (isa<AssocArrayType, WildcardAssocArrayType>(mooreNestedType)) {
        Value source = llvmPtrInput;
        // Unwrap unrealized conversion casts
        while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getInputs().size() == 1)
            source = castOp.getInputs()[0];
          else
            break;
        }
        // Check if this is from an AddressOfOp (global variable reference)
        // or a GEPOp (class property reference).
        // If neither, it's a local variable handle - pass through.
        bool needsLoad = source.getDefiningOp<LLVM::AddressOfOp>() != nullptr ||
                         source.getDefiningOp<LLVM::GEPOp>() != nullptr;
        if (!needsLoad) {
          // Local variable: the pointer is the handle itself.
          rewriter.replaceOp(op, llvmPtrInput);
          return success();
        }
        // Global variable or class property: fall through to load the handle.
      }

      auto hwResultType = typeConverter->convertType(op.getResult().getType());
      // Convert HW type to LLVM type for the load operation
      auto llvmResultType = convertToLLVMType(hwResultType);

      Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), llvmResultType,
                                          llvmPtrInput);

      // Cast back to HW type if needed (e.g., for 4-state structs)
      if (hwResultType != llvmResultType) {
        loaded = UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
                                                    hwResultType, loaded)
                     .getResult(0);
      }
      rewriter.replaceOp(op, loaded);
    } else {
      // For llhd.ref types (signal references), always use llhd.prb to probe
      // the signal value. The simulator tracks signal references through
      // function call boundaries after inlining.
      rewriter.replaceOpWithNewOp<llhd::ProbeOp>(op, input);
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
    auto loc = op.getLoc();

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
      // Convert source to LLVM-compatible type if needed
      // For 4-state hw.struct<value, unknown>, decompose and rebuild as llvm.struct
      Value storeVal = convertValueToLLVMType(adaptor.getSrc(), op.getLoc(),
                                              rewriter);
      LLVM::StoreOp::create(rewriter, op.getLoc(), storeVal, llvmPtrDst);
      rewriter.eraseOp(op);
      return success();
    }

    // Check if destination is an llhd.ref block argument in a function context.
    // Function ref parameters should use llvm.store since the simulator can't
    // track signal references through function call boundaries.
    if (auto refType = dyn_cast<llhd::RefType>(dst.getType())) {
      if (auto blockArg = dyn_cast<BlockArgument>(dst)) {
        // Check if we're in a function context (not hw.module/llhd.process)
        if (auto funcOp = dyn_cast<func::FuncOp>(
                blockArg.getOwner()->getParentOp())) {
          // For function ref parameters, use llvm.store
          auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
          Value ptrValue = UnrealizedConversionCastOp::create(
                               rewriter, loc, ptrType, dst)
                               .getResult(0);

          Value storeVal = convertValueToLLVMType(adaptor.getSrc(), loc, rewriter);
          LLVM::StoreOp::create(rewriter, loc, storeVal, ptrValue);
          rewriter.eraseOp(op);
          return success();
        }
      }
    }

    // Determine the delay for the assignment.
    Value delay;
    if constexpr (std::is_same_v<OpTy, ContinuousAssignOp> ||
                  std::is_same_v<OpTy, BlockingAssignOp>) {
      // Blocking and continuous assignments normally get a 0ns 0d 1e delay.
      // However, for assignments from module inputs (block arguments), use
      // zero epsilon delay (0ns 0d 0e) to fix initialization order issues.
      // This ensures that signals driven from inputs have the correct value
      // at t=0, rather than zero being read before the drive takes effect.
      unsigned epsilon = 1;
      if constexpr (std::is_same_v<OpTy, ContinuousAssignOp>) {
        if (isa<BlockArgument>(op.getSrc()))
          epsilon = 0;
      }
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          llhd::TimeAttr::get(op->getContext(), 0U, "ns", 0, epsilon));
    } else if constexpr (std::is_same_v<OpTy, NonBlockingAssignOp>) {
      // Non-blocking assignments get a 0ns 1d 0e delay.
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          llhd::TimeAttr::get(op->getContext(), 0U, "ns", 1, 0));
    } else {
      // Delayed assignments have a delay operand.
      // Since TimeType now converts to i64, convert it to llhd.time for llhd.drv.
      delay = llhd::IntToTimeOp::create(rewriter, op->getLoc(),
                                        adaptor.getDelay());
    }

    // Extract drive strength for continuous assignments.
    llhd::DriveStrengthAttr llhdStrength0, llhdStrength1;
    if constexpr (std::is_same_v<OpTy, ContinuousAssignOp>) {
      // Convert Moore DriveStrength to LLHD DriveStrength.
      // The enum values are identical by design.
      if (auto mooreStr0 = op.getStrength0Attr()) {
        llhdStrength0 = llhd::DriveStrengthAttr::get(
            op->getContext(),
            static_cast<llhd::DriveStrength>(
                static_cast<uint32_t>(mooreStr0.getValue())));
      }
      if (auto mooreStr1 = op.getStrength1Attr()) {
        llhdStrength1 = llhd::DriveStrengthAttr::get(
            op->getContext(),
            static_cast<llhd::DriveStrength>(
                static_cast<uint32_t>(mooreStr1.getValue())));
      }
    }

    Value srcValue = adaptor.getSrc();

    auto getConvertedValue = [&](Value value) -> Value {
      if (auto remapped = rewriter.getRemappedValue(value))
        return remapped;
      auto targetType = this->getTypeConverter()->convertType(value.getType());
      if (!targetType)
        return Value();
      return this->getTypeConverter()->materializeTargetConversion(rewriter, loc,
                                                        targetType, value);
    };

    auto emitFourStateExtractAssign = [&](Value baseRef, Value idx,
                                          Type resultType) -> LogicalResult {
      auto resultRefType = dyn_cast<llhd::RefType>(resultType);
      if (!resultRefType)
        return failure();
      auto resultStructType =
          dyn_cast<hw::StructType>(resultRefType.getNestedType());
      if (!resultStructType)
        return failure();
      auto resultIntType =
          dyn_cast<IntegerType>(resultStructType.getElements()[0].type);
      if (!resultIntType)
        return failure();
      auto baseRefType = dyn_cast<llhd::RefType>(baseRef.getType());
      if (!baseRefType)
        return failure();
      auto baseStructType =
          dyn_cast<hw::StructType>(baseRefType.getNestedType());
      if (!baseStructType)
        return failure();
      auto baseIntType =
          dyn_cast<IntegerType>(baseStructType.getElements()[0].type);
      if (!baseIntType)
        return failure();

      int64_t baseWidth = baseIntType.getWidth();
      int64_t sliceWidth = resultIntType.getWidth();
      if (sliceWidth <= 0 || sliceWidth > baseWidth)
        return failure();

      Value baseStruct = llhd::ProbeOp::create(rewriter, loc, baseRef);
      Value baseValue = extractFourStateValue(rewriter, loc, baseStruct);
      Value baseUnknown = extractFourStateUnknown(rewriter, loc, baseStruct);

      Value driveValue = srcValue;
      Value driveUnknown;
      if (isFourStateStructType(driveValue.getType())) {
        driveUnknown = extractFourStateUnknown(rewriter, loc, driveValue);
        driveValue = extractFourStateValue(rewriter, loc, driveValue);
      } else {
        driveUnknown = hw::ConstantOp::create(rewriter, loc, resultIntType, 0);
      }
      driveValue =
          adjustIntegerWidth(rewriter, driveValue, sliceWidth, loc);
      driveUnknown =
          adjustIntegerWidth(rewriter, driveUnknown, sliceWidth, loc);

      Value shiftAmount =
          adjustIntegerWidth(rewriter, idx, baseWidth, loc);
      APInt baseMask = APInt::getLowBitsSet(baseWidth, sliceWidth);
      Value maskConst = hw::ConstantOp::create(rewriter, loc, baseMask);
      Value shiftedMask = maskConst;
      if (sliceWidth != baseWidth)
        shiftedMask =
            comb::ShlOp::create(rewriter, loc, maskConst, shiftAmount);

      Value allOnes =
          hw::ConstantOp::create(rewriter, loc, APInt::getAllOnes(baseWidth));
      Value maskInv =
          comb::XorOp::create(rewriter, loc, shiftedMask, allOnes);

      Value baseValueCleared =
          comb::AndOp::create(rewriter, loc, baseValue, maskInv);
      Value baseUnknownCleared =
          comb::AndOp::create(rewriter, loc, baseUnknown, maskInv);

      Value driveValueWide =
          adjustIntegerWidth(rewriter, driveValue, baseWidth, loc);
      Value driveUnknownWide =
          adjustIntegerWidth(rewriter, driveUnknown, baseWidth, loc);
      Value driveValueShifted = driveValueWide;
      Value driveUnknownShifted = driveUnknownWide;
      if (sliceWidth != baseWidth) {
        driveValueShifted =
            comb::ShlOp::create(rewriter, loc, driveValueWide, shiftAmount);
        driveUnknownShifted =
            comb::ShlOp::create(rewriter, loc, driveUnknownWide, shiftAmount);
      }
      Value driveValueMasked =
          comb::AndOp::create(rewriter, loc, driveValueShifted, shiftedMask);
      Value driveUnknownMasked =
          comb::AndOp::create(rewriter, loc, driveUnknownShifted, shiftedMask);

      Value newValue =
          comb::OrOp::create(rewriter, loc, baseValueCleared, driveValueMasked);
      Value newUnknown =
          comb::OrOp::create(rewriter, loc, baseUnknownCleared, driveUnknownMasked);

      Value newStruct =
          createFourStateStruct(rewriter, loc, newValue, newUnknown);
      llhd::DriveOp::create(rewriter, loc, baseRef, newStruct, delay, Value{},
                            llhdStrength0, llhdStrength1);
      rewriter.eraseOp(op);
      return success();
    };

    if (auto extractRef = op.getDst().template getDefiningOp<ExtractRefOp>()) {
      Value baseRef = getConvertedValue(extractRef.getInput());
      auto baseRefType = baseRef ? dyn_cast<llhd::RefType>(baseRef.getType())
                                 : nullptr;
      if (baseRefType && isFourStateStructType(baseRefType.getNestedType())) {
        auto structType = cast<hw::StructType>(baseRefType.getNestedType());
        auto valueType = cast<IntegerType>(structType.getElements()[0].type);
        int64_t width = valueType.getWidth();
        Value idx = hw::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerType(llvm::Log2_64_Ceil(width)),
            extractRef.getLowBit());
        auto resultType =
            this->getTypeConverter()->convertType(extractRef.getResult().getType());
        if (failed(emitFourStateExtractAssign(baseRef, idx, resultType)))
          return failure();
        return success();
      }
    }

    if (auto dynExtractRef =
            op.getDst().template getDefiningOp<DynExtractRefOp>()) {
      Value baseRef = getConvertedValue(dynExtractRef.getInput());
      auto baseRefType = baseRef ? dyn_cast<llhd::RefType>(baseRef.getType())
                                 : nullptr;
      if (baseRefType && isFourStateStructType(baseRefType.getNestedType())) {
        auto structType = cast<hw::StructType>(baseRefType.getNestedType());
        auto valueType = cast<IntegerType>(structType.getElements()[0].type);
        int64_t width = valueType.getWidth();

        Value idx = getConvertedValue(dynExtractRef.getLowBit());
        if (!idx)
          return failure();
        if (isFourStateStructType(idx.getType()))
          idx = extractFourStateValue(rewriter, loc, idx);
        idx = adjustIntegerWidth(rewriter, idx, llvm::Log2_64_Ceil(width), loc);

        auto resultType =
            this->getTypeConverter()->convertType(dynExtractRef.getResult().getType());
        if (failed(emitFourStateExtractAssign(baseRef, idx, resultType)))
          return failure();
        return success();
      }
    }

    auto tryDriveReadOnlyExtractSignal = [&]() -> LogicalResult {
      auto signal = dst.getDefiningOp<llhd::SignalOp>();
      if (!signal)
        return failure();
      auto dstRefType = dyn_cast<llhd::RefType>(dst.getType());
      if (!dstRefType || !isFourStateStructType(dstRefType.getNestedType()))
        return failure();

      auto structCreate =
          signal.getInit().template getDefiningOp<hw::StructCreateOp>();
      if (!structCreate || structCreate.getOperands().size() != 2)
        return failure();

      auto valueProbe =
          structCreate.getOperand(0).getDefiningOp<llhd::ProbeOp>();
      auto unknownProbe =
          structCreate.getOperand(1).getDefiningOp<llhd::ProbeOp>();
      if (!valueProbe || !unknownProbe)
        return failure();

      auto valueExtract =
          valueProbe.getSignal().getDefiningOp<llhd::SigExtractOp>();
      auto unknownExtract =
          unknownProbe.getSignal().getDefiningOp<llhd::SigExtractOp>();
      if (!valueExtract || !unknownExtract)
        return failure();

      Value valueRef = valueExtract.getResult();
      Value unknownRef = unknownExtract.getResult();
      auto valueRefType = cast<llhd::RefType>(valueRef.getType());
      auto valueIntType = dyn_cast<IntegerType>(valueRefType.getNestedType());
      if (!valueIntType)
        return failure();

      Value driveValue = srcValue;
      Value driveUnknown;
      if (isFourStateStructType(driveValue.getType())) {
        driveUnknown = extractFourStateUnknown(rewriter, loc, driveValue);
        driveValue = extractFourStateValue(rewriter, loc, driveValue);
      } else {
        driveUnknown =
            hw::ConstantOp::create(rewriter, loc, valueIntType, 0);
      }
      driveValue =
          adjustIntegerWidth(rewriter, driveValue, valueIntType.getWidth(), loc);
      driveUnknown = adjustIntegerWidth(rewriter, driveUnknown,
                                        valueIntType.getWidth(), loc);

      llhd::DriveOp::create(rewriter, loc, valueRef, driveValue, delay, Value{},
                            llhdStrength0, llhdStrength1);
      llhd::DriveOp::create(rewriter, loc, unknownRef, driveUnknown, delay,
                            Value{}, llhdStrength0, llhdStrength1);
      rewriter.eraseOp(op);
      return success();
    };

    if (succeeded(tryDriveReadOnlyExtractSignal()))
      return success();

    // Handle 4-state struct to union assignment.
    // When assigning a constant to a union with 4-state members, the source
    // gets wrapped in a 4-state struct {value:iN, unknown:iN} which has double
    // the bitwidth, causing hw.bitcast to fail. Extract the value component.
    if (isFourStateStructType(srcValue.getType())) {
      if (auto refType = dyn_cast<llhd::RefType>(dst.getType())) {
        if (isa<hw::UnionType>(refType.getNestedType())) {
          srcValue = extractFourStateValue(rewriter, op.getLoc(), srcValue);
        }
      }
    }

    // Handle function output parameters: when the destination is a block
    // argument with llhd.ref type containing an LLVM type, and we're inside a
    // func::FuncOp, use llvm.store instead of llhd.drv. This is necessary
    // because the simulator cannot track signal references through function
    // calls, causing llhd.drv to fail when the destination is a function
    // Handle function output parameters: when the destination is a block
    // argument (function parameter) with llhd.ref type containing an LLVM type,
    // use llvm.store instead of llhd.drv. This is necessary because the
    // simulator cannot track signal references through function calls, causing
    // llhd.drv to fail when the destination is a function parameter.
    //
    // Check both the original destination (which might be the old block arg)
    // and the converted destination (which might be the new block arg).
    Value origDst = op.getDst();
    bool origIsBlockArg = isa<BlockArgument>(origDst);
    bool dstIsBlockArg = isa<BlockArgument>(dst);

    if (origIsBlockArg || dstIsBlockArg) {
      // Check if the converted type is llhd.ref - for function parameters,
      // we need to use LLVM store instead of llhd.drv because the interpreter
      // doesn't track refs passed through function calls (only signals).
      if (auto refType = dyn_cast<llhd::RefType>(dst.getType())) {
        // Only use llvm.store when the nested type is an LLVM type (e.g.,
        // !llvm.ptr for class handles). Signal refs have HW types (like i42)
        // as the nested type and should use llhd.drv.
        Type nestedType = refType.getNestedType();
        bool nestedIsLLVMType = isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                                    LLVM::LLVMArrayType>(nestedType);

        // Only apply this for function contexts with LLVM nested types.
        // Signal refs (with HW nested types) should use llhd.drv even in
        // function contexts.
        if (nestedIsLLVMType) {
          bool inFuncOp = op->template getParentOfType<func::FuncOp>() != nullptr;
          bool inProcess = op->template getParentOfType<llhd::ProcessOp>() != nullptr;
          bool inInitial = op->template getParentOfType<seq::InitialOp>() != nullptr;

          if (inFuncOp && !inProcess && !inInitial) {
            // For function output parameters with LLVM types (class handles),
            // use llvm.store instead of llhd.drv.
            // Convert the ref to an LLVM pointer and store through it.
            Value refAsPtr = UnrealizedConversionCastOp::create(
                                 rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                                 dst)
                                 .getResult(0);

            // Convert the source value to LLVM type if needed.
            Value storeVal = srcValue;
            Type storeType = storeVal.getType();
            Type llvmStoreType = convertToLLVMType(storeType);
            if (storeType != llvmStoreType) {
              storeVal = UnrealizedConversionCastOp::create(
                             rewriter, loc, llvmStoreType, storeVal)
                             .getResult(0);
            }

            LLVM::StoreOp::create(rewriter, loc, storeVal, refAsPtr);
            rewriter.eraseOp(op);
            return success();
          }
        }
      }
    }

    rewriter.replaceOpWithNewOp<llhd::DriveOp>(
        op, dst, srcValue, delay, Value{}, llhdStrength0, llhdStrength1);
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
    auto loc = op.getLoc();

    // Extract the boolean condition from a 4-state struct if needed.
    // For 4-state values, we use the "value" field and track unknown bits.
    Value condition = adaptor.getCondition();
    Value conditionVal = condition;
    Value conditionUnknown;
    if (isFourStateStructType(condition.getType())) {
      conditionVal = extractFourStateValue(rewriter, loc, condition);
      Value unknown = extractFourStateUnknown(rewriter, loc, condition);
      Value zero = hw::ConstantOp::create(rewriter, loc, unknown.getType(), 0);
      conditionUnknown = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::ne, unknown, zero);
    }
    if (!conditionUnknown) {
      conditionUnknown =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    }

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
          rewriter, loc, type, trueTerm->getOperand(0));
      Value convFalseVal = typeConverter->materializeTargetConversion(
          rewriter, loc, type, falseTerm->getOperand(0));

      rewriter.eraseOp(trueTerm);
      rewriter.eraseOp(falseTerm);

      if (!isFourStateStructType(type)) {
        rewriter.replaceOpWithNewOp<comb::MuxOp>(op, conditionVal, convTrueVal,
                                                 convFalseVal);
        return success();
      }

      Value trueVal = extractFourStateValue(rewriter, loc, convTrueVal);
      Value trueUnk = extractFourStateUnknown(rewriter, loc, convTrueVal);
      Value falseVal = extractFourStateValue(rewriter, loc, convFalseVal);
      Value falseUnk = extractFourStateUnknown(rewriter, loc, convFalseVal);

      Value selectedVal =
          comb::MuxOp::create(rewriter, loc, conditionVal, trueVal, falseVal);
      Value selectedUnk =
          comb::MuxOp::create(rewriter, loc, conditionVal, trueUnk, falseUnk);

      Value mergedVal =
          comb::AndOp::create(rewriter, loc, trueVal, falseVal, false);
      Value diff =
          comb::XorOp::create(rewriter, loc, trueVal, falseVal, false);
      Value mergedUnk =
          comb::OrOp::create(rewriter, loc, trueUnk, falseUnk, false);
      mergedUnk = comb::OrOp::create(rewriter, loc, mergedUnk, diff, false);

      Value finalVal = comb::MuxOp::create(rewriter, loc, conditionUnknown,
                                           mergedVal, selectedVal);
      Value finalUnk = comb::MuxOp::create(rewriter, loc, conditionUnknown,
                                           mergedUnk, selectedUnk);
      auto result = createFourStateStruct(rewriter, loc, finalVal, finalUnk);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto ifOp =
        scf::IfOp::create(rewriter, loc, type, conditionVal);
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

/// Convert arith.select on Moore types to comb.mux.
/// This handles the case where MLIR canonicalizers simplify control flow
/// and introduce arith.select operations on Moore types.
struct ArithSelectOpConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getType());
    if (!type)
      return failure();

    // Convert the operands with type conversion.
    Value trueVal = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), type, adaptor.getTrueValue());
    Value falseVal = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), type, adaptor.getFalseValue());

    if (!trueVal || !falseVal)
      return failure();

    rewriter.replaceOpWithNewOp<comb::MuxOp>(op, adaptor.getCondition(),
                                             trueVal, falseVal);
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
    Value cond = adaptor.getCond();
    // If the condition is a 4-state struct (from l1 type), extract the value.
    if (isFourStateStructType(cond.getType()))
      cond = extractFourStateValue(rewriter, op.getLoc(), cond);
    auto newOp =
        rewriter.replaceOpWithNewOp<VerifOpTy>(op, cond, mlir::Value(), label);
    if (op.getDefer() == DeferAssert::Final)
      newOp->setAttr("bmc.final", rewriter.getUnitAttr());
    return success();
  }
};

/// Lowering for moore.past operation.
/// The moore.past op captures a value from a previous clock cycle. In the
/// MooreToCore lowering, we try to lower 1-bit past values to explicit
/// registers using the enclosing clocked assertion's clock, so they can
/// participate in comparisons as i1. If no clock is found, fall back to
/// ltl.past for boolean values. For non-boolean types, pass through the value
/// directly (used in comparisons where the verification infrastructure handles
/// the past semantics).
struct PastOpConversion : public OpConversionPattern<PastOp> {
  using OpConversionPattern::OpConversionPattern;

  static std::optional<std::pair<Value, verif::ClockEdge>>
  findClockFromUsers(Value value) {
    SmallVector<Value, 8> worklist{value};
    llvm::DenseSet<Operation *> visited;

    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      for (auto &use : cur.getUses()) {
        Operation *user = use.getOwner();
        if (!visited.insert(user).second)
          continue;
        if (auto clocked = dyn_cast<ltl::ClockOp>(user)) {
          verif::ClockEdge edge = verif::ClockEdge::Both;
          switch (clocked.getEdge()) {
          case ltl::ClockEdge::Pos:
            edge = verif::ClockEdge::Pos;
            break;
          case ltl::ClockEdge::Neg:
            edge = verif::ClockEdge::Neg;
            break;
          case ltl::ClockEdge::Both:
            edge = verif::ClockEdge::Both;
            break;
          }
          return std::make_pair(clocked.getClock(), edge);
        }
        if (auto clocked = dyn_cast<verif::ClockedAssertOp>(user))
          return std::make_pair(clocked.getClock(), clocked.getEdge());
        if (auto clocked = dyn_cast<verif::ClockedAssumeOp>(user))
          return std::make_pair(clocked.getClock(), clocked.getEdge());
        if (auto clocked = dyn_cast<verif::ClockedCoverOp>(user))
          return std::make_pair(clocked.getClock(), clocked.getEdge());
        for (auto result : user->getResults())
          worklist.push_back(result);
      }
    }
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(PastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    int64_t delay = op.getDelay();

    if (delay == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }
    if (auto clockInfo = findClockFromUsers(op.getResult())) {
      Value clockSignal = clockInfo->first;
      auto edge = clockInfo->second;
      if (edge == verif::ClockEdge::Neg) {
        auto one = hw::ConstantOp::create(rewriter, loc,
                                          rewriter.getI1Type(), 1);
        clockSignal = comb::XorOp::create(rewriter, loc, clockSignal, one);
      } else if (edge == verif::ClockEdge::Both) {
        op.emitError("both-edge clocks are not supported for moore.past");
        return failure();
      }
      if (!isa<seq::ClockType>(clockSignal.getType()))
        clockSignal = seq::ToClockOp::create(rewriter, loc, clockSignal);

      Value current = input;
      for (int64_t i = 0; i < delay; ++i)
        current = seq::CompRegOp::create(rewriter, loc, current, clockSignal);
      rewriter.replaceOp(op, current);
      return success();
    }

    if (input.getType().isInteger(1)) {
      // Fallback: use ltl.past when no clock is found.
      rewriter.replaceOpWithNewOp<ltl::PastOp>(op, input, delay);
      return success();
    }

    op.emitError("non-boolean moore.past requires a clocked assertion");
    return failure();
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
    Location loc = op.getLoc();

    char padChar = adaptor.getPadding() == IntPadding::Space ? 32 : 48;
    IntegerAttr padCharAttr = rewriter.getI8IntegerAttr(padChar);
    auto widthAttr = adaptor.getSpecifierWidthAttr();

    bool isLeftAligned = adaptor.getAlignment() == IntAlign::Left;
    BoolAttr isLeftAlignedAttr = rewriter.getBoolAttr(isLeftAligned);

    // Get the input value, handling 4-state types which are lowered to
    // {value, unknown} structs. For formatting purposes, we extract just
    // the 'value' field since sim::Format*Op expects an integer.
    Value inputValue = adaptor.getValue();
    Type inputType = inputValue.getType();
    if (auto structType = dyn_cast<hw::StructType>(inputType)) {
      // Extract the 'value' field from the 4-state struct representation
      inputValue = hw::StructExtractOp::create(rewriter, loc, inputValue, "value");
    }

    switch (op.getFormat()) {
    case IntFormat::Decimal:
      rewriter.replaceOpWithNewOp<sim::FormatDecOp>(
          op, inputValue, isLeftAlignedAttr, padCharAttr, widthAttr,
          adaptor.getIsSignedAttr());
      return success();
    case IntFormat::Binary:
      rewriter.replaceOpWithNewOp<sim::FormatBinOp>(
          op, inputValue, isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::Octal:
      rewriter.replaceOpWithNewOp<sim::FormatOctOp>(
          op, inputValue, isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::HexLower:
      rewriter.replaceOpWithNewOp<sim::FormatHexOp>(
          op, inputValue, rewriter.getBoolAttr(false),
          isLeftAlignedAttr, padCharAttr, widthAttr);
      return success();
    case IntFormat::HexUpper:
      rewriter.replaceOpWithNewOp<sim::FormatHexOp>(
          op, inputValue, rewriter.getBoolAttr(true), isLeftAlignedAttr,
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

//===----------------------------------------------------------------------===//
// System Call Builtin Conversions
//===----------------------------------------------------------------------===//

/// Conversion for moore.builtin.strobe -> sim.print_formatted_proc
/// $strobe displays values at end of current time step
struct StrobeBIOpConversion : public OpConversionPattern<StrobeBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StrobeBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

/// Conversion for moore.builtin.fstrobe -> sim.print_formatted_proc
/// $fstrobe writes to file at end of current time step
struct FStrobeBIOpConversion : public OpConversionPattern<FStrobeBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FStrobeBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, just print the message (ignore file descriptor)
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

/// Conversion for moore.builtin.monitor -> sim.print_formatted_proc
/// $monitor enables continuous monitoring
struct MonitorBIOpConversion : public OpConversionPattern<MonitorBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MonitorBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

/// Conversion for moore.builtin.fmonitor -> sim.print_formatted_proc
/// $fmonitor enables continuous monitoring to file
struct FMonitorBIOpConversion : public OpConversionPattern<FMonitorBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FMonitorBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, just print the message (ignore file descriptor)
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

/// Conversion for moore.builtin.monitoron -> erase (no-op)
/// $monitoron re-enables monitoring
struct MonitorOnBIOpConversion : public OpConversionPattern<MonitorOnBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MonitorOnBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.monitoroff -> erase (no-op)
/// $monitoroff disables monitoring
struct MonitorOffBIOpConversion : public OpConversionPattern<MonitorOffBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MonitorOffBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.printtimescale -> erase (no-op)
/// $printtimescale prints timescale info (stub)
struct PrintTimescaleBIOpConversion
    : public OpConversionPattern<PrintTimescaleBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintTimescaleBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.ferror -> constant 0 (stub)
/// $ferror returns file error status
struct FErrorBIOpConversion : public OpConversionPattern<FErrorBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FErrorBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Return 0 (no error) as a stub
    auto i32Ty = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, i32Ty, 0);
    return success();
  }
};

/// Conversion for moore.builtin.ungetc -> return input character
/// $ungetc pushes character back into stream
struct UngetCBIOpConversion : public OpConversionPattern<UngetCBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UngetCBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Return the input character (the pushed-back value)
    rewriter.replaceOp(op, adaptor.getC());
    return success();
  }
};

/// Conversion for moore.builtin.fseek -> constant 0 (success stub)
/// $fseek sets file position
struct FSeekBIOpConversion : public OpConversionPattern<FSeekBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FSeekBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Return 0 (success) as a stub
    auto i32Ty = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, i32Ty, 0);
    return success();
  }
};

/// Conversion for moore.builtin.rewind -> erase (no-op)
/// $rewind resets file position
struct RewindBIOpConversion : public OpConversionPattern<RewindBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RewindBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.fread -> constant 0 (stub)
/// $fread reads binary data from file
struct FReadBIOpConversion : public OpConversionPattern<FReadBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FReadBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Return 0 (bytes read) as a stub
    auto i32Ty = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, i32Ty, 0);
    return success();
  }
};

/// Conversion for moore.builtin.readmemb -> erase (no-op)
/// $readmemb loads memory from binary file
struct ReadMemBBIOpConversion : public OpConversionPattern<ReadMemBBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadMemBBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.builtin.readmemh -> erase (no-op)
/// $readmemh loads memory from hex file
struct ReadMemHBIOpConversion : public OpConversionPattern<ReadMemHBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadMemHBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
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
    // Handle LLVM pointer types - they are 8 bytes on 64-bit systems.
    // This is critical for queue operations storing class objects.
    if (isa<LLVM::LLVMPointerType>(type))
      return 8;

    // Handle LLVM struct types by computing their size from members.
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type)) {
      int64_t totalSize = 0;
      for (Type elemTy : structTy.getBody()) {
        totalSize += getTypeSizeInBytes(elemTy);
      }
      return totalSize > 0 ? totalSize : 1;
    }

    // Handle LLVM array types.
    if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(type)) {
      int64_t elemSize = getTypeSizeInBytes(arrayTy.getElementType());
      return elemSize * arrayTy.getNumElements();
    }

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
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // Check if input is a fixed array or already a queue struct
    Type inputType = adaptor.getArray().getType();

    if (auto hwArrayTy = dyn_cast<hw::ArrayType>(inputType)) {
      // Fixed array case: convert to queue-like representation
      // Use __moore_array_max runtime function that takes (ptr, len, elem_size)
      auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_max", fnTy);

      // Convert hw::ArrayType to LLVM array type for alloca
      Type llvmArrayType = convertToLLVMType(hwArrayTy);

      // Store the fixed array to an alloca
      auto arrayAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayType, one);
      // Cast input to LLVM type if needed
      Value arrayToStore = adaptor.getArray();
      if (llvmArrayType != inputType) {
        arrayToStore = UnrealizedConversionCastOp::create(rewriter, loc,
                                                          llvmArrayType,
                                                          adaptor.getArray())
                           .getResult(0);
      }
      LLVM::StoreOp::create(rewriter, loc, arrayToStore, arrayAlloca);

      // Get array length and element size
      int64_t arrayLen = hwArrayTy.getNumElements();
      Type elemType = hwArrayTy.getElementType();
      int32_t elemSize = getTypeSizeInBytes(elemType);

      auto lenConst = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                               rewriter.getI64IntegerAttr(arrayLen));
      auto elemSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(elemSize));

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
          ValueRange{arrayAlloca, lenConst, elemSizeConst});
      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // Queue/dynamic array case: input is already {ptr, i64} struct
    // Use __moore_array_max with (ptr, elementSize, isSigned)
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i1Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_max", fnTy);

    // Store input to alloca and pass pointer.
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    // Get element type and size from the result queue type
    auto resultQueueTy = cast<moore::QueueType>(op.getResult().getType());
    auto elemType = typeConverter->convertType(resultQueueTy.getElementType());
    int32_t elemSize = getTypeSizeInBytes(elemType);
    auto elemSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemSize));

    // Use unsigned comparison by default (isSigned = false)
    auto isSignedConst = LLVM::ConstantOp::create(
        rewriter, loc, i1Ty, rewriter.getBoolAttr(false));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca, elemSizeConst, isSignedConst});
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
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // Check if input is a fixed array or already a queue struct
    Type inputType = adaptor.getArray().getType();

    if (auto hwArrayTy = dyn_cast<hw::ArrayType>(inputType)) {
      // Fixed array case: convert to queue-like representation
      // Use __moore_array_min runtime function that takes (ptr, len, elem_size)
      auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i32Ty});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_min", fnTy);

      // Convert hw::ArrayType to LLVM array type for alloca
      Type llvmArrayType = convertToLLVMType(hwArrayTy);

      // Store the fixed array to an alloca
      auto arrayAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayType, one);
      // Cast input to LLVM type if needed
      Value arrayToStore = adaptor.getArray();
      if (llvmArrayType != inputType) {
        arrayToStore = UnrealizedConversionCastOp::create(rewriter, loc,
                                                          llvmArrayType,
                                                          adaptor.getArray())
                           .getResult(0);
      }
      LLVM::StoreOp::create(rewriter, loc, arrayToStore, arrayAlloca);

      // Get array length and element size
      int64_t arrayLen = hwArrayTy.getNumElements();
      Type elemType = hwArrayTy.getElementType();
      int32_t elemSize = getTypeSizeInBytes(elemType);

      auto lenConst = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                               rewriter.getI64IntegerAttr(arrayLen));
      auto elemSizeConst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(elemSize));

      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
          ValueRange{arrayAlloca, lenConst, elemSizeConst});
      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // Queue/dynamic array case: input is already {ptr, i64} struct
    // Use __moore_array_min with (ptr, elementSize, isSigned)
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i1Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_min", fnTy);

    // Store input to alloca and pass pointer.
    auto inputAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), inputAlloca);

    // Get element type and size from the result queue type
    auto resultQueueTy = cast<moore::QueueType>(op.getResult().getType());
    auto elemType = typeConverter->convertType(resultQueueTy.getElementType());
    int32_t elemSize = getTypeSizeInBytes(elemType);
    auto elemSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemSize));

    // Use unsigned comparison by default (isSigned = false)
    auto isSignedConst = LLVM::ConstantOp::create(
        rewriter, loc, i1Ty, rewriter.getBoolAttr(false));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{inputAlloca, elemSizeConst, isSignedConst});
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

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

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

      // Calculate element size
      auto elemSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

      // Truncate index to i32 if necessary
      Value indexVal = adaptor.getIndex();
      if (indexVal.getType() != i32Ty)
        indexVal = LLVM::TruncOp::create(rewriter, loc, i32Ty, indexVal);

      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(fn),
                           ValueRange{queuePtr, indexVal, elemSize});
    } else {
      // delete() - clear all elements
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
      auto fn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_clear", fnTy);

      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                           ValueRange{queuePtr});
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

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void push_back(queue_ptr, element_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_push_back", fnTy);

    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

    // Store element to alloca and pass pointer
    auto elemType = typeConverter->convertType(op.getElement().getType());
    // Convert element type to pure LLVM type for LLVM operations
    Type llvmElemType = convertToLLVMType(elemType);
    auto elemAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);
    // Cast element to LLVM type if needed (hw.struct -> llvm.struct)
    Value elemToStore = adaptor.getElement();
    if (llvmElemType != elemType) {
      elemToStore = UnrealizedConversionCastOp::create(
                        rewriter, loc, llvmElemType, ValueRange{elemToStore})
                        .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, loc, elemToStore, elemAlloca);

    // Calculate element size (use LLVM type for accurate size)
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(llvmElemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{queuePtr, elemAlloca, elemSize});

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

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void push_front(queue_ptr, element_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_push_front", fnTy);

    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

    // Store element to alloca and pass pointer
    auto elemType = typeConverter->convertType(op.getElement().getType());
    // Convert element type to pure LLVM type for LLVM operations
    Type llvmElemType = convertToLLVMType(elemType);
    auto elemAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);
    // Cast element to LLVM type if needed (hw.struct -> llvm.struct)
    Value elemToStore = adaptor.getElement();
    if (llvmElemType != elemType) {
      elemToStore = UnrealizedConversionCastOp::create(
                        rewriter, loc, llvmElemType, ValueRange{elemToStore})
                        .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, loc, elemToStore, elemAlloca);

    // Calculate element size (use LLVM type for accurate size)
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(llvmElemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{queuePtr, elemAlloca, elemSize});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.insert -> runtime function call.
struct QueueInsertOpConversion
    : public RuntimeCallConversionBase<QueueInsertOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Function signature: void insert(queue_ptr, index, element_ptr, element_size)
    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty, ptrTy, i64Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_queue_insert", fnTy);

    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

    // Convert index to i32 if necessary
    Value index = adaptor.getIndex();
    if (index.getType() != i32Ty) {
      index = LLVM::TruncOp::create(rewriter, loc, i32Ty, index);
    }

    // Store element to alloca and pass pointer
    auto elemType = typeConverter->convertType(op.getElement().getType());
    // Convert element type to pure LLVM type for LLVM operations
    Type llvmElemType = convertToLLVMType(elemType);
    auto elemAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);
    // Cast element to LLVM type if needed (hw.struct -> llvm.struct)
    Value elemToStore = adaptor.getElement();
    if (llvmElemType != elemType) {
      elemToStore = UnrealizedConversionCastOp::create(
                        rewriter, loc, llvmElemType, ValueRange{elemToStore})
                        .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, loc, elemToStore, elemAlloca);

    // Calculate element size (use LLVM type for accurate size)
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(llvmElemType)));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{queuePtr, index, elemAlloca, elemSize});

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

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    auto resultType = typeConverter->convertType(op.getResult().getType());

    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(resultType)));

    // For complex types (pointers, structs), use output pointer approach
    // For simple integer types that fit in i64, use the i64 return approach
    bool useOutputPointer = isa<LLVM::LLVMPointerType>(resultType) ||
                            isa<LLVM::LLVMStructType>(resultType) ||
                            isa<hw::StructType>(resultType);

    if (useOutputPointer) {
      // Function signature: void pop_back_ptr(queue_ptr, result_ptr, element_size)
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_queue_pop_back_ptr", fnTy);

      // Convert resultType to LLVM type for alloca/load operations
      Type llvmResultType = convertToLLVMType(resultType);

      // Allocate space for the result
      auto resultAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmResultType, one);

      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                           ValueRange{queuePtr, resultAlloca, elemSize});

      // Load the result from the alloca
      Value result =
          LLVM::LoadOp::create(rewriter, loc, llvmResultType, resultAlloca);
      rewriter.replaceOp(op, result);
    } else {
      // Function signature: i64 pop_back(queue_ptr, element_size)
      // Returns the element as i64 (caller truncates/extends as needed)
      auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_queue_pop_back", fnTy);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{queuePtr, elemSize});

      // Convert result to the expected type
      Value result = call.getResult();
      if (resultType.isIntOrFloat()) {
        auto resultWidth = resultType.getIntOrFloatBitWidth();
        if (resultWidth < 64) {
          result = arith::TruncIOp::create(rewriter, loc, resultType, result);
        } else if (resultWidth > 64) {
          result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
        }
      }

      rewriter.replaceOp(op, result);
    }
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

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    auto resultType = typeConverter->convertType(op.getResult().getType());

    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));

    // adaptor.getQueue() is already a pointer to the queue struct (from RefType
    // conversion), so pass it directly to the runtime function.
    Value queuePtr = adaptor.getQueue();

    // Calculate element size
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(resultType)));

    // For complex types (pointers, structs), use output pointer approach
    // For simple integer types that fit in i64, use the i64 return approach
    bool useOutputPointer = isa<LLVM::LLVMPointerType>(resultType) ||
                            isa<LLVM::LLVMStructType>(resultType) ||
                            isa<hw::StructType>(resultType);

    if (useOutputPointer) {
      // Function signature: void pop_front_ptr(queue_ptr, result_ptr, element_size)
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_queue_pop_front_ptr", fnTy);

      // Convert resultType to LLVM type for alloca/load operations
      Type llvmResultType = convertToLLVMType(resultType);

      // Allocate space for the result
      auto resultAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmResultType, one);

      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                           ValueRange{queuePtr, resultAlloca, elemSize});

      // Load the result from the alloca
      Value result =
          LLVM::LoadOp::create(rewriter, loc, llvmResultType, resultAlloca);
      rewriter.replaceOp(op, result);
    } else {
      // Function signature: i64 pop_front(queue_ptr, element_size)
      // Returns the element as i64 (caller truncates/extends as needed)
      auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_queue_pop_front", fnTy);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                       SymbolRefAttr::get(fn),
                                       ValueRange{queuePtr, elemSize});

      // Convert result to the expected type
      Value result = call.getResult();
      if (resultType.isIntOrFloat()) {
        auto resultWidth = resultType.getIntOrFloatBitWidth();
        if (resultWidth < 64) {
          result = arith::TruncIOp::create(rewriter, loc, resultType, result);
        } else if (resultWidth > 64) {
          result = arith::ExtUIOp::create(rewriter, loc, resultType, result);
        }
      }

      rewriter.replaceOp(op, result);
    }
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

/// Conversion for moore.queue.sort.with -> inline loop with key extraction.
/// Sorts the queue in-place using a custom key expression.
/// The approach is to extract keys for all elements, sort indices by keys,
/// then reorder the queue elements.
struct QueueSortWithOpConversion
    : public OpConversionPattern<QueueSortWithOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Get the LLVM struct type for a queue: {ptr, i64}.
  static LLVM::LLVMStructType getQueueStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }

  /// Get the size in bytes of an LLVM type.
  static int64_t getTypeSize(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type))
      return (intType.getWidth() + 7) / 8;
    if (isa<LLVM::LLVMPointerType>(type))
      return 8;
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
      int64_t size = 0;
      for (Type fieldType : structType.getBody())
        size += getTypeSize(fieldType);
      return size;
    }
    return 8;
  }

  LogicalResult
  matchAndRewrite(QueueSortWithOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto queueTy = getQueueStructType(ctx);

    // Get element type from the queue reference
    auto refType = cast<moore::RefType>(op.getQueue().getType());
    auto nestedType = refType.getNestedType();
    Type mooreElemType;
    if (auto queueType = dyn_cast<moore::QueueType>(nestedType))
      mooreElemType = queueType.getElementType();
    else if (auto dynArrayType =
                 dyn_cast<moore::OpenUnpackedArrayType>(nestedType))
      mooreElemType = dynArrayType.getElementType();
    else
      return rewriter.notifyMatchFailure(op, "unsupported queue type");

    Type elemType = typeConverter->convertType(mooreElemType);
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "failed to convert element type");

    // Get the key type from the yield operation
    Block &body = op.getBody().front();
    auto yieldOp = dyn_cast<QueueSortKeyYieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(op, "expected queue.sort.key.yield");

    Type mooreKeyType = yieldOp.getKey().getType();
    Type keyType = typeConverter->convertType(mooreKeyType);
    if (!keyType)
      return rewriter.notifyMatchFailure(op, "failed to convert key type");

    // Load the queue from the reference
    Value queue = LLVM::LoadOp::create(rewriter, loc, queueTy, adaptor.getQueue());

    // Extract queue length and data pointer
    Value queueLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, queue,
                                                   ArrayRef<int64_t>{1});
    Value dataPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, queue,
                                                  ArrayRef<int64_t>{0});

    // Allocate arrays for keys and indices
    Value keysAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, queueLen);
    Value indicesAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, queueLen);

    // Initialize indices and extract keys using a loop
    Value lb = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(0));
    Value step = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    // First loop: extract keys and initialize indices
    auto extractLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(extractLoop.getBody());
      Value iv = extractLoop.getInductionVar();

      // Store index
      Value idxPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                          indicesAlloca, iv);
      LLVM::StoreOp::create(rewriter, loc, iv, idxPtr);

      // Load current element
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                           dataPtr, iv);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);

      // Cast LLVM element back to Moore type for the cloned body operations
      Value elemMoore =
          UnrealizedConversionCastOp::create(rewriter, loc,
                                             body.getArgument(0).getType(), elem)
              .getResult(0);

      // Clone the body region to compute the key
      IRMapping mapping;
      mapping.map(body.getArgument(0), elemMoore);

      for (Operation &bodyOp : body.without_terminator()) {
        Operation *cloned = rewriter.clone(bodyOp, mapping);
        for (auto [oldResult, newResult] :
             llvm::zip(bodyOp.getResults(), cloned->getResults()))
          mapping.map(oldResult, newResult);
      }

      // Get the key value from the yield and convert to LLVM type
      Value keyValueMoore = mapping.lookupOrDefault(yieldOp.getKey());
      Value keyValue = typeConverter->materializeTargetConversion(
          rewriter, loc, keyType, keyValueMoore);
      if (!keyValue)
        return rewriter.notifyMatchFailure(op, "failed to convert key type");

      // Store key
      Value keyPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                          keysAlloca, iv);
      LLVM::StoreOp::create(rewriter, loc, keyValue, keyPtr);

      rewriter.setInsertionPointAfter(extractLoop);
    }

    // Bubble sort the indices by keys (simple but works for arbitrary key types)
    // Outer loop: i from 0 to len-1
    Value lenMinus1 = arith::SubIOp::create(rewriter, loc, queueLen, step);
    auto outerLoop = scf::ForOp::create(rewriter, loc, lb, lenMinus1, step);
    {
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value i = outerLoop.getInductionVar();

      // Inner loop: j from i+1 to len
      Value iPlusOne = arith::AddIOp::create(rewriter, loc, i, step);
      auto innerLoop = scf::ForOp::create(rewriter, loc, iPlusOne, queueLen, step);
      {
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        Value j = innerLoop.getInductionVar();

        // Load indices[i] and indices[j]
        Value idxPtrI = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                             indicesAlloca, i);
        Value idxPtrJ = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                             indicesAlloca, j);
        Value idxI = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtrI);
        Value idxJ = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtrJ);

        // Load keys[indices[i]] and keys[indices[j]]
        Value keyPtrI = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                             keysAlloca, idxI);
        Value keyPtrJ = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                             keysAlloca, idxJ);
        Value keyI = LLVM::LoadOp::create(rewriter, loc, keyType, keyPtrI);
        Value keyJ = LLVM::LoadOp::create(rewriter, loc, keyType, keyPtrJ);

        // Compare keys: if keyI > keyJ, swap indices (ascending order)
        Value shouldSwap;
        if (isa<IntegerType>(keyType)) {
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::sgt,
                                              keyI, keyJ);
        } else if (isa<FloatType>(keyType)) {
          shouldSwap = arith::CmpFOp::create(rewriter, loc,
                                              arith::CmpFPredicate::OGT,
                                              keyI, keyJ);
        } else if (auto structType = dyn_cast<LLVM::LLVMStructType>(keyType)) {
          // For LLVM struct types (e.g., class handles {ptr, i64}),
          // extract the pointer and compare as integers (by address).
          // This maintains stable ordering for class references.
          Value ptrI = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                     keyI, ArrayRef<int64_t>{0});
          Value ptrJ = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                     keyJ, ArrayRef<int64_t>{0});
          // Convert pointers to integers for comparison
          Value intI = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrI);
          Value intJ = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrJ);
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::ugt,
                                              intI, intJ);
        } else if (isa<LLVM::LLVMPointerType>(keyType)) {
          // For pointer types, convert to integers and compare
          Value intI = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, keyI);
          Value intJ = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, keyJ);
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::ugt,
                                              intI, intJ);
        } else {
          // Fallback: for other types, treat as equal (no swap)
          shouldSwap = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(false));
        }

        // Conditionally swap
        auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, shouldSwap,
                                       /*withElseRegion=*/false);
        {
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          LLVM::StoreOp::create(rewriter, loc, idxJ, idxPtrI);
          LLVM::StoreOp::create(rewriter, loc, idxI, idxPtrJ);
        }

        rewriter.setInsertionPointAfter(innerLoop);
      }

      rewriter.setInsertionPointAfter(outerLoop);
    }

    // Allocate temporary storage for reordering
    Value tempAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemType, queueLen);

    // Copy elements in sorted order to temp
    auto copyLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(copyLoop.getBody());
      Value i = copyLoop.getInductionVar();

      // Get sorted index
      Value idxPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                          indicesAlloca, i);
      Value sortedIdx = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtr);

      // Copy element from original position to new position
      Value srcPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          dataPtr, sortedIdx);
      Value dstPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          tempAlloca, i);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, srcPtr);
      LLVM::StoreOp::create(rewriter, loc, elem, dstPtr);

      rewriter.setInsertionPointAfter(copyLoop);
    }

    // Copy sorted elements back to original queue
    auto copyBackLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(copyBackLoop.getBody());
      Value i = copyBackLoop.getInductionVar();

      Value srcPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          tempAlloca, i);
      Value dstPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          dataPtr, i);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, srcPtr);
      LLVM::StoreOp::create(rewriter, loc, elem, dstPtr);

      rewriter.setInsertionPointAfter(copyBackLoop);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.rsort.with -> inline loop with key extraction.
/// Same as QueueSortWithOpConversion but sorts in descending order.
struct QueueRSortWithOpConversion
    : public OpConversionPattern<QueueRSortWithOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Get the LLVM struct type for a queue: {ptr, i64}.
  static LLVM::LLVMStructType getQueueStructType(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  }

  /// Get the size in bytes of an LLVM type.
  static int64_t getTypeSize(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type))
      return (intType.getWidth() + 7) / 8;
    if (isa<LLVM::LLVMPointerType>(type))
      return 8;
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
      int64_t size = 0;
      for (Type fieldType : structType.getBody())
        size += getTypeSize(fieldType);
      return size;
    }
    return 8;
  }

  LogicalResult
  matchAndRewrite(QueueRSortWithOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto queueTy = getQueueStructType(ctx);

    // Get element type from the queue reference
    auto refType = cast<moore::RefType>(op.getQueue().getType());
    auto nestedType = refType.getNestedType();
    Type mooreElemType;
    if (auto queueType = dyn_cast<moore::QueueType>(nestedType))
      mooreElemType = queueType.getElementType();
    else if (auto dynArrayType =
                 dyn_cast<moore::OpenUnpackedArrayType>(nestedType))
      mooreElemType = dynArrayType.getElementType();
    else
      return rewriter.notifyMatchFailure(op, "unsupported queue type");

    Type elemType = typeConverter->convertType(mooreElemType);
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "failed to convert element type");

    // Get the key type from the yield operation
    Block &body = op.getBody().front();
    auto yieldOp = dyn_cast<QueueSortKeyYieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(op, "expected queue.sort.key.yield");

    Type mooreKeyType = yieldOp.getKey().getType();
    Type keyType = typeConverter->convertType(mooreKeyType);
    if (!keyType)
      return rewriter.notifyMatchFailure(op, "failed to convert key type");

    // Load the queue from the reference
    Value queue = LLVM::LoadOp::create(rewriter, loc, queueTy, adaptor.getQueue());

    // Extract queue length and data pointer
    Value queueLen = LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, queue,
                                                   ArrayRef<int64_t>{1});
    Value dataPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, queue,
                                                  ArrayRef<int64_t>{0});

    // Allocate arrays for keys and indices
    Value keysAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, keyType, queueLen);
    Value indicesAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, queueLen);

    // Initialize indices and extract keys using a loop
    Value lb = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(0));
    Value step = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    // First loop: extract keys and initialize indices
    auto extractLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(extractLoop.getBody());
      Value iv = extractLoop.getInductionVar();

      // Store index
      Value idxPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                          indicesAlloca, iv);
      LLVM::StoreOp::create(rewriter, loc, iv, idxPtr);

      // Load current element
      Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                           dataPtr, iv);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, elemPtr);

      // Cast LLVM element back to Moore type for the cloned body operations
      Value elemMoore =
          UnrealizedConversionCastOp::create(rewriter, loc,
                                             body.getArgument(0).getType(), elem)
              .getResult(0);

      // Clone the body region to compute the key
      IRMapping mapping;
      mapping.map(body.getArgument(0), elemMoore);

      for (Operation &bodyOp : body.without_terminator()) {
        Operation *cloned = rewriter.clone(bodyOp, mapping);
        for (auto [oldResult, newResult] :
             llvm::zip(bodyOp.getResults(), cloned->getResults()))
          mapping.map(oldResult, newResult);
      }

      // Get the key value from the yield and convert to LLVM type
      Value keyValueMoore = mapping.lookupOrDefault(yieldOp.getKey());
      Value keyValue = typeConverter->materializeTargetConversion(
          rewriter, loc, keyType, keyValueMoore);
      if (!keyValue)
        return rewriter.notifyMatchFailure(op, "failed to convert key type");

      // Store key
      Value keyPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                          keysAlloca, iv);
      LLVM::StoreOp::create(rewriter, loc, keyValue, keyPtr);

      rewriter.setInsertionPointAfter(extractLoop);
    }

    // Bubble sort the indices by keys (simple but works for arbitrary key types)
    // Outer loop: i from 0 to len-1
    Value lenMinus1 = arith::SubIOp::create(rewriter, loc, queueLen, step);
    auto outerLoop = scf::ForOp::create(rewriter, loc, lb, lenMinus1, step);
    {
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value i = outerLoop.getInductionVar();

      // Inner loop: j from i+1 to len
      Value iPlusOne = arith::AddIOp::create(rewriter, loc, i, step);
      auto innerLoop = scf::ForOp::create(rewriter, loc, iPlusOne, queueLen, step);
      {
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        Value j = innerLoop.getInductionVar();

        // Load indices[i] and indices[j]
        Value idxPtrI = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                             indicesAlloca, i);
        Value idxPtrJ = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                             indicesAlloca, j);
        Value idxI = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtrI);
        Value idxJ = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtrJ);

        // Load keys[indices[i]] and keys[indices[j]]
        Value keyPtrI = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                             keysAlloca, idxI);
        Value keyPtrJ = LLVM::GEPOp::create(rewriter, loc, ptrTy, keyType,
                                             keysAlloca, idxJ);
        Value keyI = LLVM::LoadOp::create(rewriter, loc, keyType, keyPtrI);
        Value keyJ = LLVM::LoadOp::create(rewriter, loc, keyType, keyPtrJ);

        // Compare keys: if keyI < keyJ, swap indices (descending order)
        Value shouldSwap;
        if (isa<IntegerType>(keyType)) {
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::slt,
                                              keyI, keyJ);
        } else if (isa<FloatType>(keyType)) {
          shouldSwap = arith::CmpFOp::create(rewriter, loc,
                                              arith::CmpFPredicate::OLT,
                                              keyI, keyJ);
        } else if (auto structType = dyn_cast<LLVM::LLVMStructType>(keyType)) {
          // For LLVM struct types (e.g., class handles {ptr, i64}),
          // extract the pointer and compare as integers (by address).
          // This maintains stable ordering for class references.
          Value ptrI = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                     keyI, ArrayRef<int64_t>{0});
          Value ptrJ = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                     keyJ, ArrayRef<int64_t>{0});
          // Convert pointers to integers for comparison
          Value intI = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrI);
          Value intJ = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrJ);
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::ult,
                                              intI, intJ);
        } else if (isa<LLVM::LLVMPointerType>(keyType)) {
          // For pointer types, convert to integers and compare
          Value intI = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, keyI);
          Value intJ = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, keyJ);
          shouldSwap = arith::CmpIOp::create(rewriter, loc,
                                              arith::CmpIPredicate::ult,
                                              intI, intJ);
        } else {
          // Fallback: for other types, treat as equal (no swap)
          shouldSwap = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(false));
        }

        // Conditionally swap
        auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, shouldSwap,
                                       /*withElseRegion=*/false);
        {
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          LLVM::StoreOp::create(rewriter, loc, idxJ, idxPtrI);
          LLVM::StoreOp::create(rewriter, loc, idxI, idxPtrJ);
        }

        rewriter.setInsertionPointAfter(innerLoop);
      }

      rewriter.setInsertionPointAfter(outerLoop);
    }

    // Allocate temporary storage for reordering
    Value tempAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemType, queueLen);

    // Copy elements in sorted order to temp
    auto copyLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(copyLoop.getBody());
      Value i = copyLoop.getInductionVar();

      // Get sorted index
      Value idxPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                          indicesAlloca, i);
      Value sortedIdx = LLVM::LoadOp::create(rewriter, loc, i64Ty, idxPtr);

      // Copy element from original position to new position
      Value srcPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          dataPtr, sortedIdx);
      Value dstPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          tempAlloca, i);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, srcPtr);
      LLVM::StoreOp::create(rewriter, loc, elem, dstPtr);

      rewriter.setInsertionPointAfter(copyLoop);
    }

    // Copy sorted elements back to original queue
    auto copyBackLoop = scf::ForOp::create(rewriter, loc, lb, queueLen, step);
    {
      rewriter.setInsertionPointToStart(copyBackLoop.getBody());
      Value i = copyBackLoop.getInductionVar();

      Value srcPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          tempAlloca, i);
      Value dstPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemType,
                                          dataPtr, i);
      Value elem = LLVM::LoadOp::create(rewriter, loc, elemType, srcPtr);
      LLVM::StoreOp::create(rewriter, loc, elem, dstPtr);

      rewriter.setInsertionPointAfter(copyBackLoop);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.queue.sort.key.yield -> no-op (handled by parent).
/// This operation is just a terminator and is handled by QueueSortWithOpConversion.
struct QueueSortKeyYieldOpConversion
    : public OpConversionPattern<QueueSortKeyYieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueueSortKeyYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This operation is cloned inside the parent's conversion and the key value
    // is extracted directly. The original operation is erased when the parent
    // (QueueSortWithOp or QueueRSortWithOp) is converted.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.array.locator.yield -> no-op (handled by parent).
/// This operation is just a terminator and is handled by ArrayLocatorOpConversion.
struct ArrayLocatorYieldOpConversion
    : public OpConversionPattern<ArrayLocatorYieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayLocatorYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This operation is handled by the parent ArrayLocatorOp conversion.
    // The original operation is erased when the parent is converted.
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

/// Conversion for moore.queue.reverse -> runtime function call.
struct QueueReverseOpConversion
    : public RuntimeCallConversionBase<QueueReverseOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(QueueReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i64Ty});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_reverse", fnTy);

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

    // Function signature: queue concat(ptr to array of queues, count, elem_size)
    auto fnTy = LLVM::LLVMFunctionType::get(queueTy, {ptrTy, i64Ty, i64Ty});
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

    auto mooreQueueTy = cast<moore::QueueType>(op.getResult().getType());
    auto elemType =
        typeConverter->convertType(mooreQueueTy.getElementType());
    auto elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(getTypeSizeInBytes(elemType)));

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{queueTy},
                                     SymbolRefAttr::get(fn),
                                     ValueRange{arrayAlloca, count, elemSize});
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

/// Conversion for moore.array.contains -> runtime function call that checks
/// if a value is contained in an unpacked array. This is used to implement
/// the SystemVerilog 'inside' operator with unpacked arrays.
struct ArrayContainsOpConversion
    : public RuntimeCallConversionBase<ArrayContainsOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(ArrayContainsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto arrayType = op.getArray().getType();
    auto valueType = op.getValue().getType();
    auto i1Ty = IntegerType::get(ctx, 1);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get the element type and calculate element size
    Type mooreElemType;
    if (auto uarrayTy = dyn_cast<UnpackedArrayType>(arrayType))
      mooreElemType = uarrayTy.getElementType();
    else if (auto openUarrayTy = dyn_cast<OpenUnpackedArrayType>(arrayType))
      mooreElemType = openUarrayTy.getElementType();
    else if (auto queueTy = dyn_cast<QueueType>(arrayType))
      mooreElemType = queueTy.getElementType();
    else
      return failure();

    // Calculate element size in bytes
    int64_t elemBitWidth = 1;
    if (auto elemIntType = dyn_cast<moore::IntType>(mooreElemType))
      elemBitWidth = elemIntType.getWidth();
    int64_t elemByteSize = (elemBitWidth + 7) / 8;

    Value elemSizeVal = arith::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemByteSize));

    // Convert the value to search for to a pointer (via alloca)
    Type convertedValueType = typeConverter->convertType(valueType);
    if (!convertedValueType)
      return failure();

    auto oneVal =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto valueAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy,
                                              convertedValueType, oneVal, 0);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getValue(), valueAlloca);

    // Handle static unpacked arrays
    if (auto uarrayTy = dyn_cast<UnpackedArrayType>(arrayType)) {
      int64_t numElements = uarrayTy.getSize();
      Value numElemsVal = arith::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(numElements));

      // Get pointer to the array data
      Value arrayPtr = adaptor.getArray();
      if (!isa<LLVM::LLVMPointerType>(arrayPtr.getType())) {
        // If it's not already a pointer, allocate and store
        auto convertedArrayType = typeConverter->convertType(arrayType);
        auto arrayAlloca = LLVM::AllocaOp::create(
            rewriter, loc, ptrTy, convertedArrayType, oneVal, 0);
        LLVM::StoreOp::create(rewriter, loc, arrayPtr, arrayAlloca);
        arrayPtr = arrayAlloca;
      }

      // Function: i1 __moore_array_contains(void* arr, i64 numElems,
      //                                     void* value, i64 elemSize)
      auto fnTy =
          LLVM::LLVMFunctionType::get(i1Ty, {ptrTy, i64Ty, ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_contains",
                                       fnTy);

      SmallVector<Value, 4> args = {arrayPtr, numElemsVal, valueAlloca,
                                    elemSizeVal};
      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i1Ty},
                                       SymbolRefAttr::get(fn), args);
      rewriter.replaceOp(op, call.getResult());
      return success();
    }

    // Handle dynamic arrays and queues
    if (isa<OpenUnpackedArrayType, QueueType>(arrayType)) {
      // Extract pointer and size from the {ptr, i64} struct
      Value dataPtr = LLVM::ExtractValueOp::create(
          rewriter, loc, ptrTy, adaptor.getArray(), ArrayRef<int64_t>{0});
      Value numElems = LLVM::ExtractValueOp::create(
          rewriter, loc, i64Ty, adaptor.getArray(), ArrayRef<int64_t>{1});

      // Function: i1 __moore_array_contains(void* arr, i64 numElems,
      //                                     void* value, i64 elemSize)
      auto fnTy =
          LLVM::LLVMFunctionType::get(i1Ty, {ptrTy, i64Ty, ptrTy, i64Ty});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_contains",
                                       fnTy);

      SmallVector<Value, 4> args = {dataPtr, numElems, valueAlloca,
                                    elemSizeVal};
      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i1Ty},
                                       SymbolRefAttr::get(fn), args);
      rewriter.replaceOp(op, call.getResult());
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
      } else {
        // Source is not an integer type (e.g., a queue struct).
        // Stream unpacking from queue to dynamic array is not yet supported.
        return rewriter.notifyMatchFailure(
            op, "streaming unpack from queue/dynamic array source not yet "
                "supported - source must be converted to bits first");
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

/// Conversion for moore.stream_concat_mixed -> runtime function call.
/// This handles streaming concatenation with a mix of static bit vectors
/// and a dynamic array/queue in the middle. Supports arbitrary-width static
/// prefixes and suffixes by storing them in byte arrays.
struct StreamConcatMixedOpConversion
    : public RuntimeCallConversionBase<StreamConcatMixedOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(StreamConcatMixedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto queueTy = getQueueStructType(ctx);

    // Get the operands
    auto staticPrefix = adaptor.getStaticPrefix();
    auto dynamicArray = adaptor.getDynamicArray();
    auto staticSuffix = adaptor.getStaticSuffix();
    bool isRightToLeft = op.getIsRightToLeft();
    int32_t sliceSize = op.getSliceSize();

    // Calculate total prefix bits and collect widths
    int64_t prefixBits = 0;
    SmallVector<int64_t> prefixWidths;
    for (auto val : staticPrefix) {
      if (val.getType().isIntOrFloat()) {
        int64_t width = val.getType().getIntOrFloatBitWidth();
        prefixWidths.push_back(width);
        prefixBits += width;
      }
    }

    // Calculate total suffix bits and collect widths
    int64_t suffixBits = 0;
    SmallVector<int64_t> suffixWidths;
    for (auto val : staticSuffix) {
      if (val.getType().isIntOrFloat()) {
        int64_t width = val.getType().getIntOrFloatBitWidth();
        suffixWidths.push_back(width);
        suffixBits += width;
      }
    }

    // Allocate byte arrays for prefix and suffix (supports arbitrary widths)
    int64_t prefixBytes = (prefixBits + 7) / 8;
    int64_t suffixBytes = (suffixBits + 7) / 8;

    // Allocate prefix byte array
    Value prefixAlloca;
    if (prefixBytes > 0) {
      auto prefixBytesConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(prefixBytes));
      prefixAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, i8Ty, prefixBytesConst);
      // Zero-initialize the array
      auto zero = LLVM::ConstantOp::create(rewriter, loc,
                                           rewriter.getI8IntegerAttr(0));
      auto isVolatile = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i1Ty, 0));
      LLVM::MemsetOp::create(rewriter, loc, prefixAlloca, zero, prefixBytesConst,
                             isVolatile);

      // Store prefix values into the byte array
      int64_t bitOffset = 0;
      for (size_t i = 0; i < staticPrefix.size(); ++i) {
        Value val = staticPrefix[i];
        int64_t width = prefixWidths[i];

        // Store the value byte by byte
        int64_t numBytesForVal = (width + 7) / 8;
        for (int64_t b = 0; b < numBytesForVal; ++b) {
          int64_t byteIdx = (bitOffset + b * 8) / 8;
          int64_t bitWithinByte = (bitOffset + b * 8) % 8;

          // Extract the byte from the value
          Value byteVal;
          if (width <= 8) {
            byteVal = val;
            if (width < 8)
              byteVal = arith::ExtUIOp::create(rewriter, loc, i8Ty, byteVal);
          } else {
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc,
                rewriter.getIntegerAttr(val.getType(), b * 8));
            Value shifted = LLVM::LShrOp::create(rewriter, loc, val, shiftAmt);
            byteVal = arith::TruncIOp::create(rewriter, loc, i8Ty, shifted);
          }

          // If not byte-aligned, we need to OR with existing content
          if (bitWithinByte == 0) {
            auto idx = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(byteIdx));
            auto gep =
                LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, prefixAlloca,
                                    ValueRange{idx});
            LLVM::StoreOp::create(rewriter, loc, byteVal, gep);
          } else {
            // Shift and OR (more complex case - for now assume byte-aligned)
            auto idx = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(byteIdx));
            auto gep =
                LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, prefixAlloca,
                                    ValueRange{idx});
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI8IntegerAttr(bitWithinByte));
            Value shiftedByte =
                LLVM::ShlOp::create(rewriter, loc, byteVal, shiftAmt);
            Value existing = LLVM::LoadOp::create(rewriter, loc, i8Ty, gep);
            Value merged =
                LLVM::OrOp::create(rewriter, loc, existing, shiftedByte);
            LLVM::StoreOp::create(rewriter, loc, merged, gep);
          }
        }
        bitOffset += width;
      }
    } else {
      // Create a null pointer for empty prefix
      prefixAlloca = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Allocate suffix byte array
    Value suffixAlloca;
    if (suffixBytes > 0) {
      auto suffixBytesConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(suffixBytes));
      suffixAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, i8Ty, suffixBytesConst);
      // Zero-initialize the array
      auto zero = LLVM::ConstantOp::create(rewriter, loc,
                                           rewriter.getI8IntegerAttr(0));
      auto isVolatile = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i1Ty, 0));
      LLVM::MemsetOp::create(rewriter, loc, suffixAlloca, zero, suffixBytesConst,
                             isVolatile);

      // Store suffix values into the byte array
      int64_t bitOffset = 0;
      for (size_t i = 0; i < staticSuffix.size(); ++i) {
        Value val = staticSuffix[i];
        int64_t width = suffixWidths[i];

        // Store the value byte by byte
        int64_t numBytesForVal = (width + 7) / 8;
        for (int64_t b = 0; b < numBytesForVal; ++b) {
          int64_t byteIdx = (bitOffset + b * 8) / 8;
          int64_t bitWithinByte = (bitOffset + b * 8) % 8;

          // Extract the byte from the value
          Value byteVal;
          if (width <= 8) {
            byteVal = val;
            if (width < 8)
              byteVal = arith::ExtUIOp::create(rewriter, loc, i8Ty, byteVal);
          } else {
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc,
                rewriter.getIntegerAttr(val.getType(), b * 8));
            Value shifted = LLVM::LShrOp::create(rewriter, loc, val, shiftAmt);
            byteVal = arith::TruncIOp::create(rewriter, loc, i8Ty, shifted);
          }

          // If not byte-aligned, we need to OR with existing content
          if (bitWithinByte == 0) {
            auto idx = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(byteIdx));
            auto gep =
                LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, suffixAlloca,
                                    ValueRange{idx});
            LLVM::StoreOp::create(rewriter, loc, byteVal, gep);
          } else {
            // Shift and OR
            auto idx = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(byteIdx));
            auto gep =
                LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, suffixAlloca,
                                    ValueRange{idx});
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI8IntegerAttr(bitWithinByte));
            Value shiftedByte =
                LLVM::ShlOp::create(rewriter, loc, byteVal, shiftAmt);
            Value existing = LLVM::LoadOp::create(rewriter, loc, i8Ty, gep);
            Value merged =
                LLVM::OrOp::create(rewriter, loc, existing, shiftedByte);
            LLVM::StoreOp::create(rewriter, loc, merged, gep);
          }
        }
        bitOffset += width;
      }
    } else {
      // Create a null pointer for empty suffix
      suffixAlloca = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Store dynamic array to alloca and pass pointer
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto arrayAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, dynamicArray, arrayAlloca);

    // Get element bit width from the dynamic array
    auto inputType = op.getDynamicArray().getType();
    Type elementType;
    if (auto queueType = dyn_cast<QueueType>(inputType)) {
      elementType = queueType.getElementType();
    } else if (auto dynArrayType = dyn_cast<OpenUnpackedArrayType>(inputType)) {
      elementType = dynArrayType.getElementType();
    }

    int64_t elementBitWidth = 8; // default
    if (elementType) {
      if (auto intType = dyn_cast<IntType>(elementType)) {
        elementBitWidth = intType.getWidth();
      } else if (auto packedType = dyn_cast<PackedType>(elementType)) {
        elementBitWidth = packedType.getBitSize().value_or(8);
      }
    }

    // Create constants
    auto prefixBitsConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(prefixBits));
    auto suffixBitsConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(suffixBits));
    auto elemWidthConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(elementBitWidth));
    auto sliceSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(sliceSize));
    auto directionConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i1Ty, isRightToLeft ? 1 : 0));

    // Call __moore_stream_concat_mixed(prefix_ptr, prefix_bits, array_ptr,
    // elem_width, suffix_ptr, suffix_bits, slice_size, direction) -> queue
    // The new signature uses byte array pointers instead of i64 values
    auto fnTy = LLVM::LLVMFunctionType::get(
        queueTy, {ptrTy, i32Ty, ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i1Ty});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_stream_concat_mixed", fnTy);

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(fn),
        ValueRange{prefixAlloca, prefixBitsConst, arrayAlloca, elemWidthConst,
                   suffixAlloca, suffixBitsConst, sliceSizeConst,
                   directionConst});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Conversion for moore.stream_unpack_mixed -> inline unpacking with runtime
/// call for the dynamic array portion. This handles streaming unpacking into
/// a mix of static lvalue references and a dynamic array/queue reference.
/// Supports arbitrary-width static prefixes and suffixes by using byte arrays.
/// For static targets (prefix/suffix), we extract bits and drive using llhd.
/// For the dynamic array target, we call the stream_unpack runtime function.
struct StreamUnpackMixedOpConversion
    : public RuntimeCallConversionBase<StreamUnpackMixedOp> {
  using RuntimeCallConversionBase::RuntimeCallConversionBase;

  LogicalResult
  matchAndRewrite(StreamUnpackMixedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto queueTy = getQueueStructType(ctx);

    auto staticPrefixRefs = adaptor.getStaticPrefixRefs();
    auto dynamicArrayRef = adaptor.getDynamicArrayRef();
    auto staticSuffixRefs = adaptor.getStaticSuffixRefs();
    auto src = adaptor.getSrc();
    bool isRightToLeft = op.getIsRightToLeft();
    int32_t sliceSize = op.getSliceSize();

    // Calculate prefix and suffix bit widths from original Moore types
    SmallVector<int64_t> prefixWidths;
    int64_t totalPrefixBits = 0;
    for (auto ref : op.getStaticPrefixRefs()) {
      if (auto refType = dyn_cast<RefType>(ref.getType())) {
        if (auto intType = dyn_cast<IntType>(refType.getNestedType())) {
          prefixWidths.push_back(intType.getWidth());
          totalPrefixBits += intType.getWidth();
        }
      }
    }

    SmallVector<int64_t> suffixWidths;
    int64_t totalSuffixBits = 0;
    for (auto ref : op.getStaticSuffixRefs()) {
      if (auto refType = dyn_cast<RefType>(ref.getType())) {
        if (auto intType = dyn_cast<IntType>(refType.getNestedType())) {
          suffixWidths.push_back(intType.getWidth());
          totalSuffixBits += intType.getWidth();
        }
      }
    }

    // Get element bit width from the dynamic array ref
    int64_t elementBitWidth = 8;
    auto dstType = op.getDynamicArrayRef().getType();
    if (auto refType = dyn_cast<RefType>(dstType)) {
      auto nestedType = refType.getNestedType();
      Type elementType;
      if (auto queueType = dyn_cast<QueueType>(nestedType)) {
        elementType = queueType.getElementType();
      } else if (auto dynArrayType =
                     dyn_cast<OpenUnpackedArrayType>(nestedType)) {
        elementType = dynArrayType.getElementType();
      }
      if (elementType) {
        if (auto intType = dyn_cast<IntType>(elementType)) {
          elementBitWidth = intType.getWidth();
        }
      }
    }

    // Store source queue to alloca for the runtime call
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto srcAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);

    if (src.getType() == queueTy) {
      LLVM::StoreOp::create(rewriter, loc, src, srcAlloca);
    } else {
      // Source should be a queue - if not, this is an error
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      os << src.getType();
      return rewriter.notifyMatchFailure(
          op, "mixed streaming unpack requires queue source, got " + typeStr);
    }

    // Allocate byte arrays for extracted prefix and suffix
    int64_t prefixBytes = (totalPrefixBits + 7) / 8;
    int64_t suffixBytes = (totalSuffixBits + 7) / 8;

    // Allocate prefix byte array
    Value prefixAlloca;
    if (prefixBytes > 0) {
      auto prefixBytesConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(prefixBytes));
      prefixAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, i8Ty, prefixBytesConst);
      // Zero-initialize the array
      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8IntegerAttr(0));
      auto isVolatile = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i1Ty, 0));
      LLVM::MemsetOp::create(rewriter, loc, prefixAlloca, zero,
                             prefixBytesConst, isVolatile);
    } else {
      prefixAlloca = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Allocate suffix byte array
    Value suffixAlloca;
    if (suffixBytes > 0) {
      auto suffixBytesConst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(suffixBytes));
      suffixAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, i8Ty, suffixBytesConst);
      // Zero-initialize the array
      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8IntegerAttr(0));
      auto isVolatile = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i1Ty, 0));
      LLVM::MemsetOp::create(rewriter, loc, suffixAlloca, zero,
                             suffixBytesConst, isVolatile);
    } else {
      suffixAlloca = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Call runtime to extract prefix, middle, and suffix
    auto prefixBitsConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(totalPrefixBits));
    auto suffixBitsConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(totalSuffixBits));
    auto elemWidthConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(elementBitWidth));
    auto sliceSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(sliceSize));
    auto directionConst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i1Ty, isRightToLeft ? 1 : 0));

    // Create result struct type for arbitrary-width:
    // { ptr prefix_data, i32 prefix_bytes, queue middle, ptr suffix_data, i32
    // suffix_bytes }
    auto resultStructTy = LLVM::LLVMStructType::getLiteral(
        ctx, {ptrTy, i32Ty, queueTy, ptrTy, i32Ty});

    // Call __moore_stream_unpack_mixed_extract(src_ptr, prefix_bits,
    // elem_width, suffix_bits, slice_size, direction) -> { prefix_ptr,
    // prefix_bytes, middle, suffix_ptr, suffix_bytes }
    auto fnTy = LLVM::LLVMFunctionType::get(
        resultStructTy, {ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});
    auto fn = getOrCreateRuntimeFunc(
        mod, rewriter, "__moore_stream_unpack_mixed_extract", fnTy);

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{resultStructTy}, SymbolRefAttr::get(fn),
        ValueRange{srcAlloca, prefixBitsConst, elemWidthConst, suffixBitsConst,
                   sliceSizeConst, directionConst});

    Value resultStruct = call.getResult();

    // Extract prefix data pointer
    Value prefixDataPtr = LLVM::ExtractValueOp::create(
        rewriter, loc, ptrTy, resultStruct, ArrayRef<int64_t>{0});

    // Extract middle queue
    Value middleQueue = LLVM::ExtractValueOp::create(
        rewriter, loc, queueTy, resultStruct, ArrayRef<int64_t>{2});

    // Extract suffix data pointer
    Value suffixDataPtr = LLVM::ExtractValueOp::create(
        rewriter, loc, ptrTy, resultStruct, ArrayRef<int64_t>{3});

    // Create 0ns 0d 1e delay for blocking assignments
    Value delay = llhd::ConstantTimeOp::create(
        rewriter, loc, llhd::TimeAttr::get(ctx, 0U, "ns", 0, 1));

    // Helper function to load bytes and compose a value of given width
    auto loadValueFromByteArray = [&](Value byteArrayPtr, int64_t bitOffset,
                                      int64_t width) -> Value {
      int64_t numBytes = (width + 7) / 8;
      auto targetType = IntegerType::get(ctx, width);

      if (width <= 64) {
        // For values up to 64 bits, compose into an i64 then truncate
        Value result = LLVM::ConstantOp::create(rewriter, loc,
                                                rewriter.getI64IntegerAttr(0));

        for (int64_t b = 0; b < numBytes; ++b) {
          int64_t byteIdx = (bitOffset / 8) + b;
          auto idx = LLVM::ConstantOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(byteIdx));
          auto gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty,
                                         byteArrayPtr, ValueRange{idx});
          Value byteVal = LLVM::LoadOp::create(rewriter, loc, i8Ty, gep);

          // Extend to i64 and shift into position
          Value extended =
              arith::ExtUIOp::create(rewriter, loc, i64Ty, byteVal);
          auto shiftAmt = LLVM::ConstantOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(b * 8));
          Value shifted =
              LLVM::ShlOp::create(rewriter, loc, extended, shiftAmt);
          result = LLVM::OrOp::create(rewriter, loc, result, shifted);
        }

        // Truncate to target width
        return arith::TruncIOp::create(rewriter, loc, targetType, result);
      } else {
        // For values > 64 bits, load into the target type directly
        // This requires building the value in multiple 64-bit chunks
        Value result = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(targetType, 0));

        int64_t chunks = (width + 63) / 64;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
          int64_t chunkStartByte = (bitOffset / 8) + chunk * 8;
          int64_t chunkBits = std::min<int64_t>(64, width - chunk * 64);
          int64_t chunkBytes = (chunkBits + 7) / 8;

          Value chunkVal = LLVM::ConstantOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));

          for (int64_t b = 0; b < chunkBytes; ++b) {
            auto idx = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(chunkStartByte + b));
            auto gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty,
                                           byteArrayPtr, ValueRange{idx});
            Value byteVal = LLVM::LoadOp::create(rewriter, loc, i8Ty, gep);

            Value extended =
                arith::ExtUIOp::create(rewriter, loc, i64Ty, byteVal);
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(b * 8));
            Value shifted =
                LLVM::ShlOp::create(rewriter, loc, extended, shiftAmt);
            chunkVal = LLVM::OrOp::create(rewriter, loc, chunkVal, shifted);
          }

          // Extend chunk to target type and shift into position
          Value extendedChunk =
              arith::ExtUIOp::create(rewriter, loc, targetType, chunkVal);
          if (chunk > 0) {
            auto shiftAmt = LLVM::ConstantOp::create(
                rewriter, loc,
                rewriter.getIntegerAttr(targetType, chunk * 64));
            extendedChunk =
                LLVM::ShlOp::create(rewriter, loc, extendedChunk, shiftAmt);
          }
          result = LLVM::OrOp::create(rewriter, loc, result, extendedChunk);
        }

        return result;
      }
    };

    // Assign prefix values using llhd.drive
    int64_t bitOffset = 0;
    for (size_t i = 0; i < staticPrefixRefs.size(); ++i) {
      Value ref = staticPrefixRefs[i];
      int64_t width = prefixWidths[i];

      // Load value from prefix byte array
      Value extracted = loadValueFromByteArray(prefixDataPtr, bitOffset, width);

      // Drive the target using llhd.drive
      llhd::DriveOp::create(rewriter, loc, ref, extracted, delay, Value{});

      bitOffset += width;
    }

    // Assign the dynamic array
    LLVM::StoreOp::create(rewriter, loc, middleQueue, dynamicArrayRef);

    // Assign suffix values using llhd.drive
    bitOffset = 0;
    for (size_t i = 0; i < staticSuffixRefs.size(); ++i) {
      Value ref = staticSuffixRefs[i];
      int64_t width = suffixWidths[i];

      // Load value from suffix byte array
      Value extracted = loadValueFromByteArray(suffixDataPtr, bitOffset, width);

      // Drive the target using llhd.drive
      llhd::DriveOp::create(rewriter, loc, ref, extracted, delay, Value{});

      bitOffset += width;
    }

    // Free the allocated prefix and suffix data
    auto freeFnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
    auto freeFn = getOrCreateRuntimeFunc(mod, rewriter, "free", freeFnTy);
    if (totalPrefixBits > 0) {
      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(freeFn),
                           ValueRange{prefixDataPtr});
    }
    if (totalSuffixBits > 0) {
      LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(freeFn),
                           ValueRange{suffixDataPtr});
    }

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
      : OpConversionPattern(tc, ctx, /*benefit=*/10), classCache(cache) {}

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

  /// Lower array.locator for fixed-size arrays with a simple predicate.
  /// This wraps the fixed-size array as a queue and calls the runtime function.
  /// This is simpler than generating inline loops and avoids issues with
  /// block manipulation in the dialect conversion framework.
  LogicalResult
  lowerFixedArrayWithSimplePredicate(ArrayLocatorOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, MLIRContext *ctx,
                                     ModuleOp mod, CmpMode cmpMode,
                                     Value constValue) const {
    // Convert the region types so the framework doesn't complain about
    // unconverted operations in the region.
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter)))
      return rewriter.notifyMatchFailure(op, "failed to convert region types");
    auto queueTy = getQueueStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    // Get array info
    auto arrayType = dyn_cast<UnpackedArrayType>(op.getArray().getType());
    if (!arrayType)
      return rewriter.notifyMatchFailure(op, "expected UnpackedArrayType");

    Type mooreElemType = arrayType.getElementType();
    int64_t arraySize = arrayType.getSize();

    Type elemType = typeConverter->convertType(mooreElemType);
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "failed to convert element type");

    int64_t elemSizeBytes = getTypeSize(elemType);

    // Convert hw::ArrayType to LLVM array and store to memory
    auto hwArrayTy = dyn_cast<hw::ArrayType>(adaptor.getArray().getType());
    if (!hwArrayTy)
      return rewriter.notifyMatchFailure(op, "expected hw::ArrayType");

    Type llvmArrayType = convertToLLVMType(hwArrayTy);
    Type llvmElemType = convertToLLVMType(elemType);

    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));

    // Allocate stack space for the array
    auto arrayAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayType, one);

    // Cast the hw.array value to llvm.array for storage
    auto castOp = UnrealizedConversionCastOp::create(rewriter, loc, llvmArrayType,
                                           adaptor.getArray());
    Value llvmArrayValue = castOp.getResult(0);
    LLVM::StoreOp::create(rewriter, loc, llvmArrayValue, arrayAlloca);

    // Create a temporary queue struct {ptr, len} pointing to the array
    Value arrayLen = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(arraySize));
    Value tempQueue = LLVM::UndefOp::create(rewriter, loc, queueTy);
    tempQueue = LLVM::InsertValueOp::create(rewriter, loc, tempQueue, arrayAlloca,
                                            ArrayRef<int64_t>{0});
    tempQueue = LLVM::InsertValueOp::create(rewriter, loc, tempQueue, arrayLen,
                                            ArrayRef<int64_t>{1});

    // Allocate stack space for the temp queue (runtime function takes ptr)
    auto tempQueueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, queueTy, one);
    LLVM::StoreOp::create(rewriter, loc, tempQueue, tempQueueAlloca);

    // Allocate stack space for the comparison value
    auto cmpValueAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);

    // Get the comparison value (convert constant to LLVM type)
    Value llvmCmpValue;
    if (auto constOp = constValue.getDefiningOp<ConstantOp>()) {
      APInt constAPInt = constOp.getValue().toAPInt(false);
      llvmCmpValue = LLVM::ConstantOp::create(rewriter, loc, llvmElemType,
                                              constAPInt.getSExtValue());
    } else {
      Value remapped = rewriter.getRemappedValue(constValue);
      if (remapped) {
        if (remapped.getType() != llvmElemType) {
          llvmCmpValue =
              UnrealizedConversionCastOp::create(rewriter, loc, llvmElemType,
                                                 remapped)
                  .getResult(0);
        } else {
          llvmCmpValue = remapped;
        }
      } else {
        return rewriter.notifyMatchFailure(
            op, "comparison value must be a constant or already converted");
      }
    }
    LLVM::StoreOp::create(rewriter, loc, llvmCmpValue, cmpValueAlloca);

    // Prepare runtime function arguments
    Value elemSize = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(elemSizeBytes));
    Value cmpModeVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int>(cmpMode)));
    Value locatorMode = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int>(op.getMode())));
    Value returnIndices = LLVM::ConstantOp::create(
        rewriter, loc, i1Ty, rewriter.getBoolAttr(op.getIndexed()));

    // Call __moore_array_find_cmp(queue*, elemSize, value*, cmpMode, locatorMode, returnIndices)
    auto fnTy = LLVM::LLVMFunctionType::get(
        queueTy, {ptrTy, i64Ty, ptrTy, i32Ty, i32Ty, i1Ty});
    auto findFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_array_find_cmp", fnTy);

    auto callOp = LLVM::CallOp::create(
        rewriter, loc, TypeRange{queueTy}, SymbolRefAttr::get(findFn),
        ValueRange{tempQueueAlloca, elemSize, cmpValueAlloca, cmpModeVal,
                   locatorMode, returnIndices});
    Value result = callOp.getResult();

    // Replace the op with the result.
    rewriter.replaceOp(op, result);
    return success();
  }

  /// Convert a single Moore operation to its LLVM/arith equivalent inline.
  /// This handles the common operations that appear in array locator predicates.
  /// The valueMap maps Moore values to their converted LLVM/arith values.
  /// Returns the converted result value, or nullptr if conversion failed.
  Value convertMooreOpInline(Operation *mooreOp,
                             ConversionPatternRewriter &rewriter,
                             Location loc, MLIRContext *ctx,
                             DenseMap<Value, Value> &valueMap,
                             ModuleOp mod) const {
    // Helper to get converted operand. This handles:
    // 1. Values already in valueMap (converted predicate ops)
    // 2. Values remapped by the rewriter (external values already converted)
    // 3. External Moore operations that need recursive conversion
    // 4. Block arguments and external values that need materialization
    std::function<Value(Value)> getConvertedOperand;
    getConvertedOperand = [&](Value mooreVal) -> Value {
      // Check if already in our value map
      auto it = valueMap.find(mooreVal);
      if (it != valueMap.end())
        return it->second;

      // Try to get remapped value from rewriter (for already-converted values)
      Value remapped = rewriter.getRemappedValue(mooreVal);
      if (remapped)
        return remapped;

      // Special handling for block arguments from enclosing functions.
      // When a function's signature is converted, the old block arguments are
      // replaced with new ones. Operations in nested regions (like predicate
      // bodies) or operations referencing the function's block arguments still
      // reference the old arguments, but those have been invalidated. We need
      // to find the new block argument at the corresponding position in the
      // converted function.
      //
      // This is critical for class method contexts where the 'this' pointer
      // is a block argument that gets remapped during function signature
      // conversion.
      if (auto blockArg = dyn_cast<BlockArgument>(mooreVal)) {
        // Get the parent function's entry block to find the new block argument
        auto funcOp = mooreOp->getParentOfType<func::FuncOp>();
        if (funcOp) {
          Block *funcEntryBlock = &funcOp.getBody().front();
          unsigned argIndex = blockArg.getArgNumber();
          if (argIndex < funcEntryBlock->getNumArguments()) {
            Value newBlockArg = funcEntryBlock->getArgument(argIndex);
            valueMap[mooreVal] = newBlockArg;
            return newBlockArg;
          }
        }
      }

      // For external values that are results of Moore operations,
      // recursively convert the defining operation
      if (Operation *definingOp = mooreVal.getDefiningOp()) {
        // Check if this is a Moore operation that we can convert inline
        if (isa<ClassPropertyRefOp, ReadOp, ConstantOp, DynExtractOp,
                ArraySizeOp, EqOp, NeOp, SubOp, AddOp, AndOp, OrOp,
                ConversionOp, StructExtractOp, StringCmpOp,
                ClassHandleCmpOp, WildcardEqOp, WildcardNeOp,
                IntToLogicOp, ClassNullOp, StringToLowerOp, StringToUpperOp,
                VTableLoadMethodOp>(definingOp) ||
            isa<func::CallOp, func::CallIndirectOp>(definingOp)) {
          // Recursively convert the defining operation
          // First, ensure all operands of the defining op are converted
          for (Value operand : definingOp->getOperands()) {
            if (valueMap.find(operand) == valueMap.end()) {
              Value converted = getConvertedOperand(operand);
              if (converted)
                valueMap[operand] = converted;
            }
          }

          // Now convert the defining operation itself
          Value result = convertMooreOpInline(definingOp, rewriter, loc, ctx,
                                              valueMap, mod);
          if (result) {
            valueMap[mooreVal] = result;
            return result;
          }
        }
      }

      // For external values (defined outside the predicate block),
      // try to materialize conversion
      Type targetType = typeConverter->convertType(mooreVal.getType());
      if (targetType) {
        // Use materializeTargetConversion which creates an UnrealizedConversionCast
        Value converted = typeConverter->materializeTargetConversion(
            rewriter, loc, targetType, mooreVal);
        if (converted) {
          valueMap[mooreVal] = converted;
          return converted;
        }
      }

      // Last resort: create an unrealized conversion cast
      if (targetType) {
        Value cast = UnrealizedConversionCastOp::create(rewriter, loc, targetType,
                                                        mooreVal)
            .getResult(0);
        valueMap[mooreVal] = cast;
        return cast;
      }

      return nullptr;
    };

    // Handle moore.constant
    if (auto constOp = dyn_cast<ConstantOp>(mooreOp)) {
      Type resultType = typeConverter->convertType(constOp.getResult().getType());
      if (!resultType)
        return nullptr;
      APInt constAPInt = constOp.getValue().toAPInt(false);
      // Check if result type is a 4-state struct
      if (auto structTy = dyn_cast<hw::StructType>(resultType)) {
        // 4-state value: create struct with {value, unknown=0}
        auto intWidth = constAPInt.getBitWidth();
        Value valPart = hw::ConstantOp::create(rewriter, loc, constAPInt);
        APInt zeroAPInt(intWidth, 0);
        Value unkPart = hw::ConstantOp::create(rewriter, loc, zeroAPInt);
        return hw::StructCreateOp::create(rewriter, loc, structTy,
                                          ValueRange{valPart, unkPart});
      }
      return hw::ConstantOp::create(rewriter, loc, constAPInt);
    }

    // Handle moore.read
    if (auto readOp = dyn_cast<ReadOp>(mooreOp)) {
      Value input = getConvertedOperand(readOp.getInput());
      if (!input)
        return nullptr;
      Type resultType = typeConverter->convertType(readOp.getResult().getType());
      if (!resultType)
        return nullptr;
      // If input is LLVM pointer, use LLVM load
      if (isa<LLVM::LLVMPointerType>(input.getType())) {
        Type llvmResultType = convertToLLVMType(resultType);
        Value loaded = LLVM::LoadOp::create(rewriter, loc, llvmResultType, input);
        if (resultType != llvmResultType) {
          loaded = UnrealizedConversionCastOp::create(rewriter, loc,
                                                      resultType, loaded)
                       .getResult(0);
        }
        return loaded;
      }
      // Otherwise use llhd.prb
      return llhd::ProbeOp::create(rewriter, loc, input);
    }

    // Handle moore.class.property_ref
    if (auto propRefOp = dyn_cast<ClassPropertyRefOp>(mooreOp)) {
      Value instance = getConvertedOperand(propRefOp.getInstance());
      if (!instance)
        return nullptr;

      // Resolve class struct info
      auto classRefTy = cast<ClassHandleType>(propRefOp.getInstance().getType());
      SymbolRefAttr classSym = classRefTy.getClassSym();
      if (failed(resolveClassStructBody(mod, classSym, *typeConverter, classCache)))
        return nullptr;

      auto structInfo = classCache.getStructInfo(classSym);
      if (!structInfo)
        return nullptr;

      auto propSym = propRefOp.getProperty();
      auto pathOpt = structInfo->getFieldPath(propSym);
      if (!pathOpt)
        return nullptr;

      auto i32Ty = IntegerType::get(ctx, 32);
      SmallVector<Value> idxVals;
      // First index 0 is needed to dereference the pointer to the struct.
      // Subsequent indices navigate into the struct fields.
      idxVals.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
      for (unsigned idx : *pathOpt)
        idxVals.push_back(LLVM::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(idx)));

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto gep = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                     structInfo->classBody, instance, idxVals);
      return gep;
    }

    // Handle moore.dyn_extract (dynamic array element extraction)
    if (auto dynExtOp = dyn_cast<DynExtractOp>(mooreOp)) {
      Value input = getConvertedOperand(dynExtOp.getInput());
      Value index = getConvertedOperand(dynExtOp.getLowBit());
      if (!input || !index)
        return nullptr;

      Type resultType = typeConverter->convertType(dynExtOp.getResult().getType());
      if (!resultType)
        return nullptr;

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);

      // Handle queue/dynamic array represented as LLVM struct {ptr, i64}
      if (auto structTy = dyn_cast<LLVM::LLVMStructType>(input.getType())) {
        // Check if this looks like a queue struct (has 2 elements: ptr and i64)
        auto body = structTy.getBody();
        if (body.size() == 2 && isa<LLVM::LLVMPointerType>(body[0]) &&
            body[1] == i64Ty) {
          // Extract data pointer from queue struct
          Value dataPtr = LLVM::ExtractValueOp::create(
              rewriter, loc, ptrTy, input, ArrayRef<int64_t>{0});

          // Extend index to i64 if needed
          if (index.getType() != i64Ty) {
            if (auto intTy = dyn_cast<IntegerType>(index.getType())) {
              if (intTy.getWidth() < 64)
                index = arith::ExtSIOp::create(rewriter, loc, i64Ty, index);
              else if (intTy.getWidth() > 64)
                index = arith::TruncIOp::create(rewriter, loc, i64Ty, index);
            }
          }

          // GEP to the element
          Type llvmResultType = convertToLLVMType(resultType);
          Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                              llvmResultType, dataPtr, index);
          Value loaded = LLVM::LoadOp::create(rewriter, loc, llvmResultType, elemPtr);
          if (resultType != llvmResultType) {
            loaded = UnrealizedConversionCastOp::create(rewriter, loc,
                                                        resultType, loaded)
                         .getResult(0);
          }
          return loaded;
        }
      }

      // Handle queue/dynamic array access via LLVM pointer
      if (isa<LLVM::LLVMPointerType>(input.getType())) {
        // Extend index to i64 if needed
        if (index.getType() != i64Ty) {
          if (auto intTy = dyn_cast<IntegerType>(index.getType())) {
            if (intTy.getWidth() < 64)
              index = arith::ExtSIOp::create(rewriter, loc, i64Ty, index);
            else if (intTy.getWidth() > 64)
              index = arith::TruncIOp::create(rewriter, loc, i64Ty, index);
          }
        }

        // GEP to the element
        Type llvmResultType = convertToLLVMType(resultType);
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                            llvmResultType, input, index);
        Value loaded = LLVM::LoadOp::create(rewriter, loc, llvmResultType, elemPtr);
        if (resultType != llvmResultType) {
          loaded = UnrealizedConversionCastOp::create(rewriter, loc,
                                                      resultType, loaded)
                       .getResult(0);
        }
        return loaded;
      }

      // Handle hw::ArrayType
      if (auto hwArrayTy = dyn_cast<hw::ArrayType>(input.getType())) {
        // Adjust index width to match array size requirements
        unsigned idxWidth = llvm::Log2_64_Ceil(hwArrayTy.getNumElements());
        if (idxWidth == 0)
          idxWidth = 1; // at least 1 bit needed
        index = adjustIntegerWidth(rewriter, index, idxWidth, loc);
        return hw::ArrayGetOp::create(rewriter, loc, input, index);
      }

      return nullptr;
    }

    // Handle moore.array.size
    if (auto arraySizeOp = dyn_cast<ArraySizeOp>(mooreOp)) {
      Value input = getConvertedOperand(arraySizeOp.getArray());
      if (!input)
        return nullptr;

      Type resultType = typeConverter->convertType(arraySizeOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Handle queue/dynamic array (LLVM struct {ptr, i64})
      if (auto structTy = dyn_cast<LLVM::LLVMStructType>(input.getType())) {
        auto i64Ty = IntegerType::get(ctx, 64);
        Value size = LLVM::ExtractValueOp::create(
            rewriter, loc, i64Ty, input, ArrayRef<int64_t>{1});
        // Truncate to result type if needed
        if (auto intResultTy = dyn_cast<IntegerType>(resultType)) {
          if (intResultTy.getWidth() < 64)
            size = arith::TruncIOp::create(rewriter, loc, resultType, size);
        }
        return size;
      }

      // Handle hw::ArrayType
      if (auto hwArrayTy = dyn_cast<hw::ArrayType>(input.getType())) {
        APInt sizeVal(cast<IntegerType>(resultType).getWidth(),
                      hwArrayTy.getNumElements());
        return hw::ConstantOp::create(rewriter, loc, sizeVal);
      }

      return nullptr;
    }

    // Handle moore.eq
    if (auto eqOp = dyn_cast<EqOp>(mooreOp)) {
      Value lhs = getConvertedOperand(eqOp.getLhs());
      Value rhs = getConvertedOperand(eqOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      Type resultType = typeConverter->convertType(eqOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Handle 4-state struct comparison
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);

        Value cmpVal = comb::ICmpOp::create(rewriter, loc,
                                            comb::ICmpPredicate::eq, lhsVal, rhsVal);
        Value zeroUnk = hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
        Value lhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                               comb::ICmpPredicate::ne, lhsUnk, zeroUnk);
        Value rhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                               comb::ICmpPredicate::ne, rhsUnk, zeroUnk);
        Value hasUnk = comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);
        Value zero = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
        Value resultVal = comb::MuxOp::create(rewriter, loc, hasUnk, zero, cmpVal);
        return createFourStateStruct(rewriter, loc, resultVal, hasUnk);
      }

      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::eq, lhs, rhs);
    }

    // Handle moore.ne
    if (auto neOp = dyn_cast<NeOp>(mooreOp)) {
      Value lhs = getConvertedOperand(neOp.getLhs());
      Value rhs = getConvertedOperand(neOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      Type resultType = typeConverter->convertType(neOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Handle 4-state struct comparison
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);

        Value cmpVal = comb::ICmpOp::create(rewriter, loc,
                                            comb::ICmpPredicate::ne, lhsVal, rhsVal);
        Value zeroUnk = hw::ConstantOp::create(rewriter, loc, lhsUnk.getType(), 0);
        Value lhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                               comb::ICmpPredicate::ne, lhsUnk, zeroUnk);
        Value rhsHasUnk = comb::ICmpOp::create(rewriter, loc,
                                               comb::ICmpPredicate::ne, rhsUnk, zeroUnk);
        Value hasUnk = comb::OrOp::create(rewriter, loc, lhsHasUnk, rhsHasUnk, false);
        Value zero = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
        Value resultVal = comb::MuxOp::create(rewriter, loc, hasUnk, zero, cmpVal);
        return createFourStateStruct(rewriter, loc, resultVal, hasUnk);
      }

      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::ne, lhs, rhs);
    }

    // Handle moore.sgt (signed greater than)
    if (auto sgtOp = dyn_cast<SgtOp>(mooreOp)) {
      Value lhs = getConvertedOperand(sgtOp.getLhs());
      Value rhs = getConvertedOperand(sgtOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      Type resultType = typeConverter->convertType(sgtOp.getResult().getType());
      if (!resultType)
        return nullptr;
      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::sgt, lhs, rhs);
    }

    // Handle moore.sge (signed greater than or equal)
    if (auto sgeOp = dyn_cast<SgeOp>(mooreOp)) {
      Value lhs = getConvertedOperand(sgeOp.getLhs());
      Value rhs = getConvertedOperand(sgeOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      Type resultType = typeConverter->convertType(sgeOp.getResult().getType());
      if (!resultType)
        return nullptr;
      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::sge, lhs, rhs);
    }

    // Handle moore.slt (signed less than)
    if (auto sltOp = dyn_cast<SltOp>(mooreOp)) {
      Value lhs = getConvertedOperand(sltOp.getLhs());
      Value rhs = getConvertedOperand(sltOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      Type resultType = typeConverter->convertType(sltOp.getResult().getType());
      if (!resultType)
        return nullptr;
      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::slt, lhs, rhs);
    }

    // Handle moore.sle (signed less than or equal)
    if (auto sleOp = dyn_cast<SleOp>(mooreOp)) {
      Value lhs = getConvertedOperand(sleOp.getLhs());
      Value rhs = getConvertedOperand(sleOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      Type resultType = typeConverter->convertType(sleOp.getResult().getType());
      if (!resultType)
        return nullptr;
      return comb::ICmpOp::create(rewriter, loc, resultType,
                                  comb::ICmpPredicate::sle, lhs, rhs);
    }

    // Handle moore.and (logical AND)
    if (auto andOp = dyn_cast<AndOp>(mooreOp)) {
      Value lhs = getConvertedOperand(andOp.getLhs());
      Value rhs = getConvertedOperand(andOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      // Handle 4-state struct: extract value parts and combine
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value result = comb::AndOp::create(rewriter, loc, lhsVal, rhsVal, false);
        // Return as 4-state struct with unknown = lhsUnk | rhsUnk
        Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
        Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);
        Value unk = comb::OrOp::create(rewriter, loc, lhsUnk, rhsUnk, false);
        return createFourStateStruct(rewriter, loc, result, unk);
      }

      return comb::AndOp::create(rewriter, loc, lhs, rhs, false);
    }

    // Handle moore.or (logical OR)
    if (auto orOp = dyn_cast<OrOp>(mooreOp)) {
      Value lhs = getConvertedOperand(orOp.getLhs());
      Value rhs = getConvertedOperand(orOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      // Handle 4-state struct: extract value parts and combine
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value result = comb::OrOp::create(rewriter, loc, lhsVal, rhsVal, false);
        // Return as 4-state struct with unknown = lhsUnk | rhsUnk
        Value lhsUnk = extractFourStateUnknown(rewriter, loc, lhs);
        Value rhsUnk = extractFourStateUnknown(rewriter, loc, rhs);
        Value unk = comb::OrOp::create(rewriter, loc, lhsUnk, rhsUnk, false);
        return createFourStateStruct(rewriter, loc, result, unk);
      }

      return comb::OrOp::create(rewriter, loc, lhs, rhs, false);
    }

    // Handle moore.conversion (type conversion)
    if (auto convOp = dyn_cast<ConversionOp>(mooreOp)) {
      Value input = getConvertedOperand(convOp.getInput());
      if (!input)
        return nullptr;
      Type resultType = typeConverter->convertType(convOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // If types match, just return input
      if (input.getType() == resultType)
        return input;

      // Handle integer width conversions
      auto srcIntTy = dyn_cast<IntegerType>(input.getType());
      auto dstIntTy = dyn_cast<IntegerType>(resultType);
      if (srcIntTy && dstIntTy) {
        if (srcIntTy.getWidth() < dstIntTy.getWidth())
          return arith::ExtSIOp::create(rewriter, loc, resultType, input);
        else if (srcIntTy.getWidth() > dstIntTy.getWidth())
          return arith::TruncIOp::create(rewriter, loc, resultType, input);
        return input;
      }

      // Use unrealized conversion cast as fallback
      return UnrealizedConversionCastOp::create(rewriter, loc, resultType, input)
                 .getResult(0);
    }

    // Handle moore.sub (subtraction for index calculation like $ - 1)
    if (auto subOp = dyn_cast<SubOp>(mooreOp)) {
      Value lhs = getConvertedOperand(subOp.getLhs());
      Value rhs = getConvertedOperand(subOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      return comb::SubOp::create(rewriter, loc, lhs, rhs, false);
    }

    // Handle moore.add
    if (auto addOp = dyn_cast<AddOp>(mooreOp)) {
      Value lhs = getConvertedOperand(addOp.getLhs());
      Value rhs = getConvertedOperand(addOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      return comb::AddOp::create(rewriter, loc, lhs, rhs, false);
    }

    // Handle moore.zext (zero extension)
    if (auto zextOp = dyn_cast<ZExtOp>(mooreOp)) {
      Value input = getConvertedOperand(zextOp.getInput());
      if (!input)
        return nullptr;
      Type resultType = typeConverter->convertType(zextOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // If types match, just return input
      if (input.getType() == resultType)
        return input;

      // Handle 4-state struct
      if (isFourStateStructType(input.getType())) {
        Value val = extractFourStateValue(rewriter, loc, input);
        Value unk = extractFourStateUnknown(rewriter, loc, input);
        if (auto dstIntTy = dyn_cast<IntegerType>(resultType)) {
          // Zero extend to target width
          auto srcIntTy = cast<IntegerType>(val.getType());
          if (srcIntTy.getWidth() < dstIntTy.getWidth()) {
            val = arith::ExtUIOp::create(rewriter, loc, resultType, val);
            unk = arith::ExtUIOp::create(rewriter, loc, resultType, unk);
          } else if (srcIntTy.getWidth() > dstIntTy.getWidth()) {
            val = arith::TruncIOp::create(rewriter, loc, resultType, val);
            unk = arith::TruncIOp::create(rewriter, loc, resultType, unk);
          }
          return createFourStateStruct(rewriter, loc, val, unk);
        }
      }

      // Standard integer extension
      auto srcIntTy = dyn_cast<IntegerType>(input.getType());
      auto dstIntTy = dyn_cast<IntegerType>(resultType);
      if (srcIntTy && dstIntTy) {
        if (srcIntTy.getWidth() < dstIntTy.getWidth())
          return arith::ExtUIOp::create(rewriter, loc, resultType, input);
        else if (srcIntTy.getWidth() > dstIntTy.getWidth())
          return arith::TruncIOp::create(rewriter, loc, resultType, input);
        return input;
      }

      // Use unrealized conversion cast as fallback
      return UnrealizedConversionCastOp::create(rewriter, loc, resultType, input)
                 .getResult(0);
    }

    // Handle moore.class.upcast (type casting in inheritance hierarchy)
    if (auto upcastOp = dyn_cast<ClassUpcastOp>(mooreOp)) {
      Value input = getConvertedOperand(upcastOp.getInstance());
      if (!input)
        return nullptr;
      Type resultType = typeConverter->convertType(upcastOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // If types match (opaque pointer mode), just return input
      if (input.getType() == resultType)
        return input;

      // Use unrealized conversion cast as fallback
      return UnrealizedConversionCastOp::create(rewriter, loc, resultType, input)
                 .getResult(0);
    }

    // Handle func.call_indirect (virtual method calls)
    if (auto callOp = dyn_cast<func::CallIndirectOp>(mooreOp)) {
      // Get the callee (function pointer)
      Value callee = getConvertedOperand(callOp.getCallee());
      if (!callee)
        return nullptr;

      // Convert all operands
      SmallVector<Value> convertedOperands;
      for (Value operand : callOp.getArgOperands()) {
        Value converted = getConvertedOperand(operand);
        if (!converted)
          return nullptr;
        convertedOperands.push_back(converted);
      }

      // If callee is an LLVM pointer (from vtable lookup), use LLVM call
      if (isa<LLVM::LLVMPointerType>(callee.getType())) {
        // Build LLVM function type from the converted operand/result types
        SmallVector<Type> inputTypes;
        for (Value operand : convertedOperands)
          inputTypes.push_back(operand.getType());

        SmallVector<Type> convResTypes;
        for (Type resType : callOp.getResultTypes()) {
          Type converted = typeConverter->convertType(resType);
          if (!converted)
            return nullptr;
          convResTypes.push_back(converted);
        }

        // Create LLVM function type
        Type llvmResType = convResTypes.empty()
                               ? LLVM::LLVMVoidType::get(ctx)
                               : convResTypes[0];
        auto llvmFnType = LLVM::LLVMFunctionType::get(llvmResType, inputTypes);

        // For LLVM indirect call, callee is the first operand
        SmallVector<Value> allOperands;
        allOperands.push_back(callee);
        allOperands.append(convertedOperands.begin(), convertedOperands.end());

        // Create LLVM call with the function pointer as first operand
        auto llvmCall = LLVM::CallOp::create(rewriter, loc, llvmFnType,
                                              allOperands);
        if (!convResTypes.empty())
          return llvmCall.getResult();

        return nullptr;
      }

      // For func-style function types, use func.call_indirect
      // Build the new function type from converted input/result types
      SmallVector<Type> inputTypes;
      for (Value operand : convertedOperands)
        inputTypes.push_back(operand.getType());

      SmallVector<Type> convResTypes;
      for (Type resType : callOp.getResultTypes()) {
        Type converted = typeConverter->convertType(resType);
        if (!converted)
          return nullptr;
        convResTypes.push_back(converted);
      }
      auto newFuncType = FunctionType::get(ctx, inputTypes, convResTypes);

      // If callee type doesn't match, cast it
      if (callee.getType() != newFuncType) {
        callee = UnrealizedConversionCastOp::create(rewriter, loc,
                                                     TypeRange{newFuncType},
                                                     ValueRange{callee})
                     .getResult(0);
      }

      auto newCallOp = func::CallIndirectOp::create(rewriter, loc, callee,
                                                     convertedOperands);
      if (newCallOp.getNumResults() > 0)
        return newCallOp.getResult(0);

      // For void calls, return a dummy value
      return nullptr;
    }

    // Handle moore.vtable.load_method (virtual method dispatch)
    // This operation loads a function pointer from the vtable for dynamic dispatch.
    if (auto vtableLoadOp = dyn_cast<VTableLoadMethodOp>(mooreOp)) {
      Value objPtr = getConvertedOperand(vtableLoadOp.getObject());
      if (!objPtr)
        return nullptr;

      // Get the class type from the object operand
      auto handleTy = cast<ClassHandleType>(vtableLoadOp.getObject().getType());
      auto classSym = handleTy.getClassSym();

      // Resolve the class struct info to get method-to-vtable-index mapping
      if (failed(resolveClassStructBody(mod, classSym, *typeConverter, classCache)))
        return nullptr;

      auto structInfoOpt = classCache.getStructInfo(classSym);
      if (!structInfoOpt)
        return nullptr;

      auto &structInfo = *structInfoOpt;

      // Get the method name and find its vtable index
      auto methodSym = vtableLoadOp.getMethodSym();
      StringRef methodName = methodSym.getLeafReference();

      auto indexIt = structInfo.methodToVtableIndex.find(methodName);
      if (indexIt == structInfo.methodToVtableIndex.end())
        return nullptr;

      unsigned vtableIndex = indexIt->second;

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get the class's struct type for GEP
      auto structTy = structInfo.classBody;

      // Build GEP path to the vtable pointer field.
      // The vtable pointer is at index 1 in the root class (after typeId).
      // For inheritance hierarchy A -> B -> C:
      //   C: { B: { A: { i32 typeId, ptr vtablePtr, ...}, ...}, ...}
      // We need to GEP through [0, 0, ..., 0, 1] to reach vtablePtr.
      SmallVector<Value> gepIndices;
      // First index is always 0 (pointer dereference)
      gepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
      // Navigate through base class chain
      for (int32_t i = 0; i < structInfo.inheritanceDepth; ++i) {
        gepIndices.push_back(LLVM::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)));
      }
      // Index to vtable pointer field (index 1 in root class)
      gepIndices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(ClassTypeCache::ClassStructInfo::vtablePtrFieldIndex)));

      // GEP to vtable pointer field
      Value vtablePtrPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, objPtr, gepIndices);

      // Load the vtable pointer
      Value vtablePtr = LLVM::LoadOp::create(rewriter, loc, ptrTy, vtablePtrPtr);

      // GEP into the vtable array at the method's index.
      // The vtable is an array of pointers, so we create a GEP that indexes
      // into the array. For LLVM GEP on an array type:
      // - First index (0) dereferences the pointer to the array
      // - Second index (vtableIndex) selects the element
      unsigned vtableSize = 0;
      for (const auto &kv : structInfo.methodToVtableIndex) {
        if (kv.second >= vtableSize)
          vtableSize = kv.second + 1;
      }
      auto vtableArrayTy = LLVM::LLVMArrayType::get(ptrTy, vtableSize > 0 ? vtableSize : 1);

      SmallVector<LLVM::GEPArg> vtableGepIndices;
      vtableGepIndices.push_back(static_cast<int64_t>(0));  // Dereference pointer
      vtableGepIndices.push_back(static_cast<int64_t>(vtableIndex));  // Array index

      Value funcPtrPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrTy, vtableArrayTy, vtablePtr, vtableGepIndices);

      // Load the function pointer from the vtable
      Value funcPtr = LLVM::LoadOp::create(rewriter, loc, ptrTy, funcPtrPtr);

      return funcPtr;
    }

    // Handle moore.string_cmp
    if (auto stringCmpOp = dyn_cast<StringCmpOp>(mooreOp)) {
      Value lhs = getConvertedOperand(stringCmpOp.getLhs());
      Value rhs = getConvertedOperand(stringCmpOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      // Create string struct type: {ptr, i64}
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
      auto i32Ty = IntegerType::get(ctx, 32);

      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
      auto runtimeFn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_cmp", fnTy);

      auto one =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
      auto lhsAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, lhs, lhsAlloca);

      auto rhsAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, rhs, rhsAlloca);

      SmallVector<Value> operands = {lhsAlloca, rhsAlloca};
      auto call = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(runtimeFn),
          operands);
      Value cmpResult = call.getResult();

      Value zero = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(0));
      arith::CmpIPredicate arithPred;
      switch (stringCmpOp.getPredicate()) {
      case StringCmpPredicate::eq:
        arithPred = arith::CmpIPredicate::eq;
        break;
      case StringCmpPredicate::ne:
        arithPred = arith::CmpIPredicate::ne;
        break;
      case StringCmpPredicate::lt:
        arithPred = arith::CmpIPredicate::slt;
        break;
      case StringCmpPredicate::le:
        arithPred = arith::CmpIPredicate::sle;
        break;
      case StringCmpPredicate::gt:
        arithPred = arith::CmpIPredicate::sgt;
        break;
      case StringCmpPredicate::ge:
        arithPred = arith::CmpIPredicate::sge;
        break;
      }
      return arith::CmpIOp::create(rewriter, loc, arithPred, cmpResult, zero);
    }

    // Handle moore.struct_extract
    if (auto structExtractOp = dyn_cast<StructExtractOp>(mooreOp)) {
      Value input = getConvertedOperand(structExtractOp.getInput());
      if (!input)
        return nullptr;

      Type resultType =
          typeConverter->convertType(structExtractOp.getResult().getType());
      if (!resultType)
        return nullptr;

      auto fieldName = structExtractOp.getFieldName();

      // Handle LLVM struct type
      if (auto llvmStructTy =
              dyn_cast<LLVM::LLVMStructType>(input.getType())) {
        // LLVM struct needs ExtractValueOp with index
        // Look up field index from Moore struct type (can be packed or unpacked)
        auto inputType = structExtractOp.getInput().getType();
        unsigned fieldIdx = 0;
        if (auto unpackedStructTy = dyn_cast<UnpackedStructType>(inputType)) {
          for (const auto &member : unpackedStructTy.getMembers()) {
            if (member.name == fieldName)
              break;
            ++fieldIdx;
          }
        } else if (auto packedStructTy = dyn_cast<moore::StructType>(inputType)) {
          for (const auto &member : packedStructTy.getMembers()) {
            if (member.name == fieldName)
              break;
            ++fieldIdx;
          }
        } else {
          return nullptr; // Unknown struct type
        }
        return LLVM::ExtractValueOp::create(rewriter, loc, input,
                                            ArrayRef<int64_t>{fieldIdx});
      }

      // Handle hw::StructType
      if (auto hwStructTy = dyn_cast<hw::StructType>(input.getType())) {
        return hw::StructExtractOp::create(rewriter, loc, input, fieldName);
      }

      return nullptr;
    }

    // Handle func::CallOp - pass through to LLVM call
    if (auto callOp = dyn_cast<func::CallOp>(mooreOp)) {
      // Convert all operands
      SmallVector<Value> convertedOperands;
      for (Value operand : callOp.getOperands()) {
        Value converted = getConvertedOperand(operand);
        if (!converted)
          return nullptr;
        convertedOperands.push_back(converted);
      }

      // Convert result type (if any)
      SmallVector<Type> convertedResultTypes;
      for (Type resultType : callOp.getResultTypes()) {
        Type converted = typeConverter->convertType(resultType);
        if (!converted)
          return nullptr;
        convertedResultTypes.push_back(converted);
      }

      // Create the call operation
      auto newCallOp = func::CallOp::create(rewriter, loc, callOp.getCallee(),
                                            convertedResultTypes,
                                            convertedOperands);
      if (newCallOp.getNumResults() > 0)
        return newCallOp.getResult(0);

      // For void calls, return a dummy value (shouldn't be used)
      return nullptr;
    }

    // Handle moore.class.null - create a null pointer
    if (auto nullOp = dyn_cast<ClassNullOp>(mooreOp)) {
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }

    // Handle moore.class_handle_cmp - compare two class handles
    if (auto cmpOp = dyn_cast<ClassHandleCmpOp>(mooreOp)) {
      Value lhs = getConvertedOperand(cmpOp.getLhs());
      Value rhs = getConvertedOperand(cmpOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      // Map the moore predicate to LLVM icmp predicate
      LLVM::ICmpPredicate pred;
      switch (cmpOp.getPredicate()) {
      case ClassHandleCmpPredicate::eq:
        pred = LLVM::ICmpPredicate::eq;
        break;
      case ClassHandleCmpPredicate::ne:
        pred = LLVM::ICmpPredicate::ne;
        break;
      }

      // Class handles are pointers, use LLVM icmp
      return LLVM::ICmpOp::create(rewriter, loc, pred, lhs, rhs);
    }

    // Handle moore.wildcard_eq - wildcard equality comparison
    if (auto wildcardEqOp = dyn_cast<WildcardEqOp>(mooreOp)) {
      Value lhs = getConvertedOperand(wildcardEqOp.getLhs());
      Value rhs = getConvertedOperand(wildcardEqOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      Type resultType =
          typeConverter->convertType(wildcardEqOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Handle 4-state struct: extract value parts and compare
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value cmpResult = comb::ICmpOp::create(
            rewriter, loc, comb::ICmpPredicate::eq, lhsVal, rhsVal);
        // Result is i1, return as 4-state struct {i1, i1}
        Value zeroI1 =
            hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
        return createFourStateStruct(rewriter, loc, cmpResult, zeroI1);
      }

      // For plain integers, use icmp
      return comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::eq, lhs,
                                  rhs);
    }

    // Handle moore.wildcard_ne - wildcard inequality comparison
    if (auto wildcardNeOp = dyn_cast<WildcardNeOp>(mooreOp)) {
      Value lhs = getConvertedOperand(wildcardNeOp.getLhs());
      Value rhs = getConvertedOperand(wildcardNeOp.getRhs());
      if (!lhs || !rhs)
        return nullptr;

      Type resultType =
          typeConverter->convertType(wildcardNeOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Handle 4-state struct: extract value parts and compare
      if (isFourStateStructType(lhs.getType())) {
        Value lhsVal = extractFourStateValue(rewriter, loc, lhs);
        Value rhsVal = extractFourStateValue(rewriter, loc, rhs);
        Value cmpResult = comb::ICmpOp::create(
            rewriter, loc, comb::ICmpPredicate::ne, lhsVal, rhsVal);
        // Result is i1, return as 4-state struct {i1, i1}
        Value zeroI1 =
            hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
        return createFourStateStruct(rewriter, loc, cmpResult, zeroI1);
      }

      // For plain integers, use icmp
      return comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::ne, lhs,
                                  rhs);
    }

    // Handle moore.int_to_logic - convert integer to logic type
    if (auto intToLogicOp = dyn_cast<IntToLogicOp>(mooreOp)) {
      Value input = getConvertedOperand(intToLogicOp.getInput());
      if (!input)
        return nullptr;

      Type resultType =
          typeConverter->convertType(intToLogicOp.getResult().getType());
      if (!resultType)
        return nullptr;

      // Convert to 4-state representation: {value, unknown=0}
      if (isFourStateStructType(resultType)) {
        auto structType = cast<hw::StructType>(resultType);
        auto valueType = structType.getElements()[0].type;
        // Ensure input is the right integer type for the struct
        if (input.getType() != valueType) {
          input = hw::BitcastOp::create(rewriter, loc, valueType, input);
        }
        // Create struct with unknown=0 (2-state values have no X/Z)
        Value zero = hw::ConstantOp::create(rewriter, loc, valueType, 0);
        return createFourStateStruct(rewriter, loc, input, zero);
      }

      // If same type, just return input
      if (input.getType() == resultType)
        return input;

      return hw::BitcastOp::create(rewriter, loc, resultType, input);
    }

    // Handle moore.string.tolower - convert string to lowercase
    if (auto toLowerOp = dyn_cast<StringToLowerOp>(mooreOp)) {
      Value input = getConvertedOperand(toLowerOp.getStr());
      if (!input)
        return nullptr;

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

      auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy});
      auto runtimeFn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_tolower", fnTy);

      auto one =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
      auto strAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, input, strAlloca);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                       SymbolRefAttr::get(runtimeFn),
                                       ValueRange{strAlloca});
      return call.getResult();
    }

    // Handle moore.string.toupper - convert string to uppercase
    if (auto toUpperOp = dyn_cast<StringToUpperOp>(mooreOp)) {
      Value input = getConvertedOperand(toUpperOp.getStr());
      if (!input)
        return nullptr;

      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto stringStructTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

      auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy});
      auto runtimeFn =
          getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_toupper", fnTy);

      auto one =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
      auto strAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
      LLVM::StoreOp::create(rewriter, loc, input, strAlloca);

      auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                       SymbolRefAttr::get(runtimeFn),
                                       ValueRange{strAlloca});
      return call.getResult();
    }

    // Unsupported operation
    return nullptr;
  }

  /// Lower array.locator for associative arrays using first/next iteration.
  /// Associative arrays require a different iteration approach than regular
  /// arrays because they use keys instead of numeric indices.
  LogicalResult lowerAssocArrayWithInlineLoop(
      ArrayLocatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx,
      ModuleOp mod, Type mooreElemType, Type mooreKeyType, Type elemType,
      int64_t elemSizeBytes, Value blockArg, ArrayLocatorYieldOp yieldOp,
      Value resultAlloca, LLVM::LLVMStructType queueTy,
      LLVM::LLVMPointerType ptrTy, IntegerType i64Ty, IntegerType i1Ty,
      Value emptyQueue, Value one) const {
    Block &body = op.getBody().front();
    auto i32Ty = IntegerType::get(ctx, 32);


    // Convert key and element types to LLVM
    Type keyType = mooreKeyType ? typeConverter->convertType(mooreKeyType)
                                : nullptr;
    Type llvmElemType = convertToLLVMType(elemType);

    // Determine key size for runtime calls
    int32_t keySize = 0; // 0 means string key
    bool isStringKey = isa_and_nonnull<StringType>(mooreKeyType);
    if (!isStringKey && keyType) {
      if (auto intTy = dyn_cast<IntegerType>(keyType))
        keySize = (intTy.getWidth() + 7) / 8;
      else
        keySize = 8; // Default to 64-bit
    }

    // For string keys, the key storage is a MooreString struct {ptr, i64}
    // For integer keys, we use the integer type directly
    Type llvmKeyType;
    if (isStringKey) {
      llvmKeyType = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
    } else if (keyType) {
      llvmKeyType = convertToLLVMType(keyType);
    } else {
      // Wildcard with no specific key type - use i64 as default
      llvmKeyType = i64Ty;
    }

    // Allocate storage for the current key
    Value keyAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmKeyType, one);

    // Initialize key storage to zero
    if (isStringKey) {
      // Initialize string to {nullptr, 0}
      Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      Value zeroLen = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                               rewriter.getI64IntegerAttr(0));
      Value emptyStr = LLVM::UndefOp::create(rewriter, loc, llvmKeyType);
      emptyStr = LLVM::InsertValueOp::create(rewriter, loc, emptyStr, nullPtr,
                                             ArrayRef<int64_t>{0});
      emptyStr = LLVM::InsertValueOp::create(rewriter, loc, emptyStr, zeroLen,
                                             ArrayRef<int64_t>{1});
      LLVM::StoreOp::create(rewriter, loc, emptyStr, keyAlloca);
    } else {
      Value zeroKey = LLVM::ConstantOp::create(
          rewriter, loc, llvmKeyType, rewriter.getIntegerAttr(llvmKeyType, 0));
      LLVM::StoreOp::create(rewriter, loc, zeroKey, keyAlloca);
    }

    // Get runtime functions
    auto firstFnTy = LLVM::LLVMFunctionType::get(i1Ty, {ptrTy, ptrTy});
    auto firstFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_first", firstFnTy);
    auto nextFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_next", firstFnTy);

    // __moore_assoc_get_ref(array, key_ptr, value_size) -> value_ptr
    auto getRefFnTy =
        LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy, i32Ty});
    auto getRefFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_get_ref", getRefFnTy);

    Value valueSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(elemSizeBytes));

    // Use scf.for loop (iterate up to array size)
    // First get the array size
    auto sizeFnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy});
    auto sizeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_assoc_size", sizeFnTy);
    Value arraySize = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Ty},
                                           SymbolRefAttr::get(sizeFn),
                                           ValueRange{adaptor.getArray()})
                          .getResult();

    Value lb = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(0));
    Value step = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    // Use scf.for loop: for i in 0..<size, use first/next to iterate
    auto forOp = scf::ForOp::create(rewriter, loc, lb, arraySize, step);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();

    // On first iteration, use hasFirst result; otherwise, assoc_next was called
    // Actually, we need to call first on iteration 0, and next on subsequent iterations
    Value zeroIdx = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(0));
    Value isFirst = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                          iv, zeroIdx);

    // Create the hasKey control variable - on first iteration, call first; otherwise next already called
    auto iterIfOp = scf::IfOp::create(rewriter, loc, TypeRange{i1Ty}, isFirst,
                                      /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&iterIfOp.getThenRegion().front());
    // First iteration: call assoc_first
    Value firstResult = LLVM::CallOp::create(rewriter, loc, TypeRange{i1Ty},
                                             SymbolRefAttr::get(firstFn),
                                             ValueRange{adaptor.getArray(), keyAlloca})
                            .getResult();
    scf::YieldOp::create(rewriter, loc, ValueRange{firstResult});

    rewriter.setInsertionPointToStart(&iterIfOp.getElseRegion().front());
    // Subsequent iterations: call assoc_next (key was updated in previous iteration)
    Value nextResult = LLVM::CallOp::create(rewriter, loc, TypeRange{i1Ty},
                                            SymbolRefAttr::get(nextFn),
                                            ValueRange{adaptor.getArray(), keyAlloca})
                           .getResult();
    scf::YieldOp::create(rewriter, loc, ValueRange{nextResult});

    rewriter.setInsertionPointAfter(iterIfOp);
    Value hasKey = iterIfOp.getResult(0);

    // Only process if hasKey is true
    auto keyValidIf = scf::IfOp::create(rewriter, loc, TypeRange{}, hasKey,
                                        /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&keyValidIf.getThenRegion().front());

    // Get the element value using the current key
    Value elemPtr = LLVM::CallOp::create(
                        rewriter, loc, TypeRange{ptrTy},
                        SymbolRefAttr::get(getRefFn),
                        ValueRange{adaptor.getArray(), keyAlloca, valueSizeConst})
                        .getResult();
    Value currentElem =
        LLVM::LoadOp::create(rewriter, loc, llvmElemType, elemPtr);

    // Build value map for predicate conversion
    DenseMap<Value, Value> valueMap;
    valueMap[blockArg] = currentElem;

    // Map the index block argument (if present) to the current key
    // For associative arrays, the "index" is actually the key
    if (body.getNumArguments() >= 2) {
      Value indexArg = body.getArgument(1);
      Value currentKey =
          LLVM::LoadOp::create(rewriter, loc, llvmKeyType, keyAlloca);
      // The index argument type may differ from the key type
      Type indexArgType = typeConverter->convertType(indexArg.getType());
      if (indexArgType && indexArgType != llvmKeyType) {
        currentKey = UnrealizedConversionCastOp::create(rewriter, loc,
                                                        indexArgType, currentKey)
                         .getResult(0);
      }
      valueMap[indexArg] = currentKey;
    }

    // Map the input array
    valueMap[op.getArray()] = adaptor.getArray();

    // Handle external values from enclosing scope
    auto funcOp = op->getParentOfType<func::FuncOp>();
    Block *funcEntryBlock = funcOp ? &funcOp.getBody().front() : nullptr;

    for (Operation &innerOp : body.without_terminator()) {
      for (unsigned i = 0; i < innerOp.getNumOperands(); ++i) {
        Value operand = innerOp.getOperand(i);
        if (valueMap.count(operand))
          continue;
        if (Operation *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == &body)
            continue;
        }
        Value converted = rewriter.getRemappedValue(operand);
        if (converted) {
          valueMap[operand] = converted;
          continue;
        }
        if (auto blockArgVal = dyn_cast<BlockArgument>(operand)) {
          Block *ownerBlock = blockArgVal.getOwner();
          if (ownerBlock != &body && funcEntryBlock) {
            unsigned argIndex = blockArgVal.getArgNumber();
            if (argIndex < funcEntryBlock->getNumArguments()) {
              Value newBlockArg = funcEntryBlock->getArgument(argIndex);
              valueMap[operand] = newBlockArg;
              innerOp.setOperand(i, newBlockArg);
              continue;
            }
          }
        }
        Type targetType = typeConverter->convertType(operand.getType());
        if (targetType && targetType != operand.getType()) {
          converted = UnrealizedConversionCastOp::create(rewriter, loc,
                                                          targetType, operand)
                          .getResult(0);
          valueMap[operand] = converted;
        } else if (targetType) {
          valueMap[operand] = operand;
        }
      }
    }

    // Convert predicate operations inline
    for (Operation &innerOp : body.without_terminator()) {
      Value result =
          convertMooreOpInline(&innerOp, rewriter, loc, ctx, valueMap, mod);
      if (!result) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert predicate operation: " +
                    innerOp.getName().getStringRef().str());
      }
      if (innerOp.getNumResults() == 1) {
        valueMap[innerOp.getResult(0)] = result;
      }
    }

    // Get the converted condition
    Value condValue = yieldOp.getCondition();
    auto condIt = valueMap.find(condValue);
    Value cond;
    if (condIt != valueMap.end()) {
      cond = condIt->second;
    } else {
      cond = typeConverter->materializeTargetConversion(rewriter, loc, i1Ty,
                                                        condValue);
    }
    if (!cond)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert predicate condition");

    // Ensure condition is i1
    if (cond.getType() != i1Ty) {
      if (isFourStateStructType(cond.getType())) {
        cond = extractFourStateValue(rewriter, loc, cond);
      }
      if (auto intTy = dyn_cast<IntegerType>(cond.getType())) {
        if (intTy.getWidth() > 1) {
          Value zero = hw::ConstantOp::create(rewriter, loc, cond.getType(), 0);
          cond = comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::ne,
                                      cond, zero);
        }
      }
    }

    // If condition matches, add to result queue
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, cond,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto pushFnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty});
    auto pushFn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_queue_push_back",
                                         pushFnTy);

    bool returnIndices = op.getIndexed();
    if (returnIndices) {
      // For find_first_index etc., push the key (not a numeric index)
      int64_t keySizeBytes = getTypeSize(llvmKeyType);
      auto keySizeVal = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(keySizeBytes));
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(pushFn),
                           ValueRange{resultAlloca, keyAlloca, keySizeVal});
    } else {
      auto elemAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmElemType, one);
      LLVM::StoreOp::create(rewriter, loc, currentElem, elemAlloca);
      auto elemSize = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(elemSizeBytes));
      LLVM::CallOp::create(rewriter, loc, TypeRange{},
                           SymbolRefAttr::get(pushFn),
                           ValueRange{resultAlloca, elemAlloca, elemSize});
    }

    // After the for loop, get the result
    rewriter.setInsertionPointAfter(forOp);
    Value result = LLVM::LoadOp::create(rewriter, loc, queueTy, resultAlloca);

    // Handle first/last mode - extract single element from result
    auto mode = op.getMode();
    if (mode != LocatorMode::All) {
      Value resultLen = LLVM::ExtractValueOp::create(
          rewriter, loc, i64Ty, result, ArrayRef<int64_t>{1});
      Value resultDataPtr = LLVM::ExtractValueOp::create(
          rewriter, loc, ptrTy, result, ArrayRef<int64_t>{0});
      Value zeroLen = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                               rewriter.getI64IntegerAttr(0));
      Value stepVal = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(1));

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
        extractIdx = arith::SubIOp::create(rewriter, loc, resultLen, stepVal);
      }

      Type resultElemType = returnIndices ? llvmKeyType : llvmElemType;
      int64_t resultElemSize =
          returnIndices ? getTypeSize(llvmKeyType) : elemSizeBytes;

      Value singleElemPtr = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, resultElemType, resultDataPtr, extractIdx);
      Value singleElem =
          LLVM::LoadOp::create(rewriter, loc, resultElemType, singleElemPtr);

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

  /// Lower array.locator with an inline loop for complex predicates.
  /// This works for arbitrary predicate expressions because the predicate
  /// operations are converted inline to LLVM/arith equivalents.
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
    Type mooreKeyType = nullptr; // Only set for associative arrays
    int64_t fixedArraySize = -1; // -1 indicates dynamic array
    bool isAssocArray = false;
    if (auto queueType = dyn_cast<QueueType>(op.getArray().getType())) {
      mooreElemType = queueType.getElementType();
    } else if (auto arrayType =
                   dyn_cast<UnpackedArrayType>(op.getArray().getType())) {
      mooreElemType = arrayType.getElementType();
      fixedArraySize = arrayType.getSize();
    } else if (auto dynArrayType =
                   dyn_cast<OpenUnpackedArrayType>(op.getArray().getType())) {
      mooreElemType = dynArrayType.getElementType();
    } else if (auto assocArrayType =
                   dyn_cast<AssocArrayType>(op.getArray().getType())) {
      mooreElemType = assocArrayType.getElementType();
      mooreKeyType = assocArrayType.getIndexType();
      isAssocArray = true;
    } else if (auto wildcardAssocType =
                   dyn_cast<WildcardAssocArrayType>(op.getArray().getType())) {
      mooreElemType = wildcardAssocType.getElementType();
      // Wildcard assoc arrays use string keys by default in our runtime
      mooreKeyType = StringType::get(ctx);
      isAssocArray = true;
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

    // Handle associative arrays with a different iteration approach
    if (isAssocArray) {
      auto result = lowerAssocArrayWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod,
                                           mooreElemType, mooreKeyType,
                                           elemType, elemSizeBytes, blockArg,
                                           yieldOp, resultAlloca, queueTy,
                                           ptrTy, i64Ty, i1Ty, emptyQueue, one);
      return result;
    }

    // Handle array length and data pointer differently for fixed-size arrays
    // vs dynamic arrays/queues
    Value arrayLen;
    Value dataPtr;

    Type convertedArrayType = adaptor.getArray().getType();
    if (auto hwArrayTy = dyn_cast<hw::ArrayType>(convertedArrayType)) {
      // Fixed-size array: hw::ArrayType needs to be stored to memory first
      // Convert hw::ArrayType to LLVM array type for storage
      Type llvmArrayType = convertToLLVMType(hwArrayTy);
      Type llvmElemType = convertToLLVMType(elemType);

      // Allocate stack space for the array
      auto arrayAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmArrayType, one);

      // Cast the hw.array value to llvm.array for storage
      Value llvmArrayValue =
          UnrealizedConversionCastOp::create(rewriter, loc, llvmArrayType,
                                             adaptor.getArray())
              .getResult(0);
      LLVM::StoreOp::create(rewriter, loc, llvmArrayValue, arrayAlloca);

      // Use the alloca as the data pointer and fixed size as length
      dataPtr = arrayAlloca;
      arrayLen = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(fixedArraySize));

      // Update elemType to use LLVM element type for GEP operations
      elemType = llvmElemType;
    } else {
      // Dynamic array/queue: already an LLVM struct {ptr, i64}
      arrayLen = LLVM::ExtractValueOp::create(
          rewriter, loc, i64Ty, adaptor.getArray(), ArrayRef<int64_t>{1});
      dataPtr = LLVM::ExtractValueOp::create(
          rewriter, loc, ptrTy, adaptor.getArray(), ArrayRef<int64_t>{0});
      // Convert elemType to LLVM type for GEP and Load operations.
      // The type converter may produce hw::StructType for packed structs,
      // which is not valid for LLVM operations.
      elemType = convertToLLVMType(elemType);
    }

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

    // Build a value map from Moore values to converted LLVM/arith values.
    // This is used by convertMooreOpInline to resolve operands.
    DenseMap<Value, Value> valueMap;

    // Map the block argument (item) to the current element
    valueMap[blockArg] = currentElem;

    // Map the index block argument (if present) to the loop induction variable
    if (body.getNumArguments() >= 2) {
      Value indexArg = body.getArgument(1);
      // The induction variable is i64, but the index block arg is typically i32
      auto i32Ty = IntegerType::get(ctx, 32);
      Value ivTrunc = arith::TruncIOp::create(rewriter, loc, i32Ty, iv);
      valueMap[indexArg] = ivTrunc;
    }

    // Map the input array to its converted value so that operations inside the
    // predicate body that reference the array can find the converted value.
    valueMap[op.getArray()] = adaptor.getArray();

    // Pre-scan all operations in the predicate body for external values
    // (values defined outside the predicate region). These need to be converted
    // before processing the operations. This is critical for handling function
    // calls that reference outer scope values like block arguments from the
    // enclosing function (e.g., 'this' pointer in class methods).
    //
    // When the enclosing function's signature is converted, its block arguments
    // are replaced with new arguments of the converted types. The operations
    // inside nested regions (like this predicate) still reference the old
    // block arguments, which have been invalidated. We need to find the remapped
    // values through the rewriter.
    //
    // First, get the parent function's entry block arguments so we can map
    // any references to old function arguments to the new converted ones.
    auto funcOp = op->getParentOfType<func::FuncOp>();
    Block *funcEntryBlock = funcOp ? &funcOp.getBody().front() : nullptr;

    // First pass: collect all external values that need remapping.
    // We need to be careful because some operands may reference invalidated
    // block arguments that can crash if we try to access their properties.
    SmallVector<std::pair<Operation *, unsigned>> operandsToRemap;
    SmallVector<Value> operandValues;

    for (Operation &innerOp : body.without_terminator()) {
      for (unsigned i = 0; i < innerOp.getNumOperands(); ++i) {
        Value operand = innerOp.getOperand(i);

        // Skip if already in valueMap
        if (valueMap.count(operand))
          continue;

        // Skip if defined inside the predicate body
        if (Operation *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == &body)
            continue;
        }

        // This is an external value - try to get the remapped version
        // The rewriter maintains a mapping from old values to new values
        // after type conversion.
        Value converted = rewriter.getRemappedValue(operand);
        if (converted) {
          valueMap[operand] = converted;
          continue;
        }

        // Special handling for block arguments from enclosing functions.
        // When a function's signature is converted, the old block arguments are
        // replaced with new ones. Operations in nested regions (like this
        // predicate body) still reference the old arguments, but those have
        // been invalidated. We need to find the new block argument at the
        // corresponding position in the converted function.
        //
        // Check if this is a block argument. Note: we can't simply check
        // getDefiningOp() == nullptr because the operand might be an
        // invalidated block argument.
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          // Check if this block argument belongs to a block outside the
          // predicate (i.e., from the enclosing function)
          Block *ownerBlock = blockArg.getOwner();
          if (ownerBlock != &body) {
            // This is a block argument from an enclosing scope.
            // Try to find the corresponding argument in the current function.
            if (funcEntryBlock) {
              unsigned argIndex = blockArg.getArgNumber();
              // The entry block should have the converted arguments
              if (argIndex < funcEntryBlock->getNumArguments()) {
                Value newBlockArg = funcEntryBlock->getArgument(argIndex);
                valueMap[operand] = newBlockArg;
                // Also update the operation's operand directly to fix the
                // invalid reference
                innerOp.setOperand(i, newBlockArg);
                continue;
              }
            }
          }
        }

        // For external values defined by operations, try to convert the type
        Type targetType = typeConverter->convertType(operand.getType());
        if (targetType && targetType != operand.getType()) {
          // Create an unrealized conversion cast for the external value
          converted = UnrealizedConversionCastOp::create(rewriter, loc,
                                                          targetType, operand)
                          .getResult(0);
          valueMap[operand] = converted;
        } else if (targetType) {
          // Same type, just use the operand directly
          valueMap[operand] = operand;
        }
      }
    }

    // Convert each Moore operation in the predicate block inline to LLVM/arith.
    // Process operations in order to ensure operand dependencies are resolved.
    for (Operation &innerOp : body.without_terminator()) {
      Value result = convertMooreOpInline(&innerOp, rewriter, loc, ctx,
                                          valueMap, mod);
      if (!result) {
        // If inline conversion failed, emit a diagnostic with the op name
        return rewriter.notifyMatchFailure(
            op, "failed to convert predicate operation: " +
                    innerOp.getName().getStringRef().str());
      }

      // Map all results (most ops have single result)
      if (innerOp.getNumResults() == 1) {
        valueMap[innerOp.getResult(0)] = result;
      }
    }

    // Get the converted condition value
    Value condValue = yieldOp.getCondition();
    auto condIt = valueMap.find(condValue);
    Value cond;
    if (condIt != valueMap.end()) {
      cond = condIt->second;
    } else {
      // Try to materialize conversion for external condition
      cond = typeConverter->materializeTargetConversion(
          rewriter, loc, i1Ty, condValue);
    }

    if (!cond)
      return rewriter.notifyMatchFailure(op, "failed to convert predicate condition");

    // Ensure condition is i1
    if (cond.getType() != i1Ty) {
      // Handle 4-state struct by extracting the value part
      if (isFourStateStructType(cond.getType())) {
        Value val = extractFourStateValue(rewriter, loc, cond);
        cond = val;
      }
      // Truncate or compare to zero if not i1
      if (auto intTy = dyn_cast<IntegerType>(cond.getType())) {
        if (intTy.getWidth() > 1) {
          Value zero = hw::ConstantOp::create(rewriter, loc, cond.getType(), 0);
          cond = comb::ICmpOp::create(rewriter, loc, comb::ICmpPredicate::ne,
                                      cond, zero);
        }
      }
    }

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
    // Body has 1 or 2 block arguments: the element and optionally the index
    if (body.getNumArguments() < 1 || body.getNumArguments() > 2)
      return rewriter.notifyMatchFailure(op, "expected 1 or 2 block arguments");

    Value blockArg = body.getArgument(0);
    // indexArg is body.getArgument(1) if present (for item.index support)

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
    if (!foundComparison)
      return lowerWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod);

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
      // For complex cases (comparing fields with other fields, or with variable
      // reads), fall back to the inline loop approach which can handle any predicate.
      Value llvmCmpValue;
      if (auto constOp = fieldCmpValue.getDefiningOp<ConstantOp>()) {
        APInt constAPInt = constOp.getValue().toAPInt(false);
        llvmCmpValue = LLVM::ConstantOp::create(rewriter, loc, convertedFieldType,
                                                constAPInt.getSExtValue());
      } else {
        // For non-constant comparison values (including reads from variables,
        // other field accesses, etc.), use the inline loop approach.
        return lowerWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod);
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
    // If not a constant, fall back to inline loop approach for variable comparisons
    auto constOp = constValue.getDefiningOp<ConstantOp>();
    if (!constOp)
      return lowerWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod);

    // For associative arrays, use the inline loop approach which handles
    // the key-based iteration via __moore_assoc_first/next/get_ref.
    // The simple runtime functions are designed for sequential arrays.
    if (isa<AssocArrayType, WildcardAssocArrayType>(op.getArray().getType()))
      return lowerWithInlineLoop(op, adaptor, rewriter, loc, ctx, mod);

    // For fixed-size arrays (UnpackedArrayType), use direct lowering
    // because the runtime functions expect a queue-like {ptr, i64} structure,
    // but fixed-size arrays are converted to hw::ArrayType.
    // We use CF dialect (basic blocks with branches) instead of SCF because
    // SCF ops are illegal inside llhd::ProcessOp.
    if (isa<UnpackedArrayType>(op.getArray().getType())) {
      return lowerFixedArrayWithSimplePredicate(op, adaptor, rewriter, loc, ctx,
                                                mod, cmpMode, constValue);
    }

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
    // Convert key type to pure LLVM type for LLVM operations
    Type llvmKeyType = convertToLLVMType(keyType);
    auto one = LLVM::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
    auto keyAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmKeyType, one);
    // Cast the key value to LLVM type if needed (hw.struct -> llvm.struct)
    Value keyToStore = adaptor.getKey();
    if (llvmKeyType != keyType) {
      keyToStore = UnrealizedConversionCastOp::create(
                       rewriter, loc, llvmKeyType, ValueRange{keyToStore})
                       .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, loc, keyToStore, keyAlloca);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, SymbolRefAttr::get(fn),
                         ValueRange{adaptor.getArray(), keyAlloca});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for moore.assoc.create -> runtime function call.
struct AssocArrayCreateOpConversion
    : public OpConversionPattern<AssocArrayCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssocArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
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
    int32_t valueSize = 4; // Default

    auto resultType = op.getResult().getType();
    if (auto assocType = dyn_cast<AssocArrayType>(resultType)) {
      auto keyType = assocType.getIndexType();
      if (!isa<StringType>(keyType)) {
        auto convertedKeyType = typeConverter->convertType(keyType);
        if (auto intTy = dyn_cast<IntegerType>(convertedKeyType))
          keySize = intTy.getWidth() / 8;
        else
          keySize = 8; // Default to 64-bit for unknown types
      }
      auto valueType = assocType.getElementType();
      auto convertedValueType = typeConverter->convertType(valueType);
      if (auto intTy = dyn_cast<IntegerType>(convertedValueType))
        valueSize = (intTy.getWidth() + 7) / 8;
    } else if (auto wildcardType = dyn_cast<WildcardAssocArrayType>(resultType)) {
      // For WildcardAssocArrayType, use string key (keySize=0) and determine
      // value size from the element type.
      auto valueType = wildcardType.getElementType();
      auto convertedValueType = typeConverter->convertType(valueType);
      if (auto intTy = dyn_cast<IntegerType>(convertedValueType))
        valueSize = (intTy.getWidth() + 7) / 8;
    }

    auto keySizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(keySize));
    auto valueSizeConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(valueSize));

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{ptrTy}, SymbolRefAttr::get(fn),
        ValueRange{keySizeConst, valueSizeConst});

    rewriter.replaceOp(op, call.getResult());
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
    Type keyValueType;     // Original key value type (may be hw.struct)
    Type llvmKeyValueType; // Pure LLVM type for alloca/store/load operations
    // Track whether keyArg is a function ref parameter (needs llvm.store writeback)
    bool isFuncRefParam = false;
    if (isa<LLVM::LLVMPointerType>(keyArg.getType())) {
      // String keys are already LLVM pointers (to struct {ptr, i64})
      keyAlloca = keyArg;
    } else if (auto refType = dyn_cast<llhd::RefType>(keyArg.getType())) {
      // Keys (can be integer or struct) are llhd.ref - we need to create an
      // alloca and copy the value to it (the runtime will update it in place
      // for first/next/etc)
      keyValueType = refType.getNestedType();
      // Convert to pure LLVM type for LLVM operations (hw.struct -> llvm.struct)
      llvmKeyValueType = convertToLLVMType(keyValueType);
      auto one = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getI64IntegerAttr(1));
      keyAlloca =
          LLVM::AllocaOp::create(rewriter, loc, ptrTy, llvmKeyValueType, one);

      // Check if keyArg is a function ref parameter (BlockArgument in func::FuncOp).
      // Function ref parameters need llvm.load/store, not llhd.prb/drv, since
      // the simulator cannot track signal references through function call boundaries.
      if (auto blockArg = dyn_cast<BlockArgument>(keyArg)) {
        auto *parentOp = blockArg.getOwner()->getParentOp();
        if (isa<func::FuncOp>(parentOp)) {
          isFuncRefParam = true;
        }
      }

      // Read current key value and store to alloca
      Value keyToStore;
      if (isFuncRefParam) {
        // For function ref parameters, cast to llvm.ptr and use llvm.load
        Value keyPtr = UnrealizedConversionCastOp::create(rewriter, loc, ptrTy,
                                                          keyArg)
                           .getResult(0);
        auto loadedKey =
            LLVM::LoadOp::create(rewriter, loc, llvmKeyValueType, keyPtr);
        keyToStore = loadedKey.getResult();
      } else {
        // For signals (llhd.sig), use llhd.prb
        auto currentKey = llhd::ProbeOp::create(rewriter, loc, keyArg);
        keyToStore = currentKey.getResult();
        // If types differ, cast the probed value to LLVM type
        if (llvmKeyValueType != keyValueType) {
          keyToStore = UnrealizedConversionCastOp::create(
                           rewriter, loc, llvmKeyValueType,
                           ValueRange{keyToStore})
                           .getResult(0);
        }
      }
      LLVM::StoreOp::create(rewriter, loc, keyToStore, keyAlloca);
    } else {
      return rewriter.notifyMatchFailure(loc, "unsupported key type");
    }

    // Determine the actual array handle.
    // For local variables, adaptor.getArray() IS the handle.
    // For class property refs (from GEP), we need to load the handle first.
    Value arrayHandle = adaptor.getArray();
    {
      Value source = arrayHandle;
      // Unwrap unrealized conversion casts
      while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1)
          source = castOp.getInputs()[0];
        else
          break;
      }
      // For GEP (class property) or AddressOfOp (global), load the handle
      if (source.getDefiningOp<LLVM::GEPOp>() ||
          source.getDefiningOp<LLVM::AddressOfOp>()) {
        arrayHandle = LLVM::LoadOp::create(rewriter, loc, ptrTy, arrayHandle);
      }
    }

    auto call = LLVM::CallOp::create(
        rewriter, loc, TypeRange{i1Ty}, SymbolRefAttr::get(fn),
        ValueRange{arrayHandle, keyAlloca});

    // For non-string keys (llhd.ref), write back the updated key value
    if (keyValueType) {
      auto updatedKey =
          LLVM::LoadOp::create(rewriter, loc, llvmKeyValueType, keyAlloca);
      Value keyToDrive = updatedKey.getResult();

      if (isFuncRefParam) {
        // For function ref parameters, use llvm.store
        Value keyPtr = UnrealizedConversionCastOp::create(rewriter, loc, ptrTy,
                                                          keyArg)
                           .getResult(0);
        LLVM::StoreOp::create(rewriter, loc, keyToDrive, keyPtr);
      } else {
        // For signals (llhd.sig), use llhd.drv
        // If types differ, cast the loaded value back to hw type for drive
        if (llvmKeyValueType != keyValueType) {
          keyToDrive = UnrealizedConversionCastOp::create(
                           rewriter, loc, keyValueType, ValueRange{keyToDrive})
                           .getResult(0);
        }
        auto delay = llhd::ConstantTimeOp::create(
            rewriter, loc, llhd::TimeAttr::get(ctx, 0U, "ns", 0, 1));
        llhd::DriveOp::create(rewriter, loc, keyArg, keyToDrive, delay,
                              Value{});
      }
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
    // The runtime returns i32, but .exists() should return i1 (bool)
    auto i1Ty = IntegerType::get(ctx, 1);
    auto zero = LLVM::ConstantOp::create(rewriter, loc,
                                         rewriter.getI32IntegerAttr(0));
    auto result = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
                                       LLVM::ICmpPredicate::ne,
                                       call.getResult(), zero);
    rewriter.replaceOp(op, result);
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

    Value value = adaptor.getValue();

    // Handle 4-state types which are lowered to {value, unknown} structs.
    // Extract the value field to get an integer for string conversion.
    if (isFourStateStructType(value.getType())) {
      value = extractFourStateValue(rewriter, loc, value);
    }

    if (!value.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "value must be an integer type");
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

// moore.string.atoreal -> call to __moore_string_atoreal runtime function
struct StringAtoRealOpConversion : public OpConversionPattern<StringAtoRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringAtoRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto f64Ty = Float64Type::get(ctx);
    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto fnTy = LLVM::LLVMFunctionType::get(f64Ty, {ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_atoreal", fnTy);

    // Store string to alloca and pass pointer.
    auto one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    auto strAlloca =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, stringStructTy, one);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getStr(), strAlloca);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{f64Ty},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{strAlloca});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.hextoa -> call to __moore_string_hextoa runtime function
struct StringHexToAOpConversion : public OpConversionPattern<StringHexToAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringHexToAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    Value value = adaptor.getValue();

    // Handle 4-state types which are lowered to {value, unknown} structs.
    if (isFourStateStructType(value.getType())) {
      value = extractFourStateValue(rewriter, loc, value);
    }

    if (!value.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "value must be an integer type");
    auto valueWidth = value.getType().getIntOrFloatBitWidth();

    Value valueI64;
    if (valueWidth < 64) {
      valueI64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
    } else if (valueWidth > 64) {
      valueI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, value);
    } else {
      valueI64 = value;
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_hextoa", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{valueI64});
    Value resultString = call.getResult();

    LLVM::StoreOp::create(rewriter, loc, resultString, adaptor.getDest());

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.string.octtoa -> call to __moore_string_octtoa runtime function
struct StringOctToAOpConversion : public OpConversionPattern<StringOctToAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringOctToAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    Value value = adaptor.getValue();

    // Handle 4-state types which are lowered to {value, unknown} structs.
    if (isFourStateStructType(value.getType())) {
      value = extractFourStateValue(rewriter, loc, value);
    }

    if (!value.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "value must be an integer type");
    auto valueWidth = value.getType().getIntOrFloatBitWidth();

    Value valueI64;
    if (valueWidth < 64) {
      valueI64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
    } else if (valueWidth > 64) {
      valueI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, value);
    } else {
      valueI64 = value;
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_octtoa", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{valueI64});
    Value resultString = call.getResult();

    LLVM::StoreOp::create(rewriter, loc, resultString, adaptor.getDest());

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.string.bintoa -> call to __moore_string_bintoa runtime function
struct StringBinToAOpConversion : public OpConversionPattern<StringBinToAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringBinToAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);

    Value value = adaptor.getValue();

    // Handle 4-state types which are lowered to {value, unknown} structs.
    if (isFourStateStructType(value.getType())) {
      value = extractFourStateValue(rewriter, loc, value);
    }

    if (!value.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "value must be an integer type");
    auto valueWidth = value.getType().getIntOrFloatBitWidth();

    Value valueI64;
    if (valueWidth < 64) {
      valueI64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
    } else if (valueWidth > 64) {
      valueI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, value);
    } else {
      valueI64 = value;
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_bintoa", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{valueI64});
    Value resultString = call.getResult();

    LLVM::StoreOp::create(rewriter, loc, resultString, adaptor.getDest());

    rewriter.eraseOp(op);
    return success();
  }
};

// moore.string.realtoa -> call to __moore_string_realtoa runtime function
struct StringRealToAOpConversion : public OpConversionPattern<StringRealToAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringRealToAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto f64Ty = Float64Type::get(ctx);

    Value value = adaptor.getValue();

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {f64Ty});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_realtoa", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{stringStructTy},
                                     SymbolRefAttr::get(runtimeFn),
                                     ValueRange{value});
    Value resultString = call.getResult();

    LLVM::StoreOp::create(rewriter, loc, resultString, adaptor.getDest());

    rewriter.eraseOp(op);
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

// moore.string.compare -> call to __moore_string_compare runtime function
struct StringCompareOpConversion : public OpConversionPattern<StringCompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringCompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_compare", fnTy);

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
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.string.icompare -> call to __moore_string_icompare runtime function
struct StringICompareOpConversion
    : public OpConversionPattern<StringICompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringICompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto stringStructTy = getStringStructType(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto runtimeFn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_string_icompare", fnTy);

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
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// moore.int_to_string -> call to __moore_packed_string_to_string runtime
// function. This operation is used to convert packed string literals (where
// ASCII characters are stored in an integer) to runtime string values.
// See IEEE 1800-2017 section 5.9 "String literals".
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
    if (!input.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "input must be an integer type");
    auto inputWidth = input.getType().getIntOrFloatBitWidth();

    // For wide constant strings (> 64 bits), extract all bytes and create
    // a global string constant to avoid truncation.
    if (inputWidth > 64) {
      // Try to get the constant value from the input
      APInt constValue;
      if (auto hwConstOp = input.getDefiningOp<hw::ConstantOp>()) {
        constValue = hwConstOp.getValue();
      } else if (auto arithConstOp = input.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(arithConstOp.getValue()))
          constValue = intAttr.getValue();
      }

      if (constValue.getBitWidth() > 0) {
        // Extract bytes from the packed integer (big-endian: MSB is first char)
        unsigned numBytes = (inputWidth + 7) / 8;
        std::string strContent;
        strContent.reserve(numBytes);

        // Extract bytes from MSB to LSB (SystemVerilog packed string order)
        for (int i = numBytes - 1; i >= 0; --i) {
          unsigned bitOffset = i * 8;
          if (bitOffset + 8 <= constValue.getBitWidth()) {
            char c = static_cast<char>(
                constValue.extractBits(8, bitOffset).getZExtValue());
            if (c != 0 || !strContent.empty())
              strContent.push_back(c);
          }
        }

        if (!strContent.empty()) {
          // Create global string constant using the helper function
          Value result = createMooreStringFromAttr(
              loc, mod, rewriter, strContent, "__packed_string");
          rewriter.replaceOp(op, result);
          return success();
        }
      }
      // Fall through to runtime call if we couldn't extract constant
    }

    Value inputI64;
    if (inputWidth < 64) {
      inputI64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, input);
    } else if (inputWidth > 64) {
      // For non-constant wide inputs, truncate (lossy, but no better option
      // without a runtime function that handles wide packed strings)
      inputI64 = arith::TruncIOp::create(rewriter, loc, i64Ty, input);
    } else {
      inputI64 = input;
    }

    auto fnTy = LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
    // Use packed_string_to_string to decode ASCII bytes from the integer
    auto runtimeFn = getOrCreateRuntimeFunc(
        mod, rewriter, "__moore_packed_string_to_string", fnTy);

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
    if (!resultType || !resultType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "result type must be an integer type");
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

        // Note: TimeType now converts to i64, so time values are handled
        // by the targetTy.isIntOrFloat() case below.
        if (isFourStateStructType(targetTy)) {
          // 4-state type: wrap the integer result in a 4-state struct
          auto structType = cast<hw::StructType>(targetTy);
          auto valueType = structType.getElements()[0].type;
          unsigned targetWidth = valueType.getIntOrFloatBitWidth();

          // Extend or truncate to match the target value type
          Value valueToStore;
          if (targetWidth < 32) {
            valueToStore =
                arith::TruncIOp::create(rewriter, loc, valueType, parsedValue);
          } else if (targetWidth > 32) {
            // Sign extend for larger types
            valueToStore =
                arith::ExtSIOp::create(rewriter, loc, valueType, parsedValue);
          } else {
            valueToStore = parsedValue;
            // Need to ensure type matches
            if (valueToStore.getType() != valueType)
              valueToStore = arith::ExtSIOp::create(rewriter, loc, valueType,
                                                    parsedValue);
          }

          // Create 4-state struct with unknown=0 (parsed values are known)
          Value zero = hw::ConstantOp::create(rewriter, loc, valueType, 0);
          Value fourStateValue =
              createFourStateStruct(rewriter, loc, valueToStore, zero);

          // Create llhd.drive to store the value
          auto timeAttr =
              llhd::TimeAttr::get(ctx, 0U, llvm::StringRef("ns"), 0, 1);
          auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
          llhd::DriveOp::create(rewriter, loc, destRef, fourStateValue, time,
                                Value{});
        } else if (targetTy.isIntOrFloat()) {
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
          auto timeAttr =
              llhd::TimeAttr::get(ctx, 0U, llvm::StringRef("ns"), 0, 1);
          auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
          llhd::DriveOp::create(rewriter, loc, destRef, valueToStore, time,
                                Value{});
        } else {
          return rewriter.notifyMatchFailure(
              op, "sscanf destination must be an integer or time type");
        }
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

    // Handle 4-state types which are lowered to {value, unknown} structs
    if (isFourStateStructType(input.getType()))
      input = extractFourStateValue(rewriter, loc, input);

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

    // Handle 4-state types which are lowered to {value, unknown} structs
    if (isFourStateStructType(input.getType()))
      input = extractFourStateValue(rewriter, loc, input);

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

    // Handle 4-state types which are lowered to {value, unknown} structs
    if (isFourStateStructType(input.getType()))
      input = extractFourStateValue(rewriter, loc, input);

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

    // Handle 4-state types which are lowered to {value, unknown} structs
    if (isFourStateStructType(input.getType()))
      input = extractFourStateValue(rewriter, loc, input);

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

// moore.builtin.severity -> runtime function call
// Calls __moore_error, __moore_warning, or __moore_info to track counts.
// Note: $fatal is handled separately with FinishBIOp.
static LogicalResult convert(SeverityBIOp op, SeverityBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto mod = op->getParentOfType<ModuleOp>();

  // Determine which runtime function to call based on severity
  std::string funcName;
  switch (op.getSeverity()) {
  case Severity::Fatal:
    // $fatal is handled by the subsequent FinishBIOp, but we still need
    // to print the message. Use __moore_error to increment the count.
    funcName = "__moore_error";
    break;
  case Severity::Error:
    funcName = "__moore_error";
    break;
  case Severity::Warning:
    funcName = "__moore_warning";
    break;
  case Severity::Info:
    funcName = "__moore_info";
    break;
  }

  // Get or create the runtime function
  // void __moore_xxx(MooreString *message)
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrTy});
  auto fn = getOrCreateRuntimeFunc(mod, rewriter, funcName, fnTy);

  // The message is already a formatted string from the sim dialect.
  // We need to convert it to a MooreString pointer.
  // For now, use sim.proc.print for the message output and track counts
  // separately. This is a simplified approach.

  // First, print the message using the sim dialect infrastructure
  std::string severityPrefix;
  switch (op.getSeverity()) {
  case Severity::Fatal:
    severityPrefix = "Fatal: ";
    break;
  case Severity::Error:
    severityPrefix = "Error: ";
    break;
  case Severity::Warning:
    severityPrefix = "Warning: ";
    break;
  case Severity::Info:
    severityPrefix = "Info: ";
    break;
  }

  auto prefix = sim::FormatLiteralOp::create(rewriter, loc, severityPrefix);
  auto message = sim::FormatStringConcatOp::create(
      rewriter, loc, ValueRange{prefix, adaptor.getMessage()});
  sim::PrintFormattedProcOp::create(rewriter, loc, message);

  // Now call the runtime function with a null message to track counts.
  // The actual message was already printed above.
  auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  LLVM::CallOp::create(rewriter, loc, fn, ValueRange{nullPtr});

  rewriter.eraseOp(op);
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
// Returns the current simulation time as i64 (femtoseconds).
static LogicalResult convert(TimeBIOp op, TimeBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  // Get current time as llhd.time, then convert to i64.
  auto llhdTime = llhd::CurrentTimeOp::create(rewriter, loc);
  auto i64Ty = rewriter.getIntegerType(64);
  rewriter.replaceOpWithNewOp<llhd::TimeToIntOp>(op, i64Ty, llhdTime);
  return success();
}

// moore.logic_to_time
// Since TimeType now converts to i64, this operation just extracts the value
// from a 4-state struct (if present) and produces an i64.
static LogicalResult convert(LogicToTimeOp op, LogicToTimeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value input = adaptor.getInput();
  // Handle 4-state struct types by extracting the value component
  if (isFourStateStructType(input.getType()))
    input = extractFourStateValue(rewriter, loc, input);
  // The result type is i64 (time in femtoseconds), no conversion needed
  rewriter.replaceOp(op, input);
  return success();
}

// moore.time_to_logic
// Since TimeType now converts to i64, this operation converts i64 to the
// 4-state l64 result type (a struct {value: i64, unknown: i64}).
struct TimeToLogicOpConversion : public OpConversionPattern<TimeToLogicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TimeToLogicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // The input is now i64 (time in femtoseconds)
    Value timeInt = adaptor.getInput();
    auto intType = rewriter.getIntegerType(64);

    // Check if the result type should be a 4-state struct
    Type resultType = typeConverter->convertType(op.getType());
    if (isFourStateStructType(resultType)) {
      // Create 4-state struct with unknown=0 (time is always a known value)
      Value zero = hw::ConstantOp::create(rewriter, loc, intType, 0);
      auto result = createFourStateStruct(rewriter, loc, timeInt, zero);
      rewriter.replaceOp(op, result);
    } else {
      // 2-state type: use the integer directly
      rewriter.replaceOp(op, timeInt);
    }
    return success();
  }
};

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
  StringRef constraintName; // Name of the constraint block
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
  StringRef constraintName; // Name of the constraint block
  int64_t defaultValue;   // Default value when no hard constraint applies
  unsigned fieldIndex;    // Index of the field in the struct
  unsigned bitWidth;      // Bit width of the property
};

/// Helper structure to hold extracted distribution constraint information.
/// Distribution constraints specify weighted probability distributions.
/// Pattern: `x dist { 0 := 10, [1:5] :/ 50, 6 := 40 }`
struct DistConstraintInfo {
  StringRef propertyName;   // Name of the constrained property
  StringRef constraintName; // Name of the constraint block
  SmallVector<std::pair<int64_t, int64_t>> ranges; // Ranges as [low, high] pairs
  SmallVector<int64_t> weights;   // Weight for each range
  SmallVector<int64_t> perRange;  // 0 = := (per-value), 1 = :/ (per-range)
  unsigned fieldIndex;      // Index of the field in the struct
  unsigned bitWidth;        // Bit width of the property
  bool isSigned = false;    // Signedness of the property
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

    StringRef constraintName = constraintBlock.getSymName();

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
        info.constraintName = constraintName;
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

    StringRef constraintName = constraintBlock.getSymName();

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
        info.constraintName = constraintName;
        info.defaultValue = defaultValue;
        info.fieldIndex = fieldIdx;
        info.bitWidth = bitWidth;
        softConstraints.push_back(info);
      }
    }
  }

  return softConstraints;
}

/// Extract distribution constraints from a class declaration.
/// Distribution constraints specify weighted probability distributions using
/// the `dist` keyword. Pattern: `x dist { 0 := 10, [1:5] :/ 50, 6 := 40 }`
static SmallVector<DistConstraintInfo>
extractDistConstraints(ClassDeclOp classDecl, ClassTypeCache &cache,
                       SymbolRefAttr classSym) {
  SmallVector<DistConstraintInfo> distConstraints;

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

  // Walk through constraint blocks looking for ConstraintDistOp
  for (auto &op : classDecl.getBody().getOps()) {
    auto constraintBlock = dyn_cast<ConstraintBlockOp>(op);
    if (!constraintBlock)
      continue;

    StringRef constraintName = constraintBlock.getSymName();

    // Walk the constraint block body looking for ConstraintDistOp
    for (auto &constraintOp : constraintBlock.getBody().getOps()) {
      if (auto distOp = dyn_cast<ConstraintDistOp>(constraintOp)) {
        // Get the variable being constrained
        Value variable = distOp.getVariable();

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

        // Get the distribution data from the op
        ArrayRef<int64_t> values = distOp.getValues();
        ArrayRef<int64_t> weights = distOp.getWeights();
        ArrayRef<int64_t> perRange = distOp.getPerRange();

        // Values come in pairs: [low0, high0, low1, high1, ...]
        if (values.size() < 2 || values.size() % 2 != 0)
          continue;
        if (weights.size() != values.size() / 2)
          continue;
        if (perRange.size() != weights.size())
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

        DistConstraintInfo info;
        info.propertyName = propName;
        info.constraintName = constraintName;
        info.fieldIndex = fieldIdx;
        info.bitWidth = bitWidth;
        info.isSigned = distOp.getIsSignedAttr() != nullptr;

        // Convert values array to ranges vector
        for (size_t i = 0; i < values.size(); i += 2) {
          info.ranges.push_back({values[i], values[i + 1]});
        }

        // Copy weights and perRange
        for (int64_t w : weights)
          info.weights.push_back(w);
        for (int64_t p : perRange)
          info.perRange.push_back(p);

        distConstraints.push_back(info);
      }
    }
  }

  return distConstraints;
}

/// Helper structure to hold solve-before ordering information.
/// Specifies that variables in 'before' should be randomized before 'after'.
/// Pattern: `solve a before b, c;` -> before={a}, after={b, c}
struct SolveBeforeInfo {
  SmallVector<StringRef> before; // Variables to solve first
  SmallVector<StringRef> after;  // Variables to solve after
};

/// Extract solve-before ordering from a class declaration.
/// Returns a list of solve-before constraints found in constraint blocks.
static SmallVector<SolveBeforeInfo>
extractSolveBeforeOrdering(ClassDeclOp classDecl) {
  SmallVector<SolveBeforeInfo> ordering;

  // Walk through constraint blocks looking for ConstraintSolveBeforeOp
  for (auto &op : classDecl.getBody().getOps()) {
    auto constraintBlock = dyn_cast<ConstraintBlockOp>(op);
    if (!constraintBlock)
      continue;

    // Walk the constraint block body looking for ConstraintSolveBeforeOp
    for (auto &constraintOp : constraintBlock.getBody().getOps()) {
      if (auto solveBeforeOp = dyn_cast<ConstraintSolveBeforeOp>(constraintOp)) {
        SolveBeforeInfo info;

        // Extract 'before' variable names
        for (auto beforeRef : solveBeforeOp.getBefore()) {
          auto symRef = cast<FlatSymbolRefAttr>(beforeRef);
          info.before.push_back(symRef.getValue());
        }

        // Extract 'after' variable names
        for (auto afterRef : solveBeforeOp.getAfter()) {
          auto symRef = cast<FlatSymbolRefAttr>(afterRef);
          info.after.push_back(symRef.getValue());
        }

        if (!info.before.empty() && !info.after.empty())
          ordering.push_back(std::move(info));
      }
    }
  }

  return ordering;
}

/// Build a dependency graph for solve-before ordering.
/// Returns a map from property name to the set of properties that must be
/// solved before it.
static llvm::DenseMap<StringRef, llvm::DenseSet<StringRef>>
buildSolveBeforeDependencies(const SmallVector<SolveBeforeInfo> &ordering) {
  llvm::DenseMap<StringRef, llvm::DenseSet<StringRef>> deps;

  for (const auto &info : ordering) {
    // For each 'after' variable, add all 'before' variables as dependencies
    for (StringRef afterVar : info.after) {
      for (StringRef beforeVar : info.before) {
        deps[afterVar].insert(beforeVar);
      }
    }
  }

  return deps;
}

/// Compute priority values for properties based on solve-before ordering.
/// Properties with lower priority values should be solved first.
/// Uses topological sort to assign priorities.
static llvm::DenseMap<StringRef, unsigned>
computeSolveOrder(const SmallVector<SolveBeforeInfo> &ordering,
                  const llvm::DenseSet<StringRef> &allProperties) {
  llvm::DenseMap<StringRef, unsigned> priority;

  // Build dependency graph
  auto deps = buildSolveBeforeDependencies(ordering);

  // Collect all properties that appear in solve-before constraints
  llvm::DenseSet<StringRef> constrainedProps;
  for (const auto &info : ordering) {
    for (StringRef prop : info.before)
      constrainedProps.insert(prop);
    for (StringRef prop : info.after)
      constrainedProps.insert(prop);
  }

  // Compute in-degree for topological sort
  llvm::DenseMap<StringRef, unsigned> inDegree;
  for (StringRef prop : constrainedProps) {
    inDegree[prop] = 0;
  }
  for (const auto &[prop, depSet] : deps) {
    inDegree[prop] = depSet.size();
  }

  // Kahn's algorithm for topological sort
  std::queue<StringRef> queue;
  for (StringRef prop : constrainedProps) {
    if (inDegree[prop] == 0)
      queue.push(prop);
  }

  unsigned currentPriority = 0;
  while (!queue.empty()) {
    StringRef current = queue.front();
    queue.pop();
    priority[current] = currentPriority++;

    // Find all properties that depend on 'current' and decrease their in-degree
    for (const auto &[prop, depSet] : deps) {
      if (depSet.contains(current)) {
        inDegree[prop]--;
        if (inDegree[prop] == 0)
          queue.push(prop);
      }
    }
  }

  // Assign default priority to unconstrained properties (they can be solved
  // in any order relative to solve-before constraints)
  unsigned defaultPriority = currentPriority;
  for (StringRef prop : allProperties) {
    if (!priority.count(prop))
      priority[prop] = defaultPriority;
  }

  return priority;
}

/// Sort constraints based on solve-before ordering.
/// Constraints for properties with lower priority are placed first.
template <typename T>
static void sortConstraintsBySolveOrder(
    SmallVector<T> &constraints,
    const llvm::DenseMap<StringRef, unsigned> &solveOrder) {
  llvm::stable_sort(constraints, [&](const T &a, const T &b) {
    unsigned prioA = solveOrder.lookup(a.propertyName);
    unsigned prioB = solveOrder.lookup(b.propertyName);
    return prioA < prioB;
  });
}

/// Helper to trace a value back through ReadOp to ClassPropertyRefOp.
/// Returns the property name if found, empty StringRef otherwise.
static StringRef traceToPropertyName(Value variable) {
  // In inline constraints, the variable is typically:
  //   %0 = moore.class.property_ref %obj[@propertyName] : ...
  //   %1 = moore.read %0 : ...
  //   moore.constraint.inside %1, [...] : ...
  // So we need to trace back through the read operation.

  // If it's directly a ClassPropertyRefOp, get the property name
  if (auto propRef = variable.getDefiningOp<ClassPropertyRefOp>())
    return propRef.getProperty();

  // If it's a ReadOp, look at its input
  if (auto readOp = variable.getDefiningOp<ReadOp>()) {
    Value input = readOp.getInput();
    if (auto propRef = input.getDefiningOp<ClassPropertyRefOp>())
      return propRef.getProperty();
  }

  return StringRef();
}

/// Extract range constraints from an inline constraint region.
/// In inline constraints, the variable operand traces back through ReadOp to
/// ClassPropertyRefOp, rather than being a block argument.
static SmallVector<RangeConstraintInfo>
extractInlineRangeConstraints(Region &inlineRegion,
                              ClassDeclOp classDecl,
                              ClassTypeCache &cache,
                              SymbolRefAttr classSym) {
  SmallVector<RangeConstraintInfo> constraints;

  if (inlineRegion.empty())
    return constraints;

  // Build a map from property names to their indices and types
  DenseMap<StringRef, std::pair<unsigned, Type>> propertyMap;
  unsigned propIdx = 0;

  // Account for type ID field (index 0 in root classes) or base class
  if (classDecl && classDecl.getBaseAttr())
    propIdx = 1;
  else
    propIdx = 1; // Type ID is at index 0

  if (classDecl) {
    for (auto &op : classDecl.getBody().getOps()) {
      if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(op)) {
        propertyMap[propDecl.getSymName()] = {propIdx, propDecl.getPropertyType()};
        propIdx++;
      }
    }
  }

  // Walk through the inline constraint region looking for ConstraintInsideOp
  for (auto &op : inlineRegion.front().getOperations()) {
    if (auto insideOp = dyn_cast<ConstraintInsideOp>(op)) {
      // Get the variable being constrained
      Value variable = insideOp.getVariable();

      // Trace back to find the property name
      StringRef propName = traceToPropertyName(variable);
      if (propName.empty())
        continue;

      // Get the ranges array
      ArrayRef<int64_t> rangesArr = insideOp.getRanges();
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
      info.constraintName = ""; // Inline constraints don't have a name
      info.fieldIndex = fieldIdx;
      info.bitWidth = bitWidth;
      info.isSoft = insideOp.getIsSoft();

      // Check if this is a single range or multiple ranges
      if (rangesArr.size() == 2) {
        info.minValue = rangesArr[0];
        info.maxValue = rangesArr[1];
        info.isMultiRange = false;
      } else {
        info.isMultiRange = true;
        info.minValue = rangesArr[0];
        info.maxValue = rangesArr[1];
        for (size_t i = 0; i < rangesArr.size(); i += 2)
          info.ranges.push_back({rangesArr[i], rangesArr[i + 1]});
      }
      constraints.push_back(info);
    }
  }

  return constraints;
}

/// Extract distribution constraints from an inline constraint region.
static SmallVector<DistConstraintInfo>
extractInlineDistConstraints(Region &inlineRegion,
                             ClassDeclOp classDecl,
                             ClassTypeCache &cache,
                             SymbolRefAttr classSym) {
  SmallVector<DistConstraintInfo> distConstraints;

  if (inlineRegion.empty())
    return distConstraints;

  // Build a map from property names to their indices and types
  DenseMap<StringRef, std::pair<unsigned, Type>> propertyMap;
  unsigned propIdx = 0;

  // Account for type ID field (index 0 in root classes) or base class
  if (classDecl && classDecl.getBaseAttr())
    propIdx = 1;
  else
    propIdx = 1; // Type ID is at index 0

  if (classDecl) {
    for (auto &op : classDecl.getBody().getOps()) {
      if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(op)) {
        propertyMap[propDecl.getSymName()] = {propIdx, propDecl.getPropertyType()};
        propIdx++;
      }
    }
  }

  // Walk through the inline constraint region looking for ConstraintDistOp
  for (auto &op : inlineRegion.front().getOperations()) {
    if (auto distOp = dyn_cast<ConstraintDistOp>(op)) {
      Value variable = distOp.getVariable();

      // Trace back to find the property name
      StringRef propName = traceToPropertyName(variable);
      if (propName.empty())
        continue;

      // Get the distribution data from the op
      ArrayRef<int64_t> values = distOp.getValues();
      ArrayRef<int64_t> weights = distOp.getWeights();
      ArrayRef<int64_t> perRange = distOp.getPerRange();

      // Values come in pairs: [low0, high0, low1, high1, ...]
      if (values.size() < 2 || values.size() % 2 != 0)
        continue;
      if (weights.size() != values.size() / 2)
        continue;
      if (perRange.size() != weights.size())
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

      DistConstraintInfo info;
      info.propertyName = propName;
      info.constraintName = ""; // Inline constraints don't have a name
      info.fieldIndex = fieldIdx;
      info.bitWidth = bitWidth;
      info.isSigned = distOp.getIsSignedAttr() != nullptr;

      // Convert values array to ranges vector
      for (size_t i = 0; i < values.size(); i += 2)
        info.ranges.push_back({values[i], values[i + 1]});

      // Copy weights and perRange
      for (int64_t w : weights)
        info.weights.push_back(w);
      for (int64_t p : perRange)
        info.perRange.push_back(p);

      distConstraints.push_back(info);
    }
  }

  return distConstraints;
}

/// Extract soft constraints from an inline constraint region.
static SmallVector<SoftConstraintInfo>
extractInlineSoftConstraints(Region &inlineRegion,
                             ClassDeclOp classDecl,
                             ClassTypeCache &cache,
                             SymbolRefAttr classSym) {
  SmallVector<SoftConstraintInfo> softConstraints;

  if (inlineRegion.empty())
    return softConstraints;

  // Build a map from property names to their indices and types
  DenseMap<StringRef, std::pair<unsigned, Type>> propertyMap;
  unsigned propIdx = 0;

  // Account for type ID field (index 0 in root classes) or base class
  if (classDecl && classDecl.getBaseAttr())
    propIdx = 1;
  else
    propIdx = 1; // Type ID is at index 0

  if (classDecl) {
    for (auto &op : classDecl.getBody().getOps()) {
      if (auto propDecl = dyn_cast<ClassPropertyDeclOp>(op)) {
        propertyMap[propDecl.getSymName()] = {propIdx, propDecl.getPropertyType()};
        propIdx++;
      }
    }
  }

  // Walk through the inline constraint region looking for soft ConstraintInsideOp
  for (auto &op : inlineRegion.front().getOperations()) {
    if (auto insideOp = dyn_cast<ConstraintInsideOp>(op)) {
      // Only process soft constraints
      if (!insideOp.getIsSoft())
        continue;

      Value variable = insideOp.getVariable();

      // Trace back to find the property name
      StringRef propName = traceToPropertyName(variable);
      if (propName.empty())
        continue;

      // Get the ranges array
      ArrayRef<int64_t> ranges = insideOp.getRanges();
      if (ranges.size() < 2)
        continue;

      int64_t defaultValue = ranges[0];

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
      info.constraintName = ""; // Inline constraints don't have a name
      info.defaultValue = defaultValue;
      info.fieldIndex = fieldIdx;
      info.bitWidth = bitWidth;
      softConstraints.push_back(info);
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
      // Use high benefit (10) to ensure this pattern runs before
      // ConstraintInsideOpConversion erases ops in the inline region.
      : OpConversionPattern<RandomizeOp>(tc, ctx, /*benefit=*/10), cache(cache) {}

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
    SmallVector<std::tuple<Value, Value, Value>> conditionalRestores;
    auto createRandEnabledCheck = [&](StringRef propertyName) -> Value {
      std::string globalName =
          ("__rand_name_" + classSym.getRootReference().str() + "_" +
           propertyName)
              .str();
      Value namePtr = createGlobalStringConstant(loc, mod, rewriter,
                                                 propertyName, globalName);
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_is_rand_enabled",
                                       fnTy);
      auto enabledVal = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(fn),
          ValueRange{classPtr, namePtr});
      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                   rewriter.getI32IntegerAttr(0));
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                   enabledVal.getResult(), zero);
    };

    if (classDecl) {
      for (auto propDecl : classDecl.getBody().getOps<ClassPropertyDeclOp>()) {
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
        if (propDecl.isRandomizable()) {
          Value enabled = createRandEnabledCheck(propDecl.getSymName());
          auto zero = arith::ConstantOp::create(
              rewriter, loc, i1Ty, rewriter.getBoolAttr(false));
          Value shouldRestore = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, enabled, zero);
          conditionalRestores.emplace_back(fieldPtr, fieldVal, shouldRestore);
        } else {
          preservedFields.push_back({fieldPtr, fieldVal});
        }
      }
    }

    auto restorePreservedFields = [&]() {
      for (auto &entry : preservedFields)
        LLVM::StoreOp::create(rewriter, loc, entry.second, entry.first);
      for (auto &entry : conditionalRestores) {
        Value fieldPtr = std::get<0>(entry);
        Value fieldVal = std::get<1>(entry);
        Value shouldRestore = std::get<2>(entry);
        auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, shouldRestore,
                                      /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        LLVM::StoreOp::create(rewriter, loc, fieldVal, fieldPtr);
      }
    };

    auto createConstraintEnabledCheck = [&](StringRef constraintName) -> Value {
      std::string globalName =
          ("__constraint_name_" + classSym.getRootReference().str() + "_" +
           constraintName)
              .str();
      Value namePtr = createGlobalStringConstant(loc, mod, rewriter,
                                                 constraintName, globalName);
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
      auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                       "__moore_is_constraint_enabled", fnTy);
      auto enabledVal = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(fn),
          ValueRange{classPtr, namePtr});
      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                   rewriter.getI32IntegerAttr(0));
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                   enabledVal.getResult(), zero);
    };

    auto applyRandcFields =
        [&](const llvm::DenseMap<StringRef, SmallVector<StringRef>>
                &constraintsByProperty) {
      if (!classDecl)
        return;

      auto randcFnTy = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty});
      auto randcFn = getOrCreateRuntimeFunc(mod, rewriter, "__moore_randc_next",
                                            randcFnTy);

      for (auto propDecl : classDecl.getBody().getOps<ClassPropertyDeclOp>()) {
        if (propDecl.getRandMode() != RandMode::RandC)
          continue;
        OpBuilder::InsertionGuard guard(rewriter);
        Value randEnabled = createRandEnabledCheck(propDecl.getSymName());
        auto constraintIt = constraintsByProperty.find(propDecl.getSymName());
        if (constraintIt != constraintsByProperty.end() &&
            !constraintIt->second.empty()) {
          Value anyEnabled = arith::ConstantOp::create(
              rewriter, loc, i1Ty, rewriter.getBoolAttr(false));
          for (auto constraintName : constraintIt->second) {
            if (constraintName.empty())
              continue;
            Value enabled = createConstraintEnabledCheck(constraintName);
            anyEnabled =
                arith::OrIOp::create(rewriter, loc, anyEnabled, enabled);
          }
          Value shouldApply = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, anyEnabled,
              arith::ConstantOp::create(rewriter, loc, i1Ty,
                                        rewriter.getBoolAttr(false)));
          Value applyCond =
              arith::AndIOp::create(rewriter, loc, randEnabled, shouldApply);
          auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, applyCond,
                                        /*withElseRegion=*/false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        } else {
          auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, randEnabled,
                                        /*withElseRegion=*/false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        }

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
    SmallVector<DistConstraintInfo> distConstraints;
    SmallVector<SolveBeforeInfo> solveBeforeOrdering;
    if (classDecl) {
      rangeConstraints = extractRangeConstraints(classDecl, cache, classSym);
      softConstraints = extractSoftConstraints(classDecl, cache, classSym);
      distConstraints = extractDistConstraints(classDecl, cache, classSym);
      solveBeforeOrdering = extractSolveBeforeOrdering(classDecl);
    }

    // Also extract constraints from the inline constraint region (if present)
    // Inline constraints are specified with `randomize() with { ... }` syntax
    // and are combined with class-level constraints (IEEE 1800-2017 Section 18.7)
    Region &inlineRegion = op.getInlineConstraints();
    if (!inlineRegion.empty()) {
      auto inlineRangeConstraints =
          extractInlineRangeConstraints(inlineRegion, classDecl, cache, classSym);
      auto inlineSoftConstraints =
          extractInlineSoftConstraints(inlineRegion, classDecl, cache, classSym);
      auto inlineDistConstraints =
          extractInlineDistConstraints(inlineRegion, classDecl, cache, classSym);

      // Merge inline constraints with class-level constraints
      // Inline constraints take precedence (they are appended last)
      rangeConstraints.append(inlineRangeConstraints.begin(),
                              inlineRangeConstraints.end());
      softConstraints.append(inlineSoftConstraints.begin(),
                             inlineSoftConstraints.end());
      distConstraints.append(inlineDistConstraints.begin(),
                             inlineDistConstraints.end());

      // Erase all ops in the inline constraint region now that we've extracted
      // the constraint information. This prevents other patterns from trying
      // to convert these ops (which would fail since we skip them).
      Block &inlineBlock = inlineRegion.front();
      SmallVector<Operation *> opsToErase;
      for (auto &op : inlineBlock.getOperations())
        opsToErase.push_back(&op);
      for (auto *op : llvm::reverse(opsToErase))
        rewriter.eraseOp(op);
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

    // Distribution constraints also mark properties as having hard constraints
    for (const auto &dist : distConstraints) {
      hardConstrainedProps.insert(dist.propertyName);
    }

    // Filter soft constraints to only those without hard constraints
    SmallVector<SoftConstraintInfo> effectiveSoftConstraints;
    for (const auto &soft : softConstraints) {
      if (!hardConstrainedProps.contains(soft.propertyName)) {
        effectiveSoftConstraints.push_back(soft);
      }
    }

    // Apply solve-before ordering to constraints if any ordering is specified
    if (!solveBeforeOrdering.empty()) {
      // Collect all properties that have constraints
      llvm::DenseSet<StringRef> allConstrainedProps;
      for (const auto &c : hardConstraints)
        allConstrainedProps.insert(c.propertyName);
      for (const auto &c : effectiveSoftConstraints)
        allConstrainedProps.insert(c.propertyName);
      for (const auto &c : distConstraints)
        allConstrainedProps.insert(c.propertyName);

      // Compute solve order (priority values for each property)
      auto solveOrder = computeSolveOrder(solveBeforeOrdering, allConstrainedProps);

      // Sort constraints by solve order so that 'before' variables are
      // randomized first
      sortConstraintsBySolveOrder(hardConstraints, solveOrder);
      sortConstraintsBySolveOrder(effectiveSoftConstraints, solveOrder);
      sortConstraintsBySolveOrder(distConstraints, solveOrder);
    }

    // Build a map of properties to constraint names for randc gating.
    llvm::DenseMap<StringRef, SmallVector<StringRef>> constraintsByProperty;
    for (const auto &constraint : hardConstraints) {
      if (!constraint.constraintName.empty())
        constraintsByProperty[constraint.propertyName].push_back(
            constraint.constraintName);
    }
    for (const auto &dist : distConstraints) {
      if (!dist.constraintName.empty())
        constraintsByProperty[dist.propertyName].push_back(dist.constraintName);
    }
    for (const auto &soft : effectiveSoftConstraints) {
      if (!soft.constraintName.empty())
        constraintsByProperty[soft.propertyName].push_back(
            soft.constraintName);
    }

    // If we have any constraints, use constraint-aware randomization
    if (!hardConstraints.empty() || !effectiveSoftConstraints.empty() ||
        !distConstraints.empty()) {
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
          OpBuilder::InsertionGuard guard(rewriter);
          Value randEnabled = createRandEnabledCheck(constraint.propertyName);
          Value applyCond = randEnabled;
          if (!constraint.constraintName.empty()) {
            Value enabled = createConstraintEnabledCheck(
                constraint.constraintName);
            applyCond = arith::AndIOp::create(rewriter, loc, randEnabled,
                                              enabled);
          }
          auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, applyCond,
                                        /*withElseRegion=*/false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

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

      // Apply distribution constraints - generate weighted random values
      // using __moore_randomize_with_dist(ranges, weights, perRange, numRanges)
      if (!distConstraints.empty()) {
        // Function type: __moore_randomize_with_dist(ptr, ptr, ptr, i64, i64) -> i64
        auto distFnTy = LLVM::LLVMFunctionType::get(
            i64Ty, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty});
        auto distFn = getOrCreateRuntimeFunc(mod, rewriter,
                                             "__moore_randomize_with_dist",
                                             distFnTy);

        for (const auto &dist : distConstraints) {
          OpBuilder::InsertionGuard guard(rewriter);
          Value randEnabled = createRandEnabledCheck(dist.propertyName);
          Value applyCond = randEnabled;
          if (!dist.constraintName.empty()) {
            Value enabled = createConstraintEnabledCheck(dist.constraintName);
            applyCond =
                arith::AndIOp::create(rewriter, loc, randEnabled, enabled);
          }
          auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, applyCond,
                                        /*withElseRegion=*/false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

          size_t numRanges = dist.ranges.size();

          // Allocate stack space for ranges array: [numRanges * 2 x i64]
          auto rangesArrayTy = LLVM::LLVMArrayType::get(i64Ty, numRanges * 2);
          auto one = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(1));
          auto rangesAlloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, rangesArrayTy, one);

          // Allocate stack space for weights array: [numRanges x i64]
          auto weightsArrayTy = LLVM::LLVMArrayType::get(i64Ty, numRanges);
          auto weightsAlloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, weightsArrayTy, one);

          // Allocate stack space for perRange array: [numRanges x i64]
          auto perRangeArrayTy = LLVM::LLVMArrayType::get(i64Ty, numRanges);
          auto perRangeAlloca =
              LLVM::AllocaOp::create(rewriter, loc, ptrTy, perRangeArrayTy, one);

          // Store ranges (as pairs of [low, high])
          for (size_t i = 0; i < numRanges; ++i) {
            // Store low value at index i*2
            auto lowIdx = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(static_cast<int64_t>(i * 2)));
            auto lowPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                              rangesAlloca, ValueRange{lowIdx});
            auto lowVal = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(dist.ranges[i].first));
            LLVM::StoreOp::create(rewriter, loc, lowVal, lowPtr);

            // Store high value at index i*2+1
            auto highIdx = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(static_cast<int64_t>(i * 2 + 1)));
            auto highPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                               rangesAlloca,
                                               ValueRange{highIdx});
            auto highVal = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(dist.ranges[i].second));
            LLVM::StoreOp::create(rewriter, loc, highVal, highPtr);

            // Store weight at index i
            auto weightIdx = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(static_cast<int64_t>(i)));
            auto weightPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                                 weightsAlloca,
                                                 ValueRange{weightIdx});
            auto weightVal = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(dist.weights[i]));
            LLVM::StoreOp::create(rewriter, loc, weightVal, weightPtr);

            // Store perRange flag at index i
            auto perRangePtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty,
                                                   perRangeAlloca,
                                                   ValueRange{weightIdx});
            auto perRangeVal = LLVM::ConstantOp::create(
                rewriter, loc, i64Ty,
                rewriter.getI64IntegerAttr(dist.perRange[i]));
            LLVM::StoreOp::create(rewriter, loc, perRangeVal, perRangePtr);
          }

          // Call __moore_randomize_with_dist(ranges, weights, perRange, numRanges, isSigned)
          auto numRangesConst = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty,
              rewriter.getI64IntegerAttr(static_cast<int64_t>(numRanges)));
          auto isSignedConst = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty,
              rewriter.getI64IntegerAttr(dist.isSigned ? 1 : 0));
          auto distResult = LLVM::CallOp::create(
              rewriter, loc, TypeRange{i64Ty}, SymbolRefAttr::get(distFn),
              ValueRange{rangesAlloca, weightsAlloca, perRangeAlloca,
                         numRangesConst, isSignedConst});

          // Truncate to the field's bit width if needed
          Type fieldIntTy = IntegerType::get(ctx, dist.bitWidth);
          Value truncatedVal = distResult.getResult();
          if (dist.bitWidth < 64) {
            truncatedVal = arith::TruncIOp::create(rewriter, loc, fieldIntTy,
                                                   distResult.getResult());
          }

          // Get GEP to the field using the property path
          auto it = structInfo->propertyPath.find(dist.propertyName);
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

            // Store the distribution-constrained random value
            LLVM::StoreOp::create(rewriter, loc, truncatedVal, fieldPtr);
          }
        }
      }

      // Apply soft constraints - set default values for properties without hard
      // constraints. Soft constraints provide fallback values that can be
      // overridden.
      for (const auto &soft : effectiveSoftConstraints) {
        OpBuilder::InsertionGuard guard(rewriter);
        Value randEnabled = createRandEnabledCheck(soft.propertyName);
        Value applyCond = randEnabled;
        if (!soft.constraintName.empty()) {
          Value enabled = createConstraintEnabledCheck(soft.constraintName);
          applyCond =
              arith::AndIOp::create(rewriter, loc, randEnabled, enabled);
        }
        auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, applyCond,
                                      /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

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

      applyRandcFields(constraintsByProperty);
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

    applyRandcFields(constraintsByProperty);
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

/// Conversion for moore.constraint_mode -> runtime function call.
/// Implements the SystemVerilog constraint_mode() method for enabling/disabling
/// constraints during randomization (IEEE 1800-2017 Section 18.8).
struct ConstraintModeOpConversion
    : public OpConversionPattern<ConstraintModeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstraintModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    Value classPtr = adaptor.getObject();
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    StringRef className = handleTy.getClassSym().getRootReference();
    Value namePtr =
        LLVM::ZeroOp::create(rewriter, loc, ptrTy);

    if (op.getConstraint().has_value()) {
      StringRef constraintName = op.getConstraint().value();
      std::string globalName =
          ("__constraint_name_" + className + "_" + constraintName).str();
      namePtr = createGlobalStringConstant(loc, mod, rewriter, constraintName,
                                           globalName);
    }

    auto convertMode = [&](Value modeVal) -> Value {
      if (!modeVal)
        return {};
      if (modeVal.getType() == i32Ty)
        return modeVal;
      auto modeIntTy = dyn_cast<IntegerType>(modeVal.getType());
      if (!modeIntTy)
        return modeVal;
      if (modeIntTy.getWidth() < 32)
        return arith::ExtUIOp::create(rewriter, loc, i32Ty, modeVal);
      if (modeIntTy.getWidth() > 32)
        return arith::TruncIOp::create(rewriter, loc, i32Ty, modeVal);
      return modeVal;
    };

    if (adaptor.getMode()) {
      Value modeI32 = convertMode(adaptor.getMode());
      if (!modeI32)
        return op.emitError() << "constraint_mode expects integer mode";

      if (op.getConstraint().has_value()) {
        auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, i32Ty});
        auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                         "__moore_constraint_mode_set", fnTy);
        auto resultVal = LLVM::CallOp::create(
            rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(fn),
            ValueRange{classPtr, namePtr, modeI32});
        rewriter.replaceOp(op, resultVal.getResult());
        return success();
      }

      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                   rewriter.getI32IntegerAttr(0));
      Value isEnable = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, modeI32, zero);
      auto ifOp =
          scf::IfOp::create(rewriter, loc, TypeRange{i32Ty}, isEnable,
                            /*withElseRegion=*/true);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto enableFnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
      auto enableFn = getOrCreateRuntimeFunc(
          mod, rewriter, "__moore_constraint_mode_enable_all", enableFnTy);
      auto enableCall = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(enableFn),
          ValueRange{classPtr});
      scf::YieldOp::create(rewriter, loc, enableCall.getResult());

      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto disableFnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
      auto disableFn = getOrCreateRuntimeFunc(
          mod, rewriter, "__moore_constraint_mode_disable_all", disableFnTy);
      auto disableCall = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(disableFn),
          ValueRange{classPtr});
      scf::YieldOp::create(rewriter, loc, disableCall.getResult());

      rewriter.replaceOp(op, ifOp.getResult(0));
      return success();
    }

    // Getter: constraint-specific or class-level.
    if (!op.getConstraint().has_value()) {
      std::string globalName =
          ("__constraint_name_" + className + "__all__").str();
      namePtr = createGlobalStringConstant(loc, mod, rewriter, "__all__",
                                           globalName);
    }
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                     "__moore_constraint_mode_get", fnTy);
    auto resultVal = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                          SymbolRefAttr::get(fn),
                                          ValueRange{classPtr, namePtr});
    rewriter.replaceOp(op, resultVal.getResult());
    return success();
  }
};

/// Conversion for moore.rand_mode -> runtime function call.
/// Implements the SystemVerilog rand_mode() method for enabling/disabling
/// random variables during randomization (IEEE 1800-2017 Section 18.8).
struct RandModeOpConversion : public OpConversionPattern<RandModeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RandModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    Value classPtr = adaptor.getObject();
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    StringRef className = handleTy.getClassSym().getRootReference();
    Value namePtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);

    if (op.getProperty().has_value()) {
      StringRef propertyName = op.getProperty().value();
      std::string globalName =
          ("__rand_name_" + className + "_" + propertyName).str();
      namePtr = createGlobalStringConstant(loc, mod, rewriter, propertyName,
                                           globalName);
    }

    auto convertMode = [&](Value modeVal) -> Value {
      if (!modeVal)
        return {};
      if (modeVal.getType() == i32Ty)
        return modeVal;
      auto modeIntTy = dyn_cast<IntegerType>(modeVal.getType());
      if (!modeIntTy)
        return modeVal;
      if (modeIntTy.getWidth() < 32)
        return arith::ExtUIOp::create(rewriter, loc, i32Ty, modeVal);
      if (modeIntTy.getWidth() > 32)
        return arith::TruncIOp::create(rewriter, loc, i32Ty, modeVal);
      return modeVal;
    };

    if (adaptor.getMode()) {
      Value modeI32 = convertMode(adaptor.getMode());
      if (!modeI32)
        return op.emitError() << "rand_mode expects integer mode";

      if (op.getProperty().has_value()) {
        auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, i32Ty});
        auto fn = getOrCreateRuntimeFunc(mod, rewriter,
                                         "__moore_rand_mode_set", fnTy);
        auto resultVal = LLVM::CallOp::create(
            rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(fn),
            ValueRange{classPtr, namePtr, modeI32});
        rewriter.replaceOp(op, resultVal.getResult());
        return success();
      }

      auto zero =
          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                   rewriter.getI32IntegerAttr(0));
      Value isEnable = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, modeI32, zero);
      auto ifOp =
          scf::IfOp::create(rewriter, loc, TypeRange{i32Ty}, isEnable,
                            /*withElseRegion=*/true);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto enableFnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
      auto enableFn = getOrCreateRuntimeFunc(
          mod, rewriter, "__moore_rand_mode_enable_all", enableFnTy);
      auto enableCall = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(enableFn),
          ValueRange{classPtr});
      scf::YieldOp::create(rewriter, loc, enableCall.getResult());

      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto disableFnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy});
      auto disableFn = getOrCreateRuntimeFunc(
          mod, rewriter, "__moore_rand_mode_disable_all", disableFnTy);
      auto disableCall = LLVM::CallOp::create(
          rewriter, loc, TypeRange{i32Ty}, SymbolRefAttr::get(disableFn),
          ValueRange{classPtr});
      scf::YieldOp::create(rewriter, loc, disableCall.getResult());

      rewriter.replaceOp(op, ifOp.getResult(0));
      return success();
    }

    if (!op.getProperty().has_value()) {
      std::string globalName = ("__rand_name_" + className + "__all__").str();
      namePtr = createGlobalStringConstant(loc, mod, rewriter, "__all__",
                                           globalName);
    }
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy});
    auto fn =
        getOrCreateRuntimeFunc(mod, rewriter, "__moore_rand_mode_get", fnTy);
    auto resultVal = LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                                          SymbolRefAttr::get(fn),
                                          ValueRange{classPtr, namePtr});
    rewriter.replaceOp(op, resultVal.getResult());
    return success();
  }
};

/// Conversion for moore.call_pre_randomize -> direct function call.
/// Invokes the pre_randomize() method before randomization begins.
/// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
struct CallPreRandomizeOpConversion
    : public OpConversionPattern<CallPreRandomizeOp> {
  CallPreRandomizeOpConversion(TypeConverter &tc, MLIRContext *ctx,
                               ClassTypeCache &cache)
      : OpConversionPattern<CallPreRandomizeOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(CallPreRandomizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // Get the class handle type and resolve class info
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    auto classSym = handleTy.getClassSym();
    StringRef className = classSym.getRootReference();

    // Check if the class has a user-defined pre_randomize method
    auto *classDeclSym = mod.lookupSymbol(classSym);
    auto classDecl = dyn_cast_or_null<ClassDeclOp>(classDeclSym);

    bool foundMethod = false;
    if (classDecl) {
      // First, look for a pre_randomize method declaration (for virtual case)
      for (auto methodDecl :
           classDecl.getBody().getOps<ClassMethodDeclOp>()) {
        if (methodDecl.getSymName() == "pre_randomize") {
          // Found a pre_randomize method - call it directly if it has an impl
          if (methodDecl.getImpl().has_value()) {
            auto implSymRef = methodDecl.getImpl().value();

            // The pre_randomize method takes 'this' as argument and returns
            // void. Call the implementation function directly.
            Value classPtr = adaptor.getObject();
            func::CallOp::create(rewriter, loc, implSymRef, TypeRange{},
                                 ValueRange{classPtr});
            foundMethod = true;
          }
          break;
        }
      }
    }

    // If not found via ClassMethodDeclOp, look for a func.func with the
    // conventional name "ClassName::pre_randomize". This handles non-virtual
    // pre_randomize methods which don't get ClassMethodDeclOp entries.
    if (!foundMethod) {
      std::string funcName = (className + "::pre_randomize").str();
      auto funcOp = mod.lookupSymbol<func::FuncOp>(funcName);
      if (funcOp) {
        Value classPtr = adaptor.getObject();
        func::CallOp::create(rewriter, loc,
                             SymbolRefAttr::get(rewriter.getContext(), funcName),
                             TypeRange{}, ValueRange{classPtr});
      }
    }

    // pre_randomize returns void, so just erase the operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  ClassTypeCache &cache;
};

/// Conversion for moore.call_post_randomize -> direct function call.
/// Invokes the post_randomize() method after successful randomization.
/// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
struct CallPostRandomizeOpConversion
    : public OpConversionPattern<CallPostRandomizeOp> {
  CallPostRandomizeOpConversion(TypeConverter &tc, MLIRContext *ctx,
                                ClassTypeCache &cache)
      : OpConversionPattern<CallPostRandomizeOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(CallPostRandomizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();

    // Get the class handle type and resolve class info
    auto handleTy = cast<ClassHandleType>(op.getObject().getType());
    auto classSym = handleTy.getClassSym();
    StringRef className = classSym.getRootReference();

    // Check if the class has a user-defined post_randomize method
    auto *classDeclSym = mod.lookupSymbol(classSym);
    auto classDecl = dyn_cast_or_null<ClassDeclOp>(classDeclSym);

    bool foundMethod = false;
    if (classDecl) {
      // First, look for a post_randomize method declaration (for virtual case)
      for (auto methodDecl :
           classDecl.getBody().getOps<ClassMethodDeclOp>()) {
        if (methodDecl.getSymName() == "post_randomize") {
          // Found a post_randomize method - call it directly if it has an impl
          if (methodDecl.getImpl().has_value()) {
            auto implSymRef = methodDecl.getImpl().value();

            // The post_randomize method takes 'this' as argument and returns
            // void. Call the implementation function directly.
            Value classPtr = adaptor.getObject();
            func::CallOp::create(rewriter, loc, implSymRef, TypeRange{},
                                 ValueRange{classPtr});
            foundMethod = true;
          }
          break;
        }
      }
    }

    // If not found via ClassMethodDeclOp, look for a func.func with the
    // conventional name "ClassName::post_randomize". This handles non-virtual
    // post_randomize methods which don't get ClassMethodDeclOp entries.
    if (!foundMethod) {
      std::string funcName = (className + "::post_randomize").str();
      auto funcOp = mod.lookupSymbol<func::FuncOp>(funcName);
      if (funcOp) {
        Value classPtr = adaptor.getObject();
        func::CallOp::create(rewriter, loc,
                             SymbolRefAttr::get(rewriter.getContext(), funcName),
                             TypeRange{}, ValueRange{classPtr});
      }
    }

    // post_randomize returns void, so just erase the operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  ClassTypeCache &cache;
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target,
                             const TypeConverter &converter) {
  target.addIllegalDialect<MooreDialect>();

  // Array locator operations must be explicitly illegal because they have
  // regions that need special handling during conversion.
  target.addIllegalOp<ArrayLocatorOp>();
  target.addIllegalOp<ArrayLocatorYieldOp>();

  // WaitEventOp and DetectEventOp need special handling: they can only be
  // converted to llhd.wait when inside an llhd.process. When they appear in
  // func.func (e.g., class tasks with timing controls), they must remain
  // unconverted until the function is inlined into a process by the
  // InlineCalls pass. Mark them as dynamically legal based on context.
  target.addDynamicallyLegalOp<WaitEventOp>([](WaitEventOp op) {
    // Legal (keep unconverted) if NOT inside an llhd.process.
    // After inlining into a process, this will become illegal and get
    // converted.
    return !op->getParentOfType<llhd::ProcessOp>();
  });
  target.addDynamicallyLegalOp<DetectEventOp>([](DetectEventOp op) {
    // DetectEventOp is always inside WaitEventOp, so follow the same rule.
    return !op->getParentOfType<llhd::ProcessOp>();
  });
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
  // Note: cf::ControlFlowDialect is NOT added as fully legal here because
  // cf ops with block arguments need type conversion. The legality for cf
  // ops is instead handled by populateCFStructuralTypeConversionsAndLegality
  // which adds dynamic legality based on whether the types are converted.

  // arith.select on Moore types needs to be converted to comb.mux.
  // This handles cases where MLIR canonicalizers introduce arith.select
  // on Moore types during control flow simplification.
  // However, arith.select on sim types (like !sim.fstring) should remain as-is
  // since they are already in a legal dialect and comb.mux doesn't support them.
  target.addDynamicallyLegalOp<arith::SelectOp>([&](arith::SelectOp op) {
    auto type = op.getTrueValue().getType();
    // Sim types are already legal and don't need conversion to comb.mux
    if (isa<sim::FormatStringType, sim::DynamicStringType>(type))
      return true;
    // LLVM types should stay as arith.select since comb.mux doesn't support them
    if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType>(type))
      return true;
    // LLHD ref types should stay as arith.select since comb.mux doesn't support
    // them. This handles cases like `sel ? reg_a : reg_b` where reg_a/reg_b are
    // signal references.
    if (isa<llhd::RefType>(type))
      return true;
    return converter.isLegal(type);
  });

  target.addLegalOp<debug::ScopeOp>();

  target.addDynamicallyLegalOp<scf::YieldOp, func::CallIndirectOp,
                               func::ReturnOp, func::ConstantOp, hw::OutputOp,
                               hw::InstanceOp, debug::ArrayOp, debug::StructOp,
                               debug::VariableOp>(
      [&](Operation *op) { return converter.isLegal(op); });

  // func::CallOp needs special handling for UVM report function interception.
  // Calls to uvm_pkg::uvm_report_* must be marked illegal so they get converted
  // to runtime function calls.
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    StringRef callee = op.getCallee();
    // UVM report free functions
    if (callee == "uvm_pkg::uvm_report_error" ||
        callee == "uvm_pkg::uvm_report_warning" ||
        callee == "uvm_pkg::uvm_report_info" ||
        callee == "uvm_pkg::uvm_report_fatal")
      return false;
    // UVM report class methods
    if (callee == "uvm_pkg::uvm_report_object::uvm_report_error" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_warning" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_info" ||
        callee == "uvm_pkg::uvm_report_object::uvm_report_fatal")
      return false;
    return converter.isLegal(op);
  });

  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&](UnrealizedConversionCastOp op) {
        if (op.getNumOperands() == 1 && op.getNumResults() == 1 &&
            op.getResult(0).getType().isInteger(1) &&
            isFourStateStructType(op.getOperand(0).getType()))
          return false;
        // Allow casts between hw.array and llvm.array types.
        // These are used when lowering fixed-size arrays that need to be
        // passed to runtime functions.
        if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
          Type inputType = op.getOperand(0).getType();
          Type outputType = op.getResult(0).getType();
          if ((isa<hw::ArrayType>(inputType) &&
               isa<LLVM::LLVMArrayType>(outputType)) ||
              (isa<LLVM::LLVMArrayType>(inputType) &&
               isa<hw::ArrayType>(outputType)))
            return true;
        }
        return converter.isLegal(op);
      });

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
  // Integer type conversion:
  // - Two-valued types (!moore.iN) are lowered to plain IntegerType (iN)
  // - Four-valued types (!moore.lN) are lowered to a struct {value: iN, unknown: iN}
  //   where unknown[i]=1 means bit i is X or Z, and when unknown[i]=1,
  //   value[i]=0 means X, value[i]=1 means Z.
  //
  // This enables proper X/Z propagation through the lowered operations.
  // Operations check for the struct type to decide whether to use 4-state logic.
  typeConverter.addConversion([&](IntType type) -> Type {
    auto width = type.getWidth();
    auto *ctx = type.getContext();

    if (type.getDomain() == Domain::FourValued) {
      // Four-valued type: use struct {value: iN, unknown: iN}
      return getFourStateStructType(ctx, width);
    }
    // Two-valued type: plain integer
    return IntegerType::get(ctx, width);
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

  // TimeType -> i64 (time in femtoseconds as a 64-bit integer)
  // We use i64 instead of llhd::TimeType because:
  // 1. hw::BitcastOp doesn't support llhd::TimeType
  // 2. Most operations treat time as a 64-bit integer anyway
  // 3. llhd.wait and other LLHD ops that need llhd.time can construct it
  typeConverter.addConversion([&](TimeType type) -> Type {
    return IntegerType::get(type.getContext(), 64);
  });

  // EventType -> i1 (tracks whether the event has been triggered).
  // In simulation, events are tracked as boolean flags indicating their
  // triggered state within the current time slot.
  typeConverter.addConversion([&](EventType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });

  // VoidType -> i0 (zero-width integer).
  // This is used in tagged unions where some members have no data (void).
  // The zero-width type allows the union structure to be preserved.
  typeConverter.addConversion([&](VoidType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 0);
  });

  typeConverter.addConversion([&](FormatStringType type) {
    return sim::FormatStringType::get(type.getContext());
  });

  // Add identity conversions for sim types that may appear in CF block arguments
  // after canonicalization. The CF structural type conversion patterns check
  // if types are legal using converter.isLegal(type), which requires an identity
  // conversion for types that are already in their final form.
  typeConverter.addConversion(
      [&](sim::FormatStringType type) { return type; });
  typeConverter.addConversion(
      [&](sim::DynamicStringType type) { return type; });

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
        auto elementType = typeConverter.convertType(type.getElementType());
        if (!elementType)
          return {};
        // If the element type converts to an LLVM type (e.g., strings which
        // become {ptr, i64} structs), use LLVM::LLVMArrayType instead of
        // hw::ArrayType since hw::ArrayType cannot contain LLVM types.
        if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType>(elementType))
          return LLVM::LLVMArrayType::get(elementType, type.getSize());
        return hw::ArrayType::get(elementType, type.getSize());
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
          // dynamic arrays, assoc arrays, or nested structs with these).
          // Note: TimeType now converts to i64, not llhd::TimeType.
          if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType>(convertedType))
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

  // WildcardAssocArrayType (wildcard associative array [*]) -> LLVM pointer
  // (opaque map handle). Wildcard associative arrays use any indexing type,
  // so they're represented the same as regular associative arrays at runtime.
  typeConverter.addConversion(
      [&](WildcardAssocArrayType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });

  // Convert packed union type to hw::UnionType
  // For packed unions, member types are converted without 4-state expansion
  // because all members share the same underlying bit storage.
  typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
    SmallVector<hw::UnionType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::UnionType::FieldInfo info;
      // Use packed conversion to avoid 4-state expansion
      info.type = convertTypeToPacked(typeConverter, field.type);
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
      // arrays including wildcard, and strings), return an LLVM pointer instead
      // of llhd.ref. These are dynamic types that don't fit the llhd
      // signal/probe model. Check the original Moore type to distinguish from
      // other pointer types.
      if (isa<QueueType, OpenUnpackedArrayType, AssocArrayType,
              WildcardAssocArrayType, StringType>(nestedType))
        return LLVM::LLVMPointerType::get(type.getContext());
      // If the inner type converted to an LLVM struct (e.g., unpacked struct
      // containing dynamic types like strings), also use LLVM pointer.
      if (isa<LLVM::LLVMStructType>(innerType))
        return LLVM::LLVMPointerType::get(type.getContext());
      // If the inner type converted to an LLVM array (e.g., fixed-size array
      // of strings), also use LLVM pointer.
      if (isa<LLVM::LLVMArrayType>(innerType))
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
        if (!resultType.isInteger(1))
          return Value();
        if (auto mooreInt = dyn_cast<moore::IntType>(inputs[0].getType())) {
          if (mooreInt.getBitSize() == 1)
            return moore::ToBuiltinBoolOp::create(builder, loc, inputs[0]);
          return Value();
        }
        if (!isFourStateStructType(inputs[0].getType()))
          return Value();
        auto structType = cast<hw::StructType>(inputs[0].getType());
        auto valueType = structType.getElements()[0].type;
        if (!valueType.isInteger(1))
          return Value();
        Value value = extractFourStateValue(builder, loc, inputs[0]);
        Value unknown = extractFourStateUnknown(builder, loc, inputs[0]);
        Value trueConst = hw::ConstantOp::create(builder, loc,
                                                builder.getI1Type(), 1);
        Value notUnknown =
            comb::XorOp::create(builder, loc, unknown, trueConst);
        return comb::AndOp::create(builder, loc,
                                   ValueRange{value, notUnknown}, true);
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
  patterns.add<ClassCopyOpConversion>(typeConverter, patterns.getContext(),
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
  patterns.add<CoverageBinDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CoverCrossDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CrossBinDeclOpConversion>(typeConverter, patterns.getContext());
  patterns.add<BinsOfOpConversion>(typeConverter, patterns.getContext());
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
  patterns.add<ConstraintDisableSoftOpConversion>(typeConverter,
                                                  patterns.getContext());
  patterns.add<ConstraintUniqueOpConversion>(typeConverter,
                                             patterns.getContext());

  // Patterns of vtable operations (with explicit benefits for ordering).
  // VTableOpConversion and VTableLoadMethodOpConversion need classCache for
  // method-to-vtable-index mapping and vtable global generation.
  patterns.add<VTableOpConversion>(typeConverter, patterns.getContext(),
                                   classCache);
  patterns.add<VTableEntryOpConversion>(typeConverter, patterns.getContext());
  patterns.add<VTableLoadMethodOpConversion>(typeConverter, patterns.getContext(),
                                             classCache);

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
    ConcatRefOpConversion,
    UnrealizedCastToBoolConversion,
    BitcastConversion<PackedToSBVOp>,
    BitcastConversion<SBVToPackedOp>,
    LogicToIntOpConversion,
    IntToLogicOpConversion,
    ToBuiltinBoolOpConversion,
    TruncOpConversion,
    ZExtOpConversion,
    SExtOpConversion,
    SIntToRealOpConversion,
    UIntToRealOpConversion,
    RealToIntOpConversion,
    ConvertRealOpConversion,

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
    FourStateNotOpConversion,  // 4-state aware NOT
    NegOpConversion,
    NegRealOpConversion,

    // Patterns of binary operations (4-state aware for arithmetic).
    FourStateArithOpConversion<AddOp, comb::AddOp>,
    FourStateArithOpConversion<SubOp, comb::SubOp>,
    FourStateArithOpConversion<MulOp, comb::MulOp>,
    FourStateArithOpConversion<DivUOp, comb::DivUOp>,
    FourStateArithOpConversion<DivSOp, comb::DivSOp>,
    FourStateArithOpConversion<ModUOp, comb::ModUOp>,
    FourStateArithOpConversion<ModSOp, comb::ModSOp>,
    FourStateAndOpConversion,  // 4-state aware AND
    FourStateOrOpConversion,   // 4-state aware OR
    FourStateXorOpConversion,  // 4-state aware XOR

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
    LogicalICmpOpConversion<UltOp, ICmpPredicate::ult>,
    LogicalICmpOpConversion<SltOp, ICmpPredicate::slt>,
    LogicalICmpOpConversion<UleOp, ICmpPredicate::ule>,
    LogicalICmpOpConversion<SleOp, ICmpPredicate::sle>,
    LogicalICmpOpConversion<UgtOp, ICmpPredicate::ugt>,
    LogicalICmpOpConversion<SgtOp, ICmpPredicate::sgt>,
    LogicalICmpOpConversion<UgeOp, ICmpPredicate::uge>,
    LogicalICmpOpConversion<SgeOp, ICmpPredicate::sge>,
    LogicalICmpOpConversion<EqOp, ICmpPredicate::eq>,
    LogicalICmpOpConversion<NeOp, ICmpPredicate::ne>,
    CaseEqOpConversion<CaseEqOp, ICmpPredicate::ceq>,
    CaseEqOpConversion<CaseNeOp, ICmpPredicate::cne>,
    WildcardEqOpConversion<WildcardEqOp, ICmpPredicate::weq>,
    WildcardEqOpConversion<WildcardNeOp, ICmpPredicate::wne>,
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
    ArithSelectOpConversion,
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

    // Patterns for system call builtins.
    StrobeBIOpConversion,
    FStrobeBIOpConversion,
    MonitorBIOpConversion,
    FMonitorBIOpConversion,
    MonitorOnBIOpConversion,
    MonitorOffBIOpConversion,
    PrintTimescaleBIOpConversion,
    FErrorBIOpConversion,
    UngetCBIOpConversion,
    FSeekBIOpConversion,
    RewindBIOpConversion,
    FReadBIOpConversion,
    ReadMemBBIOpConversion,
    ReadMemHBIOpConversion,

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
    QueueSortWithOpConversion,
    QueueRSortWithOpConversion,
    QueueSortKeyYieldOpConversion,
    ArrayLocatorYieldOpConversion,
    QueueShuffleOpConversion,
    QueueReverseOpConversion,
    QueueConcatOpConversion,
    QueueSliceOpConversion,
    QueueDeleteOpConversion,
    QueuePushBackOpConversion,
    QueuePushFrontOpConversion,
    QueueInsertOpConversion,
    QueuePopBackOpConversion,
    QueuePopFrontOpConversion,
    StreamConcatOpConversion,
    StreamUnpackOpConversion,
    StreamConcatMixedOpConversion,
    StreamUnpackMixedOpConversion,
    DynArrayNewOpConversion,
    ArraySizeOpConversion,
    ArrayContainsOpConversion,
    AssocArrayDeleteOpConversion,
    AssocArrayDeleteKeyOpConversion,
    AssocArrayCreateOpConversion,
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
    StringAtoRealOpConversion,
    StringHexToAOpConversion,
    StringOctToAOpConversion,
    StringBinToAOpConversion,
    StringRealToAOpConversion,
    StringConcatOpConversion,
    StringReplicateOpConversion,
    StringCmpOpConversion,
    StringCompareOpConversion,
    StringICompareOpConversion,
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
  patterns.add<WaitDelayOpConversion>(typeConverter, patterns.getContext());
  patterns.add<UnreachableOp>(convert);
  patterns.add<EventTriggeredOpConversion>(typeConverter, patterns.getContext());
  patterns.add<EventTriggerOpConversion>(typeConverter, patterns.getContext());
  patterns.add<WaitConditionOpConversion>(typeConverter, patterns.getContext());

  // UVM Configuration Database
  patterns.add<UVMConfigDbSetOpConversion>(typeConverter, patterns.getContext());
  patterns.add<UVMConfigDbGetOpConversion>(typeConverter, patterns.getContext());

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
  patterns.add<TimeToLogicOpConversion>(typeConverter, patterns.getContext());

  // Random number generation
  patterns.add<UrandomBIOpConversion>(typeConverter, patterns.getContext());
  patterns.add<UrandomRangeBIOpConversion>(typeConverter, patterns.getContext());
  patterns.add<RandomBIOpConversion>(typeConverter, patterns.getContext());

  // Randomization (needs class cache for struct info)
  patterns.add<RandomizeOpConversion>(typeConverter, patterns.getContext(),
                                      classCache);
  patterns.add<StdRandomizeOpConversion>(typeConverter, patterns.getContext());

  // Constraint mode and randomize callbacks
  patterns.add<ConstraintModeOpConversion>(typeConverter, patterns.getContext());
  patterns.add<RandModeOpConversion>(typeConverter, patterns.getContext());
  patterns.add<CallPreRandomizeOpConversion>(typeConverter, patterns.getContext(),
                                             classCache);
  patterns.add<CallPostRandomizeOpConversion>(typeConverter,
                                              patterns.getContext(), classCache);

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

  // Use a conversion config that allows pattern rollback, which preserves
  // value mappings longer and helps with handling nested regions that reference
  // converted block arguments.
  ConversionConfig config;
  config.allowPatternRollback = true;

  if (failed(applyFullConversion(module, target, std::move(patterns), config)))
    signalPassFailure();

  RewritePatternSet cleanupPatterns(&context);
  cleanupPatterns.add<UnrealizedCastToBoolRewrite>(&context);
  if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns))))
    signalPassFailure();

  // Fix sim.fork entry blocks that have predecessors (e.g., from forever
  // loops). During dialect conversion, inlineRegionBefore doesn't always
  // preserve predecessor relationships correctly, so we do this fixup after
  // the conversion is fully complete and block predecessors are finalized.
  OpBuilder builder(&context);
  module.walk([&](sim::SimForkOp forkOp) {
    for (Region &region : forkOp.getBranches()) {
      if (region.empty())
        continue;
      Block *entryBlock = &region.front();
      if (entryBlock->hasNoPredecessors())
        continue;

      // Create a new loop header block after the entry block.
      auto *loopHeader = new Block();
      region.getBlocks().insertAfter(entryBlock->getIterator(), loopHeader);

      // Move all operations from entry to the loop header.
      loopHeader->getOperations().splice(loopHeader->end(),
                                         entryBlock->getOperations());

      // Update all branches in the region that target entryBlock to target
      // loopHeader instead.
      for (Block &block : region) {
        Operation *terminator = block.getTerminator();
        if (!terminator)
          continue;
        if (auto brOp = dyn_cast<cf::BranchOp>(terminator)) {
          if (brOp.getDest() == entryBlock) {
            builder.setInsertionPoint(brOp);
            cf::BranchOp::create(builder, brOp.getLoc(), loopHeader);
            brOp.erase();
          }
        } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(terminator)) {
          bool updateTrue = condBrOp.getTrueDest() == entryBlock;
          bool updateFalse = condBrOp.getFalseDest() == entryBlock;
          if (updateTrue || updateFalse) {
            builder.setInsertionPoint(condBrOp);
            cf::CondBranchOp::create(
                builder, condBrOp.getLoc(), condBrOp.getCondition(),
                updateTrue ? loopHeader : condBrOp.getTrueDest(),
                condBrOp.getTrueDestOperands(),
                updateFalse ? loopHeader : condBrOp.getFalseDest(),
                condBrOp.getFalseDestOperands());
            condBrOp.erase();
          }
        }
      }

      // Add branch from entry to loop header plus a side-effect op to
      // prevent the MLIR printer from eliding the entry block.
      builder.setInsertionPointToStart(entryBlock);
      auto spaceFmt = sim::FormatLiteralOp::create(builder, forkOp.getLoc(),
                                                   " ");
      sim::PrintFormattedProcOp::create(builder, forkOp.getLoc(), spaceFmt);
      cf::BranchOp::create(builder, forkOp.getLoc(), loopHeader);
    }
  });
}

//===----------------------------------------------------------------------===//
// Init Vtables Pass
//===----------------------------------------------------------------------===//

namespace {
struct InitVtablesPass : public circt::impl::InitVtablesBase<InitVtablesPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a pass to initialize vtable globals with function pointers.
std::unique_ptr<OperationPass<ModuleOp>> circt::createInitVtablesPass() {
  return std::make_unique<InitVtablesPass>();
}

/// Initialize vtable globals with function pointer addresses.
void InitVtablesPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  // Find all LLVM globals with vtable metadata.
  SmallVector<LLVM::GlobalOp> vtableGlobals;
  module.walk([&](LLVM::GlobalOp global) {
    if (global->hasAttr("circt.vtable_entries"))
      vtableGlobals.push_back(global);
  });

  for (auto global : vtableGlobals) {
    auto entriesAttr = dyn_cast_or_null<ArrayAttr>(
        global->getAttr("circt.vtable_entries"));
    if (!entriesAttr)
      continue;

    // Get the vtable array type.
    auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(global.getType());
    if (!arrayTy)
      continue;

    unsigned vtableSize = arrayTy.getNumElements();
    if (vtableSize == 0)
      continue;

    // Build a map from index to function symbol.
    DenseMap<unsigned, SymbolRefAttr> indexToFunc;
    for (auto entry : entriesAttr) {
      auto entryArray = dyn_cast<ArrayAttr>(entry);
      if (!entryArray || entryArray.size() != 2)
        continue;

      auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
      auto funcSym = dyn_cast<SymbolRefAttr>(entryArray[1]);
      if (!indexAttr || !funcSym)
        continue;

      unsigned index = indexAttr.getInt();
      indexToFunc[index] = funcSym;
    }

    // Check if all referenced functions exist as llvm.func.
    bool allFuncsExist = true;
    for (auto &[index, funcSym] : indexToFunc) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcSym);
      if (!func) {
        // Also check for func.func (not yet converted).
        auto funcFunc = module.lookupSymbol<func::FuncOp>(funcSym);
        if (!funcFunc) {
          allFuncsExist = false;
          break;
        }
        // func.func exists but not llvm.func - skip this vtable for now.
        // It will be populated when InitVtablesPass runs after func-to-llvm.
        allFuncsExist = false;
        break;
      }
    }

    if (!allFuncsExist)
      continue;

    // All functions exist as llvm.func. Create the initializer region.
    // Clear existing initializer (zero attr) and add initializer region.
    global.removeValueAttr();
    global->removeAttr("value");

    // Create initializer region.
    Region &initRegion = global.getInitializerRegion();
    Block *block = new Block();
    initRegion.push_back(block);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block);

    // Build the vtable array value.
    // Start with an undef array, then insert function pointers at each index.
    Value vtableArray =
        LLVM::UndefOp::create(builder, global.getLoc(), arrayTy);

    for (unsigned i = 0; i < vtableSize; ++i) {
      Value funcPtr;
      auto it = indexToFunc.find(i);
      if (it != indexToFunc.end()) {
        // Get address of the function. Convert SymbolRefAttr to StringRef.
        StringRef funcName = it->second.getRootReference();
        funcPtr = LLVM::AddressOfOp::create(builder, global.getLoc(), ptrTy,
                                            funcName);
      } else {
        // Null pointer for unused slots.
        funcPtr = LLVM::ZeroOp::create(builder, global.getLoc(), ptrTy);
      }

      vtableArray = LLVM::InsertValueOp::create(builder, global.getLoc(),
                                                 vtableArray, funcPtr, i);
    }

    LLVM::ReturnOp::create(builder, global.getLoc(), vtableArray);

    // Remove the vtable metadata attribute (no longer needed).
    global->removeAttr("circt.vtable_entries");
  }
}
