//===- Types.cpp - Slang type conversion ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace ImportVerilog;
using moore::Domain;

namespace {
struct TypeVisitor {
  Context &context;
  Location loc;
  TypeVisitor(Context &context, Location loc) : context(context), loc(loc) {}

  // Handle simple bit vector types such as `bit`, `int`, or `bit [41:0]`.
  Type getSimpleBitVectorType(const slang::ast::IntegralType &type) {
    return moore::IntType::get(context.getContext(), type.bitWidth,
                               type.isFourState ? Domain::FourValued
                                                : Domain::TwoValued);
  }

  // NOLINTBEGIN(misc-no-recursion)
  Type visit(const slang::ast::VoidType &type) {
    return moore::VoidType::get(context.getContext());
  }

  Type visit(const slang::ast::ScalarType &type) {
    return getSimpleBitVectorType(type);
  }

  Type visit(const slang::ast::FloatingType &type) {
    if (type.floatKind == slang::ast::FloatingType::Kind::RealTime)
      return moore::TimeType::get(context.getContext());
    if (type.floatKind == slang::ast::FloatingType::Kind::Real)
      return moore::RealType::get(context.getContext(), moore::RealWidth::f64);
    return moore::RealType::get(context.getContext(), moore::RealWidth::f32);
  }

  Type visit(const slang::ast::PredefinedIntegerType &type) {
    if (type.integerKind == slang::ast::PredefinedIntegerType::Kind::Time)
      return moore::TimeType::get(context.getContext());
    return getSimpleBitVectorType(type);
  }

  Type visit(const slang::ast::PackedArrayType &type) {
    // Handle simple bit vector types of the form `bit [41:0]`.
    if (type.elementType.as_if<slang::ast::ScalarType>())
      return getSimpleBitVectorType(type);

    // Handle all other packed arrays.
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    // The Slang frontend guarantees the inner type to be packed.
    return moore::ArrayType::get(type.range.width(),
                                 cast<moore::PackedType>(innerType));
  }

  Type visit(const slang::ast::QueueType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::QueueType::get(cast<moore::UnpackedType>(innerType),
                                 type.maxBound);
  }

  Type visit(const slang::ast::AssociativeArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    // Handle wildcard associative arrays [*]
    if (!type.indexType) {
      return moore::WildcardAssocArrayType::get(
          cast<moore::UnpackedType>(innerType));
    }
    auto indexType = type.indexType->visit(*this);
    if (!indexType)
      return {};
    return moore::AssocArrayType::get(cast<moore::UnpackedType>(innerType),
                                      cast<moore::UnpackedType>(indexType));
  }

  Type visit(const slang::ast::FixedSizeUnpackedArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::UnpackedArrayType::get(type.range.width(),
                                         cast<moore::UnpackedType>(innerType));
  }

  Type visit(const slang::ast::DynamicArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::OpenUnpackedArrayType::get(
        cast<moore::UnpackedType>(innerType));
  }

  // Handle type defs.
  Type visit(const slang::ast::TypeAliasType &type) {
    // Use getCanonicalType() to fully resolve the underlying type.
    // This is important for typedefs to parameterized class specializations,
    // where we need to get the specialized class type (e.g., registry_0)
    // rather than the generic class type (e.g., registry).
    auto &targetType = type.targetType.getType();
    auto &canonicalType = type.getCanonicalType();
    if (auto *classType = targetType.as_if<slang::ast::ClassType>()) {
      LLVM_DEBUG(llvm::dbgs() << "TypeAliasType: " << type.name
                              << " -> targetType: " << targetType.name
                              << " (ClassType, genericClass: "
                              << (classType->genericClass ? classType->genericClass->name : "none")
                              << ")"
                              << " -> canonical: " << canonicalType.name
                              << " (kind: " << slang::ast::toString(canonicalType.kind) << ")\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "TypeAliasType: " << type.name
                              << " -> targetType: " << targetType.name
                              << " (kind: " << slang::ast::toString(targetType.kind) << ")"
                              << " -> canonical: " << canonicalType.name
                              << " (kind: " << slang::ast::toString(canonicalType.kind) << ")\n");
    }
    return canonicalType.visit(*this);
  }

  // Handle enums.
  Type visit(const slang::ast::EnumType &type) {
    // Simply return the underlying type.
    return type.baseType.visit(*this);
  }

  // Collect the members in a struct or union.
  LogicalResult
  collectMembers(const slang::ast::Scope &structType,
                 SmallVectorImpl<moore::StructLikeMember> &members) {
    for (auto &field : structType.membersOfType<slang::ast::FieldSymbol>()) {
      auto name = StringAttr::get(context.getContext(), field.name);
      auto innerType = context.convertType(*field.getDeclaredType());
      if (!innerType)
        return failure();
      members.push_back({name, cast<moore::UnpackedType>(innerType)});
    }
    return success();
  }

  // Handle packed and unpacked structs.
  Type visit(const slang::ast::PackedStructType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    return moore::StructType::get(context.getContext(), members);
  }

  Type visit(const slang::ast::UnpackedStructType &type) {
    LLVM_DEBUG(llvm::dbgs() << "Visiting UnpackedStructType: " << type.name << "\n");
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members))) {
      LLVM_DEBUG(llvm::dbgs() << "  collectMembers failed for UnpackedStructType\n");
      return {};
    }
    LLVM_DEBUG(llvm::dbgs() << "  UnpackedStructType has " << members.size() << " members\n");
    return moore::UnpackedStructType::get(context.getContext(), members);
  }

  // Helper to wrap a union type in a struct with tag and data fields.
  // Tagged unions are represented as struct<{tag: iN, data: union<...>}>
  // where N is the number of bits needed to encode the member index.
  // Uses packed struct for packed unions, unpacked struct for unpacked unions.
  Type wrapTaggedUnion(Type unionType, size_t memberCount, bool isPacked) {
    // Calculate tag width: need ceil(log2(memberCount)) bits, minimum 1
    unsigned tagWidth = 1;
    if (memberCount > 1)
      tagWidth = llvm::Log2_64_Ceil(memberCount);

    auto tagType = moore::IntType::get(context.getContext(), tagWidth,
                                       moore::Domain::TwoValued);
    SmallVector<moore::StructLikeMember> wrapperMembers;
    wrapperMembers.push_back(
        {StringAttr::get(context.getContext(), "tag"), tagType});
    wrapperMembers.push_back(
        {StringAttr::get(context.getContext(), "data"),
         cast<moore::UnpackedType>(unionType)});

    if (isPacked)
      return moore::StructType::get(context.getContext(), wrapperMembers);
    return moore::UnpackedStructType::get(context.getContext(), wrapperMembers);
  }

  Type visit(const slang::ast::PackedUnionType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    auto unionType = moore::UnionType::get(context.getContext(), members);
    // If this is a tagged union, wrap it in a struct with tag and data fields
    if (type.isTagged)
      return wrapTaggedUnion(unionType, members.size(), /*isPacked=*/true);
    return unionType;
  }

  Type visit(const slang::ast::UnpackedUnionType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    auto unionType = moore::UnpackedUnionType::get(context.getContext(), members);
    // If this is a tagged union, wrap it in a struct with tag and data fields
    if (type.isTagged)
      return wrapTaggedUnion(unionType, members.size(), /*isPacked=*/false);
    return unionType;
  }

  Type visit(const slang::ast::StringType &type) {
    return moore::StringType::get(context.getContext());
  }

  Type visit(const slang::ast::CHandleType &type) {
    return moore::ChandleType::get(context.getContext());
  }

  Type visit(const slang::ast::ClassType &type) {
    // Convert the class declaration and populate its body.
    // convertClassDeclaration handles recursive calls by checking if the
    // body is already being converted (via ClassDeclVisitor::run's guard).
    // Note: Body conversion failures are tolerated - we still return a valid
    // type so that forward-declared class types can be used in variable
    // declarations before all class members are convertible.
    LLVM_DEBUG({
      llvm::dbgs() << "ClassType: " << type.name << " ptr=" << &type;
      if (type.genericClass)
        llvm::dbgs() << " genericClass@" << type.genericClass;
      llvm::dbgs() << "\n";
    });
    // Call convertClassDeclaration but ignore failures - they may happen
    // due to forward references that will be resolved later.
    (void)context.convertClassDeclaration(type);
    auto *lowering = context.declareClass(type);
    if (!lowering || !lowering->op) {
      mlir::emitError(loc) << "no lowering generated for class type `"
                           << type.toString() << "`";
      return {};
    }
    mlir::StringAttr symName = lowering->op.getSymNameAttr();
    LLVM_DEBUG(llvm::dbgs() << "  -> lowering symName: " << symName << "\n");
    mlir::FlatSymbolRefAttr symRef = mlir::FlatSymbolRefAttr::get(symName);
    return moore::ClassHandleType::get(context.getContext(), symRef);
  }

  Type visit(const slang::ast::EventType &type) {
    return moore::EventType::get(context.getContext());
  }

  Type visit(const slang::ast::CovergroupType &type) {
    // Convert the covergroup and return a handle type.
    // The covergroup should already be declared during module body conversion.
    if (failed(context.convertCovergroup(type)))
      return {};

    auto it = context.covergroups.find(&type);
    if (it == context.covergroups.end() || !it->second || !it->second->op) {
      mlir::emitError(loc) << "no lowering generated for covergroup type `"
                           << type.name << "`";
      return {};
    }
    mlir::StringAttr symName = it->second->op.getSymNameAttr();
    mlir::FlatSymbolRefAttr symRef = mlir::FlatSymbolRefAttr::get(symName);
    return moore::CovergroupHandleType::get(context.getContext(), symRef);
  }

  Type visit(const slang::ast::VirtualInterfaceType &type) {
    // Get the interface from the virtual interface type
    auto &iface = type.iface;

    // Ensure the interface is declared (header converted). This is necessary
    // because virtual interface types reference an interface that might not
    // have been instantiated anywhere in the design.
    auto *ifaceLowering = context.convertInterfaceHeader(&iface.body);
    if (!ifaceLowering)
      return {};

    // Use the interface's symbol name from the lowering
    auto ifaceName = ifaceLowering->op.getSymName();

    // Build the symbol reference
    mlir::SymbolRefAttr symRef;
    if (type.modport) {
      // Virtual interface with modport: @interface::@modport
      symRef = mlir::SymbolRefAttr::get(
          context.getContext(),
          ifaceName,
          {mlir::FlatSymbolRefAttr::get(context.getContext(),
                                        type.modport->name)});
    } else {
      // Virtual interface without modport: @interface
      symRef = mlir::FlatSymbolRefAttr::get(context.getContext(), ifaceName);
    }

    return moore::VirtualInterfaceType::get(context.getContext(), symRef);
  }

  Type visit(const slang::ast::NullType &type) {
    // The null type represents the type of the `null` literal.
    // We represent it as a class handle with a special "__null__" symbol.
    // This allows null to be assigned to any class handle.
    auto nullSym = mlir::FlatSymbolRefAttr::get(context.getContext(), "__null__");
    return moore::ClassHandleType::get(context.getContext(), nullSym);
  }

  Type visit(const slang::ast::UntypedType &type) {
    // UntypedType is used for 'interconnect' nets. According to IEEE 1800-2017
    // Section 6.6.8, interconnect nets are used for connecting signals with
    // potentially different types. We lower this to a 1-bit 4-state logic type.
    return moore::IntType::get(context.getContext(), 1, Domain::FourValued);
  }

  /// Emit an error for all other types.
  template <typename T>
  Type visit(T &&node) {
    auto d = mlir::emitError(loc, "unsupported type: ")
             << slang::ast::toString(node.kind);
    d.attachNote() << node.template as<slang::ast::Type>().toString();
    return {};
  }
  // NOLINTEND(misc-no-recursion)
};
} // namespace

// NOLINTBEGIN(misc-no-recursion)
Type Context::convertType(const slang::ast::Type &type, LocationAttr loc) {
  if (!loc)
    loc = convertLocation(type.location);
  return type.visit(TypeVisitor(*this, loc));
}

Type Context::convertType(const slang::ast::DeclaredType &type) {
  LocationAttr loc;
  if (auto *ts = type.getTypeSyntax())
    loc = convertLocation(ts->sourceRange().start());
  return convertType(type.getType(), loc);
}
// NOLINTEND(misc-no-recursion)

// Synthesize a struct initial value from field defaults per IEEE 1800-2017
// §7.2.1. When a struct type has fields with default values (e.g.,
// `typedef struct { int a = 42; int b; } my_struct;`), this function creates
// a moore.struct_create op with the field defaults for fields that have them,
// and zero for fields that don't. Returns null if the type is not a struct
// or has no field defaults.
// NOLINTBEGIN(misc-no-recursion)
Value Context::synthesizeStructFieldDefaults(
    const slang::ast::Type &slangType, Type mooreType, Location loc) {
  const auto &canonical = slangType.getCanonicalType();

  // Determine the slang struct scope and Moore struct member list.
  const slang::ast::Scope *structScope = nullptr;
  ArrayRef<moore::StructLikeMember> members;
  bool isPacked = false;

  if (auto *packed = canonical.as_if<slang::ast::PackedStructType>()) {
    structScope = packed;
    if (auto st = dyn_cast<moore::StructType>(mooreType)) {
      members = st.getMembers();
      isPacked = true;
    } else {
      return {};
    }
  } else if (auto *unpacked =
                 canonical.as_if<slang::ast::UnpackedStructType>()) {
    structScope = unpacked;
    if (auto st = dyn_cast<moore::UnpackedStructType>(mooreType)) {
      members = st.getMembers();
      isPacked = false;
    } else {
      return {};
    }
  } else {
    return {}; // Not a struct type.
  }

  // Check if any field has a default initializer.
  bool hasAnyDefault = false;
  for (auto &field :
       structScope->membersOfType<slang::ast::FieldSymbol>()) {
    if (field.getInitializer()) {
      hasAnyDefault = true;
      break;
    }
  }
  if (!hasAnyDefault)
    return {}; // No field defaults; let normal zero-init happen.

  // Build field values: use the field's default initializer if present,
  // otherwise create a zero value of the appropriate type.
  SmallVector<Value> fieldValues;
  unsigned idx = 0;
  for (auto &field :
       structScope->membersOfType<slang::ast::FieldSymbol>()) {
    if (idx >= members.size())
      break;
    auto fieldMooreType = members[idx].type;
    Value fieldVal;

    if (const auto *init = field.getInitializer()) {
      // Convert the field's default initializer expression.
      fieldVal = convertRvalueExpression(*init, fieldMooreType);
      if (!fieldVal)
        return {};
    } else {
      // No default for this field - create a zero value.
      if (auto intType = dyn_cast<moore::IntType>(fieldMooreType)) {
        fieldVal = moore::ConstantOp::create(builder, loc, intType, 0);
      } else {
        // For non-integer fields without defaults, try to recursively
        // synthesize defaults or create a zero via conversion from 0.
        fieldVal = synthesizeStructFieldDefaults(
            field.getDeclaredType()->getType(), fieldMooreType, loc);
        if (!fieldVal) {
          // Last resort: create a 1-bit zero and convert to the field type.
          auto i1 = moore::IntType::getInt(getContext(), 1);
          auto zero = moore::ConstantOp::create(builder, loc, i1, 0);
          fieldVal = moore::ConversionOp::create(builder, loc, fieldMooreType,
                                                 zero);
        }
      }
    }
    fieldValues.push_back(fieldVal);
    ++idx;
  }

  // Build the struct_create op.
  if (isPacked) {
    return moore::StructCreateOp::create(
        builder, loc, cast<moore::StructType>(mooreType), fieldValues);
  }
  return moore::StructCreateOp::create(
      builder, loc, cast<moore::UnpackedStructType>(mooreType), fieldValues);
}
// NOLINTEND(misc-no-recursion)
