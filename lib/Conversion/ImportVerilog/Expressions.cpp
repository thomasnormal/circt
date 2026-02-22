//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Constraints.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/symbols/CoverSymbols.h"
#include "slang/ast/expressions/CallExpression.h"
#include "slang/ast/expressions/LiteralExpressions.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/syntax/AllSyntax.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/ScopeExit.h"

using namespace circt;
using namespace ImportVerilog;
using moore::Domain;

static mlir::LLVM::LLVMFuncOp
getOrCreateRuntimeFunc(Context &context, StringRef name,
                       mlir::LLVM::LLVMFunctionType funcType) {
  auto module = context.intoModuleOp;
  if (auto existing = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(context.builder);
  context.builder.setInsertionPointToStart(module.getBody());
  auto func = mlir::LLVM::LLVMFuncOp::create(context.builder, module.getLoc(),
                                             name, funcType);
  func.setLinkage(mlir::LLVM::Linkage::External);
  return func;
}

/// Convert a Slang `SVInt` to a CIRCT `FVInt`.
static FVInt convertSVIntToFVInt(const slang::SVInt &svint) {
  if (svint.hasUnknown()) {
    unsigned numWords = svint.getNumWords() / 2;
    auto value = ArrayRef<uint64_t>(svint.getRawPtr(), numWords);
    auto unknown = ArrayRef<uint64_t>(svint.getRawPtr() + numWords, numWords);
    return FVInt(APInt(svint.getBitWidth(), value),
                 APInt(svint.getBitWidth(), unknown));
  }
  auto value = ArrayRef<uint64_t>(svint.getRawPtr(), svint.getNumWords());
  return FVInt(APInt(svint.getBitWidth(), value));
}

/// Map an index into an array, with bounds `range`, to a bit offset of the
/// underlying bit storage. This is a dynamic version of
/// `slang::ConstantRange::translateIndex`.
static Value getSelectIndex(Context &context, Location loc, Value index,
                            const slang::ConstantRange &range) {
  auto &builder = context.builder;
  auto indexType = cast<moore::UnpackedType>(index.getType());

  // Compute offset first so we know if it is negative.
  auto lo = range.lower();
  auto hi = range.upper();
  auto offset = range.isLittleEndian() ? lo : hi;

  // If any bound is negative we need a signed index type.
  const bool needSigned = (lo < 0) || (hi < 0);

  // Magnitude over full range, not just the chosen offset.
  const uint64_t maxAbs = std::max<uint64_t>(std::abs(lo), std::abs(hi));

  // Bits needed from the range:
  //  - unsigned: ceil(log2(maxAbs + 1)) (ensure at least 1)
  //  - signed:   ceil(log2(maxAbs)) + 1 sign bit (ensure at least 2 when neg)
  unsigned want = needSigned
                      ? (llvm::Log2_64_Ceil(std::max<uint64_t>(1, maxAbs)) + 1)
                      : std::max<unsigned>(1, llvm::Log2_64_Ceil(maxAbs + 1));

  // Keep at least as wide as the incoming index.
  const unsigned bw = std::max<unsigned>(want, indexType.getBitSize().value());

  auto intType =
      moore::IntType::get(index.getContext(), bw, indexType.getDomain());
  index = context.materializeConversion(intType, index, needSigned, loc);

  if (offset == 0) {
    if (range.isLittleEndian())
      return index;
    else
      return moore::NegOp::create(builder, loc, index);
  }

  auto offsetConst =
      moore::ConstantOp::create(builder, loc, intType, offset, needSigned);
  if (range.isLittleEndian())
    return moore::SubOp::create(builder, loc, index, offsetConst);
  else
    return moore::SubOp::create(builder, loc, offsetConst, index);
}

/// Get the currently active timescale as an integer number of femtoseconds.
static uint64_t getTimeScaleInFemtoseconds(Context &context) {
  static_assert(int(slang::TimeUnit::Seconds) == 0);
  static_assert(int(slang::TimeUnit::Milliseconds) == 1);
  static_assert(int(slang::TimeUnit::Microseconds) == 2);
  static_assert(int(slang::TimeUnit::Nanoseconds) == 3);
  static_assert(int(slang::TimeUnit::Picoseconds) == 4);
  static_assert(int(slang::TimeUnit::Femtoseconds) == 5);

  static_assert(int(slang::TimeScaleMagnitude::One) == 1);
  static_assert(int(slang::TimeScaleMagnitude::Ten) == 10);
  static_assert(int(slang::TimeScaleMagnitude::Hundred) == 100);

  auto exp = static_cast<unsigned>(context.timeScale.base.unit);
  assert(exp <= 5);
  exp = 5 - exp;
  auto scale = static_cast<uint64_t>(context.timeScale.base.magnitude);
  while (exp-- > 0)
    scale *= 1000;
  return scale;
}

/// Check if a class handle type represents the null type.
/// Null types have the special "__null__" symbol.
static bool isNullHandleType(moore::ClassHandleType handleTy) {
  return handleTy.getClassSym() &&
         handleTy.getClassSym().getRootReference() == "__null__";
}

/// Get the value type from an lvalue type. For RefType, this returns the nested
/// type. For ClassHandleType (class handles), this returns the type itself
/// since class handles are assigned directly without dereferencing. Returns
/// null if the type is neither.
static Type getLvalueNestedType(Type lvalueType) {
  if (auto refType = dyn_cast<moore::RefType>(lvalueType))
    return refType.getNestedType();
  if (isa<moore::ClassHandleType>(lvalueType))
    return lvalueType;
  return {};
}

/// Build logical equality for unpacked aggregate values, with recursive support
/// for unpacked structs.
static Value buildUnpackedAggregateLogicalEq(Context &context, Location loc,
                                             Value lhs, Value rhs) {
  auto &builder = context.builder;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return {};

  auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);

  if (isa<moore::UnpackedArrayType>(lhs.getType()))
    return moore::UArrayCmpOp::create(builder, loc, moore::UArrayCmpPredicate::eq,
                                      lhs, rhs);

  if (auto structTy = dyn_cast<moore::UnpackedStructType>(lhs.getType())) {
    Value allEqual = moore::ConstantOp::create(builder, loc, i1Ty, 1);
    for (auto member : structTy.getMembers()) {
      Value lhsField = moore::StructExtractOp::create(builder, loc, member.type,
                                                      member.name, lhs);
      Value rhsField = moore::StructExtractOp::create(builder, loc, member.type,
                                                      member.name, rhs);
      Value fieldEq;
      if (isa<moore::UnpackedStructType, moore::UnpackedArrayType>(member.type)) {
        fieldEq = buildUnpackedAggregateLogicalEq(context, loc, lhsField, rhsField);
      } else if (isa<moore::StringType>(member.type) ||
                 isa<moore::FormatStringType>(member.type)) {
        auto strTy = moore::StringType::get(context.getContext());
        lhsField =
            context.materializeConversion(strTy, lhsField, false, lhsField.getLoc());
        rhsField =
            context.materializeConversion(strTy, rhsField, false, rhsField.getLoc());
        if (!lhsField || !rhsField)
          return {};
        fieldEq = moore::StringCmpOp::create(builder, loc,
                                             moore::StringCmpPredicate::eq,
                                             lhsField, rhsField);
      } else if (isa<moore::ChandleType>(member.type)) {
        auto intTy =
            moore::IntType::get(context.getContext(), 64, Domain::TwoValued);
        lhsField = context.materializeConversion(intTy, lhsField, false,
                                                 lhsField.getLoc());
        rhsField = context.materializeConversion(intTy, rhsField, false,
                                                 rhsField.getLoc());
        if (!lhsField || !rhsField)
          return {};
        fieldEq = moore::EqOp::create(builder, loc, lhsField, rhsField);
      } else {
        if (!isa<moore::IntType>(lhsField.getType()))
          lhsField = context.convertToSimpleBitVector(lhsField);
        if (!isa<moore::IntType>(rhsField.getType()))
          rhsField = context.convertToSimpleBitVector(rhsField);
        if (!lhsField || !rhsField)
          return {};
        if (lhsField.getType() != rhsField.getType()) {
          rhsField = context.materializeConversion(lhsField.getType(), rhsField,
                                                   /*isSigned=*/false, loc);
          if (!rhsField)
            return {};
        }
        fieldEq = moore::EqOp::create(builder, loc, lhsField, rhsField);
      }
      if (!fieldEq)
        return {};
      if (fieldEq.getType() != i1Ty)
        fieldEq = context.materializeConversion(i1Ty, fieldEq, false, loc);
      if (!fieldEq)
        return {};
      allEqual = moore::AndOp::create(builder, loc, allEqual, fieldEq);
    }
    return allEqual;
  }

  return {};
}

static Value visitClassProperty(Context &context,
                                const slang::ast::ClassPropertySymbol &expr) {
  auto loc = context.convertLocation(expr.location);
  auto builder = context.builder;

  auto type = context.convertType(expr.getType());
  if (!type)
    return {};

  // Check if this is a static property based on slang's lifetime attribute.
  // Do NOT infer static based on missing 'this' reference - that would
  // incorrectly treat non-static properties in constraint blocks as static.
  bool isStatic = expr.lifetime == slang::ast::VariableLifetime::Static;

  // Get the scope's implicit this variable and any inline constraint override.
  mlir::Value instRef = context.getImplicitThisRef();
  mlir::Value constraintThisRef = context.getInlineConstraintThisRef();

  if (isStatic) {
    // Static properties are stored as global variables.
    // Look up the global variable that was created for this static property.
    if (auto globalOp = context.globalVariables.lookup(&expr)) {
      // Use the expression's type (already converted above) rather than the
      // GlobalVariableOp's type. During recursive type conversion, the
      // GlobalVariableOp may temporarily have a placeholder type.
      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
      auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      return moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
    }

    // If the global variable hasn't been created yet (e.g., forward reference
    // or recursive class conversion), try on-demand conversion of just this
    // property. This handles cases where a method body references a static
    // property before the property has been processed during class conversion.
    if (succeeded(context.convertStaticClassProperty(expr))) {
      if (auto globalOp = context.globalVariables.lookup(&expr)) {
        // Use the expression's type (already converted above) rather than the
        // GlobalVariableOp's type.
        auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
        auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
        return moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
      }
    }

    // If we still can't find it, emit a warning and create a temporary variable
    // as a fallback.
    mlir::emitWarning(loc) << "static class property '" << expr.name
                           << "' could not be resolved to a global variable; "
                           << "treating as uninitialized variable";
    auto fieldTy = cast<moore::UnpackedType>(type);
    auto refTy = moore::RefType::get(fieldTy);
    auto nameAttr = builder.getStringAttr(expr.name);
    auto varOp = moore::VariableOp::create(builder, loc, refTy, nameAttr,
                                           /*initial=*/Value{});
    return varOp;
  }

  auto fieldSym = mlir::FlatSymbolRefAttr::get(builder.getContext(), expr.name);
  auto fieldTy = cast<moore::UnpackedType>(type);
  auto fieldRefTy = moore::RefType::get(fieldTy);

  // Get the class that declares this property from slang's AST.
  // This avoids the need to look up the property in Moore IR, which may not
  // have been fully populated yet due to circular dependencies between classes.
  const auto *parentScope = expr.getParentScope();
  if (!parentScope) {
    mlir::emitError(loc) << "class property '" << expr.name
                         << "' has no parent scope";
    return {};
  }
  const auto *declaringClass = parentScope->asSymbol().as_if<slang::ast::ClassType>();
  if (!declaringClass) {
    mlir::emitError(loc) << "class property '" << expr.name
                         << "' is not declared in a class";
    return {};
  }

  // Get the actual ClassDeclOp for the declaring class to get its
  // correct symbol name. For parameterized classes, each specialization
  // has its own ClassDeclOp with a potentially renamed symbol.
  // We must call convertClassDeclaration to ensure the class body is
  // populated (not just declared).
  (void)context.convertClassDeclaration(*declaringClass);
  auto *declaringLowering = context.declareClass(*declaringClass);
  if (!declaringLowering || !declaringLowering->op) {
    mlir::emitError(loc) << "failed to get ClassDeclOp for declaring class";
    return {};
  }
  auto declaringClassSym = declaringLowering->op.getSymNameAttr();
  auto targetClassHandle = moore::ClassHandleType::get(
      context.getContext(), mlir::FlatSymbolRefAttr::get(declaringClassSym));

  Value thisRef = instRef;
  if (constraintThisRef) {
    auto constraintTy =
        dyn_cast<moore::ClassHandleType>(constraintThisRef.getType());
    if (constraintTy &&
        context.isClassDerivedFrom(constraintTy, targetClassHandle)) {
      thisRef = constraintThisRef;
    }
  }

  if (!thisRef) {
    // No implicit 'this' reference available. This happens in constraint blocks
    // during class body conversion where properties are referenced symbolically.
    // Create a placeholder variable with the property name that will be resolved
    // by the constraint solver at runtime.
    auto fieldTy = cast<moore::UnpackedType>(type);
    auto refTy = moore::RefType::get(fieldTy);
    auto nameAttr = builder.getStringAttr(expr.name);
    auto varOp = moore::VariableOp::create(builder, loc, refTy, nameAttr,
                                           /*initial=*/Value{});
    return varOp;
  }

  moore::ClassHandleType classTy =
      cast<moore::ClassHandleType>(thisRef.getType());

  // If target class is same as current class, no conversion needed
  Value upcastRef;
  if (targetClassHandle.getClassSym() == classTy.getClassSym()) {
    upcastRef = thisRef;
  } else {
    LLVM_DEBUG({
      llvm::dbgs() << "visitClassProperty: property '" << expr.name
                   << "' declared in " << declaringClassSym
                   << ", current this type = " << classTy
                   << ", need upcast to " << targetClassHandle << "\n";
    });
    upcastRef = context.materializeConversion(targetClassHandle, thisRef, false,
                                              thisRef.getLoc());
  }
  if (!upcastRef)
    return {};

  Value fieldRef = moore::ClassPropertyRefOp::create(builder, loc, fieldRefTy,
                                                     upcastRef, fieldSym);
  return fieldRef;
}

namespace {
/// A visitor handling expressions that can be lowered as lvalue and rvalue.
struct ExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;
  bool isLvalue;

  ExprVisitor(Context &context, Location loc, bool isLvalue)
      : context(context), loc(loc), builder(context.builder),
        isLvalue(isLvalue) {}

  /// Resolve a hierarchical interface member access like `p.child.awvalid` and
  /// return either a reference (lvalue) or value (rvalue) for the signal.
  /// This supports nested interface instances inside interfaces by walking
  /// the hierarchical reference path and chaining VirtualInterfaceSignalRefOp.
  Value resolveHierarchicalInterfaceSignalRef(
      const slang::ast::HierarchicalValueExpression &expr,
      StringRef signalName) {
    Value instRef;
    size_t baseIndex = 0;

    for (size_t i = 0; i < expr.ref.path.size(); ++i) {
      const auto &elem = expr.ref.path[i];
      if (auto *instSym = elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
        if (instSym->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface) {
          if (auto it = context.interfaceInstances.find(instSym);
              it != context.interfaceInstances.end())
            instRef = it->second;
          else
            instRef = context.resolveInterfaceInstance(instSym, loc);
          if (instRef) {
            baseIndex = i;
            break;
          }
        }
      } else if (auto *ifacePort =
                     elem.symbol->as_if<slang::ast::InterfacePortSymbol>()) {
        if (auto it = context.interfacePortValues.find(ifacePort);
            it != context.interfacePortValues.end()) {
          instRef = it->second;
          baseIndex = i;
          break;
        }
      }
    }

    if (!instRef)
      return {};

    auto instRefTy = dyn_cast<moore::RefType>(instRef.getType());
    if (!instRefTy)
      return {};
    auto vifTy =
        dyn_cast<moore::VirtualInterfaceType>(instRefTy.getNestedType());
    if (!vifTy)
      return {};

    // Read the base interface instance.
    Value vifValue = moore::ReadOp::create(builder, loc, instRef);

    // Walk nested interface instances in the hierarchical path, if any.
    for (size_t i = baseIndex + 1; i < expr.ref.path.size(); ++i) {
      auto *childInst =
          expr.ref.path[i].symbol->as_if<slang::ast::InstanceSymbol>();
      if (!childInst ||
          childInst->getDefinition().definitionKind !=
              slang::ast::DefinitionKind::Interface)
        continue;

      auto *ifaceLowering = context.convertInterfaceHeader(&childInst->body);
      if (!ifaceLowering)
        return {};

      auto childIfaceRef = mlir::FlatSymbolRefAttr::get(
          builder.getContext(), ifaceLowering->op.getSymName());
      auto childVifTy =
          moore::VirtualInterfaceType::get(builder.getContext(), childIfaceRef);
      auto childRefTy = moore::RefType::get(childVifTy);
      auto signalSym = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                    childInst->name);

      Value childRef = moore::VirtualInterfaceSignalRefOp::create(
          builder, loc, childRefTy, vifValue, signalSym);
      vifValue = moore::ReadOp::create(builder, loc, childRef);
      vifTy = childVifTy;
    }

    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    auto signalSym =
        mlir::FlatSymbolRefAttr::get(builder.getContext(), signalName);
    auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
    Value signalRef = moore::VirtualInterfaceSignalRefOp::create(
        builder, loc, refTy, vifValue, signalSym);

    if (isLvalue)
      return signalRef;
    return moore::ReadOp::create(builder, loc, signalRef);
  }

  /// Convert an expression either as an lvalue or rvalue, depending on whether
  /// this is an lvalue or rvalue visitor. This is useful for projections such
  /// as `a[i]`, where you want `a` as an lvalue if you want `a[i]` as an
  /// lvalue, or `a` as an rvalue if you want `a[i]` as an rvalue.
  Value convertLvalueOrRvalueExpression(const slang::ast::Expression &expr) {
    if (isLvalue)
      return context.convertLvalueExpression(expr);
    return context.convertRvalueExpression(expr);
  }

  /// Handle virtual interface member access. When slang resolves vif.data,
  /// it creates a NamedValueExpression pointing directly to the interface
  /// member variable. We need to parse the syntax to find the virtual
  /// interface base and emit the proper VirtualInterfaceSignalRefOp.
  ///
  /// This also handles nested interface access like vif.inner.data where
  /// 'inner' is a nested interface instance inside the virtual interface.
  /// In this case, we need to build a chain of VirtualInterfaceSignalRefOp
  /// operations to traverse the interface hierarchy.
  ///
  /// Returns the result value, or nullptr if this is not a virtual interface
  /// member access that we can handle.
  Value visitVirtualInterfaceMemberAccess(
      const slang::ast::NamedValueExpression &expr,
      const slang::syntax::SyntaxNode &syntax) {
    // Use slang's expression binding to get the expression for the left side
    // (the virtual interface variable). We need the current scope where the
    // expression is being evaluated.
    if (!context.currentScope)
      return {};

    // Create an AST context using the current evaluation scope
    slang::ast::ASTContext astContext(*context.currentScope,
                                      slang::ast::LookupLocation::max);

    // Collect the path of interface member names by walking up the syntax tree.
    // For "vif.middle.inner.data", this collects ["middle", "inner"] as the
    // intermediate interface path (the final "data" is the signal name).
    SmallVector<std::string_view, 4> interfacePath;
    const slang::syntax::NameSyntax *currentSyntax = nullptr;

    // Handle scoped names like vif.data or vif.inner.data
    if (auto *scoped = syntax.as_if<slang::syntax::ScopedNameSyntax>())
      currentSyntax = scoped->left;

    if (!currentSyntax)
      return {};

    // Walk up the syntax tree to find the virtual interface base and collect
    // intermediate interface instance names.
    while (currentSyntax) {
      // First, check if this is a ScopedNameSyntax. If so, we need to
      // recursively check if we have a virtual interface at any level.
      if (auto *scoped =
              currentSyntax->as_if<slang::syntax::ScopedNameSyntax>()) {
        // Bind the left side to check if it's a virtual interface
        const auto &leftExpr =
            slang::ast::Expression::bind(*scoped->left, astContext);
        if (leftExpr.bad()) {
          // If binding fails, this might be because the left side contains
          // nested scoped names that slang can't resolve individually.
          // Try to continue by extracting the member name and walking further.
          if (auto *leftScoped =
                  scoped->left->as_if<slang::syntax::ScopedNameSyntax>()) {
            if (auto *rightIdent =
                    scoped->right->as_if<slang::syntax::IdentifierNameSyntax>()) {
              interfacePath.push_back(rightIdent->identifier.valueText());
            }
            currentSyntax = scoped->left;
            continue;
          }
        }
        if (!leftExpr.bad()) {
          if (leftExpr.type->isVirtualInterface()) {
            // The left side is a virtual interface! Add the right side as an
            // intermediate interface and then process.
            if (auto *rightIdent =
                    scoped->right->as_if<slang::syntax::IdentifierNameSyntax>()) {
              interfacePath.push_back(rightIdent->identifier.valueText());
            }
            currentSyntax = scoped->left;
            continue;
          }
          // If the left side is not a virtual interface but is an interface
          // instance reference (ArbitrarySymbolExpression pointing to an
          // Interface), we should continue walking up. The right side is an
          // intermediate member.
          if (auto *arb =
                  leftExpr.as_if<slang::ast::ArbitrarySymbolExpression>()) {
            if (auto *instSym =
                    arb->symbol->as_if<slang::ast::InstanceSymbol>()) {
              if (instSym->getDefinition().definitionKind ==
                  slang::ast::DefinitionKind::Interface) {
                // This is an intermediate interface access. Add the right side
                // to the path and continue walking up.
                if (auto *rightIdent = scoped->right->as_if<
                        slang::syntax::IdentifierNameSyntax>()) {
                  interfacePath.push_back(rightIdent->identifier.valueText());
                }
                currentSyntax = scoped->left;
                continue;
              }
            }
          }
        }
      }

      // Try to bind this syntax to see if we've reached a virtual interface
      const auto &boundExpr =
          slang::ast::Expression::bind(*currentSyntax, astContext);
      if (boundExpr.bad())
        return {};

      // If this is a virtual interface, we've found the base
      if (boundExpr.type->isVirtualInterface()) {
        // Now convert the virtual interface expression to get the MLIR value
        Value vifValue = context.convertRvalueExpression(boundExpr);
        if (!vifValue)
          return {};

        auto vifTy = dyn_cast<moore::VirtualInterfaceType>(vifValue.getType());
        if (!vifTy)
          return {};

        // Walk through intermediate interface instances to build signal ref
        // chain. The path is collected in reverse order, so process from end.
        for (auto it = interfacePath.rbegin(); it != interfacePath.rend();
             ++it) {
          std::string_view nestedIfaceName = *it;

          // Get the interface declaration for the current virtual interface
          auto ifaceSymRef = vifTy.getInterfaceRef();
          auto *ifaceOp = mlir::SymbolTable::lookupNearestSymbolFrom(
              context.intoModuleOp, ifaceSymRef);
          if (!ifaceOp)
            return {};
          auto ifaceDecl = dyn_cast_or_null<moore::InterfaceDeclOp>(ifaceOp);
          if (!ifaceDecl)
            return {};

          // If the interface body is empty, it may not have been converted yet.
          // This can happen when we're processing a class method that references
          // a virtual interface before the interface body conversion phase.
          // Look up the interface in the Context and trigger body conversion.
          if (ifaceDecl.getBody().front().empty()) {
            // Find the interface in the interfaces map by matching the op
            for (auto &[iface, lowering] : context.interfaces) {
              if (lowering && lowering->op == ifaceDecl) {
                if (failed(context.convertInterfaceBody(iface)))
                  return {};
                break;
              }
            }
          }

          // Find the nested interface signal declaration
          moore::InterfaceSignalDeclOp nestedIfaceSignal;
          for (auto &op : ifaceDecl.getBody().front()) {
            if (auto signalDecl =
                    dyn_cast<moore::InterfaceSignalDeclOp>(&op)) {
              if (signalDecl.getSymName() == StringRef(nestedIfaceName)) {
                nestedIfaceSignal = signalDecl;
                break;
              }
            }
          }

          if (!nestedIfaceSignal)
            return {};

          // Check if this signal is a virtual interface type (nested interface)
          auto nestedSignalType = nestedIfaceSignal.getType();
          auto nestedVifTy =
              dyn_cast<moore::VirtualInterfaceType>(nestedSignalType);
          if (!nestedVifTy)
            return {};

          // Create signal ref to get the nested interface
          auto signalSym =
              mlir::FlatSymbolRefAttr::get(builder.getContext(), nestedIfaceName);
          auto childRefTy = moore::RefType::get(nestedVifTy);
          Value childRef = moore::VirtualInterfaceSignalRefOp::create(
              builder, loc, childRefTy, vifValue, signalSym);

          // Read the nested interface for the next iteration
          vifValue = moore::ReadOp::create(builder, loc, childRef);
          vifTy = nestedVifTy;
        }

        // Now create the final signal reference for the actual signal
        auto type = context.convertType(*expr.type);
        if (!type)
          return {};

        // For modport ports, use the internal symbol's name (the actual signal
        // name), not the modport port name which might be different.
        std::string_view signalName = expr.symbol.name;
        if (auto *modportPort =
                expr.symbol.as_if<slang::ast::ModportPortSymbol>()) {
          if (modportPort->internalSymbol)
            signalName = modportPort->internalSymbol->name;
        }

        auto signalSym =
            mlir::FlatSymbolRefAttr::get(builder.getContext(), signalName);
        auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
        Value signalRef = moore::VirtualInterfaceSignalRefOp::create(
            builder, loc, refTy, vifValue, signalSym);

        // For rvalue, read from the reference; for lvalue, return the ref
        return isLvalue ? signalRef
                        : moore::ReadOp::create(builder, loc, signalRef);
      }

      // Check if this is an interface instance access (direct or hierarchical)
      // For "outer.inner" where outer is a direct interface instance and inner
      // is a nested interface, we need to handle this path as well.
      if (auto *arb = boundExpr.as_if<slang::ast::ArbitrarySymbolExpression>()) {
        if (auto *instSym = arb->symbol->as_if<slang::ast::InstanceSymbol>()) {
          if (instSym->getDefinition().definitionKind ==
              slang::ast::DefinitionKind::Interface) {
            // This is an interface instance. Try to resolve it.
            // If resolution fails, it might be a nested interface inside a
            // virtual interface, so we continue walking up the syntax tree.
            Value ifaceRef = context.resolveInterfaceInstance(instSym, loc);
            if (ifaceRef) {
              auto ifaceRefTy = dyn_cast<moore::RefType>(ifaceRef.getType());
              if (!ifaceRefTy)
                return {};

              auto vifTy = dyn_cast<moore::VirtualInterfaceType>(
                  ifaceRefTy.getNestedType());
              if (!vifTy)
                return {};

              // Read the interface to get the virtual interface value
              Value vifValue = moore::ReadOp::create(builder, loc, ifaceRef);

              // Walk through intermediate interface instances
              for (auto it = interfacePath.rbegin(); it != interfacePath.rend();
                   ++it) {
                std::string_view nestedIfaceName = *it;

                // Get the interface declaration
                auto ifaceSymRef = vifTy.getInterfaceRef();
                auto *ifaceOp = mlir::SymbolTable::lookupNearestSymbolFrom(
                    context.intoModuleOp, ifaceSymRef);
                auto ifaceDecl =
                    dyn_cast_or_null<moore::InterfaceDeclOp>(ifaceOp);
                if (!ifaceDecl)
                  return {};

                // Find the nested interface signal declaration
                moore::InterfaceSignalDeclOp nestedIfaceSignal;
                for (auto &op : ifaceDecl.getBody().front()) {
                  if (auto signalDecl =
                          dyn_cast<moore::InterfaceSignalDeclOp>(&op)) {
                    if (signalDecl.getSymName() == StringRef(nestedIfaceName)) {
                      nestedIfaceSignal = signalDecl;
                      break;
                    }
                  }
                }

                if (!nestedIfaceSignal)
                  return {};

                auto nestedSignalType = nestedIfaceSignal.getType();
                auto nestedVifTy =
                    dyn_cast<moore::VirtualInterfaceType>(nestedSignalType);
                if (!nestedVifTy)
                  return {};

                auto signalSym = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), nestedIfaceName);
                auto childRefTy = moore::RefType::get(nestedVifTy);
                Value childRef = moore::VirtualInterfaceSignalRefOp::create(
                    builder, loc, childRefTy, vifValue, signalSym);

                vifValue = moore::ReadOp::create(builder, loc, childRef);
                vifTy = nestedVifTy;
              }

              // Create the final signal reference
              auto type = context.convertType(*expr.type);
              if (!type)
                return {};

              std::string_view signalName = expr.symbol.name;
              if (auto *modportPort =
                      expr.symbol.as_if<slang::ast::ModportPortSymbol>()) {
                if (modportPort->internalSymbol)
                  signalName = modportPort->internalSymbol->name;
              }

              auto signalSym =
                  mlir::FlatSymbolRefAttr::get(builder.getContext(), signalName);
              auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
              Value signalRef = moore::VirtualInterfaceSignalRefOp::create(
                  builder, loc, refTy, vifValue, signalSym);

              return isLvalue ? signalRef
                              : moore::ReadOp::create(builder, loc, signalRef);
            }
            // Resolution failed - this might be a nested interface inside a
            // virtual interface. Fall through to continue walking up.
          }
        }
      }

      // This is an intermediate interface access. Extract the member name
      // and continue walking up.
      if (auto *scoped =
              currentSyntax->as_if<slang::syntax::ScopedNameSyntax>()) {
        // The right side is the member name at this level
        if (auto *rightIdent =
                scoped->right->as_if<slang::syntax::IdentifierNameSyntax>()) {
          interfacePath.push_back(rightIdent->identifier.valueText());
        }
        currentSyntax = scoped->left;
      } else {
        // Can't continue walking
        return {};
      }
    }

    return {};
  }

  /// Handle single bit selections.
  Value visit(const slang::ast::ElementSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    // We only support indexing into a few select types for now.
    auto derefType = value.getType();
    if (isLvalue)
      derefType = cast<moore::RefType>(derefType).getNestedType();

    // String character access: s[i] returns a byte (8-bit integer)
    if (isa<moore::StringType>(derefType)) {
      auto index = context.convertRvalueExpression(expr.selector());
      if (!index)
        return {};
      // String indexing uses getc() method - s[i] is equivalent to s.getc(i)
      return moore::StringGetCOp::create(builder, loc, value, index);
    }

    if (!isa<moore::IntType, moore::ArrayType, moore::UnpackedArrayType,
             moore::QueueType, moore::AssocArrayType,
             moore::WildcardAssocArrayType, moore::OpenUnpackedArrayType>(
            derefType)) {
      mlir::emitError(loc) << "unsupported expression: element select into "
                           << expr.value().type->toString() << "\n";
      return {};
    }

    auto resultType =
        isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type)) : type;

    // For queue types, associative arrays, dynamic arrays, and other
    // dynamically-sized types, we use the index directly without translation
    // since they are 0-based or use non-integer keys.
    bool isDynamicType = isa<moore::QueueType, moore::AssocArrayType,
                            moore::WildcardAssocArrayType,
                            moore::OpenUnpackedArrayType>(derefType);

    if (isDynamicType) {
      // Dynamic types (queues) use the index directly - always use dynamic
      // extract since we can't statically verify bounds.
      //
      // For queues and dynamic arrays, set the target value so that the `$`
      // (UnboundedLiteral) expression can be evaluated as `size - 1`.
      Value queueValue;
      if (isa<moore::QueueType, moore::OpenUnpackedArrayType>(derefType)) {
        // For lvalue references, read the current array value.
        if (isLvalue) {
          queueValue = moore::ReadOp::create(builder, loc,
                                             cast<moore::RefType>(value.getType())
                                                 .getNestedType(),
                                             value);
        } else {
          queueValue = value;
        }
      }

      // Set the queue target for evaluating `$` in the selector expression.
      auto prevQueueTarget = context.queueTargetValue;
      if (queueValue)
        context.queueTargetValue = queueValue;
      auto restoreQueueTarget =
          llvm::make_scope_exit([&] { context.queueTargetValue = prevQueueTarget; });

      auto indexValue = context.convertRvalueExpression(expr.selector());
      if (!indexValue)
        return {};

      // Associative arrays (including wildcard) can have non-integral keys
      // (string, class handle). Queues and dynamic arrays require integral
      // indices.
      if (isa<moore::AssocArrayType, moore::WildcardAssocArrayType>(derefType)) {
        // For associative arrays, the key type can be anything
        // (int, string, class handle, etc.) - no type check needed.
      } else if (!isa<moore::IntType>(indexValue.getType())) {
        mlir::emitError(loc) << "queue/array index is not integral: "
                             << indexValue.getType();
        return {};
      }
      if (isLvalue) {
        context.captureRef(value);
        return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                              indexValue);
      } else
        return moore::DynExtractOp::create(builder, loc, resultType, value,
                                           indexValue);
    }

    // For fixed-size types, we need to translate the index based on the range.
    auto range = expr.value().type->getFixedRange();
    if (auto *constValue = expr.selector().getConstant();
        constValue && constValue->isInteger()) {
      assert(!constValue->hasUnknown());
      assert(constValue->size() <= 32);

      auto lowBit = constValue->integer().as<uint32_t>().value();

      // If the constant index is out of bounds, emit a warning and produce a
      // no-op.  For lvalues, create a temporary variable so writes are
      // silently discarded (matching Xcelium/VCS behavior).  For rvalues,
      // return a zero constant.
      if (!range.containsPoint(static_cast<int32_t>(lowBit))) {
        mlir::emitWarning(loc) << "constant index " << lowBit
                               << " is out of bounds for range ["
                               << range.left << ":" << range.right << "]";
        if (isLvalue) {
          auto tmpVar = moore::VariableOp::create(
              builder, loc, cast<moore::RefType>(resultType),
              builder.getStringAttr("__oob_discard__"), Value());
          return tmpVar;
        }
        return moore::ConstantOp::create(builder, loc,
                                         cast<moore::IntType>(type), 0);
      }

      if (isLvalue) {
        context.captureRef(value);
        return moore::ExtractRefOp::create(builder, loc, resultType, value,
                                           range.translateIndex(lowBit));
      } else
        return moore::ExtractOp::create(builder, loc, resultType, value,
                                        range.translateIndex(lowBit));
    }

    auto lowBit = context.convertRvalueExpression(expr.selector());
    if (!lowBit)
      return {};
    lowBit = getSelectIndex(context, loc, lowBit, range);
    if (isLvalue) {
      context.captureRef(value);
      return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                            lowBit);
    } else
      return moore::DynExtractOp::create(builder, loc, resultType, value,
                                         lowBit);
  }

  /// Handle range bit selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    auto derefType = value.getType();
    if (isLvalue)
      derefType = cast<moore::RefType>(derefType).getNestedType();

    if (auto queueType = dyn_cast<moore::QueueType>(derefType)) {
      if (isLvalue) {
        mlir::emitError(loc) << "queue range select is not an lvalue";
        return {};
      }

      Value queueValue = value;

      auto prevQueueTarget = context.queueTargetValue;
      context.queueTargetValue = queueValue;
      auto restoreQueueTarget =
          llvm::make_scope_exit([&] { context.queueTargetValue = prevQueueTarget; });

      auto leftValue = context.convertRvalueExpression(expr.left());
      auto rightValue = context.convertRvalueExpression(expr.right());
      if (!leftValue || !rightValue)
        return {};

      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      auto normalizeIndex = [&](Value idx) -> Value {
        idx = context.convertToSimpleBitVector(idx);
        if (!idx)
          return {};
        if (!isa<moore::IntType>(idx.getType())) {
          mlir::emitError(loc) << "queue slice index is not integral: "
                               << idx.getType();
          return {};
        }
        return context.materializeConversion(i32Ty, idx, false, loc);
      };

      leftValue = normalizeIndex(leftValue);
      rightValue = normalizeIndex(rightValue);
      if (!leftValue || !rightValue)
        return {};

      Value startValue;
      Value endValue;
      auto one = moore::ConstantOp::create(builder, loc, i32Ty, 1);

      using slang::ast::RangeSelectionKind;
      switch (expr.getSelectionKind()) {
      case RangeSelectionKind::Simple:
        startValue = leftValue;
        endValue = rightValue;
        break;
      case RangeSelectionKind::IndexedUp: {
        auto sum = moore::AddOp::create(builder, loc, leftValue, rightValue);
        endValue = moore::SubOp::create(builder, loc, sum, one);
        startValue = leftValue;
        break;
      }
      case RangeSelectionKind::IndexedDown: {
        auto diff = moore::SubOp::create(builder, loc, leftValue, rightValue);
        startValue = moore::AddOp::create(builder, loc, diff, one);
        endValue = leftValue;
        break;
      }
      }

      return moore::QueueSliceOp::create(builder, loc, queueType, queueValue,
                                         startValue, endValue);
    }

    std::optional<int32_t> constLeft;
    std::optional<int32_t> constRight;
    if (auto *constant = expr.left().getConstant())
      constLeft = constant->integer().as<int32_t>();
    if (auto *constant = expr.right().getConstant())
      constRight = constant->integer().as<int32_t>();

    Value offsetDyn;

    if (!constRight) {
      // Allow dynamic right bounds by evaluating at runtime.
      auto dynRight = context.convertRvalueExpression(expr.right());
      if (!dynRight)
        return {};
      offsetDyn = dynRight;
    }

    // We need to determine the right bound of the range. This is the address of
    // the least significant bit of the underlying bit storage, which is the
    // offset we want to pass to the extract op.
    //
    // The arrays [6:2] and [2:6] both have 5 bits worth of underlying storage.
    // The left and right bound of the range only determine the addressing
    // scheme of the storage bits:
    //
    // Storage bits:   4  3  2  1  0  <-- extract op works on storage bits
    // [6:2] indices:  6  5  4  3  2  ("little endian" in Slang terms)
    // [2:6] indices:  2  3  4  5  6  ("big endian" in Slang terms)
    //
    // Before we can extract, we need to map the range select left and right
    // bounds from these indices to actual bit positions in the storage.

    int32_t offsetConst = 0;
    auto range = expr.value().type->getFixedRange();

    using slang::ast::RangeSelectionKind;
    if (expr.getSelectionKind() == RangeSelectionKind::Simple) {
      // For a constant range [a:b], we want the offset of the lowest storage
      // bit from which we are starting the extract. For a range [5:3] this is
      // bit index 3; for a range [3:5] this is bit index 5. Both of these are
      // later translated map to bit offset 1 (see bit indices above).
      if (constRight)
        offsetConst = *constRight;
      else if (!offsetDyn)
        offsetDyn = context.convertRvalueExpression(expr.right());
    } else {
      // For an indexed range [a+:b] or [a-:b], determining the lowest storage
      // bit is a bit more complicated. We start out with the base index `a`.
      // This is the lower *index* of the range, but not the lower *storage bit
      // position*.
      //
      // The range [a+:b] expands to [a+b-1:a] for a [6:2] range, or [a:a+b-1]
      // for a [2:6] range. The range [a-:b] expands to [a:a-b+1] for a [6:2]
      // range, or [a-b+1:a] for a [2:6] range.
      if (constLeft) {
        offsetConst = *constLeft;
      } else {
        offsetDyn = context.convertRvalueExpression(expr.left());
        if (!offsetDyn)
          return {};
      }

      // For a [a-:b] select on [2:6] and a [a+:b] select on [6:2], the range
      // expands to [a-b+1:a] and [a+b-1:a]. In this case, the right bound which
      // corresponds to the lower *storage bit offset*, is just `a` and there's
      // no further tweaking to do.
      int32_t offsetAdd = 0;

      // For a [a-:b] select on [6:2], the range expands to [a:a-b+1]. We
      // therefore have to take the `a` from above and adjust it by `-b+1` to
      // arrive at the right bound.
      if (expr.getSelectionKind() == RangeSelectionKind::IndexedDown &&
          range.isLittleEndian()) {
        if (constRight)
          offsetAdd = 1 - *constRight;
      }

      // For a [a+:b] select on [2:6], the range expands to [a:a+b-1]. We
      // therefore have to take the `a` from above and adjust it by `+b-1` to
      // arrive at the right bound.
      if (expr.getSelectionKind() == RangeSelectionKind::IndexedUp &&
          !range.isLittleEndian()) {
        if (constRight)
          offsetAdd = *constRight - 1;
      }

      // Adjust the offset such that it matches the right bound of the range.
      if (offsetAdd != 0) {
        if (offsetDyn) {
          // Convert offsetDyn to a simple bit vector (IntType) if needed.
          // This handles cases where the index expression is an enum or other
          // packed type that needs conversion before arithmetic.
          offsetDyn = context.convertToSimpleBitVector(offsetDyn);
          if (!offsetDyn)
            return {};
          auto offsetIntType = dyn_cast<moore::IntType>(offsetDyn.getType());
          if (!offsetIntType) {
            mlir::emitError(loc)
                << "indexed range select requires integer index type, got "
                << offsetDyn.getType();
            return {};
          }
          if (auto intTy = dyn_cast<moore::IntType>(offsetIntType)) {
            offsetDyn = moore::AddOp::create(
                builder, loc, offsetDyn,
                moore::ConstantOp::create(builder, loc, intTy, offsetAdd,
                                          /*isSigned=*/offsetAdd < 0));
          } else {
            mlir::emitError(loc) << "range select index is not integral: "
                                 << offsetIntType;
            return {};
          }
        }
        else
          offsetConst += offsetAdd;
      }
    }

    // Create a dynamic or constant extract. Use `getSelectIndex` and
    // `ConstantRange::translateIndex` to map from the bit indices provided by
    // the user to the actual storage bit position. Since `offset*` corresponds
    // to the right bound of the range, which provides the index of the least
    // significant selected storage bit, we get the bit offset at which we want
    // to start extracting.
    auto resultType =
        isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type)) : type;

    if (offsetDyn) {
      offsetDyn = getSelectIndex(context, loc, offsetDyn, range);
      if (isLvalue) {
        context.captureRef(value);
        return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                              offsetDyn);
      } else {
        return moore::DynExtractOp::create(builder, loc, resultType, value,
                                           offsetDyn);
      }
    } else {
      offsetConst = range.translateIndex(offsetConst);
      if (isLvalue) {
        context.captureRef(value);
        return moore::ExtractRefOp::create(builder, loc, resultType, value,
                                           offsetConst);
      } else {
        return moore::ExtractOp::create(builder, loc, resultType, value,
                                        offsetConst);
      }
    }
  }

  /// Handle concatenations.
  Value visit(const slang::ast::ConcatenationExpression &expr) {
    // Check if this is a string concatenation by looking at the result type.
    // SystemVerilog has several kinds of concatenation: bit vector, string,
    // and queue.
    bool isStringConcat = expr.type->isString();
    bool isQueueConcat = expr.type->isQueue();
    moore::QueueType queueType;
    if (isQueueConcat) {
      auto resultType = context.convertType(*expr.type);
      queueType = dyn_cast<moore::QueueType>(resultType);
      if (!queueType) {
        mlir::emitError(loc) << "queue concatenation expected queue result, got "
                             << resultType;
        return {};
      }
    }

    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      // Handle empty replications like `{0{...}}` which may occur within
      // concatenations. Slang assigns them a `void` type which we can check for
      // here.
      if (operand->type->isVoid())
        continue;
      auto value = convertLvalueOrRvalueExpression(*operand);
      if (!value)
        return {};
      if (!isLvalue && !isStringConcat && !isQueueConcat)
        value = context.convertToSimpleBitVector(value);
      if (isQueueConcat && !isLvalue) {
        if (!isa<moore::QueueType>(value.getType())) {
          auto elem = context.materializeConversion(
              queueType.getElementType(), value, operand->type->isSigned(), loc);
          if (!elem)
            return {};
          Value emptyQueue =
              moore::QueueConcatOp::create(builder, loc, queueType, {});
          auto refTy = moore::RefType::get(queueType);
          auto tmpVar = moore::VariableOp::create(
              builder, loc, refTy,
              builder.getStringAttr("queue_concat_tmp"), emptyQueue);
          moore::QueuePushBackOp::create(builder, loc, tmpVar, elem);
          value = moore::ReadOp::create(builder, loc, tmpVar);
        } else if (value.getType() != queueType) {
          mlir::emitError(loc)
              << "queue concatenation type mismatch: expected " << queueType
              << ", got " << value.getType();
          return {};
        }
      }
      if (!value)
        return {};
      operands.push_back(value);
    }
    if (isLvalue) {
      for (auto operand : operands)
        context.captureRef(operand);
      return moore::ConcatRefOp::create(builder, loc, operands);
    }
    else if (isStringConcat) {
      // Normalize operands to string type (handles format strings).
      SmallVector<Value> stringOps;
      auto strTy = moore::StringType::get(context.getContext());
      for (auto v : operands) {
        auto asStr = context.materializeConversion(strTy, v, false, v.getLoc());
        if (!asStr)
          return {};
        stringOps.push_back(asStr);
      }
      return moore::StringConcatOp::create(builder, loc, stringOps);
    }
    else if (isQueueConcat) {
      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};
      return moore::QueueConcatOp::create(builder, loc, resultType, operands);
    } else
      return moore::ConcatOp::create(builder, loc, operands);
  }

  /// Handle member accesses.
  Value visit(const slang::ast::MemberAccessExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};

    auto *valueType = expr.value().type.get();
    auto memberName = builder.getStringAttr(expr.member.name);

    // Handle structs.
    if (valueType->isStruct()) {
      auto resultType =
          isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type))
                   : type;
      auto value = convertLvalueOrRvalueExpression(expr.value());
      if (!value)
        return {};

      if (isLvalue) {
        context.captureRef(value);
        return moore::StructExtractRefOp::create(builder, loc, resultType,
                                                 memberName, value);
      }
      return moore::StructExtractOp::create(builder, loc, resultType,
                                            memberName, value);
    }

    // Handle unions.
    if (valueType->isPackedUnion() || valueType->isUnpackedUnion()) {
      auto resultType =
          isLvalue ? moore::RefType::get(cast<moore::UnpackedType>(type))
                   : type;
      auto value = convertLvalueOrRvalueExpression(expr.value());
      if (!value)
        return {};

      // For tagged unions, we need to extract through the struct wrapper.
      // Tagged unions are lowered as struct<{tag: iN, data: union<...>}>
      // First extract the data field, then extract the member from the union.
      if (valueType->isTaggedUnion()) {
        Type valueConvertedType = value.getType();
        // Handle RefType wrapper for lvalues
        if (auto refType = dyn_cast<moore::RefType>(valueConvertedType))
          valueConvertedType = refType.getNestedType();

        // Get union type from the struct wrapper's data field
        Type unionType;
        if (auto packedStruct = dyn_cast<moore::StructType>(valueConvertedType)) {
          if (packedStruct.getMembers().size() == 2)
            unionType = packedStruct.getMembers()[1].type;
        } else if (auto unpackedStruct =
                       dyn_cast<moore::UnpackedStructType>(valueConvertedType)) {
          if (unpackedStruct.getMembers().size() == 2)
            unionType = unpackedStruct.getMembers()[1].type;
        }

        if (!unionType) {
          mlir::emitError(loc)
              << "tagged union must have struct wrapper with data field";
          return {};
        }

        if (isLvalue) {
          context.captureRef(value);
          // Extract reference to data field, then reference to union member
          auto dataRefType = moore::RefType::get(
              cast<moore::UnpackedType>(unionType));
          auto dataRef = moore::StructExtractRefOp::create(
              builder, loc, dataRefType,
              builder.getStringAttr("data"), value);
          return moore::UnionExtractRefOp::create(builder, loc, resultType,
                                                  memberName, dataRef);
        }
        // Extract data field, then extract union member
        auto dataValue = moore::StructExtractOp::create(
            builder, loc, unionType,
            builder.getStringAttr("data"), value);
        return moore::UnionExtractOp::create(builder, loc, type, memberName,
                                             dataValue);
      }

      if (isLvalue) {
        context.captureRef(value);
        return moore::UnionExtractRefOp::create(builder, loc, resultType,
                                                memberName, value);
      }
      return moore::UnionExtractOp::create(builder, loc, type, memberName,
                                           value);
    }

    // Handle classes.
    if (valueType->isClass()) {
      auto valTy = context.convertType(*valueType);
      if (!valTy)
        return {};
      auto targetTy = cast<moore::ClassHandleType>(valTy);

      // Check if this is a class parameter access (e.g., obj.a where a is a
      // class parameter). Class parameters are compile-time constants and
      // can be accessed like properties but are evaluated at elaboration time.
      // IEEE 1800-2017 Section 8.25
      if (auto *paramSym =
              expr.member.as_if<slang::ast::ParameterSymbol>()) {
        // Get the constant value from slang - it has already been elaborated
        const auto &constVal = paramSym->getValue();
        if (!constVal) {
          mlir::emitError(loc)
              << "failed to evaluate class parameter '" << expr.member.name
              << "'";
          return {};
        }
        return context.materializeConstant(constVal, *expr.type, loc);
      }

      // Check if this is a static property accessed through an instance.
      // In SystemVerilog, you can write `obj.static_prop` to access a static
      // property, but static properties are stored as global variables, not
      // in the class declaration.
      if (auto *classProp =
              expr.member.as_if<slang::ast::ClassPropertySymbol>()) {
        if (classProp->lifetime == slang::ast::VariableLifetime::Static) {
          // Static properties are stored as global variables.
          if (auto globalOp = context.globalVariables.lookup(classProp)) {
            auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
            auto symRef =
                mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
            Value ref =
                moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
            return isLvalue ? ref : moore::ReadOp::create(builder, loc, ref);
          }
          // Try on-demand conversion if not found yet.
          if (succeeded(context.convertStaticClassProperty(*classProp))) {
            if (auto globalOp = context.globalVariables.lookup(classProp)) {
              auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
              auto symRef =
                  mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
              Value ref = moore::GetGlobalVariableOp::create(builder, loc,
                                                             refTy, symRef);
              return isLvalue ? ref : moore::ReadOp::create(builder, loc, ref);
            }
          }
          mlir::emitWarning(loc)
              << "static class property '" << expr.member.name
              << "' could not be resolved to a global variable";
        }
      }

      // Get the class that declares this property from slang's AST.
      // This avoids the need to look up the property in Moore IR, which may not
      // have been fully populated yet due to circular dependencies between
      // classes.
      const auto *parentScope = expr.member.getParentScope();
      if (!parentScope) {
        mlir::emitError(loc) << "class property '" << expr.member.name
                             << "' has no parent scope";
        return {};
      }
      const auto *declaringClass =
          parentScope->asSymbol().as_if<slang::ast::ClassType>();
      if (!declaringClass) {
        mlir::emitError(loc) << "class property '" << expr.member.name
                             << "' is not declared in a class";
        return {};
      }

      // Get the actual ClassDeclOp for the declaring class to get its
      // correct symbol name. For parameterized classes, each specialization
      // has its own ClassDeclOp with a potentially renamed symbol.
      // We must call convertClassDeclaration to ensure the class body is
      // populated (not just declared).
      (void)context.convertClassDeclaration(*declaringClass);
      auto *declaringLowering = context.declareClass(*declaringClass);
      if (!declaringLowering || !declaringLowering->op) {
        mlir::emitError(loc) << "failed to get ClassDeclOp for declaring class";
        return {};
      }
      auto declaringClassSym = declaringLowering->op.getSymNameAttr();
      auto upcastTargetTy = moore::ClassHandleType::get(
          context.getContext(),
          mlir::FlatSymbolRefAttr::get(declaringClassSym));

      // Convert the class handle to the required target type for property
      // shadowing purposes, if needed.
      Value baseVal;
      if (upcastTargetTy.getClassSym() == targetTy.getClassSym()) {
        baseVal = context.convertRvalueExpression(expr.value());
      } else {
        baseVal =
            context.convertRvalueExpression(expr.value(), upcastTargetTy);
      }
      if (!baseVal)
        return {};

      // @field and result type !moore.ref<T>.
      auto fieldSym =
          mlir::FlatSymbolRefAttr::get(builder.getContext(), expr.member.name);
      auto fieldRefTy = moore::RefType::get(cast<moore::UnpackedType>(type));

      // Produce a ref to the class property from the (possibly upcast) handle.
      Value fieldRef = moore::ClassPropertyRefOp::create(
          builder, loc, fieldRefTy, baseVal, fieldSym);

      // If we need an RValue, read the reference, otherwise return
      return isLvalue ? fieldRef
                      : moore::ReadOp::create(builder, loc, fieldRef);
    }

    // Handle virtual interfaces.
    if (valueType->isVirtualInterface()) {
      // Convert the base expression (the virtual interface) to get the vif
      // value.
      auto vifVal = context.convertRvalueExpression(expr.value());
      if (!vifVal)
        return {};

      auto vifTy = dyn_cast<moore::VirtualInterfaceType>(vifVal.getType());
      if (!vifTy) {
        mlir::emitError(loc)
            << "expected virtual interface type but got " << vifVal.getType();
        return {};
      }

      // Create a reference to the signal within the virtual interface.
      auto signalSym =
          mlir::FlatSymbolRefAttr::get(builder.getContext(), expr.member.name);
      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
      Value signalRef = moore::VirtualInterfaceSignalRefOp::create(
          builder, loc, refTy, vifVal, signalSym);

      // For lvalue, return the ref; for rvalue, read from it.
      return isLvalue ? signalRef
                      : moore::ReadOp::create(builder, loc, signalRef);
    }

    mlir::emitError(loc, "expression of type ")
        << valueType->toString() << " has no member fields";
    return {};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Rvalue Conversion
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct RvalueExprVisitor : public ExprVisitor {
  RvalueExprVisitor(Context &context, Location loc)
      : ExprVisitor(context, loc, /*isLvalue=*/false) {}
  using ExprVisitor::visit;

  Value convertAssertionPortBinding(const AssertionPortBinding &binding) {
    switch (binding.kind) {
    case AssertionPortBinding::Kind::Expr:
      if (!binding.expr)
        return {};
      return context.convertRvalueExpression(*binding.expr);
    case AssertionPortBinding::Kind::AssertionExpr:
      if (!binding.assertionExpr)
        return {};
      return context.convertAssertionExpression(*binding.assertionExpr, loc,
                                                /*applyDefaults=*/false);
    case AssertionPortBinding::Kind::TimingControl:
      mlir::emitError(loc)
          << "assertion timing control arguments are not yet supported";
      return {};
    }
    llvm_unreachable("unknown assertion port binding kind");
  }

  // Handle references to the left-hand side of a parent assignment.
  Value visit(const slang::ast::LValueReferenceExpression &expr) {
    assert(!context.lvalueStack.empty() && "parent assignments push lvalue");
    auto lvalue = context.lvalueStack.back();
    return moore::ReadOp::create(builder, loc, lvalue);
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    if (context.inAssertionExpr) {
      if (auto *port =
              expr.symbol.as_if<slang::ast::AssertionPortSymbol>()) {
        if (auto *binding = context.lookupAssertionPortBinding(port))
          return convertAssertionPortBinding(*binding);
      }
      if (auto *local =
              expr.symbol.as_if<slang::ast::LocalAssertionVarSymbol>()) {
        auto *binding = context.lookupAssertionLocalVarBinding(local);
        if (!binding) {
          mlir::emitError(loc, "local assertion variable referenced before "
                               "assignment");
          return {};
        }
        auto offset = context.getAssertionSequenceOffset();
        if (offset < binding->offset) {
          mlir::emitError(loc, "local assertion variable referenced before "
                               "assignment time");
          return {};
        }
        if (offset == binding->offset)
          return binding->value;
        if (!isa<moore::UnpackedType>(binding->value.getType())) {
          mlir::emitError(loc, "unsupported local assertion variable type");
          return {};
        }
        return moore::PastOp::create(builder, loc, binding->value,
                                     static_cast<int64_t>(offset -
                                                          binding->offset))
            .getResult();
      }
    }

    // Handle inline constraint receivers and compiler-generated 'this' symbols.
    if (auto inlineRef = context.getInlineConstraintThisRef()) {
      if (auto inlineSym = context.getInlineConstraintThisSymbol();
          inlineSym && inlineSym == &expr.symbol) {
        return inlineRef;
      }
      if (expr.symbol.name == "this")
        return inlineRef;
    }
    if (expr.symbol.name == "this") {
      if (auto thisRef = context.getImplicitThisRef())
        return thisRef;
    }

    // Handle local variables.
    if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
      if (isa<moore::RefType>(value.getType())) {
        auto readOp = moore::ReadOp::create(builder, loc, value);
        if (context.rvalueReadCallback)
          context.rvalueReadCallback(readOp);
        value = readOp.getResult();
      }
      return value;
    }

    // Handle global variables.
    if (auto globalOp = context.globalVariables.lookup(&expr.symbol)) {
      // Use the expression's type rather than the GlobalVariableOp's type.
      // During recursive type conversion, the GlobalVariableOp may temporarily
      // have a placeholder type while the actual type is being determined.
      auto varType = context.convertType(*expr.type);
      if (!varType)
        return {};
      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(varType));
      auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      auto value =
          moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
      return moore::ReadOp::create(builder, loc, value);
    }

    // Try on-demand conversion for global variables that haven't been converted
    // yet. This handles forward references where a variable is used before
    // being visited (e.g., in static method initializers of classes that are
    // converted before the variable itself).
    if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>()) {
      auto parentKind = var->getParentScope()->asSymbol().kind;
      if (parentKind == slang::ast::SymbolKind::Package ||
          parentKind == slang::ast::SymbolKind::Root ||
          parentKind == slang::ast::SymbolKind::CompilationUnit) {
        if (succeeded(context.convertGlobalVariable(*var))) {
          if (auto globalOp = context.globalVariables.lookup(&expr.symbol)) {
            // Use the expression's type rather than the GlobalVariableOp's type.
            auto varType = context.convertType(*expr.type);
            if (!varType)
              return {};
            auto refTy = moore::RefType::get(cast<moore::UnpackedType>(varType));
            auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
            auto value =
                moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
            return moore::ReadOp::create(builder, loc, value);
          }
        }
      }
    }

    // We're reading a class property.
    if (auto *const property =
            expr.symbol.as_if<slang::ast::ClassPropertySymbol>()) {
      auto fieldRef = visitClassProperty(context, *property);
      if (!fieldRef)
        return {};
      return moore::ReadOp::create(builder, loc, fieldRef).getResult();
    }

    // Try to materialize constant values directly.
    auto constant = context.evaluateConstant(expr);
    if (auto value = context.materializeConstant(constant, *expr.type, loc))
      return value;

    // Handle virtual interface member access. When accessing a member through
    // a virtual interface (e.g., vif.data), slang gives us a NamedValueExpression
    // where the symbol is the interface member but accessed via a virtual
    // interface variable. We detect this by checking if the symbol's parent
    // scope is an interface body and the syntax is a scoped name.
    // This applies to VariableSymbol (output/inout ports), NetSymbol
    // (input ports which are nets in SystemVerilog), and ModportPortSymbol
    // (when accessing ports through a modport-qualified virtual interface).
    {
      const slang::ast::Scope *symbolScope = nullptr;
      if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>())
        symbolScope = var->getParentScope();
      else if (auto *net = expr.symbol.as_if<slang::ast::NetSymbol>())
        symbolScope = net->getParentScope();
      else if (auto *modportPort =
                   expr.symbol.as_if<slang::ast::ModportPortSymbol>())
        symbolScope = modportPort->getParentScope();

      if (symbolScope) {
        auto parentKind = symbolScope->asSymbol().kind;
        // For ModportPortSymbol, the parent is a Modport which is inside
        // an InstanceBody. For VariableSymbol/NetSymbol, parent is directly
        // InstanceBody.
        if (parentKind == slang::ast::SymbolKind::InstanceBody ||
            parentKind == slang::ast::SymbolKind::Modport) {
          // Check if this is accessed through a virtual interface by looking at
          // the syntax.
          if (expr.syntax) {
            if (auto result =
                    visitVirtualInterfaceMemberAccess(expr, *expr.syntax))
              return result;
          }
        }
      }
    }

    // Handle interface signal access from within an interface method.
    // When we're inside an interface task/function, signal references
    // should use the implicit interface argument.
    if (context.currentInterfaceArg) {
      StringRef signalName;
      auto it = context.interfaceSignalNames.find(&expr.symbol);
      if (it != context.interfaceSignalNames.end()) {
        signalName = it->second;
      } else if (auto *scope = expr.symbol.getParentScope()) {
        if (auto *body =
                scope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()) {
          if (body == context.currentInterfaceBody)
            signalName = expr.symbol.name;
        }
      }

      if (!signalName.empty()) {
        // This is an interface signal access from within an interface context.
        // Use VirtualInterfaceSignalRefOp to access the signal through the
        // implicit interface argument.
        auto type = context.convertType(*expr.type);
        if (!type)
          return {};

        auto signalSym =
            mlir::FlatSymbolRefAttr::get(builder.getContext(), signalName);
        auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
        Value signalRef = moore::VirtualInterfaceSignalRefOp::create(
            builder, loc, refTy, context.currentInterfaceArg, signalSym);

        // For rvalue, read from the reference
        return moore::ReadOp::create(builder, loc, signalRef);
      }
    }

    // Handle clocking block signal access (ClockVar).
    // When accessing a signal through a clocking block (e.g., cb.signal),
    // slang gives us a NamedValueExpression where the symbol is a ClockVarSymbol.
    // The ClockVarSymbol has an initializer expression pointing to the
    // underlying signal.
    if (auto *clockVar = expr.symbol.as_if<slang::ast::ClockVarSymbol>()) {
      // Get the initializer expression which references the underlying signal
      auto *initExpr = clockVar->getInitializer();
      if (!initExpr) {
        mlir::emitError(loc)
            << "clocking block signal '" << clockVar->name
            << "' has no underlying signal reference";
        return {};
      }

      // For input signals (rvalue context), we read the underlying signal value.
      // The input skew delay would be applied at the clocking block level,
      // not here during individual signal access.
      return context.convertRvalueExpression(*initExpr);
    }

    // Otherwise some other part of ImportVerilog should have added an MLIR
    // value for this expression's symbol to the `context.valueSymbols` table.
    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no rvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle hierarchical values, such as `x = Top.sub.var`.
  Value visit(const slang::ast::HierarchicalValueExpression &expr) {
    auto hierLoc = context.convertLocation(expr.symbol.location);
    if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
      if (isa<moore::RefType>(value.getType())) {
        auto readOp = moore::ReadOp::create(builder, hierLoc, value);
        if (context.rvalueReadCallback)
          context.rvalueReadCallback(readOp);
        value = readOp.getResult();
      }
      return value;
    }

    // Handle direct interface member access (e.g., intf.clk where intf is a
    // direct interface instance, not a virtual interface). Check if the
    // symbol's parent is an interface body. This applies to both VariableSymbol
    // (for output/inout ports and internal variables) and NetSymbol (for input
    // ports which are nets).
    const slang::ast::Scope *parentScope = nullptr;
    if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>())
      parentScope = var->getParentScope();
    else if (auto *net = expr.symbol.as_if<slang::ast::NetSymbol>())
      parentScope = net->getParentScope();

    if (parentScope) {
      if (auto *instBody =
              parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()) {
        if (instBody->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface) {
          // This is a variable/net inside an interface. Resolve the interface
          // instance from the hierarchical path and build the signal access,
          // including nested interface instances.
          if (auto value = resolveHierarchicalInterfaceSignalRef(
                  expr, expr.symbol.name)) {
            if (context.rvalueReadCallback) {
              if (auto readOp = value.getDefiningOp<moore::ReadOp>())
                context.rvalueReadCallback(readOp);
            }
            return value;
          }
        }
      }
    }

    // Handle modport port access (e.g., port.clk where port has type
    // interface.modport). The expr.symbol is a ModportPortSymbol, and we need
    // to access the actual interface signal via its internalSymbol.
    if (auto *modportPort =
            expr.symbol.as_if<slang::ast::ModportPortSymbol>()) {
      // Get the internal symbol (the actual interface signal)
      auto *internalSym = modportPort->internalSymbol;
      if (internalSym) {
        // Get the parent scope of the internal symbol to verify it's in an
        // interface
        auto *parentScope = internalSym->getParentScope();
        if (parentScope) {
          if (auto *instBody = parentScope->asSymbol()
                                   .as_if<slang::ast::InstanceBodySymbol>()) {
            if (instBody->getDefinition().definitionKind ==
                slang::ast::DefinitionKind::Interface) {
              if (auto value = resolveHierarchicalInterfaceSignalRef(
                      expr, internalSym->name)) {
                if (context.rvalueReadCallback) {
                  if (auto readOp = value.getDefiningOp<moore::ReadOp>())
                    context.rvalueReadCallback(readOp);
                }
                return value;
              }
            }
          }
        }
      }
    }

    // Emit an error for those hierarchical values not recorded in the
    // `valueSymbols`.
    auto d = mlir::emitError(loc, "unknown hierarchical name `")
             << expr.symbol.name << "`";
    d.attachNote(hierLoc) << "no rvalue generated for "
                          << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle arbitrary symbol expressions, such as interface instance references.
  // When assigning an interface instance to a virtual interface variable,
  // slang represents the interface instance as an ArbitrarySymbolExpression.
  Value visit(const slang::ast::ArbitrarySymbolExpression &expr) {
    // Check if this is an interface instance reference
    if (auto *instSym = expr.symbol->as_if<slang::ast::InstanceSymbol>()) {
      // Look up the interface instance in our tracking map
      auto it = context.interfaceInstances.find(instSym);
      if (it != context.interfaceInstances.end()) {
        // Return the reference to the interface instance
        // The caller will use this to assign to a virtual interface variable
        return it->second;
      }
    }

    // Emit an error for other arbitrary symbol expressions we don't support
    auto d = mlir::emitError(loc, "unsupported arbitrary symbol reference `")
             << expr.symbol->name << "`";
    d.attachNote(context.convertLocation(expr.symbol->location))
        << "symbol kind: " << slang::ast::toString(expr.symbol->kind);
    return {};
  }

  // Handle type conversions (explicit and implicit).
  Value visit(const slang::ast::ConversionExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};
    return context.convertRvalueExpression(expr.operand(), type);
  }

  // Handle blocking and non-blocking assignments.
  Value visit(const slang::ast::AssignmentExpression &expr) {
    // Handle streaming concatenation lvalue with dynamic arrays/queues.
    // These require runtime streaming using StreamUnpackOp since the size
    // is not known at compile time.
    if (auto *streamExpr =
            expr.left().as_if<slang::ast::StreamingConcatenationExpression>()) {
      // Check if any stream operand is a dynamic array or queue
      int dynamicArrayIndex = -1;
      for (size_t i = 0; i < streamExpr->streams().size(); ++i) {
        auto stream = streamExpr->streams()[i];
        auto operandType = context.convertType(*stream.operand->type);
        if (operandType && isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                               operandType)) {
          if (dynamicArrayIndex >= 0) {
            // Multiple dynamic arrays in streaming lvalue not supported
            mlir::emitError(loc) << "streaming lvalue with multiple dynamic "
                                    "array operands not supported";
            return {};
          }
          dynamicArrayIndex = i;
        }
      }

      if (dynamicArrayIndex >= 0) {
        bool isMixed = streamExpr->streams().size() > 1;

        // Convert the RHS without type coercion - we want the original type
        auto rhs = context.convertRvalueExpression(expr.right());
        if (!rhs)
          return {};

        // Determine streaming direction from slice size:
        // getSliceSize() == 0 means right-to-left ({>>{}}), otherwise
        // left-to-right ({<<{}}).
        bool isRightToLeft = streamExpr->getSliceSize() != 0;
        int32_t sliceSize = streamExpr->getSliceSize();

        // Handle timing control for blocking assignments
        if (!expr.isNonBlocking()) {
          if (expr.timingControl)
            if (failed(context.convertTimingControl(*expr.timingControl)))
              return {};
        } else {
          mlir::emitError(loc)
              << "non-blocking streaming unpack not yet supported";
          return {};
        }

        if (isMixed) {
          // For mixed streaming, keep the RHS as-is (queue or dynamic array)
          // and let the runtime handle the bit distribution
          // Mixed static/dynamic streaming lvalue
          // Collect static prefix and suffix lvalue references
          SmallVector<Value> staticPrefixRefs;
          SmallVector<Value> staticSuffixRefs;
          Value dynamicArrayRef;

          for (size_t i = 0; i < streamExpr->streams().size(); ++i) {
            auto stream = streamExpr->streams()[i];
            Value ref = context.convertLvalueExpression(*stream.operand);
            if (!ref)
              return {};

            // Convert packed types to simple bit vector refs for static operands
            if (auto refType = dyn_cast<moore::RefType>(ref.getType())) {
              if (!isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                      refType.getNestedType())) {
                if (auto packed =
                        dyn_cast<moore::PackedType>(refType.getNestedType())) {
                  if (!isa<moore::IntType>(packed)) {
                    if (auto bitSize = packed.getBitSize()) {
                      auto intType = moore::RefType::get(moore::IntType::get(
                          context.getContext(), *bitSize, packed.getDomain()));
                      ref = context.materializeConversion(intType, ref, false,
                                                          loc);
                    }
                  }
                }
              }
            }

            if (static_cast<int>(i) < dynamicArrayIndex) {
              staticPrefixRefs.push_back(ref);
            } else if (static_cast<int>(i) == dynamicArrayIndex) {
              dynamicArrayRef = ref;
            } else {
              staticSuffixRefs.push_back(ref);
            }
          }

          // Create the mixed stream unpack operation
          moore::StreamUnpackMixedOp::create(builder, loc, staticPrefixRefs,
                                             dynamicArrayRef, staticSuffixRefs,
                                             rhs, sliceSize, isRightToLeft);
        } else {
          // Single dynamic array operand
          // For single array, convert queue to bits first
          if (isa<moore::QueueType, moore::OpenUnpackedArrayType>(
                  rhs.getType())) {
            // Determine the element type of the source queue/array
            Type elementType;
            if (auto queueType = dyn_cast<moore::QueueType>(rhs.getType()))
              elementType = queueType.getElementType();
            else if (auto arrayType =
                         dyn_cast<moore::OpenUnpackedArrayType>(rhs.getType()))
              elementType = arrayType.getElementType();

            // Determine result type for stream concat - use element size
            Type resultType;
            auto unpackedElem = cast<moore::UnpackedType>(elementType);
            if (auto bitSize = unpackedElem.getBitSize()) {
              resultType = moore::IntType::get(context.getContext(), *bitSize,
                                               unpackedElem.getDomain());
            } else {
              mlir::emitError(loc) << "cannot determine bit size of queue "
                                      "element for streaming unpack";
              return {};
            }

            // Create stream concat to convert queue to bits
            rhs = moore::StreamConcatOp::create(builder, loc, resultType, rhs,
                                                isRightToLeft);
          }
          auto lhs = context.convertLvalueExpression(expr.left());
          if (!lhs)
            return {};
          moore::StreamUnpackOp::create(builder, loc, lhs, rhs, isRightToLeft);
        }
        return rhs;
      }
    }

    // Handle string character assignment: str[i] = c
    // String indexing cannot return a reference type, so we handle this
    // specially by generating a StringPutCOp.
    if (auto *elemSelect =
            expr.left().as_if<slang::ast::ElementSelectExpression>()) {
      auto valueType = context.convertType(*elemSelect->value().type);
      if (valueType && isa<moore::StringType>(valueType)) {
        // Get the string reference (lvalue)
        auto strRef = context.convertLvalueExpression(elemSelect->value());
        if (!strRef)
          return {};
        // Get the index (rvalue)
        auto index = context.convertRvalueExpression(elemSelect->selector());
        if (!index)
          return {};
        // Get the character value (rvalue)
        auto charType = moore::IntType::getInt(context.getContext(), 8);
        auto rhs = context.convertRvalueExpression(expr.right(), charType);
        if (!rhs)
          return {};
        // Handle timing control for blocking assignments
        if (!expr.isNonBlocking()) {
          if (expr.timingControl)
            if (failed(context.convertTimingControl(*expr.timingControl)))
              return {};
        } else {
          // Non-blocking string character assignment is not supported
          mlir::emitError(loc)
              << "non-blocking string character assignment not supported";
          return {};
        }
        // Generate the putc operation
        moore::StringPutCOp::create(builder, loc, strRef, index, rhs);
        return rhs;
      }
    }

    auto lhs = context.convertLvalueExpression(expr.left());
    if (!lhs)
      return {};

    // Determine the right-hand side value of the assignment.
    // The lhs can be either a RefType (for variables, array elements, etc.)
    // or a ClassHandleType (for class handle assignments).
    auto lhsNestedType = getLvalueNestedType(lhs.getType());
    if (!lhsNestedType) {
      mlir::emitError(loc) << "unsupported lvalue type in assignment: "
                           << lhs.getType();
      return {};
    }
    context.lvalueStack.push_back(lhs);
    auto rhs = context.convertRvalueExpression(expr.right(), lhsNestedType);
    context.lvalueStack.pop_back();
    if (!rhs)
      return {};

    // If this is a blocking assignment, we can insert the delay/wait ops of the
    // optional timing control directly in between computing the RHS and
    // executing the assignment.
    if (!expr.isNonBlocking()) {
      if (expr.timingControl)
        if (failed(context.convertTimingControl(*expr.timingControl)))
          return {};
      auto assignOp = moore::BlockingAssignOp::create(builder, loc, lhs, rhs);
      if (context.variableAssignCallback)
        context.variableAssignCallback(assignOp);
      return rhs;
    }

    // For non-blocking assignments, we only support time delays for now.
    if (expr.timingControl) {
      // Handle regular time delays.
      if (auto *ctrl = expr.timingControl->as_if<slang::ast::DelayControl>()) {
        auto delay = context.convertRvalueExpression(
            ctrl->expr, moore::TimeType::get(builder.getContext()));
        if (!delay)
          return {};
        auto assignOp = moore::DelayedNonBlockingAssignOp::create(
            builder, loc, lhs, rhs, delay);
        if (context.variableAssignCallback)
          context.variableAssignCallback(assignOp);
        return rhs;
      }

      // All other timing controls are not supported.
      auto loc = context.convertLocation(expr.timingControl->sourceRange);
      mlir::emitError(loc)
          << "unsupported non-blocking assignment timing control: "
          << slang::ast::toString(expr.timingControl->kind);
      return {};
    }
    auto assignOp = moore::NonBlockingAssignOp::create(builder, loc, lhs, rhs);
    if (context.variableAssignCallback)
      context.variableAssignCallback(assignOp);
    return rhs;
  }

  // Helper function to convert an argument to a simple bit vector type, pass it
  // to a reduction op, and optionally invert the result.
  template <class ConcreteOp>
  Value createReduction(Value arg, bool invert) {
    arg = context.convertToSimpleBitVector(arg);
    if (!arg)
      return {};
    Value result = ConcreteOp::create(builder, loc, arg);
    if (invert)
      result = moore::NotOp::create(builder, loc, result);
    return result;
  }

  // Helper function to create pre and post increments and decrements.
  Value createIncrement(Value arg, bool isInc, bool isPost) {
    auto preValue = moore::ReadOp::create(builder, loc, arg);
    Value postValue;
    // Catch the special case where a signed 1 bit value (i1) is incremented,
    // as +1 can not be expressed as a signed 1 bit value. For any 1-bit number
    // negating is equivalent to incrementing.
    if (moore::isIntType(preValue.getType(), 1)) {
      postValue = moore::NotOp::create(builder, loc, preValue).getResult();
    } else if (auto intType = dyn_cast<moore::IntType>(preValue.getType())) {
      auto one = moore::ConstantOp::create(builder, loc, intType, 1);
      postValue =
          isInc ? moore::AddOp::create(builder, loc, preValue, one).getResult()
                : moore::SubOp::create(builder, loc, preValue, one).getResult();
      auto assignOp =
          moore::BlockingAssignOp::create(builder, loc, arg, postValue);
      if (context.variableAssignCallback)
        context.variableAssignCallback(assignOp);
    } else {
      // Non-integer type (event, class handle, etc.) - cannot be incremented
      mlir::emitError(loc, "cannot apply increment/decrement to non-integer type");
      return {};
    }

    if (isPost)
      return preValue;
    return postValue;
  }

  // Helper function to create pre and post increments and decrements.
  Value createRealIncrement(Value arg, bool isInc, bool isPost) {
    Value preValue = moore::ReadOp::create(builder, loc, arg);
    Value postValue;

    bool isTime = isa<moore::TimeType>(preValue.getType());
    if (isTime)
      preValue = context.materializeConversion(
          moore::RealType::get(context.getContext(), moore::RealWidth::f64),
          preValue, false, loc);

    moore::RealType realTy =
        llvm::dyn_cast<moore::RealType>(preValue.getType());
    if (!realTy)
      return {};

    FloatAttr oneAttr;
    if (realTy.getWidth() == moore::RealWidth::f32) {
      oneAttr = builder.getFloatAttr(builder.getF32Type(), 1.0);
    } else if (realTy.getWidth() == moore::RealWidth::f64) {
      auto oneVal = isTime ? getTimeScaleInFemtoseconds(context) : 1.0;
      oneAttr = builder.getFloatAttr(builder.getF64Type(), oneVal);
    } else {
      mlir::emitError(loc) << "cannot construct increment for " << realTy;
      return {};
    }
    auto one = moore::ConstantRealOp::create(builder, loc, oneAttr);

    postValue =
        isInc
            ? moore::AddRealOp::create(builder, loc, preValue, one).getResult()
            : moore::SubRealOp::create(builder, loc, preValue, one).getResult();

    if (isTime)
      postValue = context.materializeConversion(
          moore::TimeType::get(context.getContext()), postValue, false, loc);

    auto assignOp =
        moore::BlockingAssignOp::create(builder, loc, arg, postValue);

    if (context.variableAssignCallback)
      context.variableAssignCallback(assignOp);

    if (isPost)
      return preValue;
    return postValue;
  }

  Value visitRealUOp(const slang::ast::UnaryExpression &expr) {
    Type opFTy = context.convertType(*expr.operand().type);

    using slang::ast::UnaryOperator;
    Value arg;
    if (expr.op == UnaryOperator::Preincrement ||
        expr.op == UnaryOperator::Predecrement ||
        expr.op == UnaryOperator::Postincrement ||
        expr.op == UnaryOperator::Postdecrement)
      arg = context.convertLvalueExpression(expr.operand());
    else
      arg = context.convertRvalueExpression(expr.operand(), opFTy);
    if (!arg)
      return {};

    // Only covers expressions in 'else' branch above.
    if (isa<moore::TimeType>(arg.getType()))
      arg = context.materializeConversion(
          moore::RealType::get(context.getContext(), moore::RealWidth::f64),
          arg, false, loc);

    switch (expr.op) {
      // `+a` is simply `a`
    case UnaryOperator::Plus:
      return arg;
    case UnaryOperator::Minus:
      return moore::NegRealOp::create(builder, loc, arg);

    case UnaryOperator::Preincrement:
      return createRealIncrement(arg, true, false);
    case UnaryOperator::Predecrement:
      return createRealIncrement(arg, false, false);
    case UnaryOperator::Postincrement:
      return createRealIncrement(arg, true, true);
    case UnaryOperator::Postdecrement:
      return createRealIncrement(arg, false, true);

    case UnaryOperator::LogicalNot:
      arg = context.convertToBool(arg);
      if (!arg)
        return {};
      return moore::NotOp::create(builder, loc, arg);

    default:
      mlir::emitError(loc) << "Unary operator " << slang::ast::toString(expr.op)
                           << " not supported with real values!\n";
      return {};
    }
  }

  // Handle unary operators.
  Value visit(const slang::ast::UnaryExpression &expr) {
    // First check whether we need real or integral BOps
    const auto *floatType =
        expr.operand().type->as_if<slang::ast::FloatingType>();
    // If op is real-typed, treat as real BOp.
    if (floatType)
      return visitRealUOp(expr);

    using slang::ast::UnaryOperator;
    Value arg;
    if (expr.op == UnaryOperator::Preincrement ||
        expr.op == UnaryOperator::Predecrement ||
        expr.op == UnaryOperator::Postincrement ||
        expr.op == UnaryOperator::Postdecrement)
      arg = context.convertLvalueExpression(expr.operand());
    else
      arg = context.convertRvalueExpression(expr.operand());
    if (!arg)
      return {};

    switch (expr.op) {
      // `+a` is simply `a`, but converted to a simple bit vector type since
      // this is technically an arithmetic operation.
    case UnaryOperator::Plus:
      return context.convertToSimpleBitVector(arg);

    case UnaryOperator::Minus:
      arg = context.convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return moore::NegOp::create(builder, loc, arg);

    case UnaryOperator::BitwiseNot:
      arg = context.convertToSimpleBitVector(arg);
      if (!arg)
        return {};
      return moore::NotOp::create(builder, loc, arg);

    case UnaryOperator::BitwiseAnd:
      return createReduction<moore::ReduceAndOp>(arg, false);
    case UnaryOperator::BitwiseOr:
      return createReduction<moore::ReduceOrOp>(arg, false);
    case UnaryOperator::BitwiseXor:
      return createReduction<moore::ReduceXorOp>(arg, false);
    case UnaryOperator::BitwiseNand:
      return createReduction<moore::ReduceAndOp>(arg, true);
    case UnaryOperator::BitwiseNor:
      return createReduction<moore::ReduceOrOp>(arg, true);
    case UnaryOperator::BitwiseXnor:
      return createReduction<moore::ReduceXorOp>(arg, true);

    case UnaryOperator::LogicalNot:
      // Handle LTL types for assertion contexts.
      if (mlir::isa<ltl::PropertyType, ltl::SequenceType>(arg.getType()))
        return ltl::NotOp::create(builder, loc, arg);
      arg = context.convertToBool(arg);
      if (!arg)
        return {};
      return moore::NotOp::create(builder, loc, arg);

    case UnaryOperator::Preincrement:
      return createIncrement(arg, true, false);
    case UnaryOperator::Predecrement:
      return createIncrement(arg, false, false);
    case UnaryOperator::Postincrement:
      return createIncrement(arg, true, true);
    case UnaryOperator::Postdecrement:
      return createIncrement(arg, false, true);
    }

    mlir::emitError(loc, "unsupported unary operator");
    return {};
  }

  /// Short-circuit logical operators (IEEE 1800-2017 §11.4.7).
  /// RHS is deferred inside a ConditionalOp region so it is only evaluated
  /// when the LHS does not determine the result.
  Value buildShortCircuitLogicalOp(const slang::ast::BinaryExpression &expr,
                                   Value lhs) {
    using slang::ast::BinaryOperator;

    // Determine the domain for boolean conversion.
    Domain domain = Domain::TwoValued;
    if (expr.type->isFourState() || expr.left().type->isFourState() ||
        expr.right().type->isFourState())
      domain = Domain::FourValued;

    // Convert LHS to boolean for the condition.
    auto lhsBool = context.convertToBool(lhs, domain);
    if (!lhsBool)
      return {};

    auto resultType = lhsBool.getType();
    auto intResultType = cast<moore::IntType>(resultType);
    auto conditionalOp =
        moore::ConditionalOp::create(builder, loc, resultType, lhsBool);
    auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
    auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);

    // Helper: evaluate RHS inside current insertion point, convert to bool.
    auto evalRhsBool = [&]() -> Value {
      auto rhs = context.convertRvalueExpression(expr.right());
      if (!rhs)
        return {};
      auto rhsBool = context.convertToBool(rhs, domain);
      if (!rhsBool)
        return {};
      // Ensure the type matches the result type.
      if (rhsBool.getType() != resultType)
        rhsBool =
            moore::ConversionOp::create(builder, loc, resultType, rhsBool);
      return rhsBool;
    };

    switch (expr.op) {
    case BinaryOperator::LogicalAnd:
      // a && b: if a, evaluate b; else yield 0
      builder.setInsertionPointToStart(&trueBlock);
      if (auto rhs = evalRhsBool())
        moore::YieldOp::create(builder, loc, rhs);
      else
        return {};
      builder.setInsertionPointToStart(&falseBlock);
      moore::YieldOp::create(
          builder, loc,
          moore::ConstantOp::create(builder, loc, intResultType, 0));
      break;
    case BinaryOperator::LogicalOr:
      // a || b: if a, yield 1; else evaluate b
      builder.setInsertionPointToStart(&trueBlock);
      moore::YieldOp::create(
          builder, loc,
          moore::ConstantOp::create(builder, loc, intResultType, 1));
      builder.setInsertionPointToStart(&falseBlock);
      if (auto rhs = evalRhsBool())
        moore::YieldOp::create(builder, loc, rhs);
      else
        return {};
      break;
    case BinaryOperator::LogicalImplication:
      // a -> b: if a, evaluate b; else yield 1
      builder.setInsertionPointToStart(&trueBlock);
      if (auto rhs = evalRhsBool())
        moore::YieldOp::create(builder, loc, rhs);
      else
        return {};
      builder.setInsertionPointToStart(&falseBlock);
      moore::YieldOp::create(
          builder, loc,
          moore::ConstantOp::create(builder, loc, intResultType, 1));
      break;
    default:
      llvm_unreachable("not a short-circuit logical operator");
    }

    return conditionalOp.getResult();
  }

  /// Handles logical operators (§11.4.7), assuming lhs/rhs are rvalues already.
  Value buildLogicalBOp(slang::ast::BinaryOperator op, Value lhs, Value rhs,
                        std::optional<Domain> domain = std::nullopt) {
    using slang::ast::BinaryOperator;

    // Check if either operand is an LTL type (property or sequence).
    // In assertion contexts, SVA functions like $changed, $stable return LTL
    // types. When used in logical expressions within assertions, we should use
    // LTL operations instead of Moore operations.
    bool lhsIsLTL = mlir::isa<ltl::PropertyType, ltl::SequenceType>(lhs.getType());
    bool rhsIsLTL = mlir::isa<ltl::PropertyType, ltl::SequenceType>(rhs.getType());

    if (lhsIsLTL || rhsIsLTL) {
      // Use LTL operations for assertion contexts.
      // Convert non-LTL operand to i1 if needed.
      if (!lhsIsLTL) {
        lhs = context.convertToI1(lhs);
        if (!lhs)
          return {};
      }
      if (!rhsIsLTL) {
        rhs = context.convertToI1(rhs);
        if (!rhs)
          return {};
      }

      switch (op) {
      case BinaryOperator::LogicalAnd:
        return ltl::AndOp::create(builder, loc, SmallVector<Value, 2>{lhs, rhs});

      case BinaryOperator::LogicalOr:
        return ltl::OrOp::create(builder, loc, SmallVector<Value, 2>{lhs, rhs});

      case BinaryOperator::LogicalImplication: {
        // (lhs -> rhs) == (!lhs || rhs)
        auto notLHS = ltl::NotOp::create(builder, loc, lhs);
        return ltl::OrOp::create(builder, loc,
                                 SmallVector<Value, 2>{notLHS, rhs});
      }

      case BinaryOperator::LogicalEquivalence: {
        // (lhs <-> rhs) == (lhs && rhs) || (!lhs && !rhs)
        auto notLHS = ltl::NotOp::create(builder, loc, lhs);
        auto notRHS = ltl::NotOp::create(builder, loc, rhs);
        auto both = ltl::AndOp::create(builder, loc,
                                       SmallVector<Value, 2>{lhs, rhs});
        auto notBoth = ltl::AndOp::create(builder, loc,
                                          SmallVector<Value, 2>{notLHS, notRHS});
        return ltl::OrOp::create(builder, loc,
                                 SmallVector<Value, 2>{both, notBoth});
      }

      default:
        llvm_unreachable("not a logical BinaryOperator");
      }
    }

    // Standard boolean conversion for non-LTL operands.
    if (domain) {
      lhs = context.convertToBool(lhs, domain.value());
      rhs = context.convertToBool(rhs, domain.value());
    } else {
      lhs = context.convertToBool(lhs);
      rhs = context.convertToBool(rhs);
    }

    if (!lhs || !rhs)
      return {};

    // Ensure both operands have the same type. If they have different domains
    // (e.g., i1 vs l1), convert both to the wider domain (four-valued).
    if (lhs.getType() != rhs.getType()) {
      auto lhsInt = dyn_cast<moore::IntType>(lhs.getType());
      auto rhsInt = dyn_cast<moore::IntType>(rhs.getType());
      if (lhsInt && rhsInt) {
        // Use four-valued if either operand is four-valued.
        auto targetDomain =
            (lhsInt.getDomain() == moore::Domain::FourValued ||
             rhsInt.getDomain() == moore::Domain::FourValued)
                ? moore::Domain::FourValued
                : moore::Domain::TwoValued;
        auto targetType = moore::IntType::get(context.getContext(), 1, targetDomain);
        if (lhs.getType() != targetType)
          lhs = moore::ConversionOp::create(builder, loc, targetType, lhs);
        if (rhs.getType() != targetType)
          rhs = moore::ConversionOp::create(builder, loc, targetType, rhs);
      }
    }

    switch (op) {
    case BinaryOperator::LogicalAnd:
      return moore::AndOp::create(builder, loc, lhs, rhs);

    case BinaryOperator::LogicalOr:
      return moore::OrOp::create(builder, loc, lhs, rhs);

    case BinaryOperator::LogicalImplication: {
      // (lhs -> rhs) == (!lhs || rhs)
      auto notLHS = moore::NotOp::create(builder, loc, lhs);
      return moore::OrOp::create(builder, loc, notLHS, rhs);
    }

    case BinaryOperator::LogicalEquivalence: {
      // (lhs <-> rhs) == (lhs && rhs) || (!lhs && !rhs)
      auto notLHS = moore::NotOp::create(builder, loc, lhs);
      auto notRHS = moore::NotOp::create(builder, loc, rhs);
      auto both = moore::AndOp::create(builder, loc, lhs, rhs);
      auto notBoth = moore::AndOp::create(builder, loc, notLHS, notRHS);
      return moore::OrOp::create(builder, loc, both, notBoth);
    }

    default:
      llvm_unreachable("not a logical BinaryOperator");
    }
  }

  Value visitRealBOp(const slang::ast::BinaryExpression &expr) {
    // Convert operands to the chosen target type.
    auto lhs = context.convertRvalueExpression(expr.left());
    if (!lhs)
      return {};
    auto rhs = context.convertRvalueExpression(expr.right());
    if (!rhs)
      return {};

    if (isa<moore::TimeType>(lhs.getType()) ||
        isa<moore::TimeType>(rhs.getType())) {
      lhs = context.materializeConversion(
          moore::RealType::get(context.getContext(), moore::RealWidth::f64),
          lhs, false, loc);
      rhs = context.materializeConversion(
          moore::RealType::get(context.getContext(), moore::RealWidth::f64),
          rhs, false, loc);
    }

    using slang::ast::BinaryOperator;
    switch (expr.op) {
    case BinaryOperator::Add:
      return moore::AddRealOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::Subtract:
      return moore::SubRealOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::Multiply:
      return moore::MulRealOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::Divide:
      return moore::DivRealOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::Power:
      return moore::PowRealOp::create(builder, loc, lhs, rhs);

    case BinaryOperator::Equality:
      return moore::EqRealOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::Inequality:
      return moore::NeRealOp::create(builder, loc, lhs, rhs);

    case BinaryOperator::GreaterThan:
      return moore::FgtOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::LessThan:
      return moore::FltOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::GreaterThanEqual:
      return moore::FgeOp::create(builder, loc, lhs, rhs);
    case BinaryOperator::LessThanEqual:
      return moore::FleOp::create(builder, loc, lhs, rhs);

    case BinaryOperator::LogicalAnd:
    case BinaryOperator::LogicalOr:
    case BinaryOperator::LogicalImplication:
    case BinaryOperator::LogicalEquivalence:
      return buildLogicalBOp(expr.op, lhs, rhs);

    default:
      mlir::emitError(loc) << "Binary operator "
                           << slang::ast::toString(expr.op)
                           << " not supported with real valued operands!\n";
      return {};
    }
  }

  // Helper function to convert two arguments to a simple bit vector type and
  // pass them into a binary op.
  template <class ConcreteOp>
  Value createBinary(Value lhs, Value rhs) {
    lhs = context.convertToSimpleBitVector(lhs);
    if (!lhs)
      return {};
    rhs = context.convertToSimpleBitVector(rhs);
    if (!rhs)
      return {};
    if (lhs.getType() != rhs.getType()) {
      auto lhsInt = dyn_cast<moore::IntType>(lhs.getType());
      auto rhsInt = dyn_cast<moore::IntType>(rhs.getType());
      if (lhsInt && rhsInt && lhsInt.getWidth() == rhsInt.getWidth()) {
        auto targetDomain =
            (lhsInt.getDomain() == moore::Domain::FourValued ||
             rhsInt.getDomain() == moore::Domain::FourValued)
                ? moore::Domain::FourValued
                : moore::Domain::TwoValued;
        auto targetType = moore::IntType::get(context.getContext(),
                                              lhsInt.getWidth(), targetDomain);
        if (lhs.getType() != targetType)
          lhs = moore::ConversionOp::create(builder, loc, targetType, lhs);
        if (rhs.getType() != targetType)
          rhs = moore::ConversionOp::create(builder, loc, targetType, rhs);
      }
    }
    return ConcreteOp::create(builder, loc, lhs, rhs);
  }

  /// Create an integer constant with a defensive type check. Returns null on
  /// type mismatch and emits an error.
  Value createIntConstant(Type ty, int64_t v, bool isSigned = false) {
    if (auto intTy = dyn_cast<moore::IntType>(ty))
      return moore::ConstantOp::create(builder, loc, intTy, v, isSigned);
    mlir::emitError(loc) << "expected IntType for constant, got " << ty;
    return {};
  }

  // Handle binary operators.
  Value visit(const slang::ast::BinaryExpression &expr) {
    // First check whether we need real or integral BOps
    const auto *rhsFloatType =
        expr.right().type->as_if<slang::ast::FloatingType>();
    const auto *lhsFloatType =
        expr.left().type->as_if<slang::ast::FloatingType>();

    // If either arg is real-typed, treat as real BOp.
    if (rhsFloatType || lhsFloatType)
      return visitRealBOp(expr);

    auto lhs = context.convertRvalueExpression(expr.left());
    if (!lhs)
      return {};

    // Short-circuit evaluation for logical operators (IEEE 1800-2017 §11.4.7).
    // RHS evaluation is deferred inside a ConditionalOp region.
    // Only apply in procedural contexts (moore.procedure / func.func).
    // Skip in concurrent/assertion contexts (module body) and inside
    // array.locator regions where ConditionalOp causes legalization failures.
    {
      using slang::ast::BinaryOperator;
      if (expr.op == BinaryOperator::LogicalAnd ||
          expr.op == BinaryOperator::LogicalOr ||
          expr.op == BinaryOperator::LogicalImplication) {
        bool canShortCircuit = false;
        if (auto *block = builder.getInsertionBlock()) {
          for (auto *op = block->getParentOp(); op; op = op->getParentOp()) {
            if (isa<moore::ArrayLocatorOp>(op)) {
              canShortCircuit = false;
              break;
            }
            if (isa<moore::ProcedureOp>(op) || isa<mlir::func::FuncOp>(op)) {
              canShortCircuit = true;
              break;
            }
          }
        }
        if (canShortCircuit)
          return buildShortCircuitLogicalOp(expr, lhs);
      }
    }

    auto rhs = context.convertRvalueExpression(expr.right());
    if (!rhs)
      return {};

    // Determine the domain of the result.
    Domain domain = Domain::TwoValued;
    if (expr.type->isFourState() || expr.left().type->isFourState() ||
        expr.right().type->isFourState())
      domain = Domain::FourValued;

    using slang::ast::BinaryOperator;
    switch (expr.op) {
    case BinaryOperator::Add:
      return createBinary<moore::AddOp>(lhs, rhs);
    case BinaryOperator::Subtract:
      return createBinary<moore::SubOp>(lhs, rhs);
    case BinaryOperator::Multiply:
      return createBinary<moore::MulOp>(lhs, rhs);
    case BinaryOperator::Divide:
      if (expr.type->isSigned())
        return createBinary<moore::DivSOp>(lhs, rhs);
      else
        return createBinary<moore::DivUOp>(lhs, rhs);
    case BinaryOperator::Mod:
      if (expr.type->isSigned())
        return createBinary<moore::ModSOp>(lhs, rhs);
      else
        return createBinary<moore::ModUOp>(lhs, rhs);
    case BinaryOperator::Power: {
      // Slang casts the LHS and result of the `**` operator to a four-valued
      // type, since the operator can return X even for two-valued inputs. To
      // maintain uniform types across operands and results, cast the RHS to
      // that four-valued type as well.
      auto rhsCast = context.materializeConversion(
          lhs.getType(), rhs, expr.right().type->isSigned(), rhs.getLoc());
      if (expr.type->isSigned())
        return createBinary<moore::PowSOp>(lhs, rhsCast);
      else
        return createBinary<moore::PowUOp>(lhs, rhsCast);
    }

    case BinaryOperator::BinaryAnd:
      return createBinary<moore::AndOp>(lhs, rhs);
    case BinaryOperator::BinaryOr:
      return createBinary<moore::OrOp>(lhs, rhs);
    case BinaryOperator::BinaryXor:
      return createBinary<moore::XorOp>(lhs, rhs);
    case BinaryOperator::BinaryXnor: {
      auto result = createBinary<moore::XorOp>(lhs, rhs);
      if (!result)
        return {};
      return moore::NotOp::create(builder, loc, result);
    }

    case BinaryOperator::Equality:
      // If either operand is an LTL type, build an LTL equivalence.
      if (isa<ltl::PropertyType, ltl::SequenceType>(lhs.getType()) ||
          isa<ltl::PropertyType, ltl::SequenceType>(rhs.getType())) {
        if (!isa<ltl::PropertyType, ltl::SequenceType>(lhs.getType())) {
          lhs = context.convertToI1(lhs);
          if (!lhs)
            return {};
        }
        if (!isa<ltl::PropertyType, ltl::SequenceType>(rhs.getType())) {
          rhs = context.convertToI1(rhs);
          if (!rhs)
            return {};
        }
        auto notLhs = ltl::NotOp::create(builder, loc, lhs);
        auto notRhs = ltl::NotOp::create(builder, loc, rhs);
        auto both = ltl::AndOp::create(builder, loc,
                                       SmallVector<Value, 2>{lhs, rhs});
        auto notBoth = ltl::AndOp::create(builder, loc,
                                          SmallVector<Value, 2>{notLhs, notRhs});
        return ltl::OrOp::create(builder, loc,
                                 SmallVector<Value, 2>{both, notBoth});
      }
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::eq, lhs, rhs);
      else if (isa<moore::UnpackedStructType>(lhs.getType())) {
        auto eq = buildUnpackedAggregateLogicalEq(context, loc, lhs, rhs);
        if (!eq) {
          mlir::emitError(loc)
              << "unsupported unpacked struct equality operands";
          return {};
        }
        return eq;
      }
      else if (isa<moore::OpenUnpackedArrayType>(lhs.getType()) ||
               isa<moore::OpenUnpackedArrayType>(rhs.getType())) {
        // Open array equality is not supported; return false to allow
        // compilation of UVM compare helpers.
        auto boolTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, boolTy, 0);
      }
      else if (isa<moore::StringType>(lhs.getType()) ||
               isa<moore::StringType>(rhs.getType()) ||
               isa<moore::FormatStringType>(lhs.getType()) ||
               isa<moore::FormatStringType>(rhs.getType())) {
        auto strTy = moore::StringType::get(context.getContext());
        lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(strTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::eq, lhs, rhs);
      }
      else if (isa<moore::ChandleType>(lhs.getType()) ||
               isa<moore::ChandleType>(rhs.getType())) {
        // Treat chandles as opaque pointers: compare as 64-bit integers.
        auto intTy =
            moore::IntType::get(context.getContext(), 64, Domain::TwoValued);
        lhs = context.materializeConversion(intTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(intTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::EqOp::create(builder, loc, lhs, rhs);
      } else if (isa<moore::QueueType>(lhs.getType()) ||
                 isa<moore::QueueType>(rhs.getType())) {
        // Queue equality unsupported - return false to allow compilation.
        auto boolTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, boolTy, 0);
      } else if (isa<moore::VirtualInterfaceType>(lhs.getType()) ||
                 isa<moore::VirtualInterfaceType>(rhs.getType())) {
        // Virtual interface comparison (e.g., vif == null, vif1 == vif2).
        auto lhsVifTy = dyn_cast<moore::VirtualInterfaceType>(lhs.getType());
        auto rhsVifTy = dyn_cast<moore::VirtualInterfaceType>(rhs.getType());

        // Determine the common type for comparison.
        // If one is null (represented as ClassHandleType with __null__), use the other's type.
        moore::VirtualInterfaceType commonTy;
        Value lhsVal = lhs, rhsVal = rhs;

        if (lhsVifTy && !rhsVifTy) {
          // RHS is null or incompatible type - create a null with LHS's type
          commonTy = lhsVifTy;
          rhsVal = moore::VirtualInterfaceNullOp::create(builder, loc, commonTy);
        } else if (!lhsVifTy && rhsVifTy) {
          // LHS is null or incompatible type - create a null with RHS's type
          commonTy = rhsVifTy;
          lhsVal = moore::VirtualInterfaceNullOp::create(builder, loc, commonTy);
        } else if (lhsVifTy && rhsVifTy) {
          // Both are virtual interfaces - use LHS type
          commonTy = lhsVifTy;
          if (lhsVifTy != rhsVifTy) {
            // Types differ - emit a warning and convert RHS
            mlir::emitWarning(loc)
                << "comparing virtual interfaces of different types "
                << lhsVifTy << " and " << rhsVifTy;
            rhsVal = moore::ConversionOp::create(builder, loc, commonTy, rhs);
          }
        } else {
          // Neither is a virtual interface - should not happen
          mlir::emitError(loc) << "virtual interface comparison requires at "
                               << "least one operand to be a virtual interface";
          return {};
        }

        return moore::VirtualInterfaceCmpOp::create(
            builder, loc, moore::VirtualInterfaceCmpPredicate::eq, lhsVal,
            rhsVal);
      } else if (isa<moore::ClassHandleType>(lhs.getType()) ||
               isa<moore::ClassHandleType>(rhs.getType())) {
        // Class handle comparison (e.g., obj == null, obj1 == obj2).
        auto lhsHandleTy = dyn_cast<moore::ClassHandleType>(lhs.getType());
        auto rhsHandleTy = dyn_cast<moore::ClassHandleType>(rhs.getType());

        // Both operands must be class handles for this comparison.
        if (!lhsHandleTy || !rhsHandleTy) {
          mlir::emitError(loc) << "class handle comparison requires both "
                               << "operands to be class handles";
          return {};
        }

        // Determine the common type for comparison.
        // If one is null, use the other's type. Otherwise, they should match.
        moore::ClassHandleType commonTy = lhsHandleTy;
        Value lhsVal = lhs, rhsVal = rhs;

        if (isNullHandleType(lhsHandleTy) && !isNullHandleType(rhsHandleTy)) {
          // LHS is null - create a null with RHS's type
          commonTy = rhsHandleTy;
          lhsVal = moore::ClassNullOp::create(builder, loc, commonTy);
        } else if (!isNullHandleType(lhsHandleTy) && isNullHandleType(rhsHandleTy)) {
          // RHS is null - create a null with LHS's type
          commonTy = lhsHandleTy;
          rhsVal = moore::ClassNullOp::create(builder, loc, commonTy);
        } else if (lhsHandleTy != rhsHandleTy) {
          // Neither is null but types differ - try to find common base type.
          // For now, if they're different non-null types, upcast LHS to RHS
          // or vice versa based on inheritance.
          if (context.isClassDerivedFrom(lhsHandleTy, rhsHandleTy)) {
            commonTy = rhsHandleTy;
            lhsVal = moore::ClassUpcastOp::create(builder, loc, commonTy, lhs);
          } else if (context.isClassDerivedFrom(rhsHandleTy, lhsHandleTy)) {
            commonTy = lhsHandleTy;
            rhsVal = moore::ClassUpcastOp::create(builder, loc, commonTy, rhs);
          } else {
            // Types are not related by inheritance - emit warning
            mlir::emitWarning(loc) << "comparing unrelated class handle types "
                                   << lhsHandleTy << " and " << rhsHandleTy;
            // Use ConversionOp to cast RHS to LHS type for the comparison
            commonTy = lhsHandleTy;
            rhsVal = moore::ConversionOp::create(builder, loc, commonTy, rhs);
          }
        }

        return moore::ClassHandleCmpOp::create(
            builder, loc, moore::ClassHandleCmpPredicate::eq, lhsVal, rhsVal);
      } else
        return createBinary<moore::EqOp>(lhs, rhs);
    case BinaryOperator::Inequality:
      // If either operand is an LTL type, build an LTL inequality.
      if (isa<ltl::PropertyType, ltl::SequenceType>(lhs.getType()) ||
          isa<ltl::PropertyType, ltl::SequenceType>(rhs.getType())) {
        if (!isa<ltl::PropertyType, ltl::SequenceType>(lhs.getType())) {
          lhs = context.convertToI1(lhs);
          if (!lhs)
            return {};
        }
        if (!isa<ltl::PropertyType, ltl::SequenceType>(rhs.getType())) {
          rhs = context.convertToI1(rhs);
          if (!rhs)
            return {};
        }
        auto notLhs = ltl::NotOp::create(builder, loc, lhs);
        auto notRhs = ltl::NotOp::create(builder, loc, rhs);
        auto lhsNotRhs = ltl::AndOp::create(builder, loc,
                                            SmallVector<Value, 2>{lhs, notRhs});
        auto rhsNotLhs = ltl::AndOp::create(builder, loc,
                                            SmallVector<Value, 2>{notLhs, rhs});
        return ltl::OrOp::create(builder, loc,
                                 SmallVector<Value, 2>{lhsNotRhs, rhsNotLhs});
      }
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::ne, lhs, rhs);
      else if (isa<moore::UnpackedStructType>(lhs.getType())) {
        auto eq = buildUnpackedAggregateLogicalEq(context, loc, lhs, rhs);
        if (!eq) {
          mlir::emitError(loc)
              << "unsupported unpacked struct inequality operands";
          return {};
        }
        return moore::NotOp::create(builder, loc, eq);
      }
      else if (isa<moore::OpenUnpackedArrayType>(lhs.getType()) ||
               isa<moore::OpenUnpackedArrayType>(rhs.getType())) {
        // Open array inequality is not supported; return true to allow
        // compilation of UVM compare helpers.
        auto boolTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, boolTy, 1);
      }
      else if (isa<moore::StringType>(lhs.getType()) ||
               isa<moore::StringType>(rhs.getType()) ||
               isa<moore::FormatStringType>(lhs.getType()) ||
               isa<moore::FormatStringType>(rhs.getType())) {
        auto strTy = moore::StringType::get(context.getContext());
        lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(strTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ne, lhs, rhs);
      }
      else if (isa<moore::ChandleType>(lhs.getType()) ||
               isa<moore::ChandleType>(rhs.getType())) {
        auto intTy =
            moore::IntType::get(context.getContext(), 64, Domain::TwoValued);
        lhs = context.materializeConversion(intTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(intTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::NeOp::create(builder, loc, lhs, rhs);
      } else if (isa<moore::QueueType>(lhs.getType()) ||
                 isa<moore::QueueType>(rhs.getType())) {
        auto boolTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, boolTy, 1);
      } else if (isa<moore::VirtualInterfaceType>(lhs.getType()) ||
                 isa<moore::VirtualInterfaceType>(rhs.getType())) {
        // Virtual interface comparison (e.g., vif != null, vif1 != vif2).
        auto lhsVifTy = dyn_cast<moore::VirtualInterfaceType>(lhs.getType());
        auto rhsVifTy = dyn_cast<moore::VirtualInterfaceType>(rhs.getType());

        // Determine the common type for comparison.
        // If one is null (represented as ClassHandleType with __null__), use the other's type.
        moore::VirtualInterfaceType commonTy;
        Value lhsVal = lhs, rhsVal = rhs;

        if (lhsVifTy && !rhsVifTy) {
          // RHS is null or incompatible type - create a null with LHS's type
          commonTy = lhsVifTy;
          rhsVal = moore::VirtualInterfaceNullOp::create(builder, loc, commonTy);
        } else if (!lhsVifTy && rhsVifTy) {
          // LHS is null or incompatible type - create a null with RHS's type
          commonTy = rhsVifTy;
          lhsVal = moore::VirtualInterfaceNullOp::create(builder, loc, commonTy);
        } else if (lhsVifTy && rhsVifTy) {
          // Both are virtual interfaces - use LHS type
          commonTy = lhsVifTy;
          if (lhsVifTy != rhsVifTy) {
            // Types differ - emit a warning and convert RHS
            mlir::emitWarning(loc)
                << "comparing virtual interfaces of different types "
                << lhsVifTy << " and " << rhsVifTy;
            rhsVal = moore::ConversionOp::create(builder, loc, commonTy, rhs);
          }
        } else {
          // Neither is a virtual interface - should not happen
          mlir::emitError(loc) << "virtual interface comparison requires at "
                               << "least one operand to be a virtual interface";
          return {};
        }

        return moore::VirtualInterfaceCmpOp::create(
            builder, loc, moore::VirtualInterfaceCmpPredicate::ne, lhsVal,
            rhsVal);
      } else if (isa<moore::ClassHandleType>(lhs.getType()) ||
               isa<moore::ClassHandleType>(rhs.getType())) {
        // Class handle comparison (e.g., obj != null, obj1 != obj2).
        auto lhsHandleTy = dyn_cast<moore::ClassHandleType>(lhs.getType());
        auto rhsHandleTy = dyn_cast<moore::ClassHandleType>(rhs.getType());

        // Both operands must be class handles for this comparison.
        if (!lhsHandleTy || !rhsHandleTy) {
          mlir::emitError(loc) << "class handle comparison requires both "
                               << "operands to be class handles";
          return {};
        }

        // Determine the common type for comparison.
        // If one is null, use the other's type. Otherwise, they should match.
        moore::ClassHandleType commonTy = lhsHandleTy;
        Value lhsVal = lhs, rhsVal = rhs;

        if (isNullHandleType(lhsHandleTy) && !isNullHandleType(rhsHandleTy)) {
          // LHS is null - create a null with RHS's type
          commonTy = rhsHandleTy;
          lhsVal = moore::ClassNullOp::create(builder, loc, commonTy);
        } else if (!isNullHandleType(lhsHandleTy) && isNullHandleType(rhsHandleTy)) {
          // RHS is null - create a null with LHS's type
          commonTy = lhsHandleTy;
          rhsVal = moore::ClassNullOp::create(builder, loc, commonTy);
        } else if (lhsHandleTy != rhsHandleTy) {
          // Neither is null but types differ - try to find common base type.
          // For now, if they're different non-null types, upcast LHS to RHS
          // or vice versa based on inheritance.
          if (context.isClassDerivedFrom(lhsHandleTy, rhsHandleTy)) {
            commonTy = rhsHandleTy;
            lhsVal = moore::ClassUpcastOp::create(builder, loc, commonTy, lhs);
          } else if (context.isClassDerivedFrom(rhsHandleTy, lhsHandleTy)) {
            commonTy = lhsHandleTy;
            rhsVal = moore::ClassUpcastOp::create(builder, loc, commonTy, rhs);
          } else {
            // Types are not related by inheritance - emit warning
            mlir::emitWarning(loc) << "comparing unrelated class handle types "
                                   << lhsHandleTy << " and " << rhsHandleTy;
            // Use ConversionOp to cast RHS to LHS type for the comparison
            commonTy = lhsHandleTy;
            rhsVal = moore::ConversionOp::create(builder, loc, commonTy, rhs);
          }
        }

        return moore::ClassHandleCmpOp::create(
            builder, loc, moore::ClassHandleCmpPredicate::ne, lhsVal, rhsVal);
      } else
        return createBinary<moore::NeOp>(lhs, rhs);
    case BinaryOperator::CaseEquality:
      // Handle string comparisons with CaseEquality (===)
      if (isa<moore::StringType>(lhs.getType()) ||
          isa<moore::StringType>(rhs.getType()) ||
          isa<moore::FormatStringType>(lhs.getType()) ||
          isa<moore::FormatStringType>(rhs.getType())) {
        auto strTy = moore::StringType::get(context.getContext());
        lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(strTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::eq, lhs, rhs);
      }
      return createBinary<moore::CaseEqOp>(lhs, rhs);
    case BinaryOperator::CaseInequality:
      // Handle string comparisons with CaseInequality (!==)
      if (isa<moore::StringType>(lhs.getType()) ||
          isa<moore::StringType>(rhs.getType()) ||
          isa<moore::FormatStringType>(lhs.getType()) ||
          isa<moore::FormatStringType>(rhs.getType())) {
        auto strTy = moore::StringType::get(context.getContext());
        lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
        rhs = context.materializeConversion(strTy, rhs, false, rhs.getLoc());
        if (!lhs || !rhs)
          return {};
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ne, lhs, rhs);
      }
      return createBinary<moore::CaseNeOp>(lhs, rhs);
    case BinaryOperator::WildcardEquality:
      return createBinary<moore::WildcardEqOp>(lhs, rhs);
    case BinaryOperator::WildcardInequality:
      return createBinary<moore::WildcardNeOp>(lhs, rhs);

    case BinaryOperator::GreaterThanEqual:
      if (expr.left().type->isSigned())
        return createBinary<moore::SgeOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ge, lhs, rhs);
      else
        return createBinary<moore::UgeOp>(lhs, rhs);
    case BinaryOperator::GreaterThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SgtOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::gt, lhs, rhs);
      else
        return createBinary<moore::UgtOp>(lhs, rhs);
    case BinaryOperator::LessThanEqual:
      if (expr.left().type->isSigned())
        return createBinary<moore::SleOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::le, lhs, rhs);
      else
        return createBinary<moore::UleOp>(lhs, rhs);
    case BinaryOperator::LessThan:
      if (expr.left().type->isSigned())
        return createBinary<moore::SltOp>(lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::lt, lhs, rhs);
      else
        return createBinary<moore::UltOp>(lhs, rhs);

    case BinaryOperator::LogicalAnd:
    case BinaryOperator::LogicalOr:
    case BinaryOperator::LogicalImplication:
    case BinaryOperator::LogicalEquivalence:
      return buildLogicalBOp(expr.op, lhs, rhs, domain);

    case BinaryOperator::LogicalShiftLeft:
      return createBinary<moore::ShlOp>(lhs, rhs);
    case BinaryOperator::LogicalShiftRight:
      return createBinary<moore::ShrOp>(lhs, rhs);
    case BinaryOperator::ArithmeticShiftLeft:
      return createBinary<moore::ShlOp>(lhs, rhs);
    case BinaryOperator::ArithmeticShiftRight: {
      // The `>>>` operator is an arithmetic right shift if the LHS operand is
      // signed, or a logical right shift if the operand is unsigned.
      lhs = context.convertToSimpleBitVector(lhs);
      rhs = context.convertToSimpleBitVector(rhs);
      if (!lhs || !rhs)
        return {};
      if (expr.type->isSigned())
        return moore::AShrOp::create(builder, loc, lhs, rhs);
      return moore::ShrOp::create(builder, loc, lhs, rhs);
    }
    }

    mlir::emitError(loc, "unsupported binary operator");
    return {};
  }

  // Handle `'0`, `'1`, `'x`, and `'z` literals.
  Value visit(const slang::ast::UnbasedUnsizedIntegerLiteral &expr) {
    return context.materializeSVInt(expr.getValue(), *expr.type, loc);
  }

  // Handle integer literals.
  Value visit(const slang::ast::IntegerLiteral &expr) {
    return context.materializeSVInt(expr.getValue(), *expr.type, loc);
  }

  // Handle time literals.
  Value visit(const slang::ast::TimeLiteral &expr) {
    // The time literal is expressed in the current time scale. Determine the
    // conversion factor to convert the literal from the current time scale into
    // femtoseconds, and round the scaled value to femtoseconds.
    double scale = getTimeScaleInFemtoseconds(context);
    double value = std::round(expr.getValue() * scale);
    assert(value >= 0.0);

    // Check that the value does not exceed what we can represent in the IR.
    // Casting the maximum uint64 value to double changes its value from
    // 18446744073709551615 to 18446744073709551616, which makes the comparison
    // overestimate the largest number we can represent. To avoid this, round
    // the maximum value down to the closest number that only has the front 53
    // bits set. This matches the mantissa of a double, plus the implicit
    // leading 1, ensuring that we can accurately represent the limit.
    static constexpr uint64_t limit =
        (std::numeric_limits<uint64_t>::max() >> 11) << 11;
    if (value > limit) {
      mlir::emitError(loc) << "time value is larger than " << limit << " fs";
      return {};
    }

    return moore::ConstantTimeOp::create(builder, loc,
                                         static_cast<uint64_t>(value));
  }

  // Handle replications.
  Value visit(const slang::ast::ReplicationExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = context.convertRvalueExpression(expr.concat());
    if (!value)
      return {};

    // String replication {n{str}} produces a string
    if (isa<moore::StringType>(type)) {
      // Get the replication count
      auto countExpr = context.convertRvalueExpression(expr.count());
      if (!countExpr)
        return {};
      // Convert count to i32 if needed
      auto countType = moore::IntType::getInt(context.getContext(), 32);
      countExpr = context.convertToSimpleBitVector(
          context.convertRvalueExpression(expr.count(), countType));
      if (!countExpr)
        return {};
      return moore::StringReplicateOp::create(builder, loc, countExpr, value);
    }

    // ReplicateOp requires an IntType operand, so convert to simple bit vector
    value = context.convertToSimpleBitVector(value);
    if (!value)
      return {};
    return moore::ReplicateOp::create(builder, loc, type, value);
  }

  // Handle set membership operator.
  Value visit(const slang::ast::InsideExpression &expr) {
    auto lhs = context.convertRvalueExpression(expr.left());
    if (!lhs)
      return {};

    // String-based inside checks: compare lhs against each element.
    if (isa<moore::StringType>(lhs.getType()) ||
        isa<moore::FormatStringType>(lhs.getType())) {
      auto strTy = moore::StringType::get(context.getContext());
      lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
      if (!lhs)
        return {};

      SmallVector<Value> conditions;
      for (const auto *listExpr : expr.rangeList()) {
        if (listExpr->as_if<slang::ast::ValueRangeExpression>()) {
          mlir::emitError(loc)
              << "string 'inside' does not support ranges";
          return {};
        }
        auto value = context.convertRvalueExpression(*listExpr);
        if (!value)
          return {};
        value = context.materializeConversion(strTy, value, false, value.getLoc());
        if (!value)
          return {};
        conditions.push_back(moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::eq, lhs, value));
      }

      // Combine with OR.
      auto result = conditions.back();
      conditions.pop_back();
      while (!conditions.empty()) {
        result = moore::OrOp::create(builder, loc, conditions.back(), result);
        conditions.pop_back();
      }
      return result;
    }

    lhs = context.convertToSimpleBitVector(lhs);
    if (!lhs)
      return {};
    // All conditions for determining whether it is inside.
    SmallVector<Value> conditions;

    // Traverse open range list.
    for (const auto *listExpr : expr.rangeList()) {
      Value cond;
      // The open range list on the right-hand side of the inside operator is a
      // comma-separated list of expressions or ranges.
      if (const auto *openRange =
              listExpr->as_if<slang::ast::ValueRangeExpression>()) {
        // Handle ranges.
        auto lowBound = context.convertToSimpleBitVector(
            context.convertRvalueExpression(openRange->left()));
        auto highBound = context.convertToSimpleBitVector(
            context.convertRvalueExpression(openRange->right()));
        if (!lowBound || !highBound)
          return {};
        Value leftValue, rightValue;
        // Determine if the expression on the left-hand side is inclusively
        // within the range.
        if (openRange->left().type->isSigned() ||
            expr.left().type->isSigned()) {
          leftValue = moore::SgeOp::create(builder, loc, lhs, lowBound);
        } else {
          leftValue = moore::UgeOp::create(builder, loc, lhs, lowBound);
        }
        if (openRange->right().type->isSigned() ||
            expr.left().type->isSigned()) {
          rightValue = moore::SleOp::create(builder, loc, lhs, highBound);
        } else {
          rightValue = moore::UleOp::create(builder, loc, lhs, highBound);
        }
        cond = moore::AndOp::create(builder, loc, leftValue, rightValue);
      } else {
        // Handle expressions.
        if (listExpr->kind == slang::ast::ExpressionKind::DataType) {
          const auto &canonicalType = listExpr->type->getCanonicalType();
          if (canonicalType.kind == slang::ast::SymbolKind::EnumType) {
            const auto &enumType =
                static_cast<const slang::ast::EnumType &>(canonicalType);
            auto lhsType = cast<moore::IntType>(lhs.getType());
            SmallVector<Value> enumConds;
            for (const auto &member : enumType.values()) {
              auto cv = member.getValue();
              if (cv.bad() || !cv.isInteger())
                continue;
              auto fv = convertSVIntToFVInt(cv.integer());
              if (fv.getBitWidth() != lhsType.getWidth())
                fv = fv.zext(lhsType.getWidth());
              auto enumConst =
                  moore::ConstantOp::create(builder, loc, lhsType, fv);
              enumConds.push_back(
                  moore::WildcardEqOp::create(builder, loc, lhs, enumConst));
            }
            if (enumConds.empty()) {
              mlir::emitError(loc) << "enum type has no values";
              return {};
            }
            Value enumResult = enumConds.back();
            enumConds.pop_back();
            while (!enumConds.empty()) {
              enumResult = moore::OrOp::create(builder, loc,
                                               enumConds.back(), enumResult);
              enumConds.pop_back();
            }
            conditions.push_back(enumResult);
            continue;
          }
        }
        if (!listExpr->type->isIntegral()) {
          if (listExpr->type->isUnpackedArray()) {
            // Handle unpacked arrays by checking if lhs is contained in the
            // array. This generates a moore.array.contains operation.
            auto arrayValue = context.convertRvalueExpression(*listExpr);
            if (!arrayValue)
              return {};
            cond = moore::ArrayContainsOp::create(builder, loc, arrayValue, lhs);
            conditions.push_back(cond);
            continue;
          }
          mlir::emitError(
              loc, "only simple bit vectors supported in 'inside' expressions");
          return {};
        }

        auto value = context.convertToSimpleBitVector(
            context.convertRvalueExpression(*listExpr));
        if (!value)
          return {};
        cond = moore::WildcardEqOp::create(builder, loc, lhs, value);
      }
      conditions.push_back(cond);
    }

    // Calculate the final result by `or` op.
    auto result = conditions.back();
    conditions.pop_back();
    while (!conditions.empty()) {
      result = moore::OrOp::create(builder, loc, conditions.back(), result);
      conditions.pop_back();
    }
    return result;
  }

  // Handle conditional operator `?:`.
  Value visit(const slang::ast::ConditionalExpression &expr) {
    auto type = context.convertType(*expr.type);

    // Handle condition.
    if (expr.conditions.size() > 1) {
      mlir::emitError(loc)
          << "unsupported conditional expression with more than one condition";
      return {};
    }
    const auto &cond = expr.conditions[0];
    Value value;
    if (cond.pattern) {
      // Handle pattern matching condition (e.g., "x matches tagged a '{...}")
      auto exprValue = context.convertRvalueExpression(*cond.expr);
      if (!exprValue)
        return {};
      auto patternMatch = context.matchPattern(
          *cond.pattern, exprValue, *cond.expr->type,
          slang::ast::CaseStatementCondition::Normal, loc);
      if (failed(patternMatch))
        return {};
      value = context.convertToBool(*patternMatch);
    } else {
      value = context.convertToBool(context.convertRvalueExpression(*cond.expr));
    }
    if (!value)
      return {};
    auto conditionalOp =
        moore::ConditionalOp::create(builder, loc, type, value);

    // Create blocks for true region and false region.
    auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
    auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);

    // Handle left expression.
    builder.setInsertionPointToStart(&trueBlock);
    auto trueValue = context.convertRvalueExpression(expr.left(), type);
    if (!trueValue)
      return {};
    moore::YieldOp::create(builder, loc, trueValue);

    // Handle right expression.
    builder.setInsertionPointToStart(&falseBlock);
    auto falseValue = context.convertRvalueExpression(expr.right(), type);
    if (!falseValue)
      return {};
    moore::YieldOp::create(builder, loc, falseValue);

    return conditionalOp.getResult();
  }

  /// Handle calls.
  Value visit(const slang::ast::CallExpression &expr) {
    // Try to materialize constant values directly.
    // Skip constant evaluation for system calls that need runtime behavior
    // ($test$plusargs, $value$plusargs) since slang returns nullptr for them
    // and we want to emit runtime calls instead.
    bool skipConstEval = false;
    if (auto *sci = std::get_if<slang::ast::CallExpression::SystemCallInfo>(
            &expr.subroutine)) {
      if (sci->subroutine->name == "$test$plusargs" ||
          sci->subroutine->name == "$value$plusargs" ||
          sci->subroutine->name == "$initstate")
        skipConstEval = true;
    }
    if (!skipConstEval) {
      auto constant = context.evaluateConstant(expr);
      if (auto value =
              context.materializeConstant(constant, *expr.type, loc))
        return value;
    }

    return std::visit(
        [&](auto &subroutine) { return visitCall(expr, subroutine); },
        expr.subroutine);
  }

  /// Check if a call expression is a super.method() call.
  /// Checks both the call expression's syntax and the receiver expression
  /// for the super keyword.
  static bool isSuperCall(const slang::ast::CallExpression &callExpr) {
    // Check the call expression's own syntax for super keyword.
    // For `super.method()`, the syntax should be a MemberAccessExpressionSyntax
    // where the value part contains the super keyword.
    if (callExpr.syntax) {
      auto firstToken = callExpr.syntax->getFirstToken();
      if (firstToken.kind == slang::parsing::TokenKind::SuperKeyword)
        return true;
    }

    // Check the receiver expression (thisClass) for super reference.
    const slang::ast::Expression *recvExpr = callExpr.thisClass();
    if (!recvExpr)
      return false;

    // Check for super syntax token in the receiver.
    if (recvExpr->syntax) {
      auto firstToken = recvExpr->syntax->getFirstToken();
      if (firstToken.kind == slang::parsing::TokenKind::SuperKeyword)
        return true;
    }

    // Check if the expression is a conversion from `this` to a parent class.
    // For `super.method()`, slang creates a conversion from `this` to the
    // parent class type.
    if (const auto *conv =
            recvExpr->as_if<slang::ast::ConversionExpression>()) {
      // Check the operand's syntax for super keyword
      const auto &operand = conv->operand();
      if (operand.syntax) {
        auto firstToken = operand.syntax->getFirstToken();
        if (firstToken.kind == slang::parsing::TokenKind::SuperKeyword)
          return true;
      }

      // Also detect implicit conversion from this to parent class type.
      // This happens for super.method() calls where slang converts this
      // to the parent class type.
      auto *fromClass =
          operand.type->getCanonicalType().as_if<slang::ast::ClassType>();
      auto *toClass =
          conv->type->getCanonicalType().as_if<slang::ast::ClassType>();
      if (fromClass && toClass && fromClass != toClass) {
        // It's a conversion from one class type to another.
        // Check if toClass is actually a base class of fromClass.
        auto *baseClass = fromClass->getBaseClass();
        if (baseClass && baseClass == toClass) {
          return true;
        }
      }
    }

    return false;
  }

  /// Get both the actual `this` argument of a method call and the required
  /// class type. Also indicates if this is a super call.
  std::tuple<Value, moore::ClassHandleType, bool>
  getMethodReceiverTypeHandle(const slang::ast::CallExpression &expr) {

    moore::ClassHandleType handleTy;
    Value thisRef;
    bool superCall = isSuperCall(expr);

    // Qualified call: t.m(...), extract from thisClass.
    if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
      thisRef = context.convertRvalueExpression(*recvExpr);
      if (!thisRef)
        return {};
    } else if (context.methodReceiverOverride) {
      // Use an explicit receiver override (e.g., during constructor calls).
      thisRef = context.methodReceiverOverride;
    } else {
      // Unqualified call inside a method body: try using implicit %this.
      thisRef = context.getImplicitThisRef();
      if (!thisRef) {
        mlir::emitError(loc) << "method '" << expr.getSubroutineName()
                             << "' called without an object";
        return {};
      }
    }
    handleTy = cast<moore::ClassHandleType>(thisRef.getType());
    return {thisRef, handleTy, superCall};
  }

  /// Build a method call including implicit this argument.
  /// If isSuperCall is true, bypasses virtual dispatch and calls the method
  /// directly (as `super.method()` should do in SystemVerilog).
  mlir::CallOpInterface
  buildMethodCall(const slang::ast::SubroutineSymbol *subroutine,
                  FunctionLowering *lowering,
                  moore::ClassHandleType actualHandleTy, Value actualThisRef,
                  SmallVector<Value> &arguments,
                  SmallVector<Type> &resultTypes, bool isSuperCall = false) {

    // Get the expected receiver type from the lowered method
    auto funcTy = lowering->op.getFunctionType();
    auto expected0 = funcTy.getInput(0);
    auto expectedHdlTy = cast<moore::ClassHandleType>(expected0);

    LLVM_DEBUG({
      llvm::dbgs() << "buildMethodCall: method " << subroutine->name
                   << ", actualHandleTy: " << actualHandleTy
                   << ", expectedHdlTy: " << expectedHdlTy
                   << ", function: " << lowering->op.getSymName()
                   << ", isSuperCall: " << isSuperCall << "\n";
    });

    // Upcast the handle as necessary.
    auto implicitThisRef = context.materializeConversion(
        expectedHdlTy, actualThisRef, false, actualThisRef.getLoc());

    // Build an argument list where the this reference is the first argument.
    SmallVector<Value> explicitArguments;
    explicitArguments.reserve(arguments.size() + 1);
    explicitArguments.push_back(implicitThisRef);
    explicitArguments.append(arguments.begin(), arguments.end());

    // Method call: choose direct vs virtual.
    // For super calls, always use direct dispatch to call the parent's
    // implementation directly, bypassing virtual dispatch.
    // Use isVirtual() to catch implicit virtuality (overriding base class).
    const bool isVirtual = subroutine->isVirtual();

    if (!isVirtual || isSuperCall) {
      auto calleeSym = lowering->op.getSymName();
      // Direct (non-virtual) call -> func.call
      return mlir::func::CallOp::create(builder, loc, resultTypes, calleeSym,
                                        explicitArguments);
    }

    auto funcName = subroutine->name;
    auto method = moore::VTableLoadMethodOp::create(
        builder, loc, funcTy, actualThisRef,
        SymbolRefAttr::get(context.getContext(), funcName));
    return mlir::func::CallIndirectOp::create(builder, loc, method,
                                              explicitArguments);
  }

  /// Handle subroutine calls.
  Value visitCall(const slang::ast::CallExpression &expr,
                  const slang::ast::SubroutineSymbol *subroutine) {

    // Save any temporary receiver override; we'll clear it after computing the
    // current call's receiver so nested calls during argument evaluation use
    // the caller's implicit `this`.
    auto savedReceiverOverride = context.methodReceiverOverride;
    auto restoreReceiverOverride =
        llvm::make_scope_exit(
            [&] { context.methodReceiverOverride = savedReceiverOverride; });

    // DPI-C imports are handled by generating normal function calls to
    // stub implementations provided by the MooreRuntime library. The
    // functions are declared as external and linked at compile time.
    if (subroutine->flags & slang::ast::MethodFlags::DPIImport) {
      mlir::emitRemark(loc) << "DPI-C import '" << subroutine->name
                            << "' will use runtime stub (link with MooreRuntime)";
      // Fall through to normal call generation below
    }

    // A subroutine is a method if it has a thisVar (normal methods) or if it's
    // a virtual method (including pure virtual methods, which may not have
    // thisVar set in slang because they have no body in the abstract class).
    // Use isVirtual() to catch implicit virtuality.
    const bool isMethod = (subroutine->thisVar != nullptr) ||
                          subroutine->isVirtual();

    if (subroutine->name == "rand_mode" ||
        subroutine->name == "constraint_mode") {
      const bool isRandMode = subroutine->name == "rand_mode";
      const slang::ast::Expression *receiverExpr = nullptr;
      std::optional<mlir::FlatSymbolRefAttr> memberAttr;

      if (expr.syntax && context.currentScope) {
        if (auto *invocation = expr.syntax->as_if<
                slang::syntax::InvocationExpressionSyntax>()) {
          if (auto *memberAccess = invocation->left->as_if<
                  slang::syntax::MemberAccessExpressionSyntax>()) {
            const slang::syntax::ExpressionSyntax *baseSyntax =
                memberAccess->left;
            slang::ast::ASTContext astContext(*context.currentScope,
                                              slang::ast::LookupLocation::max);
            const auto &baseExpr =
                slang::ast::Expression::bind(*baseSyntax, astContext);
            if (!baseExpr.bad()) {
              if (auto *memberExpr =
                      baseExpr.as_if<slang::ast::MemberAccessExpression>()) {
                memberAttr = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), memberExpr->member.name);
                receiverExpr = &memberExpr->value();
              } else {
                receiverExpr = &baseExpr;
              }
            }
          }
        }
      }

      if (!receiverExpr) {
        if (const auto *thisClass = expr.thisClass())
          receiverExpr = thisClass;
      }
      if (receiverExpr) {
        if (auto *memberExpr =
                receiverExpr->as_if<slang::ast::MemberAccessExpression>()) {
          if (!memberAttr)
            memberAttr = mlir::FlatSymbolRefAttr::get(
                builder.getContext(), memberExpr->member.name);
          receiverExpr = &memberExpr->value();
        }
      }

      if (!receiverExpr) {
        mlir::emitError(loc)
            << (isRandMode ? "rand_mode" : "constraint_mode")
            << "() requires a class object receiver";
        return {};
      }

      Value classObj;
      if (auto *namedExpr =
              receiverExpr->as_if<slang::ast::NamedValueExpression>()) {
        if (isRandMode) {
          if (auto *property =
                  namedExpr->symbol.as_if<slang::ast::ClassPropertySymbol>()) {
            if (!memberAttr) {
              memberAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                        property->name);
              classObj = context.getInlineConstraintThisRef();
              if (!classObj)
                classObj = context.getImplicitThisRef();
            }
          }
        } else {
          if (auto *constraint = namedExpr->symbol.as_if<
                  slang::ast::ConstraintBlockSymbol>()) {
            if (!memberAttr) {
              memberAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                        constraint->name);
              classObj = context.getInlineConstraintThisRef();
              if (!classObj)
                classObj = context.getImplicitThisRef();
            }
          }
        }
      }

      if (!classObj) {
        classObj = context.convertRvalueExpression(*receiverExpr);
        if (!classObj)
          return {};
      }

      auto classHandleTy =
          dyn_cast<moore::ClassHandleType>(classObj.getType());
      if (!classHandleTy) {
        mlir::emitError(loc)
            << (isRandMode ? "rand_mode" : "constraint_mode")
            << "() requires a class object, got " << classObj.getType();
        return {};
      }

      Value modeValue;
      if (!expr.arguments().empty()) {
        modeValue = context.convertRvalueExpression(*expr.arguments()[0]);
        if (!modeValue)
          return {};
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        modeValue = context.materializeConversion(
            intTy, modeValue, expr.arguments()[0]->type->isSigned(), loc);
        if (!modeValue)
          return {};
      }

      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      Value result;
      if (isRandMode) {
        auto randOp = moore::RandModeOp::create(
            builder, loc, intTy, classObj,
            memberAttr ? *memberAttr : mlir::FlatSymbolRefAttr{}, modeValue);
        result = randOp.getResult();
      } else {
        auto constraintOp = moore::ConstraintModeOp::create(
            builder, loc, intTy, classObj,
            memberAttr ? *memberAttr : mlir::FlatSymbolRefAttr{}, modeValue);
        result = constraintOp.getResult();
      }

      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};
      return context.materializeConversion(resultType, result, false, loc);
    }

    // Handle covergroup method calls (sample, get_coverage, etc.)
    // IEEE 1800-2017 Section 19.8 "Covergroup methods"
    const auto &parentSym = subroutine->getParentScope()->asSymbol();
    if (parentSym.kind == slang::ast::SymbolKind::CovergroupBody) {
      // Get the covergroup instance from the call expression
      Value covergroupInstance;
      if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
        covergroupInstance = context.convertRvalueExpression(*recvExpr);
        if (!covergroupInstance)
          return {};
      }

      if (!covergroupInstance) {
        mlir::emitError(loc)
            << "covergroup method call requires covergroup instance";
        return {};
      }

      // Verify the type is a covergroup handle
      auto covergroupTy =
          dyn_cast<moore::CovergroupHandleType>(covergroupInstance.getType());
      if (!covergroupTy) {
        mlir::emitError(loc)
            << "covergroup method call requires covergroup handle, got "
            << covergroupInstance.getType();
        return {};
      }

      if (subroutine->name == "sample") {
        // sample() triggers sampling of the covergroup
        SmallVector<Value> sampleArgs;
        SmallVector<Value> iffArgs;
        // Helper lambda to collect iff conditions from a covergroup body.
        // Only populates iffArgs if at least one coverpoint has an iff
        // expression. For coverpoints without iff in a mixed group, uses
        // constant true. Returns false on conversion failure.
        auto collectIffConditions =
            [&](const slang::ast::CovergroupBodySymbol &cgBody) -> bool {
              // First pass: check if any coverpoint has an iff expression.
              bool hasAnyIff = false;
              for (const auto &member : cgBody.members()) {
                if (auto *cp =
                        member.as_if<slang::ast::CoverpointSymbol>()) {
                  if (cp->getIffExpr()) {
                    hasAnyIff = true;
                    break;
                  }
                }
              }
              if (!hasAnyIff)
                return true; // success, but no iff conditions needed

              // Second pass: evaluate iff conditions for all coverpoints.
              for (const auto &member : cgBody.members()) {
                if (auto *cp =
                        member.as_if<slang::ast::CoverpointSymbol>()) {
                  if (const auto *iffExpr = cp->getIffExpr()) {
                    Value iffVal =
                        context.convertRvalueExpression(*iffExpr);
                    if (!iffVal)
                      return false;
                    iffVal = context.convertToBool(iffVal,
                        moore::Domain::TwoValued);
                    if (!iffVal)
                      return false;
                    iffArgs.push_back(iffVal);
                  } else {
                    // No iff on this coverpoint but others have iff -
                    // use constant true (always sample).
                    auto i1Ty =
                        moore::IntType::getInt(context.getContext(), 1);
                    iffArgs.push_back(
                        moore::ConstantOp::create(builder, loc, i1Ty, 1));
                  }
                }
              }
              return true;
            };

        if (expr.arguments().empty()) {
          // Implicit sampling: evaluate each coverpoint's expression at the
          // call site. IEEE 1800-2017 Section 19.8.1 - when sample() is
          // called without arguments, each coverpoint's expression is
          // evaluated to obtain the current value for coverage tracking.
          const auto &cgBody =
              parentSym.as<slang::ast::CovergroupBodySymbol>();
          for (const auto &member : cgBody.members()) {
            if (auto *cp =
                    member.as_if<slang::ast::CoverpointSymbol>()) {
              Value val =
                  context.convertRvalueExpression(cp->getCoverageExpr());
              if (!val)
                return {};
              sampleArgs.push_back(val);
            }
          }
          // Collect iff conditions (only if any coverpoint has iff).
          if (!collectIffConditions(cgBody))
            return {};
        } else {
          // Explicit sample arguments (parametric covergroup with
          // `with function sample(...)`). IEEE 1800-2017 §19.8.1.
          // We must evaluate each coverpoint expression with the sample
          // method's formal parameters bound to the actual arguments.
          const auto &cgBody =
              parentSym.as<slang::ast::CovergroupBodySymbol>();

          // Find the sample SubroutineSymbol to get formal parameters.
          const slang::ast::SubroutineSymbol *sampleMethod = nullptr;
          for (const auto &member : cgBody.members()) {
            if (auto *sub =
                    member.as_if<slang::ast::SubroutineSymbol>()) {
              if (sub->name == "sample") {
                sampleMethod = sub;
                break;
              }
            }
          }

          if (sampleMethod) {
            auto formals = sampleMethod->getArguments();

            // Convert actual argument values.
            SmallVector<Value> actualValues;
            for (const auto *arg : expr.arguments()) {
              Value argVal = context.convertRvalueExpression(*arg);
              if (!argVal)
                return {};
              actualValues.push_back(argVal);
            }

            // Collect all FormalArgumentSymbol references from coverpoint
            // expressions. We need to bind the ACTUAL symbol pointers that
            // the expressions reference (which may differ from the
            // subroutine's getArguments() copies due to copyArg()).
            llvm::DenseMap<llvm::StringRef,
                           const slang::ast::ValueSymbol *>
                formalsByName;
            for (const auto &member : cgBody.members()) {
              if (auto *cp =
                      member.as_if<slang::ast::CoverpointSymbol>()) {
                cp->getCoverageExpr().visitSymbolReferences(
                    [&](const slang::ast::Expression &,
                        const slang::ast::Symbol &sym) {
                      if (sym.kind ==
                          slang::ast::SymbolKind::FormalArgument)
                        formalsByName[sym.name] =
                            static_cast<const slang::ast::ValueSymbol *>(
                                &sym);
                    });
              }
            }

            // Create a scope and bind each formal parameter to its
            // actual argument value, using name matching to handle
            // symbol pointer differences between copies.
            Context::ValueSymbolScope sampleScope(context.valueSymbols);
            for (size_t i = 0;
                 i < formals.size() && i < actualValues.size(); ++i) {
              // Bind the subroutine's formal (for direct references).
              context.valueSymbols.insert(formals[i], actualValues[i]);
              // Also bind the expression's actual symbol reference
              // (may be a different pointer for the same formal).
              auto it = formalsByName.find(formals[i]->name);
              if (it != formalsByName.end() &&
                  it->second != formals[i])
                context.valueSymbols.insert(it->second, actualValues[i]);
            }

            // Now evaluate each coverpoint expression in this scope.
            for (const auto &member : cgBody.members()) {
              if (auto *cp =
                      member.as_if<slang::ast::CoverpointSymbol>()) {
                Value val =
                    context.convertRvalueExpression(cp->getCoverageExpr());
                if (!val)
                  return {};
                sampleArgs.push_back(val);
              }
            }
            // Collect iff conditions (only if any coverpoint has iff).
            if (!collectIffConditions(cgBody))
              return {};
          } else {
            // Fallback: no sample method found, pass raw args.
            for (const auto *arg : expr.arguments()) {
              Value argVal = context.convertRvalueExpression(*arg);
              if (!argVal)
                return {};
              sampleArgs.push_back(argVal);
            }
          }
        }
        moore::CovergroupSampleOp::create(builder, loc, covergroupInstance,
                                          sampleArgs, iffArgs);
        // sample() returns void, but we need to return something for
        // expression context. Return a constant 0.
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        return moore::ConstantOp::create(builder, loc, intTy, 0);
      }

      if (subroutine->name == "get_coverage") {
        // get_coverage() returns the coverage percentage as a real (64-bit)
        auto realTy =
            moore::RealType::get(context.getContext(), moore::RealWidth::f64);
        return moore::CovergroupGetCoverageOp::create(builder, loc, realTy,
                                                      covergroupInstance);
      }

      if (subroutine->name == "get_inst_coverage") {
        // get_inst_coverage() returns per-instance coverage as a real (64-bit)
        auto realTy =
            moore::RealType::get(context.getContext(), moore::RealWidth::f64);
        return moore::CovergroupGetInstCoverageOp::create(
            builder, loc, realTy, covergroupInstance);
      }

      // For other covergroup methods, emit a warning and fall through
      // to normal function call handling
      mlir::emitWarning(loc) << "covergroup method '" << subroutine->name
                             << "' is not yet implemented; lowering as "
                                "regular function call";
    }

    // Handle process built-in methods.
    // IEEE 1800-2017 Section 9.7 "Process control"
    if (parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name == "process") {
        // process::self() is a static method - no 'this' object needed.
        if (subroutine->name == "self") {
          // Generate a call to __moore_process_self() runtime function which
          // returns a non-null handle when called from inside a process context
          // (llhd.process or sim.fork child), or null otherwise.
          auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
          auto selfFuncTy = mlir::LLVM::LLVMFunctionType::get(ptrTy, {});
          auto selfFunc = getOrCreateRuntimeFunc(context, "__moore_process_self",
                                                 selfFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, selfFunc,
                                                   ValueRange{});
          // Convert the result to the expected class handle type
          auto resultType = context.convertType(*expr.type);
          if (!resultType)
            return {};
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        // Instance methods: require a process handle ('this')
        Value procInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          procInstance = context.convertRvalueExpression(*recvExpr);
          if (!procInstance)
            return {};
        }

        if (!procInstance) {
          mlir::emitError(loc)
              << "process method call requires process handle";
          return {};
        }

        auto i64Ty = builder.getIntegerType(64);
        auto i32Ty = builder.getIntegerType(32);
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
        auto stringStructTy = mlir::LLVM::LLVMStructType::getLiteral(
            context.getContext(), {ptrTy, i64Ty});

        // Convert the process handle to an i64
        Value handle = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, i64Ty, procInstance)
                           .getResult(0);

        if (subroutine->name == "kill") {
          auto killFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()), {i64Ty});
          auto killFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_kill", killFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, killFunc,
                                     ValueRange{handle});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        if (subroutine->name == "await") {
          auto awaitFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()), {i64Ty});
          auto awaitFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_await", awaitFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, awaitFunc,
                                     ValueRange{handle});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        if (subroutine->name == "suspend") {
          auto suspendFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()), {i64Ty});
          auto suspendFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_suspend", suspendFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, suspendFunc,
                                     ValueRange{handle});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        if (subroutine->name == "resume") {
          auto resumeFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()), {i64Ty});
          auto resumeFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_resume", resumeFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, resumeFunc,
                                     ValueRange{handle});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        if (subroutine->name == "status") {
          auto statusFuncTy =
              mlir::LLVM::LLVMFunctionType::get(i32Ty, {i64Ty});
          auto statusFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_status", statusFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(
              builder, loc, statusFunc, ValueRange{handle});
          auto resultType = context.convertType(*expr.type);
          if (!resultType)
            return {};
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        if (subroutine->name == "get_randstate") {
          auto getFuncTy =
              mlir::LLVM::LLVMFunctionType::get(stringStructTy, {i64Ty});
          auto getFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_get_randstate", getFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, getFunc,
                                                   ValueRange{handle});
          auto resultType = context.convertType(*expr.type);
          if (!resultType)
            return {};
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        if (subroutine->name == "set_randstate") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc)
                << "process::set_randstate() requires a string argument";
            return {};
          }
          Value stateArg =
              context.convertRvalueExpression(*expr.arguments()[0]);
          if (!stateArg)
            return {};
          Value stateStruct =
              mlir::UnrealizedConversionCastOp::create(
                  builder, loc, stringStructTy, stateArg)
                  .getResult(0);
          auto setFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()),
              {i64Ty, stringStructTy});
          auto setFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_set_randstate", setFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, setFunc,
                                     ValueRange{handle, stateStruct});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        if (subroutine->name == "srandom") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc)
                << "process::srandom() requires a seed argument";
            return {};
          }
          Value seedArg =
              context.convertRvalueExpression(*expr.arguments()[0]);
          if (!seedArg)
            return {};
          auto intTy = moore::IntType::getInt(context.getContext(), 32);
          seedArg = context.materializeConversion(
              intTy, seedArg, expr.arguments()[0]->type->isSigned(), loc);
          if (!seedArg)
            return {};
          Value seedVal = mlir::UnrealizedConversionCastOp::create(
                              builder, loc, i32Ty, seedArg)
                              .getResult(0);
          auto seedFuncTy = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context.getContext()),
              {i64Ty, i32Ty});
          auto seedFunc = getOrCreateRuntimeFunc(
              context, "__moore_process_srandom", seedFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, seedFunc,
                                     ValueRange{handle, seedVal});
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Fall through for unhandled process methods.
        mlir::emitWarning(loc)
            << "process method '" << subroutine->name
            << "' is not yet implemented; lowering as regular function call";
      }
    }

    // Handle mailbox built-in method calls (put, get, try_put, try_get, num).
    // IEEE 1800-2017 Section 15.4 "Mailboxes"
    // Mailbox is a built-in parameterized class. Its methods have the BuiltIn
    // flag and the parent scope is a ClassType named "mailbox".
    if ((subroutine->flags & slang::ast::MethodFlags::BuiltIn) &&
        parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name == "mailbox") {
        // Get the mailbox instance (the 'this' object)
        Value mailboxInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          mailboxInstance = context.convertRvalueExpression(*recvExpr);
          if (!mailboxInstance)
            return {};
        }

        if (!mailboxInstance) {
          mlir::emitError(loc) << "mailbox method call requires mailbox instance";
          return {};
        }

        // The mailbox instance is a class handle. We need to get the mailbox ID
        // which is stored as a 64-bit integer inside the mailbox object.
        // For now, we'll use the class handle value directly as the mailbox ID
        // since our runtime represents mailboxes by their handle addresses.
        auto i64Ty = builder.getIntegerType(64);
        auto i1Ty = builder.getIntegerType(1);
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());

        // Convert the mailbox handle to an i64 mailbox ID
        Value mboxId = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, i64Ty, mailboxInstance)
                           .getResult(0);

        // Handle blocking put: mbox.put(msg)
        if (subroutine->name == "put") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.put() requires a message argument";
            return {};
          }
          // Convert the message argument to i64
          Value msgArg = context.convertRvalueExpression(*expr.arguments()[0]);
          if (!msgArg)
            return {};
          Value msg = mlir::UnrealizedConversionCastOp::create(
                          builder, loc, i64Ty, msgArg)
                          .getResult(0);

          // Declare and call __moore_mailbox_put
          auto putFuncTy =
              mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(context.getContext()),
                                                {i64Ty, i64Ty});
          auto putFunc =
              getOrCreateRuntimeFunc(context, "__moore_mailbox_put", putFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, putFunc, ValueRange{mboxId, msg});

          // put() returns void - return a dummy value
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Handle blocking get: mbox.get(msg)
        if (subroutine->name == "get") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.get() requires an output argument";
            return {};
          }
          // Get the lvalue for the output argument
          Value msgOutLvalue = context.convertLvalueExpression(*expr.arguments()[0]);
          if (!msgOutLvalue)
            return {};

          // Allocate temporary storage for the message
          auto c1 = builder.create<mlir::LLVM::ConstantOp>(
              loc, i64Ty, builder.getI64IntegerAttr(1));
          Value msgOut = mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, c1, 0);

          // Declare and call __moore_mailbox_get
          auto getFuncTy =
              mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(context.getContext()),
                                                {i64Ty, ptrTy});
          auto getFunc =
              getOrCreateRuntimeFunc(context, "__moore_mailbox_get", getFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, getFunc, ValueRange{mboxId, msgOut});

          // Load the result and store it to the output argument
          Value msgResult = mlir::LLVM::LoadOp::create(builder, loc, i64Ty, msgOut);
          auto outType = cast<moore::RefType>(msgOutLvalue.getType()).getNestedType();
          Value convertedMsg = mlir::UnrealizedConversionCastOp::create(
                                   builder, loc, outType, msgResult)
                                   .getResult(0);
          moore::BlockingAssignOp::create(builder, loc, msgOutLvalue, convertedMsg);

          // get() returns void - return a dummy value
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Handle non-blocking try_put: mbox.try_put(msg)
        if (subroutine->name == "try_put") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.try_put() requires a message argument";
            return {};
          }
          // Convert the message argument to i64
          Value msgArg = context.convertRvalueExpression(*expr.arguments()[0]);
          if (!msgArg)
            return {};
          Value msg = mlir::UnrealizedConversionCastOp::create(
                          builder, loc, i64Ty, msgArg)
                          .getResult(0);

          // Declare and call __moore_mailbox_tryput
          auto tryputFuncTy = mlir::LLVM::LLVMFunctionType::get(i1Ty, {i64Ty, i64Ty});
          auto tryputFunc = getOrCreateRuntimeFunc(
              context, "__moore_mailbox_tryput", tryputFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, tryputFunc,
                                                   ValueRange{mboxId, msg});

          // Convert result to moore int type
          auto resultType = context.convertType(*expr.type);
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        // Handle non-blocking try_get: mbox.try_get(msg)
        if (subroutine->name == "try_get") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.try_get() requires an output argument";
            return {};
          }
          // Get the lvalue for the output argument
          Value msgOutLvalue = context.convertLvalueExpression(*expr.arguments()[0]);
          if (!msgOutLvalue)
            return {};

          // Allocate temporary storage for the message
          auto c1 = builder.create<mlir::LLVM::ConstantOp>(
              loc, i64Ty, builder.getI64IntegerAttr(1));
          Value msgOut = mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, c1, 0);

          // Declare and call __moore_mailbox_tryget
          auto trygetFuncTy = mlir::LLVM::LLVMFunctionType::get(i1Ty, {i64Ty, ptrTy});
          auto trygetFunc = getOrCreateRuntimeFunc(
              context, "__moore_mailbox_tryget", trygetFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, trygetFunc,
                                                   ValueRange{mboxId, msgOut});

          // Load the result and store it to the output argument
          Value msgResult = mlir::LLVM::LoadOp::create(builder, loc, i64Ty, msgOut);
          auto outType = cast<moore::RefType>(msgOutLvalue.getType()).getNestedType();
          Value convertedMsg = mlir::UnrealizedConversionCastOp::create(
                                   builder, loc, outType, msgResult)
                                   .getResult(0);
          moore::BlockingAssignOp::create(builder, loc, msgOutLvalue, convertedMsg);

          // Convert result to moore int type
          auto resultType = context.convertType(*expr.type);
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        // Handle blocking peek: mbox.peek(msg)
        if (subroutine->name == "peek") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.peek() requires an output argument";
            return {};
          }
          // Get the lvalue for the output argument
          Value msgOutLvalue = context.convertLvalueExpression(*expr.arguments()[0]);
          if (!msgOutLvalue)
            return {};

          // Allocate temporary storage for the message
          auto c1 = builder.create<mlir::LLVM::ConstantOp>(
              loc, i64Ty, builder.getI64IntegerAttr(1));
          Value msgOut = mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, c1, 0);

          // Declare and call __moore_mailbox_peek
          auto peekFuncTy =
              mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(context.getContext()),
                                                {i64Ty, ptrTy});
          auto peekFunc =
              getOrCreateRuntimeFunc(context, "__moore_mailbox_peek", peekFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, peekFunc, ValueRange{mboxId, msgOut});

          // Load the result and store it to the output argument
          Value msgResult = mlir::LLVM::LoadOp::create(builder, loc, i64Ty, msgOut);
          auto outType = cast<moore::RefType>(msgOutLvalue.getType()).getNestedType();
          Value convertedMsg = mlir::UnrealizedConversionCastOp::create(
                                   builder, loc, outType, msgResult)
                                   .getResult(0);
          moore::BlockingAssignOp::create(builder, loc, msgOutLvalue, convertedMsg);

          // peek() returns void - return a dummy value
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Handle non-blocking try_peek: mbox.try_peek(msg)
        if (subroutine->name == "try_peek") {
          if (expr.arguments().size() < 1) {
            mlir::emitError(loc) << "mailbox.try_peek() requires an output argument";
            return {};
          }
          // Get the lvalue for the output argument
          Value msgOutLvalue = context.convertLvalueExpression(*expr.arguments()[0]);
          if (!msgOutLvalue)
            return {};

          // Allocate temporary storage for the message
          auto c1 = builder.create<mlir::LLVM::ConstantOp>(
              loc, i64Ty, builder.getI64IntegerAttr(1));
          Value msgOut = mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, c1, 0);

          // Declare and call __moore_mailbox_trypeek
          auto trypeekFuncTy = mlir::LLVM::LLVMFunctionType::get(i1Ty, {i64Ty, ptrTy});
          auto trypeekFunc = getOrCreateRuntimeFunc(
              context, "__moore_mailbox_trypeek", trypeekFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, trypeekFunc,
                                                   ValueRange{mboxId, msgOut});

          // Load the result and store it to the output argument
          Value msgResult = mlir::LLVM::LoadOp::create(builder, loc, i64Ty, msgOut);
          auto outType = cast<moore::RefType>(msgOutLvalue.getType()).getNestedType();
          Value convertedMsg = mlir::UnrealizedConversionCastOp::create(
                                   builder, loc, outType, msgResult)
                                   .getResult(0);
          moore::BlockingAssignOp::create(builder, loc, msgOutLvalue, convertedMsg);

          // Convert result to moore int type
          auto resultType = context.convertType(*expr.type);
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        // Handle num: mbox.num()
        if (subroutine->name == "num") {
          // Declare and call __moore_mailbox_num
          auto numFuncTy = mlir::LLVM::LLVMFunctionType::get(i64Ty, {i64Ty});
          auto numFunc =
              getOrCreateRuntimeFunc(context, "__moore_mailbox_num", numFuncTy);
          auto callOp = mlir::LLVM::CallOp::create(builder, loc, numFunc,
                                                   ValueRange{mboxId});

          // Convert result to moore int type
          auto resultType = context.convertType(*expr.type);
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callOp.getResult())
              .getResult(0);
        }

        // Fall through for unhandled mailbox methods
        mlir::emitWarning(loc) << "mailbox method '" << subroutine->name
                               << "' is not yet implemented";
      }
    }

    // Handle semaphore built-in method calls (get, put, try_get).
    // IEEE 1800-2017 Section 15.3 "Semaphores"
    // Semaphore is a built-in class. Its methods have the BuiltIn flag
    // and the parent scope is a ClassType named "semaphore".
    if ((subroutine->flags & slang::ast::MethodFlags::BuiltIn) &&
        parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name == "semaphore") {
        // Get the semaphore instance (the 'this' object)
        Value semInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          semInstance = context.convertRvalueExpression(*recvExpr);
          if (!semInstance)
            return {};
        }

        if (!semInstance) {
          mlir::emitError(loc)
              << "semaphore method call requires semaphore instance";
          return {};
        }

        auto i64Ty = builder.getIntegerType(64);
        auto i32Ty = builder.getIntegerType(32);
        auto i1Ty = builder.getIntegerType(1);
        auto voidTy = mlir::LLVM::LLVMVoidType::get(context.getContext());

        // Convert the semaphore handle to an i64 semaphore ID
        Value semId = mlir::UnrealizedConversionCastOp::create(
                          builder, loc, i64Ty, semInstance)
                          .getResult(0);

        // Handle blocking get: sem.get(keyCount = 1)
        if (subroutine->name == "get") {
          Value keyCount;
          if (expr.arguments().size() >= 1) {
            keyCount = context.convertRvalueExpression(*expr.arguments()[0]);
            if (!keyCount)
              return {};
            keyCount = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, i32Ty, keyCount)
                           .getResult(0);
          } else {
            keyCount = builder.create<mlir::LLVM::ConstantOp>(
                loc, i32Ty, builder.getI32IntegerAttr(1));
          }

          auto getFuncTy =
              mlir::LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty});
          auto getFunc = getOrCreateRuntimeFunc(
              context, "__moore_semaphore_get", getFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, getFunc,
                                     ValueRange{semId, keyCount});

          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Handle put: sem.put(keyCount = 1)
        if (subroutine->name == "put") {
          Value keyCount;
          if (expr.arguments().size() >= 1) {
            keyCount = context.convertRvalueExpression(*expr.arguments()[0]);
            if (!keyCount)
              return {};
            keyCount = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, i32Ty, keyCount)
                           .getResult(0);
          } else {
            keyCount = builder.create<mlir::LLVM::ConstantOp>(
                loc, i32Ty, builder.getI32IntegerAttr(1));
          }

          auto putFuncTy =
              mlir::LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty});
          auto putFunc = getOrCreateRuntimeFunc(
              context, "__moore_semaphore_put", putFuncTy);
          mlir::LLVM::CallOp::create(builder, loc, putFunc,
                                     ValueRange{semId, keyCount});

          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, moore::VoidType::get(context.getContext()),
                     ValueRange{})
              .getResult(0);
        }

        // Handle non-blocking try_get: sem.try_get(keyCount = 1)
        if (subroutine->name == "try_get") {
          Value keyCount;
          if (expr.arguments().size() >= 1) {
            keyCount = context.convertRvalueExpression(*expr.arguments()[0]);
            if (!keyCount)
              return {};
            keyCount = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, i32Ty, keyCount)
                           .getResult(0);
          } else {
            keyCount = builder.create<mlir::LLVM::ConstantOp>(
                loc, i32Ty, builder.getI32IntegerAttr(1));
          }

          auto tryGetFuncTy =
              mlir::LLVM::LLVMFunctionType::get(i1Ty, {i64Ty, i32Ty});
          auto tryGetFunc = getOrCreateRuntimeFunc(
              context, "__moore_semaphore_try_get", tryGetFuncTy);
          auto callResult = mlir::LLVM::CallOp::create(
              builder, loc, tryGetFunc, ValueRange{semId, keyCount});

          auto resultType = context.convertType(*expr.type);
          return mlir::UnrealizedConversionCastOp::create(
                     builder, loc, resultType, callResult.getResult())
              .getResult(0);
        }

        // Fall through for unhandled semaphore methods
        mlir::emitWarning(loc) << "semaphore method '" << subroutine->name
                               << "' is not yet implemented";
      }
    }

    // Handle class-level srandom(seed) calls.
    // IEEE 1800-2017 §18.13.3: Every class instance has a built-in srandom()
    // method that seeds the per-object RNG used by randomize().
    // We intercept this here and emit a __moore_class_srandom(objPtr, seed)
    // runtime call so the interpreter can maintain per-object RNG state.
    if (subroutine->name == "srandom" &&
        parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name != "process") {
        // Get the class instance (the 'this' object)
        Value classInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          classInstance = context.convertRvalueExpression(*recvExpr);
        }
        if (!classInstance)
          classInstance = context.getImplicitThisRef();

        if (!classInstance) {
          mlir::emitError(loc)
              << "class srandom() requires a class object receiver";
          return {};
        }

        // Convert seed argument
        if (expr.arguments().size() < 1) {
          mlir::emitError(loc)
              << "class srandom() requires a seed argument";
          return {};
        }
        Value seedArg =
            context.convertRvalueExpression(*expr.arguments()[0]);
        if (!seedArg)
          return {};
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        seedArg = context.materializeConversion(
            intTy, seedArg, expr.arguments()[0]->type->isSigned(), loc);
        if (!seedArg)
          return {};

        // Emit runtime call: __moore_class_srandom(objPtr, seed)
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
        auto i32Ty = builder.getIntegerType(32);
        Value objPtr = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, ptrTy, classInstance)
                           .getResult(0);
        Value seedVal = mlir::UnrealizedConversionCastOp::create(
                            builder, loc, i32Ty, seedArg)
                            .getResult(0);
        auto funcTy = mlir::LLVM::LLVMFunctionType::get(
            mlir::LLVM::LLVMVoidType::get(context.getContext()),
            {ptrTy, i32Ty});
        auto func = getOrCreateRuntimeFunc(
            context, "__moore_class_srandom", funcTy);
        mlir::LLVM::CallOp::create(builder, loc, func,
                                   ValueRange{objPtr, seedVal});
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, moore::VoidType::get(context.getContext()),
                   ValueRange{})
            .getResult(0);
      }
    }

    // Handle class-level get_randstate() calls.
    // IEEE 1800-2017 §18.13: Returns a string representation of the
    // per-object RNG state that can later be restored with set_randstate().
    // Emit __moore_class_get_randstate(objPtr) -> !llvm.struct<(ptr, i64)>
    if (subroutine->name == "get_randstate" &&
        parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name != "process") {
        Value classInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          classInstance = context.convertRvalueExpression(*recvExpr);
        }
        if (!classInstance)
          classInstance = context.getImplicitThisRef();

        if (!classInstance) {
          mlir::emitError(loc)
              << "class get_randstate() requires a class object receiver";
          return {};
        }

        auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
        auto i64Ty = builder.getIntegerType(64);
        auto stringStructTy = mlir::LLVM::LLVMStructType::getLiteral(
            context.getContext(), {ptrTy, i64Ty});

        Value objPtr = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, ptrTy, classInstance)
                           .getResult(0);

        auto funcTy =
            mlir::LLVM::LLVMFunctionType::get(stringStructTy, {ptrTy});
        auto func = getOrCreateRuntimeFunc(
            context, "__moore_class_get_randstate", funcTy);
        auto callOp = mlir::LLVM::CallOp::create(builder, loc, func,
                                                  ValueRange{objPtr});

        auto resultType = context.convertType(*expr.type);
        if (!resultType)
          return {};
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, resultType, callOp.getResult())
            .getResult(0);
      }
    }

    // Handle class-level set_randstate(state) calls.
    // IEEE 1800-2017 §18.13: Restores per-object RNG state from a string
    // previously obtained via get_randstate().
    // Emit __moore_class_set_randstate(objPtr, stateStruct) -> void
    if (subroutine->name == "set_randstate" &&
        parentSym.kind == slang::ast::SymbolKind::ClassType) {
      const auto *classType = parentSym.as_if<slang::ast::ClassType>();
      if (classType && classType->name != "process") {
        Value classInstance;
        if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
          classInstance = context.convertRvalueExpression(*recvExpr);
        }
        if (!classInstance)
          classInstance = context.getImplicitThisRef();

        if (!classInstance) {
          mlir::emitError(loc)
              << "class set_randstate() requires a class object receiver";
          return {};
        }

        if (expr.arguments().size() < 1) {
          mlir::emitError(loc)
              << "class set_randstate() requires a string argument";
          return {};
        }
        Value stateArg =
            context.convertRvalueExpression(*expr.arguments()[0]);
        if (!stateArg)
          return {};

        auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
        auto i64Ty = builder.getIntegerType(64);
        auto stringStructTy = mlir::LLVM::LLVMStructType::getLiteral(
            context.getContext(), {ptrTy, i64Ty});

        Value objPtr = mlir::UnrealizedConversionCastOp::create(
                           builder, loc, ptrTy, classInstance)
                           .getResult(0);
        Value stateStruct =
            mlir::UnrealizedConversionCastOp::create(
                builder, loc, stringStructTy, stateArg)
                .getResult(0);

        auto funcTy = mlir::LLVM::LLVMFunctionType::get(
            mlir::LLVM::LLVMVoidType::get(context.getContext()),
            {ptrTy, stringStructTy});
        auto func = getOrCreateRuntimeFunc(
            context, "__moore_class_set_randstate", funcTy);
        mlir::LLVM::CallOp::create(builder, loc, func,
                                   ValueRange{objPtr, stateStruct});
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, moore::VoidType::get(context.getContext()),
                   ValueRange{})
            .getResult(0);
      }
    }

    // Check if this is an interface method call (task/function defined inside
    // an interface). Interface methods have an implicit first argument for the
    // interface instance, similar to class methods with 'this'.
    bool isInterfaceMethod = false;
    Value interfaceInstance;
    if (parentSym.kind == slang::ast::SymbolKind::InstanceBody) {
      if (auto *instBody = parentSym.as_if<slang::ast::InstanceBodySymbol>()) {
        if (instBody->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface) {
          isInterfaceMethod = true;
          // Get the interface instance from the call expression.
          // For calls like `iface.set_sig()`, we need to extract the interface
          // instance from the syntax tree since slang doesn't provide thisClass()
          // for interface method calls like it does for class method calls.

          // First, try thisClass() in case slang does provide it
          if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
            interfaceInstance = context.convertRvalueExpression(*recvExpr);
          }

          // For virtual interface method calls like `vif.method()`, slang
          // doesn't populate thisClass(). We need to extract the virtual
          // interface expression from the syntax.
          if (!interfaceInstance && expr.syntax) {
            // Look for MemberAccessExpressionSyntax or ScopedNameSyntax patterns
            // that indicate a virtual interface method call.
            const slang::syntax::ExpressionSyntax *viExprSyntax = nullptr;

            if (auto *invocation = expr.syntax->as_if<
                    slang::syntax::InvocationExpressionSyntax>()) {
              // The left side of the invocation is the receiver expression
              if (auto *memberAccess = invocation->left->as_if<
                      slang::syntax::MemberAccessExpressionSyntax>()) {
                viExprSyntax = memberAccess->left;
              } else if (auto *scopedName = invocation->left->as_if<
                             slang::syntax::ScopedNameSyntax>()) {
                // For scoped names like `vi.wait_for_reset`, the left part
                // is the virtual interface expression.
                viExprSyntax = scopedName->left;
              }
            }

            if (viExprSyntax && context.currentScope) {
              // Create an AST context to bind the expression
              slang::ast::ASTContext astContext(*context.currentScope,
                                                slang::ast::LookupLocation::max);
              // Use the general expression binding method
              const auto &viExpr = slang::ast::Expression::bind(
                  *viExprSyntax, astContext);
              if (!viExpr.bad() && viExpr.type->isVirtualInterface()) {
                interfaceInstance = context.convertRvalueExpression(viExpr);
              }
            }
          }

          // If not available, check if we're inside an interface method calling
          // another method on the same interface. In this case, use the current
          // interface argument (the implicit first parameter of the interface
          // method).
          if (!interfaceInstance && context.currentInterfaceArg &&
              context.currentInterfaceBody == instBody) {
            interfaceInstance = context.currentInterfaceArg;
          }

          // If still not available, look up the interface instance by name.
          // The call syntax "iface.set_sig()" means we need the instance "iface".
          // We can find it by looking for the interface instance in our tracked
          // instances whose definition matches the method's parent interface.
          if (!interfaceInstance) {
            // Find the interface instance by scanning tracked instances
            for (const auto &[instSym, instValue] : context.interfaceInstances) {
              if (&instSym->body == instBody) {
                interfaceInstance = instValue;
                break;
              }
            }
          }

          if (!interfaceInstance) {
            // Try to detect hierarchical interface access through module
            // instances. For calls like `agentBFM.driverBFM.waitForResetn()`
            // where driverBFM is an interface inside module agentBFM, we need
            // to identify this pattern and provide a helpful error message.
            bool isHierarchicalInterfaceCall = false;

            // Check if the interface containing this method is inside a
            // different module from the current scope. The interface instance
            // is in instBody->parentInstance.
            if (instBody->parentInstance) {
              auto *ifaceInst = instBody->parentInstance;
              // Get the scope where this interface was instantiated
              if (auto *ifaceParentScope = ifaceInst->getParentScope()) {
                if (auto *containingBody =
                        ifaceParentScope->getContainingInstance()) {
                  // Check if the interface is inside a different module
                  if (containingBody->getDefinition().definitionKind ==
                      slang::ast::DefinitionKind::Module) {
                    if (context.currentScope) {
                      auto *currentBody =
                          context.currentScope->getContainingInstance();
                      if (currentBody && currentBody != containingBody) {
                        isHierarchicalInterfaceCall = true;
                      }
                    }
                  }
                }
              }
            }

            if (isHierarchicalInterfaceCall) {
              mlir::emitError(loc)
                  << "hierarchical interface method calls through module "
                     "instances are not yet supported (interface '"
                  << instBody->getDefinition().name
                  << "' is inside a child module); consider using a virtual "
                     "interface passed through module ports instead";
            } else {
              mlir::emitError(loc)
                  << "interface method call requires interface instance (could "
                     "not find instance for interface '"
                  << instBody->getDefinition().name << "')";
            }
            return {};
          }

          // If the interface instance is a reference, read it to get the value.
          // Interface instances in modules are stored as refs.
          if (auto refTy =
                  dyn_cast<moore::RefType>(interfaceInstance.getType())) {
            interfaceInstance =
                moore::ReadOp::create(builder, loc, interfaceInstance);
          }
        }
      }
    }

    auto *lowering = context.declareFunction(*subroutine);
    if (!lowering)
      return {};
    auto convertedFunction = context.convertFunction(*subroutine);
    if (failed(convertedFunction)) {
      mlir::emitError(loc) << "convertFunction failed for "
                           << subroutine->name;
      return {};
    }

    // For method calls, get the receiver `this` reference first. This is needed
    // before converting arguments because default argument expressions may
    // contain method calls with implicit `this` that should refer to the
    // receiver, not the caller's `this`.
    Value methodReceiver;
    moore::ClassHandleType methodReceiverTy;
    bool isSuperCall = false;
    if (isMethod) {
      auto [thisRef, tyHandle, superCall] = getMethodReceiverTypeHandle(expr);
      if (!thisRef)
        return {};
      methodReceiver = thisRef;
      methodReceiverTy = tyHandle;
      isSuperCall = superCall;
    }

    // Clear the override for nested calls in argument evaluation.
    context.methodReceiverOverride = {};

    // Convert the call arguments. Input arguments are converted to an rvalue.
    // `ref` arguments are passed through directly as references.
    // `output` / `inout` arguments use copy-out semantics: pass a temporary
    // reference and write the temporary back to the actual after the call.
    //
    // For method calls, default argument expressions (which contain method calls
    // with implicit `this`) should use the receiver's `this`, not the caller's.
    // We detect default arguments by checking if the argument expression's
    // source location is within the subroutine's location range.
    struct DeferredOutputCopy {
      Value actualRef;
      Value tempRef;
      bool isSigned;
    };
    SmallVector<Value> arguments;
    SmallVector<DeferredOutputCopy> deferredOutputCopies;
    auto subroutineLoc = subroutine->location;
    for (auto [callArg, declArg] :
         llvm::zip(expr.arguments(), subroutine->getArguments())) {

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for output
      // and inout arguments.
      auto *argExpr = callArg;
      if (const auto *assign = argExpr->as_if<slang::ast::AssignmentExpression>())
        argExpr = &assign->left();

      // Check if this argument is from a default value by comparing source
      // locations. Default argument expressions have source locations within
      // the function definition (same buffer as subroutine, after subroutine
      // location). Explicit arguments have source locations from the call site
      // (same buffer as expr, near expr's location).
      bool isDefaultArg = false;
      if (isMethod && methodReceiver) {
        auto argLoc = argExpr->sourceRange.start();
        auto callLoc = expr.sourceRange.start();
        // An argument is from a default value if:
        // 1. Its source location is in the same buffer as the subroutine
        //    definition (not the call site), AND
        // 2. Its source location is NOT in the same buffer as the call
        //    expression, or is far from the call site.
        // This distinguishes between:
        // - Default args: defined where the subroutine is defined
        // - Explicit args: defined where the call is made
        if (argLoc.buffer() == subroutineLoc.buffer() &&
            argLoc.buffer() != callLoc.buffer()) {
          // Argument is in subroutine's file but not call's file
          isDefaultArg = true;
        } else if (argLoc.buffer() == subroutineLoc.buffer() &&
                   argLoc.buffer() == callLoc.buffer()) {
          // Both in the same file - check if arg is near subroutine or call
          // If arg is closer to subroutine than to call, it's a default
          auto argOffset = argLoc.offset();
          auto subOffset = subroutineLoc.offset();
          auto callOffset = callLoc.offset();
          // Default args should be between subroutine start and call site,
          // and specifically in the subroutine's parameter list area.
          // A simple heuristic: if the arg location is before the call
          // location and after/at the subroutine location, it's a default.
          if (argOffset >= subOffset && argOffset < callOffset) {
            isDefaultArg = true;
          }
        }
      }

      // For default arguments in method calls, temporarily set the implicit
      // `this` to the receiver so that method calls with implicit `this` use
      // the correct receiver object.
      auto savedThis = context.currentThisRef;
      if (isDefaultArg)
        context.currentThisRef = methodReceiver;
      auto restoreThis =
          llvm::make_scope_exit([&] { context.currentThisRef = savedThis; });

      Value value;
      auto type = context.convertType(declArg->getType());
      if (declArg->direction == slang::ast::ArgumentDirection::In) {
        value = context.convertRvalueExpression(*argExpr, type);
      } else {
        Value lvalue = context.convertLvalueExpression(*argExpr);
        auto unpackedType = dyn_cast<moore::UnpackedType>(type);
        if (!unpackedType)
          return {};
        auto refType = moore::RefType::get(unpackedType);
        Value lvalueRef = context.materializeConversion(
            refType, lvalue, argExpr->type->isSigned(), loc);
        if (!lvalueRef)
          return {};

        using slang::ast::ArgumentDirection;
        switch (declArg->direction) {
        case ArgumentDirection::Ref:
          value = lvalueRef;
          break;
        case ArgumentDirection::Out:
        case ArgumentDirection::InOut: {
          Value initValue;
          if (declArg->direction == ArgumentDirection::InOut)
            initValue = moore::ReadOp::create(builder, loc, lvalueRef);
          auto tempRef = moore::VariableOp::create(
              builder, loc, refType, StringAttr{}, initValue);
          value = tempRef;
          deferredOutputCopies.push_back(
              {lvalueRef, tempRef, argExpr->type->isSigned()});
          break;
        }
        default:
          value = lvalueRef;
          break;
        }
      }
      if (!value)
        return {};
      arguments.push_back(value);
    }

    auto emitDeferredOutputCopies = [&]() -> bool {
      for (const auto &copy : deferredOutputCopies) {
        auto actualRefType = dyn_cast<moore::RefType>(copy.actualRef.getType());
        if (!actualRefType)
          return false;
        auto actualType = actualRefType.getNestedType();
        Value tempValue = moore::ReadOp::create(builder, loc, copy.tempRef);
        tempValue =
            context.materializeConversion(actualType, tempValue, copy.isSigned,
                                          loc);
        if (!tempValue)
          return false;
        moore::BlockingAssignOp::create(builder, loc, copy.actualRef, tempValue);
      }
      return true;
    };

    if (!lowering->isConverting && !lowering->captures.empty()) {
      auto materializeCaptureAtCall = [&](Value cap) -> Value {
        // Captures are expected to be moore::RefType.
        auto refTy = dyn_cast<moore::RefType>(cap.getType());
        if (!refTy) {
          lowering->op.emitError(
              "expected captured value to be moore::RefType");
          return {};
        }

        // Expected case: the capture stems from a variable of any parent
        // scope. We need to walk up, since definition might be a couple regions
        // up.
        Region *capRegion = [&]() -> Region * {
          if (auto ba = dyn_cast<BlockArgument>(cap))
            return ba.getOwner()->getParent();
          if (auto *def = cap.getDefiningOp())
            return def->getParentRegion();
          return nullptr;
        }();

        Region *callRegion =
            builder.getBlock() ? builder.getBlock()->getParent() : nullptr;

        for (Region *r = callRegion; r; r = r->getParentRegion()) {
          if (r == capRegion) {
            // Safe to use the SSA value directly here.
            return cap;
          }
        }

        // If we're inside a function that's being converted, propagate the
        // capture to that function. The capture will be replaced with a block
        // argument when the caller function is finalized.
        if (auto *callerLowering = context.currentFunctionLowering) {
          // Check if the caller already has this capture
          auto it = callerLowering->captureIndex.find(cap);
          if (it != callerLowering->captureIndex.end()) {
            // The caller already captures this value, use it directly.
            // It will be replaced with a block arg during finalization.
            return cap;
          }
          // Add this capture to the caller's capture list
          auto [newIt, inserted] = callerLowering->captureIndex.try_emplace(
              cap, callerLowering->captures.size());
          if (inserted)
            callerLowering->captures.push_back(cap);
          // Return the capture value; it will be replaced with a block arg
          // when the caller is finalized.
          return cap;
        }

        // Otherwise we can't legally rematerialize this capture here.
        lowering->op.emitError()
            << "cannot materialize captured ref at call site; non-symbol "
            << "source: "
            << (cap.getDefiningOp()
                    ? cap.getDefiningOp()->getName().getStringRef()
                    : "<block-arg>");
        return {};
      };

      for (Value cap : lowering->captures) {
        Value mat = materializeCaptureAtCall(cap);
        if (!mat)
          return {};
        arguments.push_back(mat);
      }
    }

    // Determine result types from the declared/converted func op.
    SmallVector<Type> resultTypes(
        lowering->op.getFunctionType().getResults().begin(),
        lowering->op.getFunctionType().getResults().end());

    // Inside constraint blocks, use ConstraintMethodCallOp for method calls
    // to avoid symbol lookup issues with func.call across symbol table
    // boundaries (ClassDeclOp is a SymbolTable that doesn't contain func.func).
    if (context.inConstraintExpr && isMethod) {
      auto calleeSym = lowering->op.getSymName();
      auto methodRef =
          mlir::FlatSymbolRefAttr::get(context.getContext(), calleeSym);
      Type resultType = resultTypes.empty()
                            ? moore::VoidType::get(context.getContext())
                            : resultTypes[0];

      // Get the expected receiver type and upcast as necessary.
      auto funcTy = lowering->op.getFunctionType();
      auto expectedHdlTy =
          cast<moore::ClassHandleType>(funcTy.getInput(0));
      auto implicitThisRef = context.materializeConversion(
          expectedHdlTy, methodReceiver, false, methodReceiver.getLoc());

      // Build argument list with this reference first.
      SmallVector<Value> explicitArguments;
      explicitArguments.reserve(arguments.size() + 1);
      explicitArguments.push_back(implicitThisRef);
      explicitArguments.append(arguments.begin(), arguments.end());

      auto callOp = moore::ConstraintMethodCallOp::create(
          builder, loc, resultType, methodRef, explicitArguments);
      if (!emitDeferredOutputCopies())
        return {};
      if (resultTypes.empty())
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, moore::VoidType::get(context.getContext()),
                   ValueRange{})
            .getResult(0);
      return callOp.getResult();
    }

    mlir::CallOpInterface callOp;
    if (isMethod) {
      // Class functions -> build func.call / func.indirect_call with implicit
      // this argument. Use the already-computed receiver from earlier.
      // For super calls, use direct dispatch to call the parent's method.
      callOp = buildMethodCall(subroutine, lowering, methodReceiverTy,
                               methodReceiver, arguments, resultTypes,
                               isSuperCall);
    } else if (isInterfaceMethod) {
      // Interface method call -> func.call with implicit interface argument.
      // The interface instance is passed as the first argument.
      SmallVector<Value> argsWithIface;
      argsWithIface.reserve(arguments.size() + 1);

      // If the interface instance has a modport-qualified type (e.g.,
      // @interface::@modport), we need to convert it to the base interface type
      // since interface methods are declared with the base interface type.
      Value ifaceArg = interfaceInstance;
      auto expectedType = lowering->op.getFunctionType().getInput(0);
      if (ifaceArg.getType() != expectedType) {
        ifaceArg = moore::ConversionOp::create(builder, loc, expectedType,
                                               ifaceArg);
      }
      argsWithIface.push_back(ifaceArg);
      argsWithIface.append(arguments.begin(), arguments.end());
      callOp =
          mlir::func::CallOp::create(builder, loc, lowering->op, argsWithIface);
    } else {
      // Free function -> func.call
      callOp =
          mlir::func::CallOp::create(builder, loc, lowering->op, arguments);
    }

    if (!emitDeferredOutputCopies())
      return {};

    auto result = resultTypes.size() > 0 ? callOp->getOpResult(0) : Value{};
    // For calls to void functions we need to have a value to return from this
    // function. Create a dummy `unrealized_conversion_cast`, which will get
    // deleted again later on.
    if (resultTypes.size() == 0)
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, moore::VoidType::get(context.getContext()),
                 ValueRange{})
          .getResult(0);

    return result;
  }

  /// Handle system calls.
  Value visitCall(const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;

    // $rose, $fell, $stable, $changed, $past, $sampled, and their _gclk
    // variants are only valid in the context of properties and assertions.
    // Those are treated in the LTLDialect; treat them there instead.
    bool isAssertionCall =
        llvm::StringSwitch<bool>(subroutine.name)
            .Cases({"$rose", "$fell", "$stable", "$changed", "$past", "$sampled"},
                   true)
            .Cases({"$rose_gclk", "$fell_gclk", "$stable_gclk",
                    "$changed_gclk", "$past_gclk", "$future_gclk",
                    "$rising_gclk", "$falling_gclk", "$steady_gclk",
                    "$changing_gclk"},
                   true)
            .Cases({"$global_clock", "$inferred_clock",
                    "$inferred_disable"},
                   true)
            .Default(false);

    if (isAssertionCall)
      return context.convertAssertionCallExpression(expr, info, loc);

    auto args = expr.arguments();

    FailureOr<Value> result = Value{};
    Value value;
    Value value2;

    // Handle array locator methods (find, find_index, find_first, find_first_index,
    // find_last, find_last_index). These have IteratorCallInfo with the predicate.
    // IEEE 1800-2017 Section 7.12.2 "Array locator methods".
    bool isArrayLocatorMethod =
        llvm::StringSwitch<bool>(subroutine.name)
            .Cases({"find", "find_index"}, true)
            .Cases({"find_first", "find_first_index"}, true)
            .Cases({"find_last", "find_last_index"}, true)
            .Default(false);

    if (isArrayLocatorMethod) {
      // Get the iterator info from extraInfo
      auto [iterExpr, iterVar] = info.getIteratorInfo();
      if (!iterExpr || !iterVar) {
        mlir::emitError(loc) << "array locator method requires a 'with' clause";
        return {};
      }

      // The array is the first argument
      if (args.empty()) {
        mlir::emitError(loc) << "array locator method requires an array argument";
        return {};
      }

      // Convert the array as rvalue
      Value arrayVal = context.convertRvalueExpression(*args[0]);
      if (!arrayVal)
        return {};

      // Determine the mode and whether indexed
      moore::LocatorMode mode;
      bool indexed = false;
      if (subroutine.name == "find" || subroutine.name == "find_index") {
        mode = moore::LocatorMode::All;
        indexed = (subroutine.name == "find_index");
      } else if (subroutine.name == "find_first" ||
                 subroutine.name == "find_first_index") {
        mode = moore::LocatorMode::First;
        indexed = (subroutine.name == "find_first_index");
      } else {
        mode = moore::LocatorMode::Last;
        indexed = (subroutine.name == "find_last_index");
      }

      // Get the result type (a queue)
      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};
      auto queueType = dyn_cast<moore::QueueType>(resultType);
      if (!queueType) {
        mlir::emitError(loc) << "array locator method result must be a queue type";
        return {};
      }

      // Get the element type for the iterator variable
      Type iterVarType = context.convertType(iterVar->getType());
      if (!iterVarType)
        return {};

      // Create the array locator operation
      auto locatorOp = moore::ArrayLocatorOp::create(
          builder, loc, queueType, mode, indexed, arrayVal);

      // Create the body region with block arguments for the iterator variable
      // and the index
      Block *bodyBlock = &locatorOp.getBody().emplaceBlock();
      bodyBlock->addArgument(iterVarType, loc);
      // Add index argument as i32 (SystemVerilog array indices are int)
      auto indexType = moore::IntType::getInt(context.getContext(), 32);
      bodyBlock->addArgument(indexType, loc);
      Value iterArg = bodyBlock->getArgument(0);
      Value indexArg = bodyBlock->getArgument(1);

      // Set up the value symbol for the iterator variable within the region
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);

      // Temporarily bind the iterator variable to the block argument
      Context::ValueSymbolScope scope(context.valueSymbols);
      context.valueSymbols.insert(iterVar, iterArg);

      // Bind the iterator variable's index for use with item.index
      Context::IteratorIndexSymbolScope indexScope(context.iteratorIndexSymbols);
      context.iteratorIndexSymbols.insert(iterVar, indexArg);

      // Also set the current iterator index for use by convertSystemCallArity0
      Value savedIteratorIndex = context.currentIteratorIndex;
      context.currentIteratorIndex = indexArg;

      // Convert the predicate expression inside the region
      Value predResult = context.convertRvalueExpression(*iterExpr);

      // Restore the previous iterator index
      context.currentIteratorIndex = savedIteratorIndex;

      if (!predResult)
        return {};

      // Convert predicate to bool if necessary
      predResult = context.convertToBool(predResult);
      if (!predResult)
        return {};

      // Create the yield terminator
      moore::ArrayLocatorYieldOp::create(builder, loc, predResult);

      return locatorOp.getResult();
    }

    // Handle queue sort.with and rsort.with methods
    // IEEE 1800-2017 Section 7.12.3 "Array ordering methods"
    bool isSortWithMethod =
        llvm::StringSwitch<bool>(subroutine.name)
            .Cases({"sort", "rsort"}, true)
            .Default(false);

    if (isSortWithMethod) {
      // Check if this has a 'with' clause
      auto [iterExpr, iterVar] = info.getIteratorInfo();
      if (iterExpr && iterVar) {
        // This is sort.with or rsort.with
        // The array is the first argument
        if (args.empty()) {
          mlir::emitError(loc) << "sort.with method requires an array argument";
          return {};
        }

        // Convert the array as lvalue (it's modified in place)
        Value arrayRef = context.convertLvalueExpression(*args[0]);
        if (!arrayRef)
          return {};

        // Get the element type for the iterator variable
        Type iterVarType = context.convertType(iterVar->getType());
        if (!iterVarType)
          return {};

        // Create the appropriate sort.with operation
        Operation *sortOp;
        if (subroutine.name == "sort") {
          sortOp = moore::QueueSortWithOp::create(builder, loc, arrayRef);
        } else {
          sortOp = moore::QueueRSortWithOp::create(builder, loc, arrayRef);
        }

        // Get the body region
        Region &bodyRegion = sortOp->getRegion(0);
        Block *bodyBlock = &bodyRegion.emplaceBlock();
        bodyBlock->addArgument(iterVarType, loc);
        Value iterArg = bodyBlock->getArgument(0);

        // Set up the value symbol for the iterator variable within the region
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(bodyBlock);

        // Temporarily bind the iterator variable to the block argument
        Context::ValueSymbolScope scope(context.valueSymbols);
        context.valueSymbols.insert(iterVar, iterArg);

        // Convert the key expression inside the region
        Value keyResult = context.convertRvalueExpression(*iterExpr);
        if (!keyResult)
          return {};

        // Create the yield terminator
        moore::QueueSortKeyYieldOp::create(builder, loc, keyResult);

        // sort.with returns void, return a dummy value
        auto intTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, intTy, 0);
      }
    }

    // $sformatf() and $sformat look like system tasks, but we handle string
    // formatting differently from expression evaluation, so handle them
    // separately.
    // According to IEEE 1800-2023 Section 21.3.3 "Formatting data to a
    // string" $sformatf works just like the string formatting but returns
    // a StringType.
    if (!subroutine.name.compare("$sformatf") ||
        !subroutine.name.compare("$psprintf")) {
      // Create the FormatString
      auto fmtValue = context.convertFormatString(
          expr.arguments(), loc, moore::IntFormat::Decimal, false);
      if (failed(fmtValue))
        return {};
      return fmtValue.value();
    }

    // $sscanf(str, format, args...) reads formatted data from a string.
    // IEEE 1800-2017 Section 21.3.4 "Reading data from a string".
    // The first argument is the input string, the second is the format string,
    // and the remaining arguments are output variables (wrapped in
    // AssignmentExpression by slang).
    // Returns the number of items successfully read.
    if (!subroutine.name.compare("$sscanf") && args.size() >= 2) {
      // First argument is the input string
      Value inputStr = context.convertRvalueExpression(*args[0]);
      if (!inputStr)
        return {};

      // Convert to StringType if not already (slang may pass packed integers)
      if (!isa<moore::StringType>(inputStr.getType())) {
        inputStr = moore::ConversionOp::create(
            builder, loc, moore::StringType::get(context.getContext()),
            inputStr);
      }

      // Second argument is the format string - must be a string literal
      const auto *fmtArg = args[1];
      std::string formatStr;
      if (const auto *strLit = fmtArg->as_if<slang::ast::StringLiteral>()) {
        formatStr = std::string(strLit->getValue());
      } else {
        // Try to evaluate as constant
        auto cv = context.evaluateConstant(*fmtArg);
        if (cv && cv.isString()) {
          formatStr = cv.str();
        } else {
          mlir::emitError(loc) << "$sscanf format must be a string literal";
          return {};
        }
      }

      // Remaining arguments are output variables
      SmallVector<Value> outputRefs;
      for (size_t i = 2; i < args.size(); ++i) {
        const auto *arg = args[i];
        // Slang wraps output arguments in AssignmentExpression
        if (const auto *assignExpr =
                arg->as_if<slang::ast::AssignmentExpression>()) {
          Value ref = context.convertLvalueExpression(assignExpr->left());
          if (!ref)
            return {};
          outputRefs.push_back(ref);
        } else {
          // Try direct lvalue conversion
          Value ref = context.convertLvalueExpression(*arg);
          if (!ref)
            return {};
          outputRefs.push_back(ref);
        }
      }

      // Create the sscanf operation
      auto sscanfOp = moore::SScanfBIOp::create(builder, loc, inputStr,
                                                 builder.getStringAttr(formatStr),
                                                 outputRefs);
      return sscanfOp.getResult();
    }

    // $fscanf(fd, format, args...) reads formatted data from a file.
    // IEEE 1800-2017 Section 21.3.3 "File input functions".
    // The first argument is the file descriptor, the second is the format
    // string, and the remaining arguments are output variables.
    // Returns the number of items successfully read.
    if (!subroutine.name.compare("$fscanf") && args.size() >= 2) {
      // First argument is the file descriptor
      Value fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return {};

      // Ensure fd is i32
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      if (fd.getType() != i32Ty)
        fd = moore::ConversionOp::create(builder, loc, i32Ty, fd);

      // Second argument is the format string - must be a string literal
      const auto *fmtArg = args[1];
      std::string formatStr;
      if (const auto *strLit = fmtArg->as_if<slang::ast::StringLiteral>()) {
        formatStr = std::string(strLit->getValue());
      } else {
        // Try to evaluate as constant
        auto cv = context.evaluateConstant(*fmtArg);
        if (cv && cv.isString()) {
          formatStr = cv.str();
        } else {
          mlir::emitError(loc) << "$fscanf format must be a string literal";
          return {};
        }
      }

      // Remaining arguments are output variables
      SmallVector<Value> outputRefs;
      for (size_t i = 2; i < args.size(); ++i) {
        const auto *arg = args[i];
        // Slang wraps output arguments in AssignmentExpression
        if (const auto *assignExpr =
                arg->as_if<slang::ast::AssignmentExpression>()) {
          Value ref = context.convertLvalueExpression(assignExpr->left());
          if (!ref)
            return {};
          outputRefs.push_back(ref);
        } else {
          // Try direct lvalue conversion
          Value ref = context.convertLvalueExpression(*arg);
          if (!ref)
            return {};
          outputRefs.push_back(ref);
        }
      }

      // Create the fscanf operation
      auto fscanfOp = moore::FScanfBIOp::create(
          builder, loc, fd, builder.getStringAttr(formatStr), outputRefs);
      return fscanfOp.getResult();
    }

    // Handle string substr method: str.substr(start, len) has 3 args
    if (!subroutine.name.compare("substr") && args.size() == 3) {
      Value str = context.convertRvalueExpression(*args[0]);
      Value start = context.convertRvalueExpression(*args[1]);
      Value len = context.convertRvalueExpression(*args[2]);
      if (!str || !start || !len)
        return {};
      return moore::StringSubstrOp::create(builder, loc, str, start, len);
    }

    // $cast(dest, src) is a special case because the first argument is an
    // output parameter. Slang wraps it in an AssignmentExpression with
    // EmptyArgument as RHS.
    // IEEE 1800-2017 Section 8.16: $cast attempts to assign the source
    // expression to the destination variable. When used as a function,
    // it returns 1 if the cast is legal and 0 if not.
    if (!subroutine.name.compare("$cast") && args.size() == 2) {
      // Unpack the first argument from AssignmentExpression
      auto *destExpr = args[0];
      if (const auto *assign =
              destExpr->as_if<slang::ast::AssignmentExpression>())
        destExpr = &assign->left();

      // Convert destination as lvalue and source as rvalue
      Value destLvalue = context.convertLvalueExpression(*destExpr);
      Value srcRvalue = context.convertRvalueExpression(*args[1]);
      if (!destLvalue || !srcRvalue)
        return {};

      // Check if both are class handle types - this is the common case for
      // dynamic downcasting in OOP.
      // The destination can be either a RefType (for variables holding class
      // handles) or a ClassHandleType directly (for class properties).
      auto destTy = getLvalueNestedType(destLvalue.getType());
      if (!destTy) {
        mlir::emitError(loc) << "unsupported destination type in $cast: "
                             << destLvalue.getType();
        return {};
      }
      auto srcTy = srcRvalue.getType();

      bool isClassDowncast =
          isa<moore::ClassHandleType>(destTy) &&
          isa<moore::ClassHandleType>(srcTy);

      if (isClassDowncast) {
        // For class downcasting, use ClassDynCastOp which performs runtime
        // type checking. This is essential for UVM's factory pattern.
        // IEEE 1800-2017 Section 8.16: $cast returns 1 if cast succeeds.
        auto destClassTy = cast<moore::ClassHandleType>(destTy);
        auto dynCastOp = moore::ClassDynCastOp::create(builder, loc,
                                                        destClassTy,
                                                        builder.getI1Type(),
                                                        srcRvalue);

        // Store the casted value to destination (only valid if cast succeeds)
        moore::BlockingAssignOp::create(builder, loc, destLvalue,
                                        dynCastOp.getResult());

        // Return the success flag as an int (0 or 1)
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        return moore::ConversionOp::create(builder, loc, intTy,
                                           dynCastOp.getSuccess());
      } else {
        // For non-class types, try to perform the conversion
        auto castValue = context.materializeConversion(destTy, srcRvalue,
                                                       args[1]->type->isSigned(),
                                                       loc);
        if (castValue) {
          // Store the cast value to the destination
          moore::BlockingAssignOp::create(builder, loc, destLvalue, castValue);
        }
        // Return 1 (success) - static cast always succeeds
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        return moore::ConstantOp::create(builder, loc, intTy, 1);
      }
    }

    // Handle string.itoa method: str.itoa(value) converts integer to string.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "itoa" && args.size() == 2) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value intVal = context.convertRvalueExpression(*args[1]);
      if (!strRef || !intVal)
        return {};
      moore::StringItoaOp::create(builder, loc, strRef, intVal);
      // itoa returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle string.hextoa method: str.hextoa(value) converts integer to hex string.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "hextoa" && args.size() == 2) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value intVal = context.convertRvalueExpression(*args[1]);
      if (!strRef || !intVal)
        return {};
      moore::StringHexToAOp::create(builder, loc, strRef, intVal);
      // hextoa returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle string.octtoa method: str.octtoa(value) converts integer to octal string.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "octtoa" && args.size() == 2) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value intVal = context.convertRvalueExpression(*args[1]);
      if (!strRef || !intVal)
        return {};
      moore::StringOctToAOp::create(builder, loc, strRef, intVal);
      // octtoa returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle string.bintoa method: str.bintoa(value) converts integer to binary string.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "bintoa" && args.size() == 2) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value intVal = context.convertRvalueExpression(*args[1]);
      if (!strRef || !intVal)
        return {};
      moore::StringBinToAOp::create(builder, loc, strRef, intVal);
      // bintoa returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle string.realtoa method: str.realtoa(value) converts real to string.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "realtoa" && args.size() == 2) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value realVal = context.convertRvalueExpression(*args[1]);
      if (!strRef || !realVal)
        return {};
      moore::StringRealToAOp::create(builder, loc, strRef, realVal);
      // realtoa returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle string.putc method: str.putc(index, char) sets character at index.
    // The string is modified in-place, so we need lvalue for string.
    if (subroutine.name == "putc" && args.size() == 3) {
      Value strRef = context.convertLvalueExpression(*args[0]);
      Value index = context.convertRvalueExpression(*args[1]);
      Value charVal = context.convertRvalueExpression(*args[2]);
      if (!strRef || !index || !charVal)
        return {};
      // Ensure charVal is i8
      auto charType = moore::IntType::getInt(context.getContext(), 8);
      charVal = context.materializeConversion(charType, charVal, false, loc);
      moore::StringPutCOp::create(builder, loc, strRef, index, charVal);
      // putc returns void, return a dummy value
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    // Handle randomize() - both class method and std::randomize().
    // IEEE 1800-2017 Section 18.6 "Randomization methods" (class method)
    // IEEE 1800-2017 Section 18.12 "Scope randomize function" (std::randomize)
    // IEEE 1800-2017 Section 18.7 "In-line constraints" (with clause)
    // randomize() returns 1 on success, 0 on failure.
    if (subroutine.name == "randomize" && !args.empty()) {
      // Check for inline constraints from the with clause
      const slang::ast::Constraint *inlineConstraints = nullptr;
      if (info.extraInfo.index() == 2) {
        // extraInfo index 2 is RandomizeCallInfo
        const auto &randInfo =
            std::get<slang::ast::CallExpression::RandomizeCallInfo>(
                info.extraInfo);
        inlineConstraints = randInfo.inlineConstraints;
      }

      // Detect std::randomize() by checking if args are wrapped in
      // AssignmentExpression (slang wraps lvalue args this way for
      // std::randomize).
      bool isStdRandomize =
          args[0]->as_if<slang::ast::AssignmentExpression>() != nullptr;

      if (isStdRandomize) {
        // std::randomize(var1, var2, ...) - randomize standalone variables
        SmallVector<Value> varRefs;
        for (auto *arg : args) {
          const auto *assignExpr =
              arg->as_if<slang::ast::AssignmentExpression>();
          if (!assignExpr) {
            mlir::emitError(loc)
                << "std::randomize argument must be a variable";
            return {};
          }
          // The left side of the AssignmentExpression is the actual variable
          Value varRef = context.convertLvalueExpression(assignExpr->left());
          if (!varRef)
            return {};
          context.captureRef(varRef);
          varRefs.push_back(varRef);
        }

        auto stdRandomizeOp =
            moore::StdRandomizeOp::create(builder, loc, varRefs);

        // Handle inline constraints if present
        if (inlineConstraints) {
          // Create the inline constraint region
          stdRandomizeOp.getInlineConstraints().emplaceBlock();
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(
              &stdRandomizeOp.getInlineConstraints().front());

          // Convert the inline constraints
          if (failed(context.convertConstraint(*inlineConstraints, loc)))
            return {};
        }

        auto resultType = context.convertType(*expr.type);
        if (!resultType)
          return {};

        return context.materializeConversion(
            resultType, stdRandomizeOp.getSuccess(), false, loc);
      }

      // Class randomize: obj.randomize(), obj.randomize(null),
      // obj.randomize(v, w)
      // args[0] is always the class object
      // args[1..] are either:
      //   - empty: normal randomize
      //   - null: check-only mode (IEEE 1800-2017 Section 18.11.1)
      //   - variable names: in-line random variable control (Section 18.11)

      // The first argument is the class object to randomize
      Value classObj = context.convertRvalueExpression(*args[0]);
      if (!classObj)
        return {};

      // Verify that the argument is a class handle type
      auto classHandleTy =
          dyn_cast<moore::ClassHandleType>(classObj.getType());
      if (!classHandleTy) {
        mlir::emitError(loc) << "randomize() requires a class object, got "
                             << classObj.getType();
        return {};
      }

      // Check for check-only mode (randomize(null)) or variable list
      bool checkOnly = false;
      SmallVector<mlir::Attribute> variableList;

      if (args.size() > 1) {
        // Check if this is randomize(null)
        if (args.size() == 2) {
          // slang represents null as a ConversionExpression with null operand
          // or as an EmptyArgumentExpression
          if (auto *convExpr =
                  args[1]->as_if<slang::ast::ConversionExpression>()) {
            if (convExpr->operand().type->isNull()) {
              checkOnly = true;
            }
          } else if (auto *emptyArg =
                         args[1]
                             ->as_if<slang::ast::EmptyArgumentExpression>()) {
            // Empty argument also represents null in this context
            checkOnly = true;
          } else if (args[1]->type->isNull()) {
            // Direct null type check
            checkOnly = true;
          }
        }

        // If not check-only, collect variable names for variable list mode
        if (!checkOnly) {
          for (size_t i = 1; i < args.size(); ++i) {
            // The variable name references should be NamedValueExpressions
            if (auto *namedVal =
                    args[i]->as_if<slang::ast::NamedValueExpression>()) {
              auto varName = mlir::FlatSymbolRefAttr::get(
                  builder.getContext(), namedVal->symbol.name);
              variableList.push_back(varName);
            } else {
              // Try to get the name from member access
              if (auto *memberAccess =
                      args[i]->as_if<slang::ast::MemberAccessExpression>()) {
                auto varName = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), memberAccess->member.name);
                variableList.push_back(varName);
              }
            }
          }
        }
      }

      // Call pre_randomize() before randomization
      // IEEE 1800-2017 Section 18.6.1: pre_randomize is called before
      // the randomization process begins.
      // Note: pre_randomize is not called for check-only mode
      if (!checkOnly) {
        moore::CallPreRandomizeOp::create(builder, loc, classObj);
      }

      // Create the randomize operation which returns i1 (success/failure)
      mlir::ArrayAttr varListAttr = nullptr;
      if (!variableList.empty()) {
        varListAttr = builder.getArrayAttr(variableList);
      }
      auto randomizeOp = moore::RandomizeOp::create(builder, loc, classObj,
                                                    checkOnly, varListAttr);

      // Handle inline constraints if present
      if (inlineConstraints) {
        // Create the inline constraint region
        randomizeOp.getInlineConstraints().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(
            &randomizeOp.getInlineConstraints().front());

        auto savedThisRef = context.getInlineConstraintThisRef();
        auto savedThisSym = context.getInlineConstraintThisSymbol();
        context.setInlineConstraintThisRef(classObj);
        context.setInlineConstraintThisSymbol(args[0]->getSymbolReference());
        auto restoreThisRef = llvm::make_scope_exit(
            [&] {
              context.setInlineConstraintThisRef(savedThisRef);
              context.setInlineConstraintThisSymbol(savedThisSym);
            });

        // Convert the inline constraints
        if (failed(context.convertConstraint(*inlineConstraints, loc)))
          return {};
      }

      // Call post_randomize() after successful randomization
      // IEEE 1800-2017 Section 18.6.1: post_randomize is called after
      // randomization succeeds. The success value gates the call so
      // post_randomize is skipped when constraints are infeasible (§18.6.3).
      // Note: post_randomize is not called for check-only mode
      if (!checkOnly) {
        moore::CallPostRandomizeOp::create(builder, loc, classObj,
                                           randomizeOp.getSuccess());
      }

      // The result is i1, but the expression type from slang is typically
      // int. Convert to the expected type.
      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};

      return context.materializeConversion(resultType, randomizeOp.getSuccess(),
                                           false, loc);
    }

    // Handle rand_mode() for class-level or property-level queries.
    if (subroutine.name == "rand_mode") {
      const slang::ast::Expression *receiverExpr = nullptr;
      std::optional<mlir::FlatSymbolRefAttr> propertyAttr;

      if (expr.syntax && context.currentScope) {
        if (auto *invocation = expr.syntax->as_if<
                slang::syntax::InvocationExpressionSyntax>()) {
          if (auto *memberAccess = invocation->left->as_if<
                  slang::syntax::MemberAccessExpressionSyntax>()) {
            const slang::syntax::ExpressionSyntax *baseSyntax =
                memberAccess->left;
            slang::ast::ASTContext astContext(*context.currentScope,
                                              slang::ast::LookupLocation::max);
            const auto &baseExpr =
                slang::ast::Expression::bind(*baseSyntax, astContext);
            if (!baseExpr.bad()) {
              if (auto *memberExpr =
                      baseExpr.as_if<slang::ast::MemberAccessExpression>()) {
                propertyAttr = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), memberExpr->member.name);
                receiverExpr = &memberExpr->value();
              } else {
                receiverExpr = &baseExpr;
              }
            }
          }
        }
      }

      if (!receiverExpr) {
        if (const auto *thisClass = expr.thisClass())
          receiverExpr = thisClass;
      }
      if (receiverExpr) {
        if (auto *memberExpr =
                receiverExpr->as_if<slang::ast::MemberAccessExpression>()) {
          if (!propertyAttr)
            propertyAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                        memberExpr->member.name);
          receiverExpr = &memberExpr->value();
        }
      }

      if (!receiverExpr && !args.empty()) {
        const auto *arg0 = args[0];
        if (auto *memberExpr =
                arg0->as_if<slang::ast::MemberAccessExpression>()) {
          propertyAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                      memberExpr->member.name);
          receiverExpr = &memberExpr->value();
        } else {
          receiverExpr = arg0;
        }
      }
      if (!receiverExpr) {
        mlir::emitError(loc)
            << "rand_mode() requires a class object receiver";
        return {};
      }

      Value classObj;
      const slang::ast::ClassPropertySymbol *propertySymbol = nullptr;
      if (auto *namedExpr =
              receiverExpr->as_if<slang::ast::NamedValueExpression>()) {
        propertySymbol =
            namedExpr->symbol.as_if<slang::ast::ClassPropertySymbol>();
      }
      if (propertySymbol) {
        if (!propertyAttr) {
          propertyAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                      propertySymbol->name);
          classObj = context.getInlineConstraintThisRef();
          if (!classObj)
            classObj = context.getImplicitThisRef();
          if (!classObj) {
            mlir::emitError(loc)
                << "rand_mode() requires a class object receiver";
            return {};
          }
        }
      }
      if (!classObj) {
        classObj = context.convertRvalueExpression(*receiverExpr);
        if (!classObj)
          return {};
      }

      auto classHandleTy =
          dyn_cast<moore::ClassHandleType>(classObj.getType());
      if (!classHandleTy) {
        mlir::emitError(loc) << "rand_mode() requires a class object, got "
                             << classObj.getType();
        return {};
      }

      Value modeValue;
      if (args.size() > 1) {
        modeValue = context.convertRvalueExpression(*args[1]);
        if (!modeValue)
          return {};
      }

      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto randOp = moore::RandModeOp::create(
          builder, loc, intTy, classObj,
          propertyAttr ? *propertyAttr : mlir::FlatSymbolRefAttr{}, modeValue);
      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};
      return context.materializeConversion(resultType, randOp.getResult(),
                                           false, loc);
    }

    // Handle constraint_mode() for class-level or constraint-level queries.
    if (subroutine.name == "constraint_mode") {
      const slang::ast::Expression *receiverExpr = nullptr;
      std::optional<mlir::FlatSymbolRefAttr> constraintAttr;

      if (expr.syntax && context.currentScope) {
        if (auto *invocation = expr.syntax->as_if<
                slang::syntax::InvocationExpressionSyntax>()) {
          if (auto *memberAccess = invocation->left->as_if<
                  slang::syntax::MemberAccessExpressionSyntax>()) {
            const slang::syntax::ExpressionSyntax *baseSyntax =
                memberAccess->left;
            slang::ast::ASTContext astContext(*context.currentScope,
                                              slang::ast::LookupLocation::max);
            const auto &baseExpr =
                slang::ast::Expression::bind(*baseSyntax, astContext);
            if (!baseExpr.bad()) {
              if (auto *memberExpr =
                      baseExpr.as_if<slang::ast::MemberAccessExpression>()) {
                constraintAttr = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), memberExpr->member.name);
                receiverExpr = &memberExpr->value();
              } else {
                receiverExpr = &baseExpr;
              }
            }
          }
        }
      }

      if (!receiverExpr) {
        if (const auto *thisClass = expr.thisClass())
          receiverExpr = thisClass;
      }
      if (receiverExpr) {
        if (auto *memberExpr =
                receiverExpr->as_if<slang::ast::MemberAccessExpression>()) {
          if (!constraintAttr)
            constraintAttr = mlir::FlatSymbolRefAttr::get(
                builder.getContext(), memberExpr->member.name);
          receiverExpr = &memberExpr->value();
        }
      }

      if (!receiverExpr && !args.empty()) {
        const auto *arg0 = args[0];
        if (auto *memberExpr =
                arg0->as_if<slang::ast::MemberAccessExpression>()) {
          constraintAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                        memberExpr->member.name);
          receiverExpr = &memberExpr->value();
        } else {
          receiverExpr = arg0;
        }
      }
      if (!receiverExpr) {
        mlir::emitError(loc)
            << "constraint_mode() requires a class object receiver";
        return {};
      }

      Value classObj;
      const slang::ast::ConstraintBlockSymbol *constraintSymbol = nullptr;
      if (auto *namedExpr =
              receiverExpr->as_if<slang::ast::NamedValueExpression>()) {
        constraintSymbol =
            namedExpr->symbol.as_if<slang::ast::ConstraintBlockSymbol>();
      }
      if (constraintSymbol) {
        if (!constraintAttr) {
          constraintAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                        constraintSymbol->name);
          classObj = context.getInlineConstraintThisRef();
          if (!classObj)
            classObj = context.getImplicitThisRef();
          if (!classObj) {
            mlir::emitError(loc)
                << "constraint_mode() requires a class object receiver";
            return {};
          }
        }
      }
      if (!classObj) {
        classObj = context.convertRvalueExpression(*receiverExpr);
        if (!classObj)
          return {};
      }

      auto classHandleTy =
          dyn_cast<moore::ClassHandleType>(classObj.getType());
      if (!classHandleTy) {
        mlir::emitError(loc)
            << "constraint_mode() requires a class object, got "
            << classObj.getType();
        return {};
      }

      Value modeValue;
      if (args.size() > 1) {
        modeValue = context.convertRvalueExpression(*args[1]);
        if (!modeValue)
          return {};
      }

      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto constraintOp = moore::ConstraintModeOp::create(
          builder, loc, intTy, classObj,
          constraintAttr ? *constraintAttr : mlir::FlatSymbolRefAttr{},
          modeValue);
      auto resultType = context.convertType(*expr.type);
      if (!resultType)
        return {};
      return context.materializeConversion(resultType,
                                           constraintOp.getResult(), false,
                                           loc);
    }

    // Handle queue methods that need special treatment (lvalue for queue).
    // push_back, push_front need queue as lvalue + element as rvalue.
    // pop_back, pop_front need queue as lvalue, no additional args.
    // delete, sort, rsort, shuffle, reverse need queue as lvalue, no additional args, return void.
    bool isQueuePushMethod =
        (subroutine.name == "push_back" || subroutine.name == "push_front");
    bool isQueuePopMethod =
        (subroutine.name == "pop_back" || subroutine.name == "pop_front");
    bool isQueueVoidMethod =
        (subroutine.name == "delete" || subroutine.name == "sort" ||
         subroutine.name == "rsort" || subroutine.name == "shuffle" ||
         subroutine.name == "reverse");
    bool isQueueInsertMethod = (subroutine.name == "insert");

    if (isQueuePushMethod && args.size() == 2) {
      // First arg is the queue (need lvalue), second is the element
      Value queueRef = context.convertLvalueExpression(*args[0]);
      Value element = context.convertRvalueExpression(*args[1]);
      if (!queueRef || !element)
        return {};
      result = context.convertQueueMethodCall(subroutine, loc, queueRef, element);
      if (failed(result))
        return {};
      if (*result)
        return *result;
    }

    if (isQueuePopMethod && args.size() == 1) {
      // Only the queue reference, return element type
      Value queueRef = context.convertLvalueExpression(*args[0]);
      if (!queueRef)
        return {};
      auto ty = context.convertType(*expr.type);
      result = context.convertQueueMethodCallNoArg(subroutine, loc, queueRef, ty);
      if (failed(result))
        return {};
      if (*result)
        return *result;
    }

    if (isQueueInsertMethod && args.size() == 3) {
      // queue.insert(index, elem) -- proper implementation
      Value queueRef = context.convertLvalueExpression(*args[0]);
      Value index = context.convertRvalueExpression(*args[1]);
      Value element = context.convertRvalueExpression(*args[2]);
      if (!queueRef || !index || !element)
        return {};
      if (auto refTy = dyn_cast<moore::RefType>(queueRef.getType()))
        if (auto queueTy = dyn_cast<moore::QueueType>(refTy.getNestedType()))
          element = context.materializeConversion(queueTy.getElementType(),
                                                  element, false, loc);
      // Convert index to i32 for the insert method
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      index = context.materializeConversion(i32Ty, index, false, loc);
      moore::QueueInsertOp::create(builder, loc, queueRef, index, element);
      auto intTy = moore::IntType::getInt(context.getContext(), 1);
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    }

    if (isQueueVoidMethod && args.size() == 1) {
      // Array/queue reference only, returns void
      Value queueRef = context.convertLvalueExpression(*args[0]);
      if (!queueRef)
        return {};
      result = context.convertArrayVoidMethodCall(subroutine, loc, queueRef);
      if (failed(result))
        return {};
      if (*result)
        return *result;
    }

    // Handle array/queue delete(key/index) method - 2 args: array ref + key/index
    if (subroutine.name == "delete" && args.size() == 2) {
      Value arrayRef = context.convertLvalueExpression(*args[0]);
      Value keyOrIndex = context.convertRvalueExpression(*args[1]);
      if (!arrayRef || !keyOrIndex)
        return {};
      auto refType = dyn_cast<moore::RefType>(arrayRef.getType());
      if (refType) {
        auto nestedType = refType.getNestedType();
        if (isa<moore::AssocArrayType>(nestedType)) {
          // Associative array delete(key)
          moore::AssocArrayDeleteKeyOp::create(builder, loc, arrayRef,
                                               keyOrIndex);
        } else if (isa<moore::QueueType>(nestedType)) {
          // Queue delete(index) - index must be IntType
          keyOrIndex = context.convertToSimpleBitVector(keyOrIndex);
          if (!keyOrIndex)
            return {};
          moore::QueueDeleteOp::create(builder, loc, arrayRef, keyOrIndex);
        } else {
          return {};
        }
        // delete returns void, return a dummy value
        auto intTy = moore::IntType::getInt(context.getContext(), 1);
        return moore::ConstantOp::create(builder, loc, intTy, 0);
      }
    }

    // Handle enum iteration methods: first, next, last, prev
    // IEEE 1800-2017 Section 6.19.5: Enum methods for iteration.
    // These take 1 arg (the enum value) and return the same enum type.
    bool isEnumIterMethod = (subroutine.name == "first" ||
                             subroutine.name == "next" ||
                             subroutine.name == "last" ||
                             subroutine.name == "prev");
    if (isEnumIterMethod && args.size() == 1) {
      const slang::ast::Type *slangType = args[0]->type;
      const auto &canonicalType = slangType->getCanonicalType();
      if (canonicalType.kind == slang::ast::SymbolKind::EnumType) {
        const auto &enumType =
            static_cast<const slang::ast::EnumType &>(canonicalType);
        Value enumValue = context.convertRvalueExpression(*args[0]);
        if (!enumValue)
          return {};

        // Get the integer type for the enum
        auto enumIntType = dyn_cast<moore::IntType>(enumValue.getType());
        if (!enumIntType) {
          mlir::emitError(loc) << "enum value has non-integer type: "
                               << enumValue.getType();
          return {};
        }

        // Collect enum values in declaration order
        SmallVector<std::pair<FVInt, std::string>> enumValues;
        for (const auto &member : enumType.values()) {
          const auto &cv = member.getValue();
          if (cv)
            enumValues.emplace_back(convertSVIntToFVInt(cv.integer()),
                                    std::string(member.name));
        }

        if (enumValues.empty()) {
          mlir::emitError(loc) << "enum type has no values";
          return {};
        }

        // Handle the different methods
        if (subroutine.name == "first") {
          // first() returns the first declared value
          auto &firstVal = enumValues.front();
          auto constFvint = firstVal.first;
          if (constFvint.getBitWidth() != enumIntType.getWidth())
            constFvint = constFvint.zext(enumIntType.getWidth());
          return moore::ConstantOp::create(builder, loc, enumIntType,
                                           constFvint);
        } else if (subroutine.name == "last") {
          // last() returns the last declared value
          auto &lastVal = enumValues.back();
          auto constFvint = lastVal.first;
          if (constFvint.getBitWidth() != enumIntType.getWidth())
            constFvint = constFvint.zext(enumIntType.getWidth());
          return moore::ConstantOp::create(builder, loc, enumIntType,
                                           constFvint);
        } else {
          // next() and prev() need runtime lookup
          // Build a chain of conditionals to find the next/prev value
          // For next: if value == v[0] return v[1], elif value == v[1] return v[2], ... else return v[0]
          // For prev: if value == v[n-1] return v[n-2], elif value == v[n-2] return v[n-3], ... else return v[n-1]
          bool isNext = (subroutine.name == "next");

          // Default value: for next, wrap to first; for prev, wrap to last
          auto defaultFvint = isNext ? enumValues.front().first : enumValues.back().first;
          if (defaultFvint.getBitWidth() != enumIntType.getWidth())
            defaultFvint = defaultFvint.zext(enumIntType.getWidth());
          Value result = moore::ConstantOp::create(builder, loc, enumIntType,
                                                   defaultFvint);

          // Build conditional chain (traverse from last to first for proper nesting)
          size_t n = enumValues.size();
          for (size_t i = n; i > 0; --i) {
            size_t idx = i - 1;
            // For next: if value == enumValues[idx], return enumValues[(idx+1) % n]
            // For prev: if value == enumValues[idx], return enumValues[(idx+n-1) % n]
            size_t nextIdx = isNext ? ((idx + 1) % n) : ((idx + n - 1) % n);

            auto constFvint = enumValues[idx].first;
            if (constFvint.getBitWidth() != enumIntType.getWidth())
              constFvint = constFvint.zext(enumIntType.getWidth());
            auto constVal = moore::ConstantOp::create(builder, loc, enumIntType,
                                                      constFvint);

            // Compare enum value with this constant
            Value cond = moore::EqOp::create(builder, loc, enumValue, constVal);
            cond = context.convertToBool(cond);

            // Get the next/prev value
            auto nextFvint = enumValues[nextIdx].first;
            if (nextFvint.getBitWidth() != enumIntType.getWidth())
              nextFvint = nextFvint.zext(enumIntType.getWidth());
            auto nextVal = moore::ConstantOp::create(builder, loc, enumIntType,
                                                     nextFvint);

            // Create the conditional op with regions
            auto conditionalOp = moore::ConditionalOp::create(
                builder, loc, enumIntType, cond);
            auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
            auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

            {
              OpBuilder::InsertionGuard g(builder);
              // True branch: return the next/prev enum value
              builder.setInsertionPointToStart(&trueBlock);
              moore::YieldOp::create(builder, loc, nextVal);

              // False branch: return the accumulated result
              builder.setInsertionPointToStart(&falseBlock);
              moore::YieldOp::create(builder, loc, result);
            }

            result = conditionalOp.getResult();
          }

          return result;
        }
      }
    }

    // Handle associative array iterator methods: first, next, last, prev
    // These take 2 args: array ref + key ref, and return int (1 if found)
    bool isAssocIterMethod = (subroutine.name == "first" ||
                              subroutine.name == "next" ||
                              subroutine.name == "last" ||
                              subroutine.name == "prev");
    if (isAssocIterMethod && args.size() == 2) {
      Value arrayRef = context.convertLvalueExpression(*args[0]);
      Value keyRef = context.convertLvalueExpression(*args[1]);
      if (!arrayRef || !keyRef)
        return {};
      // Check if it's an associative array
      auto refType = dyn_cast<moore::RefType>(arrayRef.getType());
      if (refType && isa<moore::AssocArrayType>(refType.getNestedType())) {
        Value found;
        if (subroutine.name == "first")
          found = moore::AssocArrayFirstOp::create(builder, loc, arrayRef,
                                                   keyRef);
        else if (subroutine.name == "next")
          found =
              moore::AssocArrayNextOp::create(builder, loc, arrayRef, keyRef);
        else if (subroutine.name == "last")
          found =
              moore::AssocArrayLastOp::create(builder, loc, arrayRef, keyRef);
        else // prev
          found =
              moore::AssocArrayPrevOp::create(builder, loc, arrayRef, keyRef);
        // IEEE 1800-2017 §7.8.2: these methods return int (signed i32).
        // The Moore ops return i1; zero-extend to i32 so that the implicit
        // signed conversion from slang does not sign-extend 1 to -1.
        auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
        found = context.materializeConversion(i32Ty, found,
                                              /*isSigned=*/false, loc);
        return found;
      }
    }

    // $typename returns the type of its argument as a string.
    // IEEE 1800-2017 Section 20.6.1: $typename returns a string representation
    // of the data type of its argument. This is evaluated at compile time.
    if (!subroutine.name.compare("$typename") && args.size() == 1) {
      // Get the type of the argument from slang and convert it to a string
      std::string typeName = args[0]->type->toString();
      auto stringType = moore::StringType::get(context.getContext());
      return moore::ConstantStringOp::create(builder, loc, stringType, typeName);
    }

    // Handle .name() method on enum types.
    // IEEE 1800-2017 Section 6.19.5.5: Returns the string representation of
    // the enum value's name, or empty string if value is not a member.
    if (!subroutine.name.compare("name") && args.size() == 1) {
      const slang::ast::Type *slangType = args[0]->type;
      // Get the canonical type to handle typedefs
      const auto &canonicalType = slangType->getCanonicalType();
      if (canonicalType.kind == slang::ast::SymbolKind::EnumType) {
        const auto &enumType =
            static_cast<const slang::ast::EnumType &>(canonicalType);
        Value enumValue = context.convertRvalueExpression(*args[0]);
        if (!enumValue)
          return {};

        auto stringType = moore::StringType::get(context.getContext());

        // Helper to create a string constant from a string literal
        auto createStringConstant = [&](const std::string &str) -> Value {
          if (str.empty()) {
            // For empty string, use a 1-bit integer
            auto intTy = moore::IntType::getInt(context.getContext(), 8);
            auto immInt = moore::ConstantStringOp::create(builder, loc, intTy, "")
                              .getResult();
            return moore::IntToStringOp::create(builder, loc, immInt).getResult();
          }
          auto intTy = moore::IntType::getInt(context.getContext(), str.size() * 8);
          auto immInt = moore::ConstantStringOp::create(builder, loc, intTy, str)
                            .getResult();
          return moore::IntToStringOp::create(builder, loc, immInt).getResult();
        };

        // Collect enum value-name pairs
        SmallVector<std::pair<FVInt, std::string>> enumValues;
        for (const auto &member : enumType.values()) {
          const auto &cv = member.getValue();
          if (cv)
            enumValues.emplace_back(convertSVIntToFVInt(cv.integer()),
                                    std::string(member.name));
        }

        // Build a chain of conditional expressions: if value == enum1 then
        // "enum1" else if value == enum2 then "enum2" else ... else ""
        // Start with the default empty string result
        Value result = createStringConstant("");

        // Get the integer type for enum value comparison
        auto enumIntType = dyn_cast<moore::IntType>(enumValue.getType());
        if (!enumIntType) {
          // Enum type might not have been converted to IntType directly
          // Try to get the underlying type
          mlir::emitError(loc) << "enum value has non-integer type: "
                               << enumValue.getType();
          return {};
        }

        // Build conditional chain from last to first
        for (auto it = enumValues.rbegin(); it != enumValues.rend(); ++it) {
          auto &[fvint, memberName] = *it;
          // Create constant for this enum value with matching width
          auto constFvint = fvint;
          if (constFvint.getBitWidth() != enumIntType.getWidth()) {
            constFvint = constFvint.zext(enumIntType.getWidth());
          }
          auto constVal = moore::ConstantOp::create(
              builder, loc, enumIntType, constFvint);
          // Compare enum value with this constant
          Value cond = moore::EqOp::create(builder, loc, enumValue, constVal);
          cond = context.convertToBool(cond);
          // Create the conditional op with regions
          auto conditionalOp = moore::ConditionalOp::create(
              builder, loc, stringType, cond);
          auto &trueBlock = conditionalOp.getTrueRegion().emplaceBlock();
          auto &falseBlock = conditionalOp.getFalseRegion().emplaceBlock();

          {
            OpBuilder::InsertionGuard g(builder);
            // True branch: return the enum member name
            builder.setInsertionPointToStart(&trueBlock);
            auto nameStr = createStringConstant(memberName);
            moore::YieldOp::create(builder, loc, nameStr);

            // False branch: return the accumulated result
            builder.setInsertionPointToStart(&falseBlock);
            moore::YieldOp::create(builder, loc, result);
          }

          result = conditionalOp.getResult();
        }

        return result;
      }
    }

    // Handle $countbits separately since it takes variable arguments.
    // $countbits(expression, control_bit...) counts bits matching any control_bit.
    // IEEE 1800-2017 Section 20.9 "Bit vector system functions"
    if (subroutine.name == "$countbits" && args.size() >= 2) {
      value = context.convertRvalueExpression(*args[0]);
      if (!value)
        return {};
      value = context.convertToSimpleBitVector(value);
      if (!value)
        return {};

      // Parse control_bit arguments to build control_bits mask:
      // - 0b0001 (1): count zeros
      // - 0b0010 (2): count ones
      // - 0b0100 (4): count X values
      // - 0b1000 (8): count Z values
      int32_t controlBitsMask = 0;
      for (size_t i = 1; i < args.size(); ++i) {
        // Check if the argument is an unbased unsized integer literal ('0, '1,
        // 'x, 'z). These are the only valid control_bit values per IEEE
        // 1800-2017 Section 20.9.
        if (auto *literal =
                args[i]->as_if<slang::ast::UnbasedUnsizedIntegerLiteral>()) {
          auto logicVal = literal->getLiteralValue();
          if (exactlyEqual(logicVal, slang::logic_t(0))) {
            controlBitsMask |= 1; // count zeros
          } else if (exactlyEqual(logicVal, slang::logic_t(1))) {
            controlBitsMask |= 2; // count ones
          } else if (exactlyEqual(logicVal, slang::logic_t::x)) {
            controlBitsMask |= 4; // count X values
          } else if (exactlyEqual(logicVal, slang::logic_t::z)) {
            controlBitsMask |= 8; // count Z values
          } else {
            mlir::emitError(loc)
                << "$countbits control_bit must be '0, '1, 'x, or 'z";
            return {};
          }
          continue;
        }

        // For other expressions, evaluate as a constant
        auto evalResult = context.evaluateConstant(*args[i]);
        if (evalResult.bad() || !evalResult.isInteger()) {
          mlir::emitError(loc)
              << "$countbits control_bit arguments must be constants";
          return {};
        }
        auto intVal = evalResult.integer().as<int32_t>();
        if (!intVal) {
          mlir::emitError(loc) << "$countbits control_bit value out of range";
          return {};
        }
        switch (*intVal) {
        case 0:
          controlBitsMask |= 1;
          break; // count zeros
        case 1:
          controlBitsMask |= 2;
          break; // count ones
        default:
          mlir::emitError(loc)
              << "$countbits control_bit must be '0, '1, 'x, or 'z";
          return {};
        }
      }

      auto intAttr = builder.getI32IntegerAttr(controlBitsMask);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(
          ty, moore::CountBitsBIOp::create(builder, loc, value, intAttr),
          expr.type->isSigned(), loc);
    }

    // Handle $ferror separately since it has an output argument.
    // $ferror(fd, str) returns error code and writes error message to str.
    // IEEE 1800-2017 Section 21.3.1 "File I/O system functions"
    if (subroutine.name == "$ferror" && args.size() == 2) {
      // First argument: file descriptor (input) — convert to i32
      value = context.convertRvalueExpression(*args[0]);
      if (!value)
        return {};
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      value = context.materializeConversion(i32Ty, value, false, loc);
      if (!value)
        return {};
      // Second argument: string output - wrapped as AssignmentExpression
      // Slang produces AssignmentExpression(str = EmptyArgument)
      Value strLhs;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>()) {
        strLhs = context.convertLvalueExpression(assignExpr->left());
        if (!strLhs)
          return {};
      } else {
        mlir::emitError(loc) << "$ferror second argument must be an output string";
        return {};
      }
      // Create the ferror operation
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto errCode = moore::FErrorBIOp::create(builder, loc, intTy, value, strLhs);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, errCode, expr.type->isSigned(), loc);
    }

    // Handle $fgets separately since it has an output argument.
    // $fgets(str, fd) returns number of chars read and writes to str.
    // IEEE 1800-2017 Section 21.3.3 "File input functions"
    if (subroutine.name == "$fgets" && args.size() == 2) {
      // First argument: string output - wrapped as AssignmentExpression
      // Slang produces AssignmentExpression(str = EmptyArgument)
      Value strLhs;
      if (auto *assignExpr =
              args[0]->as_if<slang::ast::AssignmentExpression>()) {
        strLhs = context.convertLvalueExpression(assignExpr->left());
        if (!strLhs)
          return {};
      } else {
        mlir::emitError(loc) << "$fgets first argument must be an output string";
        return {};
      }
      // Second argument: file descriptor (input)
      value = context.convertRvalueExpression(*args[1]);
      if (!value)
        return {};
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      value = context.materializeConversion(i32Ty, value, false, loc);
      if (!value)
        return {};
      // Create the fgets operation
      auto charCount =
          moore::FGetSBIOp::create(builder, loc, i32Ty, strLhs, value);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, charCount, expr.type->isSigned(), loc);
    }

    // Handle $fseek separately since it has 3 arguments.
    // $fseek(fd, offset, operation) sets file position.
    // IEEE 1800-2017 Section 21.3.3 "File positioning functions"
    if (subroutine.name == "$fseek" && args.size() == 3) {
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return {};
      auto offset = context.convertRvalueExpression(*args[1]);
      if (!offset)
        return {};
      auto operation = context.convertRvalueExpression(*args[2]);
      if (!operation)
        return {};
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      fd = context.materializeConversion(intTy, fd, false, loc);
      if (!fd)
        return {};
      offset = context.materializeConversion(intTy, offset, false, loc);
      if (!offset)
        return {};
      operation = context.materializeConversion(intTy, operation, false, loc);
      if (!operation)
        return {};
      auto result = moore::FSeekBIOp::create(builder, loc, intTy, fd, offset, operation);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(), loc);
    }

    // Handle $fread separately since it has an output argument.
    // $fread(dest, fd) reads binary data from file into dest.
    // IEEE 1800-2017 Section 21.3.3 "File input functions"
    if (subroutine.name == "$fread" && args.size() >= 2) {
      // First argument: destination variable - wrapped as AssignmentExpression
      Value destLhs;
      if (auto *assignExpr =
              args[0]->as_if<slang::ast::AssignmentExpression>()) {
        destLhs = context.convertLvalueExpression(assignExpr->left());
        if (!destLhs)
          return {};
      } else {
        mlir::emitError(loc) << "$fread first argument must be an output variable";
        return {};
      }
      // Second argument: file descriptor (input)
      value = context.convertRvalueExpression(*args[1]);
      if (!value)
        return {};
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      value = context.materializeConversion(i32Ty, value, false, loc);
      if (!value)
        return {};
      // Create the fread operation
      auto bytesRead =
          moore::FReadBIOp::create(builder, loc, i32Ty, destLhs, value);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, bytesRead, expr.type->isSigned(), loc);
    }

    // Handle $value$plusargs with runtime support.
    // $value$plusargs(format, var) returns 1 if arg found, 0 otherwise.
    // IEEE 1800-2017 Section 21.6 "Command line input"
    // Emits a runtime call to __moore_value_plusargs which checks
    // CIRCT_UVM_ARGS / UVM_ARGS environment variables.
    if (subroutine.name == "$value$plusargs" && args.size() == 2) {
      // First argument is format string (input).
      Value fmtVal = context.convertRvalueExpression(*args[0]);
      if (!fmtVal)
        return {};

      // Extract format string at compile time.
      std::string fmtStr;
      if (auto constStrOp =
              fmtVal.getDefiningOp<moore::ConstantStringOp>()) {
        fmtStr = constStrOp.getValue().str();
      } else if (auto intToStr =
                     fmtVal.getDefiningOp<moore::IntToStringOp>()) {
        if (auto constStrOp =
                intToStr.getInput()
                    .getDefiningOp<moore::ConstantStringOp>()) {
          fmtStr = constStrOp.getValue().str();
        }
      }

      if (fmtStr.empty()) {
        // Can't determine string at compile time, fall back to 0.
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        auto result = moore::ConstantOp::create(builder, loc, intTy, 0);
        auto ty = context.convertType(*expr.type);
        return context.materializeConversion(ty, result,
                                             expr.type->isSigned(), loc);
      }

      // Second argument: output variable (wrapped as AssignmentExpression).
      Value destLhs;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>()) {
        destLhs = context.convertLvalueExpression(assignExpr->left());
        if (!destLhs)
          return {};
      } else {
        // Fall back to 0 if can't get output variable.
        auto intTy = moore::IntType::getInt(context.getContext(), 32);
        auto result = moore::ConstantOp::create(builder, loc, intTy, 0);
        auto ty = context.convertType(*expr.type);
        return context.materializeConversion(ty, result,
                                             expr.type->isSigned(), loc);
      }

      // Create LLVM global for format string.
      auto module = context.intoModuleOp;
      std::string safeName;
      for (char c : fmtStr)
        safeName += (std::isalnum(c) || c == '_') ? c : '_';
      std::string globalName = "__valueplusarg_" + safeName;

      auto i8Ty = builder.getIntegerType(8);
      auto strLen = static_cast<int64_t>(fmtStr.size());
      auto arrayTy =
          mlir::LLVM::LLVMArrayType::get(i8Ty, strLen + 1);

      if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        mlir::LLVM::GlobalOp::create(
            builder, loc, arrayTy, /*isConstant=*/true,
            mlir::LLVM::Linkage::Internal, globalName,
            builder.getStringAttr(fmtStr + '\0'));
      }

      // Get pointer to global string.
      auto ptrTy = mlir::LLVM::LLVMPointerType::get(context.getContext());
      auto addrOf = mlir::LLVM::AddressOfOp::create(builder, loc, ptrTy,
                                                     globalName);
      auto i32Ty = builder.getIntegerType(32);
      auto lenConst = mlir::LLVM::ConstantOp::create(
          builder, loc, i32Ty, builder.getI32IntegerAttr(strLen));

      // Cast Moore ref to LLVM pointer for the runtime call.
      auto outputPtr = mlir::UnrealizedConversionCastOp::create(
                            builder, loc, ptrTy, destLhs)
                            .getResult(0);

      // Determine output width in bytes.
      auto refType = cast<moore::RefType>(destLhs.getType());
      auto nestedType = refType.getNestedType();
      int64_t outputBytes = 8; // default
      if (auto intType = dyn_cast<moore::IntType>(nestedType))
        outputBytes = (intType.getWidth() + 7) / 8;
      auto bytesConst = mlir::LLVM::ConstantOp::create(
          builder, loc, i32Ty, builder.getI32IntegerAttr(outputBytes));

      // Call __moore_value_plusargs(format_ptr, format_len, output_ptr,
      //                            output_bytes) -> i32
      auto funcTy = mlir::LLVM::LLVMFunctionType::get(
          i32Ty, {ptrTy, i32Ty, ptrTy, i32Ty});
      auto func = getOrCreateRuntimeFunc(
          context, "__moore_value_plusargs", funcTy);
      auto callResult = mlir::LLVM::CallOp::create(
          builder, loc, func,
          ValueRange{addrOf, lenConst, outputPtr, bytesConst});

      auto mooreIntTy =
          moore::IntType::getInt(context.getContext(), 32);
      auto castResult =
          mlir::UnrealizedConversionCastOp::create(
              builder, loc, mooreIntTy, callResult.getResult())
              .getResult(0);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, castResult,
                                           expr.type->isSigned(), loc);
    }

    // Handle $dist_* distribution functions (IEEE 1800-2017 Section 20.15).
    // These have an inout seed as the first argument (bound as lvalue).
    if (subroutine.name == "$dist_chi_square" ||
        subroutine.name == "$dist_exponential" ||
        subroutine.name == "$dist_t" ||
        subroutine.name == "$dist_poisson" ||
        subroutine.name == "$dist_uniform" ||
        subroutine.name == "$dist_normal" ||
        subroutine.name == "$dist_erlang") {
      if (args.size() < 2) {
        mlir::emitError(loc)
            << subroutine.name << " requires at least 2 arguments";
        return {};
      }
      // First arg is inout seed (lvalue ref). Slang wraps inout args in
      // AssignmentExpression — unwrap to get the actual lvalue.
      Value seedRef;
      if (auto *assignExpr =
              args[0]->as_if<slang::ast::AssignmentExpression>()) {
        seedRef = context.convertLvalueExpression(assignExpr->left());
      } else {
        seedRef = context.convertLvalueExpression(*args[0]);
      }
      if (!seedRef)
        return {};
      // Remaining args are rvalue parameters.
      SmallVector<Value> params;
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      for (size_t i = 1; i < args.size(); ++i) {
        Value v = context.convertRvalueExpression(*args[i]);
        if (!v)
          return {};
        if (v.getType() != intTy)
          v = moore::ConversionOp::create(builder, loc, intTy, v);
        params.push_back(v);
      }
      auto result = moore::DistBIOp::create(builder, loc, intTy,
                                            subroutine.name, seedRef, params);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(),
                                           loc);
    }

    // Handle coverage control functions (IEEE 1800-2017 Section 20.14).
    // $coverage_control: returns 1 on success (IEEE 1800-2017 §20.14.1)
    // $coverage_get_max: returns max coverage count (100 for percentage-based)
    // $coverage_merge: returns 0 on success
    // $coverage_save: returns 0 on success
    if (subroutine.name == "$coverage_control") {
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto result = moore::ConstantOp::create(builder, loc, intTy, 1);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(),
                                           loc);
    }
    if (subroutine.name == "$coverage_get_max") {
      // Return 100 as the maximum coverage count (percentage-based).
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto result = moore::ConstantOp::create(builder, loc, intTy, 100);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(),
                                           loc);
    }
    if (subroutine.name == "$coverage_merge" ||
        subroutine.name == "$coverage_save") {
      // Return 0 on success.
      auto intTy = moore::IntType::getInt(context.getContext(), 32);
      auto result = moore::ConstantOp::create(builder, loc, intTy, 0);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(),
                                           loc);
    }

    // $coverage_get / $get_coverage: delegate to runtime for real coverage %
    if (subroutine.name == "$coverage_get" ||
        subroutine.name == "$get_coverage") {
      auto realTy = moore::RealType::get(context.getContext(),
                                         moore::RealWidth::f64);
      auto result = moore::GetCoverageBIOp::create(builder, loc, realTy);
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, result, expr.type->isSigned(),
                                           loc);
    }

    // Helper to check if an argument is an EmptyArgumentExpression.
    auto isEmptyArg = [](const slang::ast::Expression *arg) {
      return arg->kind == slang::ast::ExpressionKind::EmptyArgument;
    };

    // Count non-empty arguments to determine effective arity.
    // EmptyArgumentExpression represents optional arguments that were not
    // provided, so we filter them out when determining the arity.
    size_t effectiveArity = 0;
    for (auto *arg : args)
      if (!isEmptyArg(arg))
        ++effectiveArity;

    // $random(seed) / $urandom(seed): update the seed argument as an inout.
    // Slang may wrap inout-style args in an AssignmentExpression; unwrap it
    // to get the real lvalue.
    if (effectiveArity == 1 &&
        (subroutine.name == "$random" || subroutine.name == "$urandom")) {
      const slang::ast::Expression *seedArgExpr = nullptr;
      for (auto *arg : args) {
        if (!isEmptyArg(arg)) {
          seedArgExpr = arg;
          break;
        }
      }

      if (seedArgExpr) {
        const slang::ast::Expression *seedLValueExpr = seedArgExpr;
        if (auto *assignExpr =
                seedArgExpr->as_if<slang::ast::AssignmentExpression>())
          seedLValueExpr = &assignExpr->left();
        while (auto *convExpr =
                   seedLValueExpr->as_if<slang::ast::ConversionExpression>())
          seedLValueExpr = &convExpr->operand();

        Value seedRef = context.convertLvalueExpression(*seedLValueExpr);
        if (seedRef) {
          auto refTy = dyn_cast<moore::RefType>(seedRef.getType());
          if (!refTy)
            seedRef = {};
          if (!seedRef)
            goto seeded_random_fallback;

          Value seedValue = context.convertRvalueExpression(*seedLValueExpr);
          if (!seedValue)
            return {};

          auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
          seedValue = context.materializeConversion(
              i32Ty, seedValue, seedLValueExpr->type->isSigned(), loc);
          if (!seedValue)
            return {};

          Value randomResult =
              (subroutine.name == "$random")
                  ? static_cast<Value>(
                        moore::RandomBIOp::create(builder, loc, seedValue))
                  : static_cast<Value>(
                        moore::UrandomBIOp::create(builder, loc, seedValue));

          Value seedNext = context.materializeConversion(
              refTy.getNestedType(), randomResult,
              seedLValueExpr->type->isSigned(), loc);
          if (!seedNext)
            return {};
          auto assignOp =
              moore::BlockingAssignOp::create(builder, loc, seedRef, seedNext);
          if (context.variableAssignCallback)
            context.variableAssignCallback(assignOp);

          auto ty = context.convertType(*expr.type);
          return context.materializeConversion(ty, randomResult,
                                               expr.type->isSigned(), loc);
        }
      }
    }
  seeded_random_fallback:

    // Call the conversion function with the appropriate arity. These return one
    // of the following:
    //
    // - `failure()` if the system call was recognized but some error occurred
    // - `Value{}` if the system call was not recognized
    // - non-null `Value` result otherwise
    switch (effectiveArity) {
    case (0):
      result = context.convertSystemCallArity0(subroutine, loc);
      break;

    case (1):
      // Find the first non-empty argument.
      for (auto *arg : args) {
        if (!isEmptyArg(arg)) {
          value = context.convertRvalueExpression(*arg);
          break;
        }
      }
      if (!value)
        return {};
      result = context.convertSystemCallArity1(subroutine, loc, value);
      break;

    case (2): {
      // Find the first two non-empty arguments.
      SmallVector<Value, 2> nonEmptyValues;
      for (auto *arg : args) {
        if (!isEmptyArg(arg)) {
          Value v = context.convertRvalueExpression(*arg);
          if (!v)
            return {};
          nonEmptyValues.push_back(v);
          if (nonEmptyValues.size() == 2)
            break;
        }
      }
      if (nonEmptyValues.size() < 2)
        return {};
      value = nonEmptyValues[0];
      value2 = nonEmptyValues[1];
      result = context.convertSystemCallArity2(subroutine, loc, value, value2);
      break;
    }

    case (3): {
      // Find the first three non-empty arguments.
      SmallVector<Value, 3> nonEmptyValues;
      for (auto *arg : args) {
        if (!isEmptyArg(arg)) {
          Value v = context.convertRvalueExpression(*arg);
          if (!v)
            return {};
          nonEmptyValues.push_back(v);
          if (nonEmptyValues.size() == 3)
            break;
        }
      }
      if (nonEmptyValues.size() < 3)
        return {};
      result = context.convertSystemCallArity3(subroutine, loc,
                                               nonEmptyValues[0],
                                               nonEmptyValues[1],
                                               nonEmptyValues[2]);
      break;
    }

    default:
      break;
    }

    // If we have recognized the system call but the conversion has encountered
    // and already reported an error, simply return the usual null `Value` to
    // indicate failure.
    if (failed(result))
      return {};

    // If we have recognized the system call and got a non-null `Value` result,
    // return that.
    if (*result) {
      auto ty = context.convertType(*expr.type);
      return context.materializeConversion(ty, *result, expr.type->isSigned(),
                                           loc);
    }

    // Otherwise we didn't recognize the system call.
    mlir::emitError(loc) << "unsupported system call `" << subroutine.name
                         << "`";
    return {};
  }

  /// Handle string literals.
  Value visit(const slang::ast::StringLiteral &expr) {
    auto type = context.convertType(*expr.type);
    return moore::ConstantStringOp::create(builder, loc, type, expr.getValue());
  }

  /// Handle real literals.
  Value visit(const slang::ast::RealLiteral &expr) {
    auto fTy = mlir::Float64Type::get(context.getContext());
    auto attr = mlir::FloatAttr::get(fTy, expr.getValue());
    return moore::ConstantRealOp::create(builder, loc, attr).getResult();
  }

  /// Handle null literals.
  Value visit(const slang::ast::NullLiteral &expr) {
    // Null represents a null class handle (no object).
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};

    // For class handles, emit a ClassNullOp.
    if (auto classHandleTy = dyn_cast<moore::ClassHandleType>(type))
      return moore::ClassNullOp::create(builder, loc, classHandleTy);

    // For virtual interfaces, emit a VirtualInterfaceNullOp.
    if (auto vifTy = dyn_cast<moore::VirtualInterfaceType>(type))
      return moore::VirtualInterfaceNullOp::create(builder, loc, vifTy);

    // For other types (like chandle), fall back to uninitialized variable.
    mlir::emitWarning(loc) << "null literal support is incomplete for type "
                           << type << "; treating as uninitialized value";

    auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
    auto nameAttr = builder.getStringAttr("null_literal");
    auto varOp = moore::VariableOp::create(builder, loc, refTy, nameAttr,
                                           /*initial=*/Value{});
    return moore::ReadOp::create(builder, loc, varOp);
  }

  /// Handle unbounded literals (`$`).
  /// In SystemVerilog, `$` represents the last index of a queue/array.
  /// For example, `q[$]` accesses the last element, which is `q[q.size()-1]`.
  Value visit(const slang::ast::UnboundedLiteral &expr) {
    // The `$` literal is only valid in certain contexts, specifically when
    // indexing into a queue or dynamic array. The queue target should have
    // been set by the parent ElementSelectExpression.
    if (!context.queueTargetValue) {
      mlir::emitError(loc) << "unbounded literal ($) used outside of queue "
                           << "or array indexing context";
      return {};
    }

    // Get the size of the queue/array.
    auto sizeValue =
        moore::ArraySizeOp::create(builder, loc, context.queueTargetValue);

    // The `$` represents `size - 1`, i.e., the last valid index.
    auto one = moore::ConstantOp::create(builder, loc, sizeValue.getType(), 1);
    return moore::SubOp::create(builder, loc, sizeValue, one);
  }

  /// Helper function to convert RValues at creation of a new Struct, Array or
  /// Int.
  FailureOr<SmallVector<Value>>
  convertElements(const slang::ast::AssignmentPatternExpressionBase &expr,
                  std::variant<Type, ArrayRef<Type>> expectedTypes,
                  unsigned replCount) {
    const auto &elts = expr.elements();
    const size_t elementCount = elts.size();

    // Inspect the variant.
    const bool hasBroadcast =
        std::holds_alternative<Type>(expectedTypes) &&
        static_cast<bool>(std::get<Type>(expectedTypes)); // non-null Type

    const bool hasPerElem =
        std::holds_alternative<ArrayRef<Type>>(expectedTypes) &&
        !std::get<ArrayRef<Type>>(expectedTypes).empty();

    // If per-element types are provided, enforce arity.
    if (hasPerElem) {
      auto types = std::get<ArrayRef<Type>>(expectedTypes);
      if (types.size() != elementCount) {
        mlir::emitError(loc)
            << "assignment pattern arity mismatch: expected " << types.size()
            << " elements, got " << elementCount;
        return failure();
      }
    }

    SmallVector<Value> converted;
    converted.reserve(elementCount * std::max(1u, replCount));

    // Convert each element heuristically, no type is expected
    if (!hasBroadcast && !hasPerElem) {
      // No expected type info.
      for (const auto *elementExpr : elts) {
        Value v = context.convertRvalueExpression(*elementExpr);
        if (!v)
          return failure();
        converted.push_back(v);
      }
    } else if (hasBroadcast) {
      // Same expected type for all elements.
      Type want = std::get<Type>(expectedTypes);
      for (const auto *elementExpr : elts) {
        Value v = want ? context.convertRvalueExpression(*elementExpr, want)
                       : context.convertRvalueExpression(*elementExpr);
        if (!v)
          return failure();
        converted.push_back(v);
      }
    } else { // hasPerElem, individual type is expected for each element
      auto types = std::get<ArrayRef<Type>>(expectedTypes);
      for (size_t i = 0; i < elementCount; ++i) {
        Type want = types[i];
        const auto *elementExpr = elts[i];
        Value v = want ? context.convertRvalueExpression(*elementExpr, want)
                       : context.convertRvalueExpression(*elementExpr);
        if (!v)
          return failure();
        converted.push_back(v);
      }
    }

    for (unsigned i = 1; i < replCount; ++i)
      converted.append(converted.begin(), converted.begin() + elementCount);

    return converted;
  }

  /// Handle assignment patterns.
  Value visitAssignmentPattern(
      const slang::ast::AssignmentPatternExpressionBase &expr,
      unsigned replCount = 1) {
    auto type = context.convertType(*expr.type);
    const auto &elts = expr.elements();

    // Handle integers.
    if (auto intType = dyn_cast<moore::IntType>(type)) {
      auto elements = convertElements(expr, {}, replCount);

      if (failed(elements))
        return {};

      assert(intType.getWidth() == elements->size());
      std::reverse(elements->begin(), elements->end());
      return moore::ConcatOp::create(builder, loc, intType, *elements);
    }

    // Handle packed structs.
    if (auto structType = dyn_cast<moore::StructType>(type)) {
      SmallVector<Type> expectedTy;
      expectedTy.reserve(structType.getMembers().size());
      for (auto member : structType.getMembers())
        expectedTy.push_back(member.type);

      FailureOr<SmallVector<Value>> elements;
      if (expectedTy.size() == elts.size())
        elements = convertElements(expr, expectedTy, replCount);
      else
        elements = convertElements(expr, {}, replCount);

      if (failed(elements))
        return {};

      assert(structType.getMembers().size() == elements->size());
      return moore::StructCreateOp::create(builder, loc, structType, *elements);
    }

    // Handle unpacked structs.
    if (auto structType = dyn_cast<moore::UnpackedStructType>(type)) {
      SmallVector<Type> expectedTy;
      expectedTy.reserve(structType.getMembers().size());
      for (auto member : structType.getMembers())
        expectedTy.push_back(member.type);

      FailureOr<SmallVector<Value>> elements;
      if (expectedTy.size() == elts.size())
        elements = convertElements(expr, expectedTy, replCount);
      else
        elements = convertElements(expr, {}, replCount);

      if (failed(elements))
        return {};

      assert(structType.getMembers().size() == elements->size());

      return moore::StructCreateOp::create(builder, loc, structType, *elements);
    }

    // Handle packed arrays.
    if (auto arrayType = dyn_cast<moore::ArrayType>(type)) {
      auto elements =
          convertElements(expr, arrayType.getElementType(), replCount);

      if (failed(elements))
        return {};

      assert(arrayType.getSize() == elements->size());
      return moore::ArrayCreateOp::create(builder, loc, arrayType, *elements);
    }

    // Handle unpacked arrays.
    if (auto arrayType = dyn_cast<moore::UnpackedArrayType>(type)) {
      auto elements =
          convertElements(expr, arrayType.getElementType(), replCount);

      if (failed(elements))
        return {};

      assert(arrayType.getSize() == elements->size());
      return moore::ArrayCreateOp::create(builder, loc, arrayType, *elements);
    }

    // Handle queues by constructing an empty queue and pushing elements.
    if (auto queueType = dyn_cast<moore::QueueType>(type)) {
      auto elements =
          convertElements(expr, queueType.getElementType(), replCount);
      if (failed(elements))
        return {};

      // Start from an empty queue (concat with zero inputs).
      Value queueValue =
          moore::QueueConcatOp::create(builder, loc, queueType, {});

      // Materialize a temporary variable to push into.
      auto refTy = moore::RefType::get(queueType);
      auto tmpVar = moore::VariableOp::create(
          builder, loc, refTy, builder.getStringAttr("queue_init_tmp"),
          queueValue);

      for (Value elem : *elements)
        moore::QueuePushBackOp::create(builder, loc, tmpVar, elem);

      return moore::ReadOp::create(builder, loc, tmpVar);
    }

    // Handle associative arrays. Assignment patterns like '{default: value}
    // create an empty associative array. The default value is currently
    // ignored - accessing non-existent keys returns the type's default value.
    // See IEEE 1800-2017 Section 7.9.11.
    if (auto assocType = dyn_cast<moore::AssocArrayType>(type)) {
      // Note: We ignore the default value from the assignment pattern.
      // Full support for associative array default values would require
      // runtime changes to store and return the specified default.
      return moore::AssocArrayCreateOp::create(builder, loc, assocType);
    }

    // Handle wildcard associative arrays similarly.
    if (auto assocType = dyn_cast<moore::WildcardAssocArrayType>(type)) {
      return moore::AssocArrayCreateOp::create(builder, loc, assocType);
    }

    mlir::emitError(loc) << "unsupported assignment pattern with type " << type;
    return {};
  }

  Value visit(const slang::ast::SimpleAssignmentPatternExpression &expr) {
    return visitAssignmentPattern(expr);
  }

  Value visit(const slang::ast::StructuredAssignmentPatternExpression &expr) {
    return visitAssignmentPattern(expr);
  }

  Value visit(const slang::ast::ReplicatedAssignmentPatternExpression &expr) {
    auto count =
        context.evaluateConstant(expr.count()).integer().as<unsigned>();
    assert(count && "Slang guarantees constant non-zero replication count");
    return visitAssignmentPattern(expr, *count);
  }

  Value visit(const slang::ast::StreamingConcatenationExpression &expr) {
    SmallVector<Value> operands;
    // Track if we have any dynamic array operands and their position
    int dynamicArrayIndex = -1;
    Value dynamicArrayValue;

    for (size_t i = 0; i < expr.streams().size(); ++i) {
      auto stream = expr.streams()[i];
      auto operandLoc = context.convertLocation(stream.operand->sourceRange);
      if (!stream.constantWithWidth.has_value() && stream.withExpr) {
        mlir::emitError(operandLoc)
            << "Moore only support streaming "
               "concatenation with fixed size 'with expression'";
        return {};
      }
      Value value;
      if (stream.constantWithWidth.has_value()) {
        value = context.convertRvalueExpression(*stream.withExpr);
        auto type = cast<moore::UnpackedType>(value.getType());
        auto intType = moore::IntType::get(
            context.getContext(), type.getBitSize().value(), type.getDomain());
        // Do not care if it's signed, because we will not do expansion.
        value = context.materializeConversion(intType, value, false, loc);
      } else {
        value = context.convertRvalueExpression(*stream.operand);
      }

      // Handle dynamic array or queue types using runtime streaming operation.
      // These cannot be converted to simple bit vectors at compile time.
      if (isa<moore::QueueType, moore::OpenUnpackedArrayType>(value.getType())) {
        if (expr.streams().size() == 1) {
          // Single dynamic array operand - use StreamConcatOp
          // Determine the result type based on the element type of the
          // queue/dynamic array. String queues produce strings, other types
          // produce integers based on element bit size.
          Type resultType;
          Type elementType;
          if (auto queueType = dyn_cast<moore::QueueType>(value.getType()))
            elementType = queueType.getElementType();
          else if (auto arrayType =
                       dyn_cast<moore::OpenUnpackedArrayType>(value.getType()))
            elementType = arrayType.getElementType();

          if (isa<moore::StringType>(elementType)) {
            // String queue streaming produces a string
            resultType = moore::StringType::get(context.getContext());
          } else {
            // For other element types, produce an integer matching element size
            // The actual size is determined at runtime, but we use element size
            // as the base type for the streaming operation.
            auto unpackedElem = cast<moore::UnpackedType>(elementType);
            if (auto bitSize = unpackedElem.getBitSize()) {
              resultType = moore::IntType::get(context.getContext(), *bitSize,
                                               unpackedElem.getDomain());
            } else {
              // For types without known bit size, use a dynamic result
              resultType = moore::StringType::get(context.getContext());
            }
          }

          // Determine streaming direction from slice size:
          // getSliceSize() == 0 means right-to-left ({>>{}}), otherwise
          // left-to-right ({<<{}}).
          bool isRightToLeft = expr.getSliceSize() != 0;

          return moore::StreamConcatOp::create(builder, loc, resultType, value,
                                               isRightToLeft);
        }

        // Mixed static/dynamic operands - track the dynamic array
        if (dynamicArrayIndex >= 0) {
          // Multiple dynamic arrays not supported
          mlir::emitError(loc) << "streaming concatenation with multiple "
                                  "dynamic array operands not supported";
          return {};
        }
        dynamicArrayIndex = i;
        dynamicArrayValue = value;
        operands.push_back(value); // Placeholder, will be split later
        continue;
      }

      value = context.convertToSimpleBitVector(value);
      if (!value)
        return {};
      operands.push_back(value);
    }

    // Handle mixed static/dynamic streaming
    if (dynamicArrayIndex >= 0) {
      // Collect static prefix and suffix operands
      SmallVector<Value> staticPrefix;
      SmallVector<Value> staticSuffix;

      for (size_t i = 0; i < operands.size(); ++i) {
        if (static_cast<int>(i) < dynamicArrayIndex) {
          staticPrefix.push_back(operands[i]);
        } else if (static_cast<int>(i) > dynamicArrayIndex) {
          staticSuffix.push_back(operands[i]);
        }
      }

      // Determine the result type - for mixed streaming to a queue,
      // we use a queue of the dynamic array's element type
      Type elementType;
      if (auto queueType =
              dyn_cast<moore::QueueType>(dynamicArrayValue.getType()))
        elementType = queueType.getElementType();
      else if (auto arrayType = dyn_cast<moore::OpenUnpackedArrayType>(
                   dynamicArrayValue.getType()))
        elementType = arrayType.getElementType();

      auto unpackedElementType =
          elementType ? dyn_cast<moore::UnpackedType>(elementType) : nullptr;
      if (!unpackedElementType) {
        mlir::emitError(loc)
            << "unsupported streaming concat element type for queue: "
            << elementType;
        return {};
      }

      // Result is a queue of the element type.
      auto resultType = moore::QueueType::get(unpackedElementType, 0);

      // Determine streaming direction from slice size
      bool isRightToLeft = expr.getSliceSize() != 0;
      int32_t sliceSize = expr.getSliceSize();

      return moore::StreamConcatMixedOp::create(
          builder, loc, resultType, staticPrefix, dynamicArrayValue,
          staticSuffix, sliceSize, isRightToLeft);
    }
    Value value;

    if (operands.size() == 1) {
      // There must be at least one element, otherwise slang will report an
      // error.
      value = operands.front();
    } else {
      value = moore::ConcatOp::create(builder, loc, operands).getResult();
    }

    if (expr.getSliceSize() == 0) {
      return value;
    }

    auto type = dyn_cast<moore::IntType>(value.getType());
    if (!type) {
      mlir::emitError(loc) << "streaming concatenation expected IntType, got "
                           << value.getType();
      return {};
    }
    SmallVector<Value> slicedOperands;
    auto iterMax = type.getWidth() / expr.getSliceSize();
    auto remainSize = type.getWidth() % expr.getSliceSize();

    for (size_t i = 0; i < iterMax; i++) {
      auto extractResultType = moore::IntType::get(
          context.getContext(), expr.getSliceSize(), type.getDomain());

      auto extracted = moore::ExtractOp::create(builder, loc, extractResultType,
                                                value, i * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }
    // Handle other wire
    if (remainSize) {
      auto extractResultType = moore::IntType::get(
          context.getContext(), remainSize, type.getDomain());

      auto extracted =
          moore::ExtractOp::create(builder, loc, extractResultType, value,
                                   iterMax * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }

    return moore::ConcatOp::create(builder, loc, slicedOperands);
  }

  Value visit(const slang::ast::AssertionInstanceExpression &expr) {
    context.pushAssertionPortScope();
    context.pushAssertionLocalVarScope();
    auto scopeGuard = llvm::make_scope_exit([&] {
      context.popAssertionLocalVarScope();
      context.popAssertionPortScope();
    });

    DenseMap<const slang::ast::AssertionPortSymbol *,
             const slang::ast::AssertionInstanceExpression::ActualArg *>
        actualArgs;
    for (const auto &arg : expr.arguments) {
      auto *formal = std::get<0>(arg);
      if (!formal)
        continue;
      actualArgs[formal] = &std::get<1>(arg);
      AssertionPortBinding binding;
      if (auto *exprArg = std::get_if<const slang::ast::Expression *>(
              &std::get<1>(arg))) {
        binding.kind = AssertionPortBinding::Kind::Expr;
        binding.expr = *exprArg;
      } else if (auto *assertArg =
                     std::get_if<const slang::ast::AssertionExpr *>(
                         &std::get<1>(arg))) {
        binding.kind = AssertionPortBinding::Kind::AssertionExpr;
        binding.assertionExpr = *assertArg;
      } else if (auto *timingArg =
                     std::get_if<const slang::ast::TimingControl *>(
                         &std::get<1>(arg))) {
        binding.kind = AssertionPortBinding::Kind::TimingControl;
        binding.timingControl = *timingArg;
      }
      context.setAssertionPortBinding(formal, binding);
    }

    auto convertActualArg =
        [&](const slang::ast::AssertionInstanceExpression::ActualArg &actual)
        -> Value {
      if (auto *exprArg =
              std::get_if<const slang::ast::Expression *>(&actual)) {
        if (!*exprArg)
          return {};
        return context.convertRvalueExpression(**exprArg);
      }
      if (auto *assertArg =
              std::get_if<const slang::ast::AssertionExpr *>(&actual)) {
        if (!*assertArg)
          return {};
        return context.convertAssertionExpression(**assertArg, loc,
                                                  /*applyDefaults=*/false);
      }
      if (std::get_if<const slang::ast::TimingControl *>(&actual)) {
        mlir::emitError(loc)
            << "assertion timing control arguments are not yet supported";
        return {};
      }
      return {};
    };

    for (auto *local : expr.localVars) {
      if (!local || !local->formalPort)
        continue;
      auto it = actualArgs.find(local->formalPort);
      if (it == actualArgs.end())
        continue;
      Value bound = convertActualArg(*it->second);
      if (!bound)
        return {};
      context.setAssertionLocalVarBinding(
          local, bound, context.getAssertionSequenceOffset());
    }

    // Defaults apply at the outer assertion; avoid double-applying default
    // clocking/disable when instantiating a named property.
    auto *parentScope = expr.symbol.getParentScope();
    auto *instBody =
        parentScope ? parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()
                    : nullptr;
    if (instBody && instBody->getDefinition().definitionKind ==
                        slang::ast::DefinitionKind::Interface) {
      Value interfaceInstance;

      // Try to extract the interface instance from the syntax tree.
      if (expr.syntax && context.currentScope) {
        const slang::syntax::ExpressionSyntax *ifaceExprSyntax = nullptr;
        if (auto *invocation =
                expr.syntax->as_if<slang::syntax::InvocationExpressionSyntax>()) {
          if (auto *memberAccess = invocation->left->as_if<
                  slang::syntax::MemberAccessExpressionSyntax>()) {
            ifaceExprSyntax = memberAccess->left;
          } else if (auto *scopedName =
                         invocation->left->as_if<slang::syntax::ScopedNameSyntax>()) {
            ifaceExprSyntax = scopedName->left;
          }
        } else if (auto *memberAccess =
                       expr.syntax->as_if<
                           slang::syntax::MemberAccessExpressionSyntax>()) {
          ifaceExprSyntax = memberAccess->left;
        } else if (auto *scopedName =
                       expr.syntax->as_if<slang::syntax::ScopedNameSyntax>()) {
          ifaceExprSyntax = scopedName->left;
        }

        if (ifaceExprSyntax) {
          slang::ast::ASTContext astContext(*context.currentScope,
                                            slang::ast::LookupLocation::max);
          const auto &ifaceExpr =
              slang::ast::Expression::bind(*ifaceExprSyntax, astContext);
          if (!ifaceExpr.bad())
            interfaceInstance = context.convertRvalueExpression(ifaceExpr);
        }
      }

      // If we're already inside an interface method, reuse the implicit arg.
      if (!interfaceInstance && context.currentInterfaceArg &&
          context.currentInterfaceBody == instBody)
        interfaceInstance = context.currentInterfaceArg;

      // Fall back to a matching tracked interface instance.
      if (!interfaceInstance) {
        for (const auto &[instSym, instValue] : context.interfaceInstances) {
          if (&instSym->body == instBody) {
            interfaceInstance = instValue;
            break;
          }
        }
      }

      if (!interfaceInstance) {
        mlir::emitError(loc)
            << "interface property instantiation requires interface instance "
               "for interface '"
            << instBody->getDefinition().name << "'";
        return {};
      }

      if (auto refTy =
              dyn_cast<moore::RefType>(interfaceInstance.getType())) {
        interfaceInstance = moore::ReadOp::create(builder, loc, interfaceInstance);
      }

      auto prevInterfaceArg = context.currentInterfaceArg;
      auto prevInterfaceBody = context.currentInterfaceBody;
      auto prevSignalNames = std::move(context.interfaceSignalNames);
      context.currentInterfaceArg = interfaceInstance;
      context.currentInterfaceBody = instBody;
      context.interfaceSignalNames.clear();
      auto restore = llvm::make_scope_exit([&] {
        context.currentInterfaceArg = prevInterfaceArg;
        context.currentInterfaceBody = prevInterfaceBody;
        context.interfaceSignalNames = std::move(prevSignalNames);
      });

      llvm::DenseSet<const slang::ast::Symbol *> portInternalSymbols;
      for (auto *symbol : instBody->getPortList()) {
        if (const auto *port = symbol->as_if<slang::ast::PortSymbol>()) {
          if (port->internalSymbol) {
            portInternalSymbols.insert(port->internalSymbol);
            context.interfaceSignalNames[port->internalSymbol] = port->name;
          }
          context.interfaceSignalNames[port] = port->name;
        } else if (const auto *multiPort =
                       symbol->as_if<slang::ast::MultiPortSymbol>()) {
          for (auto *port : multiPort->ports) {
            if (port->internalSymbol) {
              portInternalSymbols.insert(port->internalSymbol);
              context.interfaceSignalNames[port->internalSymbol] = port->name;
            }
            context.interfaceSignalNames[port] = port->name;
          }
        }
      }

      for (auto &member : instBody->members()) {
        if (member.as_if<slang::ast::PortSymbol>() ||
            member.as_if<slang::ast::MultiPortSymbol>())
          continue;
        if (portInternalSymbols.count(&member))
          continue;
        if (auto *var = member.as_if<slang::ast::VariableSymbol>()) {
          context.interfaceSignalNames[var] = var->name;
          continue;
        }
        if (auto *net = member.as_if<slang::ast::NetSymbol>()) {
          context.interfaceSignalNames[net] = net->name;
          continue;
        }
        if (auto *inst = member.as_if<slang::ast::InstanceSymbol>()) {
          if (inst->getDefinition().definitionKind ==
              slang::ast::DefinitionKind::Interface) {
            context.interfaceSignalNames[inst] = inst->name;
          }
        }
      }

      return context.convertAssertionExpression(expr.body, loc,
                                                /*applyDefaults=*/false);
    }

    return context.convertAssertionExpression(expr.body, loc,
                                              /*applyDefaults=*/false);
  }

  // Handle dynamic array new[size] expression.
  // In SystemVerilog: arr = new[size]; or arr = new[size](existing);
  Value visit(const slang::ast::NewArrayExpression &expr) {
    auto type = context.convertType(*expr.type);
    if (!type)
      return {};

    auto arrayType = dyn_cast<moore::OpenUnpackedArrayType>(type);
    if (!arrayType) {
      mlir::emitError(loc) << "new[] expression must create a dynamic array, "
                           << "got " << type;
      return {};
    }

    // Convert the size expression.
    auto sizeValue = context.convertRvalueExpression(expr.sizeExpr());
    if (!sizeValue)
      return {};

    // Convert size to i32 if needed.
    auto i32Type =
        moore::IntType::get(context.getContext(), 32, Domain::TwoValued);
    sizeValue = context.materializeConversion(i32Type, sizeValue, false, loc);

    // Convert the optional initializer expression.
    Value initValue;
    if (const auto *initExpr = expr.initExpr()) {
      initValue = context.convertRvalueExpression(*initExpr);
      if (!initValue)
        return {};
    }

    return moore::DynArrayNewOp::create(builder, loc, arrayType, sizeValue,
                                        initValue);
  }

  // A new class expression can stand for one of two things:
  // 1) A call to the `new` method (ctor) of a class made outside the scope of
  // the class
  // 2) A call to the `super.new` method, i.e. the constructor of the base
  // class, within the scope of a class, more specifically, within the new
  // method override of a class.
  // In the first case we should emit an allocation and a call to the ctor if it
  // exists (it's optional in System Verilog), in the second case we should emit
  // a call to the parent's ctor (System Verilog only has single inheritance, so
  // super is always unambiguous), but no allocation, as the child class' new
  // invocation already allocated space for both its own and its parent's
  // properties.
  Value visit(const slang::ast::NewClassExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto classTy = dyn_cast<moore::ClassHandleType>(type);
    Value newObj;

    // We are calling new from within a new function, and it's pointing to
    // super. Check the implicit this ref to figure out the super class type.
    // Do not allocate a new object.
    if (!classTy && expr.isSuperClass) {
      newObj = context.getImplicitThisRef();
      if (!newObj || !newObj.getType() ||
          !isa<moore::ClassHandleType>(newObj.getType())) {
        mlir::emitError(loc) << "implicit this ref was not set while "
                                "converting new class function";
        return {};
      }
      auto thisType = cast<moore::ClassHandleType>(newObj.getType());
      auto classDecl =
          cast<moore::ClassDeclOp>(*context.symbolTable.lookupNearestSymbolFrom(
              context.intoModuleOp, thisType.getClassSym()));
      auto baseClassSym = classDecl.getBase();
      classTy = circt::moore::ClassHandleType::get(context.getContext(),
                                                   baseClassSym.value());
    } else {
      // We are calling from outside a class; allocate space for the object.
      newObj = moore::ClassNewOp::create(builder, loc, classTy, {});
    }

    const auto *constructor = expr.constructorCall();
    // If there's no ctor, emit property initializers directly.
    // IEEE 1800-2017 §8.8: property initializers execute as part of the
    // implicit default constructor.
    if (!constructor) {
      if (auto *classType = expr.type->as_if<slang::ast::ClassType>()) {
        for (auto &member : classType->members()) {
          auto *prop = member.as_if<slang::ast::ClassPropertySymbol>();
          if (!prop)
            continue;
          if (prop->lifetime == slang::ast::VariableLifetime::Static)
            continue;
          auto *init = prop->getInitializer();
          if (!init)
            continue;

          auto propTy = context.convertType(prop->getType());
          if (!propTy)
            continue;

          Value initVal = context.convertRvalueExpression(*init, propTy);
          if (!initVal)
            continue;

          auto fieldRefTy =
              moore::RefType::get(cast<moore::UnpackedType>(propTy));
          auto fieldSym =
              mlir::FlatSymbolRefAttr::get(context.getContext(), prop->name);
          Value fieldRef = moore::ClassPropertyRefOp::create(
              builder, loc, fieldRefTy, newObj, fieldSym);
          moore::BlockingAssignOp::create(builder, loc, fieldRef, initVal);
        }
      }
      return newObj;
    }

    if (const auto *callConstructor =
            constructor->as_if<slang::ast::CallExpression>())
      if (const auto *subroutine =
              std::get_if<const slang::ast::SubroutineSymbol *>(
                  &callConstructor->subroutine)) {
        // Built-in class constructors (e.g., semaphore, mailbox) don't have a
        // thisVar because they're created programmatically, not from source.
        // For these, emit runtime create calls to pass constructor arguments.
        if ((*subroutine)->flags & slang::ast::MethodFlags::BuiltIn) {
          const auto &parentScope =
              (*subroutine)->getParentScope()->asSymbol();
          if (const auto *ct =
                  parentScope.as_if<slang::ast::ClassType>()) {
            if (ct->name == "semaphore") {
              // Emit __moore_semaphore_create(semAddr, keyCount)
              auto i64Ty = builder.getIntegerType(64);
              auto i32Ty = builder.getIntegerType(32);
              auto voidTy =
                  mlir::LLVM::LLVMVoidType::get(context.getContext());

              // Convert object handle to i64 address
              Value semAddr =
                  mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                           i64Ty, newObj)
                      .getResult(0);

              // Get keyCount argument (default 0 per IEEE 1800-2017)
              Value keyCount;
              if (callConstructor->arguments().size() >= 1) {
                keyCount = context.convertRvalueExpression(
                    *callConstructor->arguments()[0]);
                if (!keyCount)
                  return {};
                keyCount = mlir::UnrealizedConversionCastOp::create(
                               builder, loc, i32Ty, keyCount)
                               .getResult(0);
              } else {
                keyCount = builder.create<mlir::LLVM::ConstantOp>(
                    loc, i32Ty, builder.getI32IntegerAttr(0));
              }

              auto createFuncTy = mlir::LLVM::LLVMFunctionType::get(
                  voidTy, {i64Ty, i32Ty});
              auto createFunc = getOrCreateRuntimeFunc(
                  context, "__moore_semaphore_create", createFuncTy);
              mlir::LLVM::CallOp::create(builder, loc, createFunc,
                                         ValueRange{semAddr, keyCount});
            }
          }
          return newObj;
        }
        // For user-defined classes, verify that the constructor has a thisVar.
        if (!(*subroutine)->thisVar) {
          mlir::emitError(loc) << "Expected subroutine called by new to use an "
                                  "implicit this reference";
          return {};
        }
        // Debug: Check that the constructor's thisVar matches the type being constructed
        LLVM_DEBUG({
          const auto &thisVarType = (*subroutine)->thisVar->getType().getCanonicalType();
          llvm::dbgs() << "NewClassExpression: constructing " << expr.type->name
                       << ", constructor thisVar type: " << thisVarType.toString()
                       << "\n";
        });
        if (failed(context.convertFunction(**subroutine)))
          return {};
        // Set methodReceiverOverride so visitCall uses newObj as the
        // constructor's 'this' argument. Do NOT set currentThisRef here -
        // constructor arguments must be evaluated with the CALLER's 'this'
        // so property accesses like m_cntxt resolve to the correct type.
        auto savedOverride = context.methodReceiverOverride;
        context.methodReceiverOverride = newObj;
        auto restoreOverride = llvm::make_scope_exit(
            [&] { context.methodReceiverOverride = savedOverride; });
        // Emit a call to ctor
        if (!visitCall(*callConstructor, *subroutine))
          return {};
        // Return new handle
        return newObj;
      }
    return {};
  }

  // Handle covergroup instantiation: covergroup_type cg_inst = new();
  Value visit(const slang::ast::NewCovergroupExpression &expr) {
    // Convert the covergroup type to get the handle type.
    auto type = context.convertType(*expr.type);
    auto cgTy = dyn_cast<moore::CovergroupHandleType>(type);
    if (!cgTy) {
      mlir::emitError(loc) << "expected covergroup handle type, got " << type;
      return {};
    }

    // Get the covergroup symbol from the type.
    auto cgSym = cgTy.getCovergroupSym();

    // Create the covergroup instantiation op.
    return moore::CovergroupInstOp::create(builder, loc, cgTy, cgSym);
  }

  // Handle shallow copy of class instances: new <source>
  // IEEE 1800-2017 Section 8.12: Shallow copy creates a new object with the
  // same property values as the source object.
  Value visit(const slang::ast::CopyClassExpression &expr) {
    // Convert the source expression (the object being copied).
    auto source = context.convertRvalueExpression(expr.sourceExpr());
    if (!source)
      return {};

    // Get the class handle type from the source.
    auto classTy = dyn_cast<moore::ClassHandleType>(source.getType());
    if (!classTy) {
      mlir::emitError(loc) << "expected class handle type for copy source, got "
                           << source.getType();
      return {};
    }

    // Create the shallow copy operation.
    return moore::ClassCopyOp::create(builder, loc, classTy, source);
  }

  // Handle distribution expression: variable dist { items }
  // This is used within constraints to specify weighted random distributions.
  Value visit(const slang::ast::DistExpression &expr) {
    // Convert the left-hand side as an lvalue to get a reference to the
    // variable being constrained. This is necessary so the constraint can
    // be associated with the variable during randomization.
    auto variable = context.convertLvalueExpression(expr.left());
    if (!variable)
      return {};

    // Collect values, weights, and per_range markers
    // Values are stored as [low, high] pairs for each item (single values
    // have low == high)
    SmallVector<int64_t> values;
    SmallVector<int64_t> weights;
    SmallVector<int64_t> perRange;

    auto isUnboundedDistExpr = [&](const slang::ast::Expression &expr,
                                   const auto &self) -> bool {
      if (expr.type && expr.type->isUnbounded())
        return true;
      if (expr.as_if<slang::ast::UnboundedLiteral>())
        return true;
      if (auto *conv = expr.as_if<slang::ast::ConversionExpression>())
        return self(conv->operand(), self);
      return false;
    };

    auto resolveDistBound =
        [&](const slang::ast::Expression &boundExpr) -> FailureOr<int64_t> {
      if (isUnboundedDistExpr(boundExpr, isUnboundedDistExpr)) {
        if (!expr.left().type->isIntegral()) {
          mlir::emitError(loc)
              << "dist $ requires an integral left-hand side type";
          return failure();
        }
        auto bitWidth = expr.left().type->getBitWidth();
        if (bitWidth == 0 || bitWidth > 64) {
          mlir::emitError(loc)
              << "dist $ requires a fixed-width type (<= 64 bits)";
          return failure();
        }
        bool isSigned = expr.left().type->isSigned();
        if (!isSigned && bitWidth == 64) {
          mlir::emitError(loc)
              << "dist $ for 64-bit unsigned types exceeds int64_t";
          return failure();
        }
        uint64_t maxValue = 0;
        if (isSigned) {
          maxValue = (bitWidth == 1) ? 0 : ((1ULL << (bitWidth - 1)) - 1);
        } else {
          maxValue = (1ULL << bitWidth) - 1;
        }
        return static_cast<int64_t>(maxValue);
      }

      auto val = context.evaluateConstant(boundExpr);
      if (val.bad()) {
        mlir::emitError(loc) << "dist range bounds must be constant";
        return failure();
      }
      auto maybeValue = val.integer().as<int64_t>();
      if (!maybeValue) {
        mlir::emitError(loc) << "dist range bounds must be integers";
        return failure();
      }
      return *maybeValue;
    };

    for (const auto &item : expr.items()) {
      // Check if this is a range or a single value
      if (const auto *range =
              item.value.as_if<slang::ast::ValueRangeExpression>()) {
        // Range expression: [low:high]
        auto leftVal = resolveDistBound(range->left());
        auto rightVal = resolveDistBound(range->right());
        if (failed(leftVal) || failed(rightVal))
          return {};
        values.push_back(*leftVal);
        values.push_back(*rightVal);
      } else {
        // Single value
        auto singleVal = resolveDistBound(item.value);
        if (failed(singleVal))
          return {};
        int64_t v = *singleVal;
        // Single values are represented as a range [v, v]
        values.push_back(v);
        values.push_back(v);
      }

      // Get the weight (default to 1 if not specified)
      int64_t weight = 1;
      int64_t isPerRange = 0; // 0 = := (per item), 1 = :/ (per range)

      if (item.weight.has_value()) {
        auto weightVal = context.evaluateConstant(*item.weight->expr);
        if (!weightVal.bad()) {
          auto maybeWeight = weightVal.integer().as<int64_t>();
          if (maybeWeight)
            weight = *maybeWeight;
        }
        isPerRange =
            (item.weight->kind ==
             slang::ast::DistExpression::DistWeight::PerRange)
                ? 1
                : 0;
      }

      weights.push_back(weight);
      perRange.push_back(isPerRange);
    }

    // Handle default weight if specified
    if (expr.defaultWeight()) {
      // Default weight applies to any value not explicitly listed
      // For now, we don't have a mechanism to represent this in the IR
      mlir::emitWarning(loc) << "default dist weight not yet supported";
    }

    // Create the distribution constraint op.
    auto isSignedAttr =
        expr.left().type->isSigned() ? builder.getUnitAttr() : nullptr;
    moore::ConstraintDistOp::create(builder, loc, variable, values, weights,
                                    perRange, isSignedAttr);

    // DistExpression has void type in slang, but we need to return something
    // for the expression visitor. Return a constant 1 (true) as a placeholder
    // since this is typically used in constraint contexts where the result
    // indicates the constraint is satisfied.
    auto boolType = moore::IntType::getInt(context.getContext(), 1);
    return moore::ConstantOp::create(builder, loc, boolType, 1);
  }

  /// Handle tagged union expressions like `tagged Valid(42)`.
  /// These create a tagged union value by setting the tag and the value
  /// for a specific member.
  Value visit(const slang::ast::TaggedUnionExpression &expr) {
    // Convert the result type (should be struct<{tag, data}> or ustruct<{tag, data}>)
    auto resultType = context.convertType(*expr.type);
    if (!resultType)
      return {};

    // The result type should be a struct with {tag, data} fields
    // Can be either packed (StructType) or unpacked (UnpackedStructType)
    ArrayRef<moore::StructLikeMember> structMembers;
    bool isPacked = false;
    if (auto packedStruct = dyn_cast<moore::StructType>(resultType)) {
      structMembers = packedStruct.getMembers();
      isPacked = true;
    } else if (auto unpackedStruct =
                   dyn_cast<moore::UnpackedStructType>(resultType)) {
      structMembers = unpackedStruct.getMembers();
      isPacked = false;
    } else {
      mlir::emitError(loc)
          << "tagged union type must be struct or ustruct<{tag, data}>";
      return {};
    }

    if (structMembers.size() != 2) {
      mlir::emitError(loc) << "tagged union wrapper must have 2 fields";
      return {};
    }

    auto tagMember = structMembers[0];
    auto dataMember = structMembers[1];

    // Get the underlying union type from the data member
    Type unionType = dataMember.type;
    SmallVector<moore::StructLikeMember> unionMembers;
    if (auto packedUnion = dyn_cast<moore::UnionType>(unionType)) {
      unionMembers.append(packedUnion.getMembers().begin(),
                          packedUnion.getMembers().end());
    } else if (auto unpackedUnion =
                   dyn_cast<moore::UnpackedUnionType>(unionType)) {
      unionMembers.append(unpackedUnion.getMembers().begin(),
                          unpackedUnion.getMembers().end());
    } else {
      mlir::emitError(loc) << "tagged union data field must be a union type";
      return {};
    }

    // Find the tag index for the member being set
    StringRef memberName(expr.member.name);
    unsigned tagIndex = 0;
    bool found = false;
    for (size_t i = 0; i < unionMembers.size(); ++i) {
      if (unionMembers[i].name.getValue() == memberName) {
        tagIndex = i;
        found = true;
        break;
      }
    }
    if (!found) {
      mlir::emitError(loc) << "tagged union member '" << memberName
                           << "' not found";
      return {};
    }

    // Create the tag constant
    auto tagIntType = dyn_cast<moore::IntType>(tagMember.type);
    if (!tagIntType) {
      mlir::emitError(loc) << "tagged union tag must be an integer type";
      return {};
    }
    auto tagValue = moore::ConstantOp::create(builder, loc, tagIntType,
                                              static_cast<int64_t>(tagIndex));

    // Create the union value
    Value unionValue;
    if (expr.valueExpr) {
      // Convert the value expression
      auto value = context.convertRvalueExpression(*expr.valueExpr);
      if (!value)
        return {};
      // Create the union with the converted value
      unionValue = moore::UnionCreateOp::create(
          builder, loc, unionType, value,
          builder.getStringAttr(memberName));
    } else {
      // Void member - create a union with a dummy value.
      // For void members, we need to create a union value even though
      // the member itself has void type. We find the first non-void member
      // and create a zero-initialized value for it, since all union members
      // share the same storage.
      moore::StructLikeMember *firstNonVoidMember = nullptr;
      for (auto &member : unionMembers) {
        if (!isa<moore::VoidType>(member.type)) {
          firstNonVoidMember = &member;
          break;
        }
      }

      if (!firstNonVoidMember) {
        // All members are void - create a 1-bit placeholder union
        auto dummyType = moore::IntType::getInt(context.getContext(), 1);
        auto dummyValue = moore::ConstantOp::create(builder, loc, dummyType, 0);
        // Use the void member name but with an i1 placeholder value
        unionValue = moore::UnionCreateOp::create(
            builder, loc, unionType, dummyValue,
            builder.getStringAttr(memberName));
      } else {
        // Create a zero value for the first non-void member
        Value zeroValue;
        if (auto intType = dyn_cast<moore::IntType>(firstNonVoidMember->type)) {
          zeroValue = moore::ConstantOp::create(builder, loc, intType, 0);
        } else {
          // For other types, try to create a default initialization
          mlir::emitError(loc) << "cannot create default value for tagged "
                               << "union with void member and non-integer "
                               << "first member";
          return {};
        }
        unionValue = moore::UnionCreateOp::create(
            builder, loc, unionType, zeroValue,
            builder.getStringAttr(firstNonVoidMember->name.getValue()));
      }
    }

    // Create the wrapper struct with {tag, data}
    SmallVector<Value> structFields = {tagValue, unionValue};
    if (isPacked) {
      return moore::StructCreateOp::create(
          builder, loc, cast<moore::StructType>(resultType), structFields);
    }
    return moore::StructCreateOp::create(
        builder, loc, cast<moore::UnpackedStructType>(resultType), structFields);
  }

  /// Handle empty argument expressions.
  /// These represent optional arguments that were not provided in the call.
  /// Returns a null Value to indicate the argument is missing, which the
  /// calling code should handle appropriately (e.g., use default values).
  Value visit(const slang::ast::EmptyArgumentExpression &expr) {
    // Return null to indicate this is an empty/missing argument.
    // The caller (e.g., system call handling code) is responsible for
    // detecting this and using appropriate default values.
    return {};
  }

  /// Handle min:typ:max expressions.
  /// These represent timing expressions with minimum, typical, and maximum
  /// values. We convert the selected value based on compilation options.
  Value visit(const slang::ast::MinTypMaxExpression &expr) {
    // The selected() method returns the appropriate value based on which
    // timing mode is selected in compilation options (min, typ, or max).
    return context.convertRvalueExpression(expr.selected());
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }

  /// Handle InvalidExpression by attempting to unwrap and convert the child.
  /// This is particularly useful when DynamicNotProcedural is downgraded to
  /// a warning - slang wraps the original expression in InvalidExpression,
  /// but we can still attempt to convert the underlying expression.
  Value visit(const slang::ast::InvalidExpression &expr) {
    // When allowNonProceduralDynamic is set, try to unwrap and convert the
    // child expression. This allows us to handle class member accesses that
    // slang marks as invalid because they're outside a procedural context.
    if (context.options.allowNonProceduralDynamic.value_or(false)) {
      if (expr.child) {
        // Recursively visit the child expression
        return expr.child->visit(*this);
      }
    }
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Lvalue Conversion
//===----------------------------------------------------------------------===//

namespace {
struct LvalueExprVisitor : public ExprVisitor {
  LvalueExprVisitor(Context &context, Location loc)
      : ExprVisitor(context, loc, /*isLvalue=*/true) {}
  using ExprVisitor::visit;

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
    // Handle inline constraint receivers and compiler-generated 'this' symbols.
    if (auto inlineRef = context.getInlineConstraintThisRef()) {
      if (auto inlineSym = context.getInlineConstraintThisSymbol();
          inlineSym && inlineSym == &expr.symbol) {
        return inlineRef;
      }
      if (expr.symbol.name == "this")
        return inlineRef;
    }
    if (expr.symbol.name == "this") {
      if (auto thisRef = context.getImplicitThisRef())
        return thisRef;
    }

    // Handle local variables.
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;

    // Handle global variables.
    if (auto globalOp = context.globalVariables.lookup(&expr.symbol)) {
      // Use the expression's type rather than the GlobalVariableOp's type.
      // During recursive type conversion, the GlobalVariableOp may temporarily
      // have a placeholder type while the actual type is being determined.
      auto varType = context.convertType(*expr.type);
      if (!varType)
        return {};
      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(varType));
      auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      return moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
    }

    // Try on-demand conversion for global variables that haven't been converted
    // yet. This handles forward references where a variable is used before
    // being visited (e.g., in static method initializers of classes that are
    // converted before the variable itself).
    if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>()) {
      auto parentKind = var->getParentScope()->asSymbol().kind;
      if (parentKind == slang::ast::SymbolKind::Package ||
          parentKind == slang::ast::SymbolKind::Root ||
          parentKind == slang::ast::SymbolKind::CompilationUnit) {
        if (succeeded(context.convertGlobalVariable(*var))) {
          if (auto globalOp = context.globalVariables.lookup(&expr.symbol)) {
            // Use the expression's type rather than the GlobalVariableOp's type.
            auto varType = context.convertType(*expr.type);
            if (!varType)
              return {};
            auto refTy = moore::RefType::get(cast<moore::UnpackedType>(varType));
            auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
            return moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
          }
        }
      }
    }

    if (auto *const property =
            expr.symbol.as_if<slang::ast::ClassPropertySymbol>()) {
      return visitClassProperty(context, *property);
    }

    // Handle virtual interface member access. When accessing a member through
    // a virtual interface (e.g., vif.data), slang gives us a NamedValueExpression
    // where the symbol is the interface member but accessed via a virtual
    // interface variable. This applies to VariableSymbol (output/inout),
    // NetSymbol (input ports which are nets), and ModportPortSymbol
    // (when accessing ports through a modport-qualified virtual interface).
    {
      const slang::ast::Scope *symbolScope = nullptr;
      if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>())
        symbolScope = var->getParentScope();
      else if (auto *net = expr.symbol.as_if<slang::ast::NetSymbol>())
        symbolScope = net->getParentScope();
      else if (auto *modportPort =
                   expr.symbol.as_if<slang::ast::ModportPortSymbol>())
        symbolScope = modportPort->getParentScope();

      if (symbolScope) {
        auto parentKind = symbolScope->asSymbol().kind;
        // For ModportPortSymbol, the parent is a Modport which is inside
        // an InstanceBody. For VariableSymbol/NetSymbol, parent is directly
        // InstanceBody.
        if (parentKind == slang::ast::SymbolKind::InstanceBody ||
            parentKind == slang::ast::SymbolKind::Modport) {
          if (expr.syntax) {
            if (auto result =
                    visitVirtualInterfaceMemberAccess(expr, *expr.syntax))
              return result;
          }
        }
      }
    }

    // Handle interface signal access from within an interface method (lvalue).
    // When we're inside an interface task/function, signal references
    // should use the implicit interface argument.
    if (context.currentInterfaceArg) {
      StringRef signalName;
      auto it = context.interfaceSignalNames.find(&expr.symbol);
      if (it != context.interfaceSignalNames.end()) {
        signalName = it->second;
      } else if (auto *scope = expr.symbol.getParentScope()) {
        if (auto *body =
                scope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()) {
          if (body == context.currentInterfaceBody)
            signalName = expr.symbol.name;
        }
      }

      if (!signalName.empty()) {
        // This is an interface signal access from within an interface context.
        // Use VirtualInterfaceSignalRefOp to access the signal through the
        // implicit interface argument.
        auto type = context.convertType(*expr.type);
        if (!type)
          return {};

        auto signalSym =
            mlir::FlatSymbolRefAttr::get(builder.getContext(), signalName);
        auto refTy = moore::RefType::get(cast<moore::UnpackedType>(type));
        return moore::VirtualInterfaceSignalRefOp::create(
            builder, loc, refTy, context.currentInterfaceArg, signalSym);
      }
    }

    // Handle clocking block signal access (ClockVar) for lvalue.
    // When assigning to a signal through a clocking block (e.g., cb.signal = x),
    // slang gives us a NamedValueExpression where the symbol is a ClockVarSymbol.
    // The ClockVarSymbol has an initializer expression pointing to the
    // underlying signal.
    if (auto *clockVar = expr.symbol.as_if<slang::ast::ClockVarSymbol>()) {
      // Get the initializer expression which references the underlying signal
      auto *initExpr = clockVar->getInitializer();
      if (!initExpr) {
        mlir::emitError(loc)
            << "clocking block signal '" << clockVar->name
            << "' has no underlying signal reference";
        return {};
      }

      // For output signals (lvalue context), we get the underlying signal
      // reference for assignment. The output skew delay would be applied
      // at the clocking block level, not here during individual signal access.
      return context.convertLvalueExpression(*initExpr);
    }

    auto d = mlir::emitError(loc, "unknown name `") << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no lvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  // Handle hierarchical values, such as `Top.sub.var = x`.
  Value visit(const slang::ast::HierarchicalValueExpression &expr) {
    // Handle local variables.
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;

    // Handle global variables.
    if (auto globalOp = context.globalVariables.lookup(&expr.symbol)) {
      // Use the expression's type rather than the GlobalVariableOp's type.
      auto varType = context.convertType(*expr.type);
      if (!varType)
        return {};
      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(varType));
      auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      return moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
    }

    // Handle direct interface member access (e.g., intf.clk = 1 where intf is a
    // direct interface instance, not a virtual interface). Check if the
    // symbol's parent is an interface body. This applies to both VariableSymbol
    // (for output/inout ports) and NetSymbol (for input ports which are nets).
    const slang::ast::Scope *parentScope = nullptr;
    if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>())
      parentScope = var->getParentScope();
    else if (auto *net = expr.symbol.as_if<slang::ast::NetSymbol>())
      parentScope = net->getParentScope();

    if (parentScope) {
      if (auto *instBody =
              parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()) {
        if (instBody->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface) {
          if (auto value = resolveHierarchicalInterfaceSignalRef(
                  expr, expr.symbol.name))
            return value;
        }
      }
    }

    // Handle modport port access for lvalue (e.g., port.clk = value where port
    // has type interface.modport). The expr.symbol is a ModportPortSymbol, and
    // we need to access the actual interface signal via its internalSymbol.
    if (auto *modportPort =
            expr.symbol.as_if<slang::ast::ModportPortSymbol>()) {
      // Get the internal symbol (the actual interface signal)
      auto *internalSym = modportPort->internalSymbol;
      if (internalSym) {
        // Get the parent scope of the internal symbol to verify it's in an
        // interface
        auto *parentScope = internalSym->getParentScope();
        if (parentScope) {
          if (auto *instBody = parentScope->asSymbol()
                                   .as_if<slang::ast::InstanceBodySymbol>()) {
            if (instBody->getDefinition().definitionKind ==
                slang::ast::DefinitionKind::Interface) {
              if (auto value = resolveHierarchicalInterfaceSignalRef(
                      expr, internalSym->name))
                return value;
            }
          }
        }
      }
    }

    // Emit an error for those hierarchical values not recorded in the
    // `valueSymbols`.
    auto d = mlir::emitError(loc, "unknown hierarchical name `")
             << expr.symbol.name << "`";
    d.attachNote(context.convertLocation(expr.symbol.location))
        << "no lvalue generated for " << slang::ast::toString(expr.symbol.kind);
    return {};
  }

  Value visit(const slang::ast::StreamingConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto stream : expr.streams()) {
      auto operandLoc = context.convertLocation(stream.operand->sourceRange);
      if (!stream.constantWithWidth.has_value() && stream.withExpr) {
        mlir::emitError(operandLoc)
            << "Moore only support streaming "
               "concatenation with fixed size 'with expression'";
        return {};
      }
      Value value;
      if (stream.constantWithWidth.has_value()) {
        value = context.convertLvalueExpression(*stream.withExpr);
        auto type = cast<moore::UnpackedType>(
            cast<moore::RefType>(value.getType()).getNestedType());
        auto intType = moore::RefType::get(moore::IntType::get(
            context.getContext(), type.getBitSize().value(), type.getDomain()));
        // Do not care if it's signed, because we will not do expansion.
        value = context.materializeConversion(intType, value, false, loc);
      } else {
        value = context.convertLvalueExpression(*stream.operand);
        // Convert packed type references (like struct references) to simple bit
        // vector references. This is the lvalue equivalent of
        // convertToSimpleBitVector for rvalues.
        // Note: For mixed static/dynamic operands, the conversion and handling
        // is done in the AssignmentExpression visitor using StreamUnpackMixedOp.
        if (value) {
          if (auto refType = dyn_cast<moore::RefType>(value.getType())) {
            // For dynamic arrays/queues, don't convert - they're handled specially
            if (!isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                    refType.getNestedType())) {
              if (auto packed =
                      dyn_cast<moore::PackedType>(refType.getNestedType())) {
                if (!isa<moore::IntType>(packed)) {
                  if (auto bitSize = packed.getBitSize()) {
                    auto intType = moore::RefType::get(moore::IntType::get(
                        context.getContext(), *bitSize, packed.getDomain()));
                    value =
                        context.materializeConversion(intType, value, false, loc);
                  }
                }
              }
            }
          }
        }
      }

      if (!value)
        return {};
      operands.push_back(value);
    }
    Value value;
    if (operands.size() == 1) {
      // There must be at least one element, otherwise slang will report an
      // error.
      value = operands.front();
    } else {
      for (auto operand : operands)
        context.captureRef(operand);
      value = moore::ConcatRefOp::create(builder, loc, operands).getResult();
    }

    if (expr.getSliceSize() == 0) {
      return value;
    }

    auto refType = dyn_cast<moore::RefType>(value.getType());
    if (!refType) {
      mlir::emitError(loc) << "lvalue streaming expected RefType, got "
                           << value.getType();
      return {};
    }

    // Handle dynamic arrays and queues. These cannot be sliced at compile time
    // since their size is runtime-determined. Return the reference directly and
    // let the assignment handling use StreamUnpackOp for these types.
    if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(
            refType.getNestedType())) {
      // For dynamic arrays/queues, we return the reference as-is.
      // The streaming will be handled at runtime by StreamUnpackOp.
      return value;
    }

    auto type = dyn_cast<moore::IntType>(refType.getNestedType());
    if (!type) {
      mlir::emitError(loc) << "lvalue streaming expected IntType, got "
                           << refType.getNestedType();
      return {};
    }
    SmallVector<Value> slicedOperands;
    auto widthSum = type.getWidth();
    auto domain = type.getDomain();
    auto iterMax = widthSum / expr.getSliceSize();
    auto remainSize = widthSum % expr.getSliceSize();

    for (size_t i = 0; i < iterMax; i++) {
      auto extractResultType = moore::RefType::get(moore::IntType::get(
          context.getContext(), expr.getSliceSize(), domain));

      context.captureRef(value);
      auto extracted = moore::ExtractRefOp::create(
          builder, loc, extractResultType, value, i * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }
    // Handle other wire
    if (remainSize) {
      auto extractResultType = moore::RefType::get(
          moore::IntType::get(context.getContext(), remainSize, domain));

      context.captureRef(value);
      auto extracted =
          moore::ExtractRefOp::create(builder, loc, extractResultType, value,
                                      iterMax * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }

    for (auto operand : slicedOperands)
      context.captureRef(operand);
    return moore::ConcatRefOp::create(builder, loc, slicedOperands);
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    return context.convertRvalueExpression(node);
  }

  Value visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }

  /// Handle InvalidExpression by attempting to unwrap and convert the child.
  /// This is particularly useful when DynamicNotProcedural is downgraded to
  /// a warning - slang wraps the original expression in InvalidExpression,
  /// but we can still attempt to convert the underlying expression.
  Value visit(const slang::ast::InvalidExpression &expr) {
    // When allowNonProceduralDynamic is set, try to unwrap and convert the
    // child expression. This allows us to handle class member accesses that
    // slang marks as invalid because they're outside a procedural context.
    if (context.options.allowNonProceduralDynamic.value_or(false)) {
      if (expr.child) {
        // Recursively visit the child expression
        return expr.child->visit(*this);
      }
    }
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

Value Context::convertRvalueExpression(const slang::ast::Expression &expr,
                                       Type requiredType) {
  auto loc = convertLocation(expr.sourceRange);
  auto value = expr.visit(RvalueExprVisitor(*this, loc));
  if (value && requiredType)
    value =
        materializeConversion(requiredType, value, expr.type->isSigned(), loc);
  return value;
}

Value Context::convertLvalueExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange);
  return expr.visit(LvalueExprVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)

/// Helper function to convert a value to its "truthy" boolean value.
Value Context::convertToBool(Value value) {
  if (!value)
    return {};
  if (auto intTy = dyn_cast<IntegerType>(value.getType()))
    if (intTy.getWidth() == 1)
      return value;
  if (auto type = dyn_cast_or_null<moore::IntType>(value.getType()))
    if (type.getBitSize() == 1)
      return value;
  if (auto type = dyn_cast_or_null<moore::UnpackedType>(value.getType()))
    return moore::BoolCastOp::create(builder, value.getLoc(), value);
  mlir::emitError(value.getLoc(), "expression of type ")
      << value.getType() << " cannot be cast to a boolean";
  return {};
}

/// Materialize a Slang real literal as a constant op.
Value Context::materializeSVReal(const slang::ConstantValue &svreal,
                                 const slang::ast::Type &astType,
                                 Location loc) {
  const auto *floatType = astType.as_if<slang::ast::FloatingType>();
  assert(floatType);

  FloatAttr attr;
  if (svreal.isShortReal() &&
      floatType->floatKind == slang::ast::FloatingType::ShortReal) {
    attr = FloatAttr::get(builder.getF32Type(), svreal.shortReal().v);
  } else if (svreal.isReal() &&
             floatType->floatKind == slang::ast::FloatingType::Real) {
    attr = FloatAttr::get(builder.getF64Type(), svreal.real().v);
  } else {
    mlir::emitError(loc) << "invalid real constant";
    return {};
  }

  return moore::ConstantRealOp::create(builder, loc, attr);
}

/// Materialize a Slang string literal as a literal string constant op.
Value Context::materializeString(const slang::ConstantValue &stringLiteral,
                                 const slang::ast::Type &astType,
                                 Location loc) {
  if (!astType.isString())
    return {};

  // Get the actual string content (without quotes).
  // stringLiteral.str() returns the raw string content, while
  // stringLiteral.toString() would add surrounding quotes.
  const std::string &strContent = stringLiteral.str();

  // Calculate the bit width as string length * 8 bits per character.
  // This ensures the integer type can hold the entire string content.
  // IEEE 1800-2017 Section 5.9: strings are stored with first character
  // in the most significant byte.
  unsigned bitWidth = strContent.size() * 8;
  if (bitWidth == 0)
    bitWidth = 8; // Empty string still needs at least 1 byte

  auto intTy = moore::IntType::getInt(getContext(), bitWidth);
  auto immInt = moore::ConstantStringOp::create(builder, loc, intTy, strContent)
                    .getResult();
  return moore::IntToStringOp::create(builder, loc, immInt).getResult();
}

/// Materialize a Slang integer literal as a constant op.
Value Context::materializeSVInt(const slang::SVInt &svint,
                                const slang::ast::Type &astType, Location loc) {
  auto type = convertType(astType);
  if (!type)
    return {};

  bool typeIsFourValued = false;
  if (auto unpackedType = dyn_cast<moore::UnpackedType>(type))
    typeIsFourValued = unpackedType.getDomain() == moore::Domain::FourValued;

  auto fvint = convertSVIntToFVInt(svint);
  auto intType = moore::IntType::get(getContext(), fvint.getBitWidth(),
                                     fvint.hasUnknown() || typeIsFourValued
                                         ? moore::Domain::FourValued
                                         : moore::Domain::TwoValued);
  auto result = moore::ConstantOp::create(builder, loc, intType, fvint);
  return materializeConversion(type, result, astType.isSigned(), loc);
}

Value Context::materializeFixedSizeUnpackedArrayType(
    const slang::ConstantValue &constant,
    const slang::ast::FixedSizeUnpackedArrayType &astType, Location loc) {
  if (!constant.isUnpacked())
    return {};

  auto type = convertType(astType);
  if (!type)
    return {};

  // Check whether underlying type is an integer, if so, get bit width
  unsigned bitWidth;
  if (astType.elementType.isIntegral())
    bitWidth = astType.elementType.getBitWidth();
  else
    return {};

  bool typeIsFourValued = false;

  // Check whether the underlying type is four-valued
  if (auto unpackedType = dyn_cast<moore::UnpackedType>(type))
    typeIsFourValued = unpackedType.getDomain() == moore::Domain::FourValued;
  else
    return {};

  auto domain =
      typeIsFourValued ? moore::Domain::FourValued : moore::Domain::TwoValued;

  // Construct the integer type this is an unpacked array of; if possible keep
  // it two-valued, unless any entry is four-valued or the underlying type is
  // four-valued
  auto intType = moore::IntType::get(getContext(), bitWidth, domain);
  // Construct the full array type from intType
  auto arrType = moore::UnpackedArrayType::get(
      getContext(), constant.elements().size(), intType);

  llvm::SmallVector<mlir::Value> elemVals;
  moore::ConstantOp constOp;

  mlir::OpBuilder::InsertionGuard guard(builder);

  // Add one ConstantOp for every element in the array
  for (auto elem : constant.elements()) {
    FVInt fvInt = convertSVIntToFVInt(elem.integer());
    constOp = moore::ConstantOp::create(builder, loc, intType, fvInt);
    elemVals.push_back(constOp.getResult());
  }

  // Take the result of each ConstantOp and concatenate them into an array (of
  // constant values).
  auto arrayOp = moore::ArrayCreateOp::create(builder, loc, arrType, elemVals);

  return arrayOp.getResult();
}

Value Context::materializeConstant(const slang::ConstantValue &constant,
                                   const slang::ast::Type &type, Location loc) {

  if (auto *arr = type.as_if<slang::ast::FixedSizeUnpackedArrayType>())
    return materializeFixedSizeUnpackedArrayType(constant, *arr, loc);
  if (constant.isInteger())
    return materializeSVInt(constant.integer(), type, loc);
  if (constant.isReal() || constant.isShortReal())
    return materializeSVReal(constant, type, loc);
  if (constant.isString())
    return materializeString(constant, type, loc);

  return {};
}

slang::ConstantValue
Context::evaluateConstant(const slang::ast::Expression &expr) {
  using slang::ast::EvalFlags;
  slang::ast::EvalContext evalContext(
      slang::ast::ASTContext(compilation.getRoot(),
                             slang::ast::LookupLocation::max),
      EvalFlags::CacheResults | EvalFlags::SpecparamsAllowed);
  return expr.eval(evalContext);
}

/// Helper function to convert a value to its "truthy" boolean value and
/// convert it to the given domain.
Value Context::convertToBool(Value value, Domain domain) {
  value = convertToBool(value);
  if (!value)
    return {};
  auto type = moore::IntType::get(getContext(), 1, domain);
  return materializeConversion(type, value, false, value.getLoc());
}

Value Context::convertToSimpleBitVector(Value value) {
  if (!value)
    return {};
  if (isa<moore::IntType>(value.getType()))
    return value;

  // Gracefully handle queue values in contexts expecting a bit vector by
  // returning a zero of width 1. This keeps conversion progressing for UVM
  // field automation that touches queues of strings.
  if (isa<moore::QueueType>(value.getType())) {
    auto intTy = moore::IntType::getInt(getContext(), 1);
    mlir::emitRemark(value.getLoc())
        << "treating queue value as zero in bit-vector context";
    return moore::ConstantOp::create(builder, value.getLoc(), intTy, 0);
  }

  // Class handles in bit-vector context (e.g., $display formatting) are
  // represented as 64-bit integers (pointer size). Return zero as a
  // placeholder since actual pointer values are runtime-dependent.
  if (isa<moore::ClassHandleType>(value.getType())) {
    auto intTy = moore::IntType::getInt(getContext(), 64);
    mlir::emitRemark(value.getLoc())
        << "treating class handle as zero in bit-vector context";
    return moore::ConstantOp::create(builder, value.getLoc(), intTy, 0);
  }

  // Convert strings to a best-effort fixed-width bit vector. Prefer the
  // original integer when the string came from an IntToStringOp, otherwise
  // fall back to a 32-bit conversion (SV int size).
  if (isa<moore::StringType>(value.getType()) ||
      isa<moore::FormatStringType>(value.getType())) {
    if (auto formatToStr = value.getDefiningOp<moore::FormatStringToStringOp>())
      value = formatToStr.getFmtstring();

    if (auto intToStr = value.getDefiningOp<moore::IntToStringOp>())
      return intToStr.getInput();

    auto intTy = moore::IntType::getInt(getContext(), 32);
    mlir::emitRemark(value.getLoc())
        << "converting string to 32-bit integer in bit-vector context";
    return moore::StringToIntOp::create(builder, value.getLoc(),
                                        intTy.getTwoValued(), value);
  }

  // Some operations in Slang's AST, for example bitwise or `|`, don't cast
  // packed struct/array operands to simple bit vectors but directly operate
  // on the struct/array. Since the corresponding IR ops operate only on
  // simple bit vectors, insert a conversion in this case.
  if (auto packed = dyn_cast<moore::PackedType>(value.getType()))
    if (auto sbvType = packed.getSimpleBitVector())
      return materializeConversion(sbvType, value, false, value.getLoc());

  mlir::emitError(value.getLoc()) << "expression of type " << value.getType()
                                  << " cannot be cast to a simple bit vector";
  return {};
}

/// Create the necessary operations to convert from a `PackedType` to the
/// corresponding simple bit vector `IntType`. This will apply special handling
/// to time values, which requires scaling by the local timescale.
static Value materializePackedToSBVConversion(Context &context, Value value,
                                              Location loc) {
  if (isa<moore::IntType>(value.getType()))
    return value;

  auto &builder = context.builder;
  auto packedType = cast<moore::PackedType>(value.getType());
  auto intType = packedType.getSimpleBitVector();
  assert(intType);

  // If we are converting from a time to an integer, divide the integer by the
  // timescale.
  if (isa<moore::TimeType>(packedType) &&
      moore::isIntType(intType, 64, moore::Domain::FourValued)) {
    value = builder.createOrFold<moore::TimeToLogicOp>(loc, value);
    auto scale = moore::ConstantOp::create(builder, loc, intType,
                                           getTimeScaleInFemtoseconds(context));
    return builder.createOrFold<moore::DivUOp>(loc, value, scale);
  }

  // If this is an aggregate type, make sure that it does not contain any
  // `TimeType` fields. These require special conversion to ensure that the
  // local timescale is in effect.
  if (packedType.containsTimeType()) {
    mlir::emitError(loc) << "unsupported conversion: " << packedType
                         << " cannot be converted to " << intType
                         << "; contains a time type";
    return {};
  }

  // Otherwise create a simple `PackedToSBVOp` for the conversion.
  return builder.createOrFold<moore::PackedToSBVOp>(loc, value);
}

/// Create the necessary operations to convert from a simple bit vector
/// `IntType` to an equivalent `PackedType`. This will apply special handling to
/// time values, which requires scaling by the local timescale.
static Value materializeSBVToPackedConversion(Context &context,
                                              moore::PackedType packedType,
                                              Value value, Location loc) {
  if (value.getType() == packedType)
    return value;

  auto &builder = context.builder;
  auto intType = dyn_cast<moore::IntType>(value.getType());
  if (!intType) {
    mlir::emitError(loc) << "expected IntType for SBV to packed conversion, got "
                         << value.getType();
    return {};
  }
  assert(intType == packedType.getSimpleBitVector());

  // If we are converting from an integer to a time, multiply the integer by the
  // timescale.
  if (isa<moore::TimeType>(packedType) &&
      moore::isIntType(intType, 64, moore::Domain::FourValued)) {
    auto scale = moore::ConstantOp::create(builder, loc, intType,
                                           getTimeScaleInFemtoseconds(context));
    value = moore::MulOp::create(builder, loc, value, scale);
    return moore::LogicToTimeOp::create(builder, loc, value);
  }

  // If this is an aggregate type, make sure that it does not contain any
  // `TimeType` fields. These require special conversion to ensure that the
  // local timescale is in effect.
  if (packedType.containsTimeType()) {
    mlir::emitError(loc) << "unsupported conversion: " << intType
                         << " cannot be converted to " << packedType
                         << "; contains a time type";
    return {};
  }

  // Otherwise create a simple `PackedToSBVOp` for the conversion.
  return builder.createOrFold<moore::SBVToPackedOp>(loc, packedType, value);
}

/// Check if two class symbols are compatible specializations of the same
/// generic class (e.g., for the UVM this_type pattern). This is different from
/// isClassDerivedFrom which checks actual inheritance relationships.
static bool areSameGenericSpecialization(Context &context,
                                         mlir::SymbolRefAttr sym1,
                                         mlir::SymbolRefAttr sym2) {
  if (sym1 == sym2)
    return false; // Same symbol - not a specialization case

  mlir::StringAttr name1 = sym1.getRootReference();
  mlir::StringAttr name2 = sym2.getRootReference();

  // Look up the generic class for each symbol
  auto it1 = context.classSpecializationToGeneric.find(name1);
  auto it2 = context.classSpecializationToGeneric.find(name2);

  mlir::StringAttr generic1 = (it1 != context.classSpecializationToGeneric.end())
                                  ? it1->second
                                  : name1;
  mlir::StringAttr generic2 = (it2 != context.classSpecializationToGeneric.end())
                                  ? it2->second
                                  : name2;

  // If both resolve to the same generic class, they're compatible specializations
  if (generic1 == generic2)
    return true;

  // Check if one is the generic of the other
  if (generic1 == name2 || generic2 == name1)
    return true;

  return false;
}

// Forward declaration - defined later in the file after helper functions.
static mlir::SymbolRefAttr
findActualAncestorSymbol(Context &context, moore::ClassHandleType actualTy,
                         moore::ClassHandleType expectedTy);

/// Check whether the actual handle is a subclass of another handle type
/// and return a properly upcast version if so.
static mlir::Value maybeUpcastHandle(Context &context, mlir::Value actualHandle,
                                     moore::ClassHandleType expectedHandleTy) {
  auto loc = actualHandle.getLoc();

  auto actualTy = actualHandle.getType();
  auto actualHandleTy = dyn_cast<moore::ClassHandleType>(actualTy);
  if (!actualHandleTy) {
    mlir::emitError(loc) << "expected a !moore.class<...> value, got "
                         << actualTy;
    return {};
  }

  // Fast path: already the expected handle type.
  if (actualHandleTy == expectedHandleTy)
    return actualHandle;

  // Handle null type: a null handle (with special "__null__" symbol) can be
  // assigned to any class handle type. This allows "return null;" to work for
  // any class type.
  if (actualHandleTy.getClassSym() &&
      actualHandleTy.getClassSym().getRootReference() == "__null__") {
    // The source is a null handle - just cast it to the expected type.
    // Use ConversionOp since we don't have a dedicated NullCastOp.
    return moore::ConversionOp::create(context.builder, loc, expectedHandleTy,
                                       actualHandle);
  }

  // Check if the two types are compatible specializations of the same generic
  // class. This handles the UVM this_type pattern where a typedef like
  // "typedef uvm_pool#(KEY,T) this_type;" creates a different specialization
  // symbol but represents the same class with the same parameters.
  if (areSameGenericSpecialization(context, actualHandleTy.getClassSym(),
                                   expectedHandleTy.getClassSym())) {
    // Use ConversionOp for same-generic-class specialization conversions.
    // These aren't real upcasts (no inheritance relationship), just different
    // symbol names for the same logical class type.
    return moore::ConversionOp::create(context.builder, loc, expectedHandleTy,
                                       actualHandle);
  }

  if (!context.isClassDerivedFrom(actualHandleTy, expectedHandleTy)) {
    LLVM_DEBUG({
      llvm::dbgs() << "maybeUpcastHandle FAILED: actualHandleTy = "
                   << actualHandleTy << ", expectedHandleTy = " << expectedHandleTy
                   << "\n";
    });
    mlir::emitError(loc)
        << "receiver class " << actualHandleTy.getClassSym()
        << " is not the same as, or derived from, expected base class "
        << expectedHandleTy.getClassSym().getRootReference();
    return {};
  }

  // Only implicit upcasting is allowed - down casting should never be implicit.
  // Find the actual ancestor symbol that matches the expected class. This
  // handles parameterized classes where the expected type might be a generic
  // class name but the actual base is a specialization.
  auto actualAncestorSym =
      findActualAncestorSymbol(context, actualHandleTy, expectedHandleTy);
  auto upcastTy = moore::ClassHandleType::get(actualHandle.getContext(),
                                              actualAncestorSym);

  auto casted = moore::ClassUpcastOp::create(context.builder, loc,
                                             upcastTy, actualHandle)
                    .getResult();
  return casted;
}

Value Context::materializeConversion(Type type, Value value, bool isSigned,
                                     Location loc) {
  // Nothing to do if the types are already equal.
  if (type == value.getType())
    return value;

  // Handle packed types which can be converted to a simple bit vector. This
  // allows us to perform resizing and domain casting on that bit vector.
  auto dstPacked = dyn_cast<moore::PackedType>(type);
  auto srcPacked = dyn_cast<moore::PackedType>(value.getType());
  auto dstInt = dstPacked ? dstPacked.getSimpleBitVector() : moore::IntType();
  auto srcInt = srcPacked ? srcPacked.getSimpleBitVector() : moore::IntType();

  if (dstInt && srcInt) {
    // Convert the value to a simple bit vector if it isn't one already.
    value = materializePackedToSBVConversion(*this, value, loc);
    if (!value)
      return {};

    // Create truncation or sign/zero extension ops depending on the source and
    // destination width.
    auto resizedType = moore::IntType::get(
        value.getContext(), dstInt.getWidth(), srcPacked.getDomain());
    if (dstInt.getWidth() < srcInt.getWidth()) {
      value = builder.createOrFold<moore::TruncOp>(loc, resizedType, value);
    } else if (dstInt.getWidth() > srcInt.getWidth()) {
      if (isSigned)
        value = builder.createOrFold<moore::SExtOp>(loc, resizedType, value);
      else
        value = builder.createOrFold<moore::ZExtOp>(loc, resizedType, value);
    }

    // Convert the domain if needed.
    if (dstInt.getDomain() != srcInt.getDomain()) {
      if (dstInt.getDomain() == moore::Domain::TwoValued)
        value = builder.createOrFold<moore::LogicToIntOp>(loc, value);
      else if (dstInt.getDomain() == moore::Domain::FourValued)
        value = builder.createOrFold<moore::IntToLogicOp>(loc, value);
    }

    // Convert the value from a simple bit vector back to the packed type.
    value = materializeSBVToPackedConversion(*this, dstPacked, value, loc);
    if (!value)
      return {};

    assert(value.getType() == type);
    return value;
  }

  // Convert from FormatStringType to StringType
  if (isa<moore::StringType>(type) &&
      isa<moore::FormatStringType>(value.getType())) {
    return builder.createOrFold<moore::FormatStringToStringOp>(loc, value);
  }

  // Convert from StringType to FormatStringType
  if (isa<moore::FormatStringType>(type) &&
      isa<moore::StringType>(value.getType())) {
    return builder.createOrFold<moore::FormatStringOp>(loc, value);
  }

  // Handle Real To Int conversion
  if (isa<moore::IntType>(type) && isa<moore::RealType>(value.getType())) {
    auto twoValInt = builder.createOrFold<moore::RealToIntOp>(
        loc, dyn_cast<moore::IntType>(type).getTwoValued(), value);

    if (dyn_cast<moore::IntType>(type).getDomain() == moore::Domain::FourValued)
      return materializePackedToSBVConversion(*this, twoValInt, loc);
    return twoValInt;
  }

  // Handle Int to Real conversion
  if (isa<moore::RealType>(type) && isa<moore::IntType>(value.getType())) {
    Value twoValInt;
    // Check if int needs to be converted to two-valued first
    if (dyn_cast<moore::IntType>(value.getType()).getDomain() ==
        moore::Domain::TwoValued)
      twoValInt = value;
    else
      twoValInt = materializeConversion(
          dyn_cast<moore::IntType>(value.getType()).getTwoValued(), value, true,
          loc);

    if (isSigned)
      return builder.createOrFold<moore::SIntToRealOp>(loc, type, twoValInt);
    return builder.createOrFold<moore::UIntToRealOp>(loc, type, twoValInt);
  }

  auto getBuiltinFloatType = [&](moore::RealType type) -> Type {
    if (type.getWidth() == moore::RealWidth::f32)
      return mlir::Float32Type::get(builder.getContext());

    return mlir::Float64Type::get(builder.getContext());
  };

  // Handle f64/f32 to time conversion
  if (isa<moore::TimeType>(type) && isa<moore::RealType>(value.getType())) {
    auto intType =
        moore::IntType::get(builder.getContext(), 64, Domain::TwoValued);
    Type floatType =
        getBuiltinFloatType(cast<moore::RealType>(value.getType()));
    auto scale = moore::ConstantRealOp::create(
        builder, loc, value.getType(),
        FloatAttr::get(floatType, getTimeScaleInFemtoseconds(*this)));
    auto scaled = builder.createOrFold<moore::MulRealOp>(loc, value, scale);
    auto asInt = moore::RealToIntOp::create(builder, loc, intType, scaled);
    auto asLogic = moore::IntToLogicOp::create(builder, loc, asInt);
    return moore::LogicToTimeOp::create(builder, loc, asLogic);
  }

  // Handle time to f64/f32 conversion
  if (isa<moore::RealType>(type) && isa<moore::TimeType>(value.getType())) {
    auto asLogic = moore::TimeToLogicOp::create(builder, loc, value);
    auto asInt = moore::LogicToIntOp::create(builder, loc, asLogic);
    auto asReal = moore::UIntToRealOp::create(builder, loc, type, asInt);
    Type floatType = getBuiltinFloatType(cast<moore::RealType>(type));
    auto scale = moore::ConstantRealOp::create(
        builder, loc, type,
        FloatAttr::get(floatType, getTimeScaleInFemtoseconds(*this)));
    return moore::DivRealOp::create(builder, loc, asReal, scale);
  }

  // Handle int/logic to time conversion
  if (isa<moore::TimeType>(type)) {
    if (auto intType = dyn_cast<moore::IntType>(value.getType())) {
      // Convert to 4-valued 64-bit if needed, then to time.
      if (intType.getDomain() == moore::Domain::TwoValued)
        value = moore::IntToLogicOp::create(builder, loc, value);
      // Resize to 64-bit if needed.
      auto l64Type =
          moore::IntType::get(builder.getContext(), 64, moore::Domain::FourValued);
      if (value.getType() != l64Type) {
        if (intType.getWidth() < 64) {
          if (isSigned)
            value = builder.createOrFold<moore::SExtOp>(loc, l64Type, value);
          else
            value = builder.createOrFold<moore::ZExtOp>(loc, l64Type, value);
        } else if (intType.getWidth() > 64) {
          value = builder.createOrFold<moore::TruncOp>(loc, l64Type, value);
        }
      }
      return moore::LogicToTimeOp::create(builder, loc, value);
    }
  }

  // Handle time to int/logic conversion
  if (auto intType = dyn_cast<moore::IntType>(type)) {
    if (isa<moore::TimeType>(value.getType())) {
      auto asLogic = moore::TimeToLogicOp::create(builder, loc, value);
      // Convert from 4-valued to 2-valued if target is 2-valued
      if (intType.getDomain() == moore::Domain::TwoValued)
        return moore::LogicToIntOp::create(builder, loc, asLogic);
      return asLogic;
    }
  }

  // Handle Int to String
  if (isa<moore::StringType>(type)) {
    if (auto intType = dyn_cast<moore::IntType>(value.getType())) {
      if (intType.getDomain() == moore::Domain::FourValued)
        value = moore::LogicToIntOp::create(builder, loc, value);
      return moore::IntToStringOp::create(builder, loc, value);
    }
  }

  // Handle String to Int
  if (auto intType = dyn_cast<moore::IntType>(type)) {
    if (isa<moore::StringType>(value.getType())) {
      value = moore::StringToIntOp::create(builder, loc, intType.getTwoValued(),
                                           value);

      if (intType.getDomain() == moore::Domain::FourValued)
        return moore::IntToLogicOp::create(builder, loc, value);

      return value;
    }
  }

  // Handle Int to FormatString
  if (isa<moore::FormatStringType>(type)) {
    auto asStr = materializeConversion(moore::StringType::get(getContext()),
                                       value, isSigned, loc);
    if (!asStr)
      return {};
    return moore::FormatStringOp::create(builder, loc, asStr, {}, {}, {});
  }

  if (isa<moore::RealType>(type) && isa<moore::RealType>(value.getType()))
    return builder.createOrFold<moore::ConvertRealOp>(loc, type, value);

  if (isa<moore::ClassHandleType>(type) &&
      isa<moore::ClassHandleType>(value.getType())) {
    LLVM_DEBUG({
      llvm::dbgs() << "materializeConversion for class: value type = "
                   << value.getType() << ", target type = " << type << "\n";
    });
    return maybeUpcastHandle(*this, value, cast<moore::ClassHandleType>(type));
  }

  // TODO: Handle other conversions with dedicated ops.
  if (value.getType() != type)
    value = moore::ConversionOp::create(builder, loc, type, value);
  return value;
}

FailureOr<Value>
Context::convertSystemCallArity0(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc) {

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("$system",
                [&]() -> Value {
                  return moore::SystemBIOp::create(builder, loc, nullptr);
                })
          .Case("$urandom",
                [&]() -> Value {
                  return moore::UrandomBIOp::create(builder, loc, nullptr);
                })
          .Case("$random",
                [&]() -> Value {
                  return moore::RandomBIOp::create(builder, loc, nullptr);
                })
          .Case(
              "$time",
              [&]() -> Value { return moore::TimeBIOp::create(builder, loc); })
          .Case(
              "$stime",
              [&]() -> Value { return moore::TimeBIOp::create(builder, loc); })
          .Case(
              "$realtime",
              [&]() -> Value { return moore::TimeBIOp::create(builder, loc); })
          .Case("rand_mode",
                [&]() -> FailureOr<Value> {
                  // rand_mode() without args returns current mode (1 = enabled)
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                          1);
                })
          .Case("constraint_mode",
                [&]() -> FailureOr<Value> {
                  // constraint_mode() without args returns current mode (1 = enabled)
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                          1);
                })
          .Case("index",
                [&]() -> FailureOr<Value> {
                  // Handle item.index for array locator methods.
                  // IEEE 1800-2017 Section 7.12.1: item.index returns the
                  // index of the current iterator element.
                  if (currentIteratorIndex) {
                    return currentIteratorIndex;
                  }
                  mlir::emitError(loc) << "item.index is only valid within an "
                                       << "array locator method's 'with' clause";
                  return failure();
                })
          .Case("$get_initial_random_seed",
                [&]() -> FailureOr<Value> {
                  // $get_initial_random_seed returns the initial random seed.
                  // IEEE 1800-2017 §20.15.2. Emit a runtime call.
                  return (Value)moore::GetInitialRandomSeedBIOp::create(
                      builder, loc);
                })
          .Case("$initstate",
                [&]() -> FailureOr<Value> {
                  // $initstate returns 1 when simulation time == 0 and 0
                  // afterwards. IEEE 1800-2017 §20.15.
                  // Always emit a runtime check; the lowering compares
                  // CurrentTime == 0 which works in any context (initial
                  // blocks, functions called from initial blocks, etc.).
                  return (Value)moore::InitStateBIOp::create(builder, loc);
                })
          .Case("$isunbounded",
                [&]() -> Value {
                  // $isunbounded returns 0 (false) for normal parameters.
                  // Slang constant-folds this for static cases.
                  auto bitTy = moore::IntType::getInt(getContext(), 1);
                  return moore::ConstantOp::create(builder, loc, bitTy, 0);
                })
          .Case("$timeunit",
                [&]() -> Value {
                  // $timeunit with no args returns current scope's time unit
                  // as the exponent of 10 (e.g., 1ns -> -9, 10ns -> -8).
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  int exponent =
                      -3 * static_cast<int>(timeScale.base.unit);
                  auto mag = static_cast<int>(
                      timeScale.base.magnitude);
                  if (mag >= 100)
                    exponent += 2;
                  else if (mag >= 10)
                    exponent += 1;
                  return moore::ConstantOp::create(
                      builder, loc, intTy,
                      APInt(32, static_cast<uint64_t>(exponent),
                            /*isSigned=*/true));
                })
          .Case("$timeprecision",
                [&]() -> Value {
                  // $timeprecision with no args returns current scope's
                  // precision as the exponent of 10 (e.g., 1ps -> -12).
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  int exponent =
                      -3 * static_cast<int>(
                               timeScale.precision.unit);
                  auto mag = static_cast<int>(
                      timeScale.precision.magnitude);
                  if (mag >= 100)
                    exponent += 2;
                  else if (mag >= 10)
                    exponent += 1;
                  return moore::ConstantOp::create(
                      builder, loc, intTy,
                      APInt(32, static_cast<uint64_t>(exponent),
                            /*isSigned=*/true));
                })
          .Case("$reset_count",
                [&]() -> Value {
                  // Legacy: number of resets. Always 0.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$reset_value",
                [&]() -> Value {
                  // Legacy: value at reset. Always 0.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Default([&]() -> FailureOr<Value> {
            if (subroutine.name == "rand_mode" ||
                subroutine.name == "constraint_mode") {
              // Return 1 (enabled) for both getter forms
              auto intTy = moore::IntType::getInt(getContext(), 32);
              return (Value)moore::ConstantOp::create(builder, loc, intTy, 1);
            }
            mlir::emitError(loc) << "unsupported system call `"
                                 << subroutine.name << "`";
            return failure();
          });
  return systemCallRes();
}

FailureOr<Value>
Context::convertSystemCallArity1(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value) {
  auto makeIntConst = [&](Value v, int64_t c) -> Value {
    if (auto intTy = dyn_cast<moore::IntType>(v.getType()))
      return moore::ConstantOp::create(builder, loc, intTy, c);
    mlir::emitError(loc) << "expected IntType for system call size, got "
                         << v.getType();
    return {};
  };
  auto toI32 = [&](Value v) -> Value {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    return materializeConversion(i32Ty, v, /*isSigned=*/false, loc);
  };

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // OS system call.
          .Case("$system",
                [&]() -> Value {
                  return moore::SystemBIOp::create(builder, loc, value);
                })
          // Signed and unsigned system functions.
          .Case("$signed", [&]() { return value; })
          .Case("$unsigned", [&]() { return value; })
          // Real/integer conversion functions (IEEE 1800-2017 Section 20.5)
          .Case("$rtoi", [&]() { return value; })
          .Case("$itor", [&]() { return value; })

          // Math functions in SystemVerilog.
          .Case("$clog2",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::Clog2BIOp::create(builder, loc, value);
                })
          .Case("$ln",
                [&]() -> Value {
                  return moore::LnBIOp::create(builder, loc, value);
                })
          .Case("$log10",
                [&]() -> Value {
                  return moore::Log10BIOp::create(builder, loc, value);
                })
          .Case("$sin",
                [&]() -> Value {
                  return moore::SinBIOp::create(builder, loc, value);
                })
          .Case("$cos",
                [&]() -> Value {
                  return moore::CosBIOp::create(builder, loc, value);
                })
          .Case("$tan",
                [&]() -> Value {
                  return moore::TanBIOp::create(builder, loc, value);
                })
          .Case("$exp",
                [&]() -> Value {
                  return moore::ExpBIOp::create(builder, loc, value);
                })
          .Case("$sqrt",
                [&]() -> Value {
                  return moore::SqrtBIOp::create(builder, loc, value);
                })
          .Case("$floor",
                [&]() -> Value {
                  return moore::FloorBIOp::create(builder, loc, value);
                })
          .Case("$ceil",
                [&]() -> Value {
                  return moore::CeilBIOp::create(builder, loc, value);
                })
          .Case("$asin",
                [&]() -> Value {
                  return moore::AsinBIOp::create(builder, loc, value);
                })
          .Case("$acos",
                [&]() -> Value {
                  return moore::AcosBIOp::create(builder, loc, value);
                })
          .Case("$atan",
                [&]() -> Value {
                  return moore::AtanBIOp::create(builder, loc, value);
                })
          .Case("$sinh",
                [&]() -> Value {
                  return moore::SinhBIOp::create(builder, loc, value);
                })
          .Case("$cosh",
                [&]() -> Value {
                  return moore::CoshBIOp::create(builder, loc, value);
                })
          .Case("$tanh",
                [&]() -> Value {
                  return moore::TanhBIOp::create(builder, loc, value);
                })
          .Case("$asinh",
                [&]() -> Value {
                  return moore::AsinhBIOp::create(builder, loc, value);
                })
          .Case("$acosh",
                [&]() -> Value {
                  return moore::AcoshBIOp::create(builder, loc, value);
                })
          .Case("$atanh",
                [&]() -> Value {
                  return moore::AtanhBIOp::create(builder, loc, value);
                })
          .Case("$urandom",
                [&]() -> Value {
                  return moore::UrandomBIOp::create(builder, loc, value);
                })
          .Case("$urandom_range",
                [&]() -> Value {
                  // $urandom_range(max) returns a value in [0, max]
                  return moore::UrandomRangeBIOp::create(builder, loc, value,
                                                         nullptr);
                })
          .Case("$random",
                [&]() -> Value {
                  return moore::RandomBIOp::create(builder, loc, value);
                })
          .Case("$left",
                [&]() -> FailureOr<Value> {
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                          value.getType())) {
                    auto intTy = moore::IntType::getInt(getContext(), 32);
                    return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                            0);
                  }
                  return Value{};
                })
          .Case("$low",
                [&]() -> FailureOr<Value> {
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                          value.getType())) {
                    auto intTy = moore::IntType::getInt(getContext(), 32);
                    return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                            0);
                  }
                  return Value{};
                })
          .Case("$right",
                [&]() -> FailureOr<Value> {
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                          value.getType())) {
                    auto sizeVal = moore::ArraySizeOp::create(builder, loc, value);
                    auto one = makeIntConst(sizeVal, 1);
                    if (!one)
                      return failure();
                    return (Value)moore::SubOp::create(builder, loc, sizeVal,
                                                       one);
                  }
                  return Value{};
                })
          .Case("$high",
                [&]() -> FailureOr<Value> {
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(
                          value.getType())) {
                    auto sizeVal = moore::ArraySizeOp::create(builder, loc, value);
                    auto one = makeIntConst(sizeVal, 1);
                    if (!one)
                      return failure();
                    return (Value)moore::SubOp::create(builder, loc, sizeVal,
                                                       one);
                  }
                  return Value{};
                })
          .Case("rand_mode",
                [&]() -> FailureOr<Value> {
                  // rand_mode(mode) sets the mode and returns the previous mode.
                  // For now, return 1 (enabled) as the "previous" mode.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                          1);
                })
          .Case("constraint_mode",
                [&]() -> FailureOr<Value> {
                  // constraint_mode(mode) sets the mode and returns the previous mode.
                  // For now, return 1 (enabled) as the "previous" mode.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy,
                                                          1);
                })
          .Case("$realtobits",
                [&]() -> Value {
                  return moore::RealtobitsBIOp::create(builder, loc, value);
                })
          .Case("$bitstoreal",
                [&]() -> Value {
                  return moore::BitstorealBIOp::create(builder, loc, value);
                })
          .Case("$shortrealtobits",
                [&]() -> Value {
                  return moore::ShortrealtobitsBIOp::create(builder, loc,
                                                            value);
                })
          .Case("$bitstoshortreal",
                [&]() -> Value {
                  return moore::BitstoshortrealBIOp::create(builder, loc,
                                                            value);
                })
          // $test$plusargs: emit runtime call to check command-line plusargs.
          // IEEE 1800-2017 Section 21.6 "Command line input"
          .Case("$test$plusargs",
                [&]() -> FailureOr<Value> {
                  // Extract the plusarg string. The argument may be:
                  // 1. ConstantStringOp directly (string literal as packed int)
                  // 2. IntToStringOp wrapping a ConstantStringOp (implicit
                  //    conversion from packed to string type)
                  std::string plusargStr;
                  if (auto constStrOp =
                          value.getDefiningOp<moore::ConstantStringOp>()) {
                    plusargStr = constStrOp.getValue().str();
                  } else if (auto intToStr =
                                 value.getDefiningOp<moore::IntToStringOp>()) {
                    if (auto constStrOp = intToStr.getInput()
                                              .getDefiningOp<
                                                  moore::ConstantStringOp>()) {
                      plusargStr = constStrOp.getValue().str();
                    }
                  }
                  if (plusargStr.empty()) {
                    // Dynamic string — emit a TestPlusArgsBIOp for runtime
                    // evaluation. Ensure the value is a StringType.
                    auto strTy =
                        moore::StringType::get(getContext());
                    Value strVal = value;
                    if (!isa<moore::StringType>(value.getType()))
                      strVal = builder.createOrFold<moore::ConversionOp>(
                          loc, strTy, value);
                    return (Value)moore::TestPlusArgsBIOp::create(builder, loc,
                                                                   strVal);
                  }

                  // Create LLVM global string constant at module level
                  auto module = intoModuleOp;
                  // Sanitize name for MLIR symbol
                  std::string safeName;
                  for (char c : plusargStr)
                    safeName += (std::isalnum(c) || c == '_') ? c : '_';
                  std::string globalName = "__plusarg_" + safeName;

                  auto i8Ty = builder.getIntegerType(8);
                  auto strLen = static_cast<int64_t>(plusargStr.size());
                  auto arrayTy = mlir::LLVM::LLVMArrayType::get(
                      i8Ty, strLen + 1); // null-terminated

                  if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
                    OpBuilder::InsertionGuard guard(builder);
                    builder.setInsertionPointToStart(module.getBody());
                    auto global = mlir::LLVM::GlobalOp::create(
                        builder, loc, arrayTy, /*isConstant=*/true,
                        mlir::LLVM::Linkage::Internal, globalName,
                        builder.getStringAttr(plusargStr + '\0'));
                    (void)global;
                  }

                  // Get pointer to global string
                  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
                  auto addrOf = mlir::LLVM::AddressOfOp::create(
                      builder, loc, ptrTy, globalName);

                  // Call __moore_test_plusargs(ptr, len) -> i32
                  auto i32Ty = builder.getIntegerType(32);
                  auto lenConst = mlir::LLVM::ConstantOp::create(
                      builder, loc, i32Ty,
                      builder.getI32IntegerAttr(strLen));

                  auto funcTy = mlir::LLVM::LLVMFunctionType::get(
                      i32Ty, {ptrTy, i32Ty});
                  auto func = getOrCreateRuntimeFunc(
                      *this, "__moore_test_plusargs", funcTy);
                  auto callResult = mlir::LLVM::CallOp::create(
                      builder, loc, func, ValueRange{addrOf, lenConst});

                  auto mooreIntTy =
                      moore::IntType::getInt(getContext(), 32);
                  return (Value)mlir::UnrealizedConversionCastOp::create(
                             builder, loc, mooreIntTy,
                             callResult.getResult())
                      .getResult(0);
                })
          // Bit vector system functions (IEEE 1800-2017 Section 20.9)
          .Case("$isunknown",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::IsUnknownBIOp::create(builder, loc,
                                                             value);
                })
          .Case("$countones",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  Value count =
                      moore::CountOnesBIOp::create(builder, loc, value);
                  // $countones returns an integer count. Keep narrow source
                  // vectors from collapsing to 1-bit signed arithmetic by
                  // widening to at least 32 bits before later comparisons.
                  if (auto intTy = dyn_cast<moore::IntType>(count.getType());
                      intTy && intTy.getWidth() < 32) {
                    moore::IntType widenedTy =
                        intTy.getDomain() == moore::Domain::FourValued
                            ? moore::IntType::getLogic(getContext(), 32)
                            : moore::IntType::getInt(getContext(), 32);
                    count = builder.createOrFold<moore::ZExtOp>(loc, widenedTy,
                                                                count);
                  }
                  return count;
                })
          .Case("$onehot",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::OneHotBIOp::create(builder, loc, value);
                })
          .Case("$onehot0",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::OneHot0BIOp::create(builder, loc, value);
                })
          // Event triggered property (IEEE 1800-2017 Section 15.5.3)
          .Case("triggered",
                [&]() -> Value {
                  if (isa<moore::EventType>(value.getType()))
                    return moore::EventTriggeredOp::create(builder, loc, value);
                  if (isa<ltl::SequenceType>(value.getType()))
                    return ltl::TriggeredOp::create(builder, loc, value);
                  return {};
                })
          .Case("matched",
                [&]() -> Value {
                  if (isa<ltl::SequenceType>(value.getType()))
                    return ltl::MatchedOp::create(builder, loc, value);
                  return {};
                })
          .Case("len",
                [&]() -> Value {
                  if (isa<moore::StringType>(value.getType()))
                    return moore::StringLenOp::create(builder, loc, value);
                  return {};
                })
          .Case("toupper",
                [&]() -> Value {
                  return moore::StringToUpperOp::create(builder, loc, value);
                })
          .Case("tolower",
                [&]() -> Value {
                  return moore::StringToLowerOp::create(builder, loc, value);
                })
          // String to integer conversion methods (IEEE 1800-2017 Section 6.16.8)
          .Case("atoi",
                [&]() -> Value {
                  return moore::StringAtoIOp::create(builder, loc, value);
                })
          .Case("atohex",
                [&]() -> Value {
                  return moore::StringAtoHexOp::create(builder, loc, value);
                })
          .Case("atooct",
                [&]() -> Value {
                  return moore::StringAtoOctOp::create(builder, loc, value);
                })
          .Case("atobin",
                [&]() -> Value {
                  return moore::StringAtoBinOp::create(builder, loc, value);
                })
          // String to real conversion method (IEEE 1800-2017 Section 6.16.9)
          .Case("atoreal",
                [&]() -> Value {
                  auto realTy = moore::RealType::get(getContext(), moore::RealWidth::f64);
                  return moore::StringAtoRealOp::create(builder, loc, realTy, value);
                })
          // Array/queue built-in methods
          .Case("size",
                [&]() -> Value {
                  // size() is a built-in method on dynamic arrays, associative
                  // arrays, and queues
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::AssocArrayType,
                          moore::QueueType>(type))
                    return moore::ArraySizeOp::create(builder, loc, value);
                  return {};
                })
          .Case("num",
                [&]() -> Value {
                  // num() is an alias for size() on associative arrays
                  if (isa<moore::AssocArrayType>(value.getType()))
                    return moore::ArraySizeOp::create(builder, loc, value);
                  return {};
                })
          // Array locator methods (IEEE 1800-2017 Section 7.12.3)
          .Case("max",
                [&]() -> Value {
                  // max() returns a queue with the maximum value(s)
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    // Get the element type of the array
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    // Result is a queue of the element type
                    auto resultType = moore::QueueType::get(
                        cast<moore::UnpackedType>(elementType), 0);
                    return moore::QueueMaxOp::create(builder, loc, resultType,
                                                     value);
                  }
                  return {};
                })
          .Case("min",
                [&]() -> Value {
                  // min() returns a queue with the minimum value(s)
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    // Get the element type of the array
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    // Result is a queue of the element type
                    auto resultType = moore::QueueType::get(
                        cast<moore::UnpackedType>(elementType), 0);
                    return moore::QueueMinOp::create(builder, loc, resultType,
                                                     value);
                  }
                  return {};
                })
          .Case("unique",
                [&]() -> Value {
                  // unique() returns a queue with unique elements
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    // Get the element type of the array
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    // Result is a queue of the element type
                    auto resultType = moore::QueueType::get(
                        cast<moore::UnpackedType>(elementType), 0);
                    return moore::QueueUniqueOp::create(builder, loc, resultType,
                                                        value);
                  }
                  return {};
                })
          .Case("sum",
                [&]() -> Value {
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    if (!isa<moore::IntType>(elementType)) {
                      mlir::emitError(loc)
                          << "sum() only supports integer element types, got "
                          << elementType;
                      return {};
                    }
                    auto kindAttr = moore::QueueReduceKindAttr::get(
                        builder.getContext(), moore::QueueReduceKind::Sum);
                    return moore::QueueReduceOp::create(builder, loc,
                                                        elementType, kindAttr,
                                                        value);
                  }
                  return {};
                })
          .Case("product",
                [&]() -> Value {
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    if (!isa<moore::IntType>(elementType)) {
                      mlir::emitError(loc)
                          << "product() only supports integer element types, got "
                          << elementType;
                      return {};
                    }
                    auto kindAttr = moore::QueueReduceKindAttr::get(
                        builder.getContext(), moore::QueueReduceKind::Product);
                    return moore::QueueReduceOp::create(builder, loc,
                                                        elementType, kindAttr,
                                                        value);
                  }
                  return {};
                })
          .Case("and",
                [&]() -> Value {
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    if (!isa<moore::IntType>(elementType)) {
                      mlir::emitError(loc)
                          << "and() only supports integer element types, got "
                          << elementType;
                      return {};
                    }
                    auto kindAttr = moore::QueueReduceKindAttr::get(
                        builder.getContext(), moore::QueueReduceKind::And);
                    return moore::QueueReduceOp::create(builder, loc,
                                                        elementType, kindAttr,
                                                        value);
                  }
                  return {};
                })
          .Case("or",
                [&]() -> Value {
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    if (!isa<moore::IntType>(elementType)) {
                      mlir::emitError(loc)
                          << "or() only supports integer element types, got "
                          << elementType;
                      return {};
                    }
                    auto kindAttr = moore::QueueReduceKindAttr::get(
                        builder.getContext(), moore::QueueReduceKind::Or);
                    return moore::QueueReduceOp::create(builder, loc,
                                                        elementType, kindAttr,
                                                        value);
                  }
                  return {};
                })
          .Case("xor",
                [&]() -> Value {
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType,
                          moore::UnpackedArrayType>(type)) {
                    Type elementType;
                    if (auto queueType = dyn_cast<moore::QueueType>(type))
                      elementType = queueType.getElementType();
                    else if (auto dynArrayType =
                                 dyn_cast<moore::OpenUnpackedArrayType>(type))
                      elementType = dynArrayType.getElementType();
                    else if (auto arrayType =
                                 dyn_cast<moore::UnpackedArrayType>(type))
                      elementType = arrayType.getElementType();
                    if (!isa<moore::IntType>(elementType)) {
                      mlir::emitError(loc)
                          << "xor() only supports integer element types, got "
                          << elementType;
                      return {};
                    }
                    auto kindAttr = moore::QueueReduceKindAttr::get(
                        builder.getContext(), moore::QueueReduceKind::Xor);
                    return moore::QueueReduceOp::create(builder, loc,
                                                        elementType, kindAttr,
                                                        value);
                  }
                  return {};
                })
          .Case("unique_index",
                [&]() -> Value {
                  // unique_index() returns a queue with indices of unique elements
                  auto type = value.getType();
                  if (isa<moore::OpenUnpackedArrayType, moore::QueueType>(type)) {
                    auto indexType =
                        moore::IntType::get(getContext(), 64,
                                            moore::Domain::TwoValued);
                    auto resultType = moore::QueueType::get(
                        cast<moore::UnpackedType>(indexType), 0);
                    return moore::QueueUniqueIndexOp::create(builder, loc,
                                                             resultType, value);
                  }
                  return {};
                })
          // File I/O functions (IEEE 1800-2017 Section 21.3)
          .Case("$fopen",
                [&]() -> Value {
                  // $fopen(filename) opens file with default mode "r"
                  // Convert to string type if needed (handles string literals)
                  Value filename = value;
                  if (!isa<moore::StringType>(filename.getType())) {
                    // String literals may be IntType - convert to string
                    if (isa<moore::IntType>(filename.getType())) {
                      filename = moore::IntToStringOp::create(builder, loc,
                                                              filename);
                    } else {
                      return {};
                    }
                  }
                  return moore::FOpenBIOp::create(builder, loc, filename,
                                                  /*mode=*/nullptr);
                })
          .Case("$feof",
                [&]() -> Value {
                  // $feof(fd) - check if end-of-file has been reached
                  // IEEE 1800-2017 Section 21.3.3
                  auto fd = toI32(value);
                  if (!fd)
                    return {};
                  return moore::FEofBIOp::create(builder, loc, fd);
                })
          .Case("$fgetc",
                [&]() -> Value {
                  // $fgetc(fd) - read a single character from file
                  // IEEE 1800-2017 Section 21.3.3
                  auto fd = toI32(value);
                  if (!fd)
                    return {};
                  return moore::FGetCBIOp::create(builder, loc, fd);
                })
          .Case("$ftell",
                [&]() -> Value {
                  // $ftell(fd) - get current file position
                  // IEEE 1800-2017 Section 21.3.3
                  auto fd = toI32(value);
                  if (!fd)
                    return {};
                  return moore::FTellBIOp::create(builder, loc, fd);
                })
          .Case("index",
                [&]() -> FailureOr<Value> {
                  // Handle item.index for array locator methods.
                  // IEEE 1800-2017 Section 7.12.1: item.index returns the
                  // index of the current iterator element.
                  // The iterator variable is passed as the first argument.
                  if (currentIteratorIndex) {
                    return currentIteratorIndex;
                  }
                  mlir::emitError(loc) << "item.index is only valid within an "
                                       << "array locator method's 'with' clause";
                  return failure();
                })
          // $scale returns the input scaled to the time unit of another
          // module (IEEE 1800-2017 Section 20.4.2). Stub: return input.
          .Case("$scale", [&]() { return value; })
          // Query functions for dynamic types.
          // For static types, slang constant-folds these at compile time.
          .Case("$size",
                [&]() -> Value {
                  // Use ArraySizeOp for dynamic/queue/assoc array types.
                  if (isa<moore::OpenUnpackedArrayType,
                          moore::AssocArrayType,
                          moore::QueueType>(value.getType()))
                    return moore::ArraySizeOp::create(builder, loc, value);
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$bits",
                [&]() -> Value {
                  // $bits = $size * element_bit_width for dynamic types.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  if (auto qt = dyn_cast<moore::QueueType>(value.getType())) {
                    if (auto elemBits = qt.getElementType().getBitSize()) {
                      auto size =
                          moore::ArraySizeOp::create(builder, loc, value);
                      auto bitsPerElem = moore::ConstantOp::create(
                          builder, loc, intTy, *elemBits);
                      return moore::MulOp::create(builder, loc, size,
                                                  bitsPerElem);
                    }
                  }
                  if (auto dt = dyn_cast<moore::OpenUnpackedArrayType>(
                          value.getType())) {
                    if (auto elemBits = dt.getElementType().getBitSize()) {
                      auto size =
                          moore::ArraySizeOp::create(builder, loc, value);
                      auto bitsPerElem = moore::ConstantOp::create(
                          builder, loc, intTy, *elemBits);
                      return moore::MulOp::create(builder, loc, size,
                                                  bitsPerElem);
                    }
                  }
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$dimensions",
                [&]() -> Value {
                  // Dynamic types always have 1 unpacked dimension.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  if (isa<moore::OpenUnpackedArrayType,
                          moore::AssocArrayType,
                          moore::QueueType>(value.getType()))
                    return moore::ConstantOp::create(builder, loc, intTy, 1);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$unpacked_dimensions",
                [&]() -> Value {
                  // Dynamic types always have 1 unpacked dimension.
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  if (isa<moore::OpenUnpackedArrayType,
                          moore::AssocArrayType,
                          moore::QueueType>(value.getType()))
                    return moore::ConstantOp::create(builder, loc, intTy, 1);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$increment",
                [&]() -> Value {
                  // For dynamic types, $increment = -1 (ascending indices).
                  // IEEE 1800-2017 Section 7.11.1: $increment returns -1
                  // for arrays declared with ascending range [0:N].
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(
                      builder, loc, intTy,
                      APInt(32, static_cast<uint64_t>(-1),
                            /*isSigned=*/true));
                })
          // $timeunit/$timeprecision with one arg (hierarchical ref).
          // Use current scope's timescale (slang resolves the referenced
          // module's timescale at the call site for static cases).
          .Case("$timeunit",
                [&]() -> Value {
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  int exponent =
                      -3 * static_cast<int>(timeScale.base.unit);
                  auto mag = static_cast<int>(
                      timeScale.base.magnitude);
                  if (mag >= 100)
                    exponent += 2;
                  else if (mag >= 10)
                    exponent += 1;
                  return moore::ConstantOp::create(
                      builder, loc, intTy,
                      APInt(32, static_cast<uint64_t>(exponent),
                            /*isSigned=*/true));
                })
          .Case("$timeprecision",
                [&]() -> Value {
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  int exponent =
                      -3 * static_cast<int>(
                               timeScale.precision.unit);
                  auto mag = static_cast<int>(
                      timeScale.precision.magnitude);
                  if (mag >= 100)
                    exponent += 2;
                  else if (mag >= 10)
                    exponent += 1;
                  return moore::ConstantOp::create(
                      builder, loc, intTy,
                      APInt(32, static_cast<uint64_t>(exponent),
                            /*isSigned=*/true));
                })
          // $countdrivers (IEEE 1800-2017 Section 21.6). Legacy. Return 0.
          .Case("$countdrivers",
                [&]() -> Value {
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          // $getpattern (IEEE 1800-2017 Section 21.6). Legacy. Return 0.
          .Case("$getpattern",
                [&]() -> Value {
                  auto intTy = moore::IntType::getInt(getContext(), 32);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Default([&]() -> FailureOr<Value> {
            if (subroutine.name == "rand_mode" ||
                subroutine.name == "constraint_mode") {
              // Return 1 (enabled) as the "previous" mode
              auto intTy = moore::IntType::getInt(getContext(), 32);
              return (Value)moore::ConstantOp::create(builder, loc, intTy, 1);
            }
            mlir::emitError(loc) << "unsupported system call `"
                                 << subroutine.name << "`";
            return failure();
          });
  return systemCallRes();
}

FailureOr<Value>
Context::convertSystemCallArity2(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value1, Value value2) {
  auto toI32 = [&](Value v) -> Value {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    return materializeConversion(i32Ty, v, /*isSigned=*/false, loc);
  };
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("getc",
                [&]() -> Value {
                  return moore::StringGetCOp::create(builder, loc, value1,
                                                     value2);
                })
          .Case("compare",
                [&]() -> Value {
                  // str.compare(s) - lexicographic string comparison
                  // IEEE 1800-2017 Section 6.16.8
                  return moore::StringCompareOp::create(builder, loc, value1,
                                                        value2);
                })
          .Case("icompare",
                [&]() -> Value {
                  // str.icompare(s) - case-insensitive lexicographic comparison
                  // IEEE 1800-2017 Section 6.16.8
                  return moore::StringICompareOp::create(builder, loc, value1,
                                                         value2);
                })
          .Case("exists",
                [&]() -> Value {
                  // exists() checks if a key exists in an associative array.
                  // IEEE 1800-2017 §7.8.1: returns int (1 if found, 0 if not).
                  // Zero-extend from i1 to i32 to avoid sign-extension to -1.
                  if (isa<moore::AssocArrayType>(value1.getType())) {
                    Value result = moore::AssocArrayExistsOp::create(
                        builder, loc, value1, value2);
                    auto i32Ty =
                        moore::IntType::getInt(builder.getContext(), 32);
                    return materializeConversion(i32Ty, result,
                                                /*isSigned=*/false, loc);
                  }
                  auto intTy = moore::IntType::getInt(builder.getContext(), 1);
                  return moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("$urandom_range",
                [&]() -> Value {
                  // $urandom_range(max, min) returns a value in [min, max]
                  // Note: IEEE 1800-2017 says if min > max, they are swapped
                  return moore::UrandomRangeBIOp::create(builder, loc, value1,
                                                         value2);
                })
          .Case("$pow",
                [&]() -> Value {
                  // $pow(x, y) returns x raised to the power y
                  // IEEE 1800-2017 Section 20.8.2 "Real math functions"
                  return moore::PowBIOp::create(builder, loc, value1, value2);
                })
          .Case("$atan2",
                [&]() -> Value {
                  // $atan2(y, x) returns arc-tangent of y/x in radians
                  // IEEE 1800-2017 section 20.8.2 "Real math functions"
                  return moore::Atan2BIOp::create(builder, loc, value1, value2);
                })
          .Case("$hypot",
                [&]() -> Value {
                  // $hypot(x, y) returns sqrt(x^2 + y^2)
                  // IEEE 1800-2017 section 20.8.2 "Real math functions"
                  return moore::HypotBIOp::create(builder, loc, value1, value2);
                })
          // File I/O functions (IEEE 1800-2017 Section 21.3)
          .Case("$fopen",
                [&]() -> Value {
                  // $fopen(filename, mode) opens file with specified mode
                  // Convert to string type if needed (handles string literals)
                  Value filename = value1;
                  Value mode = value2;
                  if (!isa<moore::StringType>(filename.getType())) {
                    if (isa<moore::IntType>(filename.getType())) {
                      filename = moore::IntToStringOp::create(builder, loc,
                                                              filename);
                    } else {
                      return {};
                    }
                  }
                  if (!isa<moore::StringType>(mode.getType())) {
                    if (isa<moore::IntType>(mode.getType())) {
                      mode = moore::IntToStringOp::create(builder, loc, mode);
                    } else {
                      return {};
                    }
                  }
                  return moore::FOpenBIOp::create(builder, loc, filename, mode);
                })
          .Case("$ungetc",
                [&]() -> Value {
                  // $ungetc(c, fd) pushes character c back to file stream fd
                  // IEEE 1800-2017 Section 21.3.4 "File positioning functions"
                  auto intTy = moore::IntType::getInt(builder.getContext(), 32);
                  auto ch = toI32(value1);
                  if (!ch)
                    return {};
                  auto fd = toI32(value2);
                  if (!fd)
                    return {};
                  return moore::UngetCBIOp::create(builder, loc, intTy, ch, fd);
                })
          // $dist_chi_square, $dist_exponential, $dist_t, $dist_poisson:
          // Handled by the generic DistBIOp path in visitSystemCall above.
          // These arity-2 cases should never be reached.
          // Stochastic queue functions (IEEE 1800-2017 Section 21.6)
          // Legacy — deprecated, not implemented.
          .Case("$q_exam",
                [&]() -> Value {
                  mlir::emitError(loc)
                      << "unsupported legacy stochastic queue function "
                         "'$q_exam'";
                  return {};
                })
          .Case("$q_full",
                [&]() -> Value {
                  mlir::emitError(loc)
                      << "unsupported legacy stochastic queue function "
                         "'$q_full'";
                  return {};
                })
          .Default([&]() -> FailureOr<Value> {
            if (subroutine.name == "rand_mode" ||
                subroutine.name == "constraint_mode") {
              // Return 1 (enabled) as the "previous" mode
              auto intTy = moore::IntType::getInt(getContext(), 32);
              return (Value)moore::ConstantOp::create(builder, loc, intTy, 1);
            }
            mlir::emitError(loc) << "unsupported system call `"
                                 << subroutine.name << "`";
            return failure();
          });
  return systemCallRes();
}

FailureOr<Value>
Context::convertSystemCallArity3(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value1, Value value2,
                                 Value value3) {
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // $dist_uniform, $dist_normal, $dist_erlang:
          // Handled by the generic DistBIOp path in visitSystemCall above.
          // These arity-3 cases should never be reached.
          .Default([&]() -> FailureOr<Value> {
            mlir::emitError(loc) << "unsupported system call `"
                                 << subroutine.name << "`";
            return failure();
          });
  return systemCallRes();
}

FailureOr<Value>
Context::convertQueueMethodCall(const slang::ast::SystemSubroutine &subroutine,
                                Location loc, Value queueRef, Value element) {
  // Normalize element to queue element type.
  if (auto refTy = dyn_cast<moore::RefType>(queueRef.getType())) {
    if (auto queueTy = dyn_cast<moore::QueueType>(refTy.getNestedType())) {
      element = materializeConversion(queueTy.getElementType(), element,
                                      /*isSigned=*/false, element.getLoc());
    }
  }
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("push_back",
                [&]() -> FailureOr<Value> {
                  // push_back modifies the queue, so we need a reference
                  moore::QueuePushBackOp::create(builder, loc, queueRef, element);
                  // push_back returns void, return a dummy integer value
                  // since expressions need to return something
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("push_front",
                [&]() -> FailureOr<Value> {
                  moore::QueuePushFrontOp::create(builder, loc, queueRef, element);
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Default([&]() -> FailureOr<Value> { return Value{}; });
  return systemCallRes();
}

FailureOr<Value>
Context::convertQueueMethodCallNoArg(const slang::ast::SystemSubroutine &subroutine,
                                     Location loc, Value queueRef,
                                     Type elementType) {
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("pop_back",
                [&]() -> FailureOr<Value> {
                  return (Value)moore::QueuePopBackOp::create(builder, loc, elementType, queueRef);
                })
          .Case("pop_front",
                [&]() -> FailureOr<Value> {
                  return (Value)moore::QueuePopFrontOp::create(builder, loc, elementType, queueRef);
                })
          .Default([&]() -> FailureOr<Value> { return Value{}; });
  return systemCallRes();
}

FailureOr<Value>
Context::convertArrayVoidMethodCall(const slang::ast::SystemSubroutine &subroutine,
                                    Location loc, Value arrayRef) {
  // Get the underlying type of the array reference
  auto refType = cast<moore::RefType>(arrayRef.getType());
  auto nestedType = refType.getNestedType();
  bool isAssocArray = isa<moore::AssocArrayType>(nestedType);

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("delete",
                [&]() -> FailureOr<Value> {
                  if (isAssocArray) {
                    moore::AssocArrayDeleteOp::create(builder, loc, arrayRef);
                  } else {
                    // Queue delete() without index - deletes all elements
                    moore::QueueDeleteOp::create(builder, loc, arrayRef,
                                                 /*index=*/nullptr);
                  }
                  // delete returns void, return a dummy value
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("sort",
                [&]() -> FailureOr<Value> {
                  moore::QueueSortOp::create(builder, loc, arrayRef);
                  // sort returns void, return a dummy value
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("rsort",
                [&]() -> FailureOr<Value> {
                  moore::QueueRSortOp::create(builder, loc, arrayRef);
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("shuffle",
                [&]() -> FailureOr<Value> {
                  moore::QueueShuffleOp::create(builder, loc, arrayRef);
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Case("reverse",
                [&]() -> FailureOr<Value> {
                  moore::QueueReverseOp::create(builder, loc, arrayRef);
                  auto intTy = moore::IntType::getInt(getContext(), 1);
                  return (Value)moore::ConstantOp::create(builder, loc, intTy, 0);
                })
          .Default([&]() -> FailureOr<Value> { return Value{}; });
  return systemCallRes();
}

// Resolve any (possibly nested) SymbolRefAttr to an op from the root.
static mlir::Operation *resolve(Context &context, mlir::SymbolRefAttr sym) {
  return context.symbolTable.lookupNearestSymbolFrom(context.intoModuleOp, sym);
}

/// Get the generic class name for a symbol, if it's a specialization.
/// Returns the input symbol name if it's not a specialization.
static mlir::StringAttr getGenericClassName(Context &context,
                                            mlir::StringAttr symName) {
  auto it = context.classSpecializationToGeneric.find(symName);
  if (it != context.classSpecializationToGeneric.end())
    return it->second;
  return symName;
}

/// Check if two symbols refer to the same class, accounting for the fact
/// that they might be different specializations of the same generic class.
/// This is needed for the UVM this_type pattern where a typedef like
/// "typedef uvm_pool#(KEY,T) this_type;" creates a specialization that
/// should be considered the same as the enclosing class.
static bool areSameOrRelatedClass(Context &context, mlir::SymbolRefAttr sym1,
                                  mlir::SymbolRefAttr sym2) {
  if (sym1 == sym2)
    return true;

  // Get the root reference names
  mlir::StringAttr name1 = sym1.getRootReference();
  mlir::StringAttr name2 = sym2.getRootReference();

  // Check if they're the same after resolving to generic classes
  mlir::StringAttr generic1 = getGenericClassName(context, name1);
  mlir::StringAttr generic2 = getGenericClassName(context, name2);

  // If both resolve to the same generic class, they're related
  if (generic1 == generic2)
    return true;

  // Check if one is the generic of the other
  // e.g., sym1 = @test_pool_0 (specialization), sym2 = @test_pool (generic)
  if (generic1 == name2 || generic2 == name1)
    return true;

  return false;
}

/// Find the actual ancestor symbol in the inheritance chain that matches the
/// expected base class. This handles parameterized classes where the expected
/// type might be a generic class name (e.g., @uvm_reg_sequence) but the actual
/// base in the inheritance chain is a specialization (e.g., @uvm_reg_sequence_123).
/// Returns the actual symbol to use for the upcast, or the expected symbol if
/// no better match is found.
static mlir::SymbolRefAttr
findActualAncestorSymbol(Context &context, moore::ClassHandleType actualTy,
                         moore::ClassHandleType expectedTy) {
  mlir::SymbolRefAttr actualSym = actualTy.getClassSym();
  mlir::SymbolRefAttr expectedSym = expectedTy.getClassSym();

  // If they're already the same, no need to search
  if (actualSym == expectedSym)
    return expectedSym;

  // Walk up the inheritance chain to find the actual base symbol
  auto *op = resolve(context, actualSym);
  auto decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(op);

  while (decl) {
    mlir::SymbolRefAttr curBase = decl.getBaseAttr();
    if (!curBase)
      break;

    // Check if this base is related to the expected symbol
    if (areSameOrRelatedClass(context, curBase, expectedSym)) {
      // Found a match - return the actual symbol in the inheritance chain
      return curBase;
    }
    decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(resolve(context, curBase));
  }

  // Fallback to expected symbol
  return expectedSym;
}

bool Context::isClassDerivedFrom(const moore::ClassHandleType &actualTy,
                                 const moore::ClassHandleType &baseTy) {
  if (!actualTy || !baseTy)
    return false;

  mlir::SymbolRefAttr actualSym = actualTy.getClassSym();
  mlir::SymbolRefAttr baseSym = baseTy.getClassSym();

  // Direct match
  if (actualSym == baseSym)
    return true;

  // Check if they're the same class through the generic class mapping.
  // This handles the UVM this_type pattern where a typedef creates a
  // specialization that should be considered the same as the base class.
  if (areSameOrRelatedClass(*this, actualSym, baseSym))
    return true;

  auto *op = resolve(*this, actualSym);
  auto decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(op);
  if (!decl) {
    // The class declaration doesn't exist yet. This can happen when slang
    // creates a specialization (e.g., uvm_sequencer_analysis_fifo_2475) that
    // references a class we haven't converted yet.
    //
    // To handle this, we look up the generic class name and find any other
    // specialization of that generic class that HAS been converted. Since
    // all specializations of the same generic class share the same inheritance
    // chain structure, we can use any converted specialization to check
    // inheritance.
    mlir::StringAttr actualName = actualSym.getRootReference();
    mlir::StringAttr genericName = getGenericClassName(*this, actualName);

    // If the generic name is the same as the actual name, we might have a
    // specialization that hasn't been mapped yet. Try to infer the generic
    // class name by stripping a trailing _number suffix.
    // The name format is "namespace::className_number" where namespace could be
    // nested (e.g., "uvm_pkg::uvm_sequencer_analysis_fifo_2475").
    if (genericName == actualName) {
      llvm::StringRef nameStr = actualName.getValue();
      // Look for a trailing underscore followed by digits
      size_t lastUnderscore = nameStr.rfind('_');
      if (lastUnderscore != llvm::StringRef::npos) {
        llvm::StringRef suffix = nameStr.substr(lastUnderscore + 1);
        // Check if the suffix is all digits
        bool allDigits = !suffix.empty() &&
                         llvm::all_of(suffix, [](char c) { return llvm::isDigit(c); });
        if (allDigits) {
          // This looks like a specialization - extract the base name
          genericName =
              builder.getStringAttr(nameStr.substr(0, lastUnderscore));
          LLVM_DEBUG(llvm::dbgs() << "isClassDerivedFrom: inferred generic "
                                  << genericName << " from " << actualName
                                  << "\n");
        }
      }
    }

    // Try to find any other specialization of the same generic class
    for (auto &[specName, genName] : classSpecializationToGeneric) {
      if (genName == genericName && specName != actualName) {
        // Found another specialization of the same generic class
        mlir::FlatSymbolRefAttr specSym =
            mlir::FlatSymbolRefAttr::get(specName);
        auto *specOp = resolve(*this, specSym);
        decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(specOp);
        if (decl)
          break; // Use this specialization to check inheritance
      }
    }

    // If we still don't have a decl, try using the generic class itself
    if (!decl && genericName != actualName) {
      mlir::FlatSymbolRefAttr genericSym =
          mlir::FlatSymbolRefAttr::get(genericName);
      auto *genericOp = resolve(*this, genericSym);
      decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(genericOp);
    }

    // If we still can't find any related class declaration, and the actual
    // class looks like a parameterized specialization (has _number suffix),
    // trust that slang has already verified the inheritance. Slang generates
    // these specializations and would have rejected invalid type conversions.
    if (!decl) {
      if (genericName != actualName) {
        LLVM_DEBUG(llvm::dbgs()
                   << "isClassDerivedFrom: trusting slang for unconverted "
                      "specialization "
                   << actualName << " -> " << baseSym << "\n");
        return true;
      }
      return false;
    }
  }

  // Check implemented interfaces first (IEEE 1800-2017 Section 8.26).
  // A class that implements an interface class can be assigned to a variable
  // of that interface class type.
  if (auto interfaces = decl.getImplementedInterfacesAttr()) {
    for (auto ifaceAttr : interfaces) {
      auto ifaceSym = cast<mlir::SymbolRefAttr>(ifaceAttr);
      if (areSameOrRelatedClass(*this, ifaceSym, baseSym))
        return true;
    }
  }

  // Walk up the inheritance chain via ClassDeclOp::$base (SymbolRefAttr).
  unsigned depth = 0;
  while (decl && depth < 20) { // Limit depth to avoid infinite loops
    depth++;
    mlir::SymbolRefAttr curBase = decl.getBaseAttr();
    if (!curBase) {
      // If we got here through the fallback path (using a different
      // specialization), the base attributes might not be set yet.
      // In this case, try to continue by using the generic class name
      // to find an alternative base.
      mlir::StringAttr curName = decl.getSymNameAttr();
      mlir::StringAttr genericName = getGenericClassName(*this, curName);
      if (genericName != curName) {
        // Try to find the generic class and use its base
        mlir::FlatSymbolRefAttr genericSym =
            mlir::FlatSymbolRefAttr::get(genericName);
        auto *genericOp = resolve(*this, genericSym);
        auto genericDecl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(genericOp);
        if (genericDecl && genericDecl.getBaseAttr()) {
          curBase = genericDecl.getBaseAttr();
        }
      }
      // If still no base, try to infer from naming pattern
      if (!curBase) {
        llvm::StringRef nameStr = curName.getValue();
        size_t lastUnderscore = nameStr.rfind('_');
        if (lastUnderscore != llvm::StringRef::npos) {
          llvm::StringRef suffix = nameStr.substr(lastUnderscore + 1);
          if (!suffix.empty() && llvm::all_of(suffix, llvm::isDigit)) {
            genericName =
                builder.getStringAttr(nameStr.substr(0, lastUnderscore));
            mlir::FlatSymbolRefAttr genericSym =
                mlir::FlatSymbolRefAttr::get(genericName);
            auto *genericOp = resolve(*this, genericSym);
            auto genericDecl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(genericOp);
            if (genericDecl && genericDecl.getBaseAttr()) {
              curBase = genericDecl.getBaseAttr();
            }
          }
        }
      }
      if (!curBase)
        break;
    }
    // Use the related class check instead of direct equality
    if (areSameOrRelatedClass(*this, curBase, baseSym))
      return true;

    // Also check interfaces of base classes
    auto baseOp = resolve(*this, curBase);
    auto baseDecl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(baseOp);
    if (baseDecl) {
      if (auto interfaces = baseDecl.getImplementedInterfacesAttr()) {
        for (auto ifaceAttr : interfaces) {
          auto ifaceSym = cast<mlir::SymbolRefAttr>(ifaceAttr);
          if (areSameOrRelatedClass(*this, ifaceSym, baseSym))
            return true;
        }
      }
    }

    decl = baseDecl;
  }

  // If we couldn't verify inheritance but the actual class looks like a
  // parameterized specialization, trust that slang has already verified
  // the inheritance. Slang generates these specializations and would have
  // rejected invalid type conversions.
  mlir::StringAttr actualName = actualSym.getRootReference();
  llvm::StringRef nameStr = actualName.getValue();
  size_t lastUnderscore = nameStr.rfind('_');
  if (lastUnderscore != llvm::StringRef::npos) {
    llvm::StringRef suffix = nameStr.substr(lastUnderscore + 1);
    if (!suffix.empty() && llvm::all_of(suffix, llvm::isDigit)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "isClassDerivedFrom: trusting slang after failed chain "
                    "walk for specialization "
                 << actualName << " -> " << baseSym << "\n");
      return true;
    }
  }

  return false;
}

moore::ClassHandleType
Context::getAncestorClassWithProperty(const moore::ClassHandleType &actualTy,
                                      llvm::StringRef fieldName, Location loc) {
  // Start at the actual class symbol.
  mlir::SymbolRefAttr classSym = actualTy.getClassSym();

  while (classSym) {
    // Resolve the class declaration from the root symbol table owner.
    auto *op = resolve(*this, classSym);
    auto decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(op);
    if (!decl)
      break;

    // Scan the class body for a property with the requested symbol name.
    for (auto &block : decl.getBody()) {
      for (auto &opInBlock : block) {
        if (auto prop =
                llvm::dyn_cast<moore::ClassPropertyDeclOp>(&opInBlock)) {
          if (prop.getSymName() == fieldName) {
            // Found a declaring ancestor: return its handle type.
            return moore::ClassHandleType::get(actualTy.getContext(), classSym);
          }
        }
      }
    }

    // Not found here—climb to the base class (if any) and continue.
    classSym = decl.getBaseAttr(); // may be null; loop ends if so
  }

  // No ancestor declares that property.
  mlir::emitError(loc) << "unknown property `" << fieldName << "`";
  return {};
}
