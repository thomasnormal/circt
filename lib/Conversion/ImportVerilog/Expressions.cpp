//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/ADT/ScopeExit.h"

using namespace circt;
using namespace ImportVerilog;
using moore::Domain;

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

static Value visitClassProperty(Context &context,
                                const slang::ast::ClassPropertySymbol &expr) {
  auto loc = context.convertLocation(expr.location);
  auto builder = context.builder;

  auto type = context.convertType(expr.getType());
  if (!type)
    return {};

  // Check if this is a static property.
  // We check expr.lifetime first, but also handle the case where there's no
  // implicit 'this' reference - in that case, we must be in a static context
  // (e.g., static function) accessing a static property, since slang would
  // reject non-static property access without a receiver object.
  bool isStatic = expr.lifetime == slang::ast::VariableLifetime::Static;

  // Get the scope's implicit this variable
  mlir::Value instRef = context.getImplicitThisRef();

  // If there's no implicit 'this' and we're accessing a class property,
  // it must be a static property (slang validates this at parse time).
  // This handles cases where expr.lifetime may not reflect the static
  // storage class correctly (e.g., in some parameterized class contexts).
  if (!instRef) {
    isStatic = true;
  }

  if (isStatic) {
    // Static properties are stored as global variables.
    // Look up the global variable that was created for this static property.
    if (auto globalOp = context.globalVariables.lookup(&expr)) {
      return moore::GetGlobalVariableOp::create(builder, loc, globalOp);
    }

    // If the global variable hasn't been created yet (e.g., forward reference
    // or recursive class conversion), try on-demand conversion of just this
    // property. This handles cases where a method body references a static
    // property before the property has been processed during class conversion.
    if (succeeded(context.convertStaticClassProperty(expr))) {
      if (auto globalOp = context.globalVariables.lookup(&expr)) {
        return moore::GetGlobalVariableOp::create(builder, loc, globalOp);
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

  // At this point we have an implicit 'this' reference, so this is
  // an instance property access.
  if (!instRef) {
    // This should never happen based on the logic above, but keep as a safety check
    mlir::emitError(loc) << "class property '" << expr.name
                         << "' referenced without an implicit 'this'";
    return {};
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

  // Get the class symbol name from the declaring class
  auto declaringClassSym = fullyQualifiedClassName(context, *declaringClass);
  auto targetClassHandle = moore::ClassHandleType::get(
      context.getContext(),
      mlir::FlatSymbolRefAttr::get(context.getContext(), declaringClassSym));

  moore::ClassHandleType classTy =
      cast<moore::ClassHandleType>(instRef.getType());

  // If target class is same as current class, no conversion needed
  Value upcastRef;
  if (targetClassHandle.getClassSym() == classTy.getClassSym()) {
    upcastRef = instRef;
  } else {
    upcastRef = context.materializeConversion(targetClassHandle, instRef,
                                               false, instRef.getLoc());
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

  /// Convert an expression either as an lvalue or rvalue, depending on whether
  /// this is an lvalue or rvalue visitor. This is useful for projections such
  /// as `a[i]`, where you want `a` as an lvalue if you want `a[i]` as an
  /// lvalue, or `a` as an rvalue if you want `a[i]` as an rvalue.
  Value convertLvalueOrRvalueExpression(const slang::ast::Expression &expr) {
    if (isLvalue)
      return context.convertLvalueExpression(expr);
    return context.convertRvalueExpression(expr);
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
             moore::OpenUnpackedArrayType>(derefType)) {
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
                            moore::OpenUnpackedArrayType>(derefType);

    if (isDynamicType) {
      // Dynamic types (queues) use the index directly - always use dynamic
      // extract since we can't statically verify bounds.
      //
      // For queues, we need to set the queue target value so that the `$`
      // (UnboundedLiteral) expression can be evaluated as `size - 1`.
      Value queueValue;
      if (isa<moore::QueueType>(derefType)) {
        // For queues, get an rvalue of the queue to pass to ArraySizeOp.
        // If we're in lvalue context, we need to read the queue ref.
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

      auto lowBit = context.convertRvalueExpression(expr.selector());
      if (!lowBit)
        return {};
      if (isLvalue)
        return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                              lowBit);
      else
        return moore::DynExtractOp::create(builder, loc, resultType, value,
                                           lowBit);
    }

    // For fixed-size types, we need to translate the index based on the range.
    auto range = expr.value().type->getFixedRange();
    if (auto *constValue = expr.selector().getConstant();
        constValue && constValue->isInteger()) {
      assert(!constValue->hasUnknown());
      assert(constValue->size() <= 32);

      auto lowBit = constValue->integer().as<uint32_t>().value();
      if (isLvalue)
        return moore::ExtractRefOp::create(builder, loc, resultType, value,
                                           range.translateIndex(lowBit));
      else
        return moore::ExtractOp::create(builder, loc, resultType, value,
                                        range.translateIndex(lowBit));
    }

    auto lowBit = context.convertRvalueExpression(expr.selector());
    if (!lowBit)
      return {};
    lowBit = getSelectIndex(context, loc, lowBit, range);
    if (isLvalue)
      return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                            lowBit);
    else
      return moore::DynExtractOp::create(builder, loc, resultType, value,
                                         lowBit);
  }

  /// Handle range bit selections.
  Value visit(const slang::ast::RangeSelectExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto value = convertLvalueOrRvalueExpression(expr.value());
    if (!type || !value)
      return {};

    std::optional<int32_t> constLeft;
    std::optional<int32_t> constRight;
    if (auto *constant = expr.left().getConstant())
      constLeft = constant->integer().as<int32_t>();
    if (auto *constant = expr.right().getConstant())
      constRight = constant->integer().as<int32_t>();

    // We currently require the right-hand-side of the range to be constant.
    // This catches things like `[42:$]` which we don't support at the moment.
    if (!constRight) {
      mlir::emitError(loc)
          << "unsupported expression: range select with non-constant bounds";
      return {};
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

    Value offsetDyn;
    int32_t offsetConst = 0;
    auto range = expr.value().type->getFixedRange();

    using slang::ast::RangeSelectionKind;
    if (expr.getSelectionKind() == RangeSelectionKind::Simple) {
      // For a constant range [a:b], we want the offset of the lowest storage
      // bit from which we are starting the extract. For a range [5:3] this is
      // bit index 3; for a range [3:5] this is bit index 5. Both of these are
      // later translated map to bit offset 1 (see bit indices above).
      assert(constRight && "constness checked in slang");
      offsetConst = *constRight;
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
        assert(constRight && "constness checked in slang");
        offsetAdd = 1 - *constRight;
      }

      // For a [a+:b] select on [2:6], the range expands to [a:a+b-1]. We
      // therefore have to take the `a` from above and adjust it by `+b-1` to
      // arrive at the right bound.
      if (expr.getSelectionKind() == RangeSelectionKind::IndexedUp &&
          !range.isLittleEndian()) {
        assert(constRight && "constness checked in slang");
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
          offsetDyn = moore::AddOp::create(
              builder, loc, offsetDyn,
              moore::ConstantOp::create(builder, loc, offsetIntType, offsetAdd,
                                        /*isSigned=*/offsetAdd < 0));
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
        return moore::DynExtractRefOp::create(builder, loc, resultType, value,
                                              offsetDyn);
      } else {
        return moore::DynExtractOp::create(builder, loc, resultType, value,
                                           offsetDyn);
      }
    } else {
      offsetConst = range.translateIndex(offsetConst);
      if (isLvalue) {
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
      if (!value)
        return {};
      operands.push_back(value);
    }
    if (isLvalue)
      return moore::ConcatRefOp::create(builder, loc, operands);
    else if (isStringConcat)
      return moore::StringConcatOp::create(builder, loc, operands);
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

      if (isLvalue)
        return moore::StructExtractRefOp::create(builder, loc, resultType,
                                                 memberName, value);
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

      if (isLvalue)
        return moore::UnionExtractRefOp::create(builder, loc, resultType,
                                                memberName, value);
      return moore::UnionExtractOp::create(builder, loc, type, memberName,
                                           value);
    }

    // Handle classes.
    if (valueType->isClass()) {
      auto valTy = context.convertType(*valueType);
      if (!valTy)
        return {};
      auto targetTy = cast<moore::ClassHandleType>(valTy);

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

      // Get the class symbol name from the declaring class
      auto declaringClassSym =
          fullyQualifiedClassName(context, *declaringClass);
      auto upcastTargetTy = moore::ClassHandleType::get(
          context.getContext(),
          mlir::FlatSymbolRefAttr::get(context.getContext(),
                                       declaringClassSym));

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

  // Handle references to the left-hand side of a parent assignment.
  Value visit(const slang::ast::LValueReferenceExpression &expr) {
    assert(!context.lvalueStack.empty() && "parent assignments push lvalue");
    auto lvalue = context.lvalueStack.back();
    return moore::ReadOp::create(builder, loc, lvalue);
  }

  // Handle named values, such as references to declared variables.
  Value visit(const slang::ast::NamedValueExpression &expr) {
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
      auto value = moore::GetGlobalVariableOp::create(builder, loc, globalOp);
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
            auto value =
                moore::GetGlobalVariableOp::create(builder, loc, globalOp);
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

    // Emit an error for those hierarchical values not recorded in the
    // `valueSymbols`.
    auto d = mlir::emitError(loc, "unknown hierarchical name `")
             << expr.symbol.name << "`";
    d.attachNote(hierLoc) << "no rvalue generated for "
                          << slang::ast::toString(expr.symbol.kind);
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

  /// Handles logical operators (§11.4.7), assuming lhs/rhs are rvalues already.
  Value buildLogicalBOp(slang::ast::BinaryOperator op, Value lhs, Value rhs,
                        std::optional<Domain> domain = std::nullopt) {
    using slang::ast::BinaryOperator;
    // TODO: These should short-circuit; RHS should be in a separate block.

    if (domain) {
      lhs = context.convertToBool(lhs, domain.value());
      rhs = context.convertToBool(rhs, domain.value());
    } else {
      lhs = context.convertToBool(lhs);
      rhs = context.convertToBool(rhs);
    }

    if (!lhs || !rhs)
      return {};

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
    return ConcreteOp::create(builder, loc, lhs, rhs);
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
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::eq, lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::eq, lhs, rhs);
      else if (isa<moore::ClassHandleType>(lhs.getType()) ||
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
      if (isa<moore::UnpackedArrayType>(lhs.getType()))
        return moore::UArrayCmpOp::create(
            builder, loc, moore::UArrayCmpPredicate::ne, lhs, rhs);
      else if (isa<moore::StringType>(lhs.getType()))
        return moore::StringCmpOp::create(
            builder, loc, moore::StringCmpPredicate::ne, lhs, rhs);
      else if (isa<moore::ClassHandleType>(lhs.getType()) ||
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
      return createBinary<moore::CaseEqOp>(lhs, rhs);
    case BinaryOperator::CaseInequality:
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
    return moore::ReplicateOp::create(builder, loc, type, value);
  }

  // Handle set membership operator.
  Value visit(const slang::ast::InsideExpression &expr) {
    auto lhs = context.convertToSimpleBitVector(
        context.convertRvalueExpression(expr.left()));
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
        if (!listExpr->type->isIntegral()) {
          if (listExpr->type->isUnpackedArray()) {
            mlir::emitError(
                loc, "unpacked arrays in 'inside' expressions not supported");
            return {};
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
    if (cond.pattern) {
      mlir::emitError(loc) << "unsupported conditional expression with pattern";
      return {};
    }
    auto value =
        context.convertToBool(context.convertRvalueExpression(*cond.expr));
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
    auto constant = context.evaluateConstant(expr);
    if (auto value = context.materializeConstant(constant, *expr.type, loc))
      return value;

    return std::visit(
        [&](auto &subroutine) { return visitCall(expr, subroutine); },
        expr.subroutine);
  }

  /// Get both the actual `this` argument of a method call and the required
  /// class type.
  std::pair<Value, moore::ClassHandleType>
  getMethodReceiverTypeHandle(const slang::ast::CallExpression &expr) {

    moore::ClassHandleType handleTy;
    Value thisRef;

    // Qualified call: t.m(...), extract from thisClass.
    if (const slang::ast::Expression *recvExpr = expr.thisClass()) {
      thisRef = context.convertRvalueExpression(*recvExpr);
      if (!thisRef)
        return {};
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
    return {thisRef, handleTy};
  }

  /// Build a method call including implicit this argument.
  mlir::CallOpInterface
  buildMethodCall(const slang::ast::SubroutineSymbol *subroutine,
                  FunctionLowering *lowering,
                  moore::ClassHandleType actualHandleTy, Value actualThisRef,
                  SmallVector<Value> &arguments,
                  SmallVector<Type> &resultTypes) {

    // Get the expected receiver type from the lowered method
    auto funcTy = lowering->op.getFunctionType();
    auto expected0 = funcTy.getInput(0);
    auto expectedHdlTy = cast<moore::ClassHandleType>(expected0);

    // Upcast the handle as necessary.
    auto implicitThisRef = context.materializeConversion(
        expectedHdlTy, actualThisRef, false, actualThisRef.getLoc());

    // Build an argument list where the this reference is the first argument.
    SmallVector<Value> explicitArguments;
    explicitArguments.reserve(arguments.size() + 1);
    explicitArguments.push_back(implicitThisRef);
    explicitArguments.append(arguments.begin(), arguments.end());

    // Method call: choose direct vs virtual.
    const bool isVirtual =
        (subroutine->flags & slang::ast::MethodFlags::Virtual) != 0;

    if (!isVirtual) {
      auto calleeSym = lowering->op.getSymName();
      // Direct (non-virtual) call -> moore.class.call
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

    // DPI-C imports are not yet supported. Emit a remark and return a dummy
    // value to allow compilation to continue.
    if (subroutine->flags & slang::ast::MethodFlags::DPIImport) {
      mlir::emitRemark(loc) << "DPI-C imports not yet supported; call to '"
                            << subroutine->name << "' skipped";
      // Return a dummy value for the result type, or a void cast for void
      // functions.
      if (expr.type->isVoid()) {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, moore::VoidType::get(context.getContext()),
                   ValueRange{})
            .getResult(0);
      }
      auto type = context.convertType(*expr.type);
      if (!type)
        return {};
      return mlir::UnrealizedConversionCastOp::create(builder, loc, type,
                                                      ValueRange{})
          .getResult(0);
    }

    const bool isMethod = (subroutine->thisVar != nullptr);

    auto *lowering = context.declareFunction(*subroutine);
    if (!lowering)
      return {};
    auto convertedFunction = context.convertFunction(*subroutine);
    if (failed(convertedFunction))
      return {};

    // For method calls, get the receiver `this` reference first. This is needed
    // before converting arguments because default argument expressions may
    // contain method calls with implicit `this` that should refer to the
    // receiver, not the caller's `this`.
    Value methodReceiver;
    moore::ClassHandleType methodReceiverTy;
    if (isMethod) {
      auto [thisRef, tyHandle] = getMethodReceiverTypeHandle(expr);
      if (!thisRef)
        return {};
      methodReceiver = thisRef;
      methodReceiverTy = tyHandle;
    }

    // Convert the call arguments. Input arguments are converted to an rvalue.
    // All other arguments are converted to lvalues and passed into the function
    // by reference.
    //
    // For method calls, default argument expressions (which contain method calls
    // with implicit `this`) should use the receiver's `this`, not the caller's.
    // We detect default arguments by checking if the argument expression's
    // source location is within the subroutine's location range.
    SmallVector<Value> arguments;
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
        value =
            context.materializeConversion(moore::RefType::get(unpackedType),
                                          lvalue, argExpr->type->isSigned(), loc);
      }
      if (!value)
        return {};
      arguments.push_back(value);
    }

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

    mlir::CallOpInterface callOp;
    if (isMethod) {
      // Class functions -> build func.call / func.indirect_call with implicit
      // this argument. Use the already-computed receiver from earlier.
      callOp = buildMethodCall(subroutine, lowering, methodReceiverTy,
                               methodReceiver, arguments, resultTypes);
    } else {
      // Free function -> func.call
      callOp =
          mlir::func::CallOp::create(builder, loc, lowering->op, arguments);
    }

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

    // $rose, $fell, $stable, $changed, and $past are only valid in
    // the context of properties and assertions. Those are treated in the
    // LTLDialect; treat them there instead.
    bool isAssertionCall =
        llvm::StringSwitch<bool>(subroutine.name)
            .Cases({"$rose", "$fell", "$stable", "$past"}, true)
            .Default(false);

    if (isAssertionCall)
      return context.convertAssertionCallExpression(expr, info, loc);

    auto args = expr.arguments();

    FailureOr<Value> result;
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

      // Create the body region with a block argument for the iterator variable
      Block *bodyBlock = &locatorOp.getBody().emplaceBlock();
      bodyBlock->addArgument(iterVarType, loc);
      Value iterArg = bodyBlock->getArgument(0);

      // Set up the value symbol for the iterator variable within the region
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);

      // Temporarily bind the iterator variable to the block argument
      Context::ValueSymbolScope scope(context.valueSymbols);
      context.valueSymbols.insert(iterVar, iterArg);

      // Convert the predicate expression inside the region
      Value predResult = context.convertRvalueExpression(*iterExpr);
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

    // $sformatf() and $sformat look like system tasks, but we handle string
    // formatting differently from expression evaluation, so handle them
    // separately.
    // According to IEEE 1800-2023 Section 21.3.3 "Formatting data to a
    // string" $sformatf works just like the string formatting but returns
    // a StringType.
    if (!subroutine.name.compare("$sformatf")) {
      // Create the FormatString
      auto fmtValue = context.convertFormatString(
          expr.arguments(), loc, moore::IntFormat::Decimal, false);
      if (failed(fmtValue))
        return {};
      return fmtValue.value();
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

    // Handle randomize() - both class method and std::randomize().
    // IEEE 1800-2017 Section 18.6 "Randomization methods" (class method)
    // IEEE 1800-2017 Section 18.12 "Scope randomize function" (std::randomize)
    // randomize() returns 1 on success, 0 on failure.
    if (subroutine.name == "randomize" && !args.empty()) {
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
          varRefs.push_back(varRef);
        }

        auto stdRandomizeOp =
            moore::StdRandomizeOp::create(builder, loc, varRefs);

        auto resultType = context.convertType(*expr.type);
        if (!resultType)
          return {};

        return context.materializeConversion(
            resultType, stdRandomizeOp.getSuccess(), false, loc);
      }

      // Class randomize: obj.randomize()
      if (args.size() == 1) {
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

        // Create the randomize operation which returns i1 (success/failure)
        auto randomizeOp = moore::RandomizeOp::create(builder, loc, classObj);

        // The result is i1, but the expression type from slang is typically
        // int. Convert to the expected type.
        auto resultType = context.convertType(*expr.type);
        if (!resultType)
          return {};

        return context.materializeConversion(
            resultType, randomizeOp.getSuccess(), false, loc);
      }
    }

    // Handle queue methods that need special treatment (lvalue for queue).
    // push_back, push_front need queue as lvalue + element as rvalue.
    // pop_back, pop_front need queue as lvalue, no additional args.
    // delete, sort need queue as lvalue, no additional args, return void.
    bool isQueuePushMethod =
        (subroutine.name == "push_back" || subroutine.name == "push_front");
    bool isQueuePopMethod =
        (subroutine.name == "pop_back" || subroutine.name == "pop_front");
    bool isQueueVoidMethod =
        (subroutine.name == "delete" || subroutine.name == "sort");

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

    // Call the conversion function with the appropriate arity. These return one
    // of the following:
    //
    // - `failure()` if the system call was recognized but some error occurred
    // - `Value{}` if the system call was not recognized
    // - non-null `Value` result otherwise
    switch (args.size()) {
    case (0):
      result = context.convertSystemCallArity0(subroutine, loc);
      break;

    case (1):
      value = context.convertRvalueExpression(*args[0]);
      if (!value)
        return {};
      result = context.convertSystemCallArity1(subroutine, loc, value);
      break;

    case (2):
      value = context.convertRvalueExpression(*args[0]);
      value2 = context.convertRvalueExpression(*args[1]);
      if (!value || !value2)
        return {};
      result = context.convertSystemCallArity2(subroutine, loc, value, value2);
      break;

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

      value = context.convertToSimpleBitVector(value);
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
    return context.convertAssertionExpression(expr.body, loc);
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
    // If there's no ctor, we are done.
    if (!constructor)
      return newObj;

    if (const auto *callConstructor =
            constructor->as_if<slang::ast::CallExpression>())
      if (const auto *subroutine =
              std::get_if<const slang::ast::SubroutineSymbol *>(
                  &callConstructor->subroutine)) {
        // Built-in class constructors (e.g., semaphore, mailbox) don't have a
        // thisVar because they're created programmatically, not from source.
        // For these, we skip the constructor call - the runtime handles
        // initialization. Just return the allocated object.
        if ((*subroutine)->flags & slang::ast::MethodFlags::BuiltIn) {
          // TODO: Pass constructor arguments to runtime for initialization
          // (e.g., semaphore keyCount, mailbox bound)
          return newObj;
        }
        // For user-defined classes, verify that the constructor has a thisVar.
        if (!(*subroutine)->thisVar) {
          mlir::emitError(loc) << "Expected subroutine called by new to use an "
                                  "implicit this reference";
          return {};
        }
        if (failed(context.convertFunction(**subroutine)))
          return {};
        // Pass the newObj as the implicit this argument of the ctor.
        auto savedThis = context.currentThisRef;
        context.currentThisRef = newObj;
        auto restoreThis =
            llvm::make_scope_exit([&] { context.currentThisRef = savedThis; });
        // Emit a call to ctor
        if (!visitCall(*callConstructor, *subroutine))
          return {};
        // Return new handle
        return newObj;
      }
    return {};
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
    // Handle local variables.
    if (auto value = context.valueSymbols.lookup(&expr.symbol))
      return value;

    // Handle global variables.
    if (auto globalOp = context.globalVariables.lookup(&expr.symbol))
      return moore::GetGlobalVariableOp::create(builder, loc, globalOp);

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
          if (auto globalOp = context.globalVariables.lookup(&expr.symbol))
            return moore::GetGlobalVariableOp::create(builder, loc, globalOp);
        }
      }
    }

    if (auto *const property =
            expr.symbol.as_if<slang::ast::ClassPropertySymbol>()) {
      return visitClassProperty(context, *property);
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
    if (auto globalOp = context.globalVariables.lookup(&expr.symbol))
      return moore::GetGlobalVariableOp::create(builder, loc, globalOp);

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

      auto extracted = moore::ExtractRefOp::create(
          builder, loc, extractResultType, value, i * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }
    // Handle other wire
    if (remainSize) {
      auto extractResultType = moore::RefType::get(
          moore::IntType::get(context.getContext(), remainSize, domain));

      auto extracted =
          moore::ExtractRefOp::create(builder, loc, extractResultType, value,
                                      iterMax * expr.getSliceSize());
      slicedOperands.push_back(extracted);
    }

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
  slang::ConstantValue intVal = stringLiteral.convertToInt();
  auto effectiveWidth = intVal.getEffectiveWidth();
  if (!effectiveWidth)
    return {};

  auto intTy = moore::IntType::getInt(getContext(), effectiveWidth.value());

  if (astType.isString()) {
    auto immInt = moore::ConstantStringOp::create(builder, loc, intTy,
                                                  stringLiteral.toString())
                      .getResult();
    return moore::IntToStringOp::create(builder, loc, immInt).getResult();
  }
  return {};
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
    value = builder.createOrFold<moore::MulOp>(loc, value, scale);
    return builder.createOrFold<moore::LogicToTimeOp>(loc, value);
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
    mlir::emitError(loc)
        << "receiver class " << actualHandleTy.getClassSym()
        << " is not the same as, or derived from, expected base class "
        << expectedHandleTy.getClassSym().getRootReference();
    return {};
  }

  // Only implicit upcasting is allowed - down casting should never be implicit.
  auto casted = moore::ClassUpcastOp::create(context.builder, loc,
                                             expectedHandleTy, actualHandle)
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
      isa<moore::ClassHandleType>(value.getType()))
    return maybeUpcastHandle(*this, value, cast<moore::ClassHandleType>(type));

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
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

FailureOr<Value>
Context::convertSystemCallArity1(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value) {
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // Signed and unsigned system functions.
          .Case("$signed", [&]() { return value; })
          .Case("$unsigned", [&]() { return value; })

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
          // Bit vector system functions (IEEE 1800-2017 Section 20.9)
          .Case("$isunknown",
                [&]() -> FailureOr<Value> {
                  value = convertToSimpleBitVector(value);
                  if (!value)
                    return failure();
                  return (Value)moore::IsUnknownBIOp::create(builder, loc,
                                                             value);
                })
          // Event triggered property (IEEE 1800-2017 Section 15.5.3)
          .Case("triggered",
                [&]() -> Value {
                  if (isa<moore::EventType>(value.getType()))
                    return moore::EventTriggeredOp::create(builder, loc, value);
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
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

FailureOr<Value>
Context::convertSystemCallArity2(const slang::ast::SystemSubroutine &subroutine,
                                 Location loc, Value value1, Value value2) {
  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          .Case("getc",
                [&]() -> Value {
                  return moore::StringGetCOp::create(builder, loc, value1,
                                                     value2);
                })
          .Case("exists",
                [&]() -> Value {
                  // exists() checks if a key exists in an associative array
                  if (isa<moore::AssocArrayType>(value1.getType()))
                    return moore::AssocArrayExistsOp::create(builder, loc,
                                                             value1, value2);
                  return {};
                })
          .Case("$urandom_range",
                [&]() -> Value {
                  // $urandom_range(max, min) returns a value in [min, max]
                  // Note: IEEE 1800-2017 says if min > max, they are swapped
                  return moore::UrandomRangeBIOp::create(builder, loc, value1,
                                                         value2);
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
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

FailureOr<Value>
Context::convertQueueMethodCall(const slang::ast::SystemSubroutine &subroutine,
                                Location loc, Value queueRef, Value element) {
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
  if (!decl)
    return false;

  // Walk up the inheritance chain via ClassDeclOp::$base (SymbolRefAttr).
  while (decl) {
    mlir::SymbolRefAttr curBase = decl.getBaseAttr();
    if (!curBase)
      break;
    // Use the related class check instead of direct equality
    if (areSameOrRelatedClass(*this, curBase, baseSym))
      return true;
    decl = llvm::dyn_cast_or_null<moore::ClassDeclOp>(resolve(*this, curBase));
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
