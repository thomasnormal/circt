//===- AssertionExpr.cpp - Slang assertion expression conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/expressions/AssertionExpr.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "slang/ast/SystemSubroutine.h"

#include <optional>
#include <utility>

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
constexpr const char kDisableIffAttr[] = "sva.disable_iff";

struct AssertionExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  AssertionExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper to convert a range (min, optional max) to MLIR integer attributes
  std::pair<mlir::IntegerAttr, mlir::IntegerAttr>
  convertRangeToAttrs(uint32_t min,
                      std::optional<uint32_t> max = std::nullopt) {
    auto minAttr = builder.getI64IntegerAttr(min);
    mlir::IntegerAttr rangeAttr;
    if (max.has_value()) {
      rangeAttr = builder.getI64IntegerAttr(max.value() - min);
    }
    return {minAttr, rangeAttr};
  }

  /// Add repetition operation to a sequence
  Value createRepetition(Location loc,
                         const slang::ast::SequenceRepetition &repetition,
                         Value &inputSequence) {
    // Extract cycle range
    auto [minRepetitions, repetitionRange] =
        convertRangeToAttrs(repetition.range.min, repetition.range.max);

    using slang::ast::SequenceRepetition;

    // Check if repetition range is required
    if ((repetition.kind == SequenceRepetition::Nonconsecutive ||
         repetition.kind == SequenceRepetition::GoTo) &&
        !repetitionRange) {
      mlir::emitError(loc,
                      repetition.kind == SequenceRepetition::Nonconsecutive
                          ? "Nonconsecutive repetition requires a maximum value"
                          : "GoTo repetition requires a maximum value");
      return {};
    }

    switch (repetition.kind) {
    case SequenceRepetition::Consecutive:
      return ltl::RepeatOp::create(builder, loc, inputSequence, minRepetitions,
                                   repetitionRange);
    case SequenceRepetition::Nonconsecutive:
      return ltl::NonConsecutiveRepeatOp::create(
          builder, loc, inputSequence, minRepetitions, repetitionRange);
    case SequenceRepetition::GoTo:
      return ltl::GoToRepeatOp::create(builder, loc, inputSequence,
                                       minRepetitions, repetitionRange);
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::SimpleAssertionExpr &expr) {
    // Handle expression
    auto value = context.convertRvalueExpression(expr.expr);
    if (!value)
      return {};
    auto loc = context.convertLocation(expr.expr.sourceRange);
    auto valueType = value.getType();
    // For assertion instances the value is already the expected type, convert
    // boolean value
    if (!mlir::isa<ltl::SequenceType, ltl::PropertyType>(valueType)) {
      value = context.convertToI1(value);
    }
    if (!value)
      return {};

    // Handle repetition
    // The optional repetition is empty, return the converted expression
    if (!expr.repetition.has_value()) {
      return value;
    }

    // There is a repetition, embed the expression into the kind of given
    // repetition
    return createRepetition(loc, expr.repetition.value(), value);
  }

  Value visit(const slang::ast::SequenceConcatExpr &expr) {
    // Create a sequence of delayed operations, combined with a concat operation
    assert(!expr.elements.empty());

    SmallVector<Value> sequenceElements;

    for (auto it = expr.elements.begin(); it != expr.elements.end(); ++it) {
      const auto &concatElement = *it;
      Value sequenceValue =
          context.convertAssertionExpression(*concatElement.sequence, loc,
                                             /*applyDefaults=*/false);
      if (!sequenceValue)
        return {};

      Type valueType = sequenceValue.getType();
      // Sequence concatenation requires sequence types (i1 or !ltl.sequence).
      // Property types (from $rose, $fell, $changed, $stable) cannot be used
      // directly in sequence contexts.
      if (mlir::isa<ltl::PropertyType>(valueType)) {
        mlir::emitError(loc, "property type cannot be used in sequence "
                             "concatenation; consider restructuring the "
                             "assertion to use the property as a consequent");
        return {};
      }

      // Adjust inter-element delays to account for concat's cycle alignment.
      // For ##N between elements, concat already advances one cycle, so
      // subtract one when possible to align with SVA timing. The first element
      // delay is relative to the sequence start and should not be adjusted.
      uint32_t minDelay = concatElement.delay.min;
      std::optional<uint32_t> maxDelay = concatElement.delay.max;
      if (it != expr.elements.begin() && minDelay > 0) {
        --minDelay;
        if (maxDelay.has_value() && maxDelay.value() > 0)
          --maxDelay.value();
      }
      auto [delayMin, delayRange] = convertRangeToAttrs(minDelay, maxDelay);
      auto delayedSequence = ltl::DelayOp::create(builder, loc, sequenceValue,
                                                  delayMin, delayRange);
      sequenceElements.push_back(delayedSequence);
    }

    return builder.createOrFold<ltl::ConcatOp>(loc, sequenceElements);
  }

  Value visit(const slang::ast::FirstMatchAssertionExpr &expr) {
    if (!expr.matchItems.empty()) {
      mlir::emitError(loc, "first_match match items are not supported");
      return {};
    }

    auto sequenceValue =
        context.convertAssertionExpression(expr.seq, loc, /*applyDefaults=*/false);
    if (!sequenceValue)
      return {};
    return ltl::FirstMatchOp::create(builder, loc, sequenceValue);
  }

  Value visit(const slang::ast::UnaryAssertionExpr &expr) {
    auto value =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!value)
      return {};
    using slang::ast::UnaryAssertionOperator;
    switch (expr.op) {
    case UnaryAssertionOperator::Not:
      return ltl::NotOp::create(builder, loc, value);
    case UnaryAssertionOperator::SEventually:
      if (expr.range.has_value()) {
        mlir::emitError(loc, "Strong eventually with range not supported");
        return {};
      } else {
        return ltl::EventuallyOp::create(builder, loc, value);
      }
    case UnaryAssertionOperator::Always: {
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      return ltl::RepeatOp::create(builder, loc, value, attr.first,
                                   attr.second);
    }
    case UnaryAssertionOperator::NextTime: {
      auto minRepetitions = builder.getI64IntegerAttr(1);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
      }
      return ltl::DelayOp::create(builder, loc, value, minRepetitions,
                                  builder.getI64IntegerAttr(0));
    }
    case UnaryAssertionOperator::Eventually:
    case UnaryAssertionOperator::SNextTime:
    case UnaryAssertionOperator::SAlways:
      mlir::emitError(loc, "unsupported unary operator: ")
          << slang::ast::toString(expr.op);
      return {};
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::BinaryAssertionExpr &expr) {
    auto lhs =
        context.convertAssertionExpression(expr.left, loc, /*applyDefaults=*/false);
    auto rhs =
        context.convertAssertionExpression(expr.right, loc, /*applyDefaults=*/false);
    if (!lhs || !rhs)
      return {};
    SmallVector<Value, 2> operands = {lhs, rhs};
    using slang::ast::BinaryAssertionOperator;
    switch (expr.op) {
    case BinaryAssertionOperator::And:
      return ltl::AndOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Or:
      return ltl::OrOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Intersect:
      return ltl::IntersectOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Throughout: {
      auto lhsRepeat = ltl::RepeatOp::create(
          builder, loc, lhs, builder.getI64IntegerAttr(0), mlir::IntegerAttr{});
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{lhsRepeat, rhs});
    }
    case BinaryAssertionOperator::Within: {
      auto constOne =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto oneRepeat = ltl::RepeatOp::create(builder, loc, constOne,
                                             builder.getI64IntegerAttr(0),
                                             mlir::IntegerAttr{});
      auto repeatDelay = ltl::DelayOp::create(builder, loc, oneRepeat,
                                              builder.getI64IntegerAttr(1),
                                              builder.getI64IntegerAttr(0));
      auto lhsDelay =
          ltl::DelayOp::create(builder, loc, lhs, builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(0));
      auto combined = ltl::ConcatOp::create(
          builder, loc, SmallVector<Value, 3>{repeatDelay, lhsDelay, constOne});
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{combined, rhs});
    }
    case BinaryAssertionOperator::Iff: {
      auto ored = ltl::OrOp::create(builder, loc, operands);
      auto notOred = ltl::NotOp::create(builder, loc, ored);
      auto anded = ltl::AndOp::create(builder, loc, operands);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notOred, anded});
    }
    case BinaryAssertionOperator::Until:
      return ltl::UntilOp::create(builder, loc, operands);
    case BinaryAssertionOperator::UntilWith: {
      auto untilOp = ltl::UntilOp::create(builder, loc, operands);
      auto andOp = ltl::AndOp::create(builder, loc, operands);
      auto notUntil = ltl::NotOp::create(builder, loc, untilOp);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notUntil, andOp});
    }
    case BinaryAssertionOperator::Implies: {
      auto notLhs = ltl::NotOp::create(builder, loc, lhs);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notLhs, rhs});
    }
    case BinaryAssertionOperator::OverlappedImplication: {
      // The antecedent of an implication must be a sequence type (i1 or
      // !ltl.sequence), not a property type. Property types from $rose, $fell,
      // $changed, $stable cannot be used directly as antecedents.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as implication "
                             "antecedent; consider restructuring the assertion "
                             "to use the property as a consequent");
        return {};
      }
      return ltl::ImplicationOp::create(builder, loc, operands);
    }
    case BinaryAssertionOperator::NonOverlappedImplication: {
      // The antecedent of an implication must be a sequence type (i1 or
      // !ltl.sequence), not a property type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as implication "
                             "antecedent; consider restructuring the assertion "
                             "to use the property as a consequent");
        return {};
      }
      if (isa<ltl::PropertyType>(rhs.getType())) {
        // Use past-shifted antecedent to avoid concat+delay true in BMC.
        // ltl.past only accepts i1, so use delay+concat for sequences.
        Value pastAntecedent;
        if (lhs.getType().isInteger(1)) {
          pastAntecedent = ltl::PastOp::create(builder, loc, lhs, 1).getResult();
        } else {
          // For sequences, use delay to shift the antecedent back.
          auto constOne =
              hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
          auto lhsDelay = ltl::DelayOp::create(
              builder, loc, lhs, builder.getI64IntegerAttr(1),
              builder.getI64IntegerAttr(0));
          pastAntecedent = ltl::ConcatOp::create(
              builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
        }
        return ltl::ImplicationOp::create(
            builder, loc, SmallVector<Value, 2>{pastAntecedent, rhs});
      }
      auto ltlSeqType = ltl::SequenceType::get(builder.getContext());
      auto delayedRhs = ltl::DelayOp::create(
          builder, loc, ltlSeqType, rhs, builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(0));
      return ltl::ImplicationOp::create(builder, loc,
                                        SmallVector<Value, 2>{lhs, delayedRhs});
    }
    case BinaryAssertionOperator::OverlappedFollowedBy: {
      // The antecedent of an implication must be a sequence type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as followed-by "
                             "antecedent; consider restructuring the assertion");
        return {};
      }
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, notRhs});
      return ltl::NotOp::create(builder, loc, implication);
    }
    case BinaryAssertionOperator::NonOverlappedFollowedBy: {
      // The antecedent of an implication must be a sequence type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as followed-by "
                             "antecedent; consider restructuring the assertion");
        return {};
      }
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      if (isa<ltl::PropertyType>(notRhs.getType())) {
        auto constOne =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
        auto lhsDelay = ltl::DelayOp::create(
            builder, loc, lhs, builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(0));
        auto antecedent = ltl::ConcatOp::create(
            builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
        auto implication = ltl::ImplicationOp::create(
            builder, loc, SmallVector<Value, 2>{antecedent, notRhs});
        return ltl::NotOp::create(builder, loc, implication);
      }
      auto ltlSeqType = ltl::SequenceType::get(builder.getContext());
      auto delayedRhs = ltl::DelayOp::create(
          builder, loc, ltlSeqType, notRhs, builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(0));
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, delayedRhs});
      return ltl::NotOp::create(builder, loc, implication);
    }
    case BinaryAssertionOperator::SUntil: {
      // Strong until: a U b AND eventually b.
      auto untilOp = ltl::UntilOp::create(builder, loc, operands);
      auto eventuallyRhs =
          ltl::EventuallyOp::create(builder, loc, rhs);
      return ltl::AndOp::create(builder, loc,
                                SmallVector<Value, 2>{untilOp, eventuallyRhs});
    }
    case BinaryAssertionOperator::SUntilWith: {
      // Strong until-with: require overlap at termination and eventual b.
      auto andOp = ltl::AndOp::create(builder, loc, operands);
      auto untilWith = ltl::UntilOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, andOp});
      auto eventuallyAnd =
          ltl::EventuallyOp::create(builder, loc, andOp);
      return ltl::AndOp::create(builder, loc,
                                SmallVector<Value, 2>{untilWith, eventuallyAnd});
    }
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::ClockingAssertionExpr &expr) {
    auto assertionExpr =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!assertionExpr)
      return {};
    return context.convertLTLTimingControl(expr.clocking, assertionExpr);
  }

  Value visit(const slang::ast::ConditionalAssertionExpr &expr) {
    auto condition = context.convertRvalueExpression(expr.condition);
    condition = context.convertToI1(condition);
    if (!condition)
      return {};

    auto ifExpr =
        context.convertAssertionExpression(expr.ifExpr, loc, /*applyDefaults=*/false);
    if (!ifExpr)
      return {};

    auto notCond = ltl::NotOp::create(builder, loc, condition);

    if (expr.elseExpr) {
      auto elseExpr = context.convertAssertionExpression(*expr.elseExpr, loc,
                                                         /*applyDefaults=*/false);
      if (!elseExpr)
        return {};
      auto condAndIf =
          ltl::AndOp::create(builder, loc, SmallVector<Value, 2>{condition, ifExpr});
      auto notCondAndElse =
          ltl::AndOp::create(builder, loc, SmallVector<Value, 2>{notCond, elseExpr});
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{condAndIf, notCondAndElse});
    }

    return ltl::OrOp::create(builder, loc,
                             SmallVector<Value, 2>{notCond, ifExpr});
  }

  Value visit(const slang::ast::DisableIffAssertionExpr &expr) {
    auto disableCond = context.convertRvalueExpression(expr.condition);
    disableCond = context.convertToI1(disableCond);
    if (!disableCond)
      return {};

    auto assertionExpr =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!assertionExpr)
      return {};

    // Approximate disable iff by treating the property as vacuously true when
    // the disable condition holds.
    auto orOp = ltl::OrOp::create(
        builder, loc, SmallVector<Value, 2>{disableCond, assertionExpr});
    orOp->setAttr(kDisableIffAttr, builder.getUnitAttr());
    return orOp.getResult();
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::AssertionExpr &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

FailureOr<Value> Context::convertAssertionSystemCallArity1(
    const slang::ast::SystemSubroutine &subroutine, Location loc, Value value) {

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // Note: $rose/$fell/$stable/$changed are handled in
          // convertAssertionCallExpression to keep them usable in sequences.
          .Case("$fell",
                [&]() -> Value {
                  return {};
                })
          // Translate $rose to x[0] ∧ ¬x[-1]
          .Case("$rose",
                [&]() -> Value {
                  return {};
                })
          // Translate $stable to ( x[0] ∧ x[-1] ) ⋁ ( ¬x[0] ∧ ¬x[-1] )
          .Case("$stable",
                [&]() -> Value {
                  return {};
                })
          // Translate $changed to ¬$stable(x).
          .Case("$changed",
                [&]() -> Value {
                  return {};
                })
          // $sampled(x) in assertion context returns the sampled value, which
          // is effectively the current value since assertions use sampled semantics.
          .Case("$sampled", [&]() -> Value { return value; })
          // Note: $past is handled separately in convertAssertionCallExpression
          // using moore::PastOp to preserve the type for comparisons.
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

Value Context::convertAssertionCallExpression(
    const slang::ast::CallExpression &expr,
    const slang::ast::CallExpression::SystemCallInfo &info, Location loc) {

  const auto &subroutine = *info.subroutine;
  auto args = expr.arguments();

  FailureOr<Value> result;
  Value value;
  Value boolVal;

  if (subroutine.name == "$rose" || subroutine.name == "$fell" ||
      subroutine.name == "$stable" || subroutine.name == "$changed") {
    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};

    auto valueType = dyn_cast<moore::IntType>(value.getType());
    if (!valueType) {
      mlir::emitError(loc) << "unsupported sampled value type for "
                           << subroutine.name;
      return {};
    }
    bool hasClockingArg =
        args.size() > 1 &&
        args[1]->kind == slang::ast::ExpressionKind::ClockingEvent;
    if (hasClockingArg && !inAssertionExpr) {
      auto resultType = moore::IntType::get(
          builder.getContext(), 1, valueType.getDomain());
      mlir::emitWarning(loc)
          << subroutine.name
          << " with explicit clocking is not yet lowered outside assertions; "
             "returning 0 as a placeholder";
      return moore::ConstantOp::create(builder, loc, resultType, 0);
    }

    if (subroutine.name == "$stable" || subroutine.name == "$changed") {
      auto past =
          moore::PastOp::create(builder, loc, value, /*delay=*/1).getResult();
      auto stable =
          moore::CaseEqOp::create(builder, loc, value, past).getResult();
      Value resultVal = stable;
      if (subroutine.name == "$changed")
        resultVal = moore::NotOp::create(builder, loc, stable).getResult();
      return resultVal;
    }

    Value current = value;
    if (valueType.getBitSize() != 1)
      current = moore::ReduceOrOp::create(builder, loc, value).getResult();
    auto past =
        moore::PastOp::create(builder, loc, current, /*delay=*/1).getResult();
    auto currentTy = cast<moore::IntType>(current.getType());
    auto zero = moore::ConstantOp::create(builder, loc, currentTy, 0);
    auto one = moore::ConstantOp::create(builder, loc, currentTy, 1);
    auto currentIsOne =
        moore::CaseEqOp::create(builder, loc, current, one).getResult();
    auto pastIsOne =
        moore::CaseEqOp::create(builder, loc, past, one).getResult();
    auto currentIsZero =
        moore::CaseEqOp::create(builder, loc, current, zero).getResult();
    auto pastIsZero =
        moore::CaseEqOp::create(builder, loc, past, zero).getResult();
    Value resultVal;
    if (subroutine.name == "$rose") {
      auto notPastOne =
          moore::NotOp::create(builder, loc, pastIsOne).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, currentIsOne, notPastOne)
              .getResult();
    } else {
      auto notPastZero =
          moore::NotOp::create(builder, loc, pastIsZero).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, currentIsZero, notPastZero)
              .getResult();
    }
    return resultVal;
  }

  // Handle $past specially - it returns the past value with preserved type
  // so that comparisons like `$past(val) == 0` work correctly.
  if (subroutine.name == "$past") {
    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};

    // Get the delay (numTicks) from the second argument if present.
    // Default to 1 if empty or not provided.
    int64_t delay = 1;
    if (args.size() > 1 &&
        args[1]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      auto cv = evaluateConstant(*args[1]);
      if (cv.isInteger()) {
        auto intVal = cv.integer().as<int64_t>();
        if (intVal)
          delay = *intVal;
      }
    }

    // Always use moore::PastOp to preserve the type for comparisons.
    // $past(val) returns the past value with the same type as val, so that
    // comparisons like `$past(val) == 0` work correctly. LTL-specific temporal
    // operators like $rose/$fell/$stable/$changed use ltl ops internally.
    return moore::PastOp::create(builder, loc, value, delay).getResult();
  }

  switch (args.size()) {
  case (1):
    value = this->convertRvalueExpression(*args[0]);

    // $sampled returns the sampled value of the expression. In procedural
    // context (outside assertions), we return the original value to preserve
    // its type for comparisons.
    if (subroutine.name == "$sampled")
      return value;

    boolVal = builder.createOrFold<moore::ToBuiltinBoolOp>(loc, value);
    if (!boolVal)
      return {};
    result = this->convertAssertionSystemCallArity1(subroutine, loc, boolVal);
    break;

  default:
    break;
  }

  if (failed(result))
    return {};
  if (*result)
    return *result;

  mlir::emitError(loc) << "unsupported system call `" << subroutine.name << "`";
  return {};
}

Value Context::convertAssertionExpression(const slang::ast::AssertionExpr &expr,
                                          Location loc, bool applyDefaults) {
  bool prevInAssertionExpr = inAssertionExpr;
  inAssertionExpr = true;
  AssertionExprVisitor visitor{*this, loc};
  auto value = expr.visit(visitor);
  inAssertionExpr = prevInAssertionExpr;
  if (!value || !applyDefaults)
    return value;

  if (currentScope &&
      (isa<ltl::PropertyType, ltl::SequenceType>(value.getType()) ||
       value.getType().isInteger(1))) {
    if (auto *disableExpr = compilation.getDefaultDisable(*currentScope)) {
      auto disableVal = convertRvalueExpression(*disableExpr);
      disableVal = convertToI1(disableVal);
      if (disableVal) {
        auto orOp = ltl::OrOp::create(
            builder, loc, SmallVector<Value, 2>{disableVal, value});
        orOp->setAttr(kDisableIffAttr, builder.getUnitAttr());
        value = orOp.getResult();
      }
    }

    if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
      if (auto *clockBlock =
              clocking->as_if<slang::ast::ClockingBlockSymbol>()) {
        value = convertLTLTimingControl(clockBlock->getEvent(), value);
      }
    }
  }

  return value;
}
// NOLINTEND(misc-no-recursion)

/// Helper function to convert a value to an i1 value.
Value Context::convertToI1(Value value) {
  if (!value)
    return {};

  // If the value is already an i1 (e.g., from $sampled), return it directly.
  if (value.getType().isInteger(1))
    return value;

  auto type = dyn_cast<moore::IntType>(value.getType());
  if (!type || type.getBitSize() != 1) {
    mlir::emitError(value.getLoc(), "expected a 1-bit integer");
    return {};
  }

  return moore::ToBuiltinBoolOp::create(builder, value.getLoc(), value);
}
