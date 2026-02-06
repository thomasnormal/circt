//===- AssertionExpr.cpp - Slang assertion expression conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/CallExpression.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "slang/ast/SystemSubroutine.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <variant>

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
constexpr const char kDisableIffAttr[] = "sva.disable_iff";
constexpr const char kWeakEventuallyAttr[] = "ltl.weak";

static Value createUnknownOrZeroConstant(Context &context, Location loc,
                                         moore::IntType type) {
  auto &builder = context.builder;
  if (type.getDomain() == moore::Domain::TwoValued)
    return moore::ConstantOp::create(builder, loc, type, 0);
  auto width = type.getWidth();
  if (width == 0)
    return {};
  return moore::ConstantOp::create(
      builder, loc, type,
      FVInt(APInt(width, 0), APInt::getAllOnes(width)));
}

struct SequenceLengthBounds {
  uint64_t min = 0;
  std::optional<uint64_t> max;
};

static std::optional<SequenceLengthBounds>
getSequenceLengthBounds(Value seq) {
  if (!seq)
    return std::nullopt;
  if (!isa<ltl::SequenceType>(seq.getType()))
    return SequenceLengthBounds{1, 1};

  if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
    auto inputBounds = getSequenceLengthBounds(delayOp.getInput());
    if (!inputBounds)
      return std::nullopt;
    uint64_t minDelay = delayOp.getDelay();
    std::optional<uint64_t> maxDelay;
    if (auto length = delayOp.getLength())
      maxDelay = minDelay + *length;
    SequenceLengthBounds result;
    result.min = inputBounds->min + minDelay;
    if (inputBounds->max && maxDelay)
      result.max = *inputBounds->max + *maxDelay;
    return result;
  }
  if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
    SequenceLengthBounds result;
    result.min = 0;
    result.max = 0;
    for (auto input : concatOp.getInputs()) {
      auto bounds = getSequenceLengthBounds(input);
      if (!bounds)
        return std::nullopt;
      result.min += bounds->min;
      if (result.max && bounds->max)
        *result.max += *bounds->max;
      else
        result.max.reset();
    }
    return result;
  }
  if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
    auto more = repeatOp.getMore();
    auto bounds = getSequenceLengthBounds(repeatOp.getInput());
    if (!bounds)
      return std::nullopt;
    SequenceLengthBounds result;
    result.min = bounds->min * repeatOp.getBase();
    if (bounds->max && more)
      result.max = *bounds->max * (repeatOp.getBase() + *more);
    return result;
  }
  if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
    return getSequenceLengthBounds(firstMatch.getInput());

  return std::nullopt;
}

static Value lowerSampledValueFunctionWithClocking(
    Context &context, const slang::ast::Expression &valueExpr,
    const slang::ast::TimingControl &timingCtrl, StringRef funcName,
    const slang::ast::Expression *enableExpr, bool invertEnable, Location loc) {
  auto &builder = context.builder;
  auto *insertionBlock = builder.getInsertionBlock();
  if (!insertionBlock)
    return {};
  auto *parentOp = insertionBlock->getParentOp();
  moore::SVModuleOp module;
  if (parentOp) {
    if (auto direct = dyn_cast<moore::SVModuleOp>(parentOp))
      module = direct;
    else
      module = parentOp->getParentOfType<moore::SVModuleOp>();
  }
  if (!module) {
    mlir::emitWarning(loc)
        << funcName
        << " with explicit clocking is only supported within a module; "
           "returning 0 as a placeholder";
    auto resultType =
        moore::IntType::get(builder.getContext(), 1, moore::Domain::FourValued);
    return moore::ConstantOp::create(builder, loc, resultType, 0);
  }

  auto valueType = context.convertType(*valueExpr.type);
  auto intType = dyn_cast_or_null<moore::IntType>(valueType);
  if (!intType) {
    mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
    return {};
  }

  bool isRose = funcName == "$rose";
  bool isFell = funcName == "$fell";
  bool isStable = funcName == "$stable";
  bool isChanged = funcName == "$changed";
  bool boolCast = isRose || isFell;
  auto sampleType = boolCast
                        ? moore::IntType::get(builder.getContext(), 1,
                                              intType.getDomain())
                        : intType;
  auto resultType =
      moore::IntType::get(builder.getContext(), 1, sampleType.getDomain());

  Value prevVar;
  Value resultVar;
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *moduleBlock = module.getBody();
    if (moduleBlock->mightHaveTerminator()) {
      if (auto *terminator = moduleBlock->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(moduleBlock);
    } else {
      builder.setInsertionPointToEnd(moduleBlock);
    }

    Value prevInit = createUnknownOrZeroConstant(context, loc, sampleType);
    if (!prevInit)
      return {};
    Value resultInit = moore::ConstantOp::create(builder, loc, resultType, 0);

    prevVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(sampleType), StringAttr{}, prevInit);
    resultVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(resultType), StringAttr{},
        resultInit);

    auto proc =
        moore::ProcedureOp::create(builder, loc, moore::ProcedureKind::Always);
    builder.setInsertionPointToEnd(&proc.getBody().emplaceBlock());
    if (failed(context.convertTimingControl(timingCtrl)))
      return {};

    Value current = context.convertRvalueExpression(valueExpr);
    if (!current)
      return {};
    auto currentType = dyn_cast<moore::IntType>(current.getType());
    if (!currentType) {
      mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
      return {};
    }
    if (boolCast)
      current = moore::BoolCastOp::create(builder, loc, current);
    if (current.getType() != sampleType)
      current = context.materializeConversion(sampleType, current,
                                              /*isSigned=*/false, loc);

    Value enable;
    bool hasEnable = false;
    if (enableExpr) {
      enable = context.convertRvalueExpression(*enableExpr);
      if (!enable)
        return {};
      enable = context.convertToBool(enable);
      if (!enable)
        return {};
      if (invertEnable)
        enable = moore::NotOp::create(builder, loc, enable).getResult();
      hasEnable = true;
    }
    auto selectWithEnable = [&](Value onTrue, Value onFalse) -> Value {
      if (!hasEnable)
        return onTrue;
      auto conditional =
          moore::ConditionalOp::create(builder, loc, onTrue.getType(), enable);
      auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
      auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&trueBlock);
        moore::YieldOp::create(builder, loc, onTrue);
        builder.setInsertionPointToStart(&falseBlock);
        moore::YieldOp::create(builder, loc, onFalse);
      }
      return conditional.getResult();
    };

    Value prev = moore::ReadOp::create(builder, loc, prevVar);
    Value result;
    if (isStable || isChanged) {
      auto stable =
          moore::EqOp::create(builder, loc, current, prev).getResult();
      result = stable;
      if (isChanged)
        result = moore::NotOp::create(builder, loc, stable).getResult();
    } else {
      if (isRose) {
        auto notPrev = moore::NotOp::create(builder, loc, prev).getResult();
        result = moore::AndOp::create(builder, loc, current, notPrev)
                     .getResult();
      } else {
        auto notCurrent =
            moore::NotOp::create(builder, loc, current).getResult();
        result = moore::AndOp::create(builder, loc, notCurrent, prev)
                     .getResult();
      }
    }

    Value disabledValue = createUnknownOrZeroConstant(context, loc, resultType);
    if (!disabledValue)
      return {};
    Value resultValue = selectWithEnable(result, disabledValue);
    moore::BlockingAssignOp::create(builder, loc, resultVar, resultValue);
    Value nextPrev = selectWithEnable(current, prev);
    moore::BlockingAssignOp::create(builder, loc, prevVar, nextPrev);
    moore::ReturnOp::create(builder, loc);
  }

  return moore::ReadOp::create(builder, loc, resultVar);
}

static Value lowerPastWithClocking(Context &context,
                                   const slang::ast::Expression &valueExpr,
                                   const slang::ast::TimingControl &timingCtrl,
                                   int64_t delay,
                                   const slang::ast::Expression *enableExpr,
                                   bool invertEnable,
                                   Location loc) {
  auto &builder = context.builder;
  auto *insertionBlock = builder.getInsertionBlock();
  if (!insertionBlock)
    return {};
  auto *parentOp = insertionBlock->getParentOp();
  moore::SVModuleOp module;
  if (parentOp) {
    if (auto direct = dyn_cast<moore::SVModuleOp>(parentOp))
      module = direct;
    else
      module = parentOp->getParentOfType<moore::SVModuleOp>();
  }
  if (!module) {
    mlir::emitWarning(loc)
        << "$past with explicit clocking is only supported within a module; "
           "returning 0 as a placeholder";
    auto resultType =
        moore::IntType::get(builder.getContext(), 1, moore::Domain::FourValued);
    return moore::ConstantOp::create(builder, loc, resultType, 0);
  }
  if (delay < 0) {
    mlir::emitError(loc) << "$past delay must be non-negative";
    return {};
  }

  auto valueType = context.convertType(*valueExpr.type);
  auto intType = dyn_cast_or_null<moore::IntType>(valueType);
  if (!intType) {
    mlir::emitError(loc)
        << "unsupported $past value type with explicit clocking";
    return {};
  }

  int64_t historyDepth = std::max<int64_t>(delay, 1);
  Value init = createUnknownOrZeroConstant(context, loc, intType);
  if (!init)
    return {};

  SmallVector<Value, 4> historyVars;
  Value resultVar;
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *moduleBlock = module.getBody();
    if (moduleBlock->mightHaveTerminator()) {
      if (auto *terminator = moduleBlock->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(moduleBlock);
    } else {
      builder.setInsertionPointToEnd(moduleBlock);
    }

    for (int64_t i = 0; i < historyDepth; ++i) {
      historyVars.push_back(moore::VariableOp::create(
          builder, loc, moore::RefType::get(intType), StringAttr{}, init));
    }
    resultVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(intType), StringAttr{}, init);

    auto proc =
        moore::ProcedureOp::create(builder, loc, moore::ProcedureKind::Always);
    builder.setInsertionPointToEnd(&proc.getBody().emplaceBlock());
    if (failed(context.convertTimingControl(timingCtrl)))
      return {};

    Value current = context.convertRvalueExpression(valueExpr);
    if (!current)
      return {};
    auto currentType = dyn_cast<moore::IntType>(current.getType());
    if (!currentType) {
      mlir::emitError(loc)
          << "unsupported $past value type with explicit clocking";
      return {};
    }
    if (current.getType() != intType)
      current =
          context.materializeConversion(intType, current, /*isSigned=*/false,
                                         loc);

    Value enable;
    bool hasEnable = false;
    if (enableExpr) {
      enable = context.convertRvalueExpression(*enableExpr);
      if (!enable)
        return {};
      enable = context.convertToBool(enable);
      if (!enable)
        return {};
      if (invertEnable)
        enable = moore::NotOp::create(builder, loc, enable).getResult();
      hasEnable = true;
    }

    auto selectWithEnable = [&](Value onTrue, Value onFalse) -> Value {
      if (!hasEnable)
        return onTrue;
      auto conditional =
          moore::ConditionalOp::create(builder, loc, onTrue.getType(), enable);
      auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
      auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&trueBlock);
        moore::YieldOp::create(builder, loc, onTrue);
        builder.setInsertionPointToStart(&falseBlock);
        moore::YieldOp::create(builder, loc, onFalse);
      }
      return conditional.getResult();
    };

    Value pastValue = current;
    if (delay > 0)
      pastValue = moore::ReadOp::create(builder, loc, historyVars.back());
    Value disabledValue = createUnknownOrZeroConstant(context, loc, intType);
    if (!disabledValue)
      return {};
    Value resultValue = selectWithEnable(pastValue, disabledValue);
    moore::BlockingAssignOp::create(builder, loc, resultVar, resultValue);

    for (int64_t i = historyDepth - 1; i > 0; --i) {
      Value prev = moore::ReadOp::create(builder, loc, historyVars[i]);
      Value prevPrev = moore::ReadOp::create(builder, loc, historyVars[i - 1]);
      Value next = selectWithEnable(prevPrev, prev);
      moore::BlockingAssignOp::create(builder, loc, historyVars[i], next);
    }
    Value prev0 = moore::ReadOp::create(builder, loc, historyVars[0]);
    Value next0 = selectWithEnable(current, prev0);
    moore::BlockingAssignOp::create(builder, loc, historyVars[0], next0);
    moore::ReturnOp::create(builder, loc);
  }

  return moore::ReadOp::create(builder, loc, resultVar);
}

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

  LogicalResult
  handleMatchItems(std::span<const slang::ast::Expression *const> matchItems) {
    for (auto *item : matchItems) {
      if (!item)
        continue;
      switch (item->kind) {
      case slang::ast::ExpressionKind::Assignment: {
        auto &assign = item->as<slang::ast::AssignmentExpression>();
        if (assign.isCompound() || assign.isNonBlocking()) {
          mlir::emitError(loc, "unsupported match item assignment kind");
          return failure();
        }
        auto *sym = assign.left().getSymbolReference();
        auto *local =
            sym ? sym->as_if<slang::ast::LocalAssertionVarSymbol>() : nullptr;
        if (!local) {
          mlir::emitError(loc, "match item assignment must target a local "
                               "assertion variable");
          return failure();
        }
        auto rhs = context.convertRvalueExpression(assign.right());
        if (!rhs)
          return failure();
        if (!isa<moore::UnpackedType>(rhs.getType())) {
          mlir::emitError(loc, "unsupported match item assignment type")
              << rhs.getType();
          return failure();
        }
        context.setAssertionLocalVarBinding(
            local, rhs, context.getAssertionSequenceOffset());
        break;
      }
      case slang::ast::ExpressionKind::UnaryOp: {
        auto &unary = item->as<slang::ast::UnaryExpression>();
        using slang::ast::UnaryOperator;
        bool isInc = false;
        bool isDec = false;
        switch (unary.op) {
        case UnaryOperator::Preincrement:
        case UnaryOperator::Postincrement:
          isInc = true;
          break;
        case UnaryOperator::Predecrement:
        case UnaryOperator::Postdecrement:
          isDec = true;
          break;
        default:
          mlir::emitError(loc, "unsupported match item unary operator");
          return failure();
        }
        auto *sym = unary.operand().getSymbolReference();
        auto *local =
            sym ? sym->as_if<slang::ast::LocalAssertionVarSymbol>() : nullptr;
        if (!local) {
          mlir::emitError(loc, "match item unary operator must target a local "
                               "assertion variable");
          return failure();
        }
        auto base = context.convertRvalueExpression(unary.operand());
        if (!base)
          return failure();
        auto intType = dyn_cast<moore::IntType>(base.getType());
        if (!intType) {
          mlir::emitError(loc, "match item unary operator requires int type");
          return failure();
        }
        auto one = moore::ConstantOp::create(builder, loc, intType, 1);
        Value updated =
            isInc ? moore::AddOp::create(builder, loc, base, one).getResult()
                  : moore::SubOp::create(builder, loc, base, one).getResult();
        context.setAssertionLocalVarBinding(
            local, updated, context.getAssertionSequenceOffset());
        break;
      }
      case slang::ast::ExpressionKind::Call: {
        auto &call = item->as<slang::ast::CallExpression>();
        if (auto *sysInfo =
                std::get_if<slang::ast::CallExpression::SystemCallInfo>(
                    &call.subroutine)) {
          auto callLoc = context.convertLocation(call.sourceRange);
          mlir::emitRemark(callLoc)
              << "ignoring system subroutine `" << sysInfo->subroutine->name
              << "` in assertion match items";
          // Sequence match-item subroutine calls are side-effect only; formal
          // lowering ignores them.
          break;
        }
        if (!context.convertRvalueExpression(call))
          return failure();
        break;
      }
      default:
        mlir::emitError(loc, "unsupported match item expression");
        return failure();
      }
    }
    return success();
  }

  /// Add repetition operation to a sequence
  Value createRepetition(Location loc,
                         const slang::ast::SequenceRepetition &repetition,
                         Value &inputSequence) {
    // Extract cycle range
    auto [minRepetitions, repetitionRange] =
        convertRangeToAttrs(repetition.range.min, repetition.range.max);

    using slang::ast::SequenceRepetition;

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

  Value visit(const slang::ast::SequenceWithMatchExpr &expr) {
    auto value =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!value)
      return {};
    if (expr.repetition.has_value()) {
      value = createRepetition(loc, expr.repetition.value(), value);
      if (!value)
        return {};
    }
    if (failed(handleMatchItems(expr.matchItems)))
      return {};
    return value;
  }

  Value visit(const slang::ast::SequenceConcatExpr &expr) {
    // Create a sequence of delayed operations, combined with a concat operation
    assert(!expr.elements.empty());

    SmallVector<Value> sequenceElements;
    uint64_t savedOffset = context.getAssertionSequenceOffset();
    uint64_t currentOffset = savedOffset;

    for (auto it = expr.elements.begin(); it != expr.elements.end(); ++it) {
      const auto &concatElement = *it;

      // Adjust inter-element delays to account for concat's cycle alignment.
      // For ##N between elements, concat already advances one cycle, so
      // subtract one when possible to align with SVA timing. The first element
      // delay is relative to the sequence start and should not be adjusted.
      uint32_t minDelay = concatElement.delay.min;
      std::optional<uint32_t> maxDelay = concatElement.delay.max;
      uint32_t ltlMinDelay = minDelay;
      std::optional<uint32_t> ltlMaxDelay = maxDelay;
      if (it != expr.elements.begin() && ltlMinDelay > 0) {
        --ltlMinDelay;
        if (ltlMaxDelay.has_value() && ltlMaxDelay.value() > 0)
          --ltlMaxDelay.value();
      }
      // Sequence offsets track the effective cycle position of each element.
      // Concat always advances one cycle between elements, so ##0 still moves
      // by one in the lowered LTL; reflect that when computing local-var pasts.
      uint32_t offsetDelay = minDelay;
      if (it != expr.elements.begin() && offsetDelay == 0)
        offsetDelay = 1;
      currentOffset += offsetDelay;
      context.setAssertionSequenceOffset(currentOffset);

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

      auto [delayMin, delayRange] =
          convertRangeToAttrs(ltlMinDelay, ltlMaxDelay);
      auto delayedSequence = ltl::DelayOp::create(builder, loc, sequenceValue,
                                                  delayMin, delayRange);
      sequenceElements.push_back(delayedSequence);
    }

    context.setAssertionSequenceOffset(savedOffset);
    return builder.createOrFold<ltl::ConcatOp>(loc, sequenceElements);
  }

  Value visit(const slang::ast::FirstMatchAssertionExpr &expr) {
    auto sequenceValue =
        context.convertAssertionExpression(expr.seq, loc, /*applyDefaults=*/false);
    if (!sequenceValue)
      return {};
    if (auto bounds = getSequenceLengthBounds(sequenceValue)) {
      if (!bounds->max) {
        mlir::emitError(loc) << "first_match requires a bounded sequence";
        return {};
      }
    }
    if (failed(handleMatchItems(expr.matchItems)))
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
        auto minDelay = builder.getI64IntegerAttr(expr.range.value().min);
        auto lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(
              expr.range.value().max.value() - expr.range.value().min);
        }
        return ltl::DelayOp::create(builder, loc, value, minDelay,
                                    lengthAttr);
      }
      return ltl::EventuallyOp::create(builder, loc, value);
    case UnaryAssertionOperator::Eventually: {
      if (expr.range.has_value()) {
        auto minDelay = builder.getI64IntegerAttr(expr.range.value().min);
        auto lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(
              expr.range.value().max.value() - expr.range.value().min);
        }
        return ltl::DelayOp::create(builder, loc, value, minDelay,
                                    lengthAttr);
      }
      auto eventually = ltl::EventuallyOp::create(builder, loc, value);
      eventually->setAttr(kWeakEventuallyAttr, builder.getUnitAttr());
      return eventually;
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
      mlir::IntegerAttr lengthAttr = builder.getI64IntegerAttr(0);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
        lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(expr.range.value().max.value() -
                                                 expr.range.value().min);
        }
      }
      return ltl::DelayOp::create(builder, loc, value, minRepetitions,
                                  lengthAttr);
    }
    case UnaryAssertionOperator::SNextTime: {
      auto minRepetitions = builder.getI64IntegerAttr(1);
      mlir::IntegerAttr lengthAttr = builder.getI64IntegerAttr(0);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
        lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(expr.range.value().max.value() -
                                                 expr.range.value().min);
        }
      }
      return ltl::DelayOp::create(builder, loc, value, minRepetitions,
                                  lengthAttr);
    }
    case UnaryAssertionOperator::SAlways: {
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      return ltl::RepeatOp::create(builder, loc, value, attr.first,
                                   attr.second);
    }
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
      auto minAttr = builder.getI64IntegerAttr(0);
      mlir::IntegerAttr moreAttr;
      if (auto bounds = getSequenceLengthBounds(rhs)) {
        minAttr = builder.getI64IntegerAttr(bounds->min);
        if (bounds->max && *bounds->max >= bounds->min)
          moreAttr = builder.getI64IntegerAttr(*bounds->max - bounds->min);
      }
      auto lhsRepeat =
          ltl::RepeatOp::create(builder, loc, lhs, minAttr, moreAttr);
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
    auto *previousTiming = context.currentAssertionTimingControl;
    auto timingGuard = llvm::make_scope_exit(
        [&] { context.currentAssertionTimingControl = previousTiming; });
    context.currentAssertionTimingControl = &expr.clocking;
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
          // $sampled is handled in convertAssertionCallExpression.
          .Case("$sampled", [&]() -> Value { return {}; })
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
    const slang::ast::TimingControl *clockingCtrl = nullptr;
    bool hasClockingArg =
        args.size() > 1 &&
        args[1]->kind == slang::ast::ExpressionKind::ClockingEvent;
    if (hasClockingArg) {
      if (auto *clockExpr =
              args[1]->as_if<slang::ast::ClockingEventExpression>()) {
        clockingCtrl = &clockExpr->timingControl;
      } else if (!inAssertionExpr) {
        auto resultType = moore::IntType::getInt(builder.getContext(), 1);
        mlir::emitWarning(loc)
            << subroutine.name
            << " with explicit clocking is not yet lowered outside assertions; "
               "returning 0 as a placeholder";
        return moore::ConstantOp::create(builder, loc, resultType, 0);
      }
    }

    const slang::ast::Expression *enableExpr = nullptr;
    bool invertEnable = false;
    if (inAssertionExpr && currentScope) {
      if (auto *defaultDisable = compilation.getDefaultDisable(*currentScope)) {
        enableExpr = defaultDisable;
        invertEnable = true;
      }
    }

    if (inAssertionExpr && !clockingCtrl && (hasClockingArg || enableExpr)) {
      if (currentAssertionClock)
        clockingCtrl = currentAssertionClock;
      if (!clockingCtrl && currentAssertionTimingControl)
        clockingCtrl = currentAssertionTimingControl;
      if (!clockingCtrl && currentScope) {
        if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
    }

    if (clockingCtrl && inAssertionExpr && (hasClockingArg || enableExpr)) {
      return lowerSampledValueFunctionWithClocking(
          *this, *args[0], *clockingCtrl, subroutine.name, enableExpr,
          invertEnable, loc);
    }

    if (hasClockingArg && !inAssertionExpr) {
      if (clockingCtrl) {
        return lowerSampledValueFunctionWithClocking(
            *this, *args[0], *clockingCtrl, subroutine.name, nullptr, false, loc);
      }
      auto resultType = moore::IntType::getInt(builder.getContext(), 1);
      mlir::emitWarning(loc)
          << subroutine.name
          << " with explicit clocking is not yet lowered outside assertions; "
             "returning 0 as a placeholder";
      return moore::ConstantOp::create(builder, loc, resultType, 0);
    }

    if (subroutine.name == "$stable" || subroutine.name == "$changed") {
      Value sampled = value;
      Value past;
      if (inAssertionExpr) {
        // Sampled-value semantics: compare the sampled value at this edge with
        // the sampled value from the previous edge.
        sampled = value;
        past =
            moore::PastOp::create(builder, loc, value, /*delay=*/1).getResult();
      } else {
        past =
            moore::PastOp::create(builder, loc, value, /*delay=*/1).getResult();
      }
      auto stable =
          moore::EqOp::create(builder, loc, sampled, past).getResult();
      Value resultVal = stable;
      if (subroutine.name == "$changed")
        resultVal = moore::NotOp::create(builder, loc, stable).getResult();
      return resultVal;
    }

    Value current = value;
    current = moore::BoolCastOp::create(builder, loc, current).getResult();
    Value sampled = current;
    Value past;
    if (inAssertionExpr) {
      // Sampled-value semantics: use the sampled current value and sampled past.
      sampled = current;
      past =
          moore::PastOp::create(builder, loc, current, /*delay=*/1).getResult();
    } else {
      past =
          moore::PastOp::create(builder, loc, current, /*delay=*/1).getResult();
    }
    Value resultVal;
    if (subroutine.name == "$rose") {
      auto notPast = moore::NotOp::create(builder, loc, past).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, sampled, notPast).getResult();
    } else {
      auto notCurrent =
          moore::NotOp::create(builder, loc, sampled).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, notCurrent, past).getResult();
    }
    return resultVal;
  }

  // Handle $past specially - it returns the past value with preserved type
  // so that comparisons like `$past(val) == 0` work correctly.
  if (subroutine.name == "$past") {
    // Get the delay (numTicks) from the second argument if present.
    // Default to 1 if empty or not provided.
    int64_t delay = 1;
    const slang::ast::TimingControl *clockingCtrl = nullptr;
    const slang::ast::Expression *enableExpr = nullptr;
    const slang::ast::Expression *defaultDisableExpr = nullptr;
    bool invertEnable = false;
    if (args.size() > 1 &&
        args[1]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[1]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (auto *clockExpr =
                args[1]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
      } else {
        auto cv = evaluateConstant(*args[1]);
        if (cv.isInteger()) {
          auto intVal = cv.integer().as<int64_t>();
          if (intVal)
            delay = *intVal;
        }
      }
    }
    if (!clockingCtrl && args.size() > 2 &&
        args[2]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[2]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (auto *clockExpr =
                args[2]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
      } else {
        enableExpr = args[2];
      }
    }
    if (args.size() > 3 &&
        args[3]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[3]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (clockingCtrl) {
          mlir::emitError(loc) << "multiple $past clocking events";
          return {};
        }
        if (auto *clockExpr =
                args[3]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
      } else if (!enableExpr) {
        enableExpr = args[3];
      } else {
        mlir::emitError(loc) << "too many $past arguments";
        return {};
      }
    }
    auto maybeSetImplicitClocking = [&]() {
      if (!clockingCtrl && currentAssertionClock)
        clockingCtrl = currentAssertionClock;
      if (!clockingCtrl && currentAssertionTimingControl)
        clockingCtrl = currentAssertionTimingControl;
      if (!clockingCtrl && currentScope) {
        if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
    };
    if (!enableExpr && currentScope)
      defaultDisableExpr = compilation.getDefaultDisable(*currentScope);

    if (enableExpr)
      maybeSetImplicitClocking();
    else if (defaultDisableExpr) {
      maybeSetImplicitClocking();
      if (clockingCtrl) {
        enableExpr = defaultDisableExpr;
        invertEnable = true;
      }
    }
    if (clockingCtrl)
      return lowerPastWithClocking(*this, *args[0], *clockingCtrl, delay,
                                   enableExpr, invertEnable, loc);
    if (enableExpr) {
      mlir::emitError(loc)
          << "unsupported $past enable expression without explicit clocking";
      return {};
    }

    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};

    // Always use moore::PastOp to preserve the type for comparisons.
    // $past(val) returns the sampled past value with the same type as val, so
    // that comparisons like `$past(val) == 0` work correctly.
    return moore::PastOp::create(builder, loc, value, delay).getResult();
  }

  switch (args.size()) {
  case (1):
    value = this->convertRvalueExpression(*args[0]);

    // $sampled returns the sampled value of the expression.
    if (subroutine.name == "$sampled") {
      if (inAssertionExpr)
        return moore::PastOp::create(builder, loc, value, /*delay=*/0)
            .getResult();
      return value;
    }

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
  if (!prevInAssertionExpr) {
    pushAssertionLocalVarScope();
    pushAssertionSequenceOffset(0);
  }
  inAssertionExpr = true;
  AssertionExprVisitor visitor{*this, loc};
  auto value = expr.visit(visitor);
  inAssertionExpr = prevInAssertionExpr;
  if (!prevInAssertionExpr) {
    popAssertionSequenceOffset();
    popAssertionLocalVarScope();
  }
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
