//===- CrossSelect.cpp - Slang cross-select lowering ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/CallExpression.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/statements/ConditionalStatements.h"
#include "slang/ast/statements/LoopStatements.h"
#include "slang/ast/statements/MiscStatements.h"
#include "slang/ast/symbols/CoverSymbols.h"
#include "llvm/ADT/ScopeExit.h"
#include <limits>

using namespace circt;

namespace circt::ImportVerilog {

namespace {
struct CrossSelectLeaf {
  const slang::ast::ConditionBinsSelectExpr *cond = nullptr;
  bool negate = false;
};

using CrossSelectTerm = SmallVector<CrossSelectLeaf, 4>;
using CrossSelectDNF = SmallVector<CrossSelectTerm, 4>;

static mlir::SymbolRefAttr resolveCrossSelectTargetRef(
    const slang::ast::Symbol &target,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder) {
  mlir::SymbolRefAttr targetRef;
  if (target.kind == slang::ast::SymbolKind::Coverpoint) {
    auto it = coverpointSymbols.find(target.name);
    if (it != coverpointSymbols.end())
      targetRef = it->second;
    else
      targetRef =
          mlir::FlatSymbolRefAttr::get(builder.getContext(), target.name);
  } else if (target.kind == slang::ast::SymbolKind::CoverageBin) {
    auto *parentScope = target.getParentScope();
    if (parentScope) {
      auto &parentSym = parentScope->asSymbol();
      if (parentSym.kind == slang::ast::SymbolKind::Coverpoint) {
        auto it = coverpointSymbols.find(parentSym.name);
        if (it != coverpointSymbols.end()) {
          targetRef = mlir::SymbolRefAttr::get(
              builder.getContext(), it->second.getValue(),
              {mlir::FlatSymbolRefAttr::get(builder.getContext(), target.name)});
        }
      }
    }
  }

  if (!targetRef)
    targetRef = mlir::FlatSymbolRefAttr::get(builder.getContext(), target.name);
  return targetRef;
}

static std::optional<int64_t>
getConstantInt64(const slang::ConstantValue &value) {
  if (!value.isInteger())
    return std::nullopt;
  if (value.integer().hasUnknown())
    return std::nullopt;
  return value.integer().as<int64_t>();
}

static bool isUnboundedConstantExpr(
    const slang::ast::Expression &expr,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  if (expr.kind == slang::ast::ExpressionKind::UnboundedLiteral)
    return true;
  if (expr.type && expr.type->isUnbounded())
    return true;
  auto value = evaluateConstant(expr);
  return value.isUnbounded();
}

static LogicalResult evaluateIntersectToleranceRangeBounds(
    const slang::ast::ValueRangeExpression &rangeExpr, Location loc,
    int64_t &lower, int64_t &upper) {
  auto centerConst = rangeExpr.left().getConstant();
  auto toleranceConst = rangeExpr.right().getConstant();
  if (!centerConst || !toleranceConst)
    return mlir::emitError(loc)
           << "unsupported non-constant intersect value range in cross "
              "select expression";

  auto center = getConstantInt64(*centerConst);
  auto tolerance = getConstantInt64(*toleranceConst);
  if (!center || !tolerance)
    return mlir::emitError(loc)
           << "unsupported non-constant intersect value range in cross "
              "select expression";

  __int128 span = static_cast<__int128>(*tolerance);
  if (rangeExpr.rangeKind == slang::ast::ValueRangeKind::RelativeTolerance)
    span = (static_cast<__int128>(*center) * static_cast<__int128>(*tolerance)) /
           static_cast<__int128>(100);

  __int128 lowerWide = static_cast<__int128>(*center) - span;
  __int128 upperWide = static_cast<__int128>(*center) + span;
  constexpr __int128 kI64Min =
      static_cast<__int128>(std::numeric_limits<int64_t>::min());
  constexpr __int128 kI64Max =
      static_cast<__int128>(std::numeric_limits<int64_t>::max());
  if (lowerWide < kI64Min || lowerWide > kI64Max || upperWide < kI64Min ||
      upperWide > kI64Max)
    return mlir::emitError(loc)
           << "unsupported intersect value range in cross select expression";

  lower = static_cast<int64_t>(lowerWide);
  upper = static_cast<int64_t>(upperWide);
  if (lower > upper)
    std::swap(lower, upper);
  return success();
}

struct CrossMatchesPolicy {
  uint64_t minMatches = 1;
  bool requireAll = false;
};

static FailureOr<CrossMatchesPolicy> parseCrossMatchesPolicy(
    const slang::ast::Expression *matchesExpr, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  CrossMatchesPolicy policy;
  if (!matchesExpr)
    return policy;
  if (matchesExpr->kind == slang::ast::ExpressionKind::UnboundedLiteral ||
      matchesExpr->type->isUnbounded()) {
    policy.requireAll = true;
    return policy;
  }
  auto matchesValue = evaluateConstant(*matchesExpr);
  if (matchesValue.bad())
    return mlir::emitError(loc)
           << "unsupported cross select expression: non-constant 'matches' "
              "policy";
  if (matchesValue.isUnbounded()) {
    policy.requireAll = true;
    return policy;
  }
  auto matchesInt = getConstantInt64(matchesValue);
  if (!matchesInt || *matchesInt <= 0)
    return mlir::emitError(loc)
           << "unsupported cross select expression: non-constant 'matches' "
              "policy";
  policy.minMatches = static_cast<uint64_t>(*matchesInt);
  return policy;
}

static slang::ConstantValue evaluateCrossSelectScriptExpr(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation) {
  using namespace slang::ast;
  auto flags = EvalFlags::CacheResults | EvalFlags::SpecparamsAllowed |
               EvalFlags::IsScript;

  const Symbol *contextSymbol = expr.getSymbolReference();
  if (!contextSymbol) {
    expr.visitSymbolReferences(
        [&](const Expression &, const Symbol &symbol) {
          if (!contextSymbol)
            contextSymbol = &symbol;
        });
  }

  EvalContext evalContext = contextSymbol
                                ? EvalContext(*contextSymbol, flags)
                                : EvalContext(
                                      ASTContext(compilation.getRoot(),
                                                 LookupLocation::max),
                                      flags);
  evalContext.pushEmptyFrame();
  auto result = expr.eval(evalContext);
  evalContext.popFrame();
  return result;
}

static slang::ConstantValue evaluateCrossSetHelperExpr(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation,
    slang::ast::EvalContext &evalContext,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  auto value = expr.eval(evalContext);
  if (!value.bad())
    return value;
  value = evaluateConstant(expr);
  if (!value.bad())
    return value;
  return evaluateCrossSelectScriptExpr(expr, compilation);
}

enum class CrossSetPushBackCollectResult {
  Success,
  Break,
  Continue,
  Disable,
  Return,
  Fail
};

static slang::ConstantValue evaluateCrossSetExprFromPushBackHelper(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    slang::ast::EvalContext *callerEvalContext = nullptr,
    const slang::ast::SubroutineSymbol *excludeSubroutine = nullptr);

static slang::ConstantValue evaluateCrossSetExprWithFallback(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation,
    slang::ast::EvalContext &evalContext,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    const slang::ast::SubroutineSymbol *excludeSubroutine) {
  using namespace slang::ast;
  if (expr.kind == ExpressionKind::Call)
    return evaluateCrossSetExprFromPushBackHelper(
        expr, compilation, evaluateConstant, &evalContext, excludeSubroutine);

  if (auto *conversionExpr = expr.as_if<ConversionExpression>()) {
    auto convertedVal = evaluateCrossSetHelperExpr(
        conversionExpr->operand(), compilation, evalContext, evaluateConstant);
    if (!convertedVal.bad())
      return convertedVal;
    return evaluateCrossSetExprWithFallback(
        conversionExpr->operand(), compilation, evalContext, evaluateConstant,
        excludeSubroutine);
  }

  if (auto *concatExpr = expr.as_if<ConcatenationExpression>()) {
    slang::ConstantValue::Elements concatenated;
    for (auto *operand : concatExpr->operands()) {
      if (!operand || operand->type->isVoid())
        continue;

      auto operandVal = evaluateCrossSetHelperExpr(
          *operand, compilation, evalContext, evaluateConstant);
      if (operandVal.bad())
        operandVal = evaluateCrossSetExprWithFallback(
            *operand, compilation, evalContext, evaluateConstant,
            excludeSubroutine);
      if (operandVal.bad())
        return {};

      if (operandVal.isQueue()) {
        for (auto &element : *operandVal.queue())
          concatenated.push_back(element);
        continue;
      }
      if (operandVal.isUnpacked()) {
        for (auto &element : operandVal.elements())
          concatenated.push_back(element);
        continue;
      }
      concatenated.push_back(std::move(operandVal));
    }
    return slang::ConstantValue(std::move(concatenated));
  }

  auto *conditionalExpr = expr.as_if<ConditionalExpression>();
  if (!conditionalExpr)
    return {};
  if (conditionalExpr->conditions.size() != 1)
    return {};
  auto &cond = conditionalExpr->conditions.front();
  if (!cond.expr || cond.pattern)
    return {};

  auto condVal = evaluateCrossSetHelperExpr(*cond.expr, compilation, evalContext,
                                            evaluateConstant);
  if (condVal.bad())
    return {};

  auto &branchExpr = condVal.isTrue() ? conditionalExpr->left()
                                      : conditionalExpr->right();
  auto branchVal = evaluateCrossSetHelperExpr(
      branchExpr, compilation, evalContext, evaluateConstant);
  if (!branchVal.bad())
    return branchVal;
  return evaluateCrossSetExprWithFallback(
      branchExpr, compilation, evalContext, evaluateConstant, excludeSubroutine);
}

static bool extractCrossSetTuplesFromValue(const slang::ConstantValue &value,
                                           slang::ConstantValue::Elements &out) {
  out.clear();
  if (value.isQueue()) {
    for (const auto &element : *value.queue())
      out.push_back(element);
    return true;
  }
  if (value.isUnpacked()) {
    for (const auto &element : value.elements())
      out.push_back(element);
    return true;
  }
  return false;
}

static CrossSetPushBackCollectResult collectCrossSetPushBackTuples(
    const slang::ast::Statement &stmt,
    const slang::ast::ValueSymbol *returnVar,
    slang::ast::Compilation &compilation,
    slang::ast::EvalContext &evalContext,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    uint64_t &iterationBudget,
    slang::ConstantValue::Elements &tuples) {
  using namespace slang::ast;
  auto fail = CrossSetPushBackCollectResult::Fail;
  auto success = CrossSetPushBackCollectResult::Success;

  switch (stmt.kind) {
  case StatementKind::Empty:
    return success;
  case StatementKind::Break:
    return CrossSetPushBackCollectResult::Break;
  case StatementKind::Continue:
    return CrossSetPushBackCollectResult::Continue;
  case StatementKind::List: {
    auto &list = stmt.as<StatementList>();
    for (auto *child : list.list) {
      if (!child)
        continue;
      auto result =
          collectCrossSetPushBackTuples(*child, returnVar, compilation,
                                        evalContext, evaluateConstant,
                                        iterationBudget, tuples);
      if (result != success)
        return result;
    }
    return success;
  }
  case StatementKind::Block: {
    auto &block = stmt.as<BlockStatement>();
    auto result = collectCrossSetPushBackTuples(
        block.body, returnVar, compilation, evalContext, evaluateConstant,
        iterationBudget, tuples);
    if (result != CrossSetPushBackCollectResult::Disable)
      return result;

    auto *disableTarget = evalContext.getDisableTarget();
    if (block.blockSymbol && disableTarget == block.blockSymbol) {
      evalContext.setDisableTarget(nullptr, {});
      return success;
    }
    return result;
  }
  case StatementKind::ForLoop: {
    auto &forLoop = stmt.as<ForLoopStatement>();
    const slang::ast::SubroutineSymbol *excludeSubroutine = nullptr;
    if (returnVar) {
      if (auto *scope = returnVar->getParentScope())
        excludeSubroutine =
            scope->asSymbol().as_if<slang::ast::SubroutineSymbol>();
    }

    auto evaluateForExpr =
        [&](const slang::ast::Expression &expr) -> bool {
      auto exprValue = evaluateCrossSetHelperExpr(expr, compilation, evalContext,
                                                  evaluateConstant);
      if (!exprValue.bad())
        return true;

      if (expr.kind == slang::ast::ExpressionKind::Assignment) {
        auto &assign = expr.as<slang::ast::AssignmentExpression>();
        auto nested = evaluateCrossSetExprWithFallback(
            assign.right(), compilation, evalContext, evaluateConstant,
            excludeSubroutine);
        if (!nested.bad()) {
          auto *lhsSym = assign.left().getSymbolReference();
          auto *lhsValue = lhsSym ? lhsSym->as_if<slang::ast::ValueSymbol>()
                                  : nullptr;
          if (!lhsValue)
            return false;
          auto *slot = evalContext.findLocal(lhsValue);
          if (!slot)
            return false;
          *slot = std::move(nested);
          return true;
        }
      }

      auto nested = evaluateCrossSetExprWithFallback(
          expr, compilation, evalContext, evaluateConstant, excludeSubroutine);
      return !nested.bad();
    };

    for (auto *init : forLoop.initializers) {
      if (!init)
        return fail;
      if (!evaluateForExpr(*init))
        return fail;
    }

    SmallVector<const slang::ast::ValueSymbol *, 4> createdLoopVars;
    for (auto *loopVar : forLoop.loopVars) {
      if (!loopVar)
        return fail;
      auto initVal = loopVar->getType().getDefaultValue();
      if (auto *initExpr = loopVar->getInitializer()) {
        initVal = evaluateCrossSetHelperExpr(*initExpr, compilation, evalContext,
                                             evaluateConstant);
        if (initVal.bad())
          initVal = evaluateCrossSetExprWithFallback(
              *initExpr, compilation, evalContext, evaluateConstant,
              excludeSubroutine);
      }
      if (initVal.bad())
        return fail;
      evalContext.createLocal(loopVar, std::move(initVal));
      createdLoopVars.push_back(loopVar);
    }
    auto cleanupLoopVars = llvm::make_scope_exit([&] {
      for (auto *loopVar : createdLoopVars)
        evalContext.deleteLocal(loopVar);
    });

    while (true) {
      if (iterationBudget == 0)
        return fail;
      --iterationBudget;

      if (forLoop.stopExpr) {
        auto stopVal = evaluateCrossSetHelperExpr(*forLoop.stopExpr, compilation,
                                                  evalContext, evaluateConstant);
        if (stopVal.bad())
          return fail;
        if (!stopVal.isTrue())
          break;
      }

      auto bodyResult = collectCrossSetPushBackTuples(
          forLoop.body, returnVar, compilation, evalContext, evaluateConstant,
          iterationBudget, tuples);
      if (bodyResult == fail)
        return fail;
      if (bodyResult == CrossSetPushBackCollectResult::Break)
        break;
      if (bodyResult == CrossSetPushBackCollectResult::Disable)
        return bodyResult;
      if (bodyResult == CrossSetPushBackCollectResult::Return)
        return bodyResult;

      for (auto *step : forLoop.steps) {
        if (!step)
          return fail;
        if (!evaluateForExpr(*step))
          return fail;
      }
    }
    return success;
  }
  case StatementKind::WhileLoop: {
    auto &whileLoop = stmt.as<WhileLoopStatement>();
    while (true) {
      if (iterationBudget == 0)
        return fail;
      --iterationBudget;

      auto condVal = evaluateCrossSetHelperExpr(whileLoop.cond, compilation,
                                                evalContext, evaluateConstant);
      if (condVal.bad())
        return fail;
      if (!condVal.isTrue())
        break;

      auto bodyResult =
          collectCrossSetPushBackTuples(whileLoop.body, returnVar, compilation,
                                        evalContext, evaluateConstant,
                                        iterationBudget, tuples);
      if (bodyResult == fail)
        return fail;
      if (bodyResult == CrossSetPushBackCollectResult::Break)
        break;
      if (bodyResult == CrossSetPushBackCollectResult::Disable)
        return bodyResult;
      if (bodyResult == CrossSetPushBackCollectResult::Return)
        return bodyResult;
    }
    return success;
  }
  case StatementKind::DoWhileLoop: {
    auto &doWhile = stmt.as<DoWhileLoopStatement>();
    while (true) {
      if (iterationBudget == 0)
        return fail;
      --iterationBudget;

      auto bodyResult =
          collectCrossSetPushBackTuples(doWhile.body, returnVar, compilation,
                                        evalContext, evaluateConstant,
                                        iterationBudget, tuples);
      if (bodyResult == fail)
        return fail;
      if (bodyResult == CrossSetPushBackCollectResult::Break)
        break;
      if (bodyResult == CrossSetPushBackCollectResult::Disable)
        return bodyResult;
      if (bodyResult == CrossSetPushBackCollectResult::Return)
        return bodyResult;

      auto condVal = evaluateCrossSetHelperExpr(doWhile.cond, compilation,
                                                evalContext, evaluateConstant);
      if (condVal.bad())
        return fail;
      if (!condVal.isTrue())
        break;
    }
    return success;
  }
  case StatementKind::ForeverLoop: {
    auto &forever = stmt.as<ForeverLoopStatement>();
    while (true) {
      if (iterationBudget == 0)
        return fail;
      --iterationBudget;

      auto bodyResult =
          collectCrossSetPushBackTuples(forever.body, returnVar, compilation,
                                        evalContext, evaluateConstant,
                                        iterationBudget, tuples);
      if (bodyResult == fail)
        return fail;
      if (bodyResult == CrossSetPushBackCollectResult::Break)
        break;
      if (bodyResult == CrossSetPushBackCollectResult::Disable)
        return bodyResult;
      if (bodyResult == CrossSetPushBackCollectResult::Return)
        return bodyResult;
    }
    return success;
  }
  case StatementKind::RepeatLoop: {
    auto &repeat = stmt.as<RepeatLoopStatement>();
    auto countVal = evaluateCrossSetHelperExpr(repeat.count, compilation,
                                               evalContext, evaluateConstant);
    auto count = getConstantInt64(countVal);
    if (!count || *count < 0)
      return fail;

    for (int64_t i = 0; i < *count; ++i) {
      if (iterationBudget == 0)
        return fail;
      --iterationBudget;

      auto bodyResult =
          collectCrossSetPushBackTuples(repeat.body, returnVar, compilation,
                                        evalContext, evaluateConstant,
                                        iterationBudget, tuples);
      if (bodyResult == fail)
        return fail;
      if (bodyResult == CrossSetPushBackCollectResult::Break)
        break;
      if (bodyResult == CrossSetPushBackCollectResult::Disable)
        return bodyResult;
      if (bodyResult == CrossSetPushBackCollectResult::Return)
        return bodyResult;
    }
    return success;
  }
  case StatementKind::Conditional: {
    auto &condStmt = stmt.as<ConditionalStatement>();
    bool matched = true;
    for (auto &cond : condStmt.conditions) {
      if (cond.pattern)
        return fail;
      auto condVal = evaluateCrossSetHelperExpr(*cond.expr, compilation,
                                                evalContext, evaluateConstant);
      if (condVal.bad())
        return fail;
      if (!condVal.isTrue()) {
        matched = false;
        break;
      }
    }

    if (matched)
      return collectCrossSetPushBackTuples(condStmt.ifTrue, returnVar, compilation,
                                           evalContext, evaluateConstant,
                                           iterationBudget, tuples);
    if (condStmt.ifFalse)
      return collectCrossSetPushBackTuples(*condStmt.ifFalse, returnVar,
                                           compilation, evalContext,
                                           evaluateConstant, iterationBudget,
                                           tuples);
    return success;
  }
  case StatementKind::Case: {
    auto &caseStmt = stmt.as<CaseStatement>();
    auto [branch, known] = caseStmt.getKnownBranch(evalContext);
    if (!known)
      return fail;
    if (!branch)
      return success;
    return collectCrossSetPushBackTuples(*branch, returnVar, compilation,
                                         evalContext, evaluateConstant,
                                         iterationBudget, tuples);
  }
  case StatementKind::ForeachLoop: {
    auto &foreachLoop = stmt.as<ForeachLoopStatement>();
    if (foreachLoop.loopDims.empty())
      return success;

    SmallVector<const slang::ast::ValueSymbol *, 4> createdIterators;
    SmallVector<int64_t, 4> loopIndices(foreachLoop.loopDims.size(), 0);
    auto cleanupIterators = llvm::make_scope_exit([&] {
      for (auto *iter : createdIterators)
        evalContext.deleteLocal(iter);
    });

    auto setIterator = [&](const slang::ast::IteratorSymbol *iter,
                           int64_t index) -> bool {
      if (!iter)
        return true;
      auto width = iter->getType().getBitWidth();
      if (width == 0 || width > 64)
        return false;
      auto intValue = slang::SVInt(width, static_cast<uint64_t>(index),
                                   iter->getType().isSigned());
      slang::ConstantValue value(std::move(intValue));
      if (auto *slot = evalContext.findLocal(iter)) {
        *slot = std::move(value);
        return true;
      }
      evalContext.createLocal(iter, std::move(value));
      createdIterators.push_back(iter);
      return true;
    };

    auto getDynamicDimSize = [&](size_t dimIndex,
                                 uint64_t &count) -> bool {
      auto container = evaluateCrossSetHelperExpr(
          foreachLoop.arrayRef, compilation, evalContext, evaluateConstant);
      if (container.bad())
        return false;

      for (size_t i = 0; i < dimIndex; ++i) {
        auto index = loopIndices[i];
        if (index < 0)
          return false;
        if (container.isQueue()) {
          auto &queue = *container.queue();
          if (static_cast<size_t>(index) >= queue.size())
            return false;
          container = queue[static_cast<size_t>(index)];
          continue;
        }
        if (container.isUnpacked()) {
          auto elements = container.elements();
          if (static_cast<size_t>(index) >= elements.size())
            return false;
          container = elements[static_cast<size_t>(index)];
          continue;
        }
        return false;
      }

      if (container.isQueue()) {
        count = container.queue()->size();
        return true;
      }
      if (container.isUnpacked()) {
        count = container.elements().size();
        return true;
      }
      return false;
    };

    std::function<CrossSetPushBackCollectResult(size_t)> visitLoopDims =
        [&](size_t dimIndex) -> CrossSetPushBackCollectResult {
      if (dimIndex >= foreachLoop.loopDims.size())
        return collectCrossSetPushBackTuples(
            foreachLoop.body, returnVar, compilation, evalContext,
            evaluateConstant, iterationBudget, tuples);

      auto &dim = foreachLoop.loopDims[dimIndex];
      auto runIteration = [&](int64_t index)
          -> CrossSetPushBackCollectResult {
        if (iterationBudget == 0)
          return fail;
        --iterationBudget;

        loopIndices[dimIndex] = index;
        if (!setIterator(dim.loopVar, index))
          return fail;

        auto result = visitLoopDims(dimIndex + 1);
        if (result == fail)
          return fail;
        if (result == CrossSetPushBackCollectResult::Disable)
          return result;
        if (result == CrossSetPushBackCollectResult::Return)
          return result;
        return result;
      };

      if (dim.range) {
        int64_t lower = dim.range->lower();
        int64_t upper = dim.range->upper();
        int64_t step = lower <= upper ? 1 : -1;

        for (int64_t index = lower;; index += step) {
          auto result = runIteration(index);
          if (result == fail)
            return fail;
          if (result == CrossSetPushBackCollectResult::Break)
            break;
          if (result == CrossSetPushBackCollectResult::Disable ||
              result == CrossSetPushBackCollectResult::Return)
            return result;
          if (index == upper)
            break;
        }
        return success;
      }

      uint64_t count = 0;
      if (!getDynamicDimSize(dimIndex, count))
        return fail;
      for (uint64_t i = 0; i < count; ++i) {
        auto result = runIteration(static_cast<int64_t>(i));
        if (result == fail)
          return fail;
        if (result == CrossSetPushBackCollectResult::Break)
          break;
        if (result == CrossSetPushBackCollectResult::Disable ||
            result == CrossSetPushBackCollectResult::Return)
          return result;
      }
      return success;
    };

    auto result = visitLoopDims(0);
    if (result == CrossSetPushBackCollectResult::Break ||
        result == CrossSetPushBackCollectResult::Continue)
      return success;
    return result;
  }
  case StatementKind::Disable: {
    auto evalResult = stmt.eval(evalContext);
    if (evalResult == slang::ast::Statement::EvalResult::Disable)
      return CrossSetPushBackCollectResult::Disable;
    if (evalResult == slang::ast::Statement::EvalResult::Success)
      return success;
    return fail;
  }
  case StatementKind::VariableDeclaration: {
    auto &varDecl = stmt.as<VariableDeclStatement>();
    auto initialValue = varDecl.symbol.getType().getDefaultValue();
    if (auto *initializer = varDecl.symbol.getInitializer()) {
      initialValue = evaluateCrossSetHelperExpr(*initializer, compilation,
                                                evalContext, evaluateConstant);
      if (initialValue.bad()) {
        const slang::ast::SubroutineSymbol *excludeSubroutine = nullptr;
        if (returnVar) {
          if (auto *scope = returnVar->getParentScope())
            excludeSubroutine =
                scope->asSymbol().as_if<slang::ast::SubroutineSymbol>();
        }
        initialValue = evaluateCrossSetExprWithFallback(
            *initializer, compilation, evalContext, evaluateConstant,
            excludeSubroutine);
      }
    }
    if (initialValue.bad())
      return fail;
    evalContext.createLocal(&varDecl.symbol, std::move(initialValue));
    return success;
  }
  case StatementKind::ExpressionStatement: {
    auto &exprStmt = stmt.as<ExpressionStatement>();
    if (exprStmt.expr.kind == ExpressionKind::Call) {
      auto &callExpr = exprStmt.expr.as<CallExpression>();
      auto subroutineName = callExpr.getSubroutineName();
      if (subroutineName == "push_back" || subroutineName == "push_front" ||
          subroutineName == "insert") {
        auto args = callExpr.arguments();
        if (args.empty() || !args[0])
          return fail;

        auto *targetSym = args[0]->getSymbolReference();
        auto *valueSym = targetSym ? targetSym->as_if<slang::ast::ValueSymbol>()
                                   : nullptr;
        if (!valueSym)
          return fail;

        auto *queueValue = evalContext.findLocal(valueSym);
        if (!queueValue || !queueValue->isQueue())
          return fail;

        if (subroutineName == "push_back" || subroutineName == "push_front") {
          if (args.size() < 2 || !args[1])
            return fail;
          auto element = evaluateCrossSetHelperExpr(*args[1], compilation,
                                                    evalContext, evaluateConstant);
          if (element.bad())
            return fail;

          if (subroutineName == "push_back")
            queueValue->queue()->push_back(std::move(element));
          else
            queueValue->queue()->push_front(std::move(element));
          queueValue->queue()->resizeToBound();
          return success;
        }

        if (args.size() < 3 || !args[1] || !args[2])
          return fail;
        auto indexValue = evaluateCrossSetHelperExpr(*args[1], compilation,
                                                     evalContext, evaluateConstant);
        auto index = getConstantInt64(indexValue);
        if (!index || *index < 0)
          return fail;
        auto insertValue = evaluateCrossSetHelperExpr(*args[2], compilation,
                                                      evalContext, evaluateConstant);
        if (insertValue.bad())
          return fail;
        if (static_cast<size_t>(*index) > queueValue->queue()->size())
          return fail;
        auto it = queueValue->queue()->begin();
        std::advance(it, static_cast<ptrdiff_t>(*index));
        queueValue->queue()->insert(it, std::move(insertValue));
        queueValue->queue()->resizeToBound();
        return success;
      }

      if (subroutineName == "delete" || subroutineName == "sort" ||
          subroutineName == "rsort" || subroutineName == "shuffle" ||
          subroutineName == "reverse") {
        (void)callExpr.eval(evalContext);
        return success;
      }
    }

    auto exprValue =
        evaluateCrossSetHelperExpr(exprStmt.expr, compilation, evalContext,
                                   evaluateConstant);
    if (exprValue.bad()) {
      if (exprStmt.expr.kind == slang::ast::ExpressionKind::Assignment) {
        auto &assign = exprStmt.expr.as<slang::ast::AssignmentExpression>();
        const slang::ast::SubroutineSymbol *excludeSubroutine = nullptr;
        if (returnVar) {
          if (auto *scope = returnVar->getParentScope())
            excludeSubroutine =
                scope->asSymbol().as_if<slang::ast::SubroutineSymbol>();
        }
        auto nested = evaluateCrossSetExprWithFallback(
            assign.right(), compilation, evalContext, evaluateConstant,
            excludeSubroutine);
        if (!nested.bad()) {
          auto *lhsSym = assign.left().getSymbolReference();
          auto *lhsValue = lhsSym ? lhsSym->as_if<slang::ast::ValueSymbol>()
                                  : nullptr;
          if (!lhsValue)
            return fail;
          auto *slot = evalContext.findLocal(lhsValue);
          if (!slot)
            return fail;
          *slot = std::move(nested);
          return success;
        }
      }
      return fail;
    }
    return success;
  }
  case StatementKind::Return: {
    auto &ret = stmt.as<ReturnStatement>();
    if (!ret.expr)
      return CrossSetPushBackCollectResult::Return;
    auto retValue = evaluateCrossSetHelperExpr(*ret.expr, compilation, evalContext,
                                               evaluateConstant);
    if (retValue.bad()) {
      const slang::ast::SubroutineSymbol *excludeSubroutine = nullptr;
      if (returnVar) {
        if (auto *scope = returnVar->getParentScope())
          excludeSubroutine =
              scope->asSymbol().as_if<slang::ast::SubroutineSymbol>();
      }
      retValue = evaluateCrossSetExprWithFallback(
          *ret.expr, compilation, evalContext, evaluateConstant,
          excludeSubroutine);
    }
    if (returnVar && !retValue.bad())
      if (auto *slot = evalContext.findLocal(returnVar))
        *slot = retValue;
    if (!retValue.bad() && extractCrossSetTuplesFromValue(retValue, tuples))
      return CrossSetPushBackCollectResult::Return;
    // Allow "return <function_name>;" in helpers that build via push_back.
    auto *retSym = ret.expr->getSymbolReference();
    return retSym == returnVar ? CrossSetPushBackCollectResult::Return : fail;
  }
  default:
    return fail;
  }
}

static slang::ConstantValue evaluateCrossSetExprFromPushBackHelper(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    slang::ast::EvalContext *callerEvalContext,
    const slang::ast::SubroutineSymbol *excludeSubroutine) {
  using namespace slang::ast;
  if (expr.kind != ExpressionKind::Call)
    return {};
  auto &callExpr = expr.as<CallExpression>();
  if (callExpr.isSystemCall())
    return {};
  auto *subroutine = std::get<0>(callExpr.subroutine);
  if (!subroutine || subroutine->subroutineKind != SubroutineKind::Function)
    return {};
  if (excludeSubroutine && subroutine == excludeSubroutine)
    return {};
  if (callExpr.arguments().size() != subroutine->getArguments().size())
    return {};

  auto flags = slang::ast::EvalFlags::CacheResults |
               slang::ast::EvalFlags::SpecparamsAllowed |
               slang::ast::EvalFlags::IsScript;
  auto actualArgs = callExpr.arguments();
  auto formalArgs = subroutine->getArguments();

  auto initializeEvalContext = [&](slang::ast::EvalContext &evalContext) -> bool {
    for (auto [actual, formal] : llvm::zip(actualArgs, formalArgs)) {
      if (!actual || !formal)
        return false;
      auto value = callerEvalContext
                       ? evaluateCrossSetHelperExpr(*actual, compilation,
                                                    *callerEvalContext,
                                                    evaluateConstant)
                       : evaluateCrossSetHelperExpr(*actual, compilation,
                                                    evalContext,
                                                    evaluateConstant);
      if (value.bad())
        return false;
      evalContext.createLocal(formal, std::move(value));
    }
    if (subroutine->returnValVar) {
      auto returnDefault = subroutine->returnValVar->getType().getDefaultValue();
      if (returnDefault.bad())
        return false;
      evalContext.createLocal(subroutine->returnValVar, std::move(returnDefault));
    }
    return true;
  };

  auto extractReturnValueAsTuples = [&](slang::ast::EvalContext &evalContext)
      -> std::optional<slang::ConstantValue> {
    if (!subroutine->returnValVar)
      return std::nullopt;
    auto *returnValue = evalContext.findLocal(subroutine->returnValVar);
    if (!returnValue)
      return std::nullopt;
    if (!returnValue->isQueue() && !returnValue->isUnpacked())
      return std::nullopt;
    return *returnValue;
  };

  // Structural tuple extraction for helper patterns that cannot be handled by
  // plain expression evaluation (for example, queue mutator side effects).
  slang::ast::EvalContext evalContext(*subroutine, flags);
  evalContext.pushEmptyFrame();
  auto cleanupEvalFrame = llvm::make_scope_exit([&] { evalContext.popFrame(); });

  if (!initializeEvalContext(evalContext))
    return {};

  slang::ConstantValue::Elements tuples;
  uint64_t iterationBudget = 8192;
  auto result = collectCrossSetPushBackTuples(
      subroutine->getBody(), subroutine->returnValVar, compilation, evalContext,
      evaluateConstant, iterationBudget, tuples);
  if (result != CrossSetPushBackCollectResult::Success &&
      result != CrossSetPushBackCollectResult::Return)
    return {};
  if (auto extracted = extractReturnValueAsTuples(evalContext))
    return *extracted;
  return slang::ConstantValue(std::move(tuples));
}

static slang::ConstantValue evaluateCrossSetExpr(
    const slang::ast::Expression &expr, slang::ast::Compilation &compilation,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  auto value = evaluateConstant(expr);
  if (!value.bad())
    return value;
  value = evaluateCrossSelectScriptExpr(expr, compilation);
  if (!value.bad())
    return value;
  return evaluateCrossSetExprFromPushBackHelper(expr, compilation,
                                                evaluateConstant);
}

/// Build a DNF (OR-of-AND) form from a cross select expression.
/// Empty terms represent "true" (plain cross-id selection).
static LogicalResult buildCrossSelectDNF(const slang::ast::BinsSelectExpr &expr,
                                         Location loc, CrossSelectDNF &out,
                                         bool negate) {
  switch (expr.kind) {
  case slang::ast::BinsSelectExprKind::Condition: {
    auto &condExpr =
        static_cast<const slang::ast::ConditionBinsSelectExpr &>(expr);
    CrossSelectTerm term;
    term.push_back({&condExpr, negate});
    out.push_back(std::move(term));
    return success();
  }
  case slang::ast::BinsSelectExprKind::CrossId: {
    if (negate)
      return mlir::emitError(loc)
             << "unsupported negation of cross identifier in cross select "
                "expression";
    out.push_back(CrossSelectTerm{});
    return success();
  }
  case slang::ast::BinsSelectExprKind::Unary: {
    auto &unaryExpr =
        static_cast<const slang::ast::UnaryBinsSelectExpr &>(expr);
    return buildCrossSelectDNF(unaryExpr.expr, loc, out, !negate);
  }
  case slang::ast::BinsSelectExprKind::Binary: {
    auto &binaryExpr =
        static_cast<const slang::ast::BinaryBinsSelectExpr &>(expr);
    CrossSelectDNF leftTerms;
    CrossSelectDNF rightTerms;
    if (failed(buildCrossSelectDNF(binaryExpr.left, loc, leftTerms, negate)))
      return failure();
    if (failed(buildCrossSelectDNF(binaryExpr.right, loc, rightTerms, negate)))
      return failure();

    bool isOr = binaryExpr.op == slang::ast::BinaryBinsSelectExpr::Or;
    if (negate)
      isOr = !isOr;

    if (isOr) {
      out.append(leftTerms.begin(), leftTerms.end());
      out.append(rightTerms.begin(), rightTerms.end());
      return success();
    }

    // AND: compute conjunction cross product of both term sets.
    for (auto &left : leftTerms) {
      for (auto &right : rightTerms) {
        CrossSelectTerm merged = left;
        merged.append(right.begin(), right.end());
        out.push_back(std::move(merged));
      }
    }
    return success();
  }
  case slang::ast::BinsSelectExprKind::WithFilter:
    return mlir::emitError(loc)
           << "unsupported cross select expression with nested 'with' clause";
  case slang::ast::BinsSelectExprKind::SetExpr:
    return mlir::emitError(loc)
           << "unsupported cross select expression with nested cross set "
              "expression";
  case slang::ast::BinsSelectExprKind::Invalid:
    return mlir::emitError(loc) << "invalid cross select expression";
  }
  llvm_unreachable("unknown BinsSelectExpr kind");
}

static LogicalResult emitTupleBinsOf(
    ArrayRef<int64_t> tupleValues,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder, Location loc, int32_t group) {
  if (tupleValues.size() != crossTargets.size())
    return mlir::emitError(loc)
           << "internal error: tuple arity mismatch in cross select lowering";

  IntegerAttr groupAttr =
      group > 0 ? builder.getI32IntegerAttr(group) : IntegerAttr();
  for (size_t i = 0; i < tupleValues.size(); ++i) {
    auto targetRef =
        resolveCrossSelectTargetRef(*crossTargets[i], coverpointSymbols, builder);
    auto valuesAttr =
        builder.getArrayAttr({builder.getI64IntegerAttr(tupleValues[i])});
    moore::BinsOfOp::create(builder, loc, targetRef, valuesAttr,
                            mlir::DenseI64ArrayAttr(), mlir::UnitAttr(),
                            groupAttr);
  }
  return success();
}

static LogicalResult emitAlwaysFalseCrossSelect(
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder, Location loc) {
  if (crossTargets.empty())
    return mlir::emitError(loc)
           << "unsupported cross select expression with no cross targets";
  auto targetRef =
      resolveCrossSelectTargetRef(*crossTargets.front(), coverpointSymbols, builder);
  moore::BinsOfOp::create(builder, loc, targetRef, mlir::ArrayAttr(),
                          mlir::DenseI64ArrayAttr(), builder.getUnitAttr(),
                          IntegerAttr());
  return success();
}

static LogicalResult extractTupleValuesFromConstant(
    const slang::ConstantValue &tupleValue, size_t arity, Location loc,
    SmallVectorImpl<int64_t> &tupleOut) {
  tupleOut.clear();
  if (arity == 1) {
    auto single = getConstantInt64(tupleValue);
    if (single) {
      tupleOut.push_back(*single);
      return success();
    }
  }

  if (!tupleValue.isUnpacked())
    return mlir::emitError(loc)
           << "unsupported cross set expression element; expected tuple value";

  auto elements = tupleValue.elements();
  if (elements.size() != arity)
    return mlir::emitError(loc)
           << "unsupported cross set expression element with tuple arity "
           << elements.size() << "; expected " << arity;

  for (auto &element : elements) {
    auto value = getConstantInt64(element);
    if (!value)
      return mlir::emitError(loc)
             << "unsupported non-integer tuple value in cross set expression";
    tupleOut.push_back(*value);
  }
  return success();
}

static LogicalResult emitSetExprBinsSelect(
    const slang::ast::SetExprBinsSelectExpr &setExpr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    slang::ast::Compilation &compilation, OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  auto matchesPolicy =
      parseCrossMatchesPolicy(setExpr.matchesExpr, loc, evaluateConstant);
  if (failed(matchesPolicy))
    return failure();

  auto setValue = evaluateCrossSetExpr(setExpr.expr, compilation, evaluateConstant);
  if (setValue.bad())
    return mlir::emitError(loc)
           << "unsupported non-constant cross set expression";

  SmallVector<SmallVector<int64_t, 4>, 16> tuples;
  SmallVector<int64_t, 4> tupleValues;
  auto appendTuple = [&](const slang::ConstantValue &tuple) -> LogicalResult {
    if (failed(extractTupleValuesFromConstant(tuple, crossTargets.size(), loc,
                                              tupleValues)))
      return failure();
    tuples.emplace_back(tupleValues.begin(), tupleValues.end());
    return success();
  };

  if (setValue.isQueue()) {
    for (const auto &element : *setValue.queue())
      if (failed(appendTuple(element)))
        return failure();
  } else if (setValue.isUnpacked()) {
    for (const auto &element : setValue.elements())
      if (failed(appendTuple(element)))
        return failure();
  } else {
    return mlir::emitError(loc)
           << "unsupported cross set expression; expected queue or unpacked "
              "tuple list";
  }

  if (tuples.empty())
    return emitAlwaysFalseCrossSelect(crossTargets, coverpointSymbols, builder,
                                      loc);

  SmallVector<SmallVector<int64_t, 4>, 16> selectedTuples;
  selectedTuples.reserve(tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    auto candidate = ArrayRef<int64_t>(tuples[i]);
    bool seen = false;
    for (size_t j = 0; j < i; ++j) {
      if (ArrayRef<int64_t>(tuples[j]) == candidate) {
        seen = true;
        break;
      }
    }
    if (seen)
      continue;

    uint64_t matchCount = 0;
    for (auto &tuple : tuples)
      if (ArrayRef<int64_t>(tuple) == candidate)
        ++matchCount;

    // Cross set expressions are represented as explicit value tuples here; for
    // that finite representation, 'matches $' is equivalent to requiring at
    // least one tuple occurrence.
    bool selected = matchesPolicy->requireAll ? (matchCount >= 1)
                                              : (matchCount >=
                                                 matchesPolicy->minMatches);
    if (selected)
      selectedTuples.emplace_back(candidate.begin(), candidate.end());
  }

  if (selectedTuples.empty())
    return emitAlwaysFalseCrossSelect(crossTargets, coverpointSymbols, builder,
                                      loc);

  for (size_t group = 0; group < selectedTuples.size(); ++group) {
    if (failed(emitTupleBinsOf(selectedTuples[group], crossTargets,
                               coverpointSymbols,
                               builder, loc, static_cast<int32_t>(group))))
      return failure();
  }
  return success();
}

static LogicalResult evaluateIntersectList(
    std::span<const slang::ast::Expression *const> intersects, int64_t value,
    Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool &result) {
  if (intersects.empty()) {
    result = true;
    return success();
  }

  result = false;
  for (const auto *intersectExpr : intersects) {
    if (intersectExpr->kind == slang::ast::ExpressionKind::ValueRange) {
      auto &rangeExpr = intersectExpr->as<slang::ast::ValueRangeExpression>();
      if (rangeExpr.rangeKind != slang::ast::ValueRangeKind::Simple) {
        int64_t lower = 0;
        int64_t upper = 0;
        if (failed(
                evaluateIntersectToleranceRangeBounds(rangeExpr, loc, lower, upper)))
          return failure();
        if (value >= lower && value <= upper) {
          result = true;
          return success();
        }
        continue;
      }

      auto leftVal = getConstantInt64(evaluateConstant(rangeExpr.left()));
      auto rightVal = getConstantInt64(evaluateConstant(rangeExpr.right()));
      bool leftUnbounded =
          !leftVal && isUnboundedConstantExpr(rangeExpr.left(), evaluateConstant);
      bool rightUnbounded =
          !rightVal && isUnboundedConstantExpr(rangeExpr.right(), evaluateConstant);
      if ((!leftVal && !leftUnbounded) || (!rightVal && !rightUnbounded))
        return mlir::emitError(loc)
               << "unsupported non-constant intersect value range in cross "
                  "select expression";

      if (leftUnbounded && rightUnbounded) {
        result = true;
        return success();
      }
      if (leftUnbounded) {
        if (value <= *rightVal) {
          result = true;
          return success();
        }
        continue;
      }
      if (rightUnbounded) {
        if (value >= *leftVal) {
          result = true;
          return success();
        }
        continue;
      }

      int64_t lower = std::min(*leftVal, *rightVal);
      int64_t upper = std::max(*leftVal, *rightVal);
      if (value >= lower && value <= upper) {
        result = true;
        return success();
      }
      continue;
    }

    auto intVal = getConstantInt64(evaluateConstant(*intersectExpr));
    if (!intVal)
      return mlir::emitError(loc)
             << "unsupported non-constant intersect value in cross select "
                "expression";
    if (value == *intVal) {
      result = true;
      return success();
    }
  }
  return success();
}

static void collectWithFilterIterators(
    const slang::ast::Expression &filterExpr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    llvm::SmallDenseMap<size_t, const slang::ast::IteratorSymbol *, 4>
        &iterators) {
  iterators.clear();
  llvm::StringMap<size_t> targetByName;
  for (size_t i = 0; i < crossTargets.size(); ++i)
    targetByName.try_emplace(crossTargets[i]->name, i);

  filterExpr.visitSymbolReferences([&](const slang::ast::Expression &,
                                       const slang::ast::Symbol &symbol) {
    if (symbol.kind != slang::ast::SymbolKind::Iterator)
      return;
    auto it = targetByName.find(symbol.name);
    if (it == targetByName.end())
      return;
    iterators.try_emplace(it->second, &symbol.as<slang::ast::IteratorSymbol>());
  });
}

static LogicalResult evaluateWithFilterOnTuple(
    const slang::ast::BinSelectWithFilterExpr &withExpr, ArrayRef<int64_t> tuple,
    const llvm::SmallDenseMap<size_t, const slang::ast::IteratorSymbol *, 4>
        &iterators,
    slang::ast::Compilation &compilation, Location loc, bool &result) {
  using namespace slang::ast;
  EvalContext evalContext(
      ASTContext(compilation.getRoot(), LookupLocation::max),
      EvalFlags::CacheResults | EvalFlags::SpecparamsAllowed |
          EvalFlags::IsScript);

  for (auto [index, iter] : iterators) {
    if (index >= tuple.size())
      return mlir::emitError(loc)
             << "internal error: iterator index out of range in cross select "
                "'with' clause";
    auto width = iter->getType().getBitWidth();
    if (width == 0 || width > 64)
      return mlir::emitError(loc)
             << "unsupported cross select iterator width in 'with' clause";
    auto intValue = slang::SVInt(width, static_cast<uint64_t>(tuple[index]),
                                 iter->getType().isSigned());
    evalContext.createLocal(iter, slang::ConstantValue(std::move(intValue)));
  }

  auto evalResult = withExpr.filter.eval(evalContext);
  if (evalResult.bad())
    return mlir::emitError(loc)
           << "failed to evaluate cross select 'with' clause";
  result = evalResult.isTrue();
  return success();
}

struct CrossSelectBinDomainEntry {
  const slang::ast::CoverageBinSymbol *symbol = nullptr;
  SmallVector<int64_t, 16> values;
};

using CrossSelectBinDomains = SmallVector<SmallVector<CrossSelectBinDomainEntry, 16>, 4>;

struct CrossSelectBinRequirements {
  llvm::SmallDenseSet<const slang::ast::CoverageBinSymbol *, 16> requiredBins;
  llvm::SmallDenseSet<const slang::ast::CoverpointSymbol *, 4> requireAllBins;
  llvm::SmallDenseSet<const slang::ast::CoverageBinSymbol *, 16> valueBins;
  llvm::SmallDenseSet<const slang::ast::CoverpointSymbol *, 4>
      valueCoverpoints;
};

static void collectCrossSelectBinRequirements(
    const slang::ast::BinsSelectExpr &expr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    CrossSelectBinRequirements &requirements) {
  switch (expr.kind) {
  case slang::ast::BinsSelectExprKind::Condition: {
    auto &condExpr =
        static_cast<const slang::ast::ConditionBinsSelectExpr &>(expr);
    if (auto *bin = condExpr.target.as_if<slang::ast::CoverageBinSymbol>()) {
      requirements.requiredBins.insert(bin);
      if (!condExpr.intersects.empty())
        requirements.valueBins.insert(bin);
      return;
    }
    if (auto *cp = condExpr.target.as_if<slang::ast::CoverpointSymbol>()) {
      requirements.requireAllBins.insert(cp);
      if (!condExpr.intersects.empty())
        requirements.valueCoverpoints.insert(cp);
    }
    return;
  }
  case slang::ast::BinsSelectExprKind::CrossId:
    for (auto *target : crossTargets)
      requirements.requireAllBins.insert(target);
    return;
  case slang::ast::BinsSelectExprKind::Unary: {
    auto &unaryExpr =
        static_cast<const slang::ast::UnaryBinsSelectExpr &>(expr);
    collectCrossSelectBinRequirements(unaryExpr.expr, crossTargets, requirements);
    return;
  }
  case slang::ast::BinsSelectExprKind::Binary: {
    auto &binaryExpr =
        static_cast<const slang::ast::BinaryBinsSelectExpr &>(expr);
    collectCrossSelectBinRequirements(binaryExpr.left, crossTargets, requirements);
    collectCrossSelectBinRequirements(binaryExpr.right, crossTargets, requirements);
    return;
  }
  case slang::ast::BinsSelectExprKind::WithFilter: {
    auto &withExpr =
        static_cast<const slang::ast::BinSelectWithFilterExpr &>(expr);
    collectCrossSelectBinRequirements(withExpr.expr, crossTargets, requirements);
    llvm::SmallDenseMap<size_t, const slang::ast::IteratorSymbol *, 4> iterators;
    collectWithFilterIterators(withExpr.filter, crossTargets, iterators);
    for (auto [index, _] : iterators) {
      if (index < crossTargets.size())
        requirements.valueCoverpoints.insert(crossTargets[index]);
    }
    return;
  }
  case slang::ast::BinsSelectExprKind::SetExpr:
    for (auto *target : crossTargets) {
      requirements.requireAllBins.insert(target);
      requirements.valueCoverpoints.insert(target);
    }
    return;
  case slang::ast::BinsSelectExprKind::Invalid:
    return;
  }
}

static LogicalResult buildFiniteIntegralCoverpointDomain(
    const slang::ast::CoverpointSymbol &target, Location loc,
    unsigned maxDomainBits, SmallVectorImpl<int64_t> &values);

constexpr uint64_t kMaxFiniteCoverageBinValues = 4096;

static LogicalResult appendFiniteCoverageBinValue(int64_t value, Location loc,
                                                  SmallVectorImpl<int64_t> &values) {
  if (values.size() == kMaxFiniteCoverageBinValues)
    return mlir::emitError(loc)
           << "unsupported cross select expression due to large finite "
              "coverpoint bin domain";
  values.push_back(value);
  return success();
}

static LogicalResult appendFiniteCoverageBinRange(int64_t lower, int64_t upper,
                                                  Location loc,
                                                  SmallVectorImpl<int64_t> &values) {
  uint64_t rangeCount = static_cast<uint64_t>(upper - lower) + 1;
  if (values.size() > kMaxFiniteCoverageBinValues - rangeCount)
    return mlir::emitError(loc)
           << "unsupported cross select expression due to large finite "
              "coverpoint bin domain";
  for (int64_t v = lower; v <= upper; ++v)
    values.push_back(v);
  return success();
}

static LogicalResult finalizeFiniteCoverageBinValues(
    Location loc, SmallVectorImpl<int64_t> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
  if (values.empty())
    return mlir::emitError(loc)
           << "unsupported empty coverpoint bin in cross select expression";
  return success();
}

static LogicalResult collectCoverageBinExplicitValues(
    std::span<const slang::ast::Expression *const> valueExprs, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    SmallVectorImpl<int64_t> &values) {
  values.clear();

  for (const auto *valueExpr : valueExprs) {
    if (valueExpr->kind == slang::ast::ExpressionKind::ValueRange) {
      auto &rangeExpr = valueExpr->as<slang::ast::ValueRangeExpression>();
      auto leftVal = getConstantInt64(evaluateConstant(rangeExpr.left()));
      auto rightVal = getConstantInt64(evaluateConstant(rangeExpr.right()));
      if (!leftVal || !rightVal)
        return mlir::emitError(loc)
               << "unsupported non-constant coverpoint bin range in cross "
                  "select expression";
      int64_t lower = std::min(*leftVal, *rightVal);
      int64_t upper = std::max(*leftVal, *rightVal);
      if (failed(appendFiniteCoverageBinRange(lower, upper, loc, values)))
        return failure();
      continue;
    }

    auto intVal = getConstantInt64(evaluateConstant(*valueExpr));
    if (!intVal)
      return mlir::emitError(loc)
             << "unsupported non-constant coverpoint bin value in cross "
                "select expression";
    if (failed(appendFiniteCoverageBinValue(*intVal, loc, values)))
      return failure();
  }

  return finalizeFiniteCoverageBinValues(loc, values);
}

static LogicalResult filterCoverageBinWithExprValues(
    const slang::ast::Expression &withExpr, ArrayRef<int64_t> baseValues,
    slang::ast::Compilation &compilation, Location loc,
    SmallVectorImpl<int64_t> &values) {
  using namespace slang::ast;

  llvm::SmallDenseSet<const IteratorSymbol *, 2> iteratorSet;
  withExpr.visitSymbolReferences([&](const Expression &, const Symbol &symbol) {
    if (symbol.kind != SymbolKind::Iterator)
      return;
    iteratorSet.insert(&symbol.as<IteratorSymbol>());
  });

  values.clear();
  values.reserve(baseValues.size());
  for (int64_t value : baseValues) {
    EvalContext evalContext(
        ASTContext(compilation.getRoot(), LookupLocation::max),
        EvalFlags::CacheResults | EvalFlags::SpecparamsAllowed |
            EvalFlags::IsScript);
    for (auto *iter : iteratorSet) {
      auto width = iter->getType().getBitWidth();
      if (width == 0 || width > 64)
        return mlir::emitError(loc)
               << "unsupported coverpoint bin iterator width in cross select "
                  "expression";
      auto intValue =
          slang::SVInt(width, static_cast<uint64_t>(value),
                       iter->getType().isSigned());
      evalContext.createLocal(iter, slang::ConstantValue(std::move(intValue)));
    }

    auto evalResult = withExpr.eval(evalContext);
    if (evalResult.bad())
      return mlir::emitError(loc)
             << "failed to evaluate coverpoint bin 'with' clause in cross "
                "select expression";
    if (evalResult.isTrue())
      values.push_back(value);
  }

  return finalizeFiniteCoverageBinValues(loc, values);
}

static LogicalResult collectCoverageSetExprValues(
    const slang::ast::Expression &setCoverageExpr,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    SmallVectorImpl<int64_t> &values) {
  values.clear();

  auto setValue =
      evaluateCrossSetExpr(setCoverageExpr, compilation, evaluateConstant);
  if (setValue.bad()) {
    auto *targetSym = setCoverageExpr.getSymbolReference();
    auto *targetValue =
        targetSym ? targetSym->as_if<slang::ast::ValueSymbol>() : nullptr;
    if (targetValue && targetValue->getInitializer())
      setValue = evaluateCrossSetExpr(*targetValue->getInitializer(), compilation,
                                      evaluateConstant);
  }
  if (setValue.bad())
    return mlir::emitError(loc)
           << "unsupported non-constant set coverpoint bin in cross select "
              "expression";

  auto appendValue = [&](const slang::ConstantValue &element) -> LogicalResult {
    auto intValue = getConstantInt64(element);
    if (!intValue)
      return mlir::emitError(loc)
             << "unsupported non-integer set coverpoint bin value in cross "
                "select expression";
    return appendFiniteCoverageBinValue(*intValue, loc, values);
  };

  if (setValue.isQueue()) {
    for (const auto &element : *setValue.queue())
      if (failed(appendValue(element)))
        return failure();
  } else if (setValue.isUnpacked()) {
    for (const auto &element : setValue.elements())
      if (failed(appendValue(element)))
        return failure();
  } else {
    if (failed(appendValue(setValue)))
      return failure();
  }

  return finalizeFiniteCoverageBinValues(loc, values);
}

static LogicalResult collectCoverageTransitionBinValues(
    std::span<const slang::ast::CoverageBinSymbol::TransSet> transList,
    Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    SmallVectorImpl<int64_t> &values) {
  values.clear();

  for (const auto &transSet : transList) {
    for (const auto &rangeList : transSet) {
      for (const auto *itemExpr : rangeList.items) {
        if (itemExpr->kind == slang::ast::ExpressionKind::ValueRange) {
          auto &rangeExpr = itemExpr->as<slang::ast::ValueRangeExpression>();
          auto leftVal = getConstantInt64(evaluateConstant(rangeExpr.left()));
          auto rightVal = getConstantInt64(evaluateConstant(rangeExpr.right()));
          if (!leftVal || !rightVal)
            return mlir::emitError(loc)
                   << "unsupported non-constant transition bin range in cross "
                      "select expression";
          int64_t lower = std::min(*leftVal, *rightVal);
          int64_t upper = std::max(*leftVal, *rightVal);
          if (failed(appendFiniteCoverageBinRange(lower, upper, loc, values)))
            return failure();
          continue;
        }

        auto intValue = getConstantInt64(evaluateConstant(*itemExpr));
        if (!intValue)
          return mlir::emitError(loc)
                 << "unsupported non-constant transition bin value in cross "
                    "select expression";
        if (failed(appendFiniteCoverageBinValue(*intValue, loc, values)))
          return failure();
      }
    }
  }

  return finalizeFiniteCoverageBinValues(loc, values);
}

static LogicalResult collectCoverageBinValues(
    const slang::ast::CoverageBinSymbol &bin,
    const slang::ast::CoverpointSymbol &target,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    unsigned maxDomainBits,
    SmallVectorImpl<int64_t> &values) {
  values.clear();

  if (!bin.getTransList().empty())
    return collectCoverageTransitionBinValues(bin.getTransList(), loc,
                                              evaluateConstant, values);
  if (bin.isDefault || bin.isDefaultSequence)
    return mlir::emitError(loc)
           << "unsupported default coverpoint bin target in cross select "
              "expression";

  if (auto *withExpr = bin.getWithExpr()) {
    SmallVector<int64_t, 256> baseValues;
    if (!bin.getValues().empty()) {
      if (failed(collectCoverageBinExplicitValues(bin.getValues(), loc,
                                                  evaluateConstant, baseValues)))
        return failure();
    } else {
      if (failed(buildFiniteIntegralCoverpointDomain(target, loc, maxDomainBits,
                                                     baseValues)))
        return failure();
    }

    return filterCoverageBinWithExprValues(*withExpr, baseValues, compilation,
                                           loc, values);
  }

  if (auto *setCoverageExpr = bin.getSetCoverageExpr())
    return collectCoverageSetExprValues(*setCoverageExpr, compilation, loc,
                                        evaluateConstant, values);

  return collectCoverageBinExplicitValues(bin.getValues(), loc, evaluateConstant,
                                          values);
}

static LogicalResult buildFiniteIntegralCoverpointDomain(
    const slang::ast::CoverpointSymbol &target, Location loc,
    unsigned maxDomainBits, SmallVectorImpl<int64_t> &values) {
  values.clear();
  const auto &type = target.getType();
  if (!type.isIntegral())
    return mlir::emitError(loc)
           << "unsupported cross select 'with' clause over non-integral "
              "coverpoint '"
           << target.name << "'";

  auto width = type.getBitWidth();
  if (width == 0 || width > maxDomainBits)
    return mlir::emitError(loc)
           << "unsupported cross select 'with' clause over coverpoint '"
           << target.name << "' with width " << width
           << "; maximum supported width is " << maxDomainBits;

  uint64_t count = 1ull << width;
  values.reserve(static_cast<size_t>(count));
  if (type.isSigned()) {
    int64_t start = -(int64_t(1) << (width - 1));
    for (uint64_t i = 0; i < count; ++i)
      values.push_back(start + static_cast<int64_t>(i));
  } else {
    for (uint64_t i = 0; i < count; ++i)
      values.push_back(static_cast<int64_t>(i));
  }
  return success();
}

static LogicalResult buildFiniteCrossBinDomains(
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const CrossSelectBinRequirements &requirements,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    CrossSelectBinDomains &domains) {
  // Keep finite value-domain expansion aligned with the 4096-value cap used for
  // explicit bin value extraction.
  constexpr unsigned kMaxDomainBits = 12;
  constexpr uint64_t kMaxBinTupleCount = 4096;

  domains.clear();
  domains.reserve(crossTargets.size());
  uint64_t binTupleCount = 1;

  for (auto *target : crossTargets) {
    SmallVector<CrossSelectBinDomainEntry, 16> domain;
    SmallVector<const slang::ast::CoverageBinSymbol *, 2> defaultBins;
    SmallVector<int64_t, 64> coveredValues;
    for (const auto &member : target->members()) {
      auto *bin = member.as_if<slang::ast::CoverageBinSymbol>();
      if (!bin)
        continue;
      bool required = requirements.requiredBins.contains(bin) ||
                      requirements.requireAllBins.contains(target);
      if (bin->getWithExpr() && !required)
        continue;
      if (bin->getSetCoverageExpr() && !required)
        continue;

      bool unsupportedBinShape = bin->isDefaultSequence;
      if (unsupportedBinShape) {
        if (required) {
          bool valuesRequired = requirements.valueBins.contains(bin) ||
                                requirements.valueCoverpoints.contains(target);
          if (valuesRequired) {
            SmallVector<int64_t, 256> fullDomain;
            if (failed(buildFiniteIntegralCoverpointDomain(
                    *target, loc, kMaxDomainBits, fullDomain)))
              return failure();
            CrossSelectBinDomainEntry entry;
            entry.symbol = bin;
            entry.values.append(fullDomain.begin(), fullDomain.end());
            domain.push_back(std::move(entry));
            continue;
          }
          domain.push_back({bin, {}});
          continue;
        }
        continue;
      }
      if (bin->isDefault) {
        defaultBins.push_back(bin);
        continue;
      }

      SmallVector<int64_t, 16> binValues;
      if (failed(collectCoverageBinValues(*bin, *target, compilation, loc,
                                          evaluateConstant, kMaxDomainBits,
                                          binValues)))
        return failure();
      coveredValues.append(binValues.begin(), binValues.end());
      domain.push_back({bin, std::move(binValues)});
    }

    if (!defaultBins.empty()) {
      SmallVector<int64_t, 256> fullDomain;
      if (failed(buildFiniteIntegralCoverpointDomain(*target, loc, kMaxDomainBits,
                                                     fullDomain)))
        return failure();

      llvm::sort(coveredValues);
      coveredValues.erase(std::unique(coveredValues.begin(), coveredValues.end()),
                          coveredValues.end());

      SmallVector<int64_t, 256> defaultValues;
      defaultValues.reserve(fullDomain.size());
      for (int64_t value : fullDomain)
        if (!llvm::is_contained(coveredValues, value))
          defaultValues.push_back(value);

      if (!defaultValues.empty()) {
        for (auto *defaultBin : defaultBins) {
          CrossSelectBinDomainEntry entry;
          entry.symbol = defaultBin;
          entry.values.append(defaultValues.begin(), defaultValues.end());
          domain.push_back(std::move(entry));
        }
      }
    }

    if (domain.empty()) {
      SmallVector<int64_t, 256> fullDomain;
      if (failed(buildFiniteIntegralCoverpointDomain(*target, loc, kMaxDomainBits,
                                                     fullDomain)))
        return failure();
      domain.reserve(fullDomain.size());
      for (int64_t value : fullDomain)
        domain.push_back({nullptr, {value}});
    }

    if (domain.empty())
      return mlir::emitError(loc)
             << "unsupported cross select expression over coverpoint '"
             << target->name << "' with no finite bins";

    if (binTupleCount > kMaxBinTupleCount / domain.size())
      return mlir::emitError(loc)
             << "unsupported cross select expression due to large finite "
                "cross space";
    binTupleCount *= domain.size();
    domains.push_back(std::move(domain));
  }

  return success();
}

static LogicalResult emitBinTupleBinsOf(
    ArrayRef<size_t> tupleIndices, const CrossSelectBinDomains &domains,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder, Location loc, int32_t group) {
  if (tupleIndices.size() != crossTargets.size() ||
      tupleIndices.size() != domains.size())
    return mlir::emitError(loc)
           << "internal error: tuple arity mismatch in cross select lowering";

  IntegerAttr groupAttr =
      group > 0 ? builder.getI32IntegerAttr(group) : IntegerAttr();

  for (size_t i = 0; i < tupleIndices.size(); ++i) {
    if (tupleIndices[i] >= domains[i].size())
      return mlir::emitError(loc)
             << "internal error: cross select tuple index out of range";

    auto &bin = domains[i][tupleIndices[i]];
    mlir::SymbolRefAttr targetRef;
    mlir::ArrayAttr intersectValuesAttr;
    if (bin.symbol) {
      targetRef =
          resolveCrossSelectTargetRef(*bin.symbol, coverpointSymbols, builder);
    } else {
      targetRef =
          resolveCrossSelectTargetRef(*crossTargets[i], coverpointSymbols, builder);
      if (bin.values.size() != 1)
        return mlir::emitError(loc)
               << "internal error: expected singleton auto-bin domain";
      intersectValuesAttr =
          builder.getArrayAttr({builder.getI64IntegerAttr(bin.values.front())});
    }

    moore::BinsOfOp::create(builder, loc, targetRef, intersectValuesAttr,
                            mlir::DenseI64ArrayAttr(), mlir::UnitAttr(),
                            groupAttr);
  }
  return success();
}

static LogicalResult evaluateConditionOnBinTuple(
    const slang::ast::ConditionBinsSelectExpr &condExpr, ArrayRef<size_t> tuple,
    const CrossSelectBinDomains &domains,
    const llvm::SmallDenseMap<const slang::ast::CoverpointSymbol *, size_t, 4>
        &coverpointIndex,
    Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool &result) {
  const auto *coverpoint = condExpr.target.as_if<slang::ast::CoverpointSymbol>();
  const auto *coverageBin = condExpr.target.as_if<slang::ast::CoverageBinSymbol>();
  if (!coverpoint && !coverageBin)
    return mlir::emitError(loc)
           << "unsupported cross select condition target kind";

  if (coverageBin) {
    auto *parentScope = coverageBin->getParentScope();
    if (!parentScope)
      return mlir::emitError(loc)
             << "unsupported detached coverage bin in cross select expression";
    coverpoint = parentScope->asSymbol().as_if<slang::ast::CoverpointSymbol>();
    if (!coverpoint)
      return mlir::emitError(loc)
             << "unsupported coverage bin parent in cross select expression";
  }

  auto it = coverpointIndex.find(coverpoint);
  if (it == coverpointIndex.end())
    return mlir::emitError(loc)
           << "cross select expression references coverpoint outside enclosing "
              "cross";
  size_t cpIndex = it->second;
  if (cpIndex >= tuple.size() || cpIndex >= domains.size())
    return mlir::emitError(loc)
           << "internal error: coverpoint index out of range in cross select "
              "expression";
  if (tuple[cpIndex] >= domains[cpIndex].size())
    return mlir::emitError(loc)
           << "internal error: bin tuple index out of range in cross select "
              "expression";

  auto &candidateBin = domains[cpIndex][tuple[cpIndex]];
  bool baseMatch = true;
  if (coverageBin)
    baseMatch = candidateBin.symbol == coverageBin;

  bool intersectsMatch = true;
  if (!condExpr.intersects.empty()) {
    intersectsMatch = false;
    for (auto value : candidateBin.values) {
      bool valueMatch = false;
      if (failed(evaluateIntersectList(condExpr.intersects, value, loc,
                                       evaluateConstant, valueMatch)))
        return failure();
      if (valueMatch) {
        intersectsMatch = true;
        break;
      }
    }
  }

  result = baseMatch && intersectsMatch;
  return success();
}

static LogicalResult evaluateCrossSelectExprOnBinTuple(
    const slang::ast::BinsSelectExpr &expr, ArrayRef<size_t> tuple,
    const CrossSelectBinDomains &domains,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::SmallDenseMap<const slang::ast::CoverpointSymbol *, size_t, 4>
        &coverpointIndex,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool &result);

static LogicalResult countWithFilterMatchesOnBinTuple(
    const slang::ast::BinSelectWithFilterExpr &withExpr, ArrayRef<size_t> tuple,
    const CrossSelectBinDomains &domains,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    slang::ast::Compilation &compilation, Location loc, uint64_t &matchCount,
    uint64_t &totalCount) {
  constexpr uint64_t kMaxFilterTupleChecks = 1'000'000;
  matchCount = 0;
  totalCount = 0;

  llvm::SmallDenseMap<size_t, const slang::ast::IteratorSymbol *, 4> iterators;
  collectWithFilterIterators(withExpr.filter, crossTargets, iterators);

  if (tuple.empty()) {
    bool filterMatches = false;
    if (failed(evaluateWithFilterOnTuple(withExpr, {}, iterators, compilation, loc,
                                         filterMatches)))
      return failure();
    totalCount = 1;
    matchCount = filterMatches ? 1 : 0;
    return success();
  }

  SmallVector<size_t, 4> activeDims;
  activeDims.reserve(iterators.size());
  for (auto [index, _] : iterators)
    activeDims.push_back(index);
  llvm::sort(activeDims);

  if (activeDims.empty()) {
    SmallVector<int64_t, 1> dummyTuple(tuple.size(), 0);
    bool filterMatches = false;
    if (failed(evaluateWithFilterOnTuple(withExpr, dummyTuple, iterators,
                                         compilation, loc, filterMatches)))
      return failure();
    totalCount = 1;
    matchCount = filterMatches ? 1 : 0;
    return success();
  }

  SmallVector<size_t, 4> valueIndices(activeDims.size(), 0);
  SmallVector<int64_t, 4> valueTuple(tuple.size(), 0);
  bool done = false;
  while (!done) {
    for (auto [iterPos, dim] : llvm::enumerate(activeDims)) {
      if (tuple[dim] >= domains[dim].size())
        return mlir::emitError(loc)
               << "internal error: bin tuple index out of range in cross "
                  "select expression";
      auto &values = domains[dim][tuple[dim]].values;
      if (values.empty())
        return mlir::emitError(loc)
               << "unsupported empty coverpoint bin in cross select expression";
      if (valueIndices[iterPos] >= values.size())
        return mlir::emitError(loc)
               << "internal error: value index out of range in cross select "
                  "expression";
      valueTuple[dim] = values[valueIndices[iterPos]];
    }

    bool filterMatches = false;
    if (failed(evaluateWithFilterOnTuple(withExpr, valueTuple, iterators,
                                         compilation, loc, filterMatches)))
      return failure();
    ++totalCount;
    if (filterMatches)
      ++matchCount;
    if (totalCount > kMaxFilterTupleChecks)
      return mlir::emitError(loc)
             << "unsupported cross select 'with' clause due to too many "
                "value tuples in candidate bin tuple";

    for (int64_t iterPos = static_cast<int64_t>(activeDims.size()) - 1;
         iterPos >= 0; --iterPos) {
      size_t dim = activeDims[iterPos];
      auto &values = domains[dim][tuple[dim]].values;
      valueIndices[iterPos]++;
      if (valueIndices[iterPos] < values.size())
        break;
      valueIndices[iterPos] = 0;
      if (dim == 0)
        done = true;
      if (iterPos == 0)
        done = true;
    }
  }

  return success();
}

static LogicalResult evaluateSetExprOnBinTuple(
    const slang::ast::SetExprBinsSelectExpr &setExpr, ArrayRef<size_t> tuple,
    const CrossSelectBinDomains &domains,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool &result) {
  auto matchesPolicy =
      parseCrossMatchesPolicy(setExpr.matchesExpr, loc, evaluateConstant);
  if (failed(matchesPolicy))
    return failure();

  auto setValue = evaluateCrossSetExpr(setExpr.expr, compilation, evaluateConstant);
  if (setValue.bad())
    return mlir::emitError(loc)
           << "unsupported non-constant cross set expression";

  uint64_t matchCount = 0;
  uint64_t totalCount = 0;
  SmallVector<int64_t, 4> tupleValues;
  auto checkTuple = [&](const slang::ConstantValue &element) -> LogicalResult {
    if (failed(extractTupleValuesFromConstant(element, crossTargets.size(), loc,
                                              tupleValues)))
      return failure();
    ++totalCount;

    bool inCandidate = true;
    for (size_t i = 0; i < tupleValues.size(); ++i) {
      if (tuple[i] >= domains[i].size())
        return mlir::emitError(loc)
               << "internal error: bin tuple index out of range in cross set "
                  "expression";
      auto &candidateBin = domains[i][tuple[i]];
      if (!llvm::is_contained(candidateBin.values, tupleValues[i])) {
        inCandidate = false;
        break;
      }
    }
    if (inCandidate)
      ++matchCount;
    return success();
  };

  if (setValue.isQueue()) {
    for (const auto &element : *setValue.queue())
      if (failed(checkTuple(element)))
        return failure();
  } else if (setValue.isUnpacked()) {
    for (const auto &element : setValue.elements())
      if (failed(checkTuple(element)))
        return failure();
  } else {
    return mlir::emitError(loc)
           << "unsupported cross set expression; expected queue or unpacked "
              "tuple list";
  }

  if (totalCount == 0) {
    result = false;
    return success();
  }
  if (matchesPolicy->requireAll) {
    result = matchCount == totalCount;
    return success();
  }
  result = matchCount >= matchesPolicy->minMatches;
  return success();
}

static LogicalResult emitWithFilterBinsSelect(
    const slang::ast::BinSelectWithFilterExpr &withExpr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    slang::ast::Compilation &compilation, OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  llvm::SmallDenseMap<const slang::ast::CoverpointSymbol *, size_t, 4>
      coverpointIndex;
  for (size_t i = 0; i < crossTargets.size(); ++i)
    coverpointIndex.try_emplace(crossTargets[i], i);

  CrossSelectBinDomains domains;
  CrossSelectBinRequirements requirements;
  collectCrossSelectBinRequirements(withExpr, crossTargets, requirements);
  if (failed(buildFiniteCrossBinDomains(crossTargets, requirements, compilation,
                                        loc,
                                        evaluateConstant, domains)))
    return failure();

  constexpr size_t kMaxGeneratedTuples = 4096;
  SmallVector<SmallVector<size_t, 4>, 64> selectedTuples;

  if (domains.empty()) {
    bool selected = false;
    if (failed(evaluateCrossSelectExprOnBinTuple(
            withExpr, {}, domains, crossTargets, coverpointIndex, compilation,
            loc, evaluateConstant, selected)))
      return failure();
    if (selected)
      selectedTuples.emplace_back();
  } else {
    SmallVector<size_t, 4> indices(domains.size(), 0);
    bool done = false;
    while (!done) {
      bool selected = false;
      if (failed(evaluateCrossSelectExprOnBinTuple(
              withExpr, indices, domains, crossTargets, coverpointIndex,
              compilation, loc, evaluateConstant, selected)))
        return failure();
      if (selected) {
        selectedTuples.emplace_back(indices.begin(), indices.end());
        if (selectedTuples.size() > kMaxGeneratedTuples)
          return mlir::emitError(loc)
                 << "unsupported cross select 'with' clause due to too many "
                    "selected tuples";
      }

      for (int64_t dim = static_cast<int64_t>(indices.size()) - 1; dim >= 0;
           --dim) {
        indices[dim]++;
        if (indices[dim] < domains[dim].size())
          break;
        indices[dim] = 0;
        if (dim == 0)
          done = true;
      }
    }
  }

  if (selectedTuples.empty())
    return emitAlwaysFalseCrossSelect(crossTargets, coverpointSymbols, builder,
                                      loc);

  for (size_t group = 0; group < selectedTuples.size(); ++group) {
    if (failed(emitBinTupleBinsOf(selectedTuples[group], domains, crossTargets,
                                  coverpointSymbols, builder, loc,
                                  static_cast<int32_t>(group))))
      return failure();
  }
  return success();
}

static bool containsWithOrSetExpr(const slang::ast::BinsSelectExpr &expr) {
  switch (expr.kind) {
  case slang::ast::BinsSelectExprKind::WithFilter:
  case slang::ast::BinsSelectExprKind::SetExpr:
    return true;
  case slang::ast::BinsSelectExprKind::Unary: {
    auto &unaryExpr =
        static_cast<const slang::ast::UnaryBinsSelectExpr &>(expr);
    return containsWithOrSetExpr(unaryExpr.expr);
  }
  case slang::ast::BinsSelectExprKind::Binary: {
    auto &binaryExpr =
        static_cast<const slang::ast::BinaryBinsSelectExpr &>(expr);
    return containsWithOrSetExpr(binaryExpr.left) ||
           containsWithOrSetExpr(binaryExpr.right);
  }
  default:
    return false;
  }
}

static LogicalResult evaluateCrossSelectExprOnBinTuple(
    const slang::ast::BinsSelectExpr &expr, ArrayRef<size_t> tuple,
    const CrossSelectBinDomains &domains,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::SmallDenseMap<const slang::ast::CoverpointSymbol *, size_t, 4>
        &coverpointIndex,
    slang::ast::Compilation &compilation, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool &result) {
  switch (expr.kind) {
  case slang::ast::BinsSelectExprKind::Condition: {
    auto &condExpr =
        static_cast<const slang::ast::ConditionBinsSelectExpr &>(expr);
    return evaluateConditionOnBinTuple(condExpr, tuple, domains, coverpointIndex,
                                       loc, evaluateConstant, result);
  }
  case slang::ast::BinsSelectExprKind::CrossId:
    result = true;
    return success();
  case slang::ast::BinsSelectExprKind::Unary: {
    auto &unaryExpr =
        static_cast<const slang::ast::UnaryBinsSelectExpr &>(expr);
    bool child = false;
    if (failed(evaluateCrossSelectExprOnBinTuple(
            unaryExpr.expr, tuple, domains, crossTargets, coverpointIndex,
            compilation, loc, evaluateConstant, child)))
      return failure();
    result = !child;
    return success();
  }
  case slang::ast::BinsSelectExprKind::Binary: {
    auto &binaryExpr =
        static_cast<const slang::ast::BinaryBinsSelectExpr &>(expr);
    bool lhs = false;
    bool rhs = false;
    if (failed(evaluateCrossSelectExprOnBinTuple(
            binaryExpr.left, tuple, domains, crossTargets, coverpointIndex,
            compilation, loc, evaluateConstant, lhs)))
      return failure();
    if (failed(evaluateCrossSelectExprOnBinTuple(
            binaryExpr.right, tuple, domains, crossTargets, coverpointIndex,
            compilation, loc, evaluateConstant, rhs)))
      return failure();
    result = binaryExpr.op == slang::ast::BinaryBinsSelectExpr::Or ? (lhs || rhs)
                                                                    : (lhs && rhs);
    return success();
  }
  case slang::ast::BinsSelectExprKind::SetExpr: {
    auto &setExpr = static_cast<const slang::ast::SetExprBinsSelectExpr &>(expr);
    return evaluateSetExprOnBinTuple(setExpr, tuple, domains, crossTargets,
                                     compilation, loc, evaluateConstant, result);
  }
  case slang::ast::BinsSelectExprKind::WithFilter: {
    auto &withExpr =
        static_cast<const slang::ast::BinSelectWithFilterExpr &>(expr);
    auto matchesPolicy =
        parseCrossMatchesPolicy(withExpr.matchesExpr, loc, evaluateConstant);
    if (failed(matchesPolicy))
      return failure();

    bool baseMatches = false;
    if (failed(evaluateCrossSelectExprOnBinTuple(
            withExpr.expr, tuple, domains, crossTargets, coverpointIndex,
            compilation, loc, evaluateConstant, baseMatches)))
      return failure();
    if (!baseMatches) {
      result = false;
      return success();
    }

    uint64_t matchCount = 0;
    uint64_t totalCount = 0;
    if (failed(countWithFilterMatchesOnBinTuple(
            withExpr, tuple, domains, crossTargets, compilation, loc, matchCount,
            totalCount)))
      return failure();

    if (matchesPolicy->requireAll) {
      result = totalCount > 0 && matchCount == totalCount;
      return success();
    }
    result = matchCount >= matchesPolicy->minMatches;
    return success();
  }
  case slang::ast::BinsSelectExprKind::Invalid:
    return mlir::emitError(loc) << "invalid cross select expression";
  }
  llvm_unreachable("unknown BinsSelectExpr kind");
}

static LogicalResult emitFiniteTupleCrossSelect(
    const slang::ast::BinsSelectExpr &expr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    slang::ast::Compilation &compilation, OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool negate) {
  llvm::SmallDenseMap<const slang::ast::CoverpointSymbol *, size_t, 4>
      coverpointIndex;
  for (size_t i = 0; i < crossTargets.size(); ++i)
    coverpointIndex.try_emplace(crossTargets[i], i);

  CrossSelectBinDomains domains;
  CrossSelectBinRequirements requirements;
  collectCrossSelectBinRequirements(expr, crossTargets, requirements);
  if (failed(buildFiniteCrossBinDomains(crossTargets, requirements, compilation,
                                        loc,
                                        evaluateConstant, domains)))
    return failure();

  constexpr size_t kMaxSelectedTuples = 4096;
  SmallVector<SmallVector<size_t, 4>, 64> selectedTuples;

  if (domains.empty()) {
    bool selected = false;
    if (failed(evaluateCrossSelectExprOnBinTuple(
            expr, {}, domains, crossTargets, coverpointIndex, compilation, loc,
            evaluateConstant, selected)))
      return failure();
    if (negate)
      selected = !selected;
    if (selected)
      selectedTuples.emplace_back();
  } else {
    SmallVector<size_t, 4> indices(domains.size(), 0);
    bool done = false;
    while (!done) {
      bool selected = false;
      if (failed(evaluateCrossSelectExprOnBinTuple(
              expr, indices, domains, crossTargets, coverpointIndex,
              compilation, loc, evaluateConstant, selected)))
        return failure();
      if (negate)
        selected = !selected;

      if (selected) {
        selectedTuples.emplace_back(indices.begin(), indices.end());
        if (selectedTuples.size() > kMaxSelectedTuples)
          return mlir::emitError(loc)
                 << "unsupported cross select expression due to too many "
                    "selected tuples";
      }

      for (int64_t dim = static_cast<int64_t>(indices.size()) - 1; dim >= 0;
           --dim) {
        indices[dim]++;
        if (indices[dim] < domains[dim].size())
          break;
        indices[dim] = 0;
        if (dim == 0)
          done = true;
      }
    }
  }

  if (selectedTuples.empty())
    return emitAlwaysFalseCrossSelect(crossTargets, coverpointSymbols, builder,
                                      loc);

  for (size_t group = 0; group < selectedTuples.size(); ++group) {
    if (failed(emitBinTupleBinsOf(selectedTuples[group], domains, crossTargets,
                                  coverpointSymbols, builder, loc,
                                  static_cast<int32_t>(group))))
      return failure();
  }
  return success();
}
} // namespace

/// Emit one moore.binsof op for a condition leaf in a cross select expression.
static LogicalResult emitBinsOfCondition(
    const slang::ast::ConditionBinsSelectExpr &condExpr,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool negate, int32_t group) {
  auto targetRef =
      resolveCrossSelectTargetRef(condExpr.target, coverpointSymbols, builder);

  // Collect intersect values/ranges if present.
  mlir::ArrayAttr intersectValuesAttr;
  mlir::DenseI64ArrayAttr intersectRangesAttr;
  if (!condExpr.intersects.empty()) {
    constexpr uint64_t kMaxIntersectValues = 4096;
    SmallVector<int64_t> intersectValues;
    SmallVector<int64_t> intersectRanges;
    std::optional<std::pair<int64_t, int64_t>> targetBounds;
    auto getTargetBounds = [&]() -> std::optional<std::pair<int64_t, int64_t>> {
      if (targetBounds)
        return targetBounds;
      const slang::ast::CoverpointSymbol *coverpoint = nullptr;
      if (auto *cp = condExpr.target.as_if<slang::ast::CoverpointSymbol>())
        coverpoint = cp;
      else if (auto *bin =
                   condExpr.target.as_if<slang::ast::CoverageBinSymbol>()) {
        auto *parentScope = bin->getParentScope();
        if (!parentScope)
          return std::nullopt;
        coverpoint =
            parentScope->asSymbol().as_if<slang::ast::CoverpointSymbol>();
      }
      if (!coverpoint)
        return std::nullopt;
      auto &type = coverpoint->getType();
      if (!type.isIntegral())
        return std::nullopt;
      auto width = type.getBitWidth();
      if (width == 0 || width > 12)
        return std::nullopt;

      int64_t lower = 0;
      int64_t upper = 0;
      if (type.isSigned()) {
        lower = -(int64_t(1) << (width - 1));
        upper = (int64_t(1) << (width - 1)) - 1;
      } else {
        lower = 0;
        upper = (int64_t(1) << width) - 1;
      }
      targetBounds = std::pair<int64_t, int64_t>(lower, upper);
      return targetBounds;
    };

    auto appendIntersectValue = [&](int64_t value) -> LogicalResult {
      if (intersectValues.size() == kMaxIntersectValues)
        return mlir::emitError(loc)
               << "unsupported cross select expression due to large finite "
                  "value set";
      intersectValues.push_back(value);
      return success();
    };

    auto appendIntersectRange = [&](int64_t lower, int64_t upper) -> LogicalResult {
      if (lower > upper)
        std::swap(lower, upper);
      __int128 rangeCount128 =
          static_cast<__int128>(upper) - static_cast<__int128>(lower) + 1;
      if (rangeCount128 <= 0)
        return mlir::emitError(loc)
               << "unsupported cross select expression due to invalid "
                  "intersect range";

      uint64_t rangeCount = rangeCount128 > static_cast<__int128>(
                                                std::numeric_limits<uint64_t>::max())
                                ? std::numeric_limits<uint64_t>::max()
                                : static_cast<uint64_t>(rangeCount128);
      bool canExpand = rangeCount <= kMaxIntersectValues &&
                       intersectValues.size() <= kMaxIntersectValues - rangeCount;
      if (canExpand) {
        for (int64_t v = lower; v <= upper; ++v)
          intersectValues.push_back(v);
        return success();
      }

      intersectRanges.push_back(lower);
      intersectRanges.push_back(upper);
      return success();
    };

    for (const auto *intersectExpr : condExpr.intersects) {
      if (intersectExpr->kind == slang::ast::ExpressionKind::ValueRange) {
        auto &rangeExpr = intersectExpr->as<slang::ast::ValueRangeExpression>();
        if (rangeExpr.rangeKind != slang::ast::ValueRangeKind::Simple) {
          int64_t lower = 0;
          int64_t upper = 0;
          if (failed(
                  evaluateIntersectToleranceRangeBounds(rangeExpr, loc, lower, upper)))
            return failure();
          if (failed(appendIntersectRange(lower, upper)))
            return failure();
          continue;
        }

        auto leftVal = getConstantInt64(evaluateConstant(rangeExpr.left()));
        auto rightVal = getConstantInt64(evaluateConstant(rangeExpr.right()));
        bool leftUnbounded =
            !leftVal && isUnboundedConstantExpr(rangeExpr.left(), evaluateConstant);
        bool rightUnbounded =
            !rightVal && isUnboundedConstantExpr(rangeExpr.right(), evaluateConstant);
        if ((!leftVal && !leftUnbounded) || (!rightVal && !rightUnbounded))
          return mlir::emitError(loc)
                 << "unsupported non-constant intersect value range in cross "
                    "select expression";

        int64_t lower = 0;
        int64_t upper = 0;
        if (leftUnbounded || rightUnbounded) {
          auto bounds = getTargetBounds();
          if (bounds) {
            lower = leftVal ? *leftVal : bounds->first;
            upper = rightVal ? *rightVal : bounds->second;
          } else {
            lower = leftVal ? *leftVal : std::numeric_limits<int64_t>::min();
            upper = rightVal ? *rightVal : std::numeric_limits<int64_t>::max();
          }
        } else {
          lower = *leftVal;
          upper = *rightVal;
        }
        if (failed(appendIntersectRange(lower, upper)))
          return failure();
      } else {
        auto intVal = getConstantInt64(evaluateConstant(*intersectExpr));
        if (!intVal)
          return mlir::emitError(loc)
                 << "unsupported non-constant intersect value in cross select "
                    "expression";
        if (failed(appendIntersectValue(*intVal)))
          return failure();
      }
    }
    if (!intersectValues.empty()) {
      SmallVector<mlir::Attribute> intersectValueAttrs;
      intersectValueAttrs.reserve(intersectValues.size());
      for (auto value : intersectValues)
        intersectValueAttrs.push_back(builder.getI64IntegerAttr(value));
      intersectValuesAttr = builder.getArrayAttr(intersectValueAttrs);
    }
    if (!intersectRanges.empty())
      intersectRangesAttr = builder.getDenseI64ArrayAttr(intersectRanges);
  }

  auto negateAttr = negate ? builder.getUnitAttr() : mlir::UnitAttr();
  IntegerAttr groupAttr =
      group > 0 ? builder.getI32IntegerAttr(group) : IntegerAttr();
  moore::BinsOfOp::create(builder, loc, targetRef, intersectValuesAttr,
                          intersectRangesAttr, negateAttr, groupAttr);
  return success();
}

/// Helper to convert a BinsSelectExpr (binsof/intersect) to Moore IR.
/// This handles the recursive structure of bins select expressions used in
/// cross coverage bins.
LogicalResult convertBinsSelectExpr(
    const slang::ast::BinsSelectExpr &expr,
    std::span<const slang::ast::CoverpointSymbol *const> crossTargets,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    slang::ast::Compilation &compilation, OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant,
    bool negate) {
  if (expr.kind == slang::ast::BinsSelectExprKind::SetExpr) {
    if (negate)
      return mlir::emitError(loc)
             << "unsupported negation of cross set expression";
    auto &setExpr = static_cast<const slang::ast::SetExprBinsSelectExpr &>(expr);
    return emitSetExprBinsSelect(setExpr, crossTargets, coverpointSymbols,
                                 compilation, builder, loc, evaluateConstant);
  }
  if (expr.kind == slang::ast::BinsSelectExprKind::WithFilter) {
    if (negate)
      return mlir::emitError(loc)
             << "unsupported negation of cross select expression with 'with' "
                "clause";
    auto &withExpr =
        static_cast<const slang::ast::BinSelectWithFilterExpr &>(expr);
    return emitWithFilterBinsSelect(withExpr, crossTargets, coverpointSymbols,
                                    compilation, builder, loc,
                                    evaluateConstant);
  }
  if (containsWithOrSetExpr(expr))
    return emitFiniteTupleCrossSelect(expr, crossTargets, coverpointSymbols,
                                      compilation, builder, loc,
                                      evaluateConstant, negate);

  CrossSelectDNF dnf;
  if (failed(buildCrossSelectDNF(expr, loc, dnf, negate)))
    return failure();

  // Any empty DNF term means the expression can always match.
  for (auto &term : dnf)
    if (term.empty())
      return success();

  for (size_t group = 0; group < dnf.size(); ++group) {
    for (auto &leaf : dnf[group]) {
      if (failed(emitBinsOfCondition(*leaf.cond, coverpointSymbols, builder, loc,
                                     evaluateConstant, leaf.negate,
                                     static_cast<int32_t>(group))))
        return failure();
    }
  }
  return success();
}

} // namespace circt::ImportVerilog
