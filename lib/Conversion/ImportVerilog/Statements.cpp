//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include <cmath>
#include "slang/ast/Statement.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/statements/MiscStatements.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct StmtVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  StmtVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  bool isTerminated() const { return !builder.getInsertionBlock(); }
  void setTerminated() { builder.clearInsertionPoint(); }

  Block &createBlock() {
    assert(builder.getInsertionBlock());
    auto block = std::make_unique<Block>();
    block->insertAfter(builder.getInsertionBlock());
    return *block.release();
  }

  /// Handle dynamic foreach loops for associative arrays.
  /// Uses first/next pattern to iterate over keys.
  LogicalResult recursiveForeachDynamic(
      const slang::ast::ForeachLoopStatement &stmt, uint32_t level) {
    const auto &loopDim = stmt.loopDims[level];
    const auto &iter = loopDim.loopVar;

    // Get the array reference as an lvalue
    auto arrayRef = context.convertLvalueExpression(stmt.arrayRef);
    if (!arrayRef)
      return failure();

    auto iterType = context.convertType(*iter->getDeclaredType());
    if (!iterType)
      return failure();

    // Create a variable to hold the iterator key
    auto unpackedIterType = dyn_cast<moore::UnpackedType>(iterType);
    if (!unpackedIterType) {
      mlir::emitError(loc) << "foreach iterator '" << iter->name
                           << "' has non-unpacked type: " << iterType;
      return failure();
    }
    Value keyVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(unpackedIterType),
        builder.getStringAttr(iter->name), Value{});
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, keyVar);

    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&checkBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Call first() to initialize the iterator
    Value firstFound =
        moore::AssocArrayFirstOp::create(builder, loc, arrayRef, keyVar);
    firstFound = moore::ToBuiltinBoolOp::create(builder, loc, firstFound);
    cf::CondBranchOp::create(builder, loc, firstFound, &bodyBlock, &exitBlock);

    // Body block: execute body, then call next() and branch back to check
    builder.setInsertionPointToEnd(&bodyBlock);

    // find next dimension in this foreach statement
    bool hasNext = false;
    for (uint32_t nextLevel = level + 1; nextLevel < stmt.loopDims.size();
         nextLevel++) {
      if (stmt.loopDims[nextLevel].loopVar) {
        if (!stmt.loopDims[nextLevel].range.has_value()) {
          if (failed(recursiveForeachDynamic(stmt, nextLevel)))
            return failure();
        } else {
          if (failed(recursiveForeach(stmt, nextLevel)))
            return failure();
        }
        hasNext = true;
        break;
      }
    }

    if (!hasNext) {
      if (failed(context.convertStatement(stmt.body)))
        return failure();
    }

    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

    // Check block: call next() and branch to body or exit
    builder.setInsertionPointToEnd(&checkBlock);
    Value nextFound =
        moore::AssocArrayNextOp::create(builder, loc, arrayRef, keyVar);
    nextFound = moore::ToBuiltinBoolOp::create(builder, loc, nextFound);
    cf::CondBranchOp::create(builder, loc, nextFound, &bodyBlock, &exitBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult recursiveForeach(const slang::ast::ForeachLoopStatement &stmt,
                                 uint32_t level) {
    // find current dimension we are operate.
    const auto &loopDim = stmt.loopDims[level];
    if (!loopDim.range.has_value()) {
      // Dynamic loop variable - use first/next pattern for associative arrays
      return recursiveForeachDynamic(stmt, level);
    }
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    const auto &iter = loopDim.loopVar;
    auto type = context.convertType(*iter->getDeclaredType());
    if (!type)
      return failure();

    auto intType = dyn_cast<moore::IntType>(type);
    if (!intType) {
      mlir::emitError(loc) << "foreach loop variable must have integer type, "
                           << "but got " << type;
      return failure();
    }

    Value initial = moore::ConstantOp::create(
        builder, loc, intType, loopDim.range->lower(),
        /*isSigned=*/loopDim.range->lower() < 0);

    // Create loop varirable in this dimension
    Value varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(iter->name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, varOp);

    cf::BranchOp::create(builder, loc, &checkBlock);
    builder.setInsertionPointToEnd(&checkBlock);

    // When the loop variable is greater than the upper bound, goto exit
    auto upperBound = moore::ConstantOp::create(
        builder, loc, intType, loopDim.range->upper(),
        /*isSigned=*/loopDim.range->upper() < 0);

    auto var = moore::ReadOp::create(builder, loc, varOp);
    Value cond = moore::SleOp::create(builder, loc, var, upperBound);
    if (!cond)
      return failure();
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    builder.setInsertionPointToEnd(&bodyBlock);

    // find next dimension in this foreach statement, it finded then recuersive
    // resolve, else perform body statement
    bool hasNext = false;
    for (uint32_t nextLevel = level + 1; nextLevel < stmt.loopDims.size();
         nextLevel++) {
      if (stmt.loopDims[nextLevel].loopVar) {
        if (failed(recursiveForeach(stmt, nextLevel)))
          return failure();
        hasNext = true;
        break;
      }
    }

    if (!hasNext) {
      if (failed(context.convertStatement(stmt.body)))
        return failure();
    }
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    builder.setInsertionPointToEnd(&stepBlock);

    // add one to loop variable
    var = moore::ReadOp::create(builder, loc, varOp);
    if (!isa<moore::IntType>(var.getType())) {
      mlir::emitError(loc) << "foreach loop variable read produced non-integer type: "
                           << var.getType() << " (expected IntType)";
      return failure();
    }
    auto one = moore::ConstantOp::create(builder, loc, intType, 1);
    auto postValue = moore::AddOp::create(builder, loc, var, one).getResult();
    moore::BlockingAssignOp::create(builder, loc, varOp, postValue);
    cf::BranchOp::create(builder, loc, &checkBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Skip empty statements (stray semicolons).
  LogicalResult visit(const slang::ast::EmptyStatement &) { return success(); }

  // Convert every statement in a statement list. The Verilog syntax follows a
  // similar philosophy as C/C++, where things like `if` and `for` accept a
  // single statement as body. But then a `{...}` block is a valid statement,
  // which allows for the `if {...}` syntax. In Verilog, things like `final`
  // accept a single body statement, but that can be a `begin ... end` block,
  // which in turn has a single body statement, which then commonly is a list of
  // statements.
  LogicalResult visit(const slang::ast::StatementList &stmts) {
    for (auto *stmt : stmts.list) {
      if (isTerminated()) {
        auto loc = context.convertLocation(stmt->sourceRange);
        mlir::emitWarning(loc, "unreachable code");
        break;
      }
      if (failed(context.convertStatement(*stmt)))
        return failure();
    }
    return success();
  }

  // Inline `begin ... end` blocks into the parent.
  LogicalResult visit(const slang::ast::BlockStatement &stmt) {
    return context.convertStatement(stmt.body);
  }

  // Handle expression statements.
  LogicalResult visit(const slang::ast::ExpressionStatement &stmt) {
    // Special handling for calls to system tasks that return no result value.
    if (const auto *call = stmt.expr.as_if<slang::ast::CallExpression>()) {
      if (const auto *info =
              std::get_if<slang::ast::CallExpression::SystemCallInfo>(
                  &call->subroutine)) {
        auto handled = visitSystemCall(stmt, *call, *info);
        if (failed(handled))
          return failure();
        if (handled == true)
          return success();
      }

      // According to IEEE 1800-2023 Section 21.3.3 "Formatting data to a
      // string" the first argument of $sformat is its output; the other
      // arguments work like a FormatString.
      // In Moore we only support writing to a location if it is a reference;
      // However, Section 21.3.3 explains that the output of $sformat is
      // assigned as if it were cast from a string literal (Section 5.9),
      // so this implementation casts the string to the target value.
      if (!call->getSubroutineName().compare("$sformat")) {

        // Use the first argument as the output location
        auto *lhsExpr = call->arguments().front();
        // Format the second and all later arguments as a string
        auto fmtValue =
            context.convertFormatString(call->arguments().subspan(1), loc,
                                        moore::IntFormat::Decimal, false);
        if (failed(fmtValue))
          return failure();
        // Convert the FormatString to a StringType
        auto strValue = moore::FormatStringToStringOp::create(builder, loc,
                                                              fmtValue.value());
        // The Slang AST produces a `AssignmentExpression` for the first
        // argument; the RHS of this expression is invalid though
        // (`EmptyArgument`), so we only use the LHS of the
        // `AssignmentExpression` and plug in the formatted string for the RHS.
        if (auto assignExpr =
                lhsExpr->as_if<slang::ast::AssignmentExpression>()) {
          auto lhs = context.convertLvalueExpression(assignExpr->left());
          if (!lhs)
            return failure();

          auto refType = dyn_cast<moore::RefType>(lhs.getType());
          if (!refType) {
            mlir::emitError(loc) << "expected reference type for $sformat destination";
            return failure();
          }
          auto convertedValue = context.materializeConversion(
              refType.getNestedType(), strValue, false, loc);
          moore::BlockingAssignOp::create(builder, loc, lhs, convertedValue);
          return success();
        } else {
          return failure();
        }
      }

      // According to IEEE 1800-2023 Section 21.3.3 "Formatting data to a
      // string", $swrite is similar to $sformat but does not require a format
      // string. Arguments are formatted using their default representation.
      // Example: $swrite(v, pool[key]); formats pool[key] as a string into v.
      if (!call->getSubroutineName().compare("$swrite")) {

        // Use the first argument as the output location
        auto *lhsExpr = call->arguments().front();
        // Format the second and all later arguments as a string with default
        // formatting (no format string expected)
        auto fmtValue =
            context.convertFormatString(call->arguments().subspan(1), loc,
                                        moore::IntFormat::Decimal, false);
        if (failed(fmtValue))
          return failure();
        // Convert the FormatString to a StringType
        auto strValue = moore::FormatStringToStringOp::create(builder, loc,
                                                              fmtValue.value());
        // The Slang AST produces a `AssignmentExpression` for the first
        // argument; the RHS of this expression is invalid though
        // (`EmptyArgument`), so we only use the LHS of the
        // `AssignmentExpression` and plug in the formatted string for the RHS.
        if (auto assignExpr =
                lhsExpr->as_if<slang::ast::AssignmentExpression>()) {
          auto lhs = context.convertLvalueExpression(assignExpr->left());
          if (!lhs)
            return failure();

          auto refType = dyn_cast<moore::RefType>(lhs.getType());
          if (!refType) {
            mlir::emitError(loc) << "expected reference type for $swrite destination";
            return failure();
          }
          auto convertedValue = context.materializeConversion(
              refType.getNestedType(), strValue, false, loc);
          moore::BlockingAssignOp::create(builder, loc, lhs, convertedValue);
          return success();
        } else {
          return failure();
        }
      }
    }

    auto value = context.convertRvalueExpression(stmt.expr);
    if (!value)
      return failure();

    // Expressions like calls to void functions return a dummy value that has no
    // uses. If the returned value is trivially dead, remove it.
    if (auto *defOp = value.getDefiningOp()) {
      if (isOpTriviallyDead(defOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Erasing dead value-producing op: "
                                << defOp->getName() << "\n");
        defOp->erase();
      }
    }

    return success();
  }

  // Handle variable declarations.
  LogicalResult visit(const slang::ast::VariableDeclStatement &stmt) {
    const auto &var = stmt.symbol;
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "VarDecl: " << var.name << " type=" << type
                            << "\n");

    Value initial;
    if (const auto *init = var.getInitializer()) {
      initial = context.convertRvalueExpression(*init, type);
      if (!initial)
        return failure();
    }

    // Collect local temporary variables.
    auto unpackedType = dyn_cast<moore::UnpackedType>(type);
    if (!unpackedType) {
      mlir::emitError(loc) << "variable '" << var.name
                           << "' has non-unpacked type: " << type;
      return failure();
    }
    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(unpackedType),
        builder.getStringAttr(var.name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         &var, varOp);
    return success();
  }

  // Handle if statements.
  LogicalResult visit(const slang::ast::ConditionalStatement &stmt) {
    // Try to fold constant conditions up front. If all conditions evaluate to a
    // compile-time boolean, we can emit only the taken branch and skip the
    // others (avoids lowering unreachable helper code in large macros).
    std::optional<bool> constCond;
    for (const auto &condition : stmt.conditions) {
      if (condition.pattern)
        break;
      auto cv = context.evaluateConstant(*condition.expr);
      if (cv.bad())
        break;
      if (cv.isTrue()) {
        constCond = constCond ? (*constCond && true) : true;
      } else if (cv.isFalse()) {
        constCond = constCond ? (*constCond && false) : false;
      } else {
        // Non-boolean constant (e.g., string) - cannot fold.
        constCond.reset();
        break;
      }
    }
    if (constCond.has_value()) {
      if (*constCond) {
        if (failed(context.convertStatement(stmt.ifTrue)))
          return failure();
      } else if (stmt.ifFalse) {
        if (failed(context.convertStatement(*stmt.ifFalse)))
          return failure();
      }
      return success();
    }

    // Generate the condition. There may be multiple conditions linked with the
    // `&&&` operator.
    Value allConds;
    for (const auto &condition : stmt.conditions) {
      Value cond;
      if (condition.pattern) {
        auto exprValue = context.convertRvalueExpression(*condition.expr);
        if (!exprValue)
          return failure();
        auto patternMatch = matchPattern(
            *condition.pattern, exprValue, *condition.expr->type,
            slang::ast::CaseStatementCondition::Normal);
        if (failed(patternMatch))
          return failure();
        cond = *patternMatch;
      } else {
        cond = context.convertRvalueExpression(*condition.expr);
        if (!cond)
          return failure();
      }
      cond = context.convertToBool(cond);
      if (!cond)
        return failure();
      if (allConds)
        allConds = moore::AndOp::create(builder, loc, allConds, cond);
      else
        allConds = cond;
    }
    assert(allConds && "slang guarantees at least one condition");
    allConds = moore::ToBuiltinBoolOp::create(builder, loc, allConds);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, allConds, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // Generate the false branch if present.
    if (stmt.ifFalse) {
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  /// Handle pattern case statements.
  LogicalResult visit(const slang::ast::PatternCaseStatement &stmt) {
    auto caseExpr = context.convertRvalueExpression(stmt.expr);
    if (!caseExpr)
      return failure();

    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;

    for (const auto &item : stmt.items) {
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;

      auto matchResult =
          matchPattern(*item.pattern, caseExpr, *stmt.expr.type, stmt.condition);
      if (failed(matchResult))
        return failure();
      auto cond = context.convertToBool(*matchResult);
      if (!cond)
        return failure();

      if (item.filter) {
        auto filter = context.convertRvalueExpression(*item.filter);
        if (!filter)
          return failure();
        filter = context.convertToBool(filter);
        if (!filter)
          return failure();
        cond = moore::AndOp::create(builder, loc, cond, filter);
      }

      cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
      auto &nextBlock = createBlock();
      cf::CondBranchOp::create(builder, loc, cond, &matchBlock, &nextBlock);
      builder.setInsertionPointToEnd(&nextBlock);

      matchBlock.moveBefore(builder.getInsertionBlock());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      if (!isTerminated()) {
        auto itemLoc = context.convertLocation(item.stmt->sourceRange);
        cf::BranchOp::create(builder, itemLoc, &exitBlock);
      }
    }

    if (stmt.defaultCase) {
      auto &defaultBlock = createBlock();
      cf::BranchOp::create(builder, loc, &defaultBlock);
      builder.setInsertionPointToEnd(&defaultBlock);
      if (failed(context.convertStatement(*stmt.defaultCase)))
        return failure();
      if (!isTerminated()) {
        auto defLoc = context.convertLocation(stmt.defaultCase->sourceRange);
        cf::BranchOp::create(builder, defLoc, &exitBlock);
      }
    } else if (lastMatchBlock) {
      cf::BranchOp::create(builder, loc, &exitBlock);
    }

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  /// Handle case statements.
  LogicalResult visit(const slang::ast::CaseStatement &caseStmt) {
    using slang::ast::AttributeSymbol;
    using slang::ast::CaseStatementCondition;
    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    // Check each case individually. This currently ignores the `unique`,
    // `unique0`, and `priority` modifiers which would allow for additional
    // optimizations.
    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;
    SmallVector<moore::FVIntegerAttr> itemConsts;

    for (const auto &item : caseStmt.items) {
      // Create the block that will contain the main body of the expression.
      // This is where any of the comparisons will branch to if they match.
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;

      // The SV standard requires expressions to be checked in the order
      // specified by the user, and for the evaluation to stop as soon as the
      // first matching expression is encountered.
      for (const auto *expr : item.expressions) {
        auto value = context.convertRvalueExpression(*expr);
        if (!value)
          return failure();
        auto itemLoc = value.getLoc();

        // Take note if the expression is a constant.
        auto maybeConst = value;
        while (isa_and_nonnull<moore::ConversionOp, moore::IntToLogicOp,
                               moore::LogicToIntOp>(maybeConst.getDefiningOp()))
          maybeConst = maybeConst.getDefiningOp()->getOperand(0);
        if (auto defOp = maybeConst.getDefiningOp<moore::ConstantOp>())
          itemConsts.push_back(defOp.getValueAttr());

        // Generate the appropriate equality operator based on type.
        Value cond;
        if (isa<moore::StringType>(caseExpr.getType())) {
          // String case statement - use string comparison.
          cond = moore::StringCmpOp::create(
              builder, itemLoc, moore::StringCmpPredicate::eq, caseExpr, value);
          cond = context.convertToBool(cond);
          if (!cond)
            return failure();
          cond = moore::ToBuiltinBoolOp::create(builder, itemLoc, cond);
        } else {
          // Integer/enum case statement - convert to simple bit vector.
          auto caseExprSBV = context.convertToSimpleBitVector(caseExpr);
          auto valueSBV = context.convertToSimpleBitVector(value);
          if (!caseExprSBV || !valueSBV)
            return failure();
          switch (caseStmt.condition) {
          case CaseStatementCondition::Normal:
            cond = moore::CaseEqOp::create(builder, itemLoc, caseExprSBV,
                                           valueSBV);
            break;
          case CaseStatementCondition::WildcardXOrZ:
            cond = moore::CaseXZEqOp::create(builder, itemLoc, caseExprSBV,
                                             valueSBV);
            break;
          case CaseStatementCondition::WildcardJustZ:
            cond = moore::CaseZEqOp::create(builder, itemLoc, caseExprSBV,
                                            valueSBV);
            break;
          case CaseStatementCondition::Inside:
            mlir::emitError(loc, "unsupported set membership case statement");
            return failure();
          }
          cond = moore::ToBuiltinBoolOp::create(builder, itemLoc, cond);
        }

        // If the condition matches, branch to the match block. Otherwise
        // continue checking the next expression in a new block.
        auto &nextBlock = createBlock();
        mlir::cf::CondBranchOp::create(builder, itemLoc, cond, &matchBlock,
                                       &nextBlock);
        builder.setInsertionPointToEnd(&nextBlock);
      }

      // The current block is the fall-through after all conditions have been
      // checked and nothing matched. Move the match block up before this point
      // to make the IR easier to read.
      matchBlock.moveBefore(builder.getInsertionBlock());

      // Generate the code for this item's statement in the match block.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      if (!isTerminated()) {
        auto loc = context.convertLocation(item.stmt->sourceRange);
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
      }
    }

    const auto caseStmtAttrs = context.compilation.getAttributes(caseStmt);
    const bool hasFullCaseAttr =
        llvm::find_if(caseStmtAttrs, [](const AttributeSymbol *attr) {
          return attr->name == "full_case";
        }) != caseStmtAttrs.end();

    // Check if the case statement looks exhaustive assuming two-state values.
    // We use this information to work around a common bug in input Verilog
    // where a case statement enumerates all possible two-state values of the
    // case expression, but forgets to deal with cases involving X and Z bits in
    // the input.
    //
    // Once the core dialects start supporting four-state values we may want to
    // tuck this behind an import option that is on by default, since it does
    // not preserve semantics.
    auto twoStateExhaustive = false;
    if (auto intType = dyn_cast<moore::IntType>(caseExpr.getType());
        intType && intType.getWidth() < 32 &&
        itemConsts.size() == (1 << intType.getWidth())) {
      // Sort the constants by value.
      llvm::sort(itemConsts, [](auto a, auto b) {
        return a.getValue().getRawValue().ult(b.getValue().getRawValue());
      });

      // Ensure that every possible value of the case expression is present. Do
      // this by starting at 0 and iterating over all sorted items. Each item
      // must be the previous item + 1. At the end, the addition must exactly
      // overflow and take us back to zero.
      auto nextValue = FVInt::getZero(intType.getWidth());
      for (auto value : itemConsts) {
        if (value.getValue() != nextValue)
          break;
        nextValue += 1;
      }
      twoStateExhaustive = nextValue.isZero();
    }

    // If the case statement is exhaustive assuming two-state values, don't
    // generate the default case. Instead, branch to the last match block. This
    // will essentially make the last case item the "default".
    //
    // Alternatively, if the case statement has an (* full_case *) attribute
    // but no default case, it indicates that the developer has intentionally
    // covered all known possible values. Hence, the last match block is
    // treated as the implicit "default" case.
    if ((twoStateExhaustive || (hasFullCaseAttr && !caseStmt.defaultCase)) &&
        lastMatchBlock &&
        caseStmt.condition == CaseStatementCondition::Normal) {
      mlir::cf::BranchOp::create(builder, loc, lastMatchBlock);
    } else {
      // Generate the default case if present.
      if (caseStmt.defaultCase)
        if (failed(context.convertStatement(*caseStmt.defaultCase)))
          return failure();
      if (!isTerminated())
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  FailureOr<Value> matchPattern(const slang::ast::Pattern &pattern, Value value,
                                const slang::ast::Type &targetType,
                                slang::ast::CaseStatementCondition condKind) {
    using slang::ast::PatternKind;
    switch (pattern.kind) {
    case PatternKind::Wildcard: {
      auto boolType = moore::IntType::get(context.getContext(), 1,
                                          moore::Domain::TwoValued);
      return Value(moore::ConstantOp::create(builder, loc, boolType, 1));
    }
    case PatternKind::Constant: {
      auto &constPattern = pattern.as<slang::ast::ConstantPattern>();
      auto rhs = context.convertRvalueExpression(constPattern.expr);
      if (!rhs)
        return failure();
      return comparePatternValue(value, rhs, condKind);
    }
    case PatternKind::Tagged:
      return matchTaggedPattern(pattern.as<slang::ast::TaggedPattern>(), value,
                                targetType, condKind);
    default:
      mlir::emitError(loc) << "unsupported pattern in case statement: "
                           << slang::ast::toString(pattern.kind);
      return failure();
    }
  }

  FailureOr<Value>
  comparePatternValue(Value lhs, Value rhs,
                      slang::ast::CaseStatementCondition condKind) {
    if (isa<moore::StringType>(lhs.getType())) {
      auto cond = moore::StringCmpOp::create(
          builder, loc, moore::StringCmpPredicate::eq, lhs, rhs);
      return context.convertToBool(cond);
    }

    auto lhsSBV = context.convertToSimpleBitVector(lhs);
    auto rhsSBV = context.convertToSimpleBitVector(rhs);
    if (!lhsSBV || !rhsSBV)
      return failure();

    Value cond;
    switch (condKind) {
    case slang::ast::CaseStatementCondition::Normal:
      cond = moore::CaseEqOp::create(builder, loc, lhsSBV, rhsSBV);
      break;
    case slang::ast::CaseStatementCondition::WildcardXOrZ:
      cond = moore::CaseXZEqOp::create(builder, loc, lhsSBV, rhsSBV);
      break;
    case slang::ast::CaseStatementCondition::WildcardJustZ:
      cond = moore::CaseZEqOp::create(builder, loc, lhsSBV, rhsSBV);
      break;
    case slang::ast::CaseStatementCondition::Inside:
      mlir::emitError(loc, "unsupported set membership pattern match");
      return failure();
    }
    return context.convertToBool(cond);
  }

  FailureOr<Value>
  matchTaggedPattern(const slang::ast::TaggedPattern &pattern, Value value,
                     const slang::ast::Type &targetType,
                     slang::ast::CaseStatementCondition condKind) {
    if (!targetType.isTaggedUnion()) {
      mlir::emitError(loc)
          << "tagged pattern applied to non-tagged union type";
      return failure();
    }

    auto getTaggedUnionField =
        [&](Type containerType, StringRef fieldName)
        -> std::optional<moore::StructLikeMember> {
      if (auto structTy = dyn_cast<moore::StructType>(containerType)) {
        for (auto member : structTy.getMembers())
          if (member.name.getValue() == fieldName)
            return member;
      } else if (auto structTy =
                     dyn_cast<moore::UnpackedStructType>(containerType)) {
        for (auto member : structTy.getMembers())
          if (member.name.getValue() == fieldName)
            return member;
      }
      return std::nullopt;
    };

    auto containerType = value.getType();
    if (auto refTy = dyn_cast<moore::RefType>(containerType))
      containerType = refTy.getNestedType();

    auto dataMember = getTaggedUnionField(containerType, "data");
    auto tagMember = getTaggedUnionField(containerType, "tag");
    if (!dataMember || !tagMember) {
      mlir::emitError(loc)
          << "tagged union lowering expected {tag, data} struct wrapper";
      return failure();
    }

    unsigned tagIndex = 0;
    bool found = false;
    const auto &canonical = targetType.getCanonicalType();
    if (auto *packed = canonical.as_if<slang::ast::PackedUnionType>()) {
      for (auto &member : packed->membersOfType<slang::ast::FieldSymbol>()) {
        if (&member == &pattern.member) {
          found = true;
          break;
        }
        ++tagIndex;
      }
    } else if (auto *unpacked =
                   canonical.as_if<slang::ast::UnpackedUnionType>()) {
      for (auto &member : unpacked->membersOfType<slang::ast::FieldSymbol>()) {
        if (&member == &pattern.member) {
          found = true;
          break;
        }
        ++tagIndex;
      }
    }

    if (!found) {
      mlir::emitError(loc) << "could not resolve tagged union member index";
      return failure();
    }

    auto tagValue =
        moore::StructExtractOp::create(builder, loc, tagMember->type,
                                       tagMember->name, value);
    auto tagIntType = dyn_cast<moore::IntType>(tagMember->type);
    if (!tagIntType) {
      mlir::emitError(loc) << "tagged union tag member must have integer type";
      return failure();
    }
    auto tagConst = moore::ConstantOp::create(builder, loc, tagIntType,
                                              static_cast<int64_t>(tagIndex));
    auto tagMatch = comparePatternValue(tagValue, Value(tagConst),
                                        slang::ast::CaseStatementCondition::Normal);
    if (failed(tagMatch))
      return failure();

    if (!pattern.valuePattern)
      return tagMatch;

    auto unionValue =
        moore::StructExtractOp::create(builder, loc, dataMember->type,
                                       dataMember->name, value);
    auto memberName = builder.getStringAttr(pattern.member.name);
    auto memberType = context.convertType(*pattern.member.getDeclaredType());
    if (!memberType)
      return failure();
    if (isa<moore::VoidType>(memberType))
      memberType = moore::IntType::getInt(context.getContext(), 1);

    auto memberValue = moore::UnionExtractOp::create(
        builder, loc, memberType, memberName, unionValue);
    auto valueMatch =
        matchPattern(*pattern.valuePattern, memberValue,
                     pattern.member.getDeclaredType()->getType(), condKind);
    if (failed(valueMatch))
      return failure();

    auto combined =
        moore::AndOp::create(builder, loc, *tagMatch, *valueMatch);
    return context.convertToBool(combined);
  }

  // Handle `for` loops.
  LogicalResult visit(const slang::ast::ForLoopStatement &stmt) {
    // Generate the initializers.
    for (auto *initExpr : stmt.initializers)
      if (!context.convertRvalueExpression(*initExpr))
        return failure();

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    cf::BranchOp::create(builder, loc, &checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(*stmt.stopExpr);
    if (!cond)
      return failure();
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Generate the step expressions.
    builder.setInsertionPointToEnd(&stepBlock);
    for (auto *stepExpr : stmt.steps)
      if (!context.convertRvalueExpression(*stepExpr))
        return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult visit(const slang::ast::ForeachLoopStatement &stmt) {
    for (uint32_t level = 0; level < stmt.loopDims.size(); level++) {
      if (stmt.loopDims[level].loopVar)
        return recursiveForeach(stmt, level);
    }
    return success();
  }

  // Handle `repeat` loops.
  LogicalResult visit(const slang::ast::RepeatLoopStatement &stmt) {
    auto count = context.convertRvalueExpression(stmt.count);
    if (!count)
      return failure();

    // Verify the count is an integer type before proceeding.
    auto countIntType = dyn_cast<moore::IntType>(count.getType());
    if (!countIntType) {
      mlir::emitError(count.getLoc())
          << "repeat loop count must have integer type, but got "
          << count.getType();
      return failure();
    }

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    auto currentCount = checkBlock.addArgument(count.getType(), count.getLoc());
    cf::BranchOp::create(builder, loc, &checkBlock, count);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertToBool(currentCount);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Decrement the current count and branch back to the check block.
    builder.setInsertionPointToEnd(&stepBlock);
    auto one = moore::ConstantOp::create(builder, count.getLoc(), countIntType, 1);
    Value nextCount =
        moore::SubOp::create(builder, count.getLoc(), currentCount, one);
    cf::BranchOp::create(builder, loc, &checkBlock, nextCount);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `while` and `do-while` loops.
  LogicalResult createWhileLoop(const slang::ast::Expression &condExpr,
                                const slang::ast::Statement &bodyStmt,
                                bool atLeastOnce) {
    // Create the blocks for the loop condition, body, and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    cf::BranchOp::create(builder, loc, atLeastOnce ? &bodyBlock : &checkBlock);
    if (atLeastOnce)
      bodyBlock.moveBefore(&checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&checkBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(condExpr);
    if (!cond)
      return failure();
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(bodyStmt)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult visit(const slang::ast::WhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, false);
  }

  LogicalResult visit(const slang::ast::DoWhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, true);
  }

  // Handle `forever` loops.
  LogicalResult visit(const slang::ast::ForeverLoopStatement &stmt) {
    // Create the blocks for the loop body and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    cf::BranchOp::create(builder, loc, &bodyBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&bodyBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &bodyBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle timing control.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    return context.convertTimingControl(stmt.timing, stmt.stmt);
  }

  // Handle wait statements: wait(condition)
  LogicalResult visit(const slang::ast::WaitStatement &stmt) {
    auto cond = context.convertRvalueExpression(stmt.cond);
    if (!cond)
      return failure();
    // WaitConditionOp expects a 2-state bit type.
    cond = context.convertToBool(cond, Domain::TwoValued);
    if (!cond)
      return failure();
    moore::WaitConditionOp::create(builder, loc, cond);
    // Execute the body statement if any
    return context.convertStatement(stmt.stmt);
  }

  // Handle wait fork statements: wait fork
  LogicalResult visit(const slang::ast::WaitForkStatement &stmt) {
    moore::WaitForkOp::create(builder, loc);
    return success();
  }

  // Handle disable fork statements: disable fork
  LogicalResult visit(const slang::ast::DisableForkStatement &stmt) {
    moore::DisableForkOp::create(builder, loc);
    return success();
  }

  // Handle return statements.
  LogicalResult visit(const slang::ast::ReturnStatement &stmt) {
    // Check if we're inside a randsequence production - if so, return exits
    // the entire randsequence per IEEE 1800-2017 Section 18.17
    if (!context.randSequenceReturnStack.empty()) {
      // A return inside a randsequence production should not have a value
      // (it just exits the randsequence, not the function)
      if (stmt.expr) {
        // If there's an expression, it's being used to return from the function
        // containing the randsequence, so fall through to normal return handling
      } else {
        cf::BranchOp::create(builder, loc,
                             context.randSequenceReturnStack.back());
        setTerminated();
        return success();
      }
    }

    if (stmt.expr) {
      auto expr = context.convertRvalueExpression(*stmt.expr);
      if (!expr)
        return failure();
      // Ensure return type matches the function signature.
      if (auto *lowering = context.currentFunctionLowering) {
        auto funcTy = lowering->op.getFunctionType();
        if (funcTy.getNumResults() > 0) {
          auto expectedTy = funcTy.getResult(0);
          expr = context.materializeConversion(expectedTy, expr,
                                               stmt.expr->type->isSigned(),
                                               expr.getLoc());
          if (!expr) {
            mlir::emitError(loc)
                << "failed to convert return expression to expected type "
                << expectedTy;
            return failure();
          }
        }
      }
      mlir::func::ReturnOp::create(builder, loc, expr);
    } else {
      mlir::func::ReturnOp::create(builder, loc);
    }
    setTerminated();
    return success();
  }

  // Handle continue statements.
  LogicalResult visit(const slang::ast::ContinueStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc,
                             "cannot `continue` without a surrounding loop");
    cf::BranchOp::create(builder, loc, context.loopStack.back().continueBlock);
    setTerminated();
    return success();
  }

  // Handle break statements.
  LogicalResult visit(const slang::ast::BreakStatement &stmt) {
    // Check if we're inside a loop - if so, break exits the loop
    if (!context.loopStack.empty()) {
      cf::BranchOp::create(builder, loc, context.loopStack.back().breakBlock);
      setTerminated();
      return success();
    }

    // Check if we're inside a randsequence production - if so, break exits
    // the current production code block per IEEE 1800-2017 Section 18.17
    if (!context.randSequenceBreakStack.empty()) {
      cf::BranchOp::create(builder, loc,
                           context.randSequenceBreakStack.back());
      setTerminated();
      return success();
    }

    return mlir::emitError(loc, "cannot `break` without a surrounding loop "
                                "or randsequence production");
  }

  // Handle immediate assertion statements.
  LogicalResult visit(const slang::ast::ImmediateAssertionStatement &stmt) {
    auto cond = context.convertRvalueExpression(stmt.cond);
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();

    // Handle assertion statements that don't have an action block.
    if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      auto defer = moore::DeferAssert::Immediate;
      if (stmt.isFinal)
        defer = moore::DeferAssert::Final;
      else if (stmt.isDeferred)
        defer = moore::DeferAssert::Observed;

      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        moore::AssertOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        moore::AssumeOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        moore::CoverOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported immediate assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    // Regard assertion statements with an action block as the "if-else".
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, cond, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (stmt.ifTrue && failed(context.convertStatement(*stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    if (stmt.ifFalse) {
      // Generate the false branch if present.
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle concurrent assertion statements.
  LogicalResult visit(const slang::ast::ConcurrentAssertionStatement &stmt) {
    auto loc = context.convertLocation(stmt.sourceRange);
    auto property = context.convertAssertionExpression(stmt.propertySpec, loc);
    if (!property)
      return failure();

    if (stmt.ifTrue && !stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      mlir::emitWarning(loc)
          << "ignoring concurrent assertion action blocks during import";
    }

    switch (stmt.assertionKind) {
    case slang::ast::AssertionKind::Assert:
      verif::AssertOp::create(builder, loc, property, Value(), StringAttr{});
      return success();
    case slang::ast::AssertionKind::Assume:
      verif::AssumeOp::create(builder, loc, property, Value(), StringAttr{});
      return success();
    case slang::ast::AssertionKind::CoverProperty:
      verif::CoverOp::create(builder, loc, property, Value(), StringAttr{});
      return success();
    default:
      break;
    }

    mlir::emitError(loc) << "unsupported concurrent assertion kind: "
                         << slang::ast::toString(stmt.assertionKind);
    return failure();
  }

  // According to 1800-2023 Section 21.2.1 "The display and write tasks":
  // >> The $display and $write tasks display their arguments in the same
  // >> order as they appear in the argument list. Each argument can be a
  // >> string literal or an expression that returns a value.
  // According to Section 20.10 "Severity system tasks", the same
  // semantics apply to $fatal, $error, $warning, and $info.
  // This means we must first check whether the first "string-able"
  // argument is a Literal Expression which doesn't represent a fully-formatted
  // string, otherwise we convert it to a FormatStringType.
  FailureOr<Value>
  getDisplayMessage(std::span<const slang::ast::Expression *const> args) {
    if (args.size() == 0)
      return Value{};

    // Handle the string formatting.
    // If the second argument is a Literal of some type, we should either
    // treat it as a literal-to-be-formatted or a FormatStringType.
    // In this check we use a StringLiteral, but slang allows casting between
    // any literal expressions (strings, integers, reals, and time at least) so
    // this is short-hand for "any value literal"
    if (args[0]->as_if<slang::ast::StringLiteral>()) {
      return context.convertFormatString(args, loc);
    }
    // Check if there's only one argument and it's a FormatStringType
    if (args.size() == 1) {
      return context.convertRvalueExpression(
          *args[0], builder.getType<moore::FormatStringType>());
    }
    // Otherwise this looks invalid. Raise an error.
    return emitError(loc) << "Failed to convert Display Message!";
  }

  /// Handle the subset of system calls that return no result value. Return
  /// true if the called system task could be handled, false otherwise. Return
  /// failure if an error occurred.
  FailureOr<bool>
  visitSystemCall(const slang::ast::ExpressionStatement &stmt,
                  const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;
    auto args = expr.arguments();

    // Simulation Control Tasks

    if (subroutine.name == "$dumpfile" || subroutine.name == "$dumpvars") {
      // These tasks are simulator-specific; ignore for now.
      mlir::emitRemark(loc) << "ignoring system task `" << subroutine.name
                            << "`";
      return true;
    }

    if (subroutine.name == "$stop") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      moore::StopBIOp::create(builder, loc);
      return true;
    }

    if (subroutine.name == "$finish") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      moore::FinishBIOp::create(builder, loc, 0);
      moore::UnreachableOp::create(builder, loc);
      setTerminated();
      return true;
    }

    if (subroutine.name == "$exit") {
      // Calls to `$exit` from outside a `program` are ignored. Since we don't
      // yet support programs, there is nothing to do here.
      // TODO: Fix this once we support programs.
      return true;
    }

    // Display and Write Tasks (`$display[boh]?` or `$write[boh]?`)

    // Check for a `$display` or `$write` prefix.
    bool isDisplay = false;     // display or write
    bool appendNewline = false; // display
    StringRef remainingName = subroutine.name;
    if (remainingName.consume_front("$display")) {
      isDisplay = true;
      appendNewline = true;
    } else if (remainingName.consume_front("$write")) {
      isDisplay = true;
    }

    // Check for optional `b`, `o`, or `h` suffix indicating default format.
    using moore::IntFormat;
    IntFormat defaultFormat = IntFormat::Decimal;
    if (isDisplay && !remainingName.empty()) {
      if (remainingName == "b")
        defaultFormat = IntFormat::Binary;
      else if (remainingName == "o")
        defaultFormat = IntFormat::Octal;
      else if (remainingName == "h")
        defaultFormat = IntFormat::HexLower;
      else
        isDisplay = false;
    }

    if (isDisplay) {
      auto message =
          context.convertFormatString(args, loc, defaultFormat, appendNewline);
      if (failed(message))
        return failure();
      if (*message == Value{})
        return true;
      moore::DisplayBIOp::create(builder, loc, *message);
      return true;
    }

    // File I/O Tasks (IEEE 1800-2017 Section 21.3)
    if (subroutine.name == "$fclose") {
      if (args.size() != 1) {
        mlir::emitError(loc) << "$fclose expects exactly one argument";
        return failure();
      }
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return failure();
      moore::FCloseBIOp::create(builder, loc, fd);
      return true;
    }

    // File Write Tasks (`$fwrite[boh]?` or `$fdisplay[boh]?`)
    // Check for a `$fwrite` or `$fdisplay` prefix.
    bool isFWrite = false;
    bool appendNewlineFWrite = false;
    StringRef remainingNameFWrite = subroutine.name;
    if (remainingNameFWrite.consume_front("$fdisplay")) {
      isFWrite = true;
      appendNewlineFWrite = true;
    } else if (remainingNameFWrite.consume_front("$fwrite")) {
      isFWrite = true;
    }

    // Check for optional `b`, `o`, or `h` suffix indicating default format.
    using moore::IntFormat;
    IntFormat defaultFormatFWrite = IntFormat::Decimal;
    if (isFWrite && !remainingNameFWrite.empty()) {
      if (remainingNameFWrite == "b")
        defaultFormatFWrite = IntFormat::Binary;
      else if (remainingNameFWrite == "o")
        defaultFormatFWrite = IntFormat::Octal;
      else if (remainingNameFWrite == "h")
        defaultFormatFWrite = IntFormat::HexLower;
      else
        isFWrite = false;
    }

    if (isFWrite) {
      // $fwrite(fd, format, args...) - first arg is file descriptor
      if (args.empty())
        return mlir::emitError(loc, "$fwrite requires at least one argument");

      // Convert the file descriptor argument
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return failure();

      // Convert the remaining arguments as format string
      auto formatArgs = args.subspan(1);
      auto message = context.convertFormatString(formatArgs, loc,
                                                 defaultFormatFWrite,
                                                 appendNewlineFWrite);
      if (failed(message))
        return failure();
      if (*message == Value{})
        return true;
      moore::FWriteBIOp::create(builder, loc, fd, *message);
      return true;
    }

    // Severity Tasks
    using moore::Severity;
    std::optional<Severity> severity;
    if (subroutine.name == "$info")
      severity = Severity::Info;
    else if (subroutine.name == "$warning")
      severity = Severity::Warning;
    else if (subroutine.name == "$error")
      severity = Severity::Error;
    else if (subroutine.name == "$fatal")
      severity = Severity::Fatal;

    if (severity) {
      // The `$fatal` task has an optional leading verbosity argument.
      const slang::ast::Expression *verbosityExpr = nullptr;
      if (severity == Severity::Fatal && args.size() >= 1) {
        verbosityExpr = args[0];
        args = args.subspan(1);
      }

      FailureOr<Value> maybeMessage = getDisplayMessage(args);
      if (failed(maybeMessage))
        return failure();
      auto message = maybeMessage.value();

      if (message == Value{})
        message = moore::FormatLiteralOp::create(builder, loc, "");
      moore::SeverityBIOp::create(builder, loc, *severity, message);

      // Handle the `$fatal` case which behaves like a `$finish`.
      if (severity == Severity::Fatal) {
        createFinishMessage(verbosityExpr);
        moore::FinishBIOp::create(builder, loc, 1);
        moore::UnreachableOp::create(builder, loc);
        setTerminated();
      }
      return true;
    }

    // Give up on any other system tasks. These will be tried again as an
    // expression later.
    return false;
  }

  /// Create the optional diagnostic message print for finish-like ops.
  void createFinishMessage(const slang::ast::Expression *verbosityExpr) {
    unsigned verbosity = 1;
    if (verbosityExpr) {
      auto value =
          context.evaluateConstant(*verbosityExpr).integer().as<unsigned>();
      assert(value && "Slang guarantees constant verbosity parameter");
      verbosity = *value;
    }
    if (verbosity == 0)
      return;
    moore::FinishMessageBIOp::create(builder, loc, verbosity > 1);
  }

  // Handle event trigger statements.
  LogicalResult visit(const slang::ast::EventTriggerStatement &stmt) {
    if (stmt.timing) {
      mlir::emitError(loc) << "unsupported delayed event trigger";
      return failure();
    }

    // Get an lvalue ref to the event target.
    auto target = context.convertLvalueExpression(stmt.target);
    if (!target)
      return failure();

    // Read the current value of the target.
    Value readValue = moore::ReadOp::create(builder, loc, target);

    // Check if this is an event type - if so, use EventTriggerOp.
    if (isa<moore::EventType>(readValue.getType())) {
      moore::EventTriggerOp::create(builder, loc, readValue);
      return success();
    }

    // For integer types, use the toggle mechanism: invert the current value
    // and write it back to signal the event.
    if (!isa<moore::IntType>(readValue.getType())) {
      mlir::emitError(loc) << "event target must have event or integer type, "
                           << "but got " << readValue.getType();
      return failure();
    }
    Value inverted = moore::NotOp::create(builder, loc, readValue);

    if (stmt.isNonBlocking)
      moore::NonBlockingAssignOp::create(builder, loc, target, inverted);
    else
      moore::BlockingAssignOp::create(builder, loc, target, inverted);
    return success();
  }

  // Handle randsequence statements.
  LogicalResult visit(const slang::ast::RandSequenceStatement &stmt) {
    if (!stmt.firstProduction)
      return success();

    // Create an exit block for the entire randsequence. A 'return' statement
    // within any production will branch to this block.
    Block &exitBlock = createBlock();
    context.randSequenceReturnStack.push_back(&exitBlock);
    auto returnGuard = llvm::make_scope_exit(
        [&] { context.randSequenceReturnStack.pop_back(); });

    if (failed(executeProduction(*stmt.firstProduction)))
      return failure();

    // If not terminated, branch to exit block
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // Set up the exit block as the continuation point after randsequence
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      // Already terminated, nothing to do
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle randcase statements.
  // RandCase provides weighted random case selection (IEEE 1800-2017 Section
  // 18.16). We generate weighted random selection using $urandom_range,
  // computing total weight and using cascading comparisons to select an item.
  LogicalResult visit(const slang::ast::RandCaseStatement &stmt) {
    // Empty randcase - nothing to do
    if (stmt.items.empty())
      return success();

    // Collect weights from each item
    SmallVector<int64_t> weights;
    int64_t totalWeight = 0;
    for (const auto &item : stmt.items) {
      // Evaluate the weight expression (must be a constant integer)
      auto cv = context.evaluateConstant(*item.expr);
      int64_t weight = 1; // Default weight if evaluation fails
      if (!cv.bad()) {
        auto maybeWeight = cv.integer().as<int64_t>();
        if (maybeWeight)
          weight = *maybeWeight;
      }
      // Weights must be non-negative
      if (weight < 0)
        weight = 0;
      weights.push_back(weight);
      totalWeight += weight;
    }

    // If all weights are 0, treat them all as equal weight of 1
    if (totalWeight == 0) {
      totalWeight = stmt.items.size();
      for (auto &w : weights)
        w = 1;
    }

    // If there's only one item with positive weight, execute it directly
    size_t nonZeroCount = 0;
    size_t lastNonZeroIdx = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
      if (weights[i] > 0) {
        nonZeroCount++;
        lastNonZeroIdx = i;
      }
    }
    if (nonZeroCount == 1)
      return context.convertStatement(*stmt.items[lastNonZeroIdx].stmt);

    // Generate a random number in [0, totalWeight-1]
    auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
    Value maxVal =
        moore::ConstantOp::create(builder, loc, i32Ty, totalWeight - 1);
    Value randVal =
        moore::UrandomRangeBIOp::create(builder, loc, maxVal, Value{});

    // Create blocks for each item and an exit block
    auto &exitBlock = createBlock();
    SmallVector<Block *> itemBlocks;
    for (size_t i = 0; i < stmt.items.size(); ++i)
      itemBlocks.push_back(&createBlock());

    // Generate cascading comparisons to select the item
    // If rand < weight[0], goto item[0]
    // else if rand < weight[0] + weight[1], goto item[1]
    // etc.
    int64_t cumWeight = 0;
    for (size_t i = 0; i < stmt.items.size(); ++i) {
      cumWeight += weights[i];
      auto threshold = moore::ConstantOp::create(builder, loc, i32Ty, cumWeight);
      Value cond = moore::UltOp::create(builder, loc, randVal, threshold);
      cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

      if (i == stmt.items.size() - 1) {
        // Last item - unconditional branch
        cf::BranchOp::create(builder, loc, itemBlocks[i]);
      } else {
        auto &nextCheckBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, cond, itemBlocks[i],
                                 &nextCheckBlock);
        builder.setInsertionPointToEnd(&nextCheckBlock);
      }
    }

    // Generate code for each item
    for (size_t i = 0; i < stmt.items.size(); ++i) {
      builder.setInsertionPointToEnd(itemBlocks[i]);
      if (failed(context.convertStatement(*stmt.items[i].stmt)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // Continue at exit block
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult executeProduction(
      const slang::ast::RandSeqProductionSymbol &prod) {
    auto rules = prod.getRules();
    if (rules.empty())
      return success();
    if (rules.size() == 1)
      return executeRule(rules[0]);

    SmallVector<int64_t> weights;
    int64_t totalWeight = 0;
    for (const auto &rule : rules) {
      int64_t weight = 1;
      if (rule.weightExpr) {
        auto cv = context.evaluateConstant(*rule.weightExpr);
        if (!cv.bad()) {
          auto maybeWeight = cv.integer().as<int64_t>();
          if (maybeWeight)
            weight = *maybeWeight;
        }
      }
      if (weight < 0)
        weight = 0;
      weights.push_back(weight);
      totalWeight += weight;
    }

    if (totalWeight == 0) {
      totalWeight = rules.size();
      for (auto &w : weights)
        w = 1;
    }

    size_t nonZeroCount = 0;
    size_t lastNonZeroIdx = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
      if (weights[i] > 0) {
        nonZeroCount++;
        lastNonZeroIdx = i;
      }
    }
    if (nonZeroCount == 1)
      return executeRule(rules[lastNonZeroIdx]);

    auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
    Value maxVal =
        moore::ConstantOp::create(builder, loc, i32Ty, totalWeight - 1);
    Value randVal =
        moore::UrandomRangeBIOp::create(builder, loc, maxVal, Value{});

    auto &exitBlock = createBlock();
    SmallVector<Block *> ruleBlocks;
    for (size_t i = 0; i < rules.size(); ++i)
      ruleBlocks.push_back(&createBlock());

    int64_t cumWeight = 0;
    for (size_t i = 0; i < rules.size(); ++i) {
      cumWeight += weights[i];
      auto threshold =
          moore::ConstantOp::create(builder, loc, i32Ty, cumWeight);
      Value cond = moore::UltOp::create(builder, loc, randVal, threshold);
      cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

      if (i == rules.size() - 1) {
        cf::BranchOp::create(builder, loc, ruleBlocks[i]);
      } else {
        auto &nextCheckBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, cond, ruleBlocks[i],
                                 &nextCheckBlock);
        builder.setInsertionPointToEnd(&nextCheckBlock);
      }
    }

    for (size_t i = 0; i < rules.size(); ++i) {
      builder.setInsertionPointToEnd(ruleBlocks[i]);
      if (failed(executeRule(rules[i])))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult
  executeRule(const slang::ast::RandSeqProductionSymbol::Rule &rule) {
    Context::ValueSymbolScope ruleScope(context.valueSymbols);
    auto savedScope = context.currentScope;
    context.currentScope = rule.ruleBlock;
    auto scopeGuard =
        llvm::make_scope_exit([&] { context.currentScope = savedScope; });

    // Handle the initial code block before production items
    if (rule.codeBlock) {
      // Set up a break target for the code block
      Block &breakBlock = createBlock();
      context.randSequenceBreakStack.push_back(&breakBlock);
      auto breakGuard = llvm::make_scope_exit(
          [&] { context.randSequenceBreakStack.pop_back(); });

      if (failed(convertStatementBlock(*rule.codeBlock->block)))
        return failure();

      // If not terminated, branch to break block
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &breakBlock);

      // Set up the break block as the continuation point
      if (breakBlock.hasNoPredecessors()) {
        breakBlock.erase();
        // Already terminated, return early
        return success();
      } else {
        builder.setInsertionPointToEnd(&breakBlock);
      }
    }
    if (isTerminated())
      return success();

    if (rule.isRandJoin) {
      int64_t joinCount = 0;
      if (rule.randJoinExpr) {
        auto cv = context.evaluateConstant(*rule.randJoinExpr);
        if (!cv.bad()) {
          // Per IEEE 1800-2017 Section 18.17.5:
          // - Integer N: execute exactly N productions
          // - Real N (0 <= N <= 1): execute round(N * numProds) productions
          if (cv.isReal()) {
            double realVal = cv.real();
            if (realVal >= 0.0 && realVal <= 1.0) {
              // Treat as ratio of productions to execute
              joinCount = static_cast<int64_t>(
                  std::round(realVal * static_cast<double>(rule.prods.size())));
            } else {
              // Real > 1 treated as integer count
              joinCount = static_cast<int64_t>(std::round(realVal));
            }
          } else if (cv.isInteger()) {
            auto maybeCount = cv.integer().as<int64_t>();
            if (maybeCount)
              joinCount = *maybeCount;
          }
        }
      }
      // Handle edge cases per IEEE 1800-2017 Section 18.17
      // N=0 or no productions: do nothing
      if (joinCount <= 0 || rule.prods.empty()) {
        // Nothing to execute
      } else if (joinCount >= static_cast<int64_t>(rule.prods.size())) {
        joinCount = static_cast<int64_t>(rule.prods.size());
      }

      if (joinCount == 1) {
        auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
        Value maxVal = moore::ConstantOp::create(
            builder, loc, i32Ty,
            static_cast<int64_t>(rule.prods.size() - 1));
        Value randVal =
            moore::UrandomRangeBIOp::create(builder, loc, maxVal, Value{});

        auto &exitBlock = createBlock();
        SmallVector<Block *> itemBlocks;
        for (size_t i = 0; i < rule.prods.size(); ++i)
          itemBlocks.push_back(&createBlock());

        for (size_t i = 0; i < rule.prods.size(); ++i) {
          auto idxConst = moore::ConstantOp::create(
              builder, loc, i32Ty, static_cast<int64_t>(i));
          Value cond =
              moore::CaseEqOp::create(builder, loc, randVal, idxConst);
          cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
          if (i == rule.prods.size() - 1) {
            cf::BranchOp::create(builder, loc, itemBlocks[i]);
          } else {
            auto &nextCheckBlock = createBlock();
            cf::CondBranchOp::create(builder, loc, cond, itemBlocks[i],
                                     &nextCheckBlock);
            builder.setInsertionPointToEnd(&nextCheckBlock);
          }
        }

        for (size_t i = 0; i < rule.prods.size(); ++i) {
          builder.setInsertionPointToEnd(itemBlocks[i]);
          context.randSequenceReturnStack.push_back(&exitBlock);
          auto prodGuard = llvm::make_scope_exit([&] {
            context.randSequenceReturnStack.pop_back();
          });
          if (failed(executeProdBase(*rule.prods[i])))
            return failure();
          if (!isTerminated())
            cf::BranchOp::create(builder, loc, &exitBlock);
        }

        if (exitBlock.hasNoPredecessors()) {
          exitBlock.erase();
          setTerminated();
        } else {
          builder.setInsertionPointToEnd(&exitBlock);
        }
      } else if (joinCount > 1) {
        // randjoin(N) where 1 < N <= number of productions
        // Select N distinct productions using Fisher-Yates partial shuffle
        auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
        size_t numProds = rule.prods.size();

        SmallVector<Value> indexVars;
        for (size_t i = 0; i < numProds; ++i) {
          Value initVal = moore::ConstantOp::create(builder, loc, i32Ty,
                                                    static_cast<int64_t>(i));
          auto var = moore::VariableOp::create(
              builder, loc, moore::RefType::get(i32Ty),
              builder.getStringAttr("rj_idx_" + std::to_string(i)), initVal);
          indexVars.push_back(var);
        }

        for (int64_t i = 0; i < joinCount; ++i) {
          Value minVal =
              moore::ConstantOp::create(builder, loc, i32Ty, i);
          Value maxVal = moore::ConstantOp::create(
              builder, loc, i32Ty, static_cast<int64_t>(numProds - 1));
          Value randIdx =
              moore::UrandomRangeBIOp::create(builder, loc, maxVal, minVal);

          Value valAtI = moore::ReadOp::create(builder, loc, indexVars[i]);

          for (size_t j = i; j < numProds; ++j) {
            auto jConst = moore::ConstantOp::create(builder, loc, i32Ty,
                                                    static_cast<int64_t>(j));
            Value cond = moore::CaseEqOp::create(builder, loc, randIdx, jConst);
            cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

            auto &swapBlock = createBlock();
            auto &nextBlock = createBlock();
            cf::CondBranchOp::create(builder, loc, cond, &swapBlock, &nextBlock);

            builder.setInsertionPointToEnd(&swapBlock);
            Value valAtJ = moore::ReadOp::create(builder, loc, indexVars[j]);
            moore::BlockingAssignOp::create(builder, loc, indexVars[i], valAtJ);
            moore::BlockingAssignOp::create(builder, loc, indexVars[j], valAtI);
            cf::BranchOp::create(builder, loc, &nextBlock);

            builder.setInsertionPointToEnd(&nextBlock);
          }
        }

        for (int64_t i = 0; i < joinCount; ++i) {
          Value selectedIdx = moore::ReadOp::create(builder, loc, indexVars[i]);

          SmallVector<Block *> prodBlocks;
          for (size_t p = 0; p < numProds; ++p)
            prodBlocks.push_back(&createBlock());
          auto &afterProdBlock = createBlock();

          for (size_t p = 0; p < numProds; ++p) {
            auto pConst = moore::ConstantOp::create(builder, loc, i32Ty,
                                                    static_cast<int64_t>(p));
            Value cond =
                moore::CaseEqOp::create(builder, loc, selectedIdx, pConst);
            cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

            if (p == numProds - 1) {
              cf::BranchOp::create(builder, loc, prodBlocks[p]);
            } else {
              auto &nextCheckBlock = createBlock();
              cf::CondBranchOp::create(builder, loc, cond, prodBlocks[p],
                                       &nextCheckBlock);
              builder.setInsertionPointToEnd(&nextCheckBlock);
            }
          }

          for (size_t p = 0; p < numProds; ++p) {
            builder.setInsertionPointToEnd(prodBlocks[p]);
            context.randSequenceReturnStack.push_back(&afterProdBlock);
            auto prodGuard = llvm::make_scope_exit([&] {
              context.randSequenceReturnStack.pop_back();
            });
            if (failed(executeProdBase(*rule.prods[p])))
              return failure();
            if (!isTerminated())
              cf::BranchOp::create(builder, loc, &afterProdBlock);
          }

          if (afterProdBlock.hasNoPredecessors()) {
            afterProdBlock.erase();
            if (isTerminated())
              break;
          } else {
            builder.setInsertionPointToEnd(&afterProdBlock);
          }
        }
      }
    } else {
      for (auto prod : rule.prods) {
        Block &prodExit = createBlock();
        context.randSequenceReturnStack.push_back(&prodExit);
        auto prodGuard = llvm::make_scope_exit([&] {
          context.randSequenceReturnStack.pop_back();
        });

        if (failed(executeProdBase(*prod)))
          return failure();
        if (!isTerminated())
          cf::BranchOp::create(builder, loc, &prodExit);
        if (prodExit.hasNoPredecessors()) {
          prodExit.erase();
          if (isTerminated())
            break;
        } else {
          builder.setInsertionPointToEnd(&prodExit);
        }
      }
    }
    return success();
  }

  LogicalResult executeProdBase(
      const slang::ast::RandSeqProductionSymbol::ProdBase &prodBase) {
    using ProdKind = slang::ast::RandSeqProductionSymbol::ProdKind;
    switch (prodBase.kind) {
    case ProdKind::Item:
      return executeProdItem(
          prodBase.as<slang::ast::RandSeqProductionSymbol::ProdItem>());
    case ProdKind::CodeBlock: {
      // For code blocks in randsequence, we need to set up a break target
      // so that 'break' exits the code block per IEEE 1800-2017 Section 18.17
      Block &breakBlock = createBlock();
      context.randSequenceBreakStack.push_back(&breakBlock);
      auto breakGuard = llvm::make_scope_exit(
          [&] { context.randSequenceBreakStack.pop_back(); });

      auto result = convertStatementBlock(
          *prodBase.as<slang::ast::RandSeqProductionSymbol::CodeBlockProd>()
               .block);
      if (failed(result))
        return failure();

      // If not terminated, branch to break block to continue
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &breakBlock);

      // Set up the break block as the continuation point
      if (breakBlock.hasNoPredecessors()) {
        breakBlock.erase();
        // Already terminated, nothing to do
      } else {
        builder.setInsertionPointToEnd(&breakBlock);
      }
      return success();
    }
    case ProdKind::IfElse:
      return executeIfElseProd(
          prodBase.as<slang::ast::RandSeqProductionSymbol::IfElseProd>());
    case ProdKind::Repeat:
      return executeRepeatProd(
          prodBase.as<slang::ast::RandSeqProductionSymbol::RepeatProd>());
    case ProdKind::Case:
      return executeCaseProd(
          prodBase.as<slang::ast::RandSeqProductionSymbol::CaseProd>());
    }
    return success();
  }

  LogicalResult executeProdItem(
      const slang::ast::RandSeqProductionSymbol::ProdItem &item) {
    if (!item.target)
      return success();

    Context::ValueSymbolScope argScope(context.valueSymbols);
    auto args = item.args;
    auto formals = item.target->arguments;
    if (args.size() > formals.size()) {
      mlir::emitError(loc) << "too many arguments in randsequence production";
      return failure();
    }
    for (size_t i = 0; i < formals.size(); ++i) {
      auto *formal = formals[i];
      if (formal->direction != slang::ast::ArgumentDirection::In) {
        mlir::emitError(loc)
            << "randsequence production arguments must be input-only";
        return failure();
      }
      const slang::ast::Expression *argExpr = nullptr;
      if (i < args.size())
        argExpr = args[i];
      else
        argExpr = formal->getDefaultValue();
      if (!argExpr) {
        mlir::emitError(loc)
            << "missing argument for randsequence production '"
            << item.target->name << "'";
        return failure();
      }
      auto argType = context.convertType(*formal->getDeclaredType());
      if (!argType)
        return failure();
      auto argValue = context.convertRvalueExpression(*argExpr, argType);
      if (!argValue)
        return failure();

      auto refTy = moore::RefType::get(cast<moore::UnpackedType>(argType));
      auto var = moore::VariableOp::create(
          builder, loc, refTy, builder.getStringAttr(formal->name), argValue);
      context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                           formal, var);
    }

    return executeProduction(*item.target);
  }

  LogicalResult executeIfElseProd(
      const slang::ast::RandSeqProductionSymbol::IfElseProd &ifElse) {
    auto condValue = context.convertRvalueExpression(*ifElse.expr);
    if (!condValue)
      return failure();
    condValue = context.convertToBool(condValue);
    if (!condValue)
      return failure();
    condValue = moore::ToBuiltinBoolOp::create(builder, loc, condValue);

    Block &exitBlock = createBlock();
    Block &trueBlock = createBlock();
    Block *falseBlock = ifElse.elseItem ? &createBlock() : nullptr;
    cf::CondBranchOp::create(builder, loc, condValue, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    builder.setInsertionPointToEnd(&trueBlock);
    if (failed(executeProdItem(ifElse.ifItem)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    if (ifElse.elseItem) {
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(executeProdItem(*ifElse.elseItem)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult executeRepeatProd(
      const slang::ast::RandSeqProductionSymbol::RepeatProd &repeat) {
    auto cv = context.evaluateConstant(*repeat.expr);
    if (cv.bad()) {
      mlir::emitError(loc) << "randsequence repeat count must be constant";
      return failure();
    }
    auto maybeCount = cv.integer().as<int64_t>();
    if (!maybeCount)
      return success();
    int64_t count = *maybeCount;
    if (count <= 0)
      return success();

    auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
    Value countVal = moore::ConstantOp::create(builder, loc, i32Ty, count);
    Value initVal = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    auto counter = moore::VariableOp::create(
        builder, loc, moore::RefType::get(i32Ty),
        builder.getStringAttr("rs_idx"), initVal);

    Block &condBlock = createBlock();
    Block &bodyBlock = createBlock();
    Block &stepBlock = createBlock();
    Block &exitBlock = createBlock();

    cf::BranchOp::create(builder, loc, &condBlock);

    builder.setInsertionPointToEnd(&condBlock);
    Value idxVal = moore::ReadOp::create(builder, loc, counter);
    Value cond = moore::UltOp::create(builder, loc, idxVal, countVal);
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(executeProdItem(repeat.item)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    builder.setInsertionPointToEnd(&stepBlock);
    Value nextVal = moore::AddOp::create(
        builder, loc, idxVal,
        moore::ConstantOp::create(builder, loc, i32Ty, 1));
    moore::BlockingAssignOp::create(builder, loc, counter, nextVal);
    cf::BranchOp::create(builder, loc, &condBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult executeCaseProd(
      const slang::ast::RandSeqProductionSymbol::CaseProd &caseProd) {
    auto caseExpr = context.convertRvalueExpression(*caseProd.expr);
    if (!caseExpr)
      return failure();

    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;

    for (const auto &item : caseProd.items) {
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;

      for (const auto *expr : item.expressions) {
        auto value = context.convertRvalueExpression(*expr);
        if (!value)
          return failure();
        auto caseExprSBV = context.convertToSimpleBitVector(caseExpr);
        auto valueSBV = context.convertToSimpleBitVector(value);
        if (!caseExprSBV || !valueSBV)
          return failure();
        Value cond =
            moore::CaseEqOp::create(builder, loc, caseExprSBV, valueSBV);
        cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

        auto &nextBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, cond, &matchBlock, &nextBlock);
        builder.setInsertionPointToEnd(&nextBlock);
      }

      matchBlock.moveBefore(builder.getInsertionBlock());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(executeProdItem(item.item)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    if (caseProd.defaultItem) {
      if (failed(executeProdItem(*caseProd.defaultItem)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    } else {
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult
  convertStatementBlock(const slang::ast::StatementBlockSymbol &block) {
    auto savedScope = context.currentScope;
    context.currentScope = &block;
    auto scopeGuard =
        llvm::make_scope_exit([&] { context.currentScope = savedScope; });

    slang::ast::ASTContext astContext(block,
                                      slang::ast::LookupLocation::max);
    slang::ast::Statement::StatementContext stmtCtx(astContext);
    const auto &stmt = block.getStatement(astContext, stmtCtx);
    return context.convertStatement(stmt);
  }

  /// Emit an error for all other statements.
  template <typename T>
  LogicalResult visit(T &&stmt) {
    mlir::emitError(loc, "unsupported statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }

  LogicalResult visitInvalid(const slang::ast::Statement &stmt) {
    mlir::emitError(loc, "invalid statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }
};
} // namespace

LogicalResult Context::convertStatement(const slang::ast::Statement &stmt) {
  assert(builder.getInsertionBlock());
  auto loc = convertLocation(stmt.sourceRange);
  return stmt.visit(StmtVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
