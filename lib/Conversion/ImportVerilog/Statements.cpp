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
#include "slang/ast/Patterns.h"
#include "slang/ast/Statement.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/statements/MiscStatements.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/syntax/AllSyntax.h"
#include <string>
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {

/// Helper to ensure both operands have the same type before creating a binary
/// operation. If one operand is two-valued (bit) and the other is four-valued
/// (logic), promotes both to four-valued to ensure type compatibility.
static std::pair<Value, Value> unifyBoolTypes(OpBuilder &builder, Location loc,
                                              Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return {lhs, rhs};
  if (lhs.getType() == rhs.getType())
    return {lhs, rhs};

  auto lhsInt = dyn_cast<moore::IntType>(lhs.getType());
  auto rhsInt = dyn_cast<moore::IntType>(rhs.getType());
  if (!lhsInt || !rhsInt)
    return {lhs, rhs};

  // Use four-valued if either operand is four-valued.
  auto targetDomain = (lhsInt.getDomain() == moore::Domain::FourValued ||
                       rhsInt.getDomain() == moore::Domain::FourValued)
                          ? moore::Domain::FourValued
                          : moore::Domain::TwoValued;
  auto targetType = moore::IntType::get(builder.getContext(), 1, targetDomain);
  if (lhs.getType() != targetType)
    lhs = moore::ConversionOp::create(builder, loc, targetType, lhs);
  if (rhs.getType() != targetType)
    rhs = moore::ConversionOp::create(builder, loc, targetType, rhs);
  return {lhs, rhs};
}

/// Create an AndOp with type-unified operands.
static Value createUnifiedAndOp(OpBuilder &builder, Location loc, Value lhs,
                                Value rhs) {
  auto [unifiedLhs, unifiedRhs] = unifyBoolTypes(builder, loc, lhs, rhs);
  return moore::AndOp::create(builder, loc, unifiedLhs, unifiedRhs);
}

/// Create an OrOp with type-unified operands.
static Value createUnifiedOrOp(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs) {
  auto [unifiedLhs, unifiedRhs] = unifyBoolTypes(builder, loc, lhs, rhs);
  return moore::OrOp::create(builder, loc, unifiedLhs, unifiedRhs);
}

static verif::ClockEdge
convertEdgeKindVerif(const slang::ast::EdgeKind edge) {
  switch (edge) {
  case slang::ast::EdgeKind::PosEdge:
    return verif::ClockEdge::Pos;
  case slang::ast::EdgeKind::NegEdge:
    return verif::ClockEdge::Neg;
  case slang::ast::EdgeKind::BothEdges:
    return verif::ClockEdge::Both;
  case slang::ast::EdgeKind::None:
  default:
    return verif::ClockEdge::Both;
  }
}

static const slang::ast::SignalEventControl *
getCanonicalAssertionClockSignalEvent(const slang::ast::TimingControl &ctrl) {
  if (auto *signalCtrl = ctrl.as_if<slang::ast::SignalEventControl>()) {
    auto *symRef = signalCtrl->expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      return getCanonicalAssertionClockSignalEvent(clockingBlock.getEvent());
    }
    return signalCtrl;
  }
  if (auto *eventList = ctrl.as_if<slang::ast::EventListControl>()) {
    if (eventList->events.size() != 1)
      return nullptr;
    auto *event = *eventList->events.begin();
    if (!event)
      return nullptr;
    return getCanonicalAssertionClockSignalEvent(*event);
  }
  return nullptr;
}

static bool isHoistableAssertionOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (isPure(op))
    return true;
  return isa<moore::ReadOp>(op);
}

static Value cloneAssertionValueIntoBlock(Value value, OpBuilder &builder,
                                          Block *destBlock,
                                          IRMapping &mapping,
                                          llvm::DenseSet<Operation *> &active) {
  if (!value)
    return {};
  if (mapping.contains(value))
    return mapping.lookup(value);
  if (auto *defOp = value.getDefiningOp();
      defOp && defOp->getBlock() == destBlock)
    return value;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == destBlock)
      return arg;
    Value merged;
    auto *owner = arg.getOwner();
    for (auto *pred : owner->getPredecessors()) {
      Value incoming;
      Operation *term = pred->getTerminator();
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        if (br.getDest() == owner)
          incoming = br.getDestOperands()[arg.getArgNumber()];
      } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        if (condBr.getTrueDest() == owner)
          incoming = condBr.getTrueDestOperands()[arg.getArgNumber()];
        else if (condBr.getFalseDest() == owner)
          incoming = condBr.getFalseDestOperands()[arg.getArgNumber()];
      }
      if (!incoming)
        return {};
      auto cloned = cloneAssertionValueIntoBlock(incoming, builder, destBlock,
                                                 mapping, active);
      if (!cloned)
        return {};
      if (merged)
        merged = createUnifiedOrOp(builder, builder.getUnknownLoc(), merged,
                                   cloned);
      else
        merged = cloned;
    }
    return merged;
  }
  auto *defOp = value.getDefiningOp();
  if (!defOp || !isHoistableAssertionOp(defOp))
    return {};
  if (!active.insert(defOp).second)
    return {};

  for (auto operand : defOp->getOperands()) {
    auto cloned = cloneAssertionValueIntoBlock(operand, builder, destBlock,
                                               mapping, active);
    if (!cloned)
      return {};
  }

  OpBuilder::InsertionGuard guard(builder);
  if (destBlock->mightHaveTerminator()) {
    if (auto *terminator = destBlock->getTerminator())
      builder.setInsertionPoint(terminator);
    else
      builder.setInsertionPointToEnd(destBlock);
  } else {
    builder.setInsertionPointToEnd(destBlock);
  }
  builder.clone(*defOp, mapping);
  active.erase(defOp);
  return mapping.lookup(value);
}

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

  Block *lookupDisableTarget(const slang::ast::Symbol *symbol) const {
    if (!symbol)
      return nullptr;
    for (auto it = context.disableStack.rbegin();
         it != context.disableStack.rend(); ++it) {
      if (it->symbol == symbol)
        return it->targetBlock;
    }
    return nullptr;
  }

  moore::GlobalVariableOp getOrCreateProceduralAssertionsEnabledGlobal() {
    if (context.proceduralAssertionsEnabledGlobal)
      return context.proceduralAssertionsEnabledGlobal;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(context.intoModuleOp.getBody());

    std::string baseName = "__circt_proc_assertions_enabled";
    std::string symName = baseName;
    unsigned suffix = 0;
    while (context.symbolTable.lookup(symName))
      symName = baseName + "_" + std::to_string(++suffix);

    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    auto globalOp = moore::GlobalVariableOp::create(
        builder, loc, builder.getStringAttr(symName), i1Ty);

    auto &initBlock = globalOp.getInitRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&initBlock);
    auto enabled = moore::ConstantOp::create(builder, loc, i1Ty, 1);
    moore::YieldOp::create(builder, loc, enabled);

    context.proceduralAssertionsEnabledGlobal = globalOp;
    return globalOp;
  }

  Value readProceduralAssertionsEnabled() {
    auto globalOp = getOrCreateProceduralAssertionsEnabledGlobal();
    auto globalRef = moore::GetGlobalVariableOp::create(builder, loc, globalOp);
    return moore::ReadOp::create(builder, loc, globalRef);
  }

  LogicalResult writeProceduralAssertionsEnabled(Value enabled) {
    auto globalOp = getOrCreateProceduralAssertionsEnabledGlobal();
    auto targetType = globalOp.getType();
    if (enabled.getType() != targetType)
      enabled = moore::ConversionOp::create(builder, loc, targetType, enabled);
    auto globalRef = moore::GetGlobalVariableOp::create(builder, loc, globalOp);
    moore::BlockingAssignOp::create(builder, loc, globalRef, enabled);
    return success();
  }

  moore::GlobalVariableOp getOrCreateAssertionFailMessagesEnabledGlobal() {
    if (context.assertionFailMessagesEnabledGlobal)
      return context.assertionFailMessagesEnabledGlobal;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(context.intoModuleOp.getBody());

    std::string baseName = "__circt_assert_fail_msgs_enabled";
    std::string symName = baseName;
    unsigned suffix = 0;
    while (context.symbolTable.lookup(symName))
      symName = baseName + "_" + std::to_string(++suffix);

    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    auto globalOp = moore::GlobalVariableOp::create(
        builder, loc, builder.getStringAttr(symName), i1Ty);

    auto &initBlock = globalOp.getInitRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&initBlock);
    auto enabled = moore::ConstantOp::create(builder, loc, i1Ty, 1);
    moore::YieldOp::create(builder, loc, enabled);

    context.assertionFailMessagesEnabledGlobal = globalOp;
    return globalOp;
  }

  Value readAssertionFailMessagesEnabled() {
    auto globalOp = getOrCreateAssertionFailMessagesEnabledGlobal();
    auto globalRef = moore::GetGlobalVariableOp::create(builder, loc, globalOp);
    return moore::ReadOp::create(builder, loc, globalRef);
  }

  LogicalResult writeAssertionFailMessagesEnabled(Value enabled) {
    auto globalOp = getOrCreateAssertionFailMessagesEnabledGlobal();
    auto targetType = globalOp.getType();
    if (enabled.getType() != targetType)
      enabled = moore::ConversionOp::create(builder, loc, targetType, enabled);
    auto globalRef = moore::GetGlobalVariableOp::create(builder, loc, globalOp);
    moore::BlockingAssignOp::create(builder, loc, globalRef, enabled);
    return success();
  }

  Value selectBool(Value cond, Value ifTrue, Value ifFalse) {
    if (ifTrue.getType() != ifFalse.getType())
      ifFalse = moore::ConversionOp::create(builder, loc, ifTrue.getType(), ifFalse);
    cond = context.convertToBool(cond);
    if (!cond)
      return {};
    auto conditional =
        moore::ConditionalOp::create(builder, loc, ifTrue.getType(), cond);
    auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
    auto &falseBlock = conditional.getFalseRegion().emplaceBlock();

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&trueBlock);
    moore::YieldOp::create(builder, loc, ifTrue);
    builder.setInsertionPointToStart(&falseBlock);
    moore::YieldOp::create(builder, loc, ifFalse);
    return conditional.getResult();
  }

  /// Handle foreach loops for queues and dynamic arrays.
  /// Uses size-based iteration (for i=0 to size-1).
  LogicalResult recursiveForeachQueue(
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

    auto intType = dyn_cast<moore::IntType>(iterType);
    if (!intType) {
      mlir::emitError(loc)
          << "queue foreach iterator must have integer type, but got "
          << iterType;
      return failure();
    }

    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    auto done = llvm::make_scope_exit([&] { context.loopStack.pop_back(); });

    // Initialize iterator to 0
    Value initial = moore::ConstantOp::create(builder, loc, intType, 0);

    // Create loop variable
    Value varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(intType)),
        builder.getStringAttr(iter->name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, varOp);

    cf::BranchOp::create(builder, loc, &checkBlock);
    builder.setInsertionPointToEnd(&checkBlock);

    // Get array size and check if iterator < size
    Value arrayVal = moore::ReadOp::create(builder, loc, arrayRef);
    Value arraySize = moore::ArraySizeOp::create(builder, loc, arrayVal);
    Value var = moore::ReadOp::create(builder, loc, varOp);
    Value cond = moore::SltOp::create(builder, loc, var, arraySize);
    if (!cond)
      return failure();
    cond = context.convertToBool(cond);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Body block
    builder.setInsertionPointToEnd(&bodyBlock);

    // Find next dimension in this foreach statement
    bool hasNext = false;
    for (uint32_t nextLevel = level + 1; nextLevel < stmt.loopDims.size();
         nextLevel++) {
      if (stmt.loopDims[nextLevel].loopVar) {
        if (!stmt.loopDims[nextLevel].range.has_value()) {
          // Check if the array type at this level is a queue or dynamic array
          auto arrayType = arrayRef.getType();
          if (auto refType = dyn_cast<moore::RefType>(arrayType))
            arrayType = refType.getNestedType();
          if (isa<moore::QueueType, moore::OpenUnpackedArrayType>(arrayType)) {
            if (failed(recursiveForeachQueue(stmt, nextLevel)))
              return failure();
          } else {
            if (failed(recursiveForeachDynamic(stmt, nextLevel)))
              return failure();
          }
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
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Step block: increment iterator
    builder.setInsertionPointToEnd(&stepBlock);
    Value currentVar = moore::ReadOp::create(builder, loc, varOp);
    Value one = moore::ConstantOp::create(builder, loc, intType, 1);
    Value nextVar = moore::AddOp::create(builder, loc, currentVar, one);
    moore::BlockingAssignOp::create(builder, loc, varOp, nextVar);
    cf::BranchOp::create(builder, loc, &checkBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
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
      // Dynamic loop variable - check if it's a queue or associative array
      // to determine the iteration pattern.
      auto arrayRef = context.convertLvalueExpression(stmt.arrayRef);
      if (!arrayRef)
        return failure();
      auto arrayType = arrayRef.getType();
      // Unwrap RefType if present
      if (auto refType = dyn_cast<moore::RefType>(arrayType))
        arrayType = refType.getNestedType();

      // Queues and dynamic arrays use size-based iteration (for i=0 to size-1)
      // Associative arrays use first/next iteration pattern
      bool isQueueOrDynArray =
          isa<moore::QueueType, moore::OpenUnpackedArrayType>(arrayType);
      if (isQueueOrDynArray) {
        return recursiveForeachQueue(stmt, level);
      }
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

  // Handle `begin ... end` blocks and `fork ... join` blocks.
  LogicalResult visit(const slang::ast::BlockStatement &stmt) {
    // For sequential blocks, inline unnamed blocks into the parent.
    // Named blocks get explicit exit blocks so `disable <name>` can branch
    // directly to the correct continuation point.
    if (stmt.blockKind == slang::ast::StatementBlockKind::Sequential) {
      auto *blockSym = stmt.blockSymbol;
      if (!blockSym || blockSym->name.empty())
        return context.convertStatement(stmt.body);
      // Module bodies are single-block regions. Labeled module-level
      // concurrent assertions can arrive here wrapped in a named block
      // (`label: assert property ...`), but must not introduce CFG blocks.
      // Keep these linear and rely on statement lowering to emit the assertion.
      if (builder.getInsertionBlock() &&
          isa<moore::SVModuleOp>(builder.getInsertionBlock()->getParentOp()))
        return context.convertStatement(stmt.body);

      auto &exitBlock = createBlock();
      context.disableStack.push_back({blockSym, &exitBlock});
      auto done = llvm::make_scope_exit([&] { context.disableStack.pop_back(); });

      if (failed(context.convertStatement(stmt.body)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);

      if (exitBlock.hasNoPredecessors()) {
        exitBlock.erase();
        setTerminated();
      } else {
        builder.setInsertionPointToEnd(&exitBlock);
      }
      return success();
    }

    // For fork blocks (join, join_any, join_none), create a ForkOp.
    // Each statement in the fork body becomes a separate parallel branch.

    // Determine the join type.
    moore::JoinType joinType;
    switch (stmt.blockKind) {
    case slang::ast::StatementBlockKind::JoinAll:
      joinType = moore::JoinType::JoinAll;
      break;
    case slang::ast::StatementBlockKind::JoinAny:
      joinType = moore::JoinType::JoinAny;
      break;
    case slang::ast::StatementBlockKind::JoinNone:
      joinType = moore::JoinType::JoinNone;
      break;
    default:
      llvm_unreachable("unexpected block kind");
    }

    // Collect the statements that will become parallel branches.
    // The body is typically a StatementList, but could be a single statement.
    SmallVector<const slang::ast::Statement *> forkStmts;
    const slang::ast::Statement *bodyStmt = &stmt.body;
    if (auto *stmtList = bodyStmt->as_if<slang::ast::StatementList>()) {
      for (auto *s : stmtList->list)
        forkStmts.push_back(s);
    } else {
      // Single statement fork (unusual but valid).
      forkStmts.push_back(bodyStmt);
    }

    // Get optional block name.
    mlir::StringAttr nameAttr;
    if (stmt.blockSymbol && !stmt.blockSymbol->name.empty())
      nameAttr = builder.getStringAttr(stmt.blockSymbol->name);

    struct ForkBranch {
      llvm::SmallVector<const slang::ast::VariableDeclStatement *, 4> decls;
      const slang::ast::Statement *stmt = nullptr;
      // Pre-computed initializer values for automatic variables. These are
      // evaluated at fork creation time to capture the current values of
      // outer scope variables (like loop counters).
      llvm::DenseMap<const slang::ast::VariableSymbol *, Value> capturedInits;
    };
    llvm::SmallVector<ForkBranch, 4> branches;
    llvm::SmallVector<const slang::ast::VariableDeclStatement *, 4> pendingDecls;
    for (auto *s : forkStmts) {
      if (auto *decl = s->as_if<slang::ast::VariableDeclStatement>()) {
        pendingDecls.push_back(decl);
        continue;
      }
      ForkBranch branch;
      branch.decls = pendingDecls;
      branch.stmt = s;
      branches.push_back(std::move(branch));
      pendingDecls.clear();
    }

    if (branches.empty()) {
      for (auto *decl : pendingDecls) {
        if (failed(visit(*decl)))
          return failure();
      }
      return success();
    }

    Context::ValueSymbolScope forkScope(context.valueSymbols);

    // Pre-compute initializer values for automatic variable declarations
    // BEFORE creating the ForkOp. This ensures that references to outer scope
    // variables (like loop counters) capture their current values at fork
    // creation time, not when the fork branch executes.
    //
    // For example, in:
    //   for (int i = 1; i <= 3; i++) begin
    //     fork
    //       automatic int local_i = i;  // Should capture i's current value
    //       begin #(local_i * 100); end
    //     join_none
    //   end
    //
    // Each fork iteration should capture a different value of `i`.
    for (size_t branchIdx = 0; branchIdx < branches.size(); ++branchIdx) {
      auto &branch = branches[branchIdx];
      for (auto *decl : branch.decls) {
        const auto &var = decl->symbol;
        if (const auto *init = var.getInitializer()) {
          auto type = context.convertType(*var.getDeclaredType());
          if (!type)
            return failure();
          auto initialValue = context.convertRvalueExpression(*init, type);
          if (!initialValue)
            return failure();
          branch.capturedInits[&var] = initialValue;
          LLVM_DEBUG(llvm::dbgs() << "Fork capture: stored init for var '"
                                  << var.name << "' at " << &var
                                  << " in branch " << branchIdx << "\n");
        }
      }
    }

    // Create the ForkOp with the appropriate number of branches.
    auto forkOp = moore::ForkOp::create(builder, loc, joinType, nameAttr,
                                        branches.size());

    // Populate each branch region with its corresponding statement.
    for (size_t i = 0; i < branches.size(); ++i) {
      auto &region = forkOp.getBranches()[i];
      // Create entry block if not present.
      if (region.empty())
        region.emplaceBlock();

      // Set insertion point to the branch region and convert the statement.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&region.front());

      Context::ValueSymbolScope branchScope(context.valueSymbols);
      for (auto *decl : branches[i].decls) {
        // Use pre-captured initializer values for automatic variables.
        LLVM_DEBUG(llvm::dbgs() << "Fork lookup: checking var '"
                                << decl->symbol.name << "' at " << &decl->symbol
                                << " in branch " << i << ", map size="
                                << branches[i].capturedInits.size() << "\n");
        auto it = branches[i].capturedInits.find(&decl->symbol);
        if (it != branches[i].capturedInits.end()) {
          LLVM_DEBUG(llvm::dbgs() << "Fork lookup: FOUND, using captured init\n");
          if (failed(declareVariableWithInit(decl->symbol, it->second)))
            return failure();
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Fork lookup: NOT FOUND, using regular decl\n");
          if (failed(visit(*decl)))
            return failure();
        }
      }

      if (failed(context.convertStatement(*branches[i].stmt)))
        return failure();

      // After converting the statement, check all blocks in the region.
      // Add terminators to blocks that don't have one.
      // With control flow (cf.br), some blocks may already be terminated.
      for (auto &block : region) {
        builder.setInsertionPointToEnd(&block);
        if (block.empty() ||
            !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
          moore::ForkTerminatorOp::create(builder, loc);
      }
    }

    return success();
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
      StringRef swriteName = call->getSubroutineName();
      moore::IntFormat swriteFormat = moore::IntFormat::Decimal;
      bool isSWrite = false;
      if (swriteName == "$swrite") {
        isSWrite = true;
      } else if (swriteName == "$swriteb") {
        isSWrite = true;
        swriteFormat = moore::IntFormat::Binary;
      } else if (swriteName == "$swriteo") {
        isSWrite = true;
        swriteFormat = moore::IntFormat::Octal;
      } else if (swriteName == "$swriteh") {
        isSWrite = true;
        swriteFormat = moore::IntFormat::HexLower;
      }
      if (isSWrite) {

        // Use the first argument as the output location
        auto *lhsExpr = call->arguments().front();
        // Format the second and all later arguments as a string with default
        // formatting (no format string expected)
        auto fmtValue =
            context.convertFormatString(call->arguments().subspan(1), loc,
                                        swriteFormat, false);
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
    return declareVariable(stmt.symbol);
  }

  LogicalResult declareVariable(const slang::ast::VariableSymbol &var) {
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();
    // Check if this is a static function-local variable with an EXPLICIT
    // `static` keyword on the variable declaration. Static variables persist
    // across function calls and should be stored as global variables with
    // lazy initialization.
    //
    // We require an explicit `static` keyword because:
    // 1. In static functions, local variables are implicitly static, but many
    //    existing codebases (including test cases) expect them to be treated
    //    as local variables.
    // 2. The UVM pattern (uvm_component_registry::get()) uses explicit `static`
    //    to implement singletons.
    //
    // Only apply this to function-local variables (when currentFunctionLowering
    // is set), not to variables in module initial/always blocks.
    bool hasExplicitStaticKeyword = false;
    if (auto *syntax = var.getSyntax()) {
      // The variable's syntax is typically a DeclaratorSyntax. The parent
      // of the declarator is a DataDeclarationSyntax which contains the
      // modifiers (like 'static').
      const slang::syntax::SyntaxNode *checkSyntax = syntax;
      if (checkSyntax->parent)
        checkSyntax = checkSyntax->parent;
      if (auto *dataSyntax =
              checkSyntax->as_if<slang::syntax::DataDeclarationSyntax>()) {
        for (auto mod : dataSyntax->modifiers) {
          if (mod.kind == slang::parsing::TokenKind::StaticKeyword) {
            hasExplicitStaticKeyword = true;
            break;
          }
        }
      }
    }

    bool isStatic = var.lifetime == slang::ast::VariableLifetime::Static &&
                    context.currentFunctionLowering != nullptr &&
                    hasExplicitStaticKeyword;

    LLVM_DEBUG(llvm::dbgs() << "VarDecl: " << var.name << " type=" << type
                            << "\n");

    auto unpackedType = dyn_cast<moore::UnpackedType>(type);
    if (!unpackedType) {
      mlir::emitError(loc) << "variable '" << var.name
                           << "' has non-unpacked type: " << type;
      return failure();
    }

    if (isStatic) {
      // Static function-local variables are stored as global variables.
      // Check if already converted (handles multiple calls to the same function).
      if (context.globalVariables.count(&var)) {
        auto globalOp = context.globalVariables.lookup(&var);
        auto refTy = moore::RefType::get(unpackedType);
        auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
        auto ref =
            moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
        context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                             &var, ref);
        return success();
      }

      // Check by fully qualified name (handles multiple specializations).
      auto symName = fullyQualifiedSymbolName(context, var);
      if (context.globalVariablesByName.count(symName)) {
        // Reuse the existing global variable for this symbol pointer.
        context.globalVariables[&var] = context.globalVariablesByName[symName];
        auto globalOp = context.globalVariables.lookup(&var);
        auto refTy = moore::RefType::get(unpackedType);
        auto symRef = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
        auto ref =
            moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
        context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                             &var, ref);
        return success();
      }

      // Pick an insertion point for this variable at the module level.
      // Use a nested scope for InsertionGuard to restore insertion point
      // before we create the GetGlobalVariableOp reference.
      moore::GlobalVariableOp globalVarOp;
      {
        OpBuilder::InsertionGuard g(builder);
        auto it = context.orderedRootOps.upper_bound(var.location);
        if (it == context.orderedRootOps.end())
          builder.setInsertionPointToEnd(context.intoModuleOp.getBody());
        else
          builder.setInsertionPoint(it->second);

        // Create the global variable op.
        globalVarOp = moore::GlobalVariableOp::create(builder, loc, symName,
                                                      unpackedType);
        context.orderedRootOps.insert({var.location, globalVarOp});
        context.globalVariables[&var] = globalVarOp;
        context.globalVariablesByName[symName] = globalVarOp;

        // If the variable has an initializer expression, remember it for later.
        if (var.getInitializer()) {
          context.globalVariableWorklist.push_back(&var);
        } else {
          // No explicit initializer -- check for struct field defaults
          // (IEEE 1800-2017 §7.2.1). Build the init region and synthesize
          // the struct default inside it so the ops live in the right block.
          auto hasStructFieldDefaults = [&]() {
            const auto &canonical =
                var.getDeclaredType()->getType().getCanonicalType();
            const slang::ast::Scope *scope = nullptr;
            if (auto *p = canonical.as_if<slang::ast::PackedStructType>())
              scope = p;
            else if (auto *u =
                         canonical.as_if<slang::ast::UnpackedStructType>())
              scope = u;
            if (!scope)
              return false;
            for (auto &f : scope->membersOfType<slang::ast::FieldSymbol>())
              if (f.getInitializer())
                return true;
            return false;
          };
          if (hasStructFieldDefaults()) {
            auto &block = globalVarOp.getInitRegion().emplaceBlock();
            OpBuilder::InsertionGuard initGuard(builder);
            builder.setInsertionPointToEnd(&block);
            auto structDefault = context.synthesizeStructFieldDefaults(
                var.getDeclaredType()->getType(), unpackedType, loc);
            if (structDefault)
              moore::YieldOp::create(builder, loc, structDefault);
          }
        }
      }
      // InsertionGuard restores original insertion point here.

      auto refTy = moore::RefType::get(unpackedType);
      auto symRef = mlir::FlatSymbolRefAttr::get(globalVarOp.getSymNameAttr());
      auto ref =
          moore::GetGlobalVariableOp::create(builder, loc, refTy, symRef);
      context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                           &var, ref);
      return success();
    }

    // Non-static local variable - create a local VariableOp.
    Value initial;
    if (const auto *init = var.getInitializer()) {
      initial = context.convertRvalueExpression(*init, type);
      if (!initial)
        return failure();
    } else {
      // No explicit initializer -- check for struct field defaults
      // (IEEE 1800-2017 §7.2.1).
      initial = context.synthesizeStructFieldDefaults(
          var.getDeclaredType()->getType(), unpackedType, loc);
    }

    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(unpackedType),
        builder.getStringAttr(var.name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         &var, varOp);
    return success();
  }

  /// Declare a variable with a pre-computed initial value. This is used for
  /// automatic variables in fork blocks where the initializer must be evaluated
  /// at fork creation time to capture outer scope variable values.
  LogicalResult declareVariableWithInit(const slang::ast::VariableSymbol &var,
                                        Value initialValue) {
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();

    auto unpackedType = dyn_cast<moore::UnpackedType>(type);
    if (!unpackedType) {
      mlir::emitError(loc) << "variable '" << var.name
                           << "' has non-unpacked type: " << type;
      return failure();
    }

    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(unpackedType),
        builder.getStringAttr(var.name), initialValue);
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
        allConds = createUnifiedAndOp(builder, loc, allConds, cond);
      else
        allConds = cond;
    }
    assert(allConds && "slang guarantees at least one condition");
    Value allCondsLogic = allConds;
    allConds = moore::ToBuiltinBoolOp::create(builder, loc, allConds);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, allConds, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    Value savedAssertionGuard = context.currentAssertionGuard;

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    Value trueGuard = allCondsLogic;
    if (savedAssertionGuard)
      trueGuard = createUnifiedAndOp(builder, loc, savedAssertionGuard,
                                     trueGuard);
    context.currentAssertionGuard = trueGuard;
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();
    context.currentAssertionGuard = savedAssertionGuard;
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // Generate the false branch if present.
    if (stmt.ifFalse) {
      builder.setInsertionPointToEnd(falseBlock);
      Value falseGuard = moore::NotOp::create(builder, loc, allCondsLogic);
      if (savedAssertionGuard)
        falseGuard = createUnifiedAndOp(builder, loc, savedAssertionGuard,
                                        falseGuard);
      context.currentAssertionGuard = falseGuard;
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      context.currentAssertionGuard = savedAssertionGuard;
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
    Value savedAssertionGuard = context.currentAssertionGuard;
    Value anyMatch;

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
        cond = createUnifiedAndOp(builder, loc, cond, filter);
      }

      Value condLogic = cond;
      if (anyMatch)
        anyMatch = createUnifiedOrOp(builder, loc, anyMatch, condLogic);
      else
        anyMatch = condLogic;
      cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
      auto &nextBlock = createBlock();
      cf::CondBranchOp::create(builder, loc, cond, &matchBlock, &nextBlock);
      builder.setInsertionPointToEnd(&nextBlock);

      matchBlock.moveBefore(builder.getInsertionBlock());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      Value matchGuard = condLogic;
      if (savedAssertionGuard)
        matchGuard = createUnifiedAndOp(builder, loc, savedAssertionGuard,
                                        matchGuard);
      context.currentAssertionGuard = matchGuard;
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      context.currentAssertionGuard = savedAssertionGuard;
      if (!isTerminated()) {
        auto itemLoc = context.convertLocation(item.stmt->sourceRange);
        cf::BranchOp::create(builder, itemLoc, &exitBlock);
      }
    }

    if (stmt.defaultCase) {
      auto &defaultBlock = createBlock();
      cf::BranchOp::create(builder, loc, &defaultBlock);
      builder.setInsertionPointToEnd(&defaultBlock);
      Value defaultGuard = savedAssertionGuard;
      if (anyMatch) {
        Value noMatch = moore::NotOp::create(builder, loc, anyMatch);
        defaultGuard = defaultGuard
                           ? createUnifiedAndOp(builder, loc, defaultGuard,
                                                noMatch)
                           : noMatch;
      }
      context.currentAssertionGuard = defaultGuard;
      if (failed(context.convertStatement(*stmt.defaultCase)))
        return failure();
      context.currentAssertionGuard = savedAssertionGuard;
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
    using slang::ast::ExpressionKind;
    using slang::ast::TypeReferenceExpression;

    // Handle type reference case statements specially. These are compile-time
    // constructs where the case expression is a type() operator, and the items
    // are also type() expressions. We compare types at compile time and emit
    // only the matching branch.
    if (caseStmt.expr.kind == ExpressionKind::TypeReference) {
      const auto &condType =
          caseStmt.expr.as<TypeReferenceExpression>().targetType;

      // Find the matching branch by comparing types.
      const slang::ast::Statement *matchedStmt = nullptr;
      for (const auto &item : caseStmt.items) {
        for (const auto *expr : item.expressions) {
          if (expr->kind == ExpressionKind::TypeReference) {
            const auto &itemType =
                expr->as<TypeReferenceExpression>().targetType;
            if (condType.isMatching(itemType)) {
              matchedStmt = item.stmt;
              break;
            }
          }
        }
        if (matchedStmt)
          break;
      }

      // If no match found, use the default case.
      if (!matchedStmt && caseStmt.defaultCase)
        matchedStmt = caseStmt.defaultCase;

      // Emit the matched statement (or nothing if no match and no default).
      if (matchedStmt)
        return context.convertStatement(*matchedStmt);
      return success();
    }

    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    // Check each case individually. This currently ignores the `unique`,
    // `unique0`, and `priority` modifiers which would allow for additional
    // optimizations.
    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;
    SmallVector<moore::FVIntegerAttr> itemConsts;
    Value savedAssertionGuard = context.currentAssertionGuard;
    Value anyMatch;

    for (const auto &item : caseStmt.items) {
      // Create the block that will contain the main body of the expression.
      // This is where any of the comparisons will branch to if they match.
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;
      BlockArgument matchGuardArg;
      bool hasMatchGuardArg = false;

      // Snapshot the match status of previous items to preserve priority.
      Value anyMatchBeforeItem = anyMatch;
      Value itemMatch;

      // The SV standard requires expressions to be checked in the order
      // specified by the user, and for the evaluation to stop as soon as the
      // first matching expression is encountered.
      for (const auto *expr : item.expressions) {
        auto itemLoc = context.convertLocation(expr->sourceRange);

        // Generate the appropriate equality operator based on type.
        Value cond;
        Value condLogic;

        // For case inside, handle range expressions specially before
        // converting to rvalue.
        if (caseStmt.condition == CaseStatementCondition::Inside) {
          auto caseExprSBV = context.convertToSimpleBitVector(caseExpr);
          if (!caseExprSBV)
            return failure();

          if (const auto *openRange =
                  expr->as_if<slang::ast::ValueRangeExpression>()) {
            // Handle ranges: check if caseExpr is within [low, high].
            auto lowBound = context.convertToSimpleBitVector(
                context.convertRvalueExpression(openRange->left()));
            auto highBound = context.convertToSimpleBitVector(
                context.convertRvalueExpression(openRange->right()));
            if (!lowBound || !highBound)
              return failure();
            Value leftCmp, rightCmp;
            // Determine signedness for comparison.
            if (openRange->left().type->isSigned() ||
                caseStmt.expr.type->isSigned()) {
              leftCmp =
                  moore::SgeOp::create(builder, itemLoc, caseExprSBV, lowBound);
            } else {
              leftCmp =
                  moore::UgeOp::create(builder, itemLoc, caseExprSBV, lowBound);
            }
            if (openRange->right().type->isSigned() ||
                caseStmt.expr.type->isSigned()) {
              rightCmp =
                  moore::SleOp::create(builder, itemLoc, caseExprSBV, highBound);
            } else {
              rightCmp =
                  moore::UleOp::create(builder, itemLoc, caseExprSBV, highBound);
            }
            cond = moore::AndOp::create(builder, itemLoc, leftCmp, rightCmp);
          } else {
            // For non-range values, use wildcard equality (handles ?, X, Z).
            auto value = context.convertRvalueExpression(*expr);
            if (!value)
              return failure();
            auto valueSBV = context.convertToSimpleBitVector(value);
            if (!valueSBV)
              return failure();
            cond = moore::WildcardEqOp::create(builder, itemLoc, caseExprSBV,
                                               valueSBV);
          }
          condLogic = cond;
        } else {
          // Non-inside case statements.
          auto value = context.convertRvalueExpression(*expr);
          if (!value)
            return failure();
          itemLoc = value.getLoc();

          // Take note if the expression is a constant.
          auto maybeConst = value;
          while (isa_and_nonnull<moore::ConversionOp, moore::IntToLogicOp,
                                 moore::LogicToIntOp>(maybeConst.getDefiningOp()))
            maybeConst = maybeConst.getDefiningOp()->getOperand(0);
          if (auto defOp = maybeConst.getDefiningOp<moore::ConstantOp>())
            itemConsts.push_back(defOp.getValueAttr());

          if (isa<moore::StringType>(caseExpr.getType())) {
            // String case statement - use string comparison.
            cond = moore::StringCmpOp::create(
                builder, itemLoc, moore::StringCmpPredicate::eq, caseExpr, value);
            condLogic = context.convertToBool(cond);
            if (!condLogic)
              return failure();
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
              llvm_unreachable("Inside handled above");
            }
            condLogic = cond;
          }
        }

        cond = moore::ToBuiltinBoolOp::create(builder, itemLoc, condLogic);

        Value branchGuard = condLogic;
        if (anyMatchBeforeItem) {
          Value noPrevMatch =
              moore::NotOp::create(builder, itemLoc, anyMatchBeforeItem);
          branchGuard = createUnifiedAndOp(builder, itemLoc, noPrevMatch,
                                           branchGuard);
        }

        if (!hasMatchGuardArg) {
          matchGuardArg =
              matchBlock.addArgument(branchGuard.getType(), itemLoc);
          hasMatchGuardArg = true;
        }
        if (itemMatch)
          itemMatch =
              createUnifiedOrOp(builder, itemLoc, itemMatch, branchGuard);
        else
          itemMatch = branchGuard;

        // If the condition matches, branch to the match block. Otherwise
        // continue checking the next expression in a new block.
        auto &nextBlock = createBlock();
        mlir::cf::CondBranchOp::create(
            builder, itemLoc, cond, &matchBlock, ValueRange{branchGuard},
            &nextBlock, ValueRange{});
        builder.setInsertionPointToEnd(&nextBlock);
      }

      if (anyMatch)
        anyMatch = createUnifiedOrOp(builder, loc, anyMatch, itemMatch);
      else
        anyMatch = itemMatch;

      // The current block is the fall-through after all conditions have been
      // checked and nothing matched. Move the match block up before this point
      // to make the IR easier to read.
      matchBlock.moveBefore(builder.getInsertionBlock());

      // Generate the code for this item's statement in the match block.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      Value matchGuard = hasMatchGuardArg ? matchGuardArg : Value{};
      if (savedAssertionGuard)
        matchGuard = matchGuard
                         ? createUnifiedAndOp(builder, loc,
                                              savedAssertionGuard, matchGuard)
                         : savedAssertionGuard;
      context.currentAssertionGuard = matchGuard;
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      context.currentAssertionGuard = savedAssertionGuard;
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
      // If the last match block has a block argument (for tracking which
      // expression matched), we need to provide a constant true guard value
      // since this is the fallback path where no previous items matched.
      SmallVector<Value> branchArgs;
      if (lastMatchBlock->getNumArguments() > 0) {
        auto argType = cast<moore::IntType>(lastMatchBlock->getArgument(0).getType());
        auto trueGuard = moore::ConstantOp::create(builder, loc, argType, 1);
        branchArgs.push_back(trueGuard);
      }
      mlir::cf::BranchOp::create(builder, loc, lastMatchBlock, branchArgs);
    } else {
      // Generate the default case if present.
      Value defaultGuard = savedAssertionGuard;
      if (anyMatch) {
        Value noMatch = moore::NotOp::create(builder, loc, anyMatch);
        defaultGuard = defaultGuard
                           ? createUnifiedAndOp(builder, loc, defaultGuard,
                                                noMatch)
                           : noMatch;
      }
      context.currentAssertionGuard = defaultGuard;
      if (caseStmt.defaultCase)
        if (failed(context.convertStatement(*caseStmt.defaultCase)))
          return failure();
      context.currentAssertionGuard = savedAssertionGuard;
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
    case PatternKind::Variable:
      return matchVariablePattern(pattern.as<slang::ast::VariablePattern>(),
                                  value, targetType);
    case PatternKind::Structure:
      return matchStructurePattern(pattern.as<slang::ast::StructurePattern>(),
                                   value, targetType, condKind);
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

    // Ensure both operands have the same type (SameTypeOperands constraint).
    // Pattern matching may have mismatched bit widths if slang doesn't insert
    // implicit conversions (e.g., `if (x matches 42)` where x is 1-bit).
    auto lhsTy = cast<moore::IntType>(lhsSBV.getType());
    auto rhsTy = cast<moore::IntType>(rhsSBV.getType());
    if (lhsTy != rhsTy) {
      // Use the larger width and four-valued if either is four-valued.
      unsigned commonWidth = std::max(lhsTy.getWidth(), rhsTy.getWidth());
      auto commonDomain = (lhsTy.getDomain() == moore::Domain::FourValued ||
                           rhsTy.getDomain() == moore::Domain::FourValued)
                              ? moore::Domain::FourValued
                              : moore::Domain::TwoValued;
      auto commonTy =
          moore::IntType::get(context.getContext(), commonWidth, commonDomain);
      if (lhsTy != commonTy)
        lhsSBV = context.materializeConversion(commonTy, lhsSBV, false, loc);
      if (rhsTy != commonTy)
        rhsSBV = context.materializeConversion(commonTy, rhsSBV, false, loc);
      if (!lhsSBV || !rhsSBV)
        return failure();
    }

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

    auto combined = createUnifiedAndOp(builder, loc, *tagMatch, *valueMatch);
    return context.convertToBool(combined);
  }

  /// Match a variable pattern - binds the value to a pattern variable.
  /// Variable patterns always match and store the value for later use.
  FailureOr<Value>
  matchVariablePattern(const slang::ast::VariablePattern &pattern, Value value,
                       const slang::ast::Type &targetType) {
    // Get the pattern variable symbol and convert its type.
    const auto &var = pattern.variable;
    auto type = context.convertType(var.getType());
    if (!type)
      return failure();
    auto unpackedType = dyn_cast<moore::UnpackedType>(type);
    if (!unpackedType) {
      mlir::emitError(loc) << "pattern variable '" << var.name
                           << "' has non-unpacked type: " << type;
      return failure();
    }

    // Create a local variable for the pattern variable.
    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(unpackedType),
        builder.getStringAttr(var.name), value);

    // Register the pattern variable so it can be used in the case body.
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         &var, varOp);

    // Variable patterns always match - return true.
    auto boolType =
        moore::IntType::get(context.getContext(), 1, moore::Domain::TwoValued);
    return Value(moore::ConstantOp::create(builder, loc, boolType, 1));
  }

  /// Match a structure pattern - matches struct fields against sub-patterns.
  FailureOr<Value>
  matchStructurePattern(const slang::ast::StructurePattern &pattern,
                        Value value, const slang::ast::Type &targetType,
                        slang::ast::CaseStatementCondition condKind) {
    // Get the container type for extracting struct fields.
    auto containerType = value.getType();
    if (auto refTy = dyn_cast<moore::RefType>(containerType))
      containerType = refTy.getNestedType();

    // Match each field pattern.
    Value combinedMatch;
    for (const auto &fieldPattern : pattern.patterns) {
      // Get the field name and type.
      auto fieldName = builder.getStringAttr(fieldPattern.field->name);
      auto fieldType = context.convertType(fieldPattern.field->getType());
      if (!fieldType)
        return failure();

      // Extract the field value from the struct.
      Value fieldValue;
      if (auto structTy = dyn_cast<moore::StructType>(containerType)) {
        fieldValue = moore::StructExtractOp::create(builder, loc, fieldType,
                                                    fieldName, value);
      } else if (auto structTy =
                     dyn_cast<moore::UnpackedStructType>(containerType)) {
        fieldValue = moore::StructExtractOp::create(builder, loc, fieldType,
                                                    fieldName, value);
      } else {
        mlir::emitError(loc) << "structure pattern applied to non-struct type";
        return failure();
      }

      // Recursively match the sub-pattern.
      auto fieldMatch =
          matchPattern(*fieldPattern.pattern, fieldValue,
                       fieldPattern.field->getType(), condKind);
      if (failed(fieldMatch))
        return failure();

      // Combine with previous match results.
      if (combinedMatch)
        combinedMatch =
            createUnifiedAndOp(builder, loc, combinedMatch, *fieldMatch);
      else
        combinedMatch = *fieldMatch;
    }

    // If no patterns, always match.
    if (!combinedMatch) {
      auto boolType = moore::IntType::get(context.getContext(), 1,
                                          moore::Domain::TwoValued);
      return Value(moore::ConstantOp::create(builder, loc, boolType, 1));
    }

    return context.convertToBool(combinedMatch);
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

  // Handle disable statements: disable <block_name>
  // This terminates execution of a named block or task.
  // See IEEE 1800-2017 Section 9.6.2.
  LogicalResult visit(const slang::ast::DisableStatement &stmt) {
    // The target is an ArbitrarySymbolExpression that references the block/task
    const auto *ase =
        stmt.target.as_if<slang::ast::ArbitrarySymbolExpression>();
    if (!ase) {
      mlir::emitError(loc, "disable statement target must be a symbol reference");
      return failure();
    }

    // Lower disable directly to a branch when we can resolve a named block
    // target in the current lexical stack.
    if (Block *targetBlock = lookupDisableTarget(ase->symbol)) {
      cf::BranchOp::create(builder, loc, targetBlock);
      setTerminated();
      return success();
    }

    // Fallback to explicit moore.disable when target resolution isn't
    // available at import time.
    StringRef targetName = ase->symbol->name;
    moore::DisableOp::create(builder, loc, builder.getStringAttr(targetName));
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
    auto assertionsEnabled = readProceduralAssertionsEnabled();
    if (!assertionsEnabled)
      return failure();

    // Handle assertion statements that don't have an action block.
    if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      // Disabled assertions are treated as vacuous pass for immediate
      // assertion checks without action blocks.
      auto assertionsDisabled =
          moore::NotOp::create(builder, loc, assertionsEnabled);
      auto gatedCond =
          createUnifiedOrOp(builder, loc, cond, assertionsDisabled);
      auto defer = moore::DeferAssert::Immediate;
      if (stmt.isFinal)
        defer = moore::DeferAssert::Final;
      else if (stmt.isDeferred)
        defer = moore::DeferAssert::Observed;

      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        moore::AssertOp::create(builder, loc, defer, gatedCond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        moore::AssumeOp::create(builder, loc, defer, gatedCond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Restrict:
        // Immediate restrict assertions are lowered as assumes.
        moore::AssumeOp::create(builder, loc, defer, gatedCond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        moore::CoverOp::create(builder, loc, defer, gatedCond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Expect:
        moore::AssertOp::create(builder, loc, defer, gatedCond, StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported immediate assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    // Regard assertion statements with an action block as the "if-else".
    auto condLogic = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    auto enabledLogic =
        moore::ToBuiltinBoolOp::create(builder, loc, assertionsEnabled);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    Block &enabledBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, enabledLogic, &enabledBlock,
                             &exitBlock);

    // Evaluate assertion condition only while assertion checks are enabled.
    builder.setInsertionPointToEnd(&enabledBlock);
    cf::CondBranchOp::create(builder, loc, condLogic, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (stmt.ifTrue && failed(context.convertStatement(*stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    if (stmt.ifFalse) {
      // Generate the false branch if present.
      // Gate with assertionFailMessagesEnabled: if $assertfailoff was called,
      // skip the fail action (else clause) entirely.
      builder.setInsertionPointToEnd(falseBlock);
      auto failMsgsEnabled = readAssertionFailMessagesEnabled();
      if (failMsgsEnabled) {
        auto failMsgsLogic =
            moore::ToBuiltinBoolOp::create(builder, loc, failMsgsEnabled);
        Block &failBodyBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, failMsgsLogic, &failBodyBlock,
                                 &exitBlock);
        builder.setInsertionPointToEnd(&failBodyBlock);
      }
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
    if (auto *block = builder.getInsertionBlock()) {
      if (isa<moore::SVModuleOp>(block->getParentOp())) {
        if (block->mightHaveTerminator()) {
          if (auto output =
                  dyn_cast_or_null<moore::OutputOp>(block->getTerminator()))
            builder.setInsertionPoint(output);
        }
      }
    }
    auto *insertionBlock = builder.getInsertionBlock();
    auto enclosingProc = insertionBlock
                                 ? dyn_cast_or_null<moore::ProcedureOp>(
                                       insertionBlock->getParentOp())
                                 : nullptr;
    std::function<StringAttr(const slang::ast::Statement *)>
        extractActionBlockLabel =
            [&](const slang::ast::Statement *actionStmt) -> StringAttr {
      if (!actionStmt)
        return {};
      if (auto *timed = actionStmt->as_if<slang::ast::TimedStatement>())
        actionStmt = &timed->stmt;
      if (auto *stmtList = actionStmt->as_if<slang::ast::StatementList>()) {
        for (auto *stmt : stmtList->list) {
          if (auto label = extractActionBlockLabel(stmt))
            return label;
        }
        return {};
      }
      if (auto *block = actionStmt->as_if<slang::ast::BlockStatement>()) {
        return extractActionBlockLabel(&block->body);
      }
      if (auto *conditional =
              actionStmt->as_if<slang::ast::ConditionalStatement>()) {
        if (auto label = extractActionBlockLabel(&conditional->ifTrue))
          return label;
        if (conditional->ifFalse)
          return extractActionBlockLabel(conditional->ifFalse);
        return {};
      }
      if (auto *patternCase =
              actionStmt->as_if<slang::ast::PatternCaseStatement>()) {
        for (const auto &item : patternCase->items) {
          if (item.stmt) {
            if (auto label = extractActionBlockLabel(item.stmt))
              return label;
          }
        }
        if (patternCase->defaultCase)
          return extractActionBlockLabel(patternCase->defaultCase);
        return {};
      }
      if (auto *caseStmt = actionStmt->as_if<slang::ast::CaseStatement>()) {
        for (const auto &item : caseStmt->items) {
          if (item.stmt) {
            if (auto label = extractActionBlockLabel(item.stmt))
              return label;
          }
        }
        if (caseStmt->defaultCase)
          return extractActionBlockLabel(caseStmt->defaultCase);
        return {};
      }
      auto *exprStmt = actionStmt->as_if<slang::ast::ExpressionStatement>();
      if (!exprStmt)
        return {};
      auto *call = exprStmt->expr.as_if<slang::ast::CallExpression>();
      if (!call)
        return {};
      auto *sci = std::get_if<slang::ast::CallExpression::SystemCallInfo>(
          &call->subroutine);
      if (!sci)
        return {};
      StringRef taskName = sci->subroutine->name;
      if (taskName != "$error" && taskName != "$warning" &&
          taskName != "$fatal" && taskName != "$info" &&
          taskName != "$display" && taskName != "$write")
        return {};

      auto args = call->arguments();
      size_t msgArgIndex = 0;
      if (taskName == "$fatal" && !args.empty()) {
        auto firstArgConst = context.evaluateConstant(*args[0]);
        if (firstArgConst && firstArgConst.isInteger())
          msgArgIndex = 1;
      }
      if (msgArgIndex >= args.size())
        return builder.getStringAttr(taskName);

      const slang::ast::Expression *msgArg = args[msgArgIndex];
      while (auto *conv = msgArg->as_if<slang::ast::ConversionExpression>())
        msgArg = &conv->operand();
      if (auto *strLit = msgArg->as_if<slang::ast::StringLiteral>())
        return builder.getStringAttr(strLit->getValue());
      auto msgConst = context.evaluateConstant(*args[msgArgIndex]);
      if (msgConst && msgConst.isString())
        return builder.getStringAttr(msgConst.str());
      return {};
    };

    // Check for a `disable iff` expression:
    // The DisableIff construct can only occur at the top level of an assertion
    // and cannot be nested within properties.
    // Hence we only need to detect if the top level assertion expression
    // has type DisableIff, negate the `disable` expression, then pass it to
    // the `enable` parameter of AssertOp/AssumeOp.
    Value disableIffEnable;
    const slang::ast::AssertionExpr *innerPropertySpec = &stmt.propertySpec;
    if (auto *disableIff =
            stmt.propertySpec.as_if<slang::ast::DisableIffAssertionExpr>()) {
      auto disableCond = context.convertRvalueExpression(disableIff->condition);
      disableCond = context.convertToBool(disableCond);
      if (!disableCond)
        return failure();
      auto enableCond = moore::NotOp::create(builder, loc, disableCond);
      disableIffEnable = context.convertToI1(enableCond);
      if (!disableIffEnable)
        return failure();
      innerPropertySpec = &disableIff->expr;
    }

    StringAttr actionLabel;
    if (stmt.ifFalse && !stmt.ifFalse->as_if<slang::ast::EmptyStatement>())
      actionLabel = extractActionBlockLabel(stmt.ifFalse);
    if (!actionLabel && stmt.ifTrue &&
        !stmt.ifTrue->as_if<slang::ast::EmptyStatement>())
      actionLabel = extractActionBlockLabel(stmt.ifTrue);
    if ((!actionLabel && stmt.ifFalse &&
         !stmt.ifFalse->as_if<slang::ast::EmptyStatement>()) ||
        (!actionLabel && stmt.ifTrue &&
         !stmt.ifTrue->as_if<slang::ast::EmptyStatement>())) {
      mlir::emitWarning(loc)
          << "ignoring concurrent assertion action blocks during import";
    }

    if (context.currentAssertionClock && enclosingProc) {
      OpBuilder::InsertionGuard guard(builder);
      auto *moduleBlock = enclosingProc->getBlock();
      if (moduleBlock->mightHaveTerminator()) {
        if (auto *terminator = moduleBlock->getTerminator())
          builder.setInsertionPoint(terminator);
        else
          builder.setInsertionPointToEnd(moduleBlock);
      } else {
        builder.setInsertionPointToEnd(moduleBlock);
      }
      auto property = context.convertAssertionExpression(*innerPropertySpec, loc);
      if (!property) {
        // Slang uses InvalidAssertionExpr for dead generate branches.
        if (innerPropertySpec->as_if<slang::ast::InvalidAssertionExpr>())
          return success();
        return failure();
      }
      auto *assertionClock = getCanonicalAssertionClockSignalEvent(
          *context.currentAssertionClock);
      if (!assertionClock)
        assertionClock = context.currentAssertionClock;
      Value enable;
      IRMapping mapping;
      llvm::DenseSet<Operation *> active;
      if (context.currentAssertionGuard) {
        auto guardEnable = cloneAssertionValueIntoBlock(
            context.currentAssertionGuard, builder, moduleBlock, mapping, active);
        if (!guardEnable)
          mlir::emitWarning(loc)
              << "unable to hoist assertion guard; emitting unguarded assert";
        else {
          guardEnable = context.convertToBool(guardEnable);
          enable = context.convertToI1(guardEnable);
        }
      }
      if (disableIffEnable) {
        auto disableEnable = cloneAssertionValueIntoBlock(
            disableIffEnable, builder, moduleBlock, mapping, active);
        if (!disableEnable)
          return failure();
        disableEnable = context.convertToBool(disableEnable);
        if (!disableEnable)
          return failure();
        disableEnable = context.convertToI1(disableEnable);
        if (!disableEnable)
          return failure();
        enable = enable ? arith::AndIOp::create(builder, loc, enable,
                                                disableEnable)
                        : disableEnable;
      }
      if (enable && !enable.getType().isInteger(1)) {
        enable = context.convertToI1(enable);
        if (!enable)
          return failure();
      }
      auto clockVal = context.convertRvalueExpression(assertionClock->expr);
      clockVal = context.convertToI1(clockVal);
      if (!clockVal)
        return failure();
      auto edge = convertEdgeKindVerif(assertionClock->edge);
      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        verif::ClockedAssertOp::create(builder, loc, property, edge, clockVal,
                                       enable, actionLabel);
        return success();
      case slang::ast::AssertionKind::Assume:
        verif::ClockedAssumeOp::create(builder, loc, property, edge, clockVal,
                                       enable, actionLabel);
        return success();
      case slang::ast::AssertionKind::Restrict:
        // Restrict constraints are treated as assumptions in lowering.
        verif::ClockedAssumeOp::create(builder, loc, property, edge, clockVal,
                                       enable, actionLabel);
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        verif::ClockedCoverOp::create(builder, loc, property, edge, clockVal,
                                      enable, actionLabel);
        return success();
      case slang::ast::AssertionKind::CoverSequence:
        verif::ClockedCoverOp::create(builder, loc, property, edge, clockVal,
                                      enable, actionLabel);
        return success();
      case slang::ast::AssertionKind::Expect:
        verif::ClockedAssertOp::create(builder, loc, property, edge, clockVal,
                                       enable, actionLabel);
        return success();
      default:
        break;
      }
    }

    auto property = context.convertAssertionExpression(*innerPropertySpec, loc);
    if (!property) {
      // Slang uses InvalidAssertionExpr for dead generate branches.
      if (innerPropertySpec->as_if<slang::ast::InvalidAssertionExpr>())
        return success();
      return failure();
    }

    // If the property has its own clock and we're inside a procedure, hoist
    // the assertion to module level using clocked verif ops.
    if (auto clockOp = property.getDefiningOp<ltl::ClockOp>()) {
      if (enclosingProc) {
        OpBuilder::InsertionGuard guard(builder);
        auto *moduleBlock = enclosingProc->getBlock();
        if (moduleBlock->mightHaveTerminator()) {
          if (auto *terminator = moduleBlock->getTerminator())
            builder.setInsertionPoint(terminator);
          else
            builder.setInsertionPointToEnd(moduleBlock);
        } else {
          builder.setInsertionPointToEnd(moduleBlock);
        }

        // Clone the clock op and its dependencies to the module level.
        IRMapping mapping;
        llvm::DenseSet<Operation *> active;
        auto hoistedProperty = cloneAssertionValueIntoBlock(
            property, builder, moduleBlock, mapping, active);
        if (!hoistedProperty) {
          mlir::emitWarning(loc)
              << "unable to hoist assertion property; emitting inside "
                 "procedure";
        } else {
          // Get the clock op again from the hoisted property.
          auto hoistedClockOp =
              hoistedProperty.getDefiningOp<ltl::ClockOp>();
          if (hoistedClockOp) {
            auto edge = static_cast<verif::ClockEdge>(
                static_cast<int>(hoistedClockOp.getEdge()));
            auto clockVal = hoistedClockOp.getClock();
            auto innerProperty = hoistedClockOp.getInput();

            // Handle assertion guard if present.
            Value enable;
            if (context.currentAssertionGuard) {
              auto guardEnable = cloneAssertionValueIntoBlock(
                  context.currentAssertionGuard, builder, moduleBlock,
                  mapping, active);
              if (guardEnable) {
                guardEnable = context.convertToBool(guardEnable);
                enable = context.convertToI1(guardEnable);
              } else {
                mlir::emitWarning(loc)
                    << "unable to hoist assertion guard; emitting unguarded "
                       "assert";
              }
            }
            if (disableIffEnable) {
              auto disableEnable = cloneAssertionValueIntoBlock(
                  disableIffEnable, builder, moduleBlock, mapping, active);
              if (!disableEnable)
                return failure();
              disableEnable = context.convertToBool(disableEnable);
              if (!disableEnable)
                return failure();
              disableEnable = context.convertToI1(disableEnable);
              if (!disableEnable)
                return failure();
              enable = enable ? arith::AndIOp::create(builder, loc, enable,
                                                      disableEnable)
                              : disableEnable;
            }
            if (enable && !enable.getType().isInteger(1)) {
              enable = context.convertToI1(enable);
              if (!enable)
                return failure();
            }

            switch (stmt.assertionKind) {
            case slang::ast::AssertionKind::Assert:
              verif::ClockedAssertOp::create(builder, loc, innerProperty, edge,
                                             clockVal, enable, actionLabel);
              return success();
            case slang::ast::AssertionKind::Assume:
              verif::ClockedAssumeOp::create(builder, loc, innerProperty, edge,
                                             clockVal, enable, actionLabel);
              return success();
            case slang::ast::AssertionKind::Restrict:
              // Restrict constraints are treated as assumptions in lowering.
              verif::ClockedAssumeOp::create(builder, loc, innerProperty, edge,
                                             clockVal, enable, actionLabel);
              return success();
            case slang::ast::AssertionKind::CoverProperty:
              verif::ClockedCoverOp::create(builder, loc, innerProperty, edge,
                                            clockVal, enable, actionLabel);
              return success();
            case slang::ast::AssertionKind::CoverSequence:
              verif::ClockedCoverOp::create(builder, loc, innerProperty, edge,
                                            clockVal, enable, actionLabel);
              return success();
            case slang::ast::AssertionKind::Expect:
              verif::ClockedAssertOp::create(builder, loc, innerProperty, edge,
                                             clockVal, enable, actionLabel);
              return success();
            default:
              break;
            }
          }
        }
      }
    }

    switch (stmt.assertionKind) {
    case slang::ast::AssertionKind::Assert:
      verif::AssertOp::create(builder, loc, property, disableIffEnable,
                              actionLabel);
      return success();
    case slang::ast::AssertionKind::Assume:
      verif::AssumeOp::create(builder, loc, property, disableIffEnable,
                              actionLabel);
      return success();
    case slang::ast::AssertionKind::Restrict:
      // Restrict constraints are treated as assumptions in lowering.
      verif::AssumeOp::create(builder, loc, property, disableIffEnable,
                              actionLabel);
      return success();
    case slang::ast::AssertionKind::CoverProperty:
      verif::CoverOp::create(builder, loc, property, disableIffEnable,
                              actionLabel);
      return success();
    case slang::ast::AssertionKind::CoverSequence:
      verif::CoverOp::create(builder, loc, property, disableIffEnable,
                             actionLabel);
      return success();
    case slang::ast::AssertionKind::Expect:
      verif::AssertOp::create(builder, loc, property, disableIffEnable,
                              actionLabel);
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

    // VCD dump tasks (IEEE 1800-2017 Section 21.7).
    // $dumpfile opens a VCD file; $dumpvars starts tracing.
    // Other dump tasks ($dumpoff, $dumpon, etc.) are no-ops for now.
    if (subroutine.name == "$dumpfile") {
      std::string filename = "dump.vcd"; // default
      if (!args.empty()) {
        const slang::ast::Expression *arg = args[0];
        // Unwrap implicit conversions wrapping the string literal.
        while (auto *conv =
                   arg->as_if<slang::ast::ConversionExpression>())
          arg = &conv->operand();
        if (const auto *lit = arg->as_if<slang::ast::StringLiteral>())
          filename = std::string(lit->getValue());
        else {
          auto cv = context.evaluateConstant(*args[0]);
          if (cv && cv.isString())
            filename = cv.str();
        }
      }
      // Emit a print with circt.dumpfile attribute so the interpreter
      // can open the VCD file at runtime.  The format string must be
      // non-empty to survive dead-code elimination.
      std::string msgStr =
          "VCD: $dumpfile(\"" + filename + "\")\n";
      auto msg = moore::FormatLiteralOp::create(builder, loc, msgStr);
      auto displayOp = moore::DisplayBIOp::create(builder, loc, msg);
      displayOp->setAttr("circt.dumpfile",
                         builder.getStringAttr(filename));
      return true;
    }
    if (subroutine.name == "$dumpvars") {
      // Emit a print with circt.dumpvars attribute so the interpreter
      // can start VCD tracing at runtime.
      auto msg = moore::FormatLiteralOp::create(builder, loc,
                                                 "VCD: $dumpvars\n");
      auto displayOp = moore::DisplayBIOp::create(builder, loc, msg);
      displayOp->setAttr("circt.dumpvars", builder.getUnitAttr());
      return true;
    }
    if (subroutine.name == "$dumplimit" || subroutine.name == "$dumpoff" ||
        subroutine.name == "$dumpon" || subroutine.name == "$dumpflush" ||
        subroutine.name == "$dumpall" || subroutine.name == "$dumpports" ||
        subroutine.name == "$dumpportslimit" ||
        subroutine.name == "$dumpportsoff" ||
        subroutine.name == "$dumpportson" ||
        subroutine.name == "$dumpportsflush" ||
        subroutine.name == "$dumpportsall") {
      return true;
    }

    // Time formatting task (IEEE 1800-2017 Section 20.4.3)
    // $timeformat(units, precision, suffix, min_width)
    if (subroutine.name == "$timeformat") {
      auto intTy = moore::IntType::getInt(builder.getContext(), 32);
      Value units, precision, minWidth;
      std::string suffixStr;

      if (args.size() >= 1) {
        units = context.convertRvalueExpression(*args[0]);
        if (!units)
          return failure();
      }
      if (!units)
        units = moore::ConstantOp::create(
            builder, loc, intTy,
            APInt(32, static_cast<uint64_t>(-9), /*isSigned=*/true));

      if (args.size() >= 2) {
        precision = context.convertRvalueExpression(*args[1]);
        if (!precision)
          return failure();
      }
      if (!precision)
        precision = moore::ConstantOp::create(builder, loc, intTy, 0);

      if (args.size() >= 3) {
        // Extract the suffix as a compile-time constant string.
        const auto *suffArg = args[2];
        // Unwrap any implicit conversion.
        while (auto *conv =
                   suffArg->as_if<slang::ast::ConversionExpression>())
          suffArg = &conv->operand();
        if (auto *lit = suffArg->as_if<slang::ast::StringLiteral>())
          suffixStr = lit->getValue();
        else {
          // Try evaluating as a compile-time constant string.
          auto cv = context.evaluateConstant(*args[2]);
          if (cv && cv.isString())
            suffixStr = cv.str();
        }
      }

      if (args.size() >= 4) {
        minWidth = context.convertRvalueExpression(*args[3]);
        if (!minWidth)
          return failure();
      }
      if (!minWidth)
        minWidth = moore::ConstantOp::create(builder, loc, intTy, 20);

      if (units.getType() != intTy)
        units = moore::ConversionOp::create(builder, loc, intTy, units);
      if (precision.getType() != intTy)
        precision =
            moore::ConversionOp::create(builder, loc, intTy, precision);
      if (minWidth.getType() != intTy)
        minWidth =
            moore::ConversionOp::create(builder, loc, intTy, minWidth);

      moore::TimeFormatBIOp::create(builder, loc, units, precision,
                                    suffixStr, minWidth);
      return true;
    }

    // Coverage database tasks (IEEE 1800-2017 Section 20.14)
    // $set_coverage_db_name and $load_coverage_db are simulator-specific
    // persistence commands. No-op since we report coverage at sim end.
    if (subroutine.name == "$set_coverage_db_name" ||
        subroutine.name == "$load_coverage_db") {
      return true;
    }

    // Assertion control tasks (IEEE 1800-2017 Section 20.12)
    // $asserton/$assertoff/$assertkill and $assertcontrol(3/4/5) control
    // immediate assertion checking at runtime.
    if (subroutine.name == "$assertoff" || subroutine.name == "$assertkill") {
      auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
      auto disabled = moore::ConstantOp::create(builder, loc, i1Ty, 0);
      if (failed(writeProceduralAssertionsEnabled(disabled)))
        return failure();
      return true;
    }
    if (subroutine.name == "$asserton") {
      auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
      auto enabled = moore::ConstantOp::create(builder, loc, i1Ty, 1);
      if (failed(writeProceduralAssertionsEnabled(enabled)))
        return failure();
      return true;
    }
    if (subroutine.name == "$assertcontrol") {
      if (!args.empty()) {
        auto controlType = context.convertRvalueExpression(*args[0]);
        if (!controlType)
          return failure();
        auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
        if (controlType.getType() != i32Ty)
          controlType = moore::ConversionOp::create(builder, loc, i32Ty, controlType);

        auto currentEnabled = readProceduralAssertionsEnabled();
        if (!currentEnabled)
          return failure();
        auto c3 = moore::ConstantOp::create(builder, loc, i32Ty, 3);
        auto c4 = moore::ConstantOp::create(builder, loc, i32Ty, 4);
        auto c5 = moore::ConstantOp::create(builder, loc, i32Ty, 5);
        auto isOff = moore::EqOp::create(builder, loc, controlType, c3);
        auto isOn = moore::EqOp::create(builder, loc, controlType, c4);
        auto isKill = moore::EqOp::create(builder, loc, controlType, c5);
        auto offOrKill = createUnifiedOrOp(builder, loc, isOff, isKill);

        auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
        auto enabled = moore::ConstantOp::create(builder, loc, i1Ty, 1);
        auto disabled = moore::ConstantOp::create(builder, loc, i1Ty, 0);
        auto afterOff = selectBool(offOrKill, disabled, currentEnabled);
        auto nextState = selectBool(isOn, enabled, afterOff);
        if (failed(writeProceduralAssertionsEnabled(nextState)))
          return failure();

        // Also honor fail-message controls that map to $assertfailon/off.
        auto currentFailMsgsEnabled = readAssertionFailMessagesEnabled();
        if (!currentFailMsgsEnabled)
          return failure();
        auto c8 = moore::ConstantOp::create(builder, loc, i32Ty, 8);
        auto c9 = moore::ConstantOp::create(builder, loc, i32Ty, 9);
        auto isFailOn = moore::EqOp::create(builder, loc, controlType, c8);
        auto isFailOff = moore::EqOp::create(builder, loc, controlType, c9);
        auto nextFailAfterOff =
            selectBool(isFailOff, disabled, currentFailMsgsEnabled);
        auto nextFailState = selectBool(isFailOn, enabled, nextFailAfterOff);
        if (failed(writeAssertionFailMessagesEnabled(nextFailState)))
          return failure();
      }
      return true;
    }
    // $assertfailoff/$assertfailon: control assertion failure message display.
    if (subroutine.name == "$assertfailoff") {
      auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
      auto disabled = moore::ConstantOp::create(builder, loc, i1Ty, 0);
      if (failed(writeAssertionFailMessagesEnabled(disabled)))
        return failure();
      return true;
    }
    if (subroutine.name == "$assertfailon") {
      auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
      auto enabled = moore::ConstantOp::create(builder, loc, i1Ty, 1);
      if (failed(writeAssertionFailMessagesEnabled(enabled)))
        return failure();
      return true;
    }
    // Keep other non-core assertion-control forms as no-ops for now.
    if (subroutine.name == "$assertpasson" ||
        subroutine.name == "$assertpassoff" ||
        subroutine.name == "$assertnonvacuouson" ||
        subroutine.name == "$assertvacuousoff") {
      return true;
    }

    // Checkpoint/restart tasks (legacy Verilog, IEEE 1800-2017 Section 21.8)
    // These have no meaning in CIRCT's compilation flow. Emit a warning.
    if (subroutine.name == "$save" ||
        subroutine.name == "$restart" ||
        subroutine.name == "$incsave" ||
        subroutine.name == "$reset") {
      mlir::emitWarning(loc) << subroutine.name
                             << " is not supported in circt-sim"
                             << " (checkpoint/restart not implemented)";
      return true;
    }

    // $showvars — display variable names and values (IEEE 1800-2017 §21.2)
    if (subroutine.name == "$showvars") {
      SmallVector<Value> fragments;
      for (const auto *arg : args) {
        std::string varName = "?";
        if (auto *named =
                arg->as_if<slang::ast::NamedValueExpression>())
          varName = std::string(named->symbol.name);
        auto rvalue = context.convertRvalueExpression(*arg);
        if (!rvalue)
          return failure();
        auto value = context.convertToSimpleBitVector(rvalue);
        if (!value)
          return failure();
        fragments.push_back(moore::FormatLiteralOp::create(
            builder, loc, ("  " + varName + " = ")));
        fragments.push_back(moore::FormatIntOp::create(
            builder, loc, value, moore::IntFormat::Decimal,
            moore::IntAlign::Left, moore::IntPadding::Space,
            IntegerAttr(), /*isSigned=*/true));
        fragments.push_back(
            moore::FormatLiteralOp::create(builder, loc, "\n"));
      }
      if (!fragments.empty()) {
        Value msg;
        if (fragments.size() == 1)
          msg = fragments[0];
        else
          msg = moore::FormatConcatOp::create(builder, loc, fragments)
                    .getResult();
        moore::DisplayBIOp::create(builder, loc, msg);
      }
      return true;
    }

    // $stacktrace — print the scope hierarchy (IEEE 1800-2017 §21.2)
    if (subroutine.name == "$stacktrace") {
      SmallVector<StringRef> scopeNames;
      for (auto *scope = context.currentScope; scope;) {
        const auto &sym = scope->asSymbol();
        if (sym.kind == slang::ast::SymbolKind::Root ||
            sym.kind == slang::ast::SymbolKind::CompilationUnit)
          break;
        if (!sym.name.empty())
          scopeNames.push_back(sym.name);
        scope = sym.getParentScope();
      }
      std::string traceStr;
      for (const auto &name : scopeNames)
        traceStr += std::string(name) + "\n";
      if (!traceStr.empty()) {
        auto msg = moore::FormatLiteralOp::create(builder, loc, traceStr);
        moore::DisplayBIOp::create(builder, loc, msg);
      }
      return true;
    }

    // Debug/PLI tasks (IEEE 1800-2017 Sections 21.2, 21.9)
    // These are interactive simulator commands not feasible in compiled mode.
    if (subroutine.name == "$showscopes" ||
        subroutine.name == "$input" ||
        subroutine.name == "$key" ||
        subroutine.name == "$nokey" ||
        subroutine.name == "$log" ||
        subroutine.name == "$nolog" ||
        subroutine.name == "$scope" ||
        subroutine.name == "$list") {
      return true;
    }

    // PLD array tasks (IEEE 1800-2017 Section 21.7)
    // Legacy gate-array modeling functions — deprecated, not implemented.
    if (subroutine.name == "$async$and$array" ||
        subroutine.name == "$async$and$plane" ||
        subroutine.name == "$async$nand$array" ||
        subroutine.name == "$async$nand$plane" ||
        subroutine.name == "$async$nor$array" ||
        subroutine.name == "$async$nor$plane" ||
        subroutine.name == "$async$or$array" ||
        subroutine.name == "$async$or$plane" ||
        subroutine.name == "$sync$and$array" ||
        subroutine.name == "$sync$and$plane" ||
        subroutine.name == "$sync$nand$array" ||
        subroutine.name == "$sync$nand$plane" ||
        subroutine.name == "$sync$nor$array" ||
        subroutine.name == "$sync$nor$plane" ||
        subroutine.name == "$sync$or$array" ||
        subroutine.name == "$sync$or$plane") {
      mlir::emitError(loc) << "unsupported legacy PLD array task '"
                           << subroutine.name << "'";
      return failure();
    }

    // Stochastic queue tasks (IEEE 1800-2017 Section 21.6)
    // Legacy abstract queue functions — deprecated, not implemented.
    if (subroutine.name == "$q_initialize" ||
        subroutine.name == "$q_add" ||
        subroutine.name == "$q_remove" ||
        subroutine.name == "$q_exam") {
      mlir::emitError(loc) << "unsupported legacy stochastic queue task '"
                           << subroutine.name << "'";
      return failure();
    }

    // SDF annotation (IEEE 1800-2017 Section 30)
    // Timing back-annotation. Emit a warning.
    if (subroutine.name == "$sdf_annotate") {
      mlir::emitWarning(loc) << "$sdf_annotate is not supported in circt-sim"
                             << " (SDF timing annotation not implemented)";
      return true;
    }

    // $static_assert (compile-time assertion, already evaluated by slang)
    if (subroutine.name == "$static_assert") {
      return true;
    }

    // $writememb - write memory contents to binary file
    // IEEE 1800-2017 Section 21.4
    if (subroutine.name == "$writememb") {
      if (args.size() < 2) {
        mlir::emitError(loc) << "$writememb expects at least two arguments";
        return failure();
      }
      auto filename = context.convertRvalueExpression(*args[0]);
      if (!filename)
        return failure();
      if (!isa<moore::StringType>(filename.getType())) {
        if (isa<moore::IntType>(filename.getType()))
          filename =
              moore::IntToStringOp::create(builder, loc, filename).getResult();
        else {
          mlir::emitError(loc) << "$writememb filename must be a string";
          return failure();
        }
      }
      Value mem;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>())
        mem = context.convertLvalueExpression(assignExpr->left());
      else
        mem = context.convertLvalueExpression(*args[1]);
      if (!mem)
        return failure();
      context.captureRef(mem);
      moore::WriteMemBBIOp::create(builder, loc, filename, mem);
      return true;
    }

    // $writememh - write memory contents to hex file
    // IEEE 1800-2017 Section 21.4
    if (subroutine.name == "$writememh") {
      if (args.size() < 2) {
        mlir::emitError(loc) << "$writememh expects at least two arguments";
        return failure();
      }
      auto filename = context.convertRvalueExpression(*args[0]);
      if (!filename)
        return failure();
      if (!isa<moore::StringType>(filename.getType())) {
        if (isa<moore::IntType>(filename.getType()))
          filename =
              moore::IntToStringOp::create(builder, loc, filename).getResult();
        else {
          mlir::emitError(loc) << "$writememh filename must be a string";
          return failure();
        }
      }
      Value mem;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>())
        mem = context.convertLvalueExpression(assignExpr->left());
      else
        mem = context.convertLvalueExpression(*args[1]);
      if (!mem)
        return failure();
      context.captureRef(mem);
      moore::WriteMemHBIOp::create(builder, loc, filename, mem);
      return true;
    }

    // $sreadmemb/$sreadmemh - read memory from string (non-standard extension)
    // Silent no-op since these are rarely used and non-standard.
    if (subroutine.name == "$sreadmemb" ||
        subroutine.name == "$sreadmemh") {
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
      // Per IEEE 1800-2017, $exit terminates the calling program block.
      // Since programs are not yet supported, treat as $finish(0).
      moore::FinishBIOp::create(builder, loc, 0);
      moore::UnreachableOp::create(builder, loc);
      setTerminated();
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
      // Convert 4-state l32 fd to 2-state i32 if needed.
      auto intTy = moore::IntType::getInt(builder.getContext(), 32);
      if (fd.getType() != intTy)
        fd = moore::ConversionOp::create(builder, loc, intTy, fd);
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
      auto intTy = moore::IntType::getInt(builder.getContext(), 32);
      if (fd.getType() != intTy)
        fd = moore::ConversionOp::create(builder, loc, intTy, fd);

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

    // $fflush - flush file buffer
    if (subroutine.name == "$fflush") {
      Value fd;
      if (args.empty()) {
        auto intTy = moore::IntType::getInt(builder.getContext(), 32);
        fd = moore::ConstantOp::create(builder, loc, intTy, 0);
      } else {
        fd = context.convertRvalueExpression(*args[0]);
        if (!fd) return failure();
        // Convert 4-state l32 fd to 2-state i32 if needed.
        auto intTy = moore::IntType::getInt(builder.getContext(), 32);
        if (fd.getType() != intTy)
          fd = moore::ConversionOp::create(builder, loc, intTy, fd);
      }
      moore::FFlushBIOp::create(builder, loc, fd);
      return true;
    }

    // $rewind - reset file position to beginning
    if (subroutine.name == "$rewind") {
      if (args.size() != 1) {
        mlir::emitError(loc) << "$rewind expects exactly one argument";
        return failure();
      }
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return failure();
      // Convert 4-state l32 fd to 2-state i32 if needed.
      {
        auto intTy = moore::IntType::getInt(builder.getContext(), 32);
        if (fd.getType() != intTy)
          fd = moore::ConversionOp::create(builder, loc, intTy, fd);
      }
      moore::RewindBIOp::create(builder, loc, fd);
      return true;
    }

    // $readmemb - load memory from binary file
    if (subroutine.name == "$readmemb") {
      if (args.size() < 2) {
        mlir::emitError(loc) << "$readmemb expects at least two arguments";
        return failure();
      }
      auto filename = context.convertRvalueExpression(*args[0]);
      if (!filename)
        return failure();
      // Convert filename to string type if needed
      if (!isa<moore::StringType>(filename.getType())) {
        if (isa<moore::IntType>(filename.getType())) {
          filename =
              moore::IntToStringOp::create(builder, loc, filename).getResult();
        } else {
          mlir::emitError(loc) << "$readmemb filename must be a string";
          return failure();
        }
      }
      // Second argument is the memory array (output)
      // Slang wraps it as AssignmentExpression(mem = EmptyArgument)
      Value mem;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>()) {
        mem = context.convertLvalueExpression(assignExpr->left());
      } else {
        mem = context.convertLvalueExpression(*args[1]);
      }
      if (!mem)
        return failure();
      // Ensure task/function-local readmem calls capture outer memory refs so
      // generated func.func bodies remain IsolatedFromAbove.
      context.captureRef(mem);
      moore::ReadMemBBIOp::create(builder, loc, filename, mem);
      return true;
    }

    // $readmemh - load memory from hexadecimal file
    if (subroutine.name == "$readmemh") {
      if (args.size() < 2) {
        mlir::emitError(loc) << "$readmemh expects at least two arguments";
        return failure();
      }
      auto filename = context.convertRvalueExpression(*args[0]);
      if (!filename)
        return failure();
      // Convert filename to string type if needed
      if (!isa<moore::StringType>(filename.getType())) {
        if (isa<moore::IntType>(filename.getType())) {
          filename =
              moore::IntToStringOp::create(builder, loc, filename).getResult();
        } else {
          mlir::emitError(loc) << "$readmemh filename must be a string";
          return failure();
        }
      }
      // Second argument is the memory array (output)
      // Slang wraps it as AssignmentExpression(mem = EmptyArgument)
      Value mem;
      if (auto *assignExpr =
              args[1]->as_if<slang::ast::AssignmentExpression>()) {
        mem = context.convertLvalueExpression(assignExpr->left());
      } else {
        mem = context.convertLvalueExpression(*args[1]);
      }
      if (!mem)
        return failure();
      // Ensure task/function-local readmem calls capture outer memory refs so
      // generated func.func bodies remain IsolatedFromAbove.
      context.captureRef(mem);
      moore::ReadMemHBIOp::create(builder, loc, filename, mem);
      return true;
    }

    // $fscanf(fd, format, args...) - used as a task (return value discarded)
    // IEEE 1800-2017 Section 21.3.3 "File input functions"
    if (subroutine.name == "$fscanf" && args.size() >= 2) {
      Value fd = context.convertRvalueExpression(*args[0]);
      if (!fd)
        return failure();
      auto i32Ty = moore::IntType::getInt(context.getContext(), 32);
      if (fd.getType() != i32Ty)
        fd = moore::ConversionOp::create(builder, loc, i32Ty, fd);

      const auto *fmtArg = args[1];
      std::string formatStr;
      if (const auto *strLit = fmtArg->as_if<slang::ast::StringLiteral>()) {
        formatStr = std::string(strLit->getValue());
      } else {
        auto cv = context.evaluateConstant(*fmtArg);
        if (cv && cv.isString())
          formatStr = cv.str();
        else {
          mlir::emitError(loc) << "$fscanf format must be a string literal";
          return failure();
        }
      }

      SmallVector<Value> outputRefs;
      for (size_t i = 2; i < args.size(); ++i) {
        const auto *arg = args[i];
        if (const auto *assignExpr =
                arg->as_if<slang::ast::AssignmentExpression>()) {
          Value ref = context.convertLvalueExpression(assignExpr->left());
          if (!ref)
            return failure();
          outputRefs.push_back(ref);
        } else {
          Value ref = context.convertLvalueExpression(*arg);
          if (!ref)
            return failure();
          outputRefs.push_back(ref);
        }
      }

      // Create the fscanf op and discard the result
      moore::FScanfBIOp::create(builder, loc, fd,
                                builder.getStringAttr(formatStr), outputRefs);
      return true;
    }

    // $sscanf(str, format, args...) - used as a task (return value discarded)
    // IEEE 1800-2017 Section 21.3.4 "Reading data from a string"
    if (subroutine.name == "$sscanf" && args.size() >= 2) {
      Value inputStr = context.convertRvalueExpression(*args[0]);
      if (!inputStr)
        return failure();
      if (!isa<moore::StringType>(inputStr.getType())) {
        inputStr = moore::ConversionOp::create(
            builder, loc, moore::StringType::get(context.getContext()),
            inputStr);
      }

      const auto *fmtArg = args[1];
      std::string formatStr;
      if (const auto *strLit = fmtArg->as_if<slang::ast::StringLiteral>()) {
        formatStr = std::string(strLit->getValue());
      } else {
        auto cv = context.evaluateConstant(*fmtArg);
        if (cv && cv.isString())
          formatStr = cv.str();
        else {
          mlir::emitError(loc) << "$sscanf format must be a string literal";
          return failure();
        }
      }

      SmallVector<Value> outputRefs;
      for (size_t i = 2; i < args.size(); ++i) {
        const auto *arg = args[i];
        if (const auto *assignExpr =
                arg->as_if<slang::ast::AssignmentExpression>()) {
          Value ref = context.convertLvalueExpression(assignExpr->left());
          if (!ref)
            return failure();
          outputRefs.push_back(ref);
        } else {
          Value ref = context.convertLvalueExpression(*arg);
          if (!ref)
            return failure();
          outputRefs.push_back(ref);
        }
      }

      // Create the sscanf op and discard the result
      moore::SScanfBIOp::create(builder, loc, inputStr,
                                builder.getStringAttr(formatStr), outputRefs);
      return true;
    }

    // $strobe variants
    bool isStrobe = false;
    StringRef remainingStrobe = subroutine.name;
    if (remainingStrobe.consume_front("$strobe")) isStrobe = true;
    IntFormat fmtStrobe = IntFormat::Decimal;
    if (isStrobe && !remainingStrobe.empty()) {
      if (remainingStrobe == "b") fmtStrobe = IntFormat::Binary;
      else if (remainingStrobe == "o") fmtStrobe = IntFormat::Octal;
      else if (remainingStrobe == "h") fmtStrobe = IntFormat::HexLower;
      else isStrobe = false;
    }
    if (isStrobe) {
      // `$strobe` evaluates/prints at the end of the current time step.
      // Lower it as a detached `fork ... join_none` branch that waits `#0`
      // before formatting/printing, so argument values are sampled late.
      auto forkOp = moore::ForkOp::create(builder, loc, moore::JoinType::JoinNone,
                                          StringAttr{}, /*numBranches=*/1);
      auto &region = forkOp.getBranches().front();
      if (region.empty())
        region.emplaceBlock();

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&region.front());

      auto zeroDelay = moore::ConstantTimeOp::create(builder, loc, 0);
      moore::WaitDelayOp::create(builder, loc, zeroDelay);

      auto msg = context.convertFormatString(args, loc, fmtStrobe, true);
      if (failed(msg))
        return failure();
      if (*msg != Value{})
        moore::DisplayBIOp::create(builder, loc, *msg);
      moore::ForkTerminatorOp::create(builder, loc);
      return true;
    }

    // $fstrobe variants
    bool isFStrobe = false;
    StringRef remainingFS = subroutine.name;
    if (remainingFS.consume_front("$fstrobe")) isFStrobe = true;
    IntFormat fmtFS = IntFormat::Decimal;
    if (isFStrobe && !remainingFS.empty()) {
      if (remainingFS == "b") fmtFS = IntFormat::Binary;
      else if (remainingFS == "o") fmtFS = IntFormat::Octal;
      else if (remainingFS == "h") fmtFS = IntFormat::HexLower;
      else isFStrobe = false;
    }
    if (isFStrobe) {
      if (args.empty()) return mlir::emitError(loc, "$fstrobe requires fd");
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd) return failure();
      auto intTy = moore::IntType::getInt(builder.getContext(), 32);
      if (fd.getType() != intTy)
        fd = moore::ConversionOp::create(builder, loc, intTy, fd);
      auto msg = context.convertFormatString(args.subspan(1), loc, fmtFS, true);
      if (failed(msg)) return failure();
      if (*msg == Value{}) return true;
      moore::FStrobeBIOp::create(builder, loc, fd, *msg);
      return true;
    }

    // $monitor variants
    bool isMon = false;
    StringRef remainingM = subroutine.name;
    if (remainingM.consume_front("$monitor")) isMon = true;
    IntFormat fmtM = IntFormat::Decimal;
    if (isMon && !remainingM.empty()) {
      if (remainingM == "b") fmtM = IntFormat::Binary;
      else if (remainingM == "o") fmtM = IntFormat::Octal;
      else if (remainingM == "h") fmtM = IntFormat::HexLower;
      else if (remainingM == "on" || remainingM == "off") isMon = false;
      else isMon = false;
    }
    if (isMon) {
      auto msg = context.convertFormatString(args, loc, fmtM, true);
      if (failed(msg)) return failure();
      if (*msg == Value{}) return true;
      moore::MonitorBIOp::create(builder, loc, *msg);
      return true;
    }

    // $fmonitor variants
    bool isFMon = false;
    StringRef remainingFM = subroutine.name;
    if (remainingFM.consume_front("$fmonitor")) isFMon = true;
    IntFormat fmtFM = IntFormat::Decimal;
    if (isFMon && !remainingFM.empty()) {
      if (remainingFM == "b") fmtFM = IntFormat::Binary;
      else if (remainingFM == "o") fmtFM = IntFormat::Octal;
      else if (remainingFM == "h") fmtFM = IntFormat::HexLower;
      else isFMon = false;
    }
    if (isFMon) {
      if (args.empty()) return mlir::emitError(loc, "$fmonitor requires fd");
      auto fd = context.convertRvalueExpression(*args[0]);
      if (!fd) return failure();
      auto intTy = moore::IntType::getInt(builder.getContext(), 32);
      if (fd.getType() != intTy)
        fd = moore::ConversionOp::create(builder, loc, intTy, fd);
      auto msg = context.convertFormatString(args.subspan(1), loc, fmtFM, true);
      if (failed(msg)) return failure();
      if (*msg == Value{}) return true;
      moore::FMonitorBIOp::create(builder, loc, fd, *msg);
      return true;
    }

    // $monitoron/$monitoroff
    if (subroutine.name == "$monitoron") {
      moore::MonitorOnBIOp::create(builder, loc);
      return true;
    }
    if (subroutine.name == "$monitoroff") {
      moore::MonitorOffBIOp::create(builder, loc);
      return true;
    }

    // $printtimescale
    if (subroutine.name == "$printtimescale") {
      moore::PrintTimescaleBIOp::create(builder, loc);
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

    // Check if this is an event type - if so, use EventTriggerOp with the ref.
    // EventTriggerOp takes a reference so it can toggle the underlying signal.
    auto refType = dyn_cast<moore::RefType>(target.getType());
    if (refType && isa<moore::EventType>(refType.getNestedType())) {
      moore::EventTriggerOp::create(builder, loc, target);
      return success();
    }

    // Read the current value of the target.
    Value readValue = moore::ReadOp::create(builder, loc, target);

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

  // Handle procedural assign/force statements (IEEE 1800-2017 Section 10.6).
  LogicalResult visit(const slang::ast::ProceduralAssignStatement &stmt) {
    // The assignment expression contains the target and value
    const auto *assignExpr =
        stmt.assignment.as_if<slang::ast::AssignmentExpression>();
    if (!assignExpr) {
      mlir::emitError(loc) << "expected assignment expression in procedural "
                           << (stmt.isForce ? "force" : "assign");
      return failure();
    }

    auto dst = context.convertLvalueExpression(assignExpr->left());
    if (!dst)
      return failure();

    auto src = context.convertRvalueExpression(assignExpr->right());
    if (!src)
      return failure();

    if (stmt.isForce) {
      moore::ForceAssignOp::create(builder, loc, dst, src);
    } else {
      // Procedural assign (not force) — use blocking assign.
      moore::BlockingAssignOp::create(builder, loc, dst, src);
    }
    return success();
  }

  // Handle procedural deassign/release statements (IEEE 1800-2017 Section 10.6).
  LogicalResult visit(const slang::ast::ProceduralDeassignStatement &stmt) {
    if (stmt.isRelease) {
      auto dst = context.convertLvalueExpression(stmt.lvalue);
      if (!dst)
        return failure();
      moore::ReleaseAssignOp::create(builder, loc, dst);
      return success();
    }
    // Deassign — no-op for now (procedural assign uses blocking assign).
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

FailureOr<Value>
Context::matchPattern(const slang::ast::Pattern &pattern, Value value,
                      const slang::ast::Type &targetType,
                      slang::ast::CaseStatementCondition condKind,
                      Location loc) {
  // Create a temporary StmtVisitor to use its pattern matching implementation.
  StmtVisitor visitor(*this, loc);
  return visitor.matchPattern(pattern, value, targetType, condKind);
}
// NOLINTEND(misc-no-recursion)
