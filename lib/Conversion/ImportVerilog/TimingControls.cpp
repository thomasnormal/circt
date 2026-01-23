//===- TimingControl.cpp - Slang timing control conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/TimingControl.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

static ltl::ClockEdge convertEdgeKindLTL(const slang::ast::EdgeKind edge) {
  using slang::ast::EdgeKind;
  switch (edge) {
  case EdgeKind::NegEdge:
    return ltl::ClockEdge::Neg;
  case EdgeKind::PosEdge:
    return ltl::ClockEdge::Pos;
  case EdgeKind::None:
    // TODO: SV 16.16, what to do when no edge is specified?
    // For now, assume all changes (two-valued should be the same as both
    // edges)
  case EdgeKind::BothEdges:
    return ltl::ClockEdge::Both;
  }
  llvm_unreachable("all edge kinds handled");
}

static moore::Edge convertEdgeKind(const slang::ast::EdgeKind edge) {
  using slang::ast::EdgeKind;
  switch (edge) {
  case EdgeKind::None:
    return moore::Edge::AnyChange;
  case EdgeKind::PosEdge:
    return moore::Edge::PosEdge;
  case EdgeKind::NegEdge:
    return moore::Edge::NegEdge;
  case EdgeKind::BothEdges:
    return moore::Edge::BothEdges;
  }
  llvm_unreachable("all edge kinds handled");
}

// NOLINTBEGIN(misc-no-recursion)
namespace {

// Handle any of the event control constructs.
struct EventControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  // Handle single signal events like `posedge x`, `negedge y iff z`, or `w`.
  // Also handles clocking block event references like `@(cb)`.
  LogicalResult visit(const slang::ast::SignalEventControl &ctrl) {
    // Check if the expression references a clocking block.
    // In that case, we need to convert the clocking block's clock event instead.
    auto symRef = ctrl.expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      auto &clockEvent = clockingBlock.getEvent();
      // Recursively convert the clocking block's clock event
      auto visitor = *this;
      visitor.loc = context.convertLocation(clockEvent.sourceRange);
      return clockEvent.visit(visitor);
    }

    // Check if the expression references a sequence or property.
    // Using a sequence/property as an event control (@seq) is an SVA feature
    // that waits for the sequence/property to match. This is not yet supported.
    if (symRef && symRef->kind == slang::ast::SymbolKind::Sequence)
      return mlir::emitError(loc)
             << "sequence event controls (@sequence_name) are not yet "
                "supported";
    if (symRef && symRef->kind == slang::ast::SymbolKind::Property)
      return mlir::emitError(loc)
             << "property event controls (@property_name) are not yet "
                "supported";

    auto edge = convertEdgeKind(ctrl.edge);
    auto expr = context.convertRvalueExpression(ctrl.expr);
    if (!expr)
      return failure();

    // Check if the expression evaluates to an LTL sequence or property type.
    // This can happen when a sequence/property is referenced indirectly.
    if (isa<ltl::SequenceType, ltl::PropertyType>(expr.getType()))
      return mlir::emitError(loc)
             << "sequence/property event controls are not yet supported";

    Value condition;
    if (ctrl.iffCondition) {
      condition = context.convertRvalueExpression(*ctrl.iffCondition);
      condition = context.convertToBool(condition, Domain::TwoValued);
      if (!condition)
        return failure();
    }
    moore::DetectEventOp::create(builder, loc, edge, expr, condition);
    return success();
  }

  // Handle a list of signal events.
  LogicalResult visit(const slang::ast::EventListControl &ctrl) {
    for (const auto *event : ctrl.events) {
      auto visitor = *this;
      visitor.loc = context.convertLocation(event->sourceRange);
      if (failed(event->visit(visitor)))
        return failure();
    }
    return success();
  }

  // Emit an error for all other timing controls.
  template <typename T>
  LogicalResult visit(T &&ctrl) {
    return mlir::emitError(loc)
           << "unsupported event control: " << slang::ast::toString(ctrl.kind);
  }
};

// Handle any of the delay control constructs.
struct DelayControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  // Handle delays.
  LogicalResult visit(const slang::ast::DelayControl &ctrl) {
    auto delay = context.convertRvalueExpression(
        ctrl.expr, moore::TimeType::get(builder.getContext()));
    if (!delay)
      return failure();
    moore::WaitDelayOp::create(builder, loc, delay);
    return success();
  }

  // Emit an error for all other timing controls.
  template <typename T>
  LogicalResult visit(T &&ctrl) {
    return mlir::emitError(loc)
           << "unsupported delay control: " << slang::ast::toString(ctrl.kind);
  }
};

struct LTLClockControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;
  Value seqOrPro;

  Value visit(const slang::ast::SignalEventControl &ctrl) {
    // Check if the expression references a clocking block.
    // In that case, we need to convert the clocking block's clock event instead.
    auto symRef = ctrl.expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      auto &clockEvent = clockingBlock.getEvent();
      // Recursively convert the clocking block's clock event
      auto visitor = *this;
      visitor.loc = context.convertLocation(clockEvent.sourceRange);
      return clockEvent.visit(visitor);
    }

    auto edge = convertEdgeKindLTL(ctrl.edge);
    auto expr = context.convertRvalueExpression(ctrl.expr);
    if (!expr)
      return Value{};
    Value condition;
    if (ctrl.iffCondition) {
      condition = context.convertRvalueExpression(*ctrl.iffCondition);
      condition = context.convertToBool(condition, Domain::TwoValued);
      if (!condition)
        return Value{};
    }
    expr = context.convertToI1(expr);
    if (!expr)
      return Value{};
    return ltl::ClockOp::create(builder, loc, seqOrPro, edge, expr);
  }

  template <typename T>
  Value visit(T &&ctrl) {
    mlir::emitError(loc, "unsupported LTL clock control: ")
        << slang::ast::toString(ctrl.kind);
    return Value{};
  }
};

} // namespace

// Entry point to timing control handling. This deals with the layer of repeats
// that a timing control may be wrapped in, and also handles the implicit event
// control which may appear at that point. For any event control a `WaitEventOp`
// will be created and populated by `handleEventControl`. Any delay control will
// be handled by `handleDelayControl`.
static LogicalResult handleRoot(Context &context,
                                const slang::ast::TimingControl &ctrl,
                                moore::WaitEventOp *implicitWaitOp) {
  auto &builder = context.builder;
  auto loc = context.convertLocation(ctrl.sourceRange);
  if (context.options.ignoreTimingControls.value_or(false)) {
    mlir::emitWarning(loc) << "ignoring timing control due to "
                              "--ignore-timing-controls";
    return success();
  }

  using slang::ast::TimingControlKind;
  switch (ctrl.kind) {
    // Handle repeated event control like `@(repeat(N) posedge clk)`.
    // This is lowered as a countdown loop that waits for the event N times.
  case TimingControlKind::RepeatedEvent: {
    auto &repeatedCtrl = ctrl.as<slang::ast::RepeatedEventControl>();

    // Convert the count expression.
    auto count = context.convertRvalueExpression(repeatedCtrl.expr);
    if (!count)
      return failure();

    // Verify the count is an integer type.
    auto countIntType = dyn_cast<moore::IntType>(count.getType());
    if (!countIntType) {
      return mlir::emitError(loc)
             << "repeat event count must have integer type, but got "
             << count.getType();
    }

    // Get the parent region to create blocks in.
    Region *parentRegion = builder.getInsertionBlock()->getParent();

    // Create the blocks for the loop: check, body, step, exit.
    auto *exitBlock = new Block();
    auto *stepBlock = new Block();
    auto *bodyBlock = new Block();
    auto *checkBlock = new Block();

    // Insert blocks in forward order after current insertion point.
    parentRegion->getBlocks().insert(
        std::next(builder.getInsertionBlock()->getIterator()), checkBlock);
    parentRegion->getBlocks().insert(
        std::next(checkBlock->getIterator()), bodyBlock);
    parentRegion->getBlocks().insert(
        std::next(bodyBlock->getIterator()), stepBlock);
    parentRegion->getBlocks().insert(
        std::next(stepBlock->getIterator()), exitBlock);

    // Add the counter argument to the check block.
    auto currentCount = checkBlock->addArgument(count.getType(), count.getLoc());

    // Branch to the check block with the initial count.
    cf::BranchOp::create(builder, loc, checkBlock, count);

    // Generate the loop condition check: while (count != 0).
    builder.setInsertionPointToEnd(checkBlock);
    auto cond = context.convertToBool(currentCount);
    if (!cond)
      return failure();
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, bodyBlock, exitBlock);

    // Generate the loop body: wait for the event.
    builder.setInsertionPointToEnd(bodyBlock);
    if (failed(handleRoot(context, repeatedCtrl.event, implicitWaitOp)))
      return failure();
    // Branch to the step block (unless the body was terminated).
    if (builder.getInsertionBlock())
      cf::BranchOp::create(builder, loc, stepBlock);

    // Decrement the counter and branch back to the check block.
    builder.setInsertionPointToEnd(stepBlock);
    auto one =
        moore::ConstantOp::create(builder, count.getLoc(), countIntType, 1);
    Value nextCount =
        moore::SubOp::create(builder, count.getLoc(), currentCount, one);
    cf::BranchOp::create(builder, loc, checkBlock, nextCount);

    // Continue inserting in the exit block.
    if (exitBlock->hasNoPredecessors()) {
      exitBlock->erase();
      builder.clearInsertionPoint();
    } else {
      builder.setInsertionPointToEnd(exitBlock);
    }
    return success();
  }

    // Handle implicit events, i.e. `@*` and `@(*)`. This implicitly includes
    // all variables read within the statement that follows after the event
    // control. Since we haven't converted that statement yet, simply create and
    // empty wait op and let `Context::convertTimingControl` populate it once
    // the statement has been lowered.
  case TimingControlKind::ImplicitEvent:
    if (!implicitWaitOp)
      return mlir::emitError(loc) << "implicit events cannot be used here";
    *implicitWaitOp = moore::WaitEventOp::create(builder, loc);
    return success();

    // Handle event control.
  case TimingControlKind::SignalEvent:
  case TimingControlKind::EventList: {
    auto waitOp = moore::WaitEventOp::create(builder, loc);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());
    EventControlVisitor visitor{context, loc, builder};
    return ctrl.visit(visitor);
  }

    // Handle delay control.
  case TimingControlKind::Delay:
  case TimingControlKind::Delay3:
  case TimingControlKind::OneStepDelay:
  case TimingControlKind::CycleDelay: {
    DelayControlVisitor visitor{context, loc, builder};
    return ctrl.visit(visitor);
  }

  default:
    return mlir::emitError(loc, "unsupported timing control: ")
           << slang::ast::toString(ctrl.kind);
  }
}

LogicalResult
Context::convertTimingControl(const slang::ast::TimingControl &ctrl) {
  return handleRoot(*this, ctrl, nullptr);
}

LogicalResult
Context::convertTimingControl(const slang::ast::TimingControl &ctrl,
                              const slang::ast::Statement &stmt) {
  // Convert the timing control. Implicit event control will create a new empty
  // `WaitEventOp` and assign it to `implicitWaitOp`. This op will be populated
  // further down.
  moore::WaitEventOp implicitWaitOp;
  {
    auto previousCallback = rvalueReadCallback;
    auto done = llvm::make_scope_exit([&] { rvalueReadCallback = previousCallback; });
    // Reads happening as part of the event control should not be added to a
    // surrounding implicit event control's list of implicitly observed
    // variables. However, we still need to propagate reads to any outer
    // callback (e.g., for function capture tracking) so that event controls
    // inside tasks/functions that reference module-level signals can properly
    // capture those signals.
    //
    // We use a lambda that discards the read locally (doesn't add to implicit
    // event list) but still chains to the previous callback if present.
    rvalueReadCallback = previousCallback ? [previousCallback](moore::ReadOp readOp) {
      // Chain to previous callback (e.g., function capture callback)
      previousCallback(readOp);
    } : std::function<void(moore::ReadOp)>(nullptr);
    if (failed(handleRoot(*this, ctrl, &implicitWaitOp)))
      return failure();
  }

  // Convert the statement. In case `implicitWaitOp` is set, we register a
  // callback to collect all the variables read by the statement into
  // `readValues`, such that we can populate the op with implicitly observed
  // variables afterwards.
  llvm::SmallSetVector<Value, 8> readValues;
  {
    auto previousCallback = rvalueReadCallback;
    auto done = llvm::make_scope_exit([&] { rvalueReadCallback = previousCallback; });
    auto previousAssertionClock = currentAssertionClock;
    auto clockDone =
        llvm::make_scope_exit([&] { currentAssertionClock = previousAssertionClock; });
    currentAssertionClock = nullptr;
    if (auto *signalCtrl = ctrl.as_if<slang::ast::SignalEventControl>()) {
      if (!signalCtrl->iffCondition)
        currentAssertionClock = signalCtrl;
    }
    if (implicitWaitOp) {
      rvalueReadCallback = [&](moore::ReadOp readOp) {
        readValues.insert(readOp.getInput());
        if (previousCallback)
          previousCallback(readOp);
      };
    }
    if (failed(convertStatement(stmt)))
      return failure();
  }

  // Populate the implicit wait op with reads from the variables read by the
  // statement.
  if (implicitWaitOp) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&implicitWaitOp.getBody().emplaceBlock());
    for (auto readValue : readValues) {
      auto value =
          moore::ReadOp::create(builder, implicitWaitOp.getLoc(), readValue);
      moore::DetectEventOp::create(builder, implicitWaitOp.getLoc(),
                                   moore::Edge::AnyChange, value, Value{});
    }
  }

  return success();
}

Value Context::convertLTLTimingControl(const slang::ast::TimingControl &ctrl,
                                       const Value &seqOrPro) {
  auto &builder = this->builder;
  auto loc = this->convertLocation(ctrl.sourceRange);
  LTLClockControlVisitor visitor{*this, loc, builder, seqOrPro};
  return ctrl.visit(visitor);
}
// NOLINTEND(misc-no-recursion)
