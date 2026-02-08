//===- TimingControl.cpp - Slang timing control conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Support/LTLSequenceNFA.h"
#include "slang/ast/TimingControl.h"
#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/IR/IRMapping.h"
#include <cstdlib>
#include <limits>

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

static LogicalResult handleRoot(Context &context,
                                const slang::ast::TimingControl &ctrl,
                                moore::WaitEventOp *implicitWaitOp);

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

    // If this is an event-typed assertion port, substitute the bound timing
    // control directly (e.g., for $past/$rose explicit clocking arguments).
    if (context.inAssertionExpr && symRef) {
      if (auto *port = symRef->as_if<slang::ast::AssertionPortSymbol>()) {
        if (auto *binding = context.lookupAssertionPortBinding(port)) {
          if (binding->kind == AssertionPortBinding::Kind::TimingControl &&
              binding->timingControl) {
            auto nestedLoc = context.convertLocation(
                binding->timingControl->sourceRange);
            auto visitor = *this;
            visitor.loc = nestedLoc;
            return binding->timingControl->visit(visitor);
          }
        }
      }
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

static void eraseLTLDeadOps(Value root) {
  Operation *rootOp = root ? root.getDefiningOp() : nullptr;
  if (!rootOp)
    return;

  llvm::SmallVector<Operation *, 16> worklist;
  llvm::SmallVector<Operation *, 16> ltlOps;
  llvm::DenseSet<Operation *> visited;
  worklist.push_back(rootOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!op || !visited.insert(op).second)
      continue;
    if (!isa<ltl::LTLDialect>(op->getDialect()))
      continue;
    ltlOps.push_back(op);
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp())
        worklist.push_back(def);
    }
  }

  for (Operation *op : llvm::reverse(ltlOps)) {
    if (op->use_empty())
      op->erase();
  }
}

static bool isAssertionEventControl(const slang::ast::Expression &expr) {
  if (expr.kind == slang::ast::ExpressionKind::AssertionInstance)
    return true;
  if (auto symRef = expr.getSymbolReference()) {
    return symRef->kind == slang::ast::SymbolKind::Sequence ||
           symRef->kind == slang::ast::SymbolKind::Property;
  }
  return false;
}

static const slang::ast::SignalEventControl *
getCanonicalSignalEventControlForAssertions(
    const slang::ast::TimingControl &ctrl) {
  if (auto *signalCtrl = ctrl.as_if<slang::ast::SignalEventControl>()) {
    auto symRef = signalCtrl->expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      return getCanonicalSignalEventControlForAssertions(
          clockingBlock.getEvent());
    }
    return signalCtrl;
  }
  if (auto *eventList = ctrl.as_if<slang::ast::EventListControl>()) {
    if (eventList->events.size() != 1)
      return nullptr;
    auto *event = *eventList->events.begin();
    if (!event)
      return nullptr;
    return getCanonicalSignalEventControlForAssertions(*event);
  }
  return nullptr;
}

static Value cloneValueIntoBlock(Value value, OpBuilder &builder,
                                 IRMapping &mapping) {
  if (!value)
    return value;
  // Reference-like values (variables, nets, etc.) must remain shared objects;
  // cloning them into nested wait regions creates disconnected state.
  if (isa<moore::RefType>(value.getType()))
    return value;
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto *def = value.getDefiningOp()) {
    for (Value operand : def->getOperands())
      cloneValueIntoBlock(operand, builder, mapping);
    builder.clone(*def, mapping);
    return mapping.lookup(value);
  }
  return value;
}

static moore::Edge convertClockEdge(const ltl::ClockEdge edge) {
  switch (edge) {
  case ltl::ClockEdge::Neg:
    return moore::Edge::NegEdge;
  case ltl::ClockEdge::Pos:
    return moore::Edge::PosEdge;
  case ltl::ClockEdge::Both:
    return moore::Edge::BothEdges;
  }
  return moore::Edge::AnyChange;
}

static StringRef formatClockEdge(ltl::ClockEdge edge) {
  switch (edge) {
  case ltl::ClockEdge::Pos:
    return "posedge";
  case ltl::ClockEdge::Neg:
    return "negedge";
  case ltl::ClockEdge::Both:
    return "both";
  }
  return "both";
}

static void maybeAddEventExprAttr(OpBuilder &builder,
                                  SmallVectorImpl<NamedAttribute> &detailAttrs,
                                  StringRef key,
                                  const slang::ast::Expression *expr) {
  if (!expr || !expr->syntax)
    return;
  auto text = expr->syntax->toString();
  if (text.empty())
    return;
  detailAttrs.push_back(
      builder.getNamedAttr(key, builder.getStringAttr(text)));
}

struct StructuredEventExprInfo {
  StringRef baseName;
  std::optional<unsigned> lsb;
  std::optional<unsigned> msb;
  std::optional<StringRef> dynIndexName;
  std::optional<int32_t> dynSign;
  std::optional<int32_t> dynOffset;
  std::optional<unsigned> dynWidth;
  bool logicalNot = false;
  bool bitwiseNot = false;
  std::optional<StringRef> reduction;
};

struct AffineIndexInfo {
  StringRef name;
  int64_t scale = 1;
  int64_t offset = 0;
};

static std::optional<int32_t>
getConstInt32(const slang::ast::Expression &expr) {
  auto *constant = expr.getConstant();
  if (!constant || !constant->isInteger())
    return std::nullopt;
  if (constant->integer().hasUnknown())
    return std::nullopt;
  return constant->integer().as<int32_t>();
}

static bool extractAffineIndexInfo(const slang::ast::Expression &expr,
                                   AffineIndexInfo &info) {
  using slang::ast::BinaryOperator;
  using slang::ast::UnaryOperator;

  if (auto *conv = expr.as_if<slang::ast::ConversionExpression>())
    return extractAffineIndexInfo(conv->operand(), info);

  if (auto *symRef = expr.getSymbolReference()) {
    info.name = symRef->name;
    info.scale = 1;
    info.offset = 0;
    return true;
  }

  if (auto *unary = expr.as_if<slang::ast::UnaryExpression>()) {
    if (unary->op == UnaryOperator::Plus)
      return extractAffineIndexInfo(unary->operand(), info);
    if (unary->op == UnaryOperator::Minus) {
      if (!extractAffineIndexInfo(unary->operand(), info))
        return false;
      info.scale = -info.scale;
      info.offset = -info.offset;
      return true;
    }
    return false;
  }

  if (auto *binary = expr.as_if<slang::ast::BinaryExpression>()) {
    if (binary->op != BinaryOperator::Add &&
        binary->op != BinaryOperator::Subtract)
      return false;

    auto rhsConst = getConstInt32(binary->right());
    if (rhsConst && extractAffineIndexInfo(binary->left(), info)) {
      info.offset += binary->op == BinaryOperator::Add ? *rhsConst : -*rhsConst;
      return true;
    }

    auto lhsConst = getConstInt32(binary->left());
    AffineIndexInfo rhsInfo;
    if (lhsConst && extractAffineIndexInfo(binary->right(), rhsInfo)) {
      if (binary->op == BinaryOperator::Add) {
        rhsInfo.offset += *lhsConst;
      } else {
        rhsInfo.scale = -rhsInfo.scale;
        rhsInfo.offset = *lhsConst - rhsInfo.offset;
      }
      info = rhsInfo;
      return true;
    }
  }

  return false;
}

static bool extractStructuredEventExprInfo(const slang::ast::Expression &expr,
                                           StructuredEventExprInfo &info) {
  using slang::ast::RangeSelectionKind;
  using slang::ast::UnaryOperator;

  if (auto *unary = expr.as_if<slang::ast::UnaryExpression>()) {
    std::optional<StringRef> reduction;
    switch (unary->op) {
    case UnaryOperator::BitwiseAnd:
      reduction = "and";
      break;
    case UnaryOperator::BitwiseNand:
      reduction = "nand";
      break;
    case UnaryOperator::BitwiseOr:
      reduction = "or";
      break;
    case UnaryOperator::BitwiseNor:
      reduction = "nor";
      break;
    case UnaryOperator::BitwiseXor:
      reduction = "xor";
      break;
    case UnaryOperator::BitwiseXnor:
      reduction = "xnor";
      break;
    case UnaryOperator::LogicalNot:
      info.logicalNot = !info.logicalNot;
      return extractStructuredEventExprInfo(unary->operand(), info);
    case UnaryOperator::BitwiseNot:
      info.bitwiseNot = !info.bitwiseNot;
      return extractStructuredEventExprInfo(unary->operand(), info);
    default:
      break;
    }
    if (reduction) {
      if (info.reduction)
        return false;
      info.reduction = *reduction;
      return extractStructuredEventExprInfo(unary->operand(), info);
    }
  }

  if (auto *element = expr.as_if<slang::ast::ElementSelectExpression>()) {
    if (info.lsb || info.msb || info.dynIndexName)
      return false;
    auto index = getConstInt32(element->selector());
    auto range = element->value().type->getFixedRange();
    if (index) {
      int32_t bit = range.translateIndex(*index);
      if (bit < 0)
        return false;
      info.lsb = static_cast<unsigned>(bit);
      info.msb = static_cast<unsigned>(bit);
      return extractStructuredEventExprInfo(element->value(), info);
    }
    AffineIndexInfo indexInfo;
    if (!extractAffineIndexInfo(element->selector(), indexInfo))
      return false;
    int64_t sign = range.isLittleEndian() ? indexInfo.scale : -indexInfo.scale;
    int64_t offset = range.isLittleEndian() ? indexInfo.offset - range.lower()
                                            : range.upper() - indexInfo.offset;
    if ((sign != 1 && sign != -1) ||
        offset < std::numeric_limits<int32_t>::min() ||
        offset > std::numeric_limits<int32_t>::max())
      return false;
    info.dynIndexName = indexInfo.name;
    info.dynSign = static_cast<int32_t>(sign);
    info.dynOffset = static_cast<int32_t>(offset);
    info.dynWidth = 1;
    return extractStructuredEventExprInfo(element->value(), info);
  }

  if (auto *rangeSelect = expr.as_if<slang::ast::RangeSelectExpression>()) {
    if (info.lsb || info.msb || info.dynIndexName)
      return false;
    auto left = getConstInt32(rangeSelect->left());
    auto right = getConstInt32(rangeSelect->right());
    auto range = rangeSelect->value().type->getFixedRange();
    switch (rangeSelect->getSelectionKind()) {
    case RangeSelectionKind::Simple: {
      if (!left || !right)
        return false;
      int32_t lsbIndex = *right;
      uint64_t width = static_cast<uint64_t>(
                           std::abs(static_cast<int64_t>(*left) -
                                    static_cast<int64_t>(*right))) +
                       1;
      if (width == 0)
        return false;
      int32_t lsb = range.translateIndex(lsbIndex);
      if (lsb < 0)
        return false;
      uint64_t msb = static_cast<uint64_t>(lsb) + width - 1;
      if (msb > std::numeric_limits<unsigned>::max())
        return false;
      info.lsb = static_cast<unsigned>(lsb);
      info.msb = static_cast<unsigned>(msb);
      return extractStructuredEventExprInfo(rangeSelect->value(), info);
    }
    case RangeSelectionKind::IndexedUp:
    case RangeSelectionKind::IndexedDown: {
      if (!right || *right <= 0)
        return false;
      AffineIndexInfo indexInfo;
      if (!extractAffineIndexInfo(rangeSelect->left(), indexInfo))
        return false;
      int64_t width = *right;
      int64_t b = indexInfo.offset;
      if (rangeSelect->getSelectionKind() == RangeSelectionKind::IndexedUp) {
        if (!range.isLittleEndian())
          b += width - 1;
      } else {
        if (range.isLittleEndian())
          b -= width - 1;
      }

      int64_t sign =
          range.isLittleEndian() ? indexInfo.scale : -indexInfo.scale;
      int64_t offset = range.isLittleEndian() ? b - range.lower()
                                              : range.upper() - b;
      if ((sign != 1 && sign != -1) ||
          offset < std::numeric_limits<int32_t>::min() ||
          offset > std::numeric_limits<int32_t>::max())
        return false;

      info.dynIndexName = indexInfo.name;
      info.dynSign = static_cast<int32_t>(sign);
      info.dynOffset = static_cast<int32_t>(offset);
      info.dynWidth = static_cast<unsigned>(width);
      return extractStructuredEventExprInfo(rangeSelect->value(), info);
    }
    }
  }

  if (auto *symRef = expr.getSymbolReference()) {
    info.baseName = symRef->name;
    return true;
  }
  return false;
}

static std::optional<StringRef>
getStructuredBinaryEventOp(const slang::ast::BinaryExpression &expr) {
  using slang::ast::BinaryOperator;
  switch (expr.op) {
  case BinaryOperator::BinaryAnd:
  case BinaryOperator::LogicalAnd:
    return StringRef("and");
  case BinaryOperator::BinaryOr:
  case BinaryOperator::LogicalOr:
    return StringRef("or");
  case BinaryOperator::BinaryXor:
    return StringRef("xor");
  case BinaryOperator::LogicalImplication:
    return StringRef("implies");
  case BinaryOperator::LogicalEquivalence:
    return StringRef("iff");
  case BinaryOperator::Equality:
  case BinaryOperator::CaseEquality:
  case BinaryOperator::WildcardEquality:
    return StringRef("eq");
  case BinaryOperator::Inequality:
  case BinaryOperator::CaseInequality:
  case BinaryOperator::WildcardInequality:
    return StringRef("ne");
  default:
    return std::nullopt;
  }
}

static std::optional<StringRef>
getStructuredUnaryEventOp(const slang::ast::UnaryExpression &expr) {
  using slang::ast::UnaryOperator;
  switch (expr.op) {
  case UnaryOperator::LogicalNot:
    return StringRef("not");
  case UnaryOperator::BitwiseNot:
    return StringRef("bitwise_not");
  case UnaryOperator::BitwiseAnd:
    return StringRef("reduce_and");
  case UnaryOperator::BitwiseNand:
    return StringRef("reduce_nand");
  case UnaryOperator::BitwiseOr:
    return StringRef("reduce_or");
  case UnaryOperator::BitwiseNor:
    return StringRef("reduce_nor");
  case UnaryOperator::BitwiseXor:
    return StringRef("reduce_xor");
  case UnaryOperator::BitwiseXnor:
    return StringRef("reduce_xnor");
  default:
    return std::nullopt;
  }
}

static bool maybeAddStructuredEventExprAttrs(
    OpBuilder &builder, SmallVectorImpl<NamedAttribute> &detailAttrs,
    StringRef prefix, const slang::ast::Expression *expr) {
  if (!expr)
    return false;
  auto addAttrIfMissing = [&](StringRef key, Attribute value) {
    if (llvm::any_of(detailAttrs, [&](NamedAttribute namedAttr) {
          return namedAttr.getName().strref() == key;
        }))
      return;
    detailAttrs.push_back(builder.getNamedAttr(key, value));
  };
  auto keyFor = [&](StringRef suffix) {
    std::string key = prefix.str();
    key += "_";
    key += suffix.str();
    return key;
  };

  bool emittedAny = false;
  StructuredEventExprInfo info;
  if (extractStructuredEventExprInfo(*expr, info) && !info.baseName.empty()) {
    emittedAny = true;
    addAttrIfMissing(keyFor("name"), builder.getStringAttr(info.baseName));
    if (info.lsb && info.msb) {
      addAttrIfMissing(keyFor("lsb"), builder.getI32IntegerAttr(*info.lsb));
      addAttrIfMissing(keyFor("msb"), builder.getI32IntegerAttr(*info.msb));
    }
    if (info.dynIndexName && info.dynSign && info.dynOffset && info.dynWidth) {
      addAttrIfMissing(keyFor("dyn_index_name"),
                       builder.getStringAttr(*info.dynIndexName));
      addAttrIfMissing(keyFor("dyn_sign"),
                       builder.getI32IntegerAttr(*info.dynSign));
      addAttrIfMissing(keyFor("dyn_offset"),
                       builder.getI32IntegerAttr(*info.dynOffset));
      addAttrIfMissing(keyFor("dyn_width"),
                       builder.getI32IntegerAttr(*info.dynWidth));
    }
    if (info.logicalNot)
      addAttrIfMissing(keyFor("logical_not"), builder.getBoolAttr(true));
    if (info.bitwiseNot)
      addAttrIfMissing(keyFor("bitwise_not"), builder.getBoolAttr(true));
    if (info.reduction)
      addAttrIfMissing(keyFor("reduction"),
                       builder.getStringAttr(*info.reduction));
  }

  if (auto *unary = expr->as_if<slang::ast::UnaryExpression>()) {
    if (auto unaryOp = getStructuredUnaryEventOp(*unary)) {
      addAttrIfMissing(keyFor("unary_op"), builder.getStringAttr(*unaryOp));
      std::string argPrefix = (prefix + "_arg").str();
      if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, argPrefix,
                                            &unary->operand()))
        return false;
      emittedAny = true;
    }
  }

  if (auto *binary = expr->as_if<slang::ast::BinaryExpression>()) {
    auto binaryOp = getStructuredBinaryEventOp(*binary);
    if (binaryOp) {
      addAttrIfMissing(keyFor("bin_op"), builder.getStringAttr(*binaryOp));
      std::string lhsPrefix = (prefix + "_lhs").str();
      std::string rhsPrefix = (prefix + "_rhs").str();
      if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, lhsPrefix,
                                            &binary->left()))
        return false;
      if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, rhsPrefix,
                                            &binary->right()))
        return false;
      emittedAny = true;
    }
  }

  return emittedAny;
}

static void recordMixedEventSourcesOnModule(Context &context,
                                            ArrayAttr sources,
                                            ArrayAttr details = {}) {
  Block *block = context.builder.getInsertionBlock();
  if (!block)
    return;
  auto module = block->getParentOp()->getParentOfType<moore::SVModuleOp>();
  if (!module || !sources)
    return;

  SmallVector<Attribute, 4> entries;
  if (auto existing = module->getAttrOfType<ArrayAttr>("moore.event_sources"))
    entries.append(existing.begin(), existing.end());
  else if (auto existing =
               module->getAttrOfType<ArrayAttr>("moore.mixed_event_sources"))
    entries.append(existing.begin(), existing.end());
  entries.push_back(sources);
  auto entriesAttr = ArrayAttr::get(module->getContext(), entries);
  module->setAttr("moore.event_sources", entriesAttr);
  // Keep legacy naming for compatibility with in-flight downstream users/tests.
  module->setAttr("moore.mixed_event_sources", entriesAttr);
  if (details) {
    SmallVector<Attribute, 4> detailEntries;
    if (auto existing =
            module->getAttrOfType<ArrayAttr>("moore.event_source_details"))
      detailEntries.append(existing.begin(), existing.end());
    detailEntries.push_back(details);
    module->setAttr("moore.event_source_details",
                    ArrayAttr::get(module->getContext(), detailEntries));
  }
}

static LogicalResult lowerClockedSequenceEventControl(
    Context &context, Location loc, Value seqValue, Value clockValue,
    ltl::ClockEdge edge,
    std::span<const slang::ast::SignalEventControl *const> signalEvents = {},
    ArrayRef<StringAttr> sequenceSources = {},
    ArrayRef<DictionaryAttr> sequenceSourceDetails = {}) {
  OpBuilder &builder = context.builder;
  Block *startBlock = builder.getInsertionBlock();
  if (!startBlock)
    return mlir::emitError(loc) << "sequence event control requires a block";

  Region *parentRegion = startBlock->getParent();
  auto *loopBlock = new Block();
  auto *resumeBlock = new Block();
  parentRegion->getBlocks().insert(
      std::next(startBlock->getIterator()), loopBlock);
  parentRegion->getBlocks().insert(
      std::next(loopBlock->getIterator()), resumeBlock);

  builder.setInsertionPointToEnd(loopBlock);
  auto waitOp = moore::WaitEventOp::create(builder, loc);
  Block &waitBlock = waitOp.getBody().emplaceBlock();
  {
    OpBuilder waitBuilder(builder.getContext());
    waitBuilder.setInsertionPointToStart(&waitBlock);
    IRMapping mapping;
    Value clockInWait = cloneValueIntoBlock(clockValue, waitBuilder, mapping);
    auto mooreClockTy = moore::IntType::get(builder.getContext(), 1,
                                            moore::Domain::TwoValued);
    if (!isa<moore::IntType>(clockInWait.getType()))
      clockInWait = UnrealizedConversionCastOp::create(
          waitBuilder, loc, mooreClockTy, clockInWait)
                        ->getResult(0);
    moore::DetectEventOp::create(waitBuilder, loc, convertClockEdge(edge),
                                 clockInWait, Value{});
  }

  auto i1Type = builder.getI1Type();
  auto trueVal = hw::ConstantOp::create(builder, loc, i1Type, 1);
  auto falseVal = hw::ConstantOp::create(builder, loc, i1Type, 0);

  ltl::NFABuilder nfa(trueVal);
  auto fragment = nfa.build(seqValue, loc, builder);
  nfa.eliminateEpsilon();

  size_t numStates = nfa.states.size();
  if (numStates == 0)
    return mlir::emitError(loc) << "empty sequence event control";

  SmallVector<Value, 8> stateArgs;
  stateArgs.reserve(numStates);
  for (size_t i = 0; i < numStates; ++i)
    stateArgs.push_back(loopBlock->addArgument(i1Type, loc));

  // Clone sequence transition conditions into the loop body so they are
  // re-evaluated on every wakeup, not frozen at loop entry.
  IRMapping condMapping;
  SmallVector<Value, 8> loopConditions;
  loopConditions.reserve(nfa.conditions.size());
  for (Value cond : nfa.conditions)
    loopConditions.push_back(cloneValueIntoBlock(cond, builder, condMapping));

  SmallVector<SmallVector<SmallVector<Value, 4>, 4>, 8> incoming;
  incoming.resize(numStates);
  for (size_t from = 0; from < numStates; ++from) {
    for (auto &tr : nfa.states[from].transitions) {
      if (tr.isEpsilon)
        continue;
      incoming[tr.to].push_back(SmallVector<Value, 4>{
          stateArgs[from], loopConditions[tr.condIndex]});
    }
  }

  SmallVector<Value, 8> nextVals;
  nextVals.resize(numStates, falseVal);
  for (size_t i = 0; i < numStates; ++i) {
    SmallVector<Value, 8> orInputs;
    if (static_cast<int>(i) == fragment.start)
      orInputs.push_back(trueVal);
    for (auto &edgeVals : incoming[i]) {
      auto andVal = comb::AndOp::create(builder, loc, edgeVals, true);
      orInputs.push_back(andVal);
    }
    if (orInputs.empty())
      nextVals[i] = falseVal;
    else
      nextVals[i] = comb::OrOp::create(builder, loc, orInputs, true);
  }

  Value match = falseVal;
  SmallVector<Value, 8> accepting;
  for (size_t i = 0; i < numStates; ++i) {
    if (nfa.states[i].accepting)
      accepting.push_back(nextVals[i]);
  }
  if (!accepting.empty())
    match = comb::OrOp::create(builder, loc, accepting, true);

  // For mixed event lists `@(seq or signal_event)`, allow equivalent
  // same-clock signal events to trigger the wait independently of sequence
  // acceptance.
  if (!signalEvents.empty()) {
    SmallVector<Value, 4> signalMatches;
    signalMatches.push_back(match);
    for (auto [signalIdx, signalCtrl] : llvm::enumerate(signalEvents)) {
      Value condition = trueVal;
      if (signalCtrl && signalCtrl->iffCondition) {
        condition = context.convertRvalueExpression(*signalCtrl->iffCondition);
        condition = context.convertToBool(condition, Domain::TwoValued);
        if (!condition)
          return failure();
        condition = context.convertToI1(condition);
        if (!condition)
          return failure();
      }
      auto term = comb::AndOp::create(builder, loc,
                                      ValueRange{condition, trueVal}, true);
      signalMatches.push_back(term);
    }
    match = comb::OrOp::create(builder, loc, signalMatches, true);
  }

  cf::CondBranchOp::create(builder, loc, match, resumeBlock, ValueRange{},
                           loopBlock, nextVals);
  if (!signalEvents.empty() || !sequenceSources.empty()) {
    SmallVector<Attribute, 8> sources;
    SmallVector<Attribute, 8> details;
    if (!sequenceSources.empty()) {
      for (StringAttr source : sequenceSources)
        sources.push_back(source);
      for (DictionaryAttr detail : sequenceSourceDetails)
        details.push_back(detail);
    } else if (!signalEvents.empty()) {
      sources.push_back(builder.getStringAttr("sequence"));
      details.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("kind", builder.getStringAttr("sequence")),
           builder.getNamedAttr("label", builder.getStringAttr("sequence"))}));
    }
    for (auto [signalIdx, signalCtrl] : llvm::enumerate(signalEvents)) {
      std::string entry = "signal[" + std::to_string(signalIdx) + "]:" +
                          formatClockEdge(edge).str();
      if (signalCtrl && signalCtrl->iffCondition)
        entry += ":iff";
      sources.push_back(builder.getStringAttr(entry));
      SmallVector<NamedAttribute, 6> detailAttrs{
          builder.getNamedAttr("kind", builder.getStringAttr("signal")),
          builder.getNamedAttr("label", builder.getStringAttr(entry)),
          builder.getNamedAttr("edge",
                               builder.getStringAttr(formatClockEdge(edge))),
          builder.getNamedAttr("signal_index",
                               builder.getI32IntegerAttr(signalIdx))};
      if (signalCtrl) {
        if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "signal",
                                              &signalCtrl->expr))
          maybeAddEventExprAttr(builder, detailAttrs, "signal_expr",
                               &signalCtrl->expr);
        if (signalCtrl->iffCondition) {
          if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "iff",
                                                signalCtrl->iffCondition))
            maybeAddEventExprAttr(builder, detailAttrs, "iff_expr",
                                 signalCtrl->iffCondition);
        }
      }
      details.push_back(builder.getDictionaryAttr(detailAttrs));
    }
    auto sourcesAttr = builder.getArrayAttr(sources);
    auto detailsAttr = builder.getArrayAttr(details);
    waitOp->setAttr("moore.event_sources", sourcesAttr);
    waitOp->setAttr("moore.event_source_details", detailsAttr);
    recordMixedEventSourcesOnModule(context, sourcesAttr, detailsAttr);
  }

  builder.setInsertionPointToEnd(startBlock);
  auto initTrue = hw::ConstantOp::create(builder, loc, i1Type, 1);
  auto initFalse = hw::ConstantOp::create(builder, loc, i1Type, 0);
  SmallVector<Value, 8> initStates;
  initStates.reserve(numStates);
  for (size_t i = 0; i < numStates; ++i)
    initStates.push_back(static_cast<int>(i) == fragment.start ? initTrue
                                                               : initFalse);
  cf::BranchOp::create(builder, loc, loopBlock, initStates);
  builder.setInsertionPointToEnd(resumeBlock);
  return success();
}

static Value stripClockCasts(Value clock);
static bool equivalentClockSignals(Value lhs, Value rhs);

struct MultiClockSignalEventInfo {
  Value clock;
  ltl::ClockEdge edge;
  const slang::ast::Expression *iffCondition;
  const slang::ast::Expression *expr;
};

static Value computeClockTick(OpBuilder &builder, Location loc, Value prev,
                              Value curr, ltl::ClockEdge edge) {
  auto i1Type = builder.getI1Type();
  auto one = hw::ConstantOp::create(builder, loc, i1Type, 1);
  switch (edge) {
  case ltl::ClockEdge::Pos: {
    auto notPrev = comb::XorOp::create(builder, loc, prev, one);
    return comb::AndOp::create(builder, loc, ValueRange{notPrev, curr}, true);
  }
  case ltl::ClockEdge::Neg: {
    auto notCurr = comb::XorOp::create(builder, loc, curr, one);
    return comb::AndOp::create(builder, loc, ValueRange{prev, notCurr}, true);
  }
  case ltl::ClockEdge::Both:
    return comb::XorOp::create(builder, loc, prev, curr);
  }
  return Value{};
}

static bool isClockStutterCondition(Value condition, ArrayRef<Value> ticks) {
  auto xorOp = condition.getDefiningOp<comb::XorOp>();
  if (!xorOp)
    return false;
  auto inputs = xorOp.getInputs();
  if (inputs.size() != 2)
    return false;
  auto isOne = [](Value value) {
    if (auto cst = value.getDefiningOp<hw::ConstantOp>())
      return cst.getValue().isOne();
    return false;
  };
  Value tick;
  if (isOne(inputs[0]))
    tick = inputs[1];
  else if (isOne(inputs[1]))
    tick = inputs[0];
  else
    return false;
  return llvm::is_contained(ticks, tick);
}

static LogicalResult lowerMultiClockSequenceEventControl(Context &context,
                                                         Location loc,
                                                         Value seqValue,
                                                         ArrayRef<Value> clocks,
                                                         ArrayRef<MultiClockSignalEventInfo>
                                                             signalEvents = {},
                                                         ArrayRef<StringAttr>
                                                             sequenceSources = {},
                                                         ArrayRef<DictionaryAttr>
                                                             sequenceSourceDetails = {}) {
  OpBuilder &builder = context.builder;
  Block *startBlock = builder.getInsertionBlock();
  if (!startBlock)
    return mlir::emitError(loc) << "sequence event control requires a block";
  if (clocks.empty())
    return mlir::emitError(loc) << "multi-clock sequence event control requires "
                                   "at least one clock";

  Region *parentRegion = startBlock->getParent();
  auto *loopBlock = new Block();
  auto *resumeBlock = new Block();
  parentRegion->getBlocks().insert(
      std::next(startBlock->getIterator()), loopBlock);
  parentRegion->getBlocks().insert(
      std::next(loopBlock->getIterator()), resumeBlock);

  builder.setInsertionPointToEnd(loopBlock);
  auto waitOp = moore::WaitEventOp::create(builder, loc);
  Block &waitBlock = waitOp.getBody().emplaceBlock();
  {
    OpBuilder waitBuilder(builder.getContext());
    waitBuilder.setInsertionPointToStart(&waitBlock);
    IRMapping mapping;
    auto mooreClockTy = moore::IntType::get(builder.getContext(), 1,
                                            moore::Domain::TwoValued);
    for (Value clockValue : clocks) {
      Value clockInWait = cloneValueIntoBlock(clockValue, waitBuilder, mapping);
      if (!isa<moore::IntType>(clockInWait.getType()))
        clockInWait = UnrealizedConversionCastOp::create(
                          waitBuilder, loc, mooreClockTy, clockInWait)
                          ->getResult(0);
      // Wake on any change and derive edge-specific ticks in the loop body.
      moore::DetectEventOp::create(waitBuilder, loc, moore::Edge::AnyChange,
                                   clockInWait, Value{});
    }
  }

  auto i1Type = builder.getI1Type();
  auto trueVal = hw::ConstantOp::create(builder, loc, i1Type, 1);
  auto falseVal = hw::ConstantOp::create(builder, loc, i1Type, 0);

  size_t numClocks = clocks.size();
  SmallVector<BlockArgument, 4> prevClockArgs;
  prevClockArgs.reserve(numClocks);
  for (size_t i = 0; i < numClocks; ++i)
    prevClockArgs.push_back(loopBlock->addArgument(i1Type, loc));

  IRMapping clockMapping;
  SmallVector<Value, 4> currClockVals;
  currClockVals.reserve(numClocks);
  for (Value clockValue : clocks) {
    Value curr = cloneValueIntoBlock(clockValue, builder, clockMapping);
    curr = context.convertToI1(curr);
    if (!curr)
      return failure();
    currClockVals.push_back(curr);
  }

  SmallVector<Value, 4> posTicks;
  SmallVector<Value, 4> negTicks;
  SmallVector<Value, 4> bothTicks;
  posTicks.reserve(numClocks);
  negTicks.reserve(numClocks);
  bothTicks.reserve(numClocks);
  for (size_t i = 0; i < numClocks; ++i) {
    Value prev = prevClockArgs[i];
    Value curr = currClockVals[i];
    posTicks.push_back(computeClockTick(builder, loc, prev, curr,
                                        ltl::ClockEdge::Pos));
    negTicks.push_back(computeClockTick(builder, loc, prev, curr,
                                        ltl::ClockEdge::Neg));
    bothTicks.push_back(computeClockTick(builder, loc, prev, curr,
                                         ltl::ClockEdge::Both));
  }

  auto clockEdgePredicate = [&](Value clock, ltl::ClockEdge edge) -> Value {
    for (size_t i = 0; i < numClocks; ++i) {
      if (!equivalentClockSignals(clocks[i], clock))
        continue;
      switch (edge) {
      case ltl::ClockEdge::Pos:
        return posTicks[i];
      case ltl::ClockEdge::Neg:
        return negTicks[i];
      case ltl::ClockEdge::Both:
        return bothTicks[i];
      }
    }
    return Value{};
  };

  ltl::NFABuilder nfa(trueVal, clockEdgePredicate);
  auto fragment = nfa.build(seqValue, loc, builder);
  nfa.eliminateEpsilon();

  size_t numStates = nfa.states.size();
  if (numStates == 0)
    return mlir::emitError(loc) << "empty sequence event control";

  SmallVector<Value, 8> stateArgs;
  stateArgs.reserve(numStates);
  for (size_t i = 0; i < numStates; ++i)
    stateArgs.push_back(loopBlock->addArgument(i1Type, loc));

  // Clone sequence transition conditions into the loop body so they are
  // re-evaluated on every wakeup, not frozen at loop entry.
  IRMapping condMapping;
  for (Value tick : posTicks)
    condMapping.map(tick, tick);
  for (Value tick : negTicks)
    condMapping.map(tick, tick);
  for (Value tick : bothTicks)
    condMapping.map(tick, tick);
  SmallVector<Value, 8> loopConditions;
  loopConditions.reserve(nfa.conditions.size());
  for (Value cond : nfa.conditions)
    loopConditions.push_back(cloneValueIntoBlock(cond, builder, condMapping));

  SmallVector<SmallVector<SmallVector<Value, 4>, 4>, 8> incoming;
  incoming.resize(numStates);
  for (size_t from = 0; from < numStates; ++from) {
    for (auto &tr : nfa.states[from].transitions) {
      if (tr.isEpsilon)
        continue;
      incoming[tr.to].push_back(SmallVector<Value, 4>{
          stateArgs[from], loopConditions[tr.condIndex]});
    }
  }

  SmallVector<Value, 8> nextVals;
  nextVals.resize(numStates, falseVal);
  for (size_t i = 0; i < numStates; ++i) {
    SmallVector<Value, 8> orInputs;
    if (static_cast<int>(i) == fragment.start)
      orInputs.push_back(trueVal);
    for (auto &edgeVals : incoming[i]) {
      auto andVal = comb::AndOp::create(builder, loc, edgeVals, true);
      orInputs.push_back(andVal);
    }
    if (orInputs.empty())
      nextVals[i] = falseVal;
    else
      nextVals[i] = comb::OrOp::create(builder, loc, orInputs, true);
  }

  Value match = falseVal;
  SmallVector<Value, 12> allTicks;
  allTicks.reserve(posTicks.size() + negTicks.size() + bothTicks.size());
  allTicks.append(posTicks.begin(), posTicks.end());
  allTicks.append(negTicks.begin(), negTicks.end());
  allTicks.append(bothTicks.begin(), bothTicks.end());

  SmallVector<Value, 8> acceptingMatches;
  for (size_t from = 0; from < numStates; ++from) {
    for (auto &tr : nfa.states[from].transitions) {
      if (tr.isEpsilon || !nfa.states[tr.to].accepting)
        continue;
      Value condition = loopConditions[tr.condIndex];
      // Ignore pure stutter (`not tick`) transitions added by clock gating.
      // They keep states alive across unrelated clocks, but they should never
      // trigger new procedural wakeups by themselves.
      if (isClockStutterCondition(condition, allTicks))
        continue;
      acceptingMatches.push_back(comb::AndOp::create(
          builder, loc, ValueRange{stateArgs[from], condition}, true));
    }
  }
  if (!acceptingMatches.empty())
    match = comb::OrOp::create(builder, loc, acceptingMatches, true);

  if (!signalEvents.empty()) {
    SmallVector<Value, 8> mixedMatches;
    mixedMatches.push_back(match);
    for (auto [signalIdx, signalEvent] : llvm::enumerate(signalEvents)) {
      Value tick = clockEdgePredicate(signalEvent.clock, signalEvent.edge);
      if (!tick)
        return mlir::emitError(loc)
               << "mixed sequence/signal event list references a clock that "
                  "cannot be scheduled in multi-clock lowering";
      Value term = tick;
      if (signalEvent.iffCondition) {
        Value condition =
            context.convertRvalueExpression(*signalEvent.iffCondition);
        condition = context.convertToBool(condition, Domain::TwoValued);
        if (!condition)
          return failure();
        condition = context.convertToI1(condition);
        if (!condition)
          return failure();
        term = comb::AndOp::create(builder, loc, ValueRange{term, condition},
                                   true);
      }
      term = comb::AndOp::create(builder, loc, ValueRange{term, trueVal}, true);
      mixedMatches.push_back(term);
    }
    match = comb::OrOp::create(builder, loc, mixedMatches, true);
  }

  SmallVector<Value, 12> loopArgs;
  loopArgs.reserve(numStates + numClocks);
  loopArgs.append(currClockVals.begin(), currClockVals.end());
  loopArgs.append(nextVals.begin(), nextVals.end());
  cf::CondBranchOp::create(builder, loc, match, resumeBlock, ValueRange{},
                           loopBlock, loopArgs);
  if (!signalEvents.empty() || !sequenceSources.empty()) {
    SmallVector<Attribute, 8> sources;
    SmallVector<Attribute, 8> details;
    if (!sequenceSources.empty()) {
      for (StringAttr source : sequenceSources)
        sources.push_back(source);
      for (DictionaryAttr detail : sequenceSourceDetails)
        details.push_back(detail);
    } else if (!signalEvents.empty()) {
      sources.push_back(builder.getStringAttr("sequence"));
      details.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("kind", builder.getStringAttr("sequence")),
           builder.getNamedAttr("label", builder.getStringAttr("sequence"))}));
    }
    for (auto [signalIdx, signalEvent] : llvm::enumerate(signalEvents)) {
      std::string entry = "signal[" + std::to_string(signalIdx) + "]:" +
                          formatClockEdge(signalEvent.edge).str();
      if (signalEvent.iffCondition)
        entry += ":iff";
      sources.push_back(builder.getStringAttr(entry));
      SmallVector<NamedAttribute, 6> detailAttrs{
          builder.getNamedAttr("kind", builder.getStringAttr("signal")),
          builder.getNamedAttr("label", builder.getStringAttr(entry)),
          builder.getNamedAttr(
              "edge", builder.getStringAttr(formatClockEdge(signalEvent.edge))),
          builder.getNamedAttr("signal_index",
                               builder.getI32IntegerAttr(signalIdx))};
      if (signalEvent.expr) {
        if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "signal",
                                              signalEvent.expr))
          maybeAddEventExprAttr(builder, detailAttrs, "signal_expr",
                               signalEvent.expr);
      }
      if (signalEvent.iffCondition) {
        if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "iff",
                                              signalEvent.iffCondition))
          maybeAddEventExprAttr(builder, detailAttrs, "iff_expr",
                               signalEvent.iffCondition);
      }
      details.push_back(builder.getDictionaryAttr(detailAttrs));
    }
    auto sourcesAttr = builder.getArrayAttr(sources);
    auto detailsAttr = builder.getArrayAttr(details);
    waitOp->setAttr("moore.event_sources", sourcesAttr);
    waitOp->setAttr("moore.event_source_details", detailsAttr);
    recordMixedEventSourcesOnModule(context, sourcesAttr, detailsAttr);
  }

  builder.setInsertionPointToEnd(startBlock);
  SmallVector<Value, 8> initStates;
  initStates.reserve(numStates);
  auto initTrue = hw::ConstantOp::create(builder, loc, i1Type, 1);
  auto initFalse = hw::ConstantOp::create(builder, loc, i1Type, 0);
  for (size_t i = 0; i < numStates; ++i)
    initStates.push_back(static_cast<int>(i) == fragment.start ? initTrue
                                                               : initFalse);
  IRMapping initClockMapping;
  SmallVector<Value, 4> initClocks;
  initClocks.reserve(numClocks);
  for (Value clockValue : clocks) {
    Value initClock = cloneValueIntoBlock(clockValue, builder, initClockMapping);
    initClock = context.convertToI1(initClock);
    if (!initClock)
      return failure();
    initClocks.push_back(initClock);
  }
  SmallVector<Value, 12> initArgs;
  initArgs.reserve(numStates + numClocks);
  initArgs.append(initClocks.begin(), initClocks.end());
  initArgs.append(initStates.begin(), initStates.end());
  cf::BranchOp::create(builder, loc, loopBlock, initArgs);
  builder.setInsertionPointToEnd(resumeBlock);
  return success();
}

static LogicalResult lowerSequenceEventControl(Context &context, Location loc,
                                               const slang::ast::Expression &expr,
                                               const slang::ast::Expression *iffExpr) {
  OpBuilder &builder = context.builder;
  const slang::ast::Expression *assertionExpr = &expr;
  if (auto symRef = expr.getSymbolReference()) {
    if (symRef->kind == slang::ast::SymbolKind::Sequence ||
        symRef->kind == slang::ast::SymbolKind::Property ||
        symRef->kind == slang::ast::SymbolKind::LetDecl) {
      assertionExpr = &slang::ast::AssertionInstanceExpression::makeDefault(
          *symRef);
    }
  }

  Value rootValue = context.convertRvalueExpression(*assertionExpr);
  if (!rootValue)
    return failure();
  if (isa<ltl::PropertyType>(rootValue.getType()))
    return mlir::emitError(loc)
           << "property event controls are not yet supported";

  Value clockedValue = rootValue;
  if (!clockedValue.getDefiningOp<ltl::ClockOp>()) {
    if (context.currentScope) {
      if (auto *clocking = context.compilation.getDefaultClocking(
              *context.currentScope)) {
        if (auto *clockBlock =
                clocking->as_if<slang::ast::ClockingBlockSymbol>())
          clockedValue =
              context.convertLTLTimingControl(clockBlock->getEvent(),
                                              clockedValue);
      }
    }
  }

  auto clockOp = clockedValue.getDefiningOp<ltl::ClockOp>();
  if (!clockOp)
    return mlir::emitError(loc)
           << "sequence event control requires a clocking event";

  Value seqValue = clockOp.getInput();
  if (iffExpr) {
    Value condition = context.convertRvalueExpression(*iffExpr);
    condition = context.convertToBool(condition, Domain::TwoValued);
    if (!condition)
      return failure();
    condition = context.convertToI1(condition);
    if (!condition)
      return failure();
    seqValue = ltl::AndOp::create(builder, loc,
                                  SmallVector<Value, 2>{seqValue, condition});
  }
  Value clockValue = clockOp.getClock();
  auto edge = clockOp.getEdge();
  auto result =
      lowerClockedSequenceEventControl(context, loc, seqValue, clockValue, edge);
  eraseLTLDeadOps(clockedValue);
  return result;
}

static Value stripClockCasts(Value clock) {
  while (clock) {
    if (auto toBool = clock.getDefiningOp<moore::ToBuiltinBoolOp>()) {
      clock = toBool.getInput();
      continue;
    }
    if (auto boolCast = clock.getDefiningOp<moore::BoolCastOp>()) {
      clock = boolCast.getInput();
      continue;
    }
    if (auto conv = clock.getDefiningOp<moore::ConversionOp>()) {
      clock = conv.getInput();
      continue;
    }
    if (auto cast = clock.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        clock = cast->getOperand(0);
        continue;
      }
    }
    break;
  }
  return clock;
}

static bool equivalentClockSignals(Value lhs, Value rhs) {
  lhs = stripClockCasts(lhs);
  rhs = stripClockCasts(rhs);
  if (lhs == rhs)
    return true;
  auto lhsRead = lhs.getDefiningOp<moore::ReadOp>();
  auto rhsRead = rhs.getDefiningOp<moore::ReadOp>();
  if (lhsRead && rhsRead && lhsRead.getInput() == rhsRead.getInput())
    return true;
  return false;
}

static LogicalResult
lowerSequenceEventListControl(Context &context, Location loc,
                              const slang::ast::EventListControl &ctrl) {
  OpBuilder &builder = context.builder;
  if (ctrl.events.empty())
    return mlir::emitError(loc) << "empty event list control";

  SmallVector<const slang::ast::SignalEventControl *, 4> sequenceEvents;
  SmallVector<const slang::ast::SignalEventControl *, 4> signalEvents;
  for (const auto *event : ctrl.events) {
    auto *signalCtrl = event ? event->as_if<slang::ast::SignalEventControl>()
                             : nullptr;
    if (!signalCtrl)
      return mlir::emitError(loc)
             << "unsupported event kind in sequence event list";
    if (isAssertionEventControl(signalCtrl->expr))
      sequenceEvents.push_back(signalCtrl);
    else
      signalEvents.push_back(signalCtrl);
  }
  if (sequenceEvents.empty())
    return mlir::emitError(loc)
           << "sequence event-list lowering requires at least one "
              "sequence/property event";

  std::optional<ltl::ClockEdge> commonEdge;
  Value commonClock;
  bool sameClockAndEdge = true;
  SmallVector<Value, 4> sequenceInputs;
  SmallVector<Value, 4> clockedInputs;
  SmallVector<Value, 4> sequenceClocks;
  SmallVector<Value, 4> clockedValues;
  SmallVector<StringAttr, 4> sequenceSourceAttrs;
  SmallVector<DictionaryAttr, 4> sequenceSourceDetailAttrs;
  SmallVector<const slang::ast::SignalEventControl *, 4> equivalentSignals;
  SmallVector<const slang::ast::SignalEventControl *, 4> parsedSignalEvents;
  SmallVector<MultiClockSignalEventInfo, 4> multiClockSignals;
  bool useGenericSequenceLabel =
      !signalEvents.empty() && sequenceEvents.size() == 1;

  for (auto [seqIdx, signalCtrl] : llvm::enumerate(sequenceEvents)) {
    std::string source = useGenericSequenceLabel
                             ? std::string("sequence")
                             : "sequence[" + std::to_string(seqIdx) + "]";
    if (signalCtrl->iffCondition)
      source += ":iff";
    sequenceSourceAttrs.push_back(builder.getStringAttr(source));
    SmallVector<NamedAttribute, 4> detailAttrs{
        builder.getNamedAttr("kind", builder.getStringAttr("sequence")),
        builder.getNamedAttr("label", builder.getStringAttr(source)),
        builder.getNamedAttr("sequence_index",
                             builder.getI32IntegerAttr(seqIdx))};
    if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "sequence",
                                          &signalCtrl->expr))
      maybeAddEventExprAttr(builder, detailAttrs, "sequence_expr",
                           &signalCtrl->expr);
    if (signalCtrl->iffCondition) {
      if (!maybeAddStructuredEventExprAttrs(builder, detailAttrs, "iff",
                                            signalCtrl->iffCondition))
        maybeAddEventExprAttr(builder, detailAttrs, "iff_expr",
                             signalCtrl->iffCondition);
    }
    sequenceSourceDetailAttrs.push_back(builder.getDictionaryAttr(detailAttrs));

    if (signalCtrl->edge != slang::ast::EdgeKind::None)
      return mlir::emitError(loc)
             << "sequence event controls in event lists do not support edge "
                "qualifiers";

    const slang::ast::Expression *assertionExpr = &signalCtrl->expr;
    if (auto symRef = signalCtrl->expr.getSymbolReference()) {
      if (symRef->kind == slang::ast::SymbolKind::Sequence ||
          symRef->kind == slang::ast::SymbolKind::Property ||
          symRef->kind == slang::ast::SymbolKind::LetDecl) {
        assertionExpr = &slang::ast::AssertionInstanceExpression::makeDefault(
            *symRef);
      }
    }

    Value rootValue = context.convertRvalueExpression(*assertionExpr);
    if (!rootValue)
      return failure();
    if (isa<ltl::PropertyType>(rootValue.getType()))
      return mlir::emitError(loc)
             << "property event controls are not yet supported";

    Value clockedValue = rootValue;
    if (!clockedValue.getDefiningOp<ltl::ClockOp>()) {
      if (context.currentScope) {
        if (auto *clocking = context.compilation.getDefaultClocking(
                *context.currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockedValue =
                context.convertLTLTimingControl(clockBlock->getEvent(),
                                                clockedValue);
        }
      }
    }

    auto clockOp = clockedValue.getDefiningOp<ltl::ClockOp>();
    if (!clockOp)
      return mlir::emitError(loc)
             << "sequence event control requires a clocking event";

    if (!commonEdge) {
      commonEdge = clockOp.getEdge();
      commonClock = clockOp.getClock();
    } else if (*commonEdge != clockOp.getEdge() ||
               !equivalentClockSignals(commonClock, clockOp.getClock())) {
      sameClockAndEdge = false;
    }

    Value sequenceInput = clockOp.getInput();
    if (signalCtrl->iffCondition) {
      Value condition = context.convertRvalueExpression(*signalCtrl->iffCondition);
      condition = context.convertToBool(condition, Domain::TwoValued);
      if (!condition)
        return failure();
      condition = context.convertToI1(condition);
      if (!condition)
        return failure();
      sequenceInput = ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{sequenceInput, condition});
      clockedValue = ltl::ClockOp::create(builder, loc, sequenceInput,
                                          clockOp.getEdge(), clockOp.getClock());
    }
    sequenceInputs.push_back(sequenceInput);
    clockedInputs.push_back(clockedValue);
    sequenceClocks.push_back(clockOp.getClock());
    clockedValues.push_back(clockedValue);
  }

  for (auto *signalCtrl : signalEvents) {
    auto signalEdge = convertEdgeKindLTL(signalCtrl->edge);
    Value signalClock = context.convertRvalueExpression(signalCtrl->expr);
    signalClock = context.convertToI1(signalClock);
    if (!signalClock)
      return failure();
    if (sameClockAndEdge &&
        (signalEdge != *commonEdge ||
         !equivalentClockSignals(commonClock, signalClock)))
      sameClockAndEdge = false;
    parsedSignalEvents.push_back(signalCtrl);
    multiClockSignals.push_back(MultiClockSignalEventInfo{
        signalClock, signalEdge, signalCtrl->iffCondition, &signalCtrl->expr});
  }
  if (sameClockAndEdge)
    equivalentSignals = parsedSignalEvents;
  ArrayRef<StringAttr> sequenceSources = sequenceSourceAttrs;
  ArrayRef<DictionaryAttr> sequenceSourceDetails = sequenceSourceDetailAttrs;

  LogicalResult result = failure();
  if (sameClockAndEdge) {
    Value combinedSequence = sequenceInputs.front();
    if (sequenceInputs.size() > 1)
      combinedSequence = ltl::OrOp::create(builder, loc, sequenceInputs);
    result = lowerClockedSequenceEventControl(context, loc, combinedSequence,
                                              commonClock, *commonEdge,
                                              equivalentSignals,
                                              sequenceSources,
                                              sequenceSourceDetails);
  } else {
    Value combinedSequence = clockedInputs.front();
    if (clockedInputs.size() > 1)
      combinedSequence = ltl::OrOp::create(builder, loc, clockedInputs);
    SmallVector<Value, 4> uniqueClocks;
    for (Value clock : sequenceClocks) {
      bool exists = llvm::any_of(uniqueClocks, [&](Value known) {
        return equivalentClockSignals(known, clock);
      });
      if (!exists)
        uniqueClocks.push_back(clock);
    }
    for (const auto &signalEvent : multiClockSignals) {
      bool exists = llvm::any_of(uniqueClocks, [&](Value known) {
        return equivalentClockSignals(known, signalEvent.clock);
      });
      if (!exists)
        uniqueClocks.push_back(signalEvent.clock);
    }
    result = lowerMultiClockSequenceEventControl(context, loc, combinedSequence,
                                                 uniqueClocks, multiClockSignals,
                                                 sequenceSources,
                                                 sequenceSourceDetails);
  }
  for (Value clockedValue : clockedValues)
    eraseLTLDeadOps(clockedValue);
  return result;
}

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
    // If the event expression is a timing-control assertion port, substitute
    // the bound timing control directly.
    if (context.inAssertionExpr) {
      if (auto *symRef = ctrl.expr.getSymbolReference()) {
        if (auto *port =
                symRef->as_if<slang::ast::AssertionPortSymbol>()) {
          if (auto *binding = context.lookupAssertionPortBinding(port)) {
            if (binding->kind ==
                    AssertionPortBinding::Kind::TimingControl &&
                binding->timingControl) {
              auto nestedLoc =
                  context.convertLocation(binding->timingControl->sourceRange);
              auto visitor = *this;
              visitor.loc = nestedLoc;
              return binding->timingControl->visit(visitor);
            }
          }
        }
      }
    }

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
      condition = context.convertToI1(condition);
      if (!condition)
        return Value{};
    }
    expr = context.convertToI1(expr);
    if (!expr)
      return Value{};
    if (condition) {
      seqOrPro = ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{condition, seqOrPro});
    }
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
  case TimingControlKind::SignalEvent: {
    auto &signalCtrl = ctrl.as<slang::ast::SignalEventControl>();

    // Check if the expression references a clocking block.
    // In that case, we need to convert the clocking block's clock event instead.
    auto symRef = signalCtrl.expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      auto &clockEvent = clockingBlock.getEvent();
      return handleRoot(context, clockEvent, implicitWaitOp);
    }

    if (isAssertionEventControl(signalCtrl.expr)) {
      if (signalCtrl.edge != slang::ast::EdgeKind::None)
        return mlir::emitError(loc)
               << "sequence event controls do not support edge qualifiers";
      return lowerSequenceEventControl(context, loc, signalCtrl.expr,
                                       signalCtrl.iffCondition);
    }

    auto waitOp = moore::WaitEventOp::create(builder, loc);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());
    EventControlVisitor visitor{context, loc, builder};
    return ctrl.visit(visitor);
  }
  case TimingControlKind::EventList: {
    auto &eventListCtrl = ctrl.as<slang::ast::EventListControl>();
    bool hasAssertionEvents = false;
    for (const auto *event : eventListCtrl.events) {
      auto *signalCtrl = event ? event->as_if<slang::ast::SignalEventControl>()
                               : nullptr;
      bool isAssertion = signalCtrl && isAssertionEventControl(signalCtrl->expr);
      hasAssertionEvents |= isAssertion;
    }
    if (hasAssertionEvents) {
      return lowerSequenceEventListControl(context, loc, eventListCtrl);
    }
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
    if (auto *signalCtrl = getCanonicalSignalEventControlForAssertions(ctrl)) {
      if (!signalCtrl->iffCondition &&
          !isAssertionEventControl(signalCtrl->expr))
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
