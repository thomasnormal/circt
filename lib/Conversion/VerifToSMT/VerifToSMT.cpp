//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

constexpr const char kWeakEventuallyAttr[] = "ltl.weak";

static Value gatePropertyWithEnable(Value property, Value enable, bool isCover,
                                    OpBuilder &builder, Location loc) {
  if (!enable)
    return property;
  if (isCover)
    return ltl::AndOp::create(builder, loc,
                              SmallVector<Value, 2>{enable, property})
        .getResult();
  auto notEnable = ltl::NotOp::create(builder, loc, enable);
  return ltl::OrOp::create(builder, loc,
                           SmallVector<Value, 2>{notEnable, property})
      .getResult();
}

static Value gateSMTWithEnable(Value property, Value enable, bool isCover,
                               OpBuilder &builder, Location loc) {
  if (!enable)
    return property;
  if (isCover)
    return smt::AndOp::create(builder, loc, enable, property);
  auto notEnable = smt::NotOp::create(builder, loc, enable);
  return smt::OrOp::create(builder, loc, notEnable, property);
}

static void maybeAssertKnownInput(Type originalTy, Value smtVal, Location loc,
                                  OpBuilder &builder) {
  auto structTy = dyn_cast<hw::StructType>(originalTy);
  if (!structTy)
    return;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return;
  if (!elements[0].name || !elements[1].name)
    return;
  if (elements[0].name.getValue() != "value" ||
      elements[1].name.getValue() != "unknown")
    return;
  int64_t valueWidth = hw::getBitWidth(elements[0].type);
  int64_t unknownWidth = hw::getBitWidth(elements[1].type);
  if (valueWidth <= 0 || unknownWidth <= 0 || valueWidth != unknownWidth)
    return;
  auto bvTy = dyn_cast<smt::BitVectorType>(smtVal.getType());
  if (!bvTy || bvTy.getWidth() !=
                   static_cast<unsigned>(valueWidth + unknownWidth))
    return;
  auto unkTy = smt::BitVectorType::get(builder.getContext(), unknownWidth);
  auto unknownBits = smt::ExtractOp::create(builder, loc, unkTy, 0, smtVal);
  auto zero = smt::BVConstantOp::create(builder, loc, 0, unknownWidth);
  auto isKnown = smt::EqOp::create(builder, loc, unknownBits, zero);
  smt::AssertOp::create(builder, loc, isKnown);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// LTL Operation Conversion Patterns
//===----------------------------------------------------------------------===//

static Value materializeSMTBool(Value input, const TypeConverter &converter,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  if (!input)
    return Value();
  if (isa<smt::BoolType>(input.getType()))
    return input;
  if (auto bvTy = dyn_cast<smt::BitVectorType>(input.getType());
      bvTy && bvTy.getWidth() == 1) {
    auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    return smt::EqOp::create(rewriter, loc, input, one);
  }
  if (auto intTy = dyn_cast<IntegerType>(input.getType());
      intTy && intTy.getWidth() == 1) {
    auto bvTy = smt::BitVectorType::get(rewriter.getContext(), 1);
    Value bv =
        converter.materializeTargetConversion(rewriter, loc, bvTy, input);
    if (!bv)
      return Value();
    auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    return smt::EqOp::create(rewriter, loc, bv, one);
  }
  return converter.materializeTargetConversion(
      rewriter, loc, smt::BoolType::get(rewriter.getContext()), input);
}

/// Convert ltl.and to smt.and
struct LTLAndOpConversion : OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.or to smt.or
struct LTLOrOpConversion : OpConversionPattern<ltl::OrOp> {
  using OpConversionPattern<ltl::OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.intersect to smt.and
struct LTLIntersectOpConversion : OpConversionPattern<ltl::IntersectOp> {
  using OpConversionPattern<ltl::IntersectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::IntersectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.not to smt.not
struct LTLNotOpConversion : OpConversionPattern<ltl::NotOp> {
  using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();
    rewriter.replaceOpWithNewOp<smt::NotOp>(op, input);
    return success();
  }
};

/// Convert ltl.implication to smt.or(not(antecedent), consequent)
struct LTLImplicationOpConversion : OpConversionPattern<ltl::ImplicationOp> {
  using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value antecedent =
        materializeSMTBool(adaptor.getAntecedent(), *typeConverter, rewriter,
                           op.getLoc());
    Value consequent =
        materializeSMTBool(adaptor.getConsequent(), *typeConverter, rewriter,
                           op.getLoc());
    if (!antecedent || !consequent)
      return failure();
    Value notAntecedent = smt::NotOp::create(rewriter, op.getLoc(), antecedent);
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, notAntecedent, consequent);
    return success();
  }
};

/// Convert ltl.eventually to SMT boolean.
/// For bounded model checking, eventually(p) at the current time step
/// contributes p to an OR over all time steps. The BMC loop handles
/// the accumulation; here we just convert the inner property.
struct LTLEventuallyOpConversion : OpConversionPattern<ltl::EventuallyOp> {
  using OpConversionPattern<ltl::EventuallyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::EventuallyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (op->hasAttr(kWeakEventuallyAttr)) {
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }
    // For BMC: eventually(p) means p should hold at some point.
    // At each time step, we check if p holds. The BMC loop accumulates
    // these checks with OR. Here we convert the inner property.
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();
    // The eventually property at this step is just the inner property value
    // The liveness aspect (must eventually hold) is checked by the BMC framework
    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Convert ltl.until to SMT boolean.
/// p until q: p holds continuously until q holds.
/// For bounded checking at a single step: q || (p && X(p U q))
/// Since X requires next-state which BMC handles, we encode:
/// weak until semantics: q || p (either q holds or p holds at this step)
struct LTLUntilOpConversion : OpConversionPattern<ltl::UntilOp> {
  using OpConversionPattern<ltl::UntilOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::UntilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value p = materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                                 op.getLoc());
    Value q = materializeSMTBool(adaptor.getCondition(), *typeConverter,
                                 rewriter, op.getLoc());
    if (!p || !q)
      return failure();
    // Weak until: the property q || p
    // Full until semantics requires tracking across time steps in BMC
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, q, p);
    return success();
  }
};

/// Convert ltl.boolean_constant to smt.constant
struct LTLBooleanConstantOpConversion
    : OpConversionPattern<ltl::BooleanConstantOp> {
  using OpConversionPattern<ltl::BooleanConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::BooleanConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, op.getValue());
    return success();
  }
};

/// Convert ltl.delay to SMT boolean.
///
/// CURRENT LIMITATION (as of Iteration 45):
/// For BMC, delay(seq, N) with N>0 represents a sequence that starts N cycles
/// later. However, the current implementation does NOT properly track delayed
/// obligations across time steps. Instead:
///   - delay(seq, 0) → seq (correct: no delay)
///   - delay(seq, N>0) → true (INCORRECT: should track obligation)
///
/// This means temporal properties like "req |-> ##1 ack" (ack must hold 1 cycle
/// after req) are NOT properly verified. The delay is treated as trivially true.
///
/// WHY THIS IS HARD TO FIX:
/// The BMC infrastructure converts the circuit to a function before this
/// pattern runs. The function is then called in an scf.for loop, but:
/// 1. This pattern has no access to the loop's iter_args
/// 2. Delay tracking would require threading "obligation buffers" through the loop
/// 3. The architecture separates function extraction from delay conversion
///
/// WORKAROUND (proven in test/Conversion/VerifToSMT/bmc-manual-multistep.mlir):
/// Use explicit registers to store previous cycle values:
///   %prev_req = seq.compreg %req, %clk
///   %prop = comb.or %not_prev_req, %ack  // !prev_req || ack
/// This manually implements "req |-> ##1 ack" and BMC can verify it.
///
/// TODO(Future): Implement proper delay tracking:
/// 1. Before function extraction, scan circuit for ltl.delay operations
/// 2. Allocate delay_buffer[] slots in scf.for iter_args (one per delay)
/// 3. Initialize buffers to false in init region
/// 4. Shift buffers each iteration: buffer[i] = buffer[i+1], buffer[N-1] = prop
/// 5. Modify this conversion to return buffer[delay-1] for delay > 0
/// 6. Assert buffer[0] values (mature obligations)
///
/// See BMC_MULTISTEP_DESIGN.md for detailed architecture proposal.
struct LTLDelayOpConversion : OpConversionPattern<ltl::DelayOp> {
  using OpConversionPattern<ltl::DelayOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::DelayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    // Get the delay amount
    uint64_t delay = op.getDelay();

    if (delay == 0) {
      // No delay: just pass through the input sequence
      Value input =
          materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                             op.getLoc());
      if (!input)
        return failure();
      rewriter.replaceOp(op, input);
    } else {
      // LIMITATION: For BMC with delay > 0, we return true (trivially satisfied).
      // This is INCORRECT for bounded model checking but required by the current
      // architecture. The proper fix requires significant refactoring.
      // See documentation above for workarounds and future implementation.
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
    }
    return success();
  }
};

/// Convert ltl.past to SMT boolean.
/// past(signal, N) returns the value of signal from N cycles ago.
/// For BMC, this is handled specially in the BMC conversion by creating
/// past buffers that track signal history. This fallback pattern handles
/// any remaining past ops that weren't processed by BMC (e.g., past(x, 0)).
///
/// NOTE: For past with N > 0 outside of BMC, we return false (conservative
/// approximation that the past value was unknown/false). The proper handling
/// is done in the BMC infrastructure.
struct LTLPastOpConversion : OpConversionPattern<ltl::PastOp> {
  using OpConversionPattern<ltl::PastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::PastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t delay = op.getDelay();

    if (delay == 0) {
      // past(x, 0) is just x - pass through the input
      Value input =
          materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                             op.getLoc());
      if (!input)
        return failure();
      rewriter.replaceOp(op, input);
    } else {
      // For past with delay > 0 outside of BMC, return false (conservative).
      // This handles edge cases where ltl.past appears outside BMC context.
      // The proper handling with buffer tracking is done in VerifBoundedModelCheckingOpConversion.
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, false);
    }
    return success();
  }
};

/// Convert ltl.concat to SMT boolean.
/// Concatenation in LTL joins sequences end-to-end. For BMC at a single
/// time step, concat(a, b) means: a holds at its time range, then b holds
/// at its time range. When flattened to a single step check, if both
/// sequences are instantaneous (booleans), this is equivalent to AND.
/// For sequences with duration, BMC tracks them across steps.
struct LTLConcatOpConversion : OpConversionPattern<ltl::ConcatOp> {
  using OpConversionPattern<ltl::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }

    if (smtOperands.empty()) {
      // Empty concatenation is trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }

    // For BMC: concatenation of sequences at a single step is AND
    // (all parts of the sequence must hold in their respective positions)
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.repeat to SMT boolean.
/// Repetition in LTL means the sequence must match N times consecutively.
/// For BMC at a single step:
/// - repeat(seq, 0, 0) is an empty sequence (true)
/// - repeat(seq, N, 0) with N>0 means seq must hold N times
/// For a single-step check where seq is a boolean, this is just seq
/// (the sequence holding once is the same as holding N times at that instant).
struct LTLRepeatOpConversion : OpConversionPattern<ltl::RepeatOp> {
  using OpConversionPattern<ltl::RepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    // For base >= 1: the sequence must match at least once
    // In BMC single-step semantics, this is just the input sequence value
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    // For exact repetition (base=N, more=0), at a single step this is
    // equivalent to the input holding (since a boolean sequence is
    // instantaneous and N repetitions of it all overlap at the same step).
    // The temporal tracking of multiple repetitions is handled by BMC.
    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Convert ltl.goto_repeat to SMT boolean.
/// goto_repeat is a non-consecutive repetition where the final repetition
/// must hold at the end. For BMC single-step semantics:
/// - goto_repeat(seq, 0, N) with base=0 means the sequence can match 0 times (true)
/// - goto_repeat(seq, N, M) with N>0 means seq must hold at least once at this step
/// The full temporal semantics (non-consecutive with final match) requires
/// multi-step tracking which is handled by the BMC loop or LTLToCore pass.
struct LTLGoToRepeatOpConversion : OpConversionPattern<ltl::GoToRepeatOp> {
  using OpConversionPattern<ltl::GoToRepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::GoToRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    // For base >= 1: at a single step, the sequence must hold
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Convert ltl.non_consecutive_repeat to SMT boolean.
/// non_consecutive_repeat is like goto_repeat but the final match doesn't
/// need to be at the end. For BMC single-step semantics:
/// - non_consecutive_repeat(seq, 0, N) with base=0 means trivially true
/// - non_consecutive_repeat(seq, N, M) with N>0 means seq must hold at this step
struct LTLNonConsecutiveRepeatOpConversion
    : OpConversionPattern<ltl::NonConsecutiveRepeatOp> {
  using OpConversionPattern<ltl::NonConsecutiveRepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NonConsecutiveRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    // For base >= 1: at a single step, the sequence must hold
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Verif Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp,
/// negated to check for unsatisfiability.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL properties are not supported by VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/false, rewriter,
                               op.getLoc());
    }
    Value notCond = smt::NotOp::create(rewriter, op.getLoc(), cond);
    auto assertOp = rewriter.replaceOpWithNewOp<smt::AssertOp>(op, notCond);
    if (auto label = op.getLabelAttr())
      assertOp->setAttr("smtlib.name", label);
    return success();
  }
};

/// Lower a verif::AssumeOp operation with an i1 operand to a smt::AssertOp
struct VerifAssumeOpConversion : OpConversionPattern<verif::AssumeOp> {
  using OpConversionPattern<verif::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL properties are not supported by VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/false, rewriter,
                               op.getLoc());
    }
    auto assertOp = rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    if (auto label = op.getLabelAttr())
      assertOp->setAttr("smtlib.name", label);
    return success();
  }
};

/// Drop verif::CoverOp for now; cover checking is not modeled in SMT.
struct VerifCoverOpConversion : OpConversionPattern<verif::CoverOp> {
  using OpConversionPattern<verif::CoverOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::CoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL cover properties are not supported by "
                          "VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/true, rewriter,
                               op.getLoc());
    }
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

/// Lower unrealized casts between smt.bool and smt.bv<1> into explicit SMT ops.
struct BoolBVCastOpRewrite : OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);

    Value input = op.getInputs()[0];
    Type srcTy = input.getType();
    Type dstTy = op.getOutputs()[0].getType();
    Location loc = op.getLoc();

    Region *inputRegion = input.getParentRegion();
    Region *castRegion = op->getParentRegion();
    if (inputRegion && castRegion && inputRegion != castRegion) {
      auto allUsesInInputRegion = llvm::all_of(
          op->getUsers(), [&](Operation *user) {
            Region *userRegion = user->getParentRegion();
            return userRegion == inputRegion ||
                   inputRegion->isProperAncestor(userRegion);
          });
      if (!allUsesInInputRegion)
        return failure();
      if (auto *defOp = input.getDefiningOp())
        rewriter.setInsertionPointAfter(defOp);
      else if (auto *block = input.getParentBlock())
        rewriter.setInsertionPointToStart(block);
    } else {
      rewriter.setInsertionPoint(op);
    }

    auto dstBvTy = dyn_cast<smt::BitVectorType>(dstTy);
    if (dstBvTy && dstBvTy.getWidth() == 1) {
      Value boolVal;
      if (isa<smt::BoolType>(srcTy)) {
        boolVal = input;
      } else if (auto intTy = dyn_cast<IntegerType>(srcTy);
                 intTy && intTy.getWidth() == 1) {
        if (auto innerCast =
                op.getInputs()[0]
                    .getDefiningOp<UnrealizedConversionCastOp>()) {
          if (innerCast.getInputs().size() == 1 &&
              isa<smt::BoolType>(innerCast.getInputs()[0].getType()))
            boolVal = innerCast.getInputs()[0];
        }
      }
      if (boolVal) {
        auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
        auto zero = smt::BVConstantOp::create(rewriter, loc, 0, 1);
        auto ite = smt::IteOp::create(rewriter, loc, boolVal, one, zero);
        rewriter.replaceOp(op, ite);
        return success();
      }
    }

    if (isa<smt::BoolType>(dstTy)) {
      if (auto bvTy = dyn_cast<smt::BitVectorType>(srcTy);
          bvTy && bvTy.getWidth() == 1) {
        auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
        auto eq = smt::EqOp::create(rewriter, loc, input, one);
        rewriter.replaceOp(op, eq);
        return success();
      }
    }

    return failure();
  }
};

/// Move unrealized casts into the region that defines their input if the cast
/// is used exclusively there. This avoids cross-region value uses.
struct RelocateCastIntoInputRegion
    : OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value input = op.getInputs()[0];
    Region *inputRegion = input.getParentRegion();
    Region *castRegion = op->getParentRegion();
    if (!inputRegion || !castRegion || inputRegion == castRegion)
      return failure();

    auto allUsesInInputRegion = llvm::all_of(
        op->getUsers(), [&](Operation *user) {
          Region *userRegion = user->getParentRegion();
          return userRegion == inputRegion ||
                 inputRegion->isProperAncestor(userRegion);
        });
    if (!allUsesInInputRegion)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    if (auto *defOp = input.getDefiningOp()) {
      rewriter.setInsertionPointAfter(defOp);
    } else if (auto *block = input.getParentBlock()) {
      rewriter.setInsertionPointToStart(block);
    } else {
      return failure();
    }

    auto relocated = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), input);
    rewriter.replaceOp(op, relocated.getResults());
    return success();
  }
};

/// Move SMT equality ops into the region that defines their operands to avoid
/// cross-region value uses. Constants are re-materialized in the target region.
struct RelocateSMTEqIntoOperandRegion : OpRewritePattern<smt::EqOp> {
  using OpRewritePattern<smt::EqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(smt::EqOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Region *opRegion = op->getParentRegion();
    if (!opRegion)
      return failure();

    auto *lhsRegion = lhs.getParentRegion();
    auto *rhsRegion = rhs.getParentRegion();

    Region *targetRegion = nullptr;
    if (lhsRegion && lhsRegion != opRegion)
      targetRegion = lhsRegion;
    else if (rhsRegion && rhsRegion != opRegion)
      targetRegion = rhsRegion;
    else
      return failure();

    auto allUsesInTarget = llvm::all_of(op->getUsers(), [&](Operation *user) {
      Region *userRegion = user->getParentRegion();
      return userRegion == targetRegion ||
             targetRegion->isProperAncestor(userRegion);
    });
    if (!allUsesInTarget)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    if (auto *defOp = lhs.getDefiningOp())
      rewriter.setInsertionPointAfter(defOp);
    else if (auto *block = lhs.getParentBlock())
      rewriter.setInsertionPointToStart(block);
    else
      return failure();

    auto materializeInTarget = [&](Value input) -> Value {
      if (input.getParentRegion() == targetRegion)
        return input;
      if (auto cst = input.getDefiningOp<smt::BVConstantOp>())
        return smt::BVConstantOp::create(rewriter, op.getLoc(),
                                         cst.getValue());
      if (auto cst = input.getDefiningOp<smt::BoolConstantOp>())
        return smt::BoolConstantOp::create(rewriter, op.getLoc(),
                                           cst.getValue());
      return Value();
    };

    Value newLhs = materializeInTarget(lhs);
    Value newRhs = materializeInTarget(rhs);
    if (!newLhs || !newRhs)
      return failure();

    auto relocated = smt::EqOp::create(rewriter, op.getLoc(), newLhs, newRhs);
    rewriter.replaceOp(op, relocated.getResult());
    return success();
  }
};

template <typename OpTy>
struct CircuitRelationCheckOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

protected:
  using ConversionPattern::typeConverter;
  void
  createOutputsDifferentOps(Operation *firstOutputs, Operation *secondOutputs,
                            Location &loc, ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &outputsDifferent) const {
    // Convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and compare the output values.
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          smt::DistinctOp::create(rewriter, loc, o1, o2));
    }
  }

  void replaceOpWithSatCheck(OpTy &op, Location &loc,
                             ConversionPatternRewriter &rewriter,
                             smt::SolverOp &solver) const {
    // If no operation uses the result of this solver, we leave our check
    // operations empty. If the result is used, we create a check operation with
    // the result type of the operation and yield the result of the check
    // operation.
    if (op.getNumResults() == 0) {
      auto checkOp = smt::CheckOp::create(rewriter, loc, TypeRange{});
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc);

      // Erase as operation is replaced by an operator without a return value.
      rewriter.eraseOp(op);
    } else {
      Value falseVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      auto checkOp = smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc, trueVal);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc, checkOp->getResults());

      rewriter.replaceOp(op, solver->getResults());
    }
  }
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::LogicEquivalenceCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::LogicEquivalenceCheckingOp>::CircuitRelationCheckOpConversion;
  LogicEquivalenceCheckingOpConversion(TypeConverter &converter,
                                       MLIRContext *context,
                                       bool assumeKnownInputs)
      : CircuitRelationCheckOpConversion(converter, context),
        assumeKnownInputs(assumeKnownInputs) {}

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    auto hasNoResult = op.getNumResults() == 0;

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Capture the original argument types BEFORE conversion for assume-known
    // logic which needs to identify hw.struct types with value/unknown fields.
    ArrayAttr inputNames = op->getAttrOfType<ArrayAttr>("lec.input_names");
    ArrayAttr inputTypes = op->getAttrOfType<ArrayAttr>("lec.input_types");
    auto originalArgTypes = op.getFirstCircuit().getArgumentTypes();
    auto getOriginalType = [&](unsigned index) -> Type {
      if (inputTypes && index < inputTypes.size())
        if (auto typeAttr = dyn_cast<TypeAttr>(inputTypes[index]))
          return typeAttr.getValue();
      if (index < originalArgTypes.size())
        return originalArgTypes[index];
      return Type{};
    };

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments()) {
      unsigned index = arg.getArgNumber();
      StringAttr namePrefix;
      if (inputNames && index < inputNames.size())
        namePrefix = dyn_cast<StringAttr>(inputNames[index]);
      Value decl =
          smt::DeclareFunOp::create(rewriter, loc, arg.getType(), namePrefix);
      if (assumeKnownInputs) {
        if (auto originalTy = getOriginalType(index))
          maybeAssertKnownInput(originalTy, decl, loc, rewriter);
      }
      inputs.push_back(decl);
    }

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Fourth, build the assertion.
    SmallVector<Value> outputsDifferent;
    createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                              outputsDifferent);

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.empty()) {
      toAssert = smt::BoolConstantOp::create(rewriter, loc, false);
    } else if (outputsDifferent.size() == 1) {
      toAssert = outputsDifferent[0];
    } else {
      toAssert = smt::OrOp::create(rewriter, loc, outputsDifferent);
    }

    smt::AssertOp::create(rewriter, loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }

private:
  bool assumeKnownInputs = false;
};

struct RefinementCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::RefinementCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::RefinementCheckingOp>::CircuitRelationCheckOpConversion;
  RefinementCheckingOpConversion(TypeConverter &converter, MLIRContext *context,
                                 bool assumeKnownInputs)
      : CircuitRelationCheckOpConversion(converter, context),
        assumeKnownInputs(assumeKnownInputs) {}

  LogicalResult
  matchAndRewrite(verif::RefinementCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Find non-deterministic values (free variables) in the source circuit.
    // For now, only support quantification over 'primitive' types.
    SmallVector<Value> srcNonDetValues;
    bool canBind = true;
    for (auto ndOp : op.getFirstCircuit().getOps<smt::DeclareFunOp>()) {
      if (!isa<smt::IntType, smt::BoolType, smt::BitVectorType>(
              ndOp.getType())) {
        ndOp.emitError("Uninterpreted function of non-primitive type cannot be "
                       "converted.");
        canBind = false;
      }
      srcNonDetValues.push_back(ndOp.getResult());
    }
    if (!canBind)
      return failure();

    if (srcNonDetValues.empty()) {
      // If there is no non-determinism in the source circuit, the
      // refinement check becomes an equivalence check, which does not
      // need quantified expressions.
      auto eqOp = verif::LogicEquivalenceCheckingOp::create(
          rewriter, op.getLoc(), op.getNumResults() != 0);
      rewriter.moveBlockBefore(&op.getFirstCircuit().front(),
                               &eqOp.getFirstCircuit(),
                               eqOp.getFirstCircuit().end());
      rewriter.moveBlockBefore(&op.getSecondCircuit().front(),
                               &eqOp.getSecondCircuit(),
                               eqOp.getSecondCircuit().end());
      rewriter.replaceOp(op, eqOp);
      return success();
    }

    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    auto hasNoResult = op.getNumResults() == 0;

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      if (hasNoResult) {
        rewriter.eraseOp(op);
      } else {
        Value trueVal = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, trueVal);
      }
      return success();
    }

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    auto originalArgTypes = op.getFirstCircuit().getArgumentTypes();
    for (auto arg : adaptor.getFirstCircuit().getArguments()) {
      Value decl =
          smt::DeclareFunOp::create(rewriter, loc, arg.getType(), StringAttr{});
      if (assumeKnownInputs && arg.getArgNumber() < originalArgTypes.size())
        maybeAssertKnownInput(originalArgTypes[arg.getArgNumber()], decl, loc,
                              rewriter);
      inputs.push_back(decl);
    }

    // Inline the target circuit. Free variables remain free variables.
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Create the universally quantified expression containing the source
    // circuit. Free variables in the circuit's body become bound variables.
    auto forallOp = smt::ForallOp::create(
        rewriter, op.getLoc(), TypeRange(srcNonDetValues),
        [&](OpBuilder &builder, auto, ValueRange args) -> Value {
          // Inline the source circuit
          Block *body = builder.getBlock();
          rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), body,
                               inputs);

          // Replace non-deterministic values with the quantifier's bound
          // variables
          for (auto [freeVar, boundVar] : llvm::zip(srcNonDetValues, args))
            rewriter.replaceOp(freeVar.getDefiningOp(), boundVar);

          // Compare the output values
          rewriter.setInsertionPointToEnd(body);
          SmallVector<Value> outputsDifferent;
          createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                                    outputsDifferent);
          if (outputsDifferent.size() == 1)
            return outputsDifferent[0];
          else
            return rewriter.createOrFold<smt::OrOp>(loc, outputsDifferent);
        });

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    // Assert the quantified expression
    rewriter.setInsertionPointAfter(forallOp);
    smt::AssertOp::create(rewriter, op.getLoc(), forallOp.getResult());

    // Check for satisfiability and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }

private:
  bool assumeKnownInputs = false;
};

/// Information about a ltl.delay operation that needs multi-step tracking.
/// For delay N, we need N slots in the delay buffer to track the signal history.
struct DelayInfo {
  ltl::DelayOp op;           // The original delay operation
  Value inputSignal;         // The signal being delayed
  uint64_t delay;            // The delay amount (N cycles)
  uint64_t length;           // Additional range length (0 for exact delay)
  uint64_t bufferSize;       // Total buffer slots (delay + length)
  size_t bufferStartIndex;   // Index into the delay buffer iter_args
};

/// Information about a ltl.past operation that needs multi-step tracking.
/// For past N, we need N slots in the past buffer to track the signal history.
/// This is used for $rose/$fell which look at the previous cycle's value.
struct PastInfo {
  ltl::PastOp op;            // The original past operation
  Value inputSignal;         // The signal being observed in the past
  uint64_t delay;            // How many cycles ago (N)
  uint64_t bufferSize;       // Buffer slots needed (same as delay)
  size_t bufferStartIndex;   // Index into the past buffer iter_args
};

/// Rewrite ltl.implication with an exact delayed consequent into a past-form
/// implication for BMC. This allows the BMC loop to use past buffers instead
/// of looking ahead into future time steps.
static void rewriteImplicationDelaysForBMC(Block &circuitBlock,
                                           RewriterBase &rewriter) {
  SmallVector<ltl::ImplicationOp> implicationOps;
  circuitBlock.walk(
      [&](ltl::ImplicationOp op) { implicationOps.push_back(op); });

  for (auto implOp : implicationOps) {
    auto delayOp = implOp.getConsequent().getDefiningOp<ltl::DelayOp>();
    if (!delayOp)
      continue;
    uint64_t delay = delayOp.getDelay();
    if (delay == 0)
      continue;
    auto lengthAttr = delayOp.getLengthAttr();
    if (!lengthAttr || lengthAttr.getValue().getZExtValue() != 0)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(implOp);
    auto shiftedAntecedent = ltl::DelayOp::create(
        rewriter, implOp.getLoc(), implOp.getAntecedent(), delay,
        rewriter.getI64IntegerAttr(0));
    implOp.setOperand(0, shiftedAntecedent.getResult());
    implOp.setOperand(1, delayOp.getInput());
  }
}

/// Expand ltl.repeat into explicit delay/and/or sequences inside a BMC circuit.
static void expandRepeatOpsInBMC(verif::BoundedModelCheckingOp bmcOp,
                                 RewriterBase &rewriter) {
  auto &circuitBlock = bmcOp.getCircuit().front();
  SmallVector<ltl::RepeatOp> repeatOps;
  circuitBlock.walk([&](ltl::RepeatOp repeatOp) { repeatOps.push_back(repeatOp); });
  if (repeatOps.empty())
    return;

  uint64_t boundValue = bmcOp.getBound();

  for (auto repeatOp : repeatOps) {
    uint64_t base = repeatOp.getBase();
    if (base == 0)
      continue;

    uint64_t maxRepeat = base;
    if (auto moreAttr = repeatOp.getMoreAttr()) {
      maxRepeat = base + moreAttr.getValue().getZExtValue();
    } else if (boundValue > base) {
      maxRepeat = boundValue;
    }

    if (maxRepeat < base)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(repeatOp);

    auto buildExactRepeat = [&](uint64_t count) -> Value {
      SmallVector<Value> terms;
      terms.reserve(count);
      for (uint64_t i = 0; i < count; ++i) {
        if (i == 0) {
          terms.push_back(repeatOp.getInput());
          continue;
        }
        auto delayOp = ltl::DelayOp::create(rewriter, repeatOp.getLoc(),
                                            repeatOp.getInput(), i,
                                            rewriter.getI64IntegerAttr(0));
        terms.push_back(delayOp.getResult());
      }
      if (terms.size() == 1) {
        if (isa<ltl::SequenceType>(terms[0].getType()))
          return terms[0];
        auto delayOp = ltl::DelayOp::create(rewriter, repeatOp.getLoc(),
                                            terms[0], 0,
                                            rewriter.getI64IntegerAttr(0));
        return delayOp.getResult();
      }
      return ltl::AndOp::create(rewriter, repeatOp.getLoc(), terms)
          .getResult();
    };

    SmallVector<Value> choices;
    for (uint64_t count = base; count <= maxRepeat; ++count)
      choices.push_back(buildExactRepeat(count));

    Value replacement =
        choices.size() == 1
            ? choices.front()
            : ltl::OrOp::create(rewriter, repeatOp.getLoc(), choices)
                  .getResult();

    rewriter.replaceOp(repeatOp, replacement);
  }
}

static Value buildSequenceConstant(OpBuilder &builder, Location loc,
                                   bool value) {
  auto cst = hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                    value ? 1 : 0);
  auto zero = builder.getI64IntegerAttr(0);
  return ltl::DelayOp::create(builder, loc, cst, zero, zero).getResult();
}

static Value buildSequenceAnd(OpBuilder &builder, Location loc,
                              ArrayRef<Value> inputs) {
  if (inputs.empty())
    return buildSequenceConstant(builder, loc, true);
  if (inputs.size() == 1)
    return inputs.front();
  return ltl::AndOp::create(builder, loc, inputs).getResult();
}

static Value buildSequenceOr(OpBuilder &builder, Location loc,
                             ArrayRef<Value> inputs) {
  if (inputs.empty())
    return buildSequenceConstant(builder, loc, false);
  if (inputs.size() == 1)
    return inputs.front();
  return ltl::OrOp::create(builder, loc, inputs).getResult();
}

static bool exceedsCombinationLimit(uint64_t n, uint64_t k, uint64_t limit) {
  if (k > n)
    return false;
  if (k > n - k)
    k = n - k;
  uint64_t count = 1;
  for (uint64_t i = 1; i <= k; ++i) {
    uint64_t numerator = n - k + i;
    if (count > limit / numerator)
      return true;
    count *= numerator;
    count /= i;
    if (count > limit)
      return true;
  }
  return count > limit;
}

static LogicalResult
expandGotoRepeatOpsInBMC(verif::BoundedModelCheckingOp bmcOp,
                         RewriterBase &rewriter) {
  auto &circuitBlock = bmcOp.getCircuit().front();
  SmallVector<ltl::GoToRepeatOp> gotoOps;
  SmallVector<ltl::NonConsecutiveRepeatOp> nonConsecOps;
  circuitBlock.walk([&](ltl::GoToRepeatOp op) { gotoOps.push_back(op); });
  circuitBlock.walk(
      [&](ltl::NonConsecutiveRepeatOp op) { nonConsecOps.push_back(op); });
  if (gotoOps.empty() && nonConsecOps.empty())
    return success();

  uint64_t boundValue = bmcOp.getBound();
  uint64_t maxOffset = boundValue > 0 ? boundValue - 1 : 0;
  constexpr uint64_t kMaxCombos = 16384;

  auto expandOp = [&](auto op, bool requireCurrentMatch) -> LogicalResult {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    uint64_t base = op.getBase();
    uint64_t maxCount = base + op.getMore();

    auto delayZero = rewriter.getI64IntegerAttr(0);
    Value input = op.getInput();
    Value seqInput = input;
    if (!isa<ltl::SequenceType>(input.getType()))
      seqInput =
          ltl::DelayOp::create(rewriter, loc, input, delayZero, delayZero)
              .getResult();

    SmallVector<Value> offsetValues;
    offsetValues.reserve(maxOffset + 1);
    offsetValues.push_back(seqInput);
    for (uint64_t offset = 1; offset <= maxOffset; ++offset) {
      offsetValues.push_back(ltl::DelayOp::create(rewriter, loc, seqInput,
                                                 rewriter.getI64IntegerAttr(
                                                     offset),
                                                 delayZero)
                                  .getResult());
    }

    SmallVector<Value> choices;
    for (uint64_t count = base; count <= maxCount; ++count) {
      if (count == 0) {
        choices.push_back(buildSequenceConstant(rewriter, loc, true));
        continue;
      }
      if (requireCurrentMatch && count > 0 && maxOffset + 1 < count)
        continue;
      if (!requireCurrentMatch && maxOffset + 1 < count)
        continue;

      uint64_t remaining = requireCurrentMatch ? count - 1 : count;
      uint64_t available = requireCurrentMatch ? maxOffset : maxOffset + 1;
      if (exceedsCombinationLimit(available, remaining, kMaxCombos))
        return op.emitError("goto/non-consecutive repeat expansion too large; "
                            "reduce the BMC bound or repetition range");

      SmallVector<Value> current;
      if (requireCurrentMatch)
        current.push_back(offsetValues[0]);
      auto buildCombos = [&](uint64_t start, uint64_t need,
                             auto &&buildCombosRef) -> void {
        if (need == 0) {
          choices.push_back(buildSequenceAnd(rewriter, loc, current));
          return;
        }
        for (uint64_t idx = start; idx + need <= maxOffset + 1; ++idx) {
          current.push_back(offsetValues[idx]);
          buildCombosRef(idx + 1, need - 1, buildCombosRef);
          current.pop_back();
        }
      };
      buildCombos(requireCurrentMatch ? 1 : 0, remaining, buildCombos);
    }

    Value replacement = buildSequenceOr(rewriter, loc, choices);
    rewriter.replaceOp(op, replacement);
    return success();
  };

  for (auto op : gotoOps)
    if (failed(expandOp(op, /*requireCurrentMatch=*/true)))
      return failure();
  for (auto op : nonConsecOps)
    if (failed(expandOp(op, /*requireCurrentMatch=*/false)))
      return failure();
  return success();
}

/// Lower a verif::BMCOp operation to an MLIR program that performs the bounded
/// model check
struct VerifBoundedModelCheckingOpConversion
    : OpConversionPattern<verif::BoundedModelCheckingOp> {
  using OpConversionPattern<verif::BoundedModelCheckingOp>::OpConversionPattern;

  VerifBoundedModelCheckingOpConversion(
      TypeConverter &converter, MLIRContext *context, Namespace &names,
      bool risingClocksOnly, SmallVectorImpl<Operation *> &propertylessBMCOps,
      SmallVectorImpl<Operation *> &coverBMCOps)
      : OpConversionPattern(converter, context), names(names),
        risingClocksOnly(risingClocksOnly),
        propertylessBMCOps(propertylessBMCOps), coverBMCOps(coverBMCOps) {}
  LogicalResult
  matchAndRewrite(verif::BoundedModelCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    bool isCoverCheck =
        std::find(coverBMCOps.begin(), coverBMCOps.end(), op) !=
        coverBMCOps.end();

    if (std::find(propertylessBMCOps.begin(), propertylessBMCOps.end(), op) !=
        propertylessBMCOps.end()) {
      // No properties to check, so we don't bother solving, we just return true
      // (without this we would incorrectly find violations, since the solver
      // will always return SAT)
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    auto &circuitBlock = op.getCircuit().front();
    // Expand non-consecutive repetitions into bounded delay patterns so BMC
    // can model them with delay buffers.
    if (failed(expandGotoRepeatOpsInBMC(op, rewriter)))
      return failure();
    // Expand repeat ops inside the BMC circuit before delay scanning so
    // delay buffers can be allocated for the resulting ltl.delay ops.
    expandRepeatOpsInBMC(op, rewriter);
    // Rewrite implication with exact delayed consequent to a past-form
    // implication so the BMC delay buffers can track the antecedent history.
    rewriteImplicationDelaysForBMC(circuitBlock, rewriter);

    // Combine multiple non-final properties into a single check value so BMC
    // can detect any violating property.
    SmallVector<Operation *> nonFinalOps;
    SmallVector<Value> checkProps;
    if (isCoverCheck) {
      circuitBlock.walk([&](verif::CoverOp coverOp) {
        if (coverOp->hasAttr("bmc.final"))
          return;
        nonFinalOps.push_back(coverOp);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(coverOp);
        checkProps.push_back(
            gatePropertyWithEnable(coverOp.getProperty(), coverOp.getEnable(),
                                   /*isCover=*/true, rewriter,
                                   coverOp.getLoc()));
      });
    } else {
      circuitBlock.walk([&](verif::AssertOp assertOp) {
        if (assertOp->hasAttr("bmc.final"))
          return;
        nonFinalOps.push_back(assertOp);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(assertOp);
        checkProps.push_back(
            gatePropertyWithEnable(assertOp.getProperty(), assertOp.getEnable(),
                                   /*isCover=*/false, rewriter,
                                   assertOp.getLoc()));
      });
    }
    SmallVector<Value> nonFinalCheckValues;
    if (!nonFinalOps.empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(circuitBlock.getTerminator());
      Value combined = checkProps.front();
      if (checkProps.size() > 1) {
        combined = isCoverCheck
                       ? ltl::OrOp::create(rewriter, loc, checkProps).getResult()
                       : ltl::AndOp::create(rewriter, loc, checkProps)
                             .getResult();
      }
      nonFinalCheckValues.push_back(combined);
      for (auto *opToErase : nonFinalOps)
        rewriter.eraseOp(opToErase);
    }

    // Hoist any final-only checks into circuit outputs so we can check them
    // only at the final step.
    SmallVector<Value> finalCheckValues;
    SmallVector<bool> finalCheckIsCover;
    SmallVector<Operation *> opsToErase;
    circuitBlock.walk([&](Operation *curOp) {
      if (!curOp->hasAttr("bmc.final"))
        return;
      if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(assertOp);
        finalCheckValues.push_back(
            gatePropertyWithEnable(assertOp.getProperty(), assertOp.getEnable(),
                                   /*isCover=*/false, rewriter,
                                   assertOp.getLoc()));
        finalCheckIsCover.push_back(false);
        opsToErase.push_back(curOp);
        return;
      }
      if (auto assumeOp = dyn_cast<verif::AssumeOp>(curOp)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(assumeOp);
        finalCheckValues.push_back(
            gatePropertyWithEnable(assumeOp.getProperty(), assumeOp.getEnable(),
                                   /*isCover=*/false, rewriter,
                                   assumeOp.getLoc()));
        finalCheckIsCover.push_back(false);
        opsToErase.push_back(curOp);
        return;
      }
      if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(coverOp);
        finalCheckValues.push_back(
            gatePropertyWithEnable(coverOp.getProperty(), coverOp.getEnable(),
                                   /*isCover=*/true, rewriter,
                                   coverOp.getLoc()));
        finalCheckIsCover.push_back(true);
        opsToErase.push_back(curOp);
        return;
      }
    });
    // Erase the bmc.final ops using the rewriter to properly notify the
    // conversion framework
    for (auto *opToErase : opsToErase)
      rewriter.eraseOp(opToErase);
    size_t numNonFinalChecks = nonFinalCheckValues.size();
    size_t numFinalChecks = finalCheckValues.size();
    uint64_t boundValue = op.getBound();

    SmallVector<Type> oldLoopInputTy(op.getLoop().getArgumentTypes());
    SmallVector<Type> oldCircuitInputTy(op.getCircuit().getArgumentTypes());
    // TODO: the init and loop regions should be able to be concrete instead of
    // symbolic which is probably preferable - just need to convert back and
    // forth
    SmallVector<Type> loopInputTy, circuitInputTy, initOutputTy,
        circuitOutputTy;
    if (failed(typeConverter->convertTypes(oldLoopInputTy, loopInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getInit().front().back().getOperandTypes(), initOutputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();

    unsigned numRegs = op.getNumRegs();
    auto initialValues = op.getInitialValues();

    // Count clocks from the original init yield types (BEFORE region conversion)
    // This handles both explicit seq::ClockType and i1 clocks converted via
    // ToClockOp
    size_t numInitClocks = 0;
    for (auto ty : op.getInit().front().back().getOperandTypes())
      if (isa<seq::ClockType>(ty))
        numInitClocks++;

    // Collect circuit argument names for debug-friendly SMT declarations.
    size_t originalArgCount = oldCircuitInputTy.size();
    SmallVector<StringAttr> inputNamePrefixes;
    if (auto nameAttr =
            op->getAttrOfType<ArrayAttr>("bmc_input_names")) {
      inputNamePrefixes.reserve(nameAttr.size());
      for (auto attr : nameAttr) {
        if (auto strAttr = dyn_cast<StringAttr>(attr))
          inputNamePrefixes.push_back(strAttr);
        else
          inputNamePrefixes.push_back(StringAttr{});
      }
    } else {
      inputNamePrefixes.assign(originalArgCount, StringAttr{});
    }

    auto maybeAssertKnown = [&](Type originalTy, Value smtVal,
                                OpBuilder &builder) {
      maybeAssertKnownInput(originalTy, smtVal, loc, builder);
    };

    // =========================================================================
    // Multi-step delay tracking:
    // Scan the circuit for ltl.delay operations with delay > 0 and set up
    // the infrastructure to track delayed signals across time steps.
    //
    // For each ltl.delay(signal, N) with N > 0:
    // 1. Add N new block arguments to the circuit (for the delay buffer)
    // 2. Add N new yield operands (shifted buffer + current signal)
    // 3. Replace the delay op with the oldest buffer entry (what was N steps ago)
    //
    // The delay buffer is a shift register: each iteration, values shift down
    // and the current signal value enters at position N-1.
    // buffer[0] contains the value from N steps ago (the delayed value to use).
    // =========================================================================
    SmallVector<DelayInfo> delayInfos;
    size_t totalDelaySlots = 0;
    bool delaySetupFailed = false;

    // First pass: collect all delay ops with meaningful temporal ranges.
    circuitBlock.walk([&](ltl::DelayOp delayOp) {
      if (delayOp.use_empty())
        return;
      uint64_t delay = delayOp.getDelay();

      uint64_t length = 0;
      if (auto lengthAttr = delayOp.getLengthAttr()) {
        length = lengthAttr.getValue().getZExtValue();
      } else {
        // Unbounded delay (##[N:$]) is approximated within the BMC bound as
        // a bounded range [N : bound-1]. When the delay exceeds the bound,
        // there is no match within the window.
        if (boundValue > 0 && delay < boundValue)
          length = boundValue - 1 - delay;
      }

      // NOTE: Unbounded delay (missing length) is approximated by the BMC
      // bound, bounded range is supported by widening the delay buffer.
      if (delay == 0 && length == 0)
        return;
      uint64_t bufferSize = delay + length;
      if (bufferSize == 0)
        return;

      DelayInfo info;
      info.op = delayOp;
      info.inputSignal = delayOp.getInput();
      info.delay = delay;
      info.length = length;
      info.bufferSize = bufferSize;
      info.bufferStartIndex = totalDelaySlots;
      delayInfos.push_back(info);
      totalDelaySlots += bufferSize;
    });

    // Track delay buffer names (added immediately after original inputs).
    size_t delaySlots = totalDelaySlots;

    // =========================================================================
    // Past operation tracking for $rose/$fell:
    // ltl.past operations look at signal values from previous cycles.
    // For past(signal, N), we need N buffer slots to track signal history.
    // The buffer works identically to delay buffers - buffer[0] holds the
    // oldest value (from N cycles ago).
    // =========================================================================
    SmallVector<PastInfo> pastInfos;
    size_t totalPastSlots = 0;

    circuitBlock.walk([&](ltl::PastOp pastOp) {
      if (pastOp.use_empty())
        return;
      uint64_t delay = pastOp.getDelay();
      if (delay == 0)
        return;

      PastInfo info;
      info.op = pastOp;
      info.inputSignal = pastOp.getInput();
      info.delay = delay;
      info.bufferSize = delay;
      info.bufferStartIndex = totalPastSlots;
      pastInfos.push_back(info);
      totalPastSlots += delay;
    });

    // Second pass: modify the circuit block to add delay buffer infrastructure
    // We need to do this BEFORE region type conversion.
    //
    // Output order after modification:
    // [original outputs (registers)] [delay buffer outputs]
    SmallVector<ltl::DelayOp> delayOpsToErase;
    if (!delayInfos.empty()) {
      // For each delay op, add buffer arguments and modify the yield
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());

      // Get current operands (no check outputs appended yet)
      SmallVector<Value> origOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      SmallVector<Value> newYieldOperands(origOperands.begin(),
                                          origOperands.end());
      SmallVector<Value> delayBufferOutputs;

      for (auto &info : delayInfos) {
        // The input signal type - used for delay buffer slots.
        Type bufferElementType = info.inputSignal.getType();

        // Add buffer arguments for the delay buffer (oldest to newest)
        SmallVector<Value> bufferArgs;
        for (uint64_t i = 0; i < info.bufferSize; ++i) {
          auto arg = circuitBlock.addArgument(bufferElementType, loc);
          bufferArgs.push_back(arg);
        }

        // Replace all uses of the delay op with the delayed value.
        // For exact delay: use the oldest buffer entry (value from N steps ago).
        // For bounded range: OR over the window [delay, delay+length].
        Value inputSig = info.inputSignal;
        Value delayedValue;
        auto toSequenceValue = [&](Value val) -> Value {
          if (isa<ltl::SequenceType>(val.getType()))
            return val;
          if (isa<IntegerType>(val.getType())) {
            auto zero = rewriter.getI64IntegerAttr(0);
            return ltl::DelayOp::create(rewriter, loc, val, zero, zero)
                .getResult();
          }
          return Value{};
        };

        if (info.delay == 0) {
          // Range includes the current cycle, so OR the input with the buffer.
          bool isI1 = false;
          if (auto intTy = dyn_cast<IntegerType>(bufferElementType))
            isI1 = intTy.getWidth() == 1;
          if (!isI1 && !isa<ltl::SequenceType>(bufferElementType)) {
            info.op.emitError(
                "bounded delay in BMC requires i1 or ltl.sequence input");
            delaySetupFailed = true;
            continue;
          }
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.op);
          delayedValue = toSequenceValue(inputSig);
          if (!delayedValue) {
            info.op.emitError(
                "failed to build sequence value for bounded delay input");
            delaySetupFailed = true;
            continue;
          }
          for (auto arg : bufferArgs) {
            delayedValue =
                ltl::OrOp::create(rewriter, loc,
                                  ValueRange{delayedValue, arg})
                    .getResult();
          }
        } else {
          delayedValue = toSequenceValue(bufferArgs[0]);
          if (!delayedValue) {
            info.op.emitError(
                "failed to build sequence value for bounded delay input");
            delaySetupFailed = true;
            continue;
          }
          if (info.length > 0) {
            bool isI1 = false;
            if (auto intTy = dyn_cast<IntegerType>(bufferElementType))
              isI1 = intTy.getWidth() == 1;
            if (!isI1 && !isa<ltl::SequenceType>(bufferElementType)) {
              info.op.emitError(
                  "bounded delay in BMC requires i1 or ltl.sequence input");
              delaySetupFailed = true;
              continue;
            }
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(info.op);
            for (uint64_t i = 1; i <= info.length; ++i) {
              delayedValue =
                  ltl::OrOp::create(rewriter, loc,
                                    ValueRange{delayedValue, bufferArgs[i]})
                      .getResult();
            }
          }
        }
        info.op.replaceAllUsesWith(delayedValue);
        delayOpsToErase.push_back(info.op);

        // Add yield operands: shifted buffer (drop oldest, add current)
        // new_buffer = [buffer[1], buffer[2], ..., buffer[bufferSize-1], current_signal]
        for (uint64_t i = 1; i < info.bufferSize; ++i)
          delayBufferOutputs.push_back(bufferArgs[i]);
        delayBufferOutputs.push_back(inputSig);
      }

      // Construct final yield: [orig outputs] [delay buffers]
      newYieldOperands.append(delayBufferOutputs);

      // Update the yield with new operands
      yieldOp->setOperands(newYieldOperands);

      if (delaySetupFailed)
        return failure();

      // Erase the delay ops (after all replacements are done)
      for (auto delayOp : delayOpsToErase)
        rewriter.eraseOp(delayOp);

      // Update the type vectors to include the new arguments and outputs
      oldCircuitInputTy.clear();
      oldCircuitInputTy.append(op.getCircuit().getArgumentTypes().begin(),
                               op.getCircuit().getArgumentTypes().end());
    }

    // =========================================================================
    // Process ltl.past operations for $rose/$fell support.
    // Past operations look at signal values from previous cycles.
    // The buffer mechanism is the same as delay ops: buffer[0] is the oldest
    // value (from N cycles ago), which is exactly what past(signal, N) needs.
    // =========================================================================
    SmallVector<ltl::PastOp> pastOpsToErase;
    if (!pastInfos.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());

      // Get current operands (no check outputs appended yet)
      SmallVector<Value> origOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      SmallVector<Value> newYieldOperands(origOperands.begin(),
                                          origOperands.end());
      SmallVector<Value> pastBufferOutputs;

      for (auto &info : pastInfos) {
        // The input signal type - past ops work on i1 signals
        Type bufferElementType = info.inputSignal.getType();

        // Add buffer arguments for the past buffer (oldest to newest)
        SmallVector<Value> bufferArgs;
        for (uint64_t i = 0; i < info.bufferSize; ++i) {
          auto arg = circuitBlock.addArgument(bufferElementType, loc);
          bufferArgs.push_back(arg);
        }

        // Replace all uses of the past op with the past value.
        // past(signal, N) returns the value from N cycles ago, which is buffer[0]
        // after N cycles of shifting.
        Value pastValue = bufferArgs[0];
        info.op.replaceAllUsesWith(pastValue);
        pastOpsToErase.push_back(info.op);

        // Add yield operands: shifted buffer (drop oldest, add current)
        // new_buffer = [buffer[1], buffer[2], ..., buffer[N-1], current_signal]
        for (uint64_t i = 1; i < info.bufferSize; ++i)
          pastBufferOutputs.push_back(bufferArgs[i]);
        pastBufferOutputs.push_back(info.inputSignal);
      }

      // Construct final yield: [orig outputs] [past buffers]
      newYieldOperands.append(pastBufferOutputs);

      yieldOp->setOperands(newYieldOperands);

      // Erase the past ops
      for (auto pastOp : pastOpsToErase)
        rewriter.eraseOp(pastOp);

      // Update the type vectors
      oldCircuitInputTy.clear();
      oldCircuitInputTy.append(op.getCircuit().getArgumentTypes().begin(),
                               op.getCircuit().getArgumentTypes().end());

      // Update totalDelaySlots to include past slots for buffer initialization
      totalDelaySlots += totalPastSlots;
    }

    // Append non-final and final check outputs after delay/past buffers so
    // circuit outputs are ordered as:
    // [original outputs] [delay/past buffers] [non-final checks] [final checks]
    if (!nonFinalCheckValues.empty() || !finalCheckValues.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());
      SmallVector<Value> newYieldOperands(yieldOp.getOperands());
      newYieldOperands.append(nonFinalCheckValues.begin(),
                              nonFinalCheckValues.end());
      newYieldOperands.append(finalCheckValues.begin(),
                              finalCheckValues.end());
      yieldOp->setOperands(newYieldOperands);
    }

    // Extend name list with delay/past buffer slots appended to the circuit
    // arguments. These slots are appended after the original inputs.
    if (inputNamePrefixes.size() != originalArgCount)
      inputNamePrefixes.resize(originalArgCount, StringAttr{});
    if (delaySlots > 0) {
      for (size_t i = 0; i < delaySlots; ++i) {
        auto name =
            rewriter.getStringAttr("delay_buf_" + Twine(i).str());
        inputNamePrefixes.push_back(name);
      }
    }
    if (totalPastSlots > 0) {
      for (size_t i = 0; i < totalPastSlots; ++i) {
        auto name =
            rewriter.getStringAttr("past_buf_" + Twine(i).str());
        inputNamePrefixes.push_back(name);
      }
    }

    // Re-compute circuit types after potential modification
    circuitInputTy.clear();
    circuitOutputTy.clear();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();

    auto initFuncTy = rewriter.getFunctionType({}, initOutputTy);
    // Loop and init output types are necessarily the same, so just use init
    // output types
    auto loopFuncTy = rewriter.getFunctionType(loopInputTy, initOutputTy);
    auto circuitFuncTy =
        rewriter.getFunctionType(circuitInputTy, circuitOutputTy);

    func::FuncOp initFuncOp, loopFuncOp, circuitFuncOp;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          op->getParentOfType<ModuleOp>().getBody());
      initFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_init"), initFuncTy);
      rewriter.inlineRegionBefore(op.getInit(), initFuncOp.getFunctionBody(),
                                  initFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&initFuncOp.getFunctionBody(),
                                             *typeConverter)))
        return failure();
      loopFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_loop"), loopFuncTy);
      rewriter.inlineRegionBefore(op.getLoop(), loopFuncOp.getFunctionBody(),
                                  loopFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&loopFuncOp.getFunctionBody(),
                                             *typeConverter)))
        return failure();
      circuitFuncOp = func::FuncOp::create(
          rewriter, loc, names.newName("bmc_circuit"), circuitFuncTy);
      rewriter.inlineRegionBefore(op.getCircuit(),
                                  circuitFuncOp.getFunctionBody(),
                                  circuitFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&circuitFuncOp.getFunctionBody(),
                                             *typeConverter)))
        return failure();
      auto funcOps = {&initFuncOp, &loopFuncOp, &circuitFuncOp};
      // initOutputTy is the same as loop output types
      auto outputTys = {initOutputTy, initOutputTy, circuitOutputTy};
      for (auto [funcOp, outputTy] : llvm::zip(funcOps, outputTys)) {
        auto operands = funcOp->getBody().front().back().getOperands();
        rewriter.eraseOp(&funcOp->getFunctionBody().front().back());
        rewriter.setInsertionPointToEnd(&funcOp->getBody().front());
        SmallVector<Value> toReturn;
        for (unsigned i = 0; i < outputTy.size(); ++i)
          toReturn.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, outputTy[i], operands[i]));
        func::ReturnOp::create(rewriter, loc, toReturn);
      }
    }

    auto solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                        ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Call init func to get initial clock values
    ValueRange initVals =
        func::CallOp::create(rewriter, loc, initFuncOp)->getResults();

    // InputDecls order should be <circuit arguments> <state arguments>
    // <finalChecks> <wasViolated>
    // Get list of clock indexes in circuit args
    //
    // Circuit arguments layout (after delay buffer modification):
    // [original args (clocks, inputs, regs)] [delay buffer slots]
    //
    // Original args have size: oldCircuitInputTy.size() - totalDelaySlots
    // Delay buffer slots have size: totalDelaySlots
    size_t origCircuitArgsSize = oldCircuitInputTy.size() - totalDelaySlots;

    size_t initIndex = 0;
    SmallVector<Value> inputDecls;
    SmallVector<int> clockIndexes;
    size_t nonRegIndex = 0; // Track position among non-register inputs
    for (auto [curIndex, oldTy, newTy] :
         llvm::enumerate(oldCircuitInputTy, circuitInputTy)) {
      // Check if this is a delay buffer slot (added at the end)
      bool isDelayBuffer = curIndex >= origCircuitArgsSize;
      if (isDelayBuffer) {
        // Initialize delay buffers to false (no prior history at step 0)
        if (auto bvTy = dyn_cast<smt::BitVectorType>(newTy)) {
          auto initVal =
              smt::BVConstantOp::create(rewriter, loc, 0, bvTy.getWidth());
          inputDecls.push_back(initVal);
        } else if (isa<smt::BoolType>(newTy)) {
          auto initVal = smt::BoolConstantOp::create(rewriter, loc, false);
          inputDecls.push_back(initVal);
        } else {
          op.emitError("unsupported delay buffer type in BMC conversion");
          return failure();
        }
        continue;
      }

      // Check if this is a register input (registers are at the end of original args)
      bool isRegister = curIndex >= origCircuitArgsSize - numRegs;

      // Check if this is a clock - either explicit seq::ClockType or
      // an i1 that corresponds to an init clock (for i1 clocks converted via
      // ToClockOp inside the circuit)
      bool isI1Type = isa<IntegerType>(oldTy) &&
                      cast<IntegerType>(oldTy).getWidth() == 1;
      bool isClock = isa<seq::ClockType>(oldTy) ||
                     (!isRegister && isI1Type && nonRegIndex < numInitClocks);

      if (isClock) {
        inputDecls.push_back(initVals[initIndex++]);
        clockIndexes.push_back(curIndex);
        nonRegIndex++;
        continue;
      }
      if (!isRegister)
        nonRegIndex++;
      if (isRegister) {
        auto initVal =
            initialValues[curIndex - origCircuitArgsSize + numRegs];
        if (auto initIntAttr = dyn_cast<IntegerAttr>(initVal)) {
          const auto &cstInt = initIntAttr.getValue();
          if (auto bvTy = dyn_cast<smt::BitVectorType>(newTy)) {
            assert(cstInt.getBitWidth() == bvTy.getWidth() &&
                   "Width mismatch between initial value and target type");
            auto initVal = smt::BVConstantOp::create(rewriter, loc, cstInt);
            inputDecls.push_back(initVal);
            maybeAssertKnown(oldTy, initVal, rewriter);
            continue;
          }
          if (isa<smt::BoolType>(newTy)) {
            auto initVal =
                smt::BoolConstantOp::create(rewriter, loc, !cstInt.isZero());
            inputDecls.push_back(initVal);
            maybeAssertKnown(oldTy, initVal, rewriter);
            continue;
          }
          op.emitError("unsupported integer initial value in BMC conversion");
          return failure();
        }
        if (auto initBoolAttr = dyn_cast<BoolAttr>(initVal)) {
          if (auto bvTy = dyn_cast<smt::BitVectorType>(newTy)) {
            auto initVal = smt::BVConstantOp::create(
                rewriter, loc, initBoolAttr.getValue() ? 1 : 0,
                bvTy.getWidth());
            inputDecls.push_back(initVal);
            maybeAssertKnown(oldTy, initVal, rewriter);
            continue;
          }
          if (isa<smt::BoolType>(newTy)) {
            auto initVal = smt::BoolConstantOp::create(
                rewriter, loc, initBoolAttr.getValue());
            inputDecls.push_back(initVal);
            maybeAssertKnown(oldTy, initVal, rewriter);
            continue;
          }
          op.emitError("unsupported bool initial value in BMC conversion");
          return failure();
        }
      }
      StringAttr namePrefix;
      if (curIndex < inputNamePrefixes.size())
        namePrefix = inputNamePrefixes[curIndex];
      auto decl =
          smt::DeclareFunOp::create(rewriter, loc, newTy, namePrefix);
      inputDecls.push_back(decl);
      maybeAssertKnown(oldTy, decl, rewriter);
    }

    auto numStateArgs = initVals.size() - initIndex;
    // Add the rest of the init vals (state args)
    for (; initIndex < initVals.size(); ++initIndex)
      inputDecls.push_back(initVals[initIndex]);

    SmallVector<unsigned> regClockToLoopIndex;
    bool usePerRegClocks =
        !risingClocksOnly && clockIndexes.size() > 1 && numRegs > 0;
    if (usePerRegClocks) {
      auto regClocksAttr = op->getAttrOfType<ArrayAttr>("bmc_reg_clocks");
      if (!regClocksAttr || regClocksAttr.size() != numRegs) {
        op.emitError("multi-clock BMC requires bmc_reg_clocks with one entry "
                     "per register");
        return failure();
      }
      DenseMap<StringRef, unsigned> inputNameToIndex;
      for (auto [idx, nameAttr] : llvm::enumerate(inputNamePrefixes)) {
        if (nameAttr && !nameAttr.getValue().empty())
          inputNameToIndex[nameAttr.getValue()] = idx;
      }
      regClockToLoopIndex.reserve(numRegs);
      for (auto attr : regClocksAttr) {
        auto nameAttr = dyn_cast<StringAttr>(attr);
        if (!nameAttr || nameAttr.getValue().empty()) {
          op.emitError("multi-clock BMC requires named clock entries in "
                       "bmc_reg_clocks");
          return failure();
        }
        auto nameIt = inputNameToIndex.find(nameAttr.getValue());
        if (nameIt == inputNameToIndex.end()) {
          op.emitError("bmc_reg_clocks entry does not match any input name");
          return failure();
        }
        unsigned clockInputIndex = nameIt->second;
        auto clockIt =
            llvm::find(clockIndexes, static_cast<int>(clockInputIndex));
        if (clockIt == clockIndexes.end()) {
          op.emitError("bmc_reg_clocks entry does not name a clock input");
          return failure();
        }
        regClockToLoopIndex.push_back(
            static_cast<unsigned>(clockIt - clockIndexes.begin()));
      }
    }

    bool checkFinalAtEnd = true;
    if (!risingClocksOnly && clockIndexes.size() == 1 &&
        (boundValue % 2 != 0)) {
      // In non-rising mode, odd bounds end on a negedge; skip final-only checks.
      checkFinalAtEnd = false;
    }

    Value lowerBound =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value step =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    Value upperBound =
        arith::ConstantOp::create(rewriter, loc, adaptor.getBoundAttr());
    Value constFalse =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
    Value constTrue =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    // Initialize final check iter_args with false values matching their types
    // (these will be updated with circuit outputs). Types may be !smt.bv<1>
    // for i1 properties or !smt.bool for LTL properties.
    for (size_t i = 0; i < numFinalChecks; ++i) {
      Type checkTy = circuitOutputTy[circuitOutputTy.size() - numFinalChecks + i];
      if (isa<smt::BoolType>(checkTy))
        inputDecls.push_back(smt::BoolConstantOp::create(rewriter, loc, false));
      else
        inputDecls.push_back(smt::BVConstantOp::create(rewriter, loc, 0, 1));
    }
    inputDecls.push_back(constFalse); // wasViolated?

    // TODO: swapping to a whileOp here would allow early exit once the property
    // is violated
    // Perform model check up to the provided bound
    auto forOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Assert 2-state constraints on the current iteration inputs.
          size_t numCircuitArgs = circuitFuncOp.getNumArguments();
          for (auto [oldTy, arg] :
               llvm::zip(TypeRange(oldCircuitInputTy).take_front(numCircuitArgs),
                         iterArgs.take_front(numCircuitArgs))) {
            maybeAssertKnown(oldTy, arg, builder);
          }

          // Execute the circuit
          ValueRange circuitCallOuts =
              func::CallOp::create(
                  builder, loc, circuitFuncOp,
                  iterArgs.take_front(circuitFuncOp.getNumArguments()))
                  ->getResults();

          // Circuit outputs are ordered as:
          // [original outputs (registers)] [delay buffer outputs]
          // [non-final checks] [final checks]
          //
          // Note: totalDelaySlots is captured from the outer scope
          ValueRange finalCheckOutputs =
              numFinalChecks == 0 ? ValueRange{}
                                  : circuitCallOuts.take_back(numFinalChecks);
          ValueRange beforeFinal =
              numFinalChecks == 0 ? circuitCallOuts
                                  : circuitCallOuts.drop_back(numFinalChecks);
          ValueRange nonFinalCheckOutputs =
              numNonFinalChecks == 0
                  ? ValueRange{}
                  : beforeFinal.take_back(numNonFinalChecks);
          ValueRange nonFinalOutputs =
              numNonFinalChecks == 0
                  ? beforeFinal
                  : beforeFinal.drop_back(numNonFinalChecks);
          // Split non-final outputs into register outputs and delay buffer outputs
          ValueRange delayBufferOutputs = nonFinalOutputs.take_back(totalDelaySlots);
          ValueRange circuitOutputs = nonFinalOutputs.drop_back(totalDelaySlots);

          Value violated = iterArgs.back();
          if (numNonFinalChecks > 0) {
            // If we have a cycle up to which we ignore assertions, we need an
            // IfOp to track this.
            auto insideForPoint = builder.saveInsertionPoint();
            // We need to still have the yielded result of the op in scope after
            // we've built the check.
            Value yieldedValue;
            bool gateOnPosedge = !risingClocksOnly && clockIndexes.size() == 1;
            auto ignoreAssertionsUntil =
                op->getAttrOfType<IntegerAttr>("ignore_asserts_until");
            if (ignoreAssertionsUntil || gateOnPosedge) {
              Value shouldSkip;
              if (ignoreAssertionsUntil) {
                auto ignoreUntilConstant = arith::ConstantOp::create(
                    builder, loc,
                    rewriter.getI32IntegerAttr(
                        ignoreAssertionsUntil.getValue().getZExtValue()));
                auto shouldIgnore = arith::CmpIOp::create(
                    builder, loc, arith::CmpIPredicate::ult, i,
                    ignoreUntilConstant);
                shouldSkip = shouldIgnore;
              }
              if (gateOnPosedge) {
                auto one = arith::ConstantOp::create(
                    builder, loc, rewriter.getI32IntegerAttr(1));
                auto zero = arith::ConstantOp::create(
                    builder, loc, rewriter.getI32IntegerAttr(0));
                auto lsb = arith::AndIOp::create(builder, loc, i, one);
                // Skip even iterations; with initial clock = 0, posedges occur
                // on odd loop iterations after toggling.
                auto isEven = arith::CmpIOp::create(
                    builder, loc, arith::CmpIPredicate::eq, lsb, zero);
                if (shouldSkip)
                  shouldSkip = arith::OrIOp::create(builder, loc, shouldSkip,
                                                    isEven);
                else
                  shouldSkip = isEven;
              }
              auto ifShouldSkip = scf::IfOp::create(
                  builder, loc, builder.getI1Type(), shouldSkip, true);
              // If we should skip, yield the existing value.
              builder.setInsertionPointToEnd(
                  &ifShouldSkip.getThenRegion().front());
              scf::YieldOp::create(builder, loc, ValueRange(iterArgs.back()));
              builder.setInsertionPointToEnd(
                  &ifShouldSkip.getElseRegion().front());
              yieldedValue = ifShouldSkip.getResult(0);
            }

            // Check the non-final property for this iteration.
            Value checkVal = nonFinalCheckOutputs.front();
            Value isTrue;
            if (isa<smt::BoolType>(checkVal.getType())) {
              // LTL properties are converted to !smt.bool, use directly
              isTrue = checkVal;
            } else {
              // i1 properties are converted to !smt.bv<1>, compare with 1
              auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
              isTrue = smt::EqOp::create(builder, loc, checkVal, trueBV);
            }
            Value checkCond = isCoverCheck
                                   ? isTrue
                                   : smt::NotOp::create(builder, loc, isTrue);

            smt::PushOp::create(builder, loc, 1);
            smt::AssertOp::create(builder, loc, checkCond);
            auto checkOp =
                smt::CheckOp::create(rewriter, loc, builder.getI1Type());
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.createBlock(&checkOp.getSatRegion());
              smt::YieldOp::create(builder, loc, constTrue);
              builder.createBlock(&checkOp.getUnknownRegion());
              smt::YieldOp::create(builder, loc, constTrue);
              builder.createBlock(&checkOp.getUnsatRegion());
              smt::YieldOp::create(builder, loc, constFalse);
            }
            smt::PopOp::create(builder, loc, 1);

            Value newViolated = arith::OrIOp::create(
                builder, loc, checkOp.getResult(0), iterArgs.back());

            // If we've packaged everything in an IfOp, we need to yield the
            // new violated value.
            if (ignoreAssertionsUntil || gateOnPosedge) {
              scf::YieldOp::create(builder, loc, newViolated);
              // Replace the variable with the yielded value.
              violated = yieldedValue;
            } else {
              violated = newViolated;
            }

            // If we created an IfOp, make sure we start inserting after it
            // again.
            builder.restoreInsertionPoint(insideForPoint);
          }

          // Call loop func to update clock & state arg values
          SmallVector<Value> loopCallInputs;
          // Fetch clock values to feed to loop
          for (auto index : clockIndexes)
            loopCallInputs.push_back(iterArgs[index]);
          // Fetch state args to feed to loop
          for (auto stateArg :
               iterArgs.drop_back(1 + numFinalChecks).take_back(numStateArgs))
            loopCallInputs.push_back(stateArg);
          ValueRange loopVals =
              func::CallOp::create(builder, loc, loopFuncOp, loopCallInputs)
                  ->getResults();

          size_t loopIndex = 0;
          // Collect decls to yield at end of iteration
          SmallVector<Value> newDecls;
          size_t nonRegIdx = 0;
          size_t argIndex = 0;
          // Circuit args are: [clocks, inputs] [registers] [delay buffers]
          // Drop both registers and delay buffers to get just clocks and inputs
          size_t numNonStateArgs = oldCircuitInputTy.size() - numRegs - totalDelaySlots;
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldCircuitInputTy).take_front(numNonStateArgs),
                         TypeRange(circuitInputTy).take_front(numNonStateArgs))) {
            // Check if this is a clock - either explicit seq::ClockType or
            // an i1 that corresponds to an init clock
            bool isI1Type = isa<IntegerType>(oldTy) &&
                            cast<IntegerType>(oldTy).getWidth() == 1;
            bool isClock = isa<seq::ClockType>(oldTy) ||
                           (isI1Type && nonRegIdx < numInitClocks);
            if (isClock)
              newDecls.push_back(loopVals[loopIndex++]);
            else {
              auto decl = smt::DeclareFunOp::create(
                  builder, loc, newTy,
                  argIndex < inputNamePrefixes.size()
                      ? inputNamePrefixes[argIndex]
                      : StringAttr{});
              newDecls.push_back(decl);
              maybeAssertKnown(oldTy, decl, builder);
            }
            nonRegIdx++;
            argIndex++;
          }

          // Only update the registers on a clock posedge unless in rising
          // clocks only mode
          // Multi-clock designs use per-register clock gating when available.
          Value isPosedge;
          SmallVector<Value> posedges;
          bool usePosedge = !risingClocksOnly && clockIndexes.size() == 1;
          bool usePerRegPosedge =
              !risingClocksOnly && clockIndexes.size() > 1 && numRegs > 0;
          if (usePosedge) {
            auto clockIndex = clockIndexes[0];
            auto oldClock = iterArgs[clockIndex];
            // The clock is necessarily the first value returned by the loop
            // region
            auto newClock = loopVals[0];
            auto oldClockLow = smt::BVNotOp::create(builder, loc, oldClock);
            auto isPosedgeBV =
                smt::BVAndOp::create(builder, loc, oldClockLow, newClock);
            // Convert posedge bv<1> to bool
            auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
            isPosedge = smt::EqOp::create(builder, loc, isPosedgeBV, trueBV);
          } else if (usePerRegPosedge) {
            posedges.reserve(clockIndexes.size());
            auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
            for (auto [idx, clockIndex] : llvm::enumerate(clockIndexes)) {
              auto oldClock = iterArgs[clockIndex];
              auto newClock = loopVals[idx];
              auto oldClockLow =
                  smt::BVNotOp::create(builder, loc, oldClock);
              auto isPosedgeBV =
                  smt::BVAndOp::create(builder, loc, oldClockLow, newClock);
              posedges.push_back(
                  smt::EqOp::create(builder, loc, isPosedgeBV, trueBV));
            }
          }
          if (clockIndexes.size() >= 1) {
            SmallVector<Value> regInputs = circuitOutputs.take_back(numRegs);
            if (risingClocksOnly || clockIndexes.size() == 1) {
              if (risingClocksOnly) {
                // In rising clocks only mode we don't need to worry about
                // whether there was a posedge.
                newDecls.append(regInputs);
              } else {
                auto regStates =
                    iterArgs.take_front(circuitFuncOp.getNumArguments())
                        .take_back(numRegs + totalDelaySlots)
                        .drop_back(totalDelaySlots);
                SmallVector<Value> nextRegStates;
                for (auto [regState, regInput] :
                     llvm::zip(regStates, regInputs)) {
                  // Create an ITE to calculate the next reg state
                  // TODO: we create a lot of ITEs here that will slow things down
                  // - these could be avoided by making init/loop regions concrete
                  nextRegStates.push_back(smt::IteOp::create(
                      builder, loc, isPosedge, regInput, regState));
                }
                newDecls.append(nextRegStates);
              }
            } else if (usePerRegPosedge) {
              auto regStates =
                  iterArgs.take_front(circuitFuncOp.getNumArguments())
                      .take_back(numRegs + totalDelaySlots)
                      .drop_back(totalDelaySlots);
              SmallVector<Value> nextRegStates;
              nextRegStates.reserve(numRegs);
              for (auto [idx, pair] :
                   llvm::enumerate(llvm::zip(regStates, regInputs))) {
                auto [regState, regInput] = pair;
                Value regPosedge = posedges[regClockToLoopIndex[idx]];
                nextRegStates.push_back(smt::IteOp::create(
                    builder, loc, regPosedge, regInput, regState));
              }
              newDecls.append(nextRegStates);
            }
          }

          // Add delay buffer outputs for the next iteration
          // These are the shifted buffer values from the circuit
          if (totalDelaySlots > 0) {
            if (usePosedge) {
              auto delayStates =
                  iterArgs.take_front(circuitFuncOp.getNumArguments())
                      .take_back(totalDelaySlots);
              for (auto [delayState, delayVal] :
                   llvm::zip(delayStates, delayBufferOutputs)) {
                newDecls.push_back(smt::IteOp::create(builder, loc, isPosedge,
                                                     delayVal, delayState));
              }
            } else {
              for (Value delayVal : delayBufferOutputs)
                newDecls.push_back(delayVal);
            }
          }

          // Add the rest of the loop state args
          for (; loopIndex < loopVals.size(); ++loopIndex)
            newDecls.push_back(loopVals[loopIndex]);

          // Pass through finalCheckOutputs (already !smt.bv<1>) for next
          // iteration
          for (auto finalVal : finalCheckOutputs)
            newDecls.push_back(finalVal);
          newDecls.push_back(violated);

          scf::YieldOp::create(builder, loc, newDecls);
        });

    // Get the violation flag from the loop
    Value violated = forOp->getResults().back();

    // If there are final checks, compute any final assertion violation and
    // any final cover success.
    Value finalCheckViolated = constFalse;
    Value finalCoverHit = constFalse;
    if (numFinalChecks > 0 && checkFinalAtEnd) {
      auto results = forOp->getResults();
      size_t finalStart = results.size() - 1 - numFinalChecks;
      SmallVector<Value> finalAssertOutputs;
      SmallVector<Value> finalCoverOutputs;
      finalAssertOutputs.reserve(numFinalChecks);
      finalCoverOutputs.reserve(numFinalChecks);
      for (size_t i = 0; i < numFinalChecks; ++i) {
        Value finalVal = results[finalStart + i];
        if (finalCheckIsCover[i])
          finalCoverOutputs.push_back(finalVal);
        else
          finalAssertOutputs.push_back(finalVal);
      }

      if (!finalAssertOutputs.empty()) {
        for (Value finalVal : finalAssertOutputs) {
          Value isTrue;
          if (isa<smt::BoolType>(finalVal.getType())) {
            // LTL properties are converted to !smt.bool, use directly
            isTrue = finalVal;
          } else {
            // i1 properties are converted to !smt.bv<1>, compare with 1
            auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
            isTrue = smt::EqOp::create(rewriter, loc, finalVal, trueBV);
          }
          // Assert the negation: we're looking for cases where the check FAILS
          Value isFalse = smt::NotOp::create(rewriter, loc, isTrue);
          smt::AssertOp::create(rewriter, loc, isFalse);
        }
        // Now check if there's a satisfying assignment (i.e., a violation)
        auto finalCheckOp =
            smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.createBlock(&finalCheckOp.getSatRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCheckOp.getUnknownRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCheckOp.getUnsatRegion());
          smt::YieldOp::create(rewriter, loc, constFalse);
        }
        finalCheckViolated = finalCheckOp.getResult(0);
      }

      if (!finalCoverOutputs.empty()) {
        smt::PushOp::create(rewriter, loc, 1);
        SmallVector<Value> coverTerms;
        coverTerms.reserve(finalCoverOutputs.size());
        for (Value finalVal : finalCoverOutputs) {
          if (isa<smt::BoolType>(finalVal.getType())) {
            coverTerms.push_back(finalVal);
          } else {
            auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
            coverTerms.push_back(
                smt::EqOp::create(rewriter, loc, finalVal, trueBV));
          }
        }
        Value anyCover =
            coverTerms.size() == 1
                ? coverTerms.front()
                : smt::OrOp::create(rewriter, loc, coverTerms).getResult();
        smt::AssertOp::create(rewriter, loc, anyCover);
        auto finalCoverOp =
            smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.createBlock(&finalCoverOp.getSatRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCoverOp.getUnknownRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCoverOp.getUnsatRegion());
          smt::YieldOp::create(rewriter, loc, constFalse);
        }
        smt::PopOp::create(rewriter, loc, 1);
        finalCoverHit = finalCoverOp.getResult(0);
      }
    }

    // Combine results: true if no violations found
    // For assert check: !violated && !finalCheckViolated
    // For cover check: violated (we want to find a trace)
    Value res;
    if (isCoverCheck) {
      res = arith::OrIOp::create(rewriter, loc, violated, finalCoverHit);
    } else {
      Value anyViolation =
          arith::OrIOp::create(rewriter, loc, violated, finalCheckViolated);
      res = arith::XOrIOp::create(rewriter, loc, anyViolation, constTrue);
    }
    smt::YieldOp::create(rewriter, loc, res);
    rewriter.replaceOp(op, solver.getResults());
    return success();
  }

  Namespace &names;
  bool risingClocksOnly;
  SmallVectorImpl<Operation *> &propertylessBMCOps;
  SmallVectorImpl<Operation *> &coverBMCOps;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static void populateLTLToSMTConversionPatterns(TypeConverter &converter,
                                               RewritePatternSet &patterns) {
  patterns.add<LTLAndOpConversion, LTLOrOpConversion, LTLIntersectOpConversion,
               LTLNotOpConversion, LTLImplicationOpConversion,
               LTLEventuallyOpConversion, LTLUntilOpConversion,
               LTLBooleanConstantOpConversion, LTLDelayOpConversion,
               LTLPastOpConversion, LTLConcatOpConversion,
               LTLRepeatOpConversion, LTLGoToRepeatOpConversion,
               LTLNonConsecutiveRepeatOpConversion>(converter,
                                                    patterns.getContext());
}

void circt::populateVerifToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns, Namespace &names,
    bool risingClocksOnly, bool assumeKnownInputs,
    SmallVectorImpl<Operation *> &propertylessBMCOps,
    SmallVectorImpl<Operation *> &coverBMCOps) {
  // Add LTL operation conversion patterns
  populateLTLToSMTConversionPatterns(converter, patterns);

  // Add Verif operation conversion patterns
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion>(converter, patterns.getContext());
  patterns.add<LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(
      converter, patterns.getContext(), assumeKnownInputs);
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, risingClocksOnly,
      propertylessBMCOps, coverBMCOps);
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget verifTarget(getContext());
  verifTarget.addIllegalDialect<verif::VerifDialect>();
  verifTarget.addLegalDialect<smt::SMTDialect, arith::ArithDialect,
                              scf::SCFDialect, func::FuncDialect,
                              ltl::LTLDialect>();
  verifTarget.addLegalOp<UnrealizedConversionCastOp>();

  // Check BMC ops contain only one assertion (done outside pattern to avoid
  // issues with whether assertions are/aren't lowered yet)
  SymbolTable symbolTable(getOperation());
  SmallVector<Operation *> propertylessBMCOps;
  SmallVector<Operation *> coverBMCOps;
  WalkResult assertionCheck = getOperation().walk(
      [&](Operation *op) { // Check there is exactly one assertion and clock
        if (auto bmcOp = dyn_cast<verif::BoundedModelCheckingOp>(op)) {
          auto regTypes = TypeRange(bmcOp.getCircuit().getArgumentTypes())
                              .take_back(bmcOp.getNumRegs());
          for (auto [regType, initVal] :
               llvm::zip(regTypes, bmcOp.getInitialValues())) {
            if (!isa<UnitAttr>(initVal)) {
              int64_t regWidth = hw::getBitWidth(regType);
              if (regWidth <= 0) {
                op->emitError(
                    "initial values require registers with known bit widths");
                return WalkResult::interrupt();
              }
              if (auto initIntAttr = dyn_cast<IntegerAttr>(initVal)) {
                if (initIntAttr.getValue().getBitWidth() !=
                    static_cast<unsigned>(regWidth)) {
                  op->emitError(
                      "bit width of initial value does not match register");
                  return WalkResult::interrupt();
                }
                continue;
              }
              if (auto initBoolAttr = dyn_cast<BoolAttr>(initVal)) {
                if (regWidth != 1) {
                  op->emitError(
                      "bool initial value requires 1-bit register width");
                  return WalkResult::interrupt();
                }
                continue;
              }
              auto tyAttr = dyn_cast<TypedAttr>(initVal);
              if (!tyAttr || tyAttr.getType() != regType) {
                op->emitError("unsupported initial value for register");
                return WalkResult::interrupt();
              }
            }
          }
          // Check only one clock is present in the circuit inputs
          auto numClockArgs = 0;
          for (auto argType : bmcOp.getCircuit().getArgumentTypes())
            if (isa<seq::ClockType>(argType))
              numClockArgs++;
          // TODO: this can be removed once we have a way to associate reg
          // ins/outs with clocks
          if (numClockArgs > 1) {
            if (risingClocksOnly) {
              op->emitError("multi-clock BMC is not supported with "
                            "--rising-clocks-only");
              return WalkResult::interrupt();
            }
            unsigned numRegs = bmcOp.getNumRegs();
            auto regClocks =
                bmcOp->getAttrOfType<ArrayAttr>("bmc_reg_clocks");
            if (numRegs > 0 &&
                (!regClocks || regClocks.size() != numRegs)) {
              op->emitError("multi-clock BMC requires bmc_reg_clocks with one "
                            "entry per register");
              return WalkResult::interrupt();
            }
            if (numRegs > 0 &&
                !bmcOp->getAttrOfType<ArrayAttr>("bmc_input_names")) {
              op->emitError(
                  "multi-clock BMC requires bmc_input_names for clock mapping");
              return WalkResult::interrupt();
            }
          }
          SmallVector<mlir::Operation *> worklist;
          int numAssertions = 0;
          int numCovers = 0;
          op->walk([&](Operation *curOp) {
            if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
              numAssertions++;
            }
            if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
              numCovers++;
            }
            if (auto inst = dyn_cast<InstanceOp>(curOp))
              worklist.push_back(symbolTable.lookup(inst.getModuleName()));
          });
          // TODO: probably negligible compared to actual model checking time
          // but cacheing the assertion count of modules would speed this up
          while (!worklist.empty()) {
            auto *module = worklist.pop_back_val();
            module->walk([&](Operation *curOp) {
              if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
                numAssertions++;
              }
              if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
                numCovers++;
              }
              if (auto inst = dyn_cast<InstanceOp>(curOp))
                worklist.push_back(symbolTable.lookup(inst.getModuleName()));
            });
            if (numAssertions > 1 || numCovers > 1)
              break;
          }
          if (numAssertions == 0 && numCovers == 0) {
            op->emitWarning("no property provided to check in module - will "
                            "trivially find no violations.");
            propertylessBMCOps.push_back(bmcOp);
          }
          if (numAssertions > 0 && numCovers > 0) {
            op->emitError(
                "bounded model checking problems with mixed assert/cover "
                "properties are not yet correctly handled - instead, check one "
                "kind at a time");
            return WalkResult::interrupt();
          }
          if (numCovers > 0)
            coverBMCOps.push_back(bmcOp);
        }
        return WalkResult::advance();
      });
  if (assertionCheck.wasInterrupted())
    return signalPassFailure();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);

  // Add LTL type conversions to SMT boolean
  // LTL sequences and properties are converted to SMT booleans for BMC
  converter.addConversion([](ltl::SequenceType type) -> Type {
    return smt::BoolType::get(type.getContext());
  });
  converter.addConversion([](ltl::PropertyType type) -> Type {
    return smt::BoolType::get(type.getContext());
  });

  // Keep assert/assume/cover legal inside BMC so the BMC conversion can handle
  // them, and illegal elsewhere so they get lowered normally.
  auto isInsideBMC = [](Operation *op) {
    return op->getParentOfType<verif::BoundedModelCheckingOp>() != nullptr;
  };
  verifTarget.addDynamicallyLegalOp<verif::AssertOp>(
      [&](verif::AssertOp op) { return isInsideBMC(op); });
  verifTarget.addDynamicallyLegalOp<verif::AssumeOp>(
      [&](verif::AssumeOp op) { return isInsideBMC(op); });
  verifTarget.addDynamicallyLegalOp<verif::CoverOp>(
      [&](verif::CoverOp op) { return isInsideBMC(op); });

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);

  // First phase: lower Verif operations, leaving LTL ops untouched.
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion>(converter, patterns.getContext());
  patterns.add<LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(
      converter, patterns.getContext(), assumeKnownInputs);
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, risingClocksOnly,
      propertylessBMCOps, coverBMCOps);

  if (failed(mlir::applyPartialConversion(getOperation(), verifTarget,
                                          std::move(patterns))))
    return signalPassFailure();

  // Second phase: lower remaining LTL operations to SMT.
  ConversionTarget ltlTarget(getContext());
  ltlTarget.addLegalDialect<smt::SMTDialect, arith::ArithDialect,
                            scf::SCFDialect, func::FuncDialect,
                            comb::CombDialect, hw::HWDialect, seq::SeqDialect,
                            verif::VerifDialect>();
  ltlTarget.addLegalOp<UnrealizedConversionCastOp>();
  ltlTarget.addIllegalOp<ltl::AndOp, ltl::OrOp, ltl::IntersectOp, ltl::NotOp,
                         ltl::ImplicationOp, ltl::EventuallyOp, ltl::UntilOp,
                         ltl::BooleanConstantOp, ltl::DelayOp, ltl::ConcatOp,
                         ltl::RepeatOp, ltl::GoToRepeatOp,
                         ltl::NonConsecutiveRepeatOp, ltl::PastOp>();

  RewritePatternSet ltlPatterns(&getContext());
  populateLTLToSMTConversionPatterns(converter, ltlPatterns);
  if (failed(mlir::applyPartialConversion(getOperation(), ltlTarget,
                                          std::move(ltlPatterns))))
    return signalPassFailure();

  RewritePatternSet relocatePatterns(&getContext());
  relocatePatterns.add<RelocateCastIntoInputRegion>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(relocatePatterns))))
    return signalPassFailure();

  RewritePatternSet postPatterns(&getContext());
  postPatterns.add<BoolBVCastOpRewrite>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(postPatterns))))
    return signalPassFailure();

  RewritePatternSet regionFixups(&getContext());
  regionFixups.add<RelocateSMTEqIntoOperandRegion>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(regionFixups))))
    return signalPassFailure();
}
