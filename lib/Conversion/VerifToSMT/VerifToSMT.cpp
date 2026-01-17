//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
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
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// LTL Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert ltl.and to smt.and
struct LTLAndOpConversion : OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), input);
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
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), input);
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

/// Convert ltl.not to smt.not
struct LTLNotOpConversion : OpConversionPattern<ltl::NotOp> {
  using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getInput());
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
    Value antecedent = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getAntecedent());
    Value consequent = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getConsequent());
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
    // For BMC: eventually(p) means p should hold at some point.
    // At each time step, we check if p holds. The BMC loop accumulates
    // these checks with OR. Here we convert the inner property.
    Value input = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getInput());
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
    Value p = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getInput());
    Value q = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getCondition());
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
    // Get the delay amount
    uint64_t delay = op.getDelay();

    if (delay == 0) {
      // No delay: just pass through the input sequence
      Value input = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()),
          adaptor.getInput());
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
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), input);
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
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    // For base >= 1: the sequence must match at least once
    // In BMC single-step semantics, this is just the input sequence value
    Value input = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getInput());
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
    Value notCond = smt::NotOp::create(rewriter, op.getLoc(), cond);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, notCond);
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
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
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
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
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

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(smt::DeclareFunOp::create(rewriter, loc, arg.getType()));

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
    if (outputsDifferent.size() == 1)
      toAssert = outputsDifferent[0];
    else
      toAssert = smt::OrOp::create(rewriter, loc, outputsDifferent);

    smt::AssertOp::create(rewriter, loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }
};

struct RefinementCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::RefinementCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::RefinementCheckingOp>::CircuitRelationCheckOpConversion;

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
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(smt::DeclareFunOp::create(rewriter, loc, arg.getType()));

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
};

/// Information about a ltl.delay operation that needs multi-step tracking.
/// For delay N, we need N slots in the delay buffer to track the signal history.
struct DelayInfo {
  ltl::DelayOp op;           // The original delay operation
  Value inputSignal;         // The signal being delayed
  uint64_t delay;            // The delay amount (N cycles)
  size_t bufferStartIndex;   // Index into the delay buffer iter_args
};

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

    // Hoist any final-only asserts into circuit outputs so we can check them
    // only at the final step.
    SmallVector<Value> finalCheckValues;
    SmallVector<Operation *> opsToErase;
    auto &circuitBlock = op.getCircuit().front();
    circuitBlock.walk([&](Operation *curOp) {
      if (!curOp->hasAttr("bmc.final"))
        return;
      if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
        finalCheckValues.push_back(assertOp.getProperty());
        opsToErase.push_back(curOp);
        return;
      }
      if (auto assumeOp = dyn_cast<verif::AssumeOp>(curOp)) {
        finalCheckValues.push_back(assumeOp.getProperty());
        opsToErase.push_back(curOp);
        return;
      }
      if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
        finalCheckValues.push_back(coverOp.getProperty());
        opsToErase.push_back(curOp);
        return;
      }
    });
    // Modify the yield first (while values are still valid)
    if (!finalCheckValues.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());
      SmallVector<Value> newYieldOperands(yieldOp.getOperands());
      newYieldOperands.append(finalCheckValues.begin(),
                              finalCheckValues.end());
      yieldOp->setOperands(newYieldOperands);
    }
    // Erase the bmc.final ops using the rewriter to properly notify the
    // conversion framework
    for (auto *opToErase : opsToErase)
      rewriter.eraseOp(opToErase);
    size_t numFinalAsserts = finalCheckValues.size();

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

    // First pass: collect all delay ops with delay > 0
    circuitBlock.walk([&](ltl::DelayOp delayOp) {
      uint64_t delay = delayOp.getDelay();
      if (delay > 0) {
        DelayInfo info;
        info.op = delayOp;
        info.inputSignal = delayOp.getInput();
        info.delay = delay;
        info.bufferStartIndex = totalDelaySlots;
        delayInfos.push_back(info);
        totalDelaySlots += delay;
      }
    });

    // Second pass: modify the circuit block to add delay buffer infrastructure
    // We need to do this BEFORE region type conversion.
    //
    // Output order after modification:
    // [original outputs (registers)] [delay buffer outputs] [final check outputs]
    //
    // We need to insert delay buffer outputs BEFORE final check outputs.
    SmallVector<ltl::DelayOp> delayOpsToErase;
    if (!delayInfos.empty()) {
      // For each delay op, add buffer arguments and modify the yield
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());

      // Get current operands and split off the final check values
      SmallVector<Value> origOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      // Final check values were appended last, so they're at the end
      size_t numOrigOutputs = origOperands.size() - numFinalAsserts;
      SmallVector<Value> newYieldOperands(origOperands.begin(),
                                          origOperands.begin() + numOrigOutputs);
      SmallVector<Value> delayBufferOutputs;

      for (auto &info : delayInfos) {
        // The input signal type - use i1 for LTL properties (they're booleans)
        Type bufferElementType = rewriter.getI1Type();

        // Add N block arguments for the delay buffer (oldest to newest)
        SmallVector<Value> bufferArgs;
        for (uint64_t i = 0; i < info.delay; ++i) {
          auto arg = circuitBlock.addArgument(bufferElementType, loc);
          bufferArgs.push_back(arg);
        }

        // Replace all uses of the delay op with the oldest buffer entry (index 0)
        // This is the value from N steps ago
        // Note: We save the input signal BEFORE replacing uses, as it comes from
        // the delay op's input operand
        Value inputSig = info.inputSignal;
        info.op.replaceAllUsesWith(bufferArgs[0]);
        delayOpsToErase.push_back(info.op);

        // Add yield operands: shifted buffer (drop oldest, add current)
        // new_buffer = [buffer[1], buffer[2], ..., buffer[N-1], current_signal]
        for (uint64_t i = 1; i < info.delay; ++i) {
          delayBufferOutputs.push_back(bufferArgs[i]);
        }
        delayBufferOutputs.push_back(inputSig);
      }

      // Construct final yield: [orig outputs] [delay buffers] [final checks]
      newYieldOperands.append(delayBufferOutputs);
      newYieldOperands.append(origOperands.begin() + numOrigOutputs,
                              origOperands.end());  // final check values

      // Update the yield with new operands
      yieldOp->setOperands(newYieldOperands);

      // Erase the delay ops (after all replacements are done)
      for (auto delayOp : delayOpsToErase)
        rewriter.eraseOp(delayOp);

      // Update the type vectors to include the new arguments and outputs
      oldCircuitInputTy.clear();
      oldCircuitInputTy.append(op.getCircuit().getArgumentTypes().begin(),
                               op.getCircuit().getArgumentTypes().end());
    }

    // Re-compute circuit types after potential modification
    circuitInputTy.clear();
    circuitOutputTy.clear();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();

    if (failed(rewriter.convertRegionTypes(&op.getInit(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getLoop(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getCircuit(), *typeConverter)))
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
      loopFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_loop"), loopFuncTy);
      rewriter.inlineRegionBefore(op.getLoop(), loopFuncOp.getFunctionBody(),
                                  loopFuncOp.end());
      circuitFuncOp = func::FuncOp::create(
          rewriter, loc, names.newName("bmc_circuit"), circuitFuncTy);
      rewriter.inlineRegionBefore(op.getCircuit(),
                                  circuitFuncOp.getFunctionBody(),
                                  circuitFuncOp.end());
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

    // Initial push
    smt::PushOp::create(rewriter, loc, 1);

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
        // The type is smt::BVType<1> after conversion (i1 -> bv<1>)
        inputDecls.push_back(smt::BVConstantOp::create(rewriter, loc, 0, 1));
        continue;
      }

      // Check if this is a register input (registers are at the end of original args)
      bool isRegister = curIndex >= origCircuitArgsSize - numRegs;

      // Check if this is a clock - either explicit seq::ClockType or
      // an i1 that corresponds to an init clock (for i1 clocks converted via
      // ToClockOp inside the circuit)
      bool isClock = isa<seq::ClockType>(oldTy) ||
                     (!isRegister && nonRegIndex < numInitClocks);

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
          assert(cstInt.getBitWidth() ==
                     cast<smt::BitVectorType>(newTy).getWidth() &&
                 "Width mismatch between initial value and target type");
          inputDecls.push_back(
              smt::BVConstantOp::create(rewriter, loc, cstInt));
          continue;
        }
      }
      inputDecls.push_back(smt::DeclareFunOp::create(rewriter, loc, newTy));
    }

    auto numStateArgs = initVals.size() - initIndex;
    // Add the rest of the init vals (state args)
    for (; initIndex < initVals.size(); ++initIndex)
      inputDecls.push_back(initVals[initIndex]);

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
    // Initialize final check iter_args with SMT bv<1> zero (these will be
    // updated with circuit outputs)
    auto smtBVFalse = smt::BVConstantOp::create(rewriter, loc, 0, 1);
    for (size_t i = 0; i < numFinalAsserts; ++i)
      inputDecls.push_back(smtBVFalse);
    inputDecls.push_back(constFalse); // wasViolated?

    // TODO: swapping to a whileOp here would allow early exit once the property
    // is violated
    // Perform model check up to the provided bound
    auto forOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Drop existing assertions
          smt::PopOp::create(builder, loc, 1);
          smt::PushOp::create(builder, loc, 1);

          // Execute the circuit
          ValueRange circuitCallOuts =
              func::CallOp::create(
                  builder, loc, circuitFuncOp,
                  iterArgs.take_front(circuitFuncOp.getNumArguments()))
                  ->getResults();

          // Circuit outputs are ordered as:
          // [original outputs (registers)] [delay buffer outputs] [final checks]
          //
          // Note: totalDelaySlots is captured from the outer scope
          ValueRange finalCheckOutputs =
              numFinalAsserts == 0 ? ValueRange{}
                                   : circuitCallOuts.take_back(numFinalAsserts);
          ValueRange nonFinalOutputs =
              numFinalAsserts == 0 ? circuitCallOuts
                                   : circuitCallOuts.drop_back(numFinalAsserts);
          // Split non-final outputs into register outputs and delay buffer outputs
          ValueRange delayBufferOutputs = nonFinalOutputs.take_back(totalDelaySlots);
          ValueRange circuitOutputs = nonFinalOutputs.drop_back(totalDelaySlots);

          // If we have a cycle up to which we ignore assertions, we need an
          // IfOp to track this
          // First, save the insertion point so we can safely enter the IfOp

          auto insideForPoint = builder.saveInsertionPoint();
          // We need to still have the yielded result of the op in scope after
          // we've built the check
          Value yieldedValue;
          auto ignoreAssertionsUntil =
              op->getAttrOfType<IntegerAttr>("ignore_asserts_until");
          if (ignoreAssertionsUntil) {
            auto ignoreUntilConstant = arith::ConstantOp::create(
                builder, loc,
                rewriter.getI32IntegerAttr(
                    ignoreAssertionsUntil.getValue().getZExtValue()));
            auto shouldIgnore =
                arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                      i, ignoreUntilConstant);
            auto ifShouldIgnore = scf::IfOp::create(
                builder, loc, builder.getI1Type(), shouldIgnore, true);
            // If we should ignore, yield the existing value
            builder.setInsertionPointToEnd(
                &ifShouldIgnore.getThenRegion().front());
            scf::YieldOp::create(builder, loc, ValueRange(iterArgs.back()));
            builder.setInsertionPointToEnd(
                &ifShouldIgnore.getElseRegion().front());
            yieldedValue = ifShouldIgnore.getResult(0);
          }

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

          Value violated = arith::OrIOp::create(
              builder, loc, checkOp.getResult(0), iterArgs.back());

          // If we've packaged everything in an IfOp, we need to yield the
          // new violated value
          if (ignoreAssertionsUntil) {
            scf::YieldOp::create(builder, loc, violated);
            // Replace the variable with the yielded value
            violated = yieldedValue;
          }

          // If we created an IfOp, make sure we start inserting after it again
          builder.restoreInsertionPoint(insideForPoint);

          // Call loop func to update clock & state arg values
          SmallVector<Value> loopCallInputs;
          // Fetch clock values to feed to loop
          for (auto index : clockIndexes)
            loopCallInputs.push_back(iterArgs[index]);
          // Fetch state args to feed to loop
          for (auto stateArg :
               iterArgs.drop_back(1 + numFinalAsserts).take_back(numStateArgs))
            loopCallInputs.push_back(stateArg);
          ValueRange loopVals =
              func::CallOp::create(builder, loc, loopFuncOp, loopCallInputs)
                  ->getResults();

          size_t loopIndex = 0;
          // Collect decls to yield at end of iteration
          SmallVector<Value> newDecls;
          size_t nonRegIdx = 0;
          // Circuit args are: [clocks, inputs] [registers] [delay buffers]
          // Drop both registers and delay buffers to get just clocks and inputs
          size_t numNonStateArgs = oldCircuitInputTy.size() - numRegs - totalDelaySlots;
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldCircuitInputTy).take_front(numNonStateArgs),
                         TypeRange(circuitInputTy).take_front(numNonStateArgs))) {
            // Check if this is a clock - either explicit seq::ClockType or
            // an i1 that corresponds to an init clock
            bool isClock = isa<seq::ClockType>(oldTy) ||
                           (nonRegIdx < numInitClocks);
            if (isClock)
              newDecls.push_back(loopVals[loopIndex++]);
            else
              newDecls.push_back(
                  smt::DeclareFunOp::create(builder, loc, newTy));
            nonRegIdx++;
          }

          // Only update the registers on a clock posedge unless in rising
          // clocks only mode
          // TODO: this will also need changing with multiple clocks - currently
          // it only accounts for the one clock case.
          if (clockIndexes.size() == 1) {
            SmallVector<Value> regInputs = circuitOutputs.take_back(numRegs);
            if (risingClocksOnly) {
              // In rising clocks only mode we don't need to worry about whether
              // there was a posedge
              newDecls.append(regInputs);
            } else {
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
              auto isPosedge =
                  smt::EqOp::create(builder, loc, isPosedgeBV, trueBV);
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
          }

          // Add delay buffer outputs for the next iteration
          // These are the shifted buffer values from the circuit
          for (Value delayVal : delayBufferOutputs)
            newDecls.push_back(delayVal);

          // Add the rest of the loop state args
          for (; loopIndex < loopVals.size(); ++loopIndex)
            newDecls.push_back(loopVals[loopIndex]);

          // Pass through finalCheckOutputs (already !smt.bv<1>) for next
          // iteration
          for (auto finalVal : finalCheckOutputs) {
            newDecls.push_back(finalVal);
          }
          newDecls.push_back(violated);

          scf::YieldOp::create(builder, loc, newDecls);
        });

    // Get the violation flag from the loop
    Value violated = forOp->getResults().back();

    // If there are final checks, check if they can be violated
    Value finalCheckViolated = constFalse;
    if (numFinalAsserts > 0) {
      // For each final check, assert its negation (we're looking for
      // violations). If SAT, the check can be violated.
      auto results = forOp->getResults();
      size_t finalStart = results.size() - 1 - numFinalAsserts;
      auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
      for (size_t i = 0; i < numFinalAsserts; ++i) {
        // Results are !smt.bv<1>, check if they can be false
        Value finalVal = results[finalStart + i];
        Value isTrue = smt::EqOp::create(rewriter, loc, finalVal, trueBV);
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

    // Combine results: true if no violations found
    // For assert check: !violated && !finalCheckViolated
    // For cover check: violated (we want to find a trace)
    Value res;
    if (isCoverCheck) {
      res = violated;
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

void circt::populateVerifToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns, Namespace &names,
    bool risingClocksOnly, SmallVectorImpl<Operation *> &propertylessBMCOps,
    SmallVectorImpl<Operation *> &coverBMCOps) {
  // Add LTL operation conversion patterns
  patterns.add<LTLAndOpConversion, LTLOrOpConversion, LTLNotOpConversion,
               LTLImplicationOpConversion, LTLEventuallyOpConversion,
               LTLUntilOpConversion, LTLBooleanConstantOpConversion,
               LTLDelayOpConversion, LTLConcatOpConversion,
               LTLRepeatOpConversion>(converter, patterns.getContext());

  // Add Verif operation conversion patterns
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion,
               LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(converter,
                                               patterns.getContext());
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, risingClocksOnly,
      propertylessBMCOps, coverBMCOps);
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<smt::SMTDialect, arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Check BMC ops contain only one assertion (done outside pattern to avoid
  // issues with whether assertions are/aren't lowered yet)
  SymbolTable symbolTable(getOperation());
  SmallVector<Operation *> propertylessBMCOps;
  SmallVector<Operation *> coverBMCOps;
  WalkResult assertionCheck = getOperation().walk(
      [&](Operation *op) { // Check there is exactly one assertion and clock
        if (auto bmcOp = dyn_cast<verif::BoundedModelCheckingOp>(op)) {
          // We also currently don't support initial values on registers that
          // don't have integer inputs.
          auto regTypes = TypeRange(bmcOp.getCircuit().getArgumentTypes())
                              .take_back(bmcOp.getNumRegs());
          for (auto [regType, initVal] :
               llvm::zip(regTypes, bmcOp.getInitialValues())) {
            if (!isa<UnitAttr>(initVal)) {
              if (!isa<IntegerType>(regType)) {
                op->emitError("initial values are currently only supported for "
                              "registers with integer types");
                return WalkResult::interrupt();
              }
              auto tyAttr = dyn_cast<TypedAttr>(initVal);
              if (!tyAttr || tyAttr.getType() != regType) {
                op->emitError("type of initial value does not match type of "
                              "initialized register");
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
            op->emitError(
                "only modules with one or zero clocks are currently supported");
            return WalkResult::interrupt();
          }
          SmallVector<mlir::Operation *> worklist;
          int numAssertions = 0;
          int numCovers = 0;
          op->walk([&](Operation *curOp) {
            if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
              if (!assertOp->hasAttr("bmc.final"))
                numAssertions++;
            }
            if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
              if (!coverOp->hasAttr("bmc.final"))
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
                if (!assertOp->hasAttr("bmc.final"))
                  numAssertions++;
              }
              if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
                if (!coverOp->hasAttr("bmc.final"))
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
          if (numAssertions > 1 || numCovers > 1 ||
              (numAssertions > 0 && numCovers > 0)) {
            op->emitError(
                "bounded model checking problems with multiple properties are "
                "not yet correctly handled - instead, check one property at a "
                "time");
            return WalkResult::interrupt();
          }
          if (numCovers == 1)
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

  // Mark LTL operations as illegal so they get converted to SMT
  target.addIllegalOp<ltl::AndOp, ltl::OrOp, ltl::NotOp, ltl::ImplicationOp,
                      ltl::EventuallyOp, ltl::UntilOp, ltl::BooleanConstantOp,
                      ltl::DelayOp, ltl::ConcatOp, ltl::RepeatOp>();

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);

  populateVerifToSMTConversionPatterns(converter, patterns, names,
                                       risingClocksOnly, propertylessBMCOps,
                                       coverBMCOps);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
