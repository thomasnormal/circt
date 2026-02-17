//===- CombToSMT.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to smt::RepeatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<smt::RepeatOp>(op, op.getMultiple(),
                                               adaptor.getInput());
    return success();
  }
};

/// Lower a comb::TruthTableOp operation to an SMT-LIB-compatible boolean
/// expression tree.
struct TruthTableOpConversion : OpConversionPattern<TruthTableOp> {
  using OpConversionPattern<TruthTableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruthTableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tableAttr = op.getLookupTable();
    SmallVector<bool> table;
    table.reserve(tableAttr.size());
    for (bool entry : tableAttr)
      table.push_back(entry);

    auto allTrue = llvm::all_of(table, [](bool v) { return v; });
    auto allFalse = llvm::all_of(table, [](bool v) { return !v; });

    auto resultType =
        dyn_cast_or_null<smt::BitVectorType>(typeConverter->convertType(
            op.getType()));
    if (!resultType || resultType.getWidth() != 1)
      return rewriter.notifyMatchFailure(op, "expected smt.bv<1> result type");

    if (allTrue || allFalse) {
      rewriter.replaceOpWithNewOp<smt::BVConstantOp>(
          op, APInt(1, allTrue ? 1 : 0));
      return success();
    }

    Value falseVal = smt::BVConstantOp::create(rewriter, loc, APInt(1, 0));
    Value trueVal = smt::BVConstantOp::create(rewriter, loc, APInt(1, 1));

    auto inputs = adaptor.getInputs();
    if (inputs.empty()) {
      rewriter.replaceOp(op, table.front() ? trueVal : falseVal);
      return success();
    }

    auto domainType = smt::BitVectorType::get(getContext(), inputs.size());
    auto arrayType = smt::ArrayType::get(getContext(), domainType, resultType);
    Value array =
        smt::ArrayBroadcastOp::create(rewriter, loc, arrayType, falseVal);
    for (auto [idx, value] : llvm::enumerate(table)) {
      if (!value)
        continue;
      Value idxVal =
          smt::BVConstantOp::create(rewriter, loc, APInt(inputs.size(), idx));
      array = smt::ArrayStoreOp::create(rewriter, loc, array, idxVal, trueVal);
    }

    Value index = inputs.front();
    for (Value bit : inputs.drop_front())
      index = smt::ConcatOp::create(rewriter, loc, index, bit);

    rewriter.replaceOpWithNewOp<smt::ArraySelectOp>(op, array, index);
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a smt::BVCmpOp, smt::EqOp or
/// smt::DistinctOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SMT lowering is 2-state; treat case/wild equality as eq/ne.
    auto predicate = adaptor.getPredicate();
    if (predicate == ICmpPredicate::weq || predicate == ICmpPredicate::ceq)
      predicate = ICmpPredicate::eq;
    if (predicate == ICmpPredicate::wne || predicate == ICmpPredicate::cne)
      predicate = ICmpPredicate::ne;

    Value result;
    if (predicate == ICmpPredicate::eq) {
      result = smt::EqOp::create(rewriter, op.getLoc(), adaptor.getLhs(),
                                 adaptor.getRhs());
    } else if (predicate == ICmpPredicate::ne) {
      result = smt::DistinctOp::create(rewriter, op.getLoc(), adaptor.getLhs(),
                                       adaptor.getRhs());
    } else {
      smt::BVCmpPredicate pred;
      switch (predicate) {
      case ICmpPredicate::sge:
        pred = smt::BVCmpPredicate::sge;
        break;
      case ICmpPredicate::sgt:
        pred = smt::BVCmpPredicate::sgt;
        break;
      case ICmpPredicate::sle:
        pred = smt::BVCmpPredicate::sle;
        break;
      case ICmpPredicate::slt:
        pred = smt::BVCmpPredicate::slt;
        break;
      case ICmpPredicate::uge:
        pred = smt::BVCmpPredicate::uge;
        break;
      case ICmpPredicate::ugt:
        pred = smt::BVCmpPredicate::ugt;
        break;
      case ICmpPredicate::ule:
        pred = smt::BVCmpPredicate::ule;
        break;
      case ICmpPredicate::ult:
        pred = smt::BVCmpPredicate::ult;
        break;
      default:
        llvm_unreachable("all cases handled above");
      }

      result = smt::BVCmpOp::create(rewriter, op.getLoc(), pred,
                                    adaptor.getLhs(), adaptor.getRhs());
    }

    Value convVal = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), typeConverter->convertType(op.getType()),
        result);
    if (!convVal)
      return failure();

    rewriter.replaceOp(op, convVal);
    return success();
  }
};

/// Lower a comb::ExtractOp operation to an smt::ExtractOp
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<smt::ExtractOp>(
        op, typeConverter->convertType(op.getResult().getType()),
        adaptor.getLowBitAttr(), adaptor.getInput());
    return success();
  }
};

/// Lower a comb::MuxOp operation to an smt::IteOp
struct MuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getCond());
    rewriter.replaceOpWithNewOp<smt::IteOp>(
        op, condition, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// Lower a comb::SubOp operation to an smt::BVNegOp + smt::BVAddOp
struct SubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value negRhs =
        smt::BVNegOp::create(rewriter, op.getLoc(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::BVAddOp>(op, adaptor.getLhs(), negRhs);
    return success();
  }
};

/// Lower a comb::ParityOp operation to a chain of smt::Extract + XOr ops
struct ParityOpConversion : OpConversionPattern<ParityOp> {
  using OpConversionPattern<ParityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned bitwidth =
        cast<smt::BitVectorType>(adaptor.getInput().getType()).getWidth();

    // Note: the SMT bitvector type does not support 0 bitwidth vectors and thus
    // the type conversion should already fail.
    Type oneBitTy = smt::BitVectorType::get(getContext(), 1);
    Value runner =
        smt::ExtractOp::create(rewriter, loc, oneBitTy, 0, adaptor.getInput());
    for (unsigned i = 1; i < bitwidth; ++i) {
      Value ext = smt::ExtractOp::create(rewriter, loc, oneBitTy, i,
                                         adaptor.getInput());
      runner = smt::BVXOrOp::create(rewriter, loc, runner, ext);
    }

    rewriter.replaceOp(op, runner);
    return success();
  }
};

/// Lower the SourceOp to the TargetOp one-to-one.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(
        op,
        OpConversionPattern<SourceOp>::typeConverter->convertType(
            op.getResult().getType()),
        adaptor.getOperands());
    return success();
  }
};

/// Lower the SourceOp to the TargetOp special-casing if the second operand is
/// zero to return a new symbolic value.
template <typename SourceOp, typename TargetOp>
struct DivisionOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = dyn_cast<smt::BitVectorType>(adaptor.getRhs().getType());
    if (!type)
      return failure();

    auto resultType = OpConversionPattern<SourceOp>::typeConverter->convertType(
        op.getResult().getType());
    Value zero =
        smt::BVConstantOp::create(rewriter, loc, APInt(type.getWidth(), 0));
    Value isZero = smt::EqOp::create(rewriter, loc, adaptor.getRhs(), zero);
    Value symbolicVal = smt::DeclareFunOp::create(rewriter, loc, resultType);
    Value division =
        TargetOp::create(rewriter, loc, resultType, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, isZero, symbolicVal, division);
    return success();
  }
};

/// Converts an operation with a variadic number of operands to a chain of
/// binary operations assuming left-associativity of the operation.
template <typename SourceOp, typename TargetOp>
struct VariadicToBinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 2)
      return failure();

    Value runner = operands[0];
    for (Value operand : operands.drop_front())
      runner = TargetOp::create(rewriter, op.getLoc(), runner, operand);

    rewriter.replaceOp(op, runner);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LLVM Dialect Conversion Patterns (for operations used by MooreToCore)
//===----------------------------------------------------------------------===//

/// Lower an llvm.intr.ctpop (count population / popcount) operation to SMT.
/// SMT-LIB2 does not have a native popcount operation, so we implement it
/// by extracting each bit and summing them.
///
/// For a bitvector of width N, the result is:
///   sum(i=0 to N-1) zext(extract(x, i, i))
///
/// This is used for $countones, $onehot, and $onehot0 SystemVerilog functions.
struct LLVMCtPopOpConversion : OpConversionPattern<LLVM::CtPopOp> {
  using OpConversionPattern<LLVM::CtPopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CtPopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getIn();
    auto inputType = dyn_cast<smt::BitVectorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a bitvector type");

    unsigned bitWidth = inputType.getWidth();
    if (bitWidth == 0)
      return rewriter.notifyMatchFailure(op, "0-bit vectors not supported");

    // Result type is the same width as input (popcount result fits in input width)
    Type oneBitTy = smt::BitVectorType::get(getContext(), 1);

    // Start with zero
    Value result =
        smt::BVConstantOp::create(rewriter, loc, APInt(bitWidth, 0));

    // Sum each bit: for each bit position, extract it, zero-extend to full
    // width, and add to the running sum.
    for (unsigned i = 0; i < bitWidth; ++i) {
      // Extract bit i
      Value bit =
          smt::ExtractOp::create(rewriter, loc, oneBitTy, i, input);

      // Zero-extend to the result width
      Value extended;
      if (bitWidth > 1) {
        // Zero-extend by concatenating zeros on the left
        Value zeros = smt::BVConstantOp::create(rewriter, loc,
                                                 APInt(bitWidth - 1, 0));
        extended = smt::ConcatOp::create(rewriter, loc, zeros, bit);
      } else {
        extended = bit;
      }

      // Add to running sum
      result = smt::BVAddOp::create(rewriter, loc, result, extended);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToSMTPass
    : public circt::impl::ConvertCombToSMTBase<ConvertCombToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToSMTConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns) {
  patterns.add<CombReplicateOpConversion, IcmpOpConversion, ExtractOpConversion,
               SubOpConversion, MuxOpConversion, ParityOpConversion,
               TruthTableOpConversion,
               OneToOneOpConversion<ShlOp, smt::BVShlOp>,
               OneToOneOpConversion<ShrUOp, smt::BVLShrOp>,
               OneToOneOpConversion<ShrSOp, smt::BVAShrOp>,
               DivisionOpConversion<DivSOp, smt::BVSDivOp>,
               DivisionOpConversion<DivUOp, smt::BVUDivOp>,
               DivisionOpConversion<ModSOp, smt::BVSRemOp>,
               DivisionOpConversion<ModUOp, smt::BVURemOp>,
               VariadicToBinaryOpConversion<ConcatOp, smt::ConcatOp>,
               VariadicToBinaryOpConversion<AddOp, smt::BVAddOp>,
               VariadicToBinaryOpConversion<MulOp, smt::BVMulOp>,
               VariadicToBinaryOpConversion<AndOp, smt::BVAndOp>,
               VariadicToBinaryOpConversion<OrOp, smt::BVOrOp>,
               VariadicToBinaryOpConversion<XorOp, smt::BVXOrOp>,
               // LLVM intrinsics used by MooreToCore for bit manipulation
               LLVMCtPopOpConversion>(converter, patterns.getContext());

  // NOTE: SMT lowering is 2-state; comb.truth_table's xprop-oriented semantics
  // are approximated by a 2-state lookup table.
}

void ConvertCombToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalOp<seq::FromClockOp>();
  target.addIllegalOp<seq::ToClockOp>();
  target.addIllegalDialect<comb::CombDialect>();
  // LLVM intrinsics used by MooreToCore for bit manipulation functions
  // ($countones, $onehot, $onehot0)
  target.addIllegalOp<LLVM::CtPopOp>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  // Also add HW patterns because some 'comb' canonicalizers produce constant
  // operations, i.e., even if there is absolutely no HW operation present
  // initially, we might have to convert one.
  populateHWToSMTConversionPatterns(converter, patterns, false);
  populateCombToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
