//===- SVAToLTL.cpp - SVA to LTL Conversion -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SVA dialect operations to LTL and Verif dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SVAToLTL.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SVA/SVADialect.h"
#include "circt/Dialect/SVA/SVAOps.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_LOWERSVATOLTL
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace sva;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

namespace {

class SVATypeConverter : public TypeConverter {
public:
  SVATypeConverter() {
    addConversion([](Type type) { return type; });

    // Convert SVA sequence type to LTL sequence type
    addConversion([](SequenceType type) -> Type {
      return ltl::SequenceType::get(type.getContext());
    });

    // Convert SVA property type to LTL property type
    addConversion([](PropertyType type) -> Type {
      return ltl::PropertyType::get(type.getContext());
    });
  }
};

//===----------------------------------------------------------------------===//
// Sequence Operation Conversions
//===----------------------------------------------------------------------===//

struct SequenceDelayOpConversion
    : public OpConversionPattern<SequenceDelayOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceDelayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::DelayOp>(
        op, ltlSeqType, adaptor.getInput(), op.getDelayAttr(),
        op.getLengthAttr());
    return success();
  }
};

struct SequenceRepeatOpConversion
    : public OpConversionPattern<SequenceRepeatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::RepeatOp>(
        op, ltlSeqType, adaptor.getInput(), op.getBaseAttr(), op.getMoreAttr());
    return success();
  }
};

struct SequenceGotoRepeatOpConversion
    : public OpConversionPattern<SequenceGotoRepeatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceGotoRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::GoToRepeatOp>(
        op, ltlSeqType, adaptor.getInput(), op.getBaseAttr(), op.getMoreAttr());
    return success();
  }
};

struct SequenceNonConsecutiveRepeatOpConversion
    : public OpConversionPattern<SequenceNonConsecutiveRepeatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceNonConsecutiveRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::NonConsecutiveRepeatOp>(
        op, ltlSeqType, adaptor.getInput(), op.getBaseAttr(), op.getMoreAttr());
    return success();
  }
};

struct SequenceFirstMatchOpConversion
    : public OpConversionPattern<SequenceFirstMatchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceFirstMatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::FirstMatchOp>(op, ltlSeqType,
                                                  adaptor.getInput());
    return success();
  }
};

struct SequenceConcatOpConversion
    : public OpConversionPattern<SequenceConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::ConcatOp>(op, ltlSeqType,
                                                adaptor.getInputs());
    return success();
  }
};

struct SequenceOrOpConversion : public OpConversionPattern<SequenceOrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::OrOp>(op, ltlSeqType, adaptor.getInputs());
    return success();
  }
};

struct SequenceAndOpConversion : public OpConversionPattern<SequenceAndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::AndOp>(op, ltlSeqType,
                                             adaptor.getInputs());
    return success();
  }
};

struct SequenceIntersectOpConversion
    : public OpConversionPattern<SequenceIntersectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceIntersectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::IntersectOp>(op, ltlSeqType,
                                                   adaptor.getInputs());
    return success();
  }
};

struct SequenceClockOpConversion
    : public OpConversionPattern<SequenceClockOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SequenceClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlSeqType = ltl::SequenceType::get(op.getContext());
    // Convert SVA clock edge to LTL clock edge
    ltl::ClockEdge ltlEdge = ltl::ClockEdge::Pos;
    switch (op.getEdge()) {
    case ClockEdge::Pos:
      ltlEdge = ltl::ClockEdge::Pos;
      break;
    case ClockEdge::Neg:
      ltlEdge = ltl::ClockEdge::Neg;
      break;
    case ClockEdge::Both:
      ltlEdge = ltl::ClockEdge::Both;
      break;
    }
    rewriter.replaceOpWithNewOp<ltl::ClockOp>(op, ltlSeqType, adaptor.getInput(),
                                               ltlEdge, adaptor.getClock());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Property Operation Conversions
//===----------------------------------------------------------------------===//

struct PropertyNotOpConversion : public OpConversionPattern<PropertyNotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::NotOp>(op, ltlPropType,
                                             adaptor.getInput());
    return success();
  }
};

struct PropertyAndOpConversion : public OpConversionPattern<PropertyAndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::AndOp>(op, ltlPropType,
                                             adaptor.getInputs());
    return success();
  }
};

struct PropertyOrOpConversion : public OpConversionPattern<PropertyOrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::OrOp>(op, ltlPropType,
                                            adaptor.getInputs());
    return success();
  }
};

struct PropertyImplicationOpConversion
    : public OpConversionPattern<PropertyImplicationOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());

    Value antecedent = adaptor.getAntecedent();
    Value consequent = adaptor.getConsequent();

    // For non-overlapping implication (|=>), we need to add a delay of 1 cycle
    // to the consequent
    if (!op.getOverlapping()) {
      auto ltlSeqType = ltl::SequenceType::get(op.getContext());
      consequent = ltl::DelayOp::create(
          rewriter, op.getLoc(), ltlSeqType, consequent,
          rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(0));
    }

    rewriter.replaceOpWithNewOp<ltl::ImplicationOp>(op, ltlPropType, antecedent,
                                                     consequent);
    return success();
  }
};

struct PropertyEventuallyOpConversion
    : public OpConversionPattern<PropertyEventuallyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyEventuallyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    rewriter.replaceOpWithNewOp<ltl::EventuallyOp>(op, ltlPropType,
                                                    adaptor.getInput());
    return success();
  }
};

struct PropertyUntilOpConversion : public OpConversionPattern<PropertyUntilOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyUntilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    // Note: LTL's until is weak; for strong until we would need additional
    // handling
    rewriter.replaceOpWithNewOp<ltl::UntilOp>(op, ltlPropType,
                                               adaptor.getInput(),
                                               adaptor.getCondition());
    return success();
  }
};

struct PropertyClockOpConversion
    : public OpConversionPattern<PropertyClockOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PropertyClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ltlPropType = ltl::PropertyType::get(op.getContext());
    // Convert SVA clock edge to LTL clock edge
    ltl::ClockEdge ltlEdge = ltl::ClockEdge::Pos;
    switch (op.getEdge()) {
    case ClockEdge::Pos:
      ltlEdge = ltl::ClockEdge::Pos;
      break;
    case ClockEdge::Neg:
      ltlEdge = ltl::ClockEdge::Neg;
      break;
    case ClockEdge::Both:
      ltlEdge = ltl::ClockEdge::Both;
      break;
    }
    rewriter.replaceOpWithNewOp<ltl::ClockOp>(op, ltlPropType,
                                               adaptor.getInput(), ltlEdge,
                                               adaptor.getClock());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Assertion Directive Conversions
//===----------------------------------------------------------------------===//

struct AssertPropertyOpConversion
    : public OpConversionPattern<AssertPropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssertPropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<verif::AssertOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

struct AssumePropertyOpConversion
    : public OpConversionPattern<AssumePropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssumePropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<verif::AssumeOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

struct CoverPropertyOpConversion
    : public OpConversionPattern<CoverPropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoverPropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<verif::CoverOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

struct ClockedAssertPropertyOpConversion
    : public OpConversionPattern<ClockedAssertPropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockedAssertPropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert SVA clock edge to Verif clock edge
    verif::ClockEdge verifEdge = verif::ClockEdge::Pos;
    switch (op.getEdge()) {
    case ClockEdge::Pos:
      verifEdge = verif::ClockEdge::Pos;
      break;
    case ClockEdge::Neg:
      verifEdge = verif::ClockEdge::Neg;
      break;
    case ClockEdge::Both:
      verifEdge = verif::ClockEdge::Both;
      break;
    }
    rewriter.replaceOpWithNewOp<verif::ClockedAssertOp>(
        op, adaptor.getProperty(), verifEdge, adaptor.getClock(),
        adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

struct ClockedAssumePropertyOpConversion
    : public OpConversionPattern<ClockedAssumePropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockedAssumePropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    verif::ClockEdge verifEdge = verif::ClockEdge::Pos;
    switch (op.getEdge()) {
    case ClockEdge::Pos:
      verifEdge = verif::ClockEdge::Pos;
      break;
    case ClockEdge::Neg:
      verifEdge = verif::ClockEdge::Neg;
      break;
    case ClockEdge::Both:
      verifEdge = verif::ClockEdge::Both;
      break;
    }
    rewriter.replaceOpWithNewOp<verif::ClockedAssumeOp>(
        op, adaptor.getProperty(), verifEdge, adaptor.getClock(),
        adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

struct ClockedCoverPropertyOpConversion
    : public OpConversionPattern<ClockedCoverPropertyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClockedCoverPropertyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    verif::ClockEdge verifEdge = verif::ClockEdge::Pos;
    switch (op.getEdge()) {
    case ClockEdge::Pos:
      verifEdge = verif::ClockEdge::Pos;
      break;
    case ClockEdge::Neg:
      verifEdge = verif::ClockEdge::Neg;
      break;
    case ClockEdge::Both:
      verifEdge = verif::ClockEdge::Both;
      break;
    }
    rewriter.replaceOpWithNewOp<verif::ClockedCoverOp>(
        op, adaptor.getProperty(), verifEdge, adaptor.getClock(),
        adaptor.getEnable(), op.getLabelAttr());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower SVA to LTL Pass
//===----------------------------------------------------------------------===//

namespace {

struct LowerSVAToLTLPass
    : public circt::impl::LowerSVAToLTLBase<LowerSVAToLTLPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSVAToLTLPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addIllegalDialect<SVADialect>();

  // Mark types as legal
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [](UnrealizedConversionCastOp op) {
        // Allow casts between SVA and LTL types during conversion
        return true;
      });

  SVATypeConverter typeConverter;
  RewritePatternSet patterns(&getContext());

  // Add sequence conversions
  patterns.add<SequenceDelayOpConversion, SequenceRepeatOpConversion,
               SequenceGotoRepeatOpConversion,
               SequenceNonConsecutiveRepeatOpConversion,
               SequenceFirstMatchOpConversion,
               SequenceConcatOpConversion, SequenceOrOpConversion,
               SequenceAndOpConversion, SequenceIntersectOpConversion,
               SequenceClockOpConversion>(typeConverter, &getContext());

  // Add property conversions
  patterns.add<PropertyNotOpConversion, PropertyAndOpConversion,
               PropertyOrOpConversion, PropertyImplicationOpConversion,
               PropertyEventuallyOpConversion, PropertyUntilOpConversion,
               PropertyClockOpConversion>(typeConverter, &getContext());

  // Add assertion directive conversions
  patterns.add<AssertPropertyOpConversion, AssumePropertyOpConversion,
               CoverPropertyOpConversion, ClockedAssertPropertyOpConversion,
               ClockedAssumePropertyOpConversion,
               ClockedCoverPropertyOpConversion>(typeConverter, &getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createLowerSVAToLTLPass() {
  return std::make_unique<LowerSVAToLTLPass>();
}
