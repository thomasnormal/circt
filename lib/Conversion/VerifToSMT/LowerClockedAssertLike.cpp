//===- LowerClockedAssertLike.cpp - Lower clocked assertions to unclocked -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers clocked assertions (verif.clocked_assert,
// verif.clocked_assume, verif.clocked_cover) with i1 properties to their
// unclocked equivalents (verif.assert, verif.assume, verif.cover) by wrapping
// the property in an ltl.clock so LTLToCore can preserve edge sampling.
// For edge-triggered checks, we produce a clocked sequence and rely on the
// LTL lowering to model the tick semantics.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

namespace circt {
#define GEN_PASS_DEF_LOWERCLOCKEDASSERTLIKE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

static Value buildClockedProperty(OpBuilder &builder, Location loc, Value prop,
                                  verif::ClockEdge edge, Value clock) {
  auto makeClocked = [&](ltl::ClockEdge ltlEdge) {
    return ltl::ClockOp::create(builder, loc, prop, ltlEdge, clock).getResult();
  };
  switch (edge) {
  case verif::ClockEdge::Pos:
    return makeClocked(ltl::ClockEdge::Pos);
  case verif::ClockEdge::Neg:
    return makeClocked(ltl::ClockEdge::Neg);
  case verif::ClockEdge::Both: {
    Value pos = makeClocked(ltl::ClockEdge::Pos);
    Value neg = makeClocked(ltl::ClockEdge::Neg);
    return ltl::OrOp::create(builder, loc, ValueRange{pos, neg}).getResult();
  }
  }
  llvm_unreachable("unknown clock edge");
}

/// Convert verif.clocked_assert with i1 property to verif.assert
struct ClockedAssertOpConversion
    : OpConversionPattern<verif::ClockedAssertOp> {
  using OpConversionPattern<verif::ClockedAssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedAssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle i1 properties - LTL properties should be handled by LTLToCore
    if (isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      return failure();

    Value clockedProp = buildClockedProperty(
        rewriter, op.getLoc(), adaptor.getProperty(), op.getEdge(),
        adaptor.getClock());
    rewriter.replaceOpWithNewOp<verif::AssertOp>(op, clockedProp,
                                                 adaptor.getEnable(),
                                                 op.getLabelAttr());
    return success();
  }
};

/// Convert verif.clocked_assume with i1 property to verif.assume
struct ClockedAssumeOpConversion
    : OpConversionPattern<verif::ClockedAssumeOp> {
  using OpConversionPattern<verif::ClockedAssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedAssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle i1 properties - LTL properties should be handled by LTLToCore
    if (isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      return failure();

    Value clockedProp = buildClockedProperty(
        rewriter, op.getLoc(), adaptor.getProperty(), op.getEdge(),
        adaptor.getClock());
    rewriter.replaceOpWithNewOp<verif::AssumeOp>(op, clockedProp,
                                                 adaptor.getEnable(),
                                                 op.getLabelAttr());
    return success();
  }
};

/// Convert verif.clocked_cover with i1 property to verif.cover
struct ClockedCoverOpConversion
    : OpConversionPattern<verif::ClockedCoverOp> {
  using OpConversionPattern<verif::ClockedCoverOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedCoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle i1 properties - LTL properties should be handled by LTLToCore
    if (isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      return failure();

    Value clockedProp = buildClockedProperty(
        rewriter, op.getLoc(), adaptor.getProperty(), op.getEdge(),
        adaptor.getClock());
    rewriter.replaceOpWithNewOp<verif::CoverOp>(op, clockedProp,
                                                adaptor.getEnable(),
                                                op.getLabelAttr());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower Clocked Assert Like pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerClockedAssertLikePass
    : public circt::impl::LowerClockedAssertLikeBase<
          LowerClockedAssertLikePass> {
  void runOnOperation() override;
};
} // namespace

void LowerClockedAssertLikePass::runOnOperation() {
  ConversionTarget target(getContext());

  // Keep the verif dialect legal, but make clocked ops with i1 properties
  // illegal
  target.addLegalDialect<verif::VerifDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<ltl::LTLDialect>();

  // Mark clocked assertions with i1 properties as illegal
  target.addDynamicallyLegalOp<verif::ClockedAssertOp>(
      [](verif::ClockedAssertOp op) {
        return isa<ltl::PropertyType, ltl::SequenceType>(
            op.getProperty().getType());
      });
  target.addDynamicallyLegalOp<verif::ClockedAssumeOp>(
      [](verif::ClockedAssumeOp op) {
        return isa<ltl::PropertyType, ltl::SequenceType>(
            op.getProperty().getType());
      });
  target.addDynamicallyLegalOp<verif::ClockedCoverOp>(
      [](verif::ClockedCoverOp op) {
        return isa<ltl::PropertyType, ltl::SequenceType>(
            op.getProperty().getType());
      });

  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });

  RewritePatternSet patterns(&getContext());
  patterns.add<ClockedAssertOpConversion, ClockedAssumeOpConversion,
               ClockedCoverOpConversion>(converter, patterns.getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createLowerClockedAssertLikePass() {
  return std::make_unique<LowerClockedAssertLikePass>();
}
