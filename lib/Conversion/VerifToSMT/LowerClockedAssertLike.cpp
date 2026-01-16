//===- LowerClockedAssertLike.cpp - Lower clocked assertions to unclocked -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers clocked assertions (verif.clocked_assert, verif.clocked_assume,
// verif.clocked_cover) with i1 properties to their unclocked equivalents
// (verif.assert, verif.assume, verif.cover). This is useful for BMC where
// the clocked assertions with simple boolean properties need to be converted
// to unclocked form before VerifToSMT conversion.
//
// The transformation drops the clock and edge information since for BMC the
// circuit is already being evaluated at each time step.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

    rewriter.replaceOpWithNewOp<verif::AssertOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
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

    rewriter.replaceOpWithNewOp<verif::AssumeOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
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

    rewriter.replaceOpWithNewOp<verif::CoverOp>(
        op, adaptor.getProperty(), adaptor.getEnable(), op.getLabelAttr());
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
