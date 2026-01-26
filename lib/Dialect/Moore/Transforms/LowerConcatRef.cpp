//===- LowerConcatRef.cpp - moore.concat_ref lowering ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerConcatRef pass.
// It's used to disassemble the moore.concat_ref. Which is tricky to lower
// directly. For example, disassemble "{a, b} = c" onto "a = c[7:3]"
// and "b = c[2:0]".
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace circt {
namespace moore {
#define GEN_PASS_DEF_LOWERCONCATREF
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {

// A helper function for collecting the non-concatRef operands of concatRef.
static void collectOperands(Value operand, SmallVectorImpl<Value> &operands) {
  if (auto concatRefOp = operand.getDefiningOp<ConcatRefOp>())
    for (auto nestedOperand : concatRefOp.getValues())
      collectOperands(nestedOperand, operands);
  else
    operands.push_back(operand);
}

static std::optional<RefType> getSliceRefType(Type nestedType, int64_t width) {
  auto intType = dyn_cast<IntType>(nestedType);
  if (!intType)
    return std::nullopt;
  auto *ctx = nestedType.getContext();
  IntType sliceType = intType.getDomain() == Domain::TwoValued
                          ? IntType::getInt(ctx, width)
                          : IntType::getLogic(ctx, width);
  return RefType::get(sliceType);
}

template <typename OpTy>
struct ConcatRefLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use to collect the operands of concatRef.
    SmallVector<Value, 4> operands;
    collectOperands(op.getDst(), operands);
    auto srcWidth =
        cast<UnpackedType>(op.getSrc().getType()).getBitSize().value();

    // Disassemble assignments with the LHS is concatRef. And create new
    // corresponding assignments using non-concatRef LHS.
    for (auto operand : operands) {
      auto type = cast<RefType>(operand.getType()).getNestedType();
      auto width = type.getBitSize().value();

      rewriter.setInsertionPoint(op);
      // FIXME: Need to estimate whether the bits range is from large to
      // small or vice versa. Like "logic [7:0] or [0:7]".

      // Only able to correctly handle the situation like "[7:0]" now.
      auto extract = ExtractOp::create(rewriter, op.getLoc(), type, op.getSrc(),
                                       srcWidth - width);

      // Update the real bit width of RHS of assignment. Like "c" the above
      // description mentioned.
      srcWidth = srcWidth - width;

      OpTy::create(rewriter, op.getLoc(), operand, extract);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExtractRefFromConcatRefLowering
    : public OpConversionPattern<ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatRef = op.getInput().getDefiningOp<ConcatRefOp>();
    if (!concatRef)
      return failure();

    SmallVector<Value, 4> operands;
    collectOperands(concatRef, operands);

    SmallVector<int64_t, 4> widths;
    widths.reserve(operands.size());
    for (auto operand : operands) {
      auto refType = cast<RefType>(operand.getType());
      auto nestedType = cast<UnpackedType>(refType.getNestedType());
      auto bitSize = nestedType.getBitSize();
      if (!bitSize)
        return failure();
      widths.push_back(*bitSize);
    }

    auto resultRefType = cast<RefType>(op.getResult().getType());
    auto resultSize = resultRefType.getBitSize();
    if (!resultSize)
      return failure();

    int64_t extractLow = op.getLowBit();
    int64_t extractHigh = extractLow + *resultSize;

    SmallVector<int64_t, 4> operandLow(operands.size());
    int64_t offset = 0;
    for (int i = static_cast<int>(operands.size()) - 1; i >= 0; --i) {
      operandLow[i] = offset;
      offset += widths[i];
    }

    SmallVector<Value, 4> slices;
    for (size_t i = 0; i < operands.size(); ++i) {
      int64_t opLow = operandLow[i];
      int64_t opHigh = opLow + widths[i];
      int64_t overlapLow = std::max(opLow, extractLow);
      int64_t overlapHigh = std::min(opHigh, extractHigh);
      if (overlapLow >= overlapHigh)
        continue;

      int64_t sliceLow = overlapLow - opLow;
      int64_t sliceWidth = overlapHigh - overlapLow;
      Value sliceRef = operands[i];

      if (!(sliceLow == 0 && sliceWidth == widths[i] &&
            operands[i].getType() == resultRefType)) {
        auto nestedType =
            cast<RefType>(operands[i].getType()).getNestedType();
        auto sliceRefType = getSliceRefType(nestedType, sliceWidth);
        if (!sliceRefType)
          return failure();
        sliceRef = ExtractRefOp::create(
            rewriter, op.getLoc(), *sliceRefType, operands[i],
            rewriter.getI32IntegerAttr(sliceLow));
      }

      slices.push_back(sliceRef);
    }

    if (slices.empty())
      return failure();

    if (slices.size() == 1) {
      rewriter.replaceOp(op, slices.front());
      return success();
    }

    auto concatSlice = ConcatRefOp::create(rewriter, op.getLoc(), slices);
    rewriter.replaceOp(op, concatSlice.getResult());
    return success();
  }
};

struct ReadConcatRefLowering : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatRef = op.getInput().getDefiningOp<ConcatRefOp>();
    if (!concatRef)
      return failure();

    SmallVector<Value, 4> operands;
    collectOperands(concatRef, operands);

    SmallVector<Value, 4> values;
    values.reserve(operands.size());
    for (auto operand : operands)
      values.push_back(ReadOp::create(rewriter, op.getLoc(), operand));

    if (values.empty())
      return failure();

    if (values.size() == 1) {
      rewriter.replaceOp(op, values.front());
      return success();
    }

    auto concat = ConcatOp::create(rewriter, op.getLoc(), values);
    rewriter.replaceOp(op, concat.getResult());
    return success();
  }
};

struct LowerConcatRefPass
    : public circt::moore::impl::LowerConcatRefBase<LowerConcatRefPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createLowerConcatRefPass() {
  return std::make_unique<LowerConcatRefPass>();
}

void LowerConcatRefPass::runOnOperation() {
  MLIRContext &context = getContext();
  ConversionTarget target(context);

  target.addDynamicallyLegalOp<ContinuousAssignOp, BlockingAssignOp,
                               NonBlockingAssignOp>([](auto op) {
    return !op->getOperand(0).template getDefiningOp<ConcatRefOp>();
  });
  target.addDynamicallyLegalOp<ExtractRefOp>([](ExtractRefOp op) {
    return !op.getInput().getDefiningOp<ConcatRefOp>();
  });
  target.addDynamicallyLegalOp<ReadOp>([](ReadOp op) {
    return !op.getInput().getDefiningOp<ConcatRefOp>();
  });

  target.addLegalDialect<MooreDialect>();
  RewritePatternSet patterns(&context);
  patterns.add<ConcatRefLowering<ContinuousAssignOp>,
               ConcatRefLowering<BlockingAssignOp>,
               ConcatRefLowering<NonBlockingAssignOp>,
               ExtractRefFromConcatRefLowering, ReadConcatRefLowering>(
      &context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
