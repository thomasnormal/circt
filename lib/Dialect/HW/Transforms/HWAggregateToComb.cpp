//===- HWAggregateToComb.cpp - HW aggregate to comb -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWAGGREGATETOCOMB
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

// Lower hw.array_create and hw.array_concat to comb.concat.
template <typename OpTy>
struct HWArrayCreateLikeOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInputs());
    return success();
  }
};

struct HWAggregateConstantOpConversion
    : OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  static LogicalResult peelAttribute(Location loc, Attribute attr,
                                     ConversionPatternRewriter &rewriter,
                                     APInt &intVal) {
    SmallVector<Attribute> worklist;
    worklist.push_back(attr);
    unsigned nextInsertion = intVal.getBitWidth();

    while (!worklist.empty()) {
      auto current = worklist.pop_back_val();
      if (auto innerArray = dyn_cast<ArrayAttr>(current)) {
        for (auto elem : llvm::reverse(innerArray))
          worklist.push_back(elem);
        continue;
      }

      if (auto intAttr = dyn_cast<IntegerAttr>(current)) {
        auto chunk = intAttr.getValue();
        nextInsertion -= chunk.getBitWidth();
        intVal.insertBits(chunk, nextInsertion);
        continue;
      }

      return failure();
    }

    return success();
  }

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower to concat.
    SmallVector<Value> results;
    auto bitWidth = hw::getBitWidth(op.getType());
    assert(bitWidth >= 0 && "bit width must be known for constant");
    APInt intVal(bitWidth, 0);
    if (failed(peelAttribute(op.getLoc(), adaptor.getFieldsAttr(), rewriter,
                             intVal)))
      return failure();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, intVal);
    return success();
  }
};

struct HWArrayGetOpConversion : OpConversionPattern<hw::ArrayGetOp> {
  using OpConversionPattern<hw::ArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> results;
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    auto lowered = adaptor.getInput();
    for (size_t i = 0; i < numElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, elemWidth));

    SmallVector<Value> bits;
    comb::extractBits(rewriter, op.getIndex(), bits);
    auto result = comb::constructMuxTree(rewriter, op.getLoc(), bits, results,
                                         results.back());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWArrayInjectOpConversion : OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    Location loc = op.getLoc();

    // Extract all elements from the input array
    SmallVector<Value> originalElements;
    auto inputArray = adaptor.getInput();
    for (size_t i = 0; i < numElements; ++i) {
      originalElements.push_back(rewriter.createOrFold<comb::ExtractOp>(
          loc, inputArray, i * elemWidth, elemWidth));
    }

    // Create 2D array: each row represents what the array would look like
    // if injection happened at that specific index
    SmallVector<Value> arrayRows;
    arrayRows.reserve(numElements);
    for (int injectIdx = numElements - 1; injectIdx >= 0; --injectIdx) {
      SmallVector<Value> rowElements;
      rowElements.reserve(numElements);

      // Build the row: array[n-1], array[n-2], ..., but replace element at
      // injectIdx with newVal
      for (int originalIdx = numElements - 1; originalIdx >= 0; --originalIdx) {
        if (originalIdx == injectIdx) {
          rowElements.push_back(adaptor.getElement());
        } else {
          rowElements.push_back(originalElements[originalIdx]);
        }
      }

      // Concatenate elements to form this row
      Value row = hw::ArrayCreateOp::create(rewriter, loc, rowElements);
      arrayRows.push_back(row);
    }

    // Create the 2D array by concatenating all rows
    // arrayRows[0] corresponds to injection at index 0
    // arrayRows[1] corresponds to injection at index 1, etc.
    Value array2D = hw::ArrayCreateOp::create(rewriter, loc, arrayRows);

    // Create array_get operation to select the row
    auto arrayGetOp =
        hw::ArrayGetOp::create(rewriter, loc, array2D, adaptor.getIndex());

    rewriter.replaceOp(op, arrayGetOp);
    return success();
  }
};

static unsigned getStructFieldOffset(hw::StructType type, unsigned fieldIdx) {
  unsigned offset = 0;
  auto fields = type.getElements();
  for (unsigned i = fieldIdx + 1, e = fields.size(); i < e; ++i) {
    auto width = hw::getBitWidth(fields[i].type);
    assert(width >= 0 && "bit width must be known for struct field");
    offset += width;
  }
  return offset;
}

struct HWStructCreateOpConversion
    : OpConversionPattern<hw::StructCreateOp> {
  using OpConversionPattern<hw::StructCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> fields(adaptor.getInput());
    if (fields.empty()) {
      auto bitWidth = hw::getBitWidth(op.getType());
      if (bitWidth < 0)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "unknown struct width");
      auto resultTy = rewriter.getIntegerType(bitWidth);
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, APInt(resultTy.getWidth(), 0));
      return success();
    }
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, fields);
    return success();
  }
};

struct HWStructExtractOpConversion
    : OpConversionPattern<hw::StructExtractOp> {
  using OpConversionPattern<hw::StructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = cast<hw::StructType>(op.getInput().getType());
    unsigned fieldIdx = op.getFieldIndex();
    auto fieldTy = structTy.getElements()[fieldIdx].type;
    auto width = hw::getBitWidth(fieldTy);
    if (width < 0)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unknown struct field width");
    auto offset = getStructFieldOffset(structTy, fieldIdx);
    auto resultTy = rewriter.getIntegerType(width);
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, APInt(resultTy.getWidth(), 0));
      return success();
    }
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultTy,
                                                 adaptor.getInput(), offset);
    return success();
  }
};

struct HWStructExplodeOpConversion
    : OpConversionPattern<hw::StructExplodeOp> {
  using OpConversionPattern<hw::StructExplodeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructExplodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = cast<hw::StructType>(op.getInput().getType());
    SmallVector<Value> results;
    results.reserve(structTy.getElements().size());
    for (unsigned i = 0, e = structTy.getElements().size(); i < e; ++i) {
      auto fieldTy = structTy.getElements()[i].type;
      auto width = hw::getBitWidth(fieldTy);
      if (width < 0)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "unknown struct field width");
      auto offset = getStructFieldOffset(structTy, i);
      auto resultTy = rewriter.getIntegerType(width);
      if (width == 0) {
        results.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(resultTy.getWidth(), 0)));
        continue;
      }
      results.push_back(comb::ExtractOp::create(
          rewriter, op.getLoc(), resultTy, adaptor.getInput(), offset));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MuxOpConversion : OpConversionPattern<comb::MuxOp> {
  using OpConversionPattern<comb::MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Re-create Mux with legalized types.
    rewriter.replaceOpWithNewOp<comb::MuxOp>(
        op, adaptor.getCond(), adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// A type converter is needed to perform the in-flight materialization of
/// aggregate types to integer types.
class AggregateTypeConverter : public TypeConverter {
public:
  AggregateTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](hw::ArrayType t) -> Type {
      return IntegerType::get(t.getContext(), hw::getBitWidth(t));
    });
    addConversion([](hw::StructType t) -> Type {
      return IntegerType::get(t.getContext(), hw::getBitWidth(t));
    });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });

    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });
  }
};
} // namespace

static void populateHWAggregateToCombOpConversionPatterns(
    RewritePatternSet &patterns, AggregateTypeConverter &typeConverter) {
  patterns.add<HWArrayGetOpConversion,
               HWArrayCreateLikeOpConversion<hw::ArrayCreateOp>,
               HWArrayCreateLikeOpConversion<hw::ArrayConcatOp>,
               HWAggregateConstantOpConversion, HWArrayInjectOpConversion,
               HWStructCreateOpConversion, HWStructExtractOpConversion,
               HWStructExplodeOpConversion, MuxOpConversion>(
      typeConverter, patterns.getContext());
}

namespace {
struct HWAggregateToCombPass
    : public hw::impl::HWAggregateToCombBase<HWAggregateToCombPass> {
  void runOnOperation() override;
  using HWAggregateToCombBase<HWAggregateToCombPass>::HWAggregateToCombBase;
};
} // namespace

void HWAggregateToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  // TODO: Add ArraySliceOp as well.
  target.addIllegalOp<hw::ArrayGetOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
                      hw::AggregateConstantOp, hw::ArrayInjectOp,
                      hw::StructCreateOp, hw::StructExtractOp,
                      hw::StructExplodeOp>();
  target.addDynamicallyLegalOp<comb::MuxOp>(
      [](comb::MuxOp op) { return hw::type_isa<IntegerType>(op.getType()); });
  target.addLegalDialect<hw::HWDialect, comb::CombDialect>();

  RewritePatternSet patterns(&getContext());
  AggregateTypeConverter typeConverter;
  populateHWAggregateToCombOpConversionPatterns(patterns, typeConverter);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
