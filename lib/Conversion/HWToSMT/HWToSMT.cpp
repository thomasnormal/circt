//===- HWToSMT.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTHWTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
static unsigned getArrayDomainWidth(size_t numElements) {
  // SMT bit-vectors must have width >= 1. For 1-element arrays use a single
  // bit index and rely on explicit in-bounds checks.
  return llvm::Log2_64_Ceil(numElements > 1 ? numElements : 2);
}

static FailureOr<Value> packValueToBitVector(Type hwType, Value value,
                                             PatternRewriter &rewriter,
                                             Location loc) {
  if (auto alias = dyn_cast<hw::TypeAliasType>(hwType))
    return packValueToBitVector(alias.getCanonicalType(), value, rewriter, loc);

  if (auto valueBvTy = dyn_cast<mlir::smt::BitVectorType>(value.getType())) {
    int64_t expectedWidth = hw::getBitWidth(hwType);
    if (expectedWidth < 0)
      return failure();
    if (valueBvTy.getWidth() == static_cast<unsigned>(expectedWidth))
      return value;
    return failure();
  }

  if (auto arrayType = dyn_cast<hw::ArrayType>(hwType)) {
    auto arraySMT = dyn_cast<mlir::smt::ArrayType>(value.getType());
    if (!arraySMT)
      return failure();
    unsigned numElements = arrayType.getNumElements();
    unsigned indexWidth = getArrayDomainWidth(numElements);
    Value flattened;
    for (int64_t i = static_cast<int64_t>(numElements) - 1; i >= 0; --i) {
      Value index =
          mlir::smt::BVConstantOp::create(rewriter, loc, i, indexWidth);
      Value element =
          mlir::smt::ArraySelectOp::create(rewriter, loc, value, index);
      auto packedElement =
          packValueToBitVector(arrayType.getElementType(), element, rewriter, loc);
      if (failed(packedElement))
        return failure();
      flattened = flattened
                      ? mlir::smt::ConcatOp::create(rewriter, loc, flattened,
                                                    *packedElement)
                      : *packedElement;
    }
    return flattened;
  }

  return failure();
}

static FailureOr<Value> unpackBitVectorToValue(Type hwType, Value bitVector,
                                               const TypeConverter &converter,
                                               PatternRewriter &rewriter,
                                               Location loc) {
  if (auto alias = dyn_cast<hw::TypeAliasType>(hwType))
    return unpackBitVectorToValue(alias.getCanonicalType(), bitVector, converter,
                                  rewriter, loc);

  auto bitVectorTy = dyn_cast<mlir::smt::BitVectorType>(bitVector.getType());
  if (!bitVectorTy)
    return failure();
  int64_t expectedWidth = hw::getBitWidth(hwType);
  if (expectedWidth < 0)
    return failure();
  if (bitVectorTy.getWidth() != static_cast<unsigned>(expectedWidth))
    return failure();

  Type convertedTy = converter.convertType(hwType);
  if (!convertedTy)
    return failure();
  if (isa<mlir::smt::BitVectorType>(convertedTy))
    return bitVector;

  if (auto arrayType = dyn_cast<hw::ArrayType>(hwType)) {
    auto arraySMT = dyn_cast<mlir::smt::ArrayType>(convertedTy);
    if (!arraySMT)
      return failure();

    unsigned numElements = arrayType.getNumElements();
    unsigned indexWidth = getArrayDomainWidth(numElements);
    int64_t elementWidth = hw::getBitWidth(arrayType.getElementType());
    if (elementWidth <= 0)
      return failure();
    auto elementBvTy = mlir::smt::BitVectorType::get(bitVector.getContext(),
                                                     elementWidth);

    Value array = mlir::smt::DeclareFunOp::create(rewriter, loc, arraySMT);
    for (unsigned i = 0; i < numElements; ++i) {
      Value elementBits = mlir::smt::ExtractOp::create(
          rewriter, loc, elementBvTy, i * static_cast<unsigned>(elementWidth),
          bitVector);
      auto element = unpackBitVectorToValue(arrayType.getElementType(),
                                            elementBits, converter, rewriter, loc);
      if (failed(element))
        return failure();
      Value index =
          mlir::smt::BVConstantOp::create(rewriter, loc, i, indexWidth);
      array = mlir::smt::ArrayStoreOp::create(rewriter, loc, array, index,
                                              *element);
    }
    return array;
  }

  return failure();
}

static LogicalResult flattenAggregateConstantAttr(Attribute attr, Type type,
                                                  APInt &flattened,
                                                  unsigned &nextInsertion) {
  if (auto alias = dyn_cast<hw::TypeAliasType>(type))
    return flattenAggregateConstantAttr(attr, alias.getCanonicalType(),
                                        flattened, nextInsertion);

  if (auto structType = dyn_cast<hw::StructType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != structType.getElements().size())
      return failure();
    for (auto [fieldAttr, fieldInfo] :
         llvm::zip(arrayAttr.getValue(), structType.getElements()))
      if (failed(flattenAggregateConstantAttr(fieldAttr, fieldInfo.type,
                                              flattened, nextInsertion)))
        return failure();
    return success();
  }

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != arrayType.getNumElements())
      return failure();
    for (auto elementAttr : arrayAttr.getValue())
      if (failed(flattenAggregateConstantAttr(elementAttr,
                                              arrayType.getElementType(),
                                              flattened, nextInsertion)))
        return failure();
    return success();
  }

  if (auto arrayType = dyn_cast<hw::UnpackedArrayType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != arrayType.getNumElements())
      return failure();
    for (auto elementAttr : arrayAttr.getValue())
      if (failed(flattenAggregateConstantAttr(elementAttr,
                                              arrayType.getElementType(),
                                              flattened, nextInsertion)))
        return failure();
    return success();
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return failure();
    auto chunk = intAttr.getValue().zextOrTrunc(intType.getWidth());
    nextInsertion -= chunk.getBitWidth();
    flattened.insertBits(chunk, nextInsertion);
    return success();
  }

  if (auto enumType = dyn_cast<hw::EnumType>(type)) {
    auto enumAttr = dyn_cast<StringAttr>(attr);
    if (!enumAttr)
      return failure();
    auto idx = enumType.indexOf(enumAttr.getValue());
    auto width = enumType.getBitWidth();
    if (!idx || !width)
      return failure();
    APInt chunk(*width, *idx);
    nextInsertion -= chunk.getBitWidth();
    flattened.insertBits(chunk, nextInsertion);
    return success();
  }

  if (auto clockConst = dyn_cast<seq::ClockConstAttr>(attr)) {
    if (!isa<seq::ClockType>(type))
      return failure();
    APInt chunk(1, clockConst.getValue() == seq::ClockConst::High ? 1 : 0);
    nextInsertion -= 1;
    flattened.insertBits(chunk, nextInsertion);
    return success();
  }

  return failure();
}

static Value materializeAggregateConstant(Attribute attr, Type sourceType,
                                          Type targetType,
                                          ConversionPatternRewriter &rewriter,
                                          Location loc) {
  if (auto targetBV = dyn_cast<mlir::smt::BitVectorType>(targetType)) {
    APInt flattened(targetBV.getWidth(), 0);
    unsigned nextInsertion = targetBV.getWidth();
    if (failed(flattenAggregateConstantAttr(attr, sourceType, flattened,
                                            nextInsertion)))
      return Value();
    if (nextInsertion != 0)
      return Value();
    return mlir::smt::BVConstantOp::create(rewriter, loc, flattened);
  }

  if (auto targetArray = dyn_cast<mlir::smt::ArrayType>(targetType)) {
    auto sourceArray = dyn_cast<hw::ArrayType>(sourceType);
    if (!sourceArray)
      return Value();
    auto elementsAttr = dyn_cast<ArrayAttr>(attr);
    if (!elementsAttr || elementsAttr.size() != sourceArray.getNumElements())
      return Value();

    Value array = mlir::smt::DeclareFunOp::create(rewriter, loc, targetArray);
    unsigned numElements = sourceArray.getNumElements();
    unsigned indexWidth = getArrayDomainWidth(numElements);
    auto elementType = sourceArray.getElementType();
    auto targetElementType = targetArray.getRangeType();
    for (auto [idx, elementAttr] : llvm::enumerate(elementsAttr.getValue())) {
      Value element =
          materializeAggregateConstant(elementAttr, elementType,
                                       targetElementType, rewriter, loc);
      if (!element)
        return Value();
      Value index = mlir::smt::BVConstantOp::create(rewriter, loc,
                                                    numElements - idx - 1,
                                                    indexWidth);
      array = mlir::smt::ArrayStoreOp::create(rewriter, loc, array, index,
                                              element);
    }
    return array;
  }

  return Value();
}

/// Lower a hw::ConstantOp operation to smt::BVConstantOp
struct HWConstantOpConversion : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getValue().getBitWidth() < 1) {
      // Constants of type i0 cannot be represented in SMT. They can still
      // appear as singleton-array indices and become dead once the consumer is
      // lowered.
      if (op->use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "0-bit constants with uses not "
                                         "supported");
    }
    rewriter.replaceOpWithNewOp<mlir::smt::BVConstantOp>(op,
                                                         adaptor.getValue());
    return success();
  }
};

/// Lower a hw::AggregateConstantOp operation to SMT constants.
struct HWAggregateConstantOpConversion
    : OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type targetType = getTypeConverter()->convertType(op.getType());
    if (!targetType)
      return failure();
    Value lowered = materializeAggregateConstant(
        op.getFieldsAttr(), op.getType(), targetType, rewriter, op.getLoc());
    if (!lowered)
      return rewriter.notifyMatchFailure(
          op, "unsupported aggregate constant attribute/type combination");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

/// Lower a hw::HWModuleOp operation to func::FuncOp.
struct HWModuleOpConversion : OpConversionPattern<HWModuleOp> {
  using OpConversionPattern<HWModuleOp>::OpConversionPattern;

  HWModuleOpConversion(TypeConverter &converter, MLIRContext *context,
                       bool replaceWithSolver)
      : OpConversionPattern<HWModuleOp>::OpConversionPattern(converter,
                                                             context),
        replaceWithSolver(replaceWithSolver) {}

  LogicalResult
  matchAndRewrite(HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcTy = op.getModuleType().getFuncType();
    SmallVector<Type> inputTypes, resultTypes;
    if (failed(typeConverter->convertTypes(funcTy.getInputs(), inputTypes)))
      return failure();
    if (failed(typeConverter->convertTypes(funcTy.getResults(), resultTypes)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter)))
      return failure();
    auto loc = op.getLoc();
    if (replaceWithSolver) {
      // If we're exporting to SMTLIB we need to move the module into an
      // smt.solver op (pre-pattern checks make sure we only have one module).
      auto solverOp = mlir::smt::SolverOp::create(rewriter, loc, {}, {});
      auto *solverBlock = rewriter.createBlock(&solverOp.getBodyRegion());
      // Create a new symbolic value to replace each input
      rewriter.setInsertionPointToStart(solverBlock);
      SmallVector<Value> symVals;
      for (auto inputType : inputTypes) {
        auto symVal = mlir::smt::DeclareFunOp::create(rewriter, loc, inputType);
        symVals.push_back(symVal);
      }
      // Inline module body into solver op and replace args with new symbolic
      // values
      rewriter.inlineBlockBefore(op.getBodyBlock(), solverBlock,
                                 solverBlock->end(), symVals);
      rewriter.eraseOp(op);
      return success();
    }
    auto funcOp = mlir::func::FuncOp::create(
        rewriter, loc, adaptor.getSymNameAttr(),
        rewriter.getFunctionType(inputTypes, resultTypes));
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
    rewriter.eraseOp(op);
    return success();
  }

  bool replaceWithSolver;
};

/// Lower a hw::HWModuleExternOp declaration to an external func::FuncOp.
struct HWModuleExternOpConversion : OpConversionPattern<hw::HWModuleExternOp> {
  using OpConversionPattern<hw::HWModuleExternOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::HWModuleExternOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcTy = op.getModuleType().getFuncType();
    SmallVector<Type> inputTypes, resultTypes;
    if (failed(typeConverter->convertTypes(funcTy.getInputs(), inputTypes)))
      return failure();
    if (failed(typeConverter->convertTypes(funcTy.getResults(), resultTypes)))
      return failure();
    auto funcOp = mlir::func::FuncOp::create(
        rewriter, op.getLoc(), adaptor.getSymNameAttr(),
        rewriter.getFunctionType(inputTypes, resultTypes));
    funcOp.setVisibility(SymbolTable::Visibility::Private);
    rewriter.replaceOp(op, funcOp);
    return success();
  }
};

/// Lower a hw::OutputOp operation to func::ReturnOp.
struct OutputOpConversion : OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  OutputOpConversion(TypeConverter &converter, MLIRContext *context,
                     bool assertModuleOutputs)
      : OpConversionPattern<OutputOp>::OpConversionPattern(converter, context),
        assertModuleOutputs(assertModuleOutputs) {}

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (assertModuleOutputs) {
      Location loc = op.getLoc();
      for (auto output : adaptor.getOutputs()) {
        Value constOutput =
            mlir::smt::DeclareFunOp::create(rewriter, loc, output.getType());
        Value eq = mlir::smt::EqOp::create(rewriter, loc, output, constOutput);
        mlir::smt::AssertOp::create(rewriter, loc, eq);
      }
      rewriter.replaceOpWithNewOp<mlir::smt::YieldOp>(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }

  bool assertModuleOutputs;
};

/// Lower a hw::WireOp by dropping the naming edge. In SMT we do not model
/// external forces/observability of SSA edges, so the identity semantics of
/// hw.wire is sufficient.
struct WireOpConversion : OpConversionPattern<WireOp> {
  using OpConversionPattern<WireOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Lower a hw::BitcastOp when source and result share the same SMT
/// representation.
struct BitcastOpConversion : OpConversionPattern<BitcastOp> {
  using OpConversionPattern<BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedResultType = typeConverter->convertType(op.getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported bitcast result type");
    if (adaptor.getInput().getType() == convertedResultType) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Lower array<elem> -> iN bitcasts by flattening array elements into a BV.
    if (auto srcArrayTy = dyn_cast<hw::ArrayType>(op.getInput().getType())) {
      auto dstBvTy = dyn_cast<mlir::smt::BitVectorType>(convertedResultType);
      if (dstBvTy) {
        auto flattened = packValueToBitVector(
            srcArrayTy, adaptor.getInput(), rewriter, loc);
        if (failed(flattened))
          return rewriter.notifyMatchFailure(
              op.getLoc(), "array bitcast requires bitvector element sort");
        auto flattenedTy =
            cast<mlir::smt::BitVectorType>((*flattened).getType());
        if (flattenedTy.getWidth() != dstBvTy.getWidth())
          return rewriter.notifyMatchFailure(op.getLoc(),
                                             "array bitcast width mismatch");
        rewriter.replaceOp(op, *flattened);
        return success();
      }
    }

    // Lower iN -> array<elem> bitcasts by extracting element chunks.
    if (auto dstArrayTy = dyn_cast<hw::ArrayType>(op.getType())) {
      auto srcBvTy = dyn_cast<mlir::smt::BitVectorType>(adaptor.getInput().getType());
      auto dstArraySMT = dyn_cast<mlir::smt::ArrayType>(convertedResultType);
      if (srcBvTy && dstArraySMT) {
        auto elemBvTy =
            dyn_cast<mlir::smt::BitVectorType>(dstArraySMT.getRangeType());
        if (!elemBvTy)
          return rewriter.notifyMatchFailure(
              op.getLoc(), "array bitcast requires bitvector element sort");
        unsigned numElements = dstArrayTy.getNumElements();
        unsigned elemWidth = elemBvTy.getWidth();
        if (srcBvTy.getWidth() != numElements * elemWidth)
          return rewriter.notifyMatchFailure(
              op.getLoc(), "array bitcast width mismatch");
        unsigned indexWidth = getArrayDomainWidth(numElements);

        Value array = mlir::smt::DeclareFunOp::create(rewriter, loc, dstArraySMT);
        for (unsigned i = 0; i < numElements; ++i) {
          unsigned offset = i * elemWidth;
          Value element = mlir::smt::ExtractOp::create(
              rewriter, loc, elemBvTy, offset, adaptor.getInput());
          Value index = mlir::smt::BVConstantOp::create(rewriter, loc, i,
                                                        indexWidth);
          array = mlir::smt::ArrayStoreOp::create(rewriter, loc, array, index,
                                                  element);
        }
        rewriter.replaceOp(op, array);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(
        op.getLoc(), "bitcast source/result SMT representations differ");
  }
};

/// Lower a hw::InstanceOp operation to func::CallOp.
struct InstanceOpConversion : OpConversionPattern<InstanceOp> {
  using OpConversionPattern<InstanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    bool calleeIsExtern = false;
    if (Operation *symbol =
            SymbolTable::lookupNearestSymbolFrom(op, op.getModuleNameAttr())) {
      if (isa<hw::HWModuleExternOp>(symbol))
        calleeIsExtern = true;
      if (auto funcOp = dyn_cast<mlir::func::FuncOp>(symbol);
          funcOp && funcOp.isExternal())
        calleeIsExtern = true;
    }

    // For extern modules, model each result as a fresh uninterpreted function
    // application over the converted inputs. This avoids emitting external
    // func.call ops that SMT-LIB export cannot represent.
    if (calleeIsExtern) {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      SmallVector<Type> domainTypes(adaptor.getInputs().getTypes().begin(),
                                    adaptor.getInputs().getTypes().end());
      for (auto resultType : resultTypes) {
        // SMT allows zero-argument function symbols (`declare-const` form).
        // Model extern instances without inputs as fresh symbolic values
        // directly, avoiding `!smt.func<()>` construction.
        if (domainTypes.empty()) {
          Value sym = mlir::smt::DeclareFunOp::create(rewriter, op.getLoc(),
                                                      resultType);
          results.push_back(sym);
          continue;
        }
        auto funcType = mlir::smt::SMTFuncType::get(domainTypes, resultType);
        Value fn = mlir::smt::DeclareFunOp::create(rewriter, op.getLoc(), funcType);
        Value app = mlir::smt::ApplyFuncOp::create(rewriter, op.getLoc(), fn,
                                                   adaptor.getInputs());
        results.push_back(app);
      }
      rewriter.replaceOp(op, results);
      return success();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, adaptor.getModuleNameAttr(), resultTypes, adaptor.getInputs());
    return success();
  }
};

/// Lower a hw::ArrayCreateOp operation to smt::DeclareFun and an
/// smt::ArrayStoreOp for each operand.
struct ArrayCreateOpConversion : OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern<ArrayCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type arrTy = typeConverter->convertType(op.getType());
    if (!arrTy)
      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");

    unsigned width = adaptor.getInputs().size();
    unsigned indexWidth = getArrayDomainWidth(width);

    Value arr = mlir::smt::DeclareFunOp::create(rewriter, loc, arrTy);
    for (auto [i, el] : llvm::enumerate(adaptor.getInputs())) {
      Value idx = mlir::smt::BVConstantOp::create(rewriter, loc, width - i - 1,
                                                  indexWidth);
      arr = mlir::smt::ArrayStoreOp::create(rewriter, loc, arr, idx, el);
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};

/// Lower a hw::ArrayGetOp operation to smt::ArraySelectOp
struct ArrayGetOpConversion : OpConversionPattern<ArrayGetOp> {
  using OpConversionPattern<ArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numElements =
        cast<hw::ArrayType>(op.getInput().getType()).getNumElements();
    unsigned indexWidth = getArrayDomainWidth(numElements);

    Type type = typeConverter->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported array element type");

    if (numElements == 1) {
      // The only legal index value for a singleton array is zero. Keep the
      // semantics by selecting element 0 directly and avoid carrying i0 index
      // values into SMT lowering.
      Value zeroIndex = mlir::smt::BVConstantOp::create(rewriter, loc, 0, 1);
      rewriter.replaceOpWithNewOp<mlir::smt::ArraySelectOp>(
          op, adaptor.getInput(), zeroIndex);
      return success();
    }

    Value oobVal = mlir::smt::DeclareFunOp::create(rewriter, loc, type);
    Value numElementsVal = mlir::smt::BVConstantOp::create(
        rewriter, loc, numElements - 1, indexWidth);
    Value inBounds = mlir::smt::BVCmpOp::create(
        rewriter, loc, mlir::smt::BVCmpPredicate::ule, adaptor.getIndex(),
        numElementsVal);
    Value indexed = mlir::smt::ArraySelectOp::create(
        rewriter, loc, adaptor.getInput(), adaptor.getIndex());
    rewriter.replaceOpWithNewOp<mlir::smt::IteOp>(op, inBounds, indexed,
                                                  oobVal);
    return success();
  }
};

/// Lower a hw::ArrayInjectOp operation to smt::ArrayStoreOp.
struct ArrayInjectOpConversion : OpConversionPattern<ArrayInjectOp> {
  using OpConversionPattern<ArrayInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numElements =
        cast<hw::ArrayType>(op.getInput().getType()).getNumElements();
    unsigned indexWidth = getArrayDomainWidth(numElements);

    Type arrType = typeConverter->convertType(op.getType());
    if (!arrType)
      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");

    Value oobVal = mlir::smt::DeclareFunOp::create(rewriter, loc, arrType);
    // Check if the index is within bounds
    Value numElementsVal = mlir::smt::BVConstantOp::create(
        rewriter, loc, numElements - 1, indexWidth);
    Value inBounds = mlir::smt::BVCmpOp::create(
        rewriter, loc, mlir::smt::BVCmpPredicate::ule, adaptor.getIndex(),
        numElementsVal);

    // Store the element at the given index
    Value stored = mlir::smt::ArrayStoreOp::create(
        rewriter, loc, adaptor.getInput(), adaptor.getIndex(),
        adaptor.getElement());

    // Return unbounded array if out of bounds
    rewriter.replaceOpWithNewOp<mlir::smt::IteOp>(op, inBounds, stored, oobVal);
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

/// Lower a hw::StructCreateOp operation to smt::ConcatOp.
struct StructCreateOpConversion : OpConversionPattern<StructCreateOp> {
  using OpConversionPattern<StructCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getInput();
    if (inputs.empty())
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "empty struct create not supported");
    auto structTy = cast<hw::StructType>(op.getType());
    if (inputs.size() != structTy.getElements().size())
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "struct field/input arity mismatch");

    SmallVector<Value> packedInputs;
    packedInputs.reserve(inputs.size());
    for (auto [input, fieldInfo] : llvm::zip_equal(inputs, structTy.getElements())) {
      auto packed =
          packValueToBitVector(fieldInfo.type, input, rewriter, op.getLoc());
      if (failed(packed))
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "struct field must lower to bitvector");
      if (!isa<mlir::smt::BitVectorType>((*packed).getType()))
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "struct fields must be bitvectors");
      packedInputs.push_back(*packed);
    }

    Value result = packedInputs.front();
    for (auto input : llvm::drop_begin(packedInputs))
      result = mlir::smt::ConcatOp::create(rewriter, op.getLoc(), result, input);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower a hw::StructExtractOp operation to smt::ExtractOp.
struct StructExtractOpConversion : OpConversionPattern<StructExtractOp> {
  using OpConversionPattern<StructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = cast<hw::StructType>(op.getInput().getType());
    unsigned fieldIdx = op.getFieldIndex();
    auto fieldTy = structTy.getElements()[fieldIdx].type;
    int64_t fieldWidth = hw::getBitWidth(fieldTy);
    if (fieldWidth <= 0)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "0-bit struct fields not supported");
    auto fieldBvTy =
        mlir::smt::BitVectorType::get(op.getContext(), fieldWidth);
    auto offset = getStructFieldOffset(structTy, fieldIdx);
    Value fieldBits = mlir::smt::ExtractOp::create(
        rewriter, op.getLoc(), fieldBvTy, offset, adaptor.getInput());
    auto unpacked = unpackBitVectorToValue(fieldTy, fieldBits, *typeConverter,
                                           rewriter, op.getLoc());
    if (failed(unpacked))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "struct field unpack failed");
    rewriter.replaceOp(op, *unpacked);
    return success();
  }
};

/// Lower a hw::StructExplodeOp operation to smt::ExtractOp operations.
struct StructExplodeOpConversion : OpConversionPattern<StructExplodeOp> {
  using OpConversionPattern<StructExplodeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExplodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = cast<hw::StructType>(op.getInput().getType());
    SmallVector<Value> results;
    results.reserve(structTy.getElements().size());
    for (unsigned i = 0, e = structTy.getElements().size(); i < e; ++i) {
      auto fieldTy = structTy.getElements()[i].type;
      int64_t fieldWidth = hw::getBitWidth(fieldTy);
      if (fieldWidth <= 0)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "0-bit struct fields not supported");
      auto fieldBvTy =
          mlir::smt::BitVectorType::get(op.getContext(), fieldWidth);
      auto offset = getStructFieldOffset(structTy, i);
      Value fieldBits = mlir::smt::ExtractOp::create(
          rewriter, op.getLoc(), fieldBvTy, offset, adaptor.getInput());
      auto unpacked = unpackBitVectorToValue(fieldTy, fieldBits, *typeConverter,
                                             rewriter, op.getLoc());
      if (failed(unpacked))
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "struct field unpack failed");
      results.push_back(*unpacked);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Remove redundant (seq::FromClock and seq::ToClock) ops.
template <typename OpTy>
struct ReplaceWithInput : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Lower seq.const_clock to a 1-bit SMT bit-vector constant.
struct ConstClockOpConversion : OpConversionPattern<seq::ConstClockOp> {
  using OpConversionPattern<seq::ConstClockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::ConstClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    uint64_t value = op.getValue() == seq::ClockConst::High ? 1 : 0;
    rewriter.replaceOpWithNewOp<mlir::smt::BVConstantOp>(op, value, 1);
    return success();
  }
};

/// Normalize singleton array ops that use i0 indices to i1 zero indices.
/// This avoids carrying zero-width index values into SMT conversion.
struct NormalizeSingletonArrayGetIndex : OpRewritePattern<ArrayGetOp> {
  using OpRewritePattern<ArrayGetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArrayGetOp op,
                                PatternRewriter &rewriter) const override {
    auto indexType = dyn_cast<IntegerType>(op.getIndex().getType());
    if (!indexType || indexType.getWidth() != 0)
      return failure();
    if (cast<ArrayType>(op.getInput().getType()).getNumElements() != 1)
      return failure();

    auto oldConst = op.getIndex().getDefiningOp<ConstantOp>();
    bool eraseOldConst = oldConst && oldConst->hasOneUse();
    auto zero =
        ConstantOp::create(rewriter, op.getLoc(), rewriter.getIntegerType(1), 0);
    rewriter.replaceOpWithNewOp<ArrayGetOp>(op, op.getInput(), zero);
    if (eraseOldConst)
      rewriter.eraseOp(oldConst);
    return success();
  }
};

struct NormalizeSingletonArrayInjectIndex : OpRewritePattern<ArrayInjectOp> {
  using OpRewritePattern<ArrayInjectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArrayInjectOp op,
                                PatternRewriter &rewriter) const override {
    auto indexType = dyn_cast<IntegerType>(op.getIndex().getType());
    if (!indexType || indexType.getWidth() != 0)
      return failure();
    if (cast<ArrayType>(op.getInput().getType()).getNumElements() != 1)
      return failure();

    auto oldConst = op.getIndex().getDefiningOp<ConstantOp>();
    bool eraseOldConst = oldConst && oldConst->hasOneUse();
    auto zero =
        ConstantOp::create(rewriter, op.getLoc(), rewriter.getIntegerType(1), 0);
    rewriter.replaceOpWithNewOp<ArrayInjectOp>(op, op.getInput(), zero,
                                               op.getElement());
    if (eraseOldConst)
      rewriter.eraseOp(oldConst);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert HW to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHWToSMTPass
    : public impl::ConvertHWToSMTBase<ConvertHWToSMTPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void circt::populateHWToSMTTypeConverter(TypeConverter &converter) {
  // The semantics of the builtin integer at the CIRCT core level is currently
  // not very well defined. It is used for two-valued, four-valued, and possible
  // other multi-valued logic. Here, we interpret it as two-valued for now.
  // From a formal perspective, CIRCT would ideally define its own types for
  // two-valued, four-valued, nine-valued (etc.) logic each. In MLIR upstream
  // the integer type also carries poison information (which we don't have in
  // CIRCT?).
  converter.addConversion([](IntegerType type) -> std::optional<Type> {
    if (type.getWidth() <= 0)
      return std::nullopt;
    return mlir::smt::BitVectorType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([](seq::ClockType type) -> std::optional<Type> {
    return mlir::smt::BitVectorType::get(type.getContext(), 1);
  });
  converter.addConversion([&](llhd::RefType type) -> std::optional<Type> {
    return converter.convertType(type.getNestedType());
  });
  converter.addConversion([&](hw::TypeAliasType type) -> std::optional<Type> {
    return converter.convertType(type.getCanonicalType());
  });
  converter.addConversion([&](ArrayType type) -> std::optional<Type> {
    auto rangeType = converter.convertType(type.getElementType());
    if (!rangeType)
      return {};
    auto domainType = mlir::smt::BitVectorType::get(
        type.getContext(), getArrayDomainWidth(type.getNumElements()));
    return mlir::smt::ArrayType::get(type.getContext(), domainType, rangeType);
  });
  converter.addConversion([](StructType type) -> std::optional<Type> {
    auto width = hw::getBitWidth(type);
    if (width <= 0)
      return std::nullopt;
    return mlir::smt::BitVectorType::get(type.getContext(), width);
  });
  converter.addConversion([](Type type) -> std::optional<Type> {
    auto &dialect = type.getDialect();
    if (dialect.getNamespace() != "llvm")
      return std::nullopt;
    std::string sortName;
    llvm::raw_string_ostream os(sortName);
    type.print(os);
    return mlir::smt::SortType::get(type.getContext(), os.str());
  });

  auto ensureInsertionInValueRegion = [](OpBuilder &builder, Value input) {
    if (auto *defOp = input.getDefiningOp()) {
      builder.setInsertionPointAfter(defOp);
      return;
    }
    if (auto *block = input.getParentBlock())
      builder.setInsertionPointToStart(block);
  };

  // Default target materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                    inputs)
        ->getResult(0);
  });

  // Convert a 'smt.bool'-typed value to a 'smt.bv<N>'-typed value
  converter.addTargetMaterialization([&](OpBuilder &builder,
                                         mlir::smt::BitVectorType resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    if (!isa<mlir::smt::BoolType>(inputs[0].getType()))
      return Value();

    OpBuilder::InsertionGuard guard(builder);
    ensureInsertionInValueRegion(builder, inputs[0]);

    unsigned width = resultType.getWidth();
    Value constZero = mlir::smt::BVConstantOp::create(builder, loc, 0, width);
    Value constOne = mlir::smt::BVConstantOp::create(builder, loc, 1, width);
    return mlir::smt::IteOp::create(builder, loc, inputs[0], constOne,
                                    constZero);
  });

  // Convert an unrealized conversion cast from 'smt.bool' to i1
  // into a direct conversion from 'smt.bool' to 'smt.bv<1>'.
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, mlir::smt::BitVectorType resultType,
          ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1 || resultType.getWidth() != 1)
          return Value();

        auto intType = dyn_cast<IntegerType>(inputs[0].getType());
        if (!intType || intType.getWidth() != 1)
          return Value();

        auto castOp =
            inputs[0].getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          return Value();

        if (!isa<mlir::smt::BoolType>(castOp.getInputs()[0].getType()))
          return Value();

        OpBuilder::InsertionGuard guard(builder);
        ensureInsertionInValueRegion(builder, inputs[0]);

        Value constZero = mlir::smt::BVConstantOp::create(builder, loc, 0, 1);
        Value constOne = mlir::smt::BVConstantOp::create(builder, loc, 1, 1);
        return mlir::smt::IteOp::create(builder, loc, castOp.getInputs()[0],
                                        constOne, constZero);
      });

  // Convert a 'smt.bv<1>'-typed value to a 'smt.bool'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, mlir::smt::BoolType resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1)
          return Value();

        auto bvType = dyn_cast<mlir::smt::BitVectorType>(inputs[0].getType());
        if (!bvType || bvType.getWidth() != 1)
          return Value();

        OpBuilder::InsertionGuard guard(builder);
        ensureInsertionInValueRegion(builder, inputs[0]);

        Value constOne = mlir::smt::BVConstantOp::create(builder, loc, 1, 1);
        return mlir::smt::EqOp::create(builder, loc, inputs[0], constOne);
      });

  // Default source materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                    inputs)
        ->getResult(0);
  });
}

void circt::populateHWToSMTConversionPatterns(TypeConverter &converter,
                                              RewritePatternSet &patterns,
                                              bool forSMTLIBExport) {
  patterns.add<HWConstantOpConversion, HWAggregateConstantOpConversion,
               WireOpConversion, BitcastOpConversion, InstanceOpConversion,
               ReplaceWithInput<seq::ToClockOp>,
               ReplaceWithInput<seq::FromClockOp>, ConstClockOpConversion,
               ArrayCreateOpConversion, ArrayGetOpConversion,
               ArrayInjectOpConversion, StructCreateOpConversion,
               StructExtractOpConversion, StructExplodeOpConversion>(
      converter, patterns.getContext());
  patterns.add<OutputOpConversion, HWModuleOpConversion,
               HWModuleExternOpConversion>(
      converter, patterns.getContext(), forSMTLIBExport);
}

void ConvertHWToSMTPass::runOnOperation() {
  {
    RewritePatternSet normalizePatterns(&getContext());
    normalizePatterns
        .add<NormalizeSingletonArrayGetIndex,
             NormalizeSingletonArrayInjectIndex>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(normalizePatterns));
  }

  if (forSMTLIBExport) {
    auto numModules = 0;
    auto numInstances = 0;
    getOperation().walk([&](Operation *op) {
      if (isa<hw::HWModuleOp>(op))
        numModules++;
      if (isa<hw::InstanceOp>(op))
        numInstances++;
    });
    // Error out if there is any module hierarchy or multiple modules
    // Currently there's no need as this flag is intended for SMTLIB export and
    // we can just flatten modules
    if (numModules > 1) {
      getOperation()->emitError("multiple hw.module operations are not "
                                "supported with for-smtlib-export");
      return;
    }
    if (numInstances > 0) {
      getOperation()->emitError("hw.instance operations are not supported "
                                "with for-smtlib-export");
      return;
    }
  }

  ConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalOp<seq::FromClockOp>();
  target.addIllegalOp<seq::ToClockOp>();
  target.addIllegalOp<seq::ConstClockOp>();
  target.addLegalDialect<mlir::smt::SMTDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  populateHWToSMTConversionPatterns(converter, patterns, forSMTLIBExport);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Sort the functions topologically because 'hw.module' has a graph region
  // while 'func.func' is a regular SSACFG region. Real combinational cycles or
  // pseudo cycles through module instances are not supported yet.
  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    // Skip functions that are definitely not the result of lowering from
    // 'hw.module'
    if (func.getBody().getBlocks().size() != 1)
      continue;

    mlir::sortTopologically(&func.getBody().front());
  }
}
