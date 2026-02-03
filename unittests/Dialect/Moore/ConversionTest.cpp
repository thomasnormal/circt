#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

bool isFourStateStructType(Type type) {
  auto structType = dyn_cast<hw::StructType>(type);
  if (!structType)
    return false;
  auto fields = structType.getElements();
  if (fields.size() != 2)
    return false;
  return fields[0].name.getValue() == "value" &&
         fields[1].name.getValue() == "unknown";
}

bool isLLVMFourStateStructType(Type type, unsigned &width) {
  auto llvmStructType = dyn_cast<LLVM::LLVMStructType>(type);
  if (!llvmStructType || llvmStructType.isOpaque())
    return false;
  auto body = llvmStructType.getBody();
  if (body.size() != 2)
    return false;
  auto valueType = dyn_cast<IntegerType>(body[0]);
  auto unknownType = dyn_cast<IntegerType>(body[1]);
  if (!valueType || !unknownType)
    return false;
  if (valueType.getWidth() != unknownType.getWidth())
    return false;
  width = valueType.getWidth();
  return true;
}

bool hasMaskedUnknownStructCreate(ModuleOp module) {
  bool found = false;
  module.walk([&](hw::StructCreateOp create) {
    if (found)
      return;
    if (!isFourStateStructType(create.getType()))
      return;
    if (create.getInput().size() != 2)
      return;
    Value unk = create.getInput()[1];
    if (auto unkConst = unk.getDefiningOp<hw::ConstantOp>()) {
      if (unkConst.getValue().isZero())
        return;
    }
    auto andOp = create.getInput()[0].getDefiningOp<comb::AndOp>();
    if (!andOp)
      return;

    auto isAllOnesConst = [](Value v) {
      auto cst = v.getDefiningOp<hw::ConstantOp>();
      return cst && cst.getValue().isAllOnes();
    };
    auto isNotUnknown = [&](Value v) {
      auto xorOp = v.getDefiningOp<comb::XorOp>();
      if (!xorOp)
        return false;
      Value lhs = xorOp.getOperand(0);
      Value rhs = xorOp.getOperand(1);
      return (lhs == unk && isAllOnesConst(rhs)) ||
             (rhs == unk && isAllOnesConst(lhs));
    };

    if (isNotUnknown(andOp.getOperand(0)) ||
        isNotUnknown(andOp.getOperand(1)))
      found = true;
  });
  return found;
}

} // namespace

TEST(MooreToCoreConversionTest, IntDomainConversionPreservesValueOrder) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @intDomainConversion(out o: !moore.l1) {
      %c = moore.constant 1 : !moore.i1
      %l = moore.conversion %c : !moore.i1 -> !moore.l1
      moore.output %l : !moore.l1
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool sawBitcastToStruct = false;
  bool sawZeroUnknown = false;

  module->walk([&](hw::BitcastOp op) {
    if (isFourStateStructType(op.getType()))
      sawBitcastToStruct = true;
  });

  module->walk([&](Operation *op) {
    if (auto agg = dyn_cast<hw::AggregateConstantOp>(op)) {
      if (!isFourStateStructType(agg.getType()))
        return;
      auto fields = agg.getFieldsAttr();
      if (fields.size() != 2)
        return;
      auto valueAttr = dyn_cast<IntegerAttr>(fields[0]);
      auto unknownAttr = dyn_cast<IntegerAttr>(fields[1]);
      if (!valueAttr || !unknownAttr)
        return;
      if (valueAttr.getValue().isOne() && unknownAttr.getValue().isZero())
        sawZeroUnknown = true;
      return;
    }

    if (auto create = dyn_cast<hw::StructCreateOp>(op)) {
      if (!isFourStateStructType(create.getType()))
        return;
      if (create.getInput().size() != 2)
        return;
      auto valueCst =
          create.getInput()[0].getDefiningOp<hw::ConstantOp>();
      auto unknownCst =
          create.getInput()[1].getDefiningOp<hw::ConstantOp>();
      if (!valueCst || !unknownCst)
        return;
      if (valueCst.getValue().isOne() && unknownCst.getValue().isZero())
        sawZeroUnknown = true;
    }
  });

  EXPECT_FALSE(sawBitcastToStruct);
  EXPECT_TRUE(sawZeroUnknown);
}

TEST(MooreToCoreConversionTest, FourStateXorMasksUnknownValue) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @xorMask(in %a : !moore.l4, in %b : !moore.l4, out out : !moore.l4) {
      %x = moore.xor %a, %b : l4
      moore.output %x : !moore.l4
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  EXPECT_TRUE(hasMaskedUnknownStructCreate(*module));
}

TEST(MooreToCoreConversionTest, FourStateAddMasksUnknownValue) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @addMask(in %a : !moore.l4, in %b : !moore.l4, out out : !moore.l4) {
      %x = moore.add %a, %b : l4
      moore.output %x : !moore.l4
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  EXPECT_TRUE(hasMaskedUnknownStructCreate(*module));
}

TEST(MooreToCoreConversionTest, FourStateConsensusSimplifiesToInput) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @consensus(in %a : !moore.l4, in %b : !moore.l4, out out : !moore.l4) {
      %nb = moore.not %b : l4
      %and0 = moore.and %a, %b : l4
      %and1 = moore.and %a, %nb : l4
      %x = moore.xor %and0, %and1 : l4
      moore.output %x : !moore.l4
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  auto hwModule = module->lookupSymbol<hw::HWModuleOp>("consensus");
  ASSERT_TRUE(hwModule);

  auto outputOp = *hwModule.getBodyBlock()->getOps<hw::OutputOp>().begin();
  ASSERT_EQ(outputOp.getNumOperands(), 1u);

  Value outVal = outputOp.getOperand(0);
  Value arg0 = hwModule.getBodyBlock()->getArgument(0);
  bool passThrough = outVal == arg0;
  if (!passThrough) {
    if (auto create = outVal.getDefiningOp<hw::StructCreateOp>()) {
      if (create.getInput().size() == 2) {
        auto valExtract = create.getInput()[0].getDefiningOp<hw::StructExtractOp>();
        auto unkExtract = create.getInput()[1].getDefiningOp<hw::StructExtractOp>();
        if (valExtract && unkExtract && valExtract.getInput() == arg0 &&
            unkExtract.getInput() == arg0)
          passThrough = true;
      }
    }
  }
  EXPECT_TRUE(passThrough);
}

TEST(MooreToCoreConversionTest, FourStateAbsorptionSimplifiesToInput) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @absorb(in %a : !moore.l4, in %b : !moore.l4, out out : !moore.l4) {
      %and0 = moore.and %a, %b : l4
      %or0 = moore.or %a, %b : l4
      %x0 = moore.or %a, %and0 : l4
      %x1 = moore.and %a, %or0 : l4
      %x2 = moore.xor %x0, %x1 : l4
      moore.output %x2 : !moore.l4
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  auto hwModule = module->lookupSymbol<hw::HWModuleOp>("absorb");
  ASSERT_TRUE(hwModule);

  auto outputOp = *hwModule.getBodyBlock()->getOps<hw::OutputOp>().begin();
  ASSERT_EQ(outputOp.getNumOperands(), 1u);

  Value outVal = outputOp.getOperand(0);
  bool zeroConst = false;
  if (auto agg = outVal.getDefiningOp<hw::AggregateConstantOp>()) {
    if (isFourStateStructType(agg.getType())) {
      auto fields = agg.getFieldsAttr();
      auto valueAttr = dyn_cast<IntegerAttr>(fields[0]);
      auto unknownAttr = dyn_cast<IntegerAttr>(fields[1]);
      if (valueAttr && unknownAttr && valueAttr.getValue().isZero() &&
          unknownAttr.getValue().isZero())
        zeroConst = true;
    }
  } else if (auto create = outVal.getDefiningOp<hw::StructCreateOp>()) {
    if (create.getInput().size() == 2) {
      auto valueCst = create.getInput()[0].getDefiningOp<hw::ConstantOp>();
      auto unknownCst = create.getInput()[1].getDefiningOp<hw::ConstantOp>();
      if (valueCst && unknownCst && valueCst.getValue().isZero() &&
          unknownCst.getValue().isZero())
        zeroConst = true;
    }
  }

  EXPECT_TRUE(zeroConst);
}

TEST(MooreToCoreConversionTest, FourStateGlobalVariableUsesLLVMType) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  // A 4-state global variable (l8) should produce an LLVM global with
  // llvm.struct<(i8, i8)> rather than hw.struct<value: i8, unknown: i8>.
  const char *ir = R"mlir(
    moore.global_variable @fourStateGlobal : !moore.l8
    moore.module @useFourStateGlobal() {
      %g = moore.get_global_variable @fourStateGlobal : <l8>
      moore.output
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  // Verify that the LLVM global has an LLVM struct type, not hw::StructType.
  bool foundLLVMGlobal = false;
  bool hasHWStructType = false;
  module->walk([&](LLVM::GlobalOp globalOp) {
    if (globalOp.getSymName() == "fourStateGlobal") {
      foundLLVMGlobal = true;
      Type globalType = globalOp.getGlobalType();
      // Should be LLVM struct type, not hw struct type.
      if (isa<hw::StructType>(globalType))
        hasHWStructType = true;
    }
  });

  EXPECT_TRUE(foundLLVMGlobal);
  EXPECT_FALSE(hasHWStructType);
}

TEST(MooreToCoreConversionTest, FourStateLLVMStructCastPreservesFieldOrder) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    hw.module @cast_from_llvm(out out : !hw.struct<value: i8, unknown: i8>) {
      %undef = llvm.mlir.undef : !llvm.struct<(i8, i8)>
      %val = hw.constant 90 : i8
      %unk = hw.constant 15 : i8
      %s0 = llvm.insertvalue %val, %undef[0] : !llvm.struct<(i8, i8)>
      %s1 = llvm.insertvalue %unk, %s0[1] : !llvm.struct<(i8, i8)>
      %hw = builtin.unrealized_conversion_cast %s1 : !llvm.struct<(i8, i8)> to !hw.struct<value: i8, unknown: i8>
      hw.output %hw : !hw.struct<value: i8, unknown: i8>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool sawOrderedAggregate = false;
  module->walk([&](hw::AggregateConstantOp agg) {
    if (!isFourStateStructType(agg.getType()))
      return;
    auto fields = agg.getFieldsAttr();
    if (fields.size() != 2)
      return;
    auto valueAttr = dyn_cast<IntegerAttr>(fields[0]);
    auto unknownAttr = dyn_cast<IntegerAttr>(fields[1]);
    if (!valueAttr || !unknownAttr)
      return;
    if (valueAttr.getValue().getLimitedValue() == 90 &&
        unknownAttr.getValue().getLimitedValue() == 15)
      sawOrderedAggregate = true;
  });

  bool sawFourStateCast = false;
  module->walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    Type srcType = cast.getOperand(0).getType();
    Type dstType = cast.getResult(0).getType();
    unsigned width = 0;
    if ((isFourStateStructType(srcType) &&
         isLLVMFourStateStructType(dstType, width)) ||
        (isFourStateStructType(dstType) &&
         isLLVMFourStateStructType(srcType, width)))
      sawFourStateCast = true;
  });

  EXPECT_TRUE(sawOrderedAggregate);
  EXPECT_FALSE(sawFourStateCast);
}

TEST(MooreToCoreConversionTest, FourStateConditionalMasksUnknownValue) {
  MLIRContext context;
  context.loadDialect<moore::MooreDialect, hw::HWDialect, comb::CombDialect,
                      seq::SeqDialect, llhd::LLHDDialect, ltl::LTLDialect,
                      verif::VerifDialect, LLVM::LLVMDialect,
                      arith::ArithDialect>();

  const char *ir = R"mlir(
    moore.module @condMask(in %c : !moore.l1, in %a : !moore.l4, in %b : !moore.l4, out out : !moore.l4) {
      %x = moore.conditional %c : l1 -> l4 {
        moore.yield %a : l4
      } {
        moore.yield %b : l4
      }
      moore.output %x : !moore.l4
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createConvertMooreToCorePass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  EXPECT_TRUE(hasMaskedUnknownStructCreate(*module));
}
