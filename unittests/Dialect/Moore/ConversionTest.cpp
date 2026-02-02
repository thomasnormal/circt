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
