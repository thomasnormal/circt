#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombDialect.h"
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
