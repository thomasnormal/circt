#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(StripLLHDSignalPtrCastTest, HandlesPtrCastLoadStore) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @signal_ptr_cast(out out : !hw.struct<value: i1, unknown: i1>) {
      %init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
      %sig = llhd.sig %init : !hw.struct<value: i1, unknown: i1>
      %ptr = builtin.unrealized_conversion_cast %sig : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
      %val = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
      %val_llvm = builtin.unrealized_conversion_cast %val : !hw.struct<value: i1, unknown: i1> to !llvm.struct<(i1, i1)>
      llvm.store %val_llvm, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %load_hw = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %load_hw : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createStripLLHDInterfaceSignals());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  bool hasLLHD = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llvm")
        hasLLVM = true;
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLVM);
  EXPECT_FALSE(hasLLHD);
}
