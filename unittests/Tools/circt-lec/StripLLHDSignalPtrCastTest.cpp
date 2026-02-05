#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(StripLLHDSignalPtrCastTest, AliasesPortNamedSignalsWithoutDrives) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect>();

  const char *ir = R"mlir(
    hw.module @port_named_signal(in %data_i : i8, out out : i8) {
      %c0 = hw.constant 0 : i8
      %sig = llhd.sig name "data_i" %c0 : i8
      %val = llhd.prb %sig : i8
      hw.output %val : i8
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createStripLLHDInterfaceSignals());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  module->walk([&](Operation *op) {
    if (auto *dialect = op->getDialect())
      if (dialect->getNamespace() == "llhd")
        hasLLHD = true;
  });
  EXPECT_FALSE(hasLLHD);

  bool checkedOutput = false;
  module->walk([&](circt::hw::HWModuleOp hwModule) {
    if (hwModule.getSymName() != "port_named_signal")
      return;
    auto outputOp =
        cast<circt::hw::OutputOp>(hwModule.getBodyBlock()->getTerminator());
    Value input = hwModule.getArgumentForInput(0);
    EXPECT_EQ(outputOp.getOperand(0), input);
    checkedOutput = true;
  });
  EXPECT_TRUE(checkedOutput);
}

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
  circt::StripLLHDInterfaceSignalsOptions options;
  options.strict = true;
  pm.addPass(circt::createStripLLHDInterfaceSignals(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  bool hasCombInput = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLHD);
  module->walk([&](circt::hw::HWModuleOp hwModule) {
    for (auto nameAttr : hwModule.getModuleType().getInputNames()) {
      if (cast<StringAttr>(nameAttr).getValue() == "llhd_comb")
        hasCombInput = true;
    }
  });
  EXPECT_FALSE(hasCombInput);
}

TEST(StripLLHDSignalPtrCastTest, HandlesAllocaPhiRefMerge) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect, mlir::cf::ControlFlowDialect>();

  const char *ir = R"mlir(
    hw.module @alloca_phi_ref_merge(in %cond : i1, in %in0 : i1, in %in1 : i1,
                                    out out : i1) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %t0 = llhd.constant_time <0ns, 0d, 1e>
      %comb = llhd.combinational -> i1 {
        %ptr0 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
        %ptr1 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
        cf.cond_br %cond, ^bb1, ^bb2
      ^bb1:
        llvm.store %in0, %ptr0 : i1, !llvm.ptr
        cf.br ^bb3(%ptr0 : !llvm.ptr)
      ^bb2:
        llvm.store %in1, %ptr1 : i1, !llvm.ptr
        cf.br ^bb3(%ptr1 : !llvm.ptr)
      ^bb3(%arg : !llvm.ptr):
        %ref = builtin.unrealized_conversion_cast %arg : !llvm.ptr to !llhd.ref<i1>
        %val = llhd.prb %ref : i1
        llhd.drv %ref, %val after %t0 : i1
        %load = llvm.load %arg : !llvm.ptr -> i1
        llhd.yield %load : i1
      }
      hw.output %comb : i1
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createStripLLHDInterfaceSignals());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  bool hasCombInput = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLHD);
  module->walk([&](circt::hw::HWModuleOp hwModule) {
    for (auto nameAttr : hwModule.getModuleType().getInputNames()) {
      if (cast<StringAttr>(nameAttr).getValue() == "llhd_comb")
        hasCombInput = true;
    }
  });
  EXPECT_FALSE(hasCombInput);
}

TEST(StripLLHDSignalPtrCastTest, CollapsesPointerPhiStoreLoad) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect, mlir::cf::ControlFlowDialect>();

  const char *ir = R"mlir(
    hw.module @ptr_phi_store_load(in %cond : i1, in %a : i8, in %b : i8, out out : i8) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %comb = llhd.combinational -> i8 {
        %p1 = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
        %p2 = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
        cf.cond_br %cond, ^bb1(%a, %p1 : i8, !llvm.ptr), ^bb1(%b, %p2 : i8, !llvm.ptr)
      ^bb1(%val: i8, %ptr: !llvm.ptr):
        llvm.store %val, %ptr : i8, !llvm.ptr
        %loaded = llvm.load %ptr : !llvm.ptr -> i8
        llhd.yield %loaded : i8
      }
      hw.output %comb : i8
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  circt::StripLLHDInterfaceSignalsOptions options;
  options.strict = true;
  pm.addPass(circt::createStripLLHDInterfaceSignals(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  bool hasCombInput = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLHD);
  module->walk([&](circt::hw::HWModuleOp hwModule) {
    for (auto nameAttr : hwModule.getModuleType().getInputNames()) {
      if (cast<StringAttr>(nameAttr).getValue() == "llhd_comb")
        hasCombInput = true;
    }
  });
  EXPECT_FALSE(hasCombInput);
}

TEST(StripLLHDSignalPtrCastTest, HandlesLocalRefExtractUpdate) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @sig_ref_extract(in %bit : i1, out out : i8) {
      %t0 = llhd.constant_time <0ns, 0d, 1e>
      %c0_i3 = hw.constant 0 : i3
      %c0_i8 = hw.constant 0 : i8
      %sig = llhd.sig %c0_i8 {lec.local} : i8
      %comb = llhd.combinational -> i8 {
        %bitref = llhd.sig.extract %sig from %c0_i3 : <i8> -> <i1>
        llhd.drv %bitref, %bit after %t0 : i1
        %val = llhd.prb %sig : i8
        llhd.yield %val : i8
      }
      hw.output %comb : i8
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  circt::StripLLHDInterfaceSignalsOptions options;
  options.strict = true;
  pm.addPass(circt::createStripLLHDInterfaceSignals(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  bool hasCombInput = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLHD);
  module->walk([&](circt::hw::HWModuleOp hwModule) {
    for (auto nameAttr : hwModule.getModuleType().getInputNames()) {
      if (cast<StringAttr>(nameAttr).getValue() == "llhd_comb")
        hasCombInput = true;
    }
  });
  EXPECT_FALSE(hasCombInput);
}

TEST(StripLLHDSignalPtrCastTest, ClearsUnknownOnValueFieldUpdate) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @sig_ref_value_4state(in %bit : i1,
                                    out out : !hw.struct<value: i1, unknown: i1>) {
      %t0 = llhd.constant_time <0ns, 0d, 1e>
      %c0 = hw.constant 0 : i1
      %c1 = hw.constant 1 : i1
      %init = hw.struct_create (%c0, %c1) : !hw.struct<value: i1, unknown: i1>
      %sig = llhd.sig %init {lec.local} : !hw.struct<value: i1, unknown: i1>
      %comb = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
        %ref = llhd.sig.struct_extract %sig["value"] : <!hw.struct<value: i1, unknown: i1>>
        llhd.drv %ref, %bit after %t0 : i1
        %val = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
        llhd.yield %val : !hw.struct<value: i1, unknown: i1>
      }
      hw.output %comb : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  circt::StripLLHDInterfaceSignalsOptions options;
  options.strict = true;
  pm.addPass(circt::createStripLLHDInterfaceSignals(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLHD = false;
  module->walk([&](Operation *op) {
    if (op->getDialect()) {
      if (op->getDialect()->getNamespace() == "llhd")
        hasLLHD = true;
    }
  });
  EXPECT_FALSE(hasLLHD);

  auto hwModule = *module->getOps<circt::hw::HWModuleOp>().begin();
  Value bitArg = hwModule.getBodyBlock()->getArgument(0);
  bool foundUpdate = false;
  module->walk([&](circt::hw::StructCreateOp op) {
    if (op.getNumOperands() != 2)
      return;
    if (op.getOperand(0) != bitArg)
      return;
    auto constant = op.getOperand(1).getDefiningOp<circt::hw::ConstantOp>();
    if (!constant)
      return;
    if (constant.getValue().isZero())
      foundUpdate = true;
  });
  EXPECT_TRUE(foundUpdate);
}
