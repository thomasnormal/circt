#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(LowerLECLLVMTest, RemovesLLVMStructOps) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_structs(in %in : !hw.struct<value: i1, unknown: i1>,
                                      out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %cast = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %cast : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawStructCreate = false;
  module->walk([&](circt::hw::StructCreateOp) { sawStructCreate = true; });
  EXPECT_TRUE(sawStructCreate);
}

TEST(LowerLECLLVMTest, HandlesMultipleStoresInSingleBlock) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_multi_store(
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %tmp2 = llvm.insertvalue %unknown, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp3 = llvm.insertvalue %value, %tmp2[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      %dummy = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !hw.struct<value: i1, unknown: i1>
      llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      llvm.store %tmp3, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %cast = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %cast : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawExpectedCreate = false;
  module->walk([&](circt::hw::StructCreateOp create) {
    if (create.getOperands().size() != 2)
      return;
    auto first =
        create.getOperands()[0].getDefiningOp<circt::hw::StructExtractOp>();
    auto second =
        create.getOperands()[1].getDefiningOp<circt::hw::StructExtractOp>();
    if (!first || !second)
      return;
    if (first.getFieldName() == "unknown" && second.getFieldName() == "value")
      sawExpectedCreate = true;
  });
  EXPECT_TRUE(sawExpectedCreate);
}

TEST(LowerLECLLVMTest, HandlesHWStructRoundtripCasts) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_roundtrip(
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %struct = hw.struct_create (%value, %unknown) : !hw.struct<value: i1, unknown: i1>
      %llvm = builtin.unrealized_conversion_cast %struct : !hw.struct<value: i1, unknown: i1> to !llvm.struct<(i1, i1)>
      %back = builtin.unrealized_conversion_cast %llvm : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %back : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawStructCreate = false;
  module->walk([&](circt::hw::StructCreateOp) { sawStructCreate = true; });
  EXPECT_TRUE(sawStructCreate);
}

TEST(LowerLECLLVMTest, HandlesPartialInsertFromLoadedStruct) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_partial_insert(
        in %lhs : !hw.struct<value: i1, unknown: i1>,
        in %rhs : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %lhs_value = hw.struct_extract %lhs["value"] : !hw.struct<value: i1, unknown: i1>
      %lhs_unknown = hw.struct_extract %lhs["unknown"] : !hw.struct<value: i1, unknown: i1>
      %lhs0 = llvm.insertvalue %lhs_value, %undef[0] : !llvm.struct<(i1, i1)>
      %lhs1 = llvm.insertvalue %lhs_unknown, %lhs0[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      llvm.store %lhs1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %rhs_value = hw.struct_extract %rhs["value"] : !hw.struct<value: i1, unknown: i1>
      %new = llvm.insertvalue %rhs_value, %load[0] : !llvm.struct<(i1, i1)>
      %cast = builtin.unrealized_conversion_cast %new : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %cast : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawStructCreate = false;
  module->walk([&](circt::hw::StructCreateOp) { sawStructCreate = true; });
  EXPECT_TRUE(sawStructCreate);
}

TEST(LowerLECLLVMTest, LowersLLVMSelectStruct) {
  MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_select(
        in %cond : i1,
        in %lhs : !hw.struct<value: i1, unknown: i1>,
        in %rhs : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %lhs_value = hw.struct_extract %lhs["value"] : !hw.struct<value: i1, unknown: i1>
      %lhs_unknown = hw.struct_extract %lhs["unknown"] : !hw.struct<value: i1, unknown: i1>
      %lhs0 = llvm.insertvalue %lhs_value, %undef[0] : !llvm.struct<(i1, i1)>
      %lhs1 = llvm.insertvalue %lhs_unknown, %lhs0[1] : !llvm.struct<(i1, i1)>
      %rhs_value = hw.struct_extract %rhs["value"] : !hw.struct<value: i1, unknown: i1>
      %rhs_unknown = hw.struct_extract %rhs["unknown"] : !hw.struct<value: i1, unknown: i1>
      %rhs0 = llvm.insertvalue %rhs_value, %undef[0] : !llvm.struct<(i1, i1)>
      %rhs1 = llvm.insertvalue %rhs_unknown, %rhs0[1] : !llvm.struct<(i1, i1)>
      %sel = llvm.select %cond, %lhs1, %rhs1 : i1, !llvm.struct<(i1, i1)>
      %cast = builtin.unrealized_conversion_cast %sel : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
      hw.output %cast : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawMux = false;
  module->walk([&](circt::comb::MuxOp) { sawMux = true; });
  EXPECT_TRUE(sawMux);
}

TEST(LowerLECLLVMTest, LowersAllocaBackedLLHDRef) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_ref_alloca(
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
      llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
      %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
      hw.output %probe : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawSignal = false;
  module->walk([&](circt::llhd::SignalOp) { sawSignal = true; });
  EXPECT_TRUE(sawSignal);
}

TEST(LowerLECLLVMTest, LowersAllocaBackedLLHDRefWithCast) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_ref_alloca_cast(
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      %ptr_as1 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<1>
      %ptr_as0 = llvm.addrspacecast %ptr_as1 : !llvm.ptr<1> to !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %ptr_as0 : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
      llvm.store %tmp1, %ptr_as0 : !llvm.struct<(i1, i1)>, !llvm.ptr
      %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
      hw.output %probe : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawSignal = false;
  module->walk([&](circt::llhd::SignalOp) { sawSignal = true; });
  EXPECT_TRUE(sawSignal);
}

TEST(LowerLECLLVMTest, LowersAllocaBackedLLHDRefWithBlockArg) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect, mlir::cf::ControlFlowDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_ref_alloca_block_arg(
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %outval = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
        %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
        cf.br ^bb1(%ptr : !llvm.ptr)

      ^bb1(%arg0: !llvm.ptr):
        %ref = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
        llvm.store %tmp1, %arg0 : !llvm.struct<(i1, i1)>, !llvm.ptr
        %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
        llhd.yield %probe : !hw.struct<value: i1, unknown: i1>
      }
      hw.output %outval : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawSignal = false;
  module->walk([&](circt::llhd::SignalOp) { sawSignal = true; });
  EXPECT_TRUE(sawSignal);
}

TEST(LowerLECLLVMTest, LowersAllocaBackedLLHDRefWithSelect) {
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::llhd::LLHDDialect,
                      LLVM::LLVMDialect>();

  const char *ir = R"mlir(
    hw.module @lower_lec_llvm_ref_alloca_select(
        in %cond : i1,
        in %in : !hw.struct<value: i1, unknown: i1>,
        out out : !hw.struct<value: i1, unknown: i1>) {
      %one = llvm.mlir.constant(1 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
      %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
      %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
      %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
      %ptr_as1 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<1>
      %ptr_as0 = llvm.addrspacecast %ptr_as1 : !llvm.ptr<1> to !llvm.ptr
      %sel = llvm.select %cond, %ptr, %ptr_as0 : i1, !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %sel : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
      llvm.store %tmp1, %sel : !llvm.struct<(i1, i1)>, !llvm.ptr
      %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
      hw.output %probe : !hw.struct<value: i1, unknown: i1>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createLowerLECLLVM());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool hasLLVM = false;
  module->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm")
      hasLLVM = true;
  });
  EXPECT_FALSE(hasLLVM);

  bool sawSignal = false;
  module->walk([&](circt::llhd::SignalOp) { sawSignal = true; });
  EXPECT_TRUE(sawSignal);
}
