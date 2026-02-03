#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(VerifToSMTTest, XOptimisticOutputsBuildsMaskedDiff) {
  MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::verif::VerifDialect,
                      mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::scf::SCFDialect, mlir::smt::SMTDialect>();

  const char *ir = R"mlir(
    verif.lec first {
    ^bb0(%arg0: !hw.struct<value: i2, unknown: i2>):
      verif.yield %arg0 : !hw.struct<value: i2, unknown: i2>
    } second {
    ^bb0(%arg0: !hw.struct<value: i2, unknown: i2>):
      verif.yield %arg0 : !hw.struct<value: i2, unknown: i2>
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  circt::ConvertVerifToSMTOptions options;
  options.xOptimisticOutputs = true;
  pm.addPass(circt::createConvertVerifToSMT(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  bool sawMasking = false;
  module->walk([&](mlir::smt::BVAndOp) { sawMasking = true; });
  EXPECT_TRUE(sawMasking);
}
