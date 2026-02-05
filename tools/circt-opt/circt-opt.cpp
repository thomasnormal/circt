//===- circt-opt.cpp - The circt-opt driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-opt' tool, which is the circt analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/Version.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

// Defined in the test directory, no public header.
namespace circt {
namespace test {
void registerAnalysisTestPasses();
} // namespace test
} // namespace circt

namespace {
struct ResourceGuardPassInstrumentation final : public mlir::PassInstrumentation {
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::StringRef arg = pass->getArgument();
    llvm::StringRef label = arg.empty() ? pass->getName() : arg;
    llvm::StringRef opName =
        op ? op->getName().getStringRef() : llvm::StringRef("<null>");
    std::string combined = (label + "[" + opName + "]").str();
    circt::setResourceGuardPhase(combined);
  }
};
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::emitc::EmitCDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::index::IndexDialect>();

  circt::registerAllDialects(registry);
  circt::registerAllPasses();

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();
  mlir::registerReconcileUnrealizedCastsPass();
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Register test passes
  circt::test::registerAnalysisTestPasses();
  mlir::registerSROA();
  mlir::registerMem2RegPass();

  auto filenames = mlir::registerAndParseCLIOptions(
      argc, argv, "CIRCT modular optimizer driver", registry);
  circt::installResourceGuard();

  // Inject pass-level phase labels into the resource guard, while preserving
  // the default pass pipeline parsing behavior from `MlirOptMainConfig`.
  auto config = mlir::MlirOptMainConfig::createFromCLOptions();
  auto baseConfig = config;
  config.setPassPipelineSetupFn(
      [baseConfig](mlir::PassManager &pm) mutable -> mlir::LogicalResult {
        pm.addInstrumentation(
            std::make_unique<ResourceGuardPassInstrumentation>());
        return baseConfig.setupPassPipeline(pm);
      });

  std::string errorMessage;
  auto file = mlir::openInputFile(filenames.first, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }
  auto output = mlir::openOutputFile(filenames.second, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  circt::setResourceGuardPhase("pass pipeline");
  if (mlir::failed(mlir::MlirOptMain(output->os(), std::move(file), registry,
                                     config)))
    return EXIT_FAILURE;
  output->keep();
  return EXIT_SUCCESS;
}
