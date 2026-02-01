//===-- Passes.td - LEC Transforms pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for LEC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_LEC_PASSES_TD
#define CIRCT_TOOLS_CIRCT_LEC_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConstructLEC : Pass<"construct-lec", "::mlir::ModuleOp"> {
  let summary = "Lower CIRCTs core dialects to a LEC problem statement";
  let description = [{
    Takes two `hw.module` operations and lowers them to a `verif.lec` operation
    inside a `func.func` matching the name of the first module.

    This pass can also (optionally) insert a 'main' function that calls the
    function with the `verif.lec` and adheres to the typical main function
    signature. This is useful when compiling to a standalone binary rather than
    using the JIT compiler.

    This pass also inserts a few LLVM dialect operation to print the result of
    the `verif.lec` using the `printf` function of the standard library.
  }];

  let options = [
    Option<"firstModule", "first-module", "std::string",
           /*default=*/"",
           "Name of the first of the two modules to compare.">,
    Option<"secondModule", "second-module", "std::string",
           /*default=*/"",
           "Name of the second of the two modules to compare.">,
    Option<"allowIOAlignment", "align-io", "bool",
           /*default=*/"false",
           "Allow aligning mismatched module inputs for approximate LEC.">,
    Option<"insertMode", "insert-mode", "lec::InsertAdditionalModeEnum",
           /*default=*/"lec::InsertAdditionalModeEnum::Reporting",           
            "Select what additional code to add.",
            [{::llvm::cl::values(
             clEnumValN(lec::InsertAdditionalModeEnum::None,
             "none", "Do not insert any additional code."),
             clEnumValN(lec::InsertAdditionalModeEnum::Reporting,
             "reporting", "Insert LLVM code to report LEC result."),
             clEnumValN(lec::InsertAdditionalModeEnum::Main,
             "main", "Insert main function for AOT compilation and reporting.")
           )}]>,
  ];

  let dependentDialects = [
    "verif::VerifDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def StripLLHDInterfaceSignals
    : Pass<"strip-llhd-interface-signals", "::mlir::ModuleOp"> {
  let summary = "Strip LLHD interface signal storage for LEC";
  let description = [{
    Convert interface-backed LLHD signal storage patterns into direct SSA
    values so that LEC can run on HW/Comb/Verif without LLHD/LLVM ops.
  }];
  let options = [
    Option<"strict", "strict-llhd", "bool",
           /*default=*/"false",
           "Fail when LLHD interface stripping would require abstraction.">
  ];

  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::llhd::LLHDDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def LowerLECLLVM : Pass<"lower-lec-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower simple LLVM struct patterns for LEC";
  let description = [{
    Converts common LLVM struct construction patterns (alloca/store/load and
    insertvalue chains feeding `unrealized_conversion_cast`) into HW structs so
    the LEC pipeline can proceed without LLVM dialect ops.
  }];

  let dependentDialects = [
    "circt::hw::HWDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def LowerLLHDRefPorts
    : Pass<"lower-llhd-ref-ports", "::mlir::ModuleOp"> {
  let summary = "Lower LLHD ref ports to value ports for LEC";
  let description = [{
    Convert `llhd.ref` ports to value ports by reclassifying externally-driven
    ref outputs as inputs and rewriting instance connections. This enables
    subsequent LLHD-to-core lowering to eliminate LLHD ops in formal flows.
  }];
  let options = [
    Option<"strict", "strict-llhd", "bool",
           /*default=*/"false",
           "Fail when LLHD ref lowering would drop internal drives.">
  ];

  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::llhd::LLHDDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_LEC_PASSES_TD
