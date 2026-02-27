// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: demoted functions with LLVM array return types must trampoline
// correctly instead of leaving unresolved extern symbols in compiled mode.
//
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
// SIM: ok=-1
// COMPILED: ok=-1

module {
  // Force demotion via verif.assume.
  func.func @"uvm_pkg::uvm_demo::arr_ret"() -> !llvm.array<2 x i64> {
    %true = arith.constant true
    verif.assume %true : i1

    %a0 = arith.constant 17 : i64
    %a1 = arith.constant 42 : i64
    %undef = llvm.mlir.undef : !llvm.array<2 x i64>
    %arr0 = llvm.insertvalue %a0, %undef[0] : !llvm.array<2 x i64>
    %arr1 = llvm.insertvalue %a1, %arr0[1] : !llvm.array<2 x i64>
    return %arr1 : !llvm.array<2 x i64>
  }

  // Compile this function; it calls demoted callee via generated trampoline.
  func.func @caller() -> i1 {
    %arr = func.call @"uvm_pkg::uvm_demo::arr_ret"() : () -> !llvm.array<2 x i64>
    %x = llvm.extractvalue %arr[1] : !llvm.array<2 x i64>
    %exp = arith.constant 42 : i64
    %ok = arith.cmpi eq, %x, %exp : i64
    return %ok : i1
  }

  hw.module @test() {
    %fmt_ok = sim.fmt.literal "ok="
    %fmt_nl = sim.fmt.literal "\0A"

    %c10_i64 = hw.constant 10000000 : i64

    llhd.process {
      %ok = func.call @caller() : () -> i1
      %fok = sim.fmt.dec %ok signed : i1
      %msg = sim.fmt.concat (%fmt_ok, %fok, %fmt_nl)
      sim.proc.print %msg
      llhd.halt
    }

    llhd.process {
      %d = llhd.int_to_time %c10_i64
      llhd.wait delay %d, ^done
    ^done:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
