// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: compiled->interpreted trampoline return packing must preserve
// wide integer bits for non-struct returns (e.g. i128).
//
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
// SIM: hi=-1
// COMPILED: hi=-1

module {
  // Mark function as non-compilable (verif.assume) so it is forced through a
  // trampoline when compiled code calls it.
  func.func @"uvm_pkg::uvm_demo::wide_ret"() -> i128 {
    %true = arith.constant true
    verif.assume %true : i1

    %one = arith.constant 1 : i128
    %shamt = arith.constant 100 : i128
    %hi = arith.shli %one, %shamt : i128
    %ret = arith.ori %hi, %one : i128
    return %ret : i128
  }

  // This function is compilable and calls the demoted callee through a
  // generated trampoline in compiled mode.
  func.func @caller() -> i1 {
    %r = func.call @"uvm_pkg::uvm_demo::wide_ret"() : () -> i128
    %hi = comb.extract %r from 100 : (i128) -> i1
    return %hi : i1
  }

  hw.module @test() {
    %fmt_hi = sim.fmt.literal "hi="
    %fmt_nl = sim.fmt.literal "\0A"

    %c10_i64 = hw.constant 10000000 : i64

    llhd.process {
      %hi = func.call @caller() : () -> i1
      %fhi = sim.fmt.dec %hi signed : i1
      %msg = sim.fmt.concat (%fmt_hi, %fhi, %fmt_nl)
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
