// RUN: not env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s

// Regression: referenced externs with unsupported trampoline ABI types must be
// diagnosed at compile time, not left as unresolved runtime symbols.
//
// CHECK: error: cannot generate interpreter trampoline for referenced external function: unsupported trampoline ABI (return type vector<2xi32>)
// CHECK: [circt-compile] Failed to generate trampolines

module {
  // Force demotion via verif.assume.
  func.func @"uvm_pkg::uvm_demo::vec_ret"() -> vector<2xi32> {
    %true = arith.constant true
    verif.assume %true : i1
    %v = arith.constant dense<[7, 9]> : vector<2xi32>
    return %v : vector<2xi32>
  }

  // Keep the call live so lowering creates an external declaration for vec_ret.
  func.func @caller() -> i1 {
    %v = func.call @"uvm_pkg::uvm_demo::vec_ret"() : () -> vector<2xi32>
    %bits = llvm.bitcast %v : vector<2xi32> to i64
    %exp = arith.constant 38654705671 : i64
    %ok = arith.cmpi eq, %bits, %exp : i64
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
