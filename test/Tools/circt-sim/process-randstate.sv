// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

// Test process::srandom(), process::get_randstate(), and process::set_randstate().
// IEEE 1800-2017 Section 9.7 "Process control"

module test_process_randstate;
  process p;
  string s1;
  string s2;

  initial begin
    p = process::self();
    p.srandom(123);
    s1 = p.get_randstate();

    // Change state, then restore it.
    p.srandom(456);
    p.set_randstate(s1);
    s2 = p.get_randstate();

    if (s1.len() == 0)
      $display("RANDSTATE_EMPTY");

    if (s1 == s2)
      $display("RANDSTATE_OK");
    else
      $display("RANDSTATE_MISMATCH");

    $finish;
  end
endmodule

// CHECK-NOT: RANDSTATE_EMPTY
// CHECK-NOT: RANDSTATE_MISMATCH
// CHECK: RANDSTATE_OK
