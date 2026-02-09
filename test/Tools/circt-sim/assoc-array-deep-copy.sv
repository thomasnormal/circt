// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that associative array whole-assignment creates a deep copy.
// In SystemVerilog (IEEE 1800-2017 Section 7.9), assigning one associative
// array to another copies all entries.  Modifications to one array must
// not affect the other after the copy.

module top;
  int aa1[string];
  int aa2[string];

  initial begin
    // Populate aa1.
    aa1["x"] = 1;
    aa1["y"] = 2;
    aa1["z"] = 3;

    // Deep-copy aa1 into aa2.
    aa2 = aa1;

    // Verify aa2 has all entries from aa1.
    // CHECK: aa2[x]=1
    $display("aa2[x]=%0d", aa2["x"]);
    // CHECK: aa2[y]=2
    $display("aa2[y]=%0d", aa2["y"]);
    // CHECK: aa2[z]=3
    $display("aa2[z]=%0d", aa2["z"]);
    // CHECK: aa2_size=3
    $display("aa2_size=%0d", aa2.size());

    // Delete all entries from aa1.
    aa1.delete();
    // CHECK: aa1_size=0
    $display("aa1_size=%0d", aa1.size());

    // aa2 must still have all 3 entries (deep copy, not shared pointer).
    // CHECK: aa2_after_delete_size=3
    $display("aa2_after_delete_size=%0d", aa2.size());
    // CHECK: aa2_after[x]=1
    $display("aa2_after[x]=%0d", aa2["x"]);
    // CHECK: aa2_after[y]=2
    $display("aa2_after[y]=%0d", aa2["y"]);
    // CHECK: aa2_after[z]=3
    $display("aa2_after[z]=%0d", aa2["z"]);

    // Modify aa2 and verify aa1 is unaffected (already empty).
    aa2["w"] = 99;
    // CHECK: aa2_final_size=4
    $display("aa2_final_size=%0d", aa2.size());

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
