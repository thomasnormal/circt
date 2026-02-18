// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test associative array: exists, num, first, next, delete
module top;
  int aa[string];
  string key;
  int found;

  initial begin
    // Initially empty
    // CHECK: num_init=0
    $display("num_init=%0d", aa.num());

    // Insert elements
    aa["alpha"] = 1;
    aa["beta"] = 2;
    aa["gamma"] = 3;

    // CHECK: num_after=3
    $display("num_after=%0d", aa.num());

    // exists
    // CHECK: exists_alpha=1
    $display("exists_alpha=%0d", aa.exists("alpha"));
    // CHECK: exists_delta=0
    $display("exists_delta=%0d", aa.exists("delta"));

    // Read values
    // CHECK: val_beta=2
    $display("val_beta=%0d", aa["beta"]);

    // Delete one entry
    aa.delete("beta");
    // CHECK: num_after_del=2
    $display("num_after_del=%0d", aa.num());
    // CHECK: exists_beta=0
    $display("exists_beta=%0d", aa.exists("beta"));

    // first/next iteration
    found = aa.first(key);
    // CHECK: first_found=1
    $display("first_found=%0d", found);

    found = aa.next(key);
    // CHECK: next_found=1
    $display("next_found=%0d", found);

    found = aa.next(key);
    // No more keys
    // CHECK: next_end=0
    $display("next_end=%0d", found);

    // delete all
    aa.delete();
    // CHECK: num_final=0
    $display("num_final=%0d", aa.num());

    $finish;
  end
endmodule
