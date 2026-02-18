// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test associative array methods: size, num, exists, delete, first, last, next, prev
module top;
  int aa[string];
  string key;
  int found;

  initial begin
    aa["alpha"] = 1;
    aa["beta"] = 2;
    aa["gamma"] = 3;

    // CHECK: aa_size=3
    $display("aa_size=%0d", aa.size());

    // CHECK: aa_num=3
    $display("aa_num=%0d", aa.num());

    // exists — returns nonzero if key exists
    // CHECK: exists_alpha_nz=1
    $display("exists_alpha_nz=%0d", aa.exists("alpha") != 0);
    // CHECK: exists_delta=0
    $display("exists_delta=%0d", aa.exists("delta"));

    // first — returns nonzero if aa is non-empty
    found = aa.first(key);
    // CHECK: first_found_nz=1
    $display("first_found_nz=%0d", found != 0);

    // next
    found = aa.next(key);
    // CHECK: next_found_nz=1
    $display("next_found_nz=%0d", found != 0);

    // last
    found = aa.last(key);
    // CHECK: last_found_nz=1
    $display("last_found_nz=%0d", found != 0);

    // prev
    found = aa.prev(key);
    // CHECK: prev_found_nz=1
    $display("prev_found_nz=%0d", found != 0);

    // delete specific key
    aa.delete("beta");
    // CHECK: after_delete=2
    $display("after_delete=%0d", aa.size());

    // delete all
    aa.delete();
    // CHECK: after_delete_all=0
    $display("after_delete_all=%0d", aa.size());

    $finish;
  end
endmodule
