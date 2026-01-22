// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test $readmemb and $readmemh (IEEE 1800-2017 Section 21.4)

module readmem_test;
  logic [31:0] mem_b [0:15];
  logic [31:0] mem_h [0:15];
  string fname_b = "test_b.mem";
  string fname_h = "test_h.mem";

  initial begin
    // Test with string literal filename
    $readmemb("data_b.mem", mem_b);
    $readmemh("data_h.mem", mem_h);

    // Test with string variable filename
    $readmemb(fname_b, mem_b);
    $readmemh(fname_h, mem_h);
  end
endmodule

// CHECK: moore.module @readmem_test
// CHECK: moore.builtin.readmemb
// CHECK: moore.builtin.readmemh
// CHECK: moore.builtin.readmemb
// CHECK: moore.builtin.readmemh
