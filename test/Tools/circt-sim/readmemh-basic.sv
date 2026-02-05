// RUN: mkdir -p %t.dir
// RUN: printf '0A 1B 2C 3D\n' > %t.dir/mem_data.hex
// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: cd %t.dir && circt-sim %t.mlir --top readmemh_test 2>&1 | FileCheck %s

// Test $readmemh runtime support: load hex values into a memory array
// and verify the contents via $display.

module readmemh_test;
  logic [7:0] mem [0:3];

  initial begin
    $readmemh("mem_data.hex", mem);
    // CHECK: mem[0] = 10
    $display("mem[0] = %0d", mem[0]);
    // CHECK: mem[1] = 27
    $display("mem[1] = %0d", mem[1]);
    // CHECK: mem[2] = 44
    $display("mem[2] = %0d", mem[2]);
    // CHECK: mem[3] = 61
    $display("mem[3] = %0d", mem[3]);
    // CHECK: PASS
    $display("PASS");
  end
endmodule
