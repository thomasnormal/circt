// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test dynamic array: new, size, delete, assignment
module top;
  int dyn[];
  int dyn2[];

  initial begin
    // Initially empty
    // CHECK: size_init=0
    $display("size_init=%0d", dyn.size());

    // Allocate
    dyn = new[5];
    // CHECK: size_alloc=5
    $display("size_alloc=%0d", dyn.size());

    // Write and read
    dyn[0] = 10;
    dyn[1] = 20;
    dyn[2] = 30;
    dyn[3] = 40;
    dyn[4] = 50;

    // CHECK: dyn0=10
    $display("dyn0=%0d", dyn[0]);
    // CHECK: dyn4=50
    $display("dyn4=%0d", dyn[4]);

    // Copy
    dyn2 = dyn;
    // CHECK: copy_size=5
    $display("copy_size=%0d", dyn2.size());
    // CHECK: copy_val=30
    $display("copy_val=%0d", dyn2[2]);

    // Resize with new — note: preserving old elements may not work in all impls
    dyn = new[3](dyn);
    // CHECK: resize_size=3
    $display("resize_size=%0d", dyn.size());
    // Preserved value: dyn[0] should be 10 from original
    // CHECK: resize_val=10
    $display("resize_val=%0d", dyn[0]);

    // Delete
    dyn.delete();
    // CHECK: size_delete=0
    $display("size_delete=%0d", dyn.size());

    $finish;
  end
endmodule
