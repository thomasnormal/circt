// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [3:0] vx;
  logic [3:0] vz;

  initial begin
    vx = 4'b10xz;
    vz = 4'bzzzz;

    $display("bvx<%b>", vx);
    $display("bvz<%b>", vz);
    $display("ovx<%o>", vx);
    $display("ovz<%o>", vz);
    $display("hvx<%h>", vx);
    $display("hvz<%h>", vz);
    $display("dvx<%d>", vx);
    $display("dvz<%d>", vz);
    $display("d04vx<%04d>", vx);
    $finish;
  end
endmodule

// CHECK: bvx<10xz>
// CHECK: bvz<zzzz>
// CHECK: ovx<1X>
// CHECK: ovz<zz>
// CHECK: hvx<X>
// CHECK: hvz<z>
// CHECK: dvx< X>
// CHECK: dvz< z>
// CHECK: d04vx<000X>
