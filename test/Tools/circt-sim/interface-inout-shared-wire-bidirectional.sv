// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s

// Regression for shared inout wires driven by multiple interface instances.
// Both instances use identical interface-body continuous assignments:
//   assign S = s_oe ? s_o : 1'bz;
//   assign s_i = S;
// Distinct-driver identity must include instance context; keying only by
// DriveOp pointer aliases sibling drivers and can drop reverse-direction
// propagation (e.g. target ACK not visible to controller).

interface tri_if(inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;

  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

module driver_a(tri_if intf);
  initial begin
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;
    #2;
    intf.s_o = 1'b0;
    intf.s_oe = 1'b1;
    #2;
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;
  end
endmodule

module driver_b(tri_if intf);
  initial begin
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;
    #6;
    intf.s_o = 1'b0;
    intf.s_oe = 1'b1;
    #2;
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;
  end
endmodule

module monitor(tri_if a, tri_if b);
  initial begin
    #3;
    if (b.s_i === 1'b0)
      $display("B_SEES_A_LOW_OK");
    else
      $display("B_SEES_A_LOW_FAIL:%b", b.s_i);

    #4;
    if (a.s_i === 1'b0)
      $display("A_SEES_B_LOW_OK");
    else
      $display("A_SEES_B_LOW_FAIL:%b", a.s_i);

    #3;
    if (a.s_i === 1'b1 && b.s_i === 1'b1)
      $display("BOTH_RELEASE_HIGH_OK");
    else
      $display("BOTH_RELEASE_HIGH_FAIL:a=%b b=%b", a.s_i, b.s_i);

    $finish;
  end
endmodule

module top;
  wire S;
  pullup(S);

  tri_if if_a(S);
  tri_if if_b(S);

  driver_a da(if_a);
  driver_b db(if_b);
  monitor m(if_a, if_b);

  // CHECK: B_SEES_A_LOW_OK
  // CHECK: A_SEES_B_LOW_OK
  // CHECK: BOTH_RELEASE_HIGH_OK
  // CHECK-NOT: _FAIL:
endmodule
