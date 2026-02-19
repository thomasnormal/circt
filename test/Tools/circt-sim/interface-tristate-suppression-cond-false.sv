// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s

// Regression: mirror-store suppression for tri-state destination fields must
// engage when cond=0 selects known elseValue (high-Z), even if src is X.
// Otherwise source->dest mirror links can keep stale low values alive after
// release on shared inout wires.

interface tri_if(inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;

  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

module driver_a(tri_if intf);
  initial begin
    intf.s_oe = 1'b1;
    intf.s_o = 1'b0;
    #2;
    intf.s_o = 1'bx;
    intf.s_oe = 1'b0;
  end
endmodule

module monitor_b(tri_if intf);
  initial begin
    intf.s_oe = 1'b0;
    intf.s_o = 1'bx;

    #1;
    if (intf.s_i === 1'b0)
      $display("OBS_LOW_OK");
    else
      $display("OBS_LOW_FAIL:%b", intf.s_i);

    #3;
    if (intf.s_i === 1'b1)
      $display("OBS_RELEASE_HIGH_OK");
    else
      $display("OBS_RELEASE_HIGH_FAIL:%b", intf.s_i);

    $finish;
  end
endmodule

module top;
  wire S;
  pullup(S);

  tri_if if_a(S);
  tri_if if_b(S);

  driver_a da(if_a);
  monitor_b mb(if_b);

  // CHECK: OBS_LOW_OK
  // CHECK: OBS_RELEASE_HIGH_OK
  // CHECK-NOT: _FAIL:
endmodule
