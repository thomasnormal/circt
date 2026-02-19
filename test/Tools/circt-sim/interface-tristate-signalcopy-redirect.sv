// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s

// Regression: signal-copy propagation from shared inout nets must not update
// tri-state destination fields directly. Otherwise a passive interface instance
// can latch a sampled low and keep driving the bus low after release.

interface tri_if(inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;

  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

module active_driver(tri_if intf);
  initial begin
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;

    #2;
    repeat (6) begin
      intf.s_o = 1'b0;
      intf.s_oe = 1'b1;
      #1;
      intf.s_o = 1'b1;
      intf.s_oe = 1'b0;
      #2;
    end
  end
endmodule

module passive_agent(tri_if intf);
  initial begin
    intf.s_o = 1'bx;
    intf.s_oe = 1'b0;
  end
endmodule

module monitor(tri_if passive_if, input logic S);
  integer i;
  initial begin
    #4;
    for (i = 0; i < 5; i = i + 1) begin
      if (S === 1'b1 && passive_if.s_i === 1'b1)
        $display("PULSE_%0d_OK", i);
      else
        $display("PULSE_%0d_FAIL:S=%b s_i=%b", i, S, passive_if.s_i);
      #3;
    end
    $finish;
  end
endmodule

module top;
  wire S;
  pullup(S);

  tri_if if_active(S);
  tri_if if_passive(S);

  active_driver a(if_active);
  passive_agent p(if_passive);
  monitor m(if_passive, S);

  // CHECK: PULSE_0_OK
  // CHECK: PULSE_1_OK
  // CHECK: PULSE_2_OK
  // CHECK: PULSE_3_OK
  // CHECK: PULSE_4_OK
  // CHECK-NOT: _FAIL:
endmodule
