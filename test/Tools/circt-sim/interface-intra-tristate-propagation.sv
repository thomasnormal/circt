// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s

// Test that interface-internal tri-state wiring remains reactive at runtime.
// The interface computes scl from (scl_oen ? scl_o : z) and mirrors scl_i=scl.
// This catches cases where these interface internal assigns were only executed
// once at init and not updated after later field stores.

interface i3c_like_if;
  logic scl;
  logic scl_i;
  logic scl_o;
  logic scl_oen;

  assign scl = scl_oen ? scl_o : 1'bz;
  assign scl_i = scl;
endinterface

module driver(i3c_like_if vif);
  initial begin
    vif.scl_o = 1'b0;
    vif.scl_oen = 1'b0;

    #1;
    vif.scl_o = 1'b1;
    vif.scl_oen = 1'b1;

    #1;
    vif.scl_o = 1'b0;
    vif.scl_oen = 1'b1;

    #1;
    vif.scl_o = 1'b1;
    vif.scl_oen = 1'b0;
  end
endmodule

module monitor(i3c_like_if vif);
  initial begin
    #2;
    if (vif.scl_i === 1'b1)
      $display("MON_OK_DRIVEN_HIGH");
    else
      $display("MON_FAIL_DRIVEN_HIGH:%b", vif.scl_i);

    #1;
    if (vif.scl_i === 1'b0)
      $display("MON_OK_DRIVEN_LOW");
    else
      $display("MON_FAIL_DRIVEN_LOW:%b", vif.scl_i);

    #1;
    if (vif.scl_i === 1'b1)
      $display("MON_FAIL_RELEASED_STILL_HIGH");
    else
      $display("MON_OK_RELEASED_NOT_HIGH");

    $finish;
  end
endmodule

module top;
  i3c_like_if bus();
  driver d(bus);
  monitor m(bus);

  // CHECK: MON_OK_DRIVEN_HIGH
  // CHECK: MON_OK_DRIVEN_LOW
  // CHECK: MON_OK_RELEASED_NOT_HIGH
  // CHECK-NOT: MON_FAIL_
endmodule
