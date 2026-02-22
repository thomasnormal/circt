// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: env CIRCT_SIM_TRACE_IFACE_STORE=1 circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s

// Regression: mirrored probe-copy stores into passive tri-state interface
// fields must not be suppressed when the local tri-state rule is released (Z).
// Suppressing those writes can freeze passive observation and break monitor
// progression in dual-interface bus topologies.

interface tri_if(inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;
  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

class active_drv;
  virtual tri_if vif;
  function new(virtual tri_if vif);
    this.vif = vif;
  endfunction
  task run();
    vif.s_o = 1'b1;
    vif.s_oe = 1'b0;
    #2;
    repeat (4) begin
      vif.s_o = 1'b0;
      vif.s_oe = 1'b1;
      #2;
      vif.s_o = 1'b1;
      vif.s_oe = 1'b0;
      #2;
    end
  endtask
endclass

class passive_mon;
  virtual tri_if vif;
  function new(virtual tri_if vif);
    this.vif = vif;
  endfunction
  task run();
    int i;
    #3;
    for (i = 0; i < 4; i++) begin
      if (vif.s_i === 1'b0)
        $display("PASSIVE_LOW_%0d_OK", i);
      else
        $display("PASSIVE_LOW_%0d_FAIL:%b", i, vif.s_i);
      #2;
      if (vif.s_i === 1'b1)
        $display("PASSIVE_HIGH_%0d_OK", i);
      else
        $display("PASSIVE_HIGH_%0d_FAIL:%b", i, vif.s_i);
      #2;
    end
  endtask
endclass

module top;
  wire S;
  pullup(S);

  tri_if if_active(S);
  tri_if if_passive(S);

  active_drv drv;
  passive_mon mon;

  initial begin
    if_passive.s_o = 1'bx;
    if_passive.s_oe = 1'b0;
    drv = new(if_active);
    mon = new(if_passive);
    fork
      drv.run();
      mon.run();
    join
    $finish;
  end

  // CHECK: [IFACE-STORE]{{.*}}sig_1.field_0{{.*}}copySrc=
  // CHECK: PASSIVE_LOW_0_OK
  // CHECK-NOT: [IFACE-STORE]{{.*}}sig_1.field_0{{.*}}suppressed=1
  // CHECK: PASSIVE_HIGH_0_OK
  // CHECK: PASSIVE_LOW_1_OK
  // CHECK: PASSIVE_HIGH_1_OK
  // CHECK: PASSIVE_LOW_2_OK
  // CHECK: PASSIVE_HIGH_2_OK
  // CHECK: PASSIVE_LOW_3_OK
  // CHECK: PASSIVE_HIGH_3_OK
  // CHECK-NOT: _FAIL:
endmodule
