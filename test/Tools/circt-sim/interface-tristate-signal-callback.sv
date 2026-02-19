// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s

// Regression for interface tri-state rule re-evaluation on signal callbacks.
// The interface source field `s_o` is driven from a top-level signal via
// continuous assignment (not llvm.store). When `drive` toggles, runtime
// signal callbacks must re-run tri-state rules so `S`/`s_i` update.

interface tri_if(inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;

  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

module top;
  logic drive;
  wire S;

  pullup(S);
  tri_if bus(S);

  assign bus.s_o = drive;

  initial begin
    bus.s_oe = 1'b1;
    drive = 1'b1;
    #1;
    if (bus.s_i === 1'b1)
      $display("OBS_HIGH_OK");
    else
      $display("OBS_HIGH_FAIL:%b", bus.s_i);

    drive = 1'b0;
    #1;
    if (bus.s_i === 1'b0)
      $display("OBS_LOW_OK");
    else
      $display("OBS_LOW_FAIL:%b", bus.s_i);

    $finish;
  end

  // CHECK: OBS_HIGH_OK
  // CHECK: OBS_LOW_OK
  // CHECK-NOT: OBS_{{.*}}_FAIL:
endmodule
