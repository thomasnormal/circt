// RUN: circt-verilog --ir-hw %s 2>&1 | FileCheck %s
// XFAIL: *

// Test that interface tasks with timing controls (@posedge, @negedge) are
// properly converted after inlining. The MooreToCore pass leaves these
// unconverted when they're in func.func, and they get converted after
// inlining into llhd.process.
// FIXME: moore.wait_event/detect_event inside func.func are not yet lowered.

// CHECK-LABEL: hw.module @top
// CHECK: llhd.process
// CHECK: llhd.wait
// CHECK-NOT: moore.wait_event
// CHECK-NOT: moore.detect_event

interface driver_bfm (
  input bit clk,
  input bit reset_n,
  output bit enable
);

  task wait_for_reset();
    @(negedge reset_n);
    @(posedge reset_n);
  endtask

  task drive_enable();
    @(posedge clk);
    enable <= 1'b1;
  endtask

endinterface

module top;
  bit clk;
  bit reset_n;
  bit enable;

  driver_bfm bfm (
    .clk(clk),
    .reset_n(reset_n),
    .enable(enable)
  );

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    reset_n = 0;
    #10 reset_n = 1;
    bfm.wait_for_reset();
    bfm.drive_enable();
    $finish;
  end

endmodule
