// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top fork_event_tb 2>&1 | FileCheck %s

// Test: wait_event inside fork should compile and simulate correctly.
// Previously, this failed with:
//   error: 'llhd.wait' op expects parent op 'llhd.process'
// because WaitEventOpConversion always generated llhd.wait, but inside
// a fork branch the immediate parent is sim.fork, not llhd.process.
// The fix generates __moore_wait_event runtime calls in fork context.

// CHECK: [circt-sim] Simulation completed
module fork_event_tb();
  event ev;
  reg [3:0] a = 0;
  initial fork
    begin
      a = 4'h3;
      #20;
      ->ev;
    end
    begin
      @ev
      a = 4'h4;
    end
  join
endmodule
