// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test force/release on a wire driven by continuous assignment.
// Bug: force is a blocking assignment, release is a no-op.
// IEEE 1800-2017 Section 10.6: force on a net overrides its drivers;
// release allows the continuous assignment to drive again.
module top;
  reg [7:0] driver;
  wire [7:0] w;
  assign w = driver;

  initial begin
    // Drive via continuous assignment
    driver = 50;
    #1;
    // CHECK: wire_driven=50
    $display("wire_driven=%0d", w);

    // Force the wire to a different value
    force w = 222;
    #1;
    // CHECK: wire_forced=222
    $display("wire_forced=%0d", w);

    // Change the driver while forced — wire should still show forced value
    driver = 100;
    #1;
    // CHECK: wire_still_forced=222
    $display("wire_still_forced=%0d", w);

    // Release — wire should now reflect the driver (100)
    release w;
    #1;
    // CHECK: wire_released=100
    $display("wire_released=%0d", w);

    $finish;
  end
endmodule
