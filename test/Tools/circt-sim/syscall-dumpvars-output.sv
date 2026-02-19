// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that $dumpfile/$dumpvars produce VCD output.
// Bug: $dumpfile/$dumpvars are completely silent no-ops — no file is created,
// no VCD content is output, and no diagnostic is emitted.
// IEEE 1800-2017 Section 21.7: $dumpfile opens a VCD file and
// $dumpvars selects variables to dump.
module top;
  reg [7:0] val;

  initial begin
    $dumpfile("syscall_dumpvars_test.vcd");
    $dumpvars(0, top);
    val = 8'hAA;
    #1;
    val = 8'h55;
    #1;
    $display("vcd_test_complete");
    $finish;
  end
  // The simulator emits a diagnostic with the filename from $dumpfile.
  // CHECK: syscall_dumpvars_test.vcd
  // CHECK: vcd_test_complete
endmodule
