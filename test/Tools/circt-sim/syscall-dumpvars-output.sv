// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that $dumpfile/$dumpvars produce VCD output or at minimum a diagnostic.
// Bug: $dumpfile/$dumpvars are completely silent no-ops — no file is created,
// no VCD content is output, and no warning is emitted.
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
  // When properly implemented, the output should contain VCD content
  // or at least reference the dump file. Currently it's totally silent.
  // CHECK: vcd_test_complete
  // CHECK: {{[Ww]rote.*[Vv][Cc][Dd]|[Dd]ump|\.vcd}}
endmodule
