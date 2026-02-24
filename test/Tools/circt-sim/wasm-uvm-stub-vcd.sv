// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top wasm_uvm_stub_tb --max-time=50000 --vcd %t.vcd 2>&1 | FileCheck %s
// RUN: FileCheck %s --check-prefix=VCD < %t.vcd
//
// CHECK: uvm stub tb start
// CHECK: [circt-sim] Wrote waveform to
//
// VCD: $var wire {{[0-9]+}} {{.*}} sig $end
// VCD: $enddefinitions $end

// Minimal UVM-stub testbench for wasm frontend+sim VCD checks.
`include "uvm_macros.svh"

module wasm_uvm_stub_tb();
  import uvm_pkg::*;

  logic clk = 1'b0;
  logic sig = 1'b0;

  always #5 clk = ~clk;

  initial begin
    $display("uvm stub tb start");
    #7 sig = 1'b1;
    #10 sig = 1'b0;
    #10 $finish;
  end
endmodule
