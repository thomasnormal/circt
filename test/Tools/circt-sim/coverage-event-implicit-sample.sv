// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top cov_intro 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: declaration-time covergroup construction with an implicit sampling
// event (`cg cov = new;` + `covergroup cg @(...)`) must synthesize sampling.

// VERILOG-NOT: error

module cov_intro;
  logic clk = 0;
  always #5 clk = ~clk;

  logic       we   = 0;
  logic [3:0] addr = 0;
  logic [7:0] wdata = 0, rdata;

  sram dut(.clk, .we, .addr, .wdata, .rdata);

  covergroup sram_cg @(posedge clk);
    cp_addr: coverpoint addr;
    cp_we:   coverpoint we;
  endgroup

  sram_cg cov = new;

  initial begin
    repeat (20) begin
      @(posedge clk);
      we   <= $random;
      addr <= $random;
      wdata <= $random;
    end
    #1;
    $display("DONE");
    $finish;
  end
endmodule

module sram #(parameter int DEPTH = 16, parameter int WIDTH = 8) (
  input  logic                      clk,
  input  logic                      we,
  input  logic [$clog2(DEPTH)-1:0] addr,
  input  logic [WIDTH-1:0]         wdata,
  output logic [WIDTH-1:0]         rdata
);
  logic [WIDTH-1:0] mem [0:DEPTH-1];
  always_ff @(posedge clk) begin
    if (we) mem[addr] <= wdata;
    rdata <= mem[addr];
  end
endmodule

// CHECK: DONE
// CHECK: Coverage Report
// CHECK: Covergroup: sram_cg
// CHECK: cp_addr: {{[1-9][0-9]*}} hits
// CHECK: cp_we: {{[1-9][0-9]*}} hits
