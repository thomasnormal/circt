// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: covergroup sampling events must work when the covergroup handle
// is assigned via `cov = new;` after declaration (not only `cg cov = new;`).

// VERILOG-NOT: error

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  logic [3:0] addr = 0;
  logic we = 0;

  covergroup cg @(posedge clk);
    cp_addr: coverpoint addr;
    cp_we: coverpoint we;
  endgroup

  cg cov;

  initial begin
    cov = new;
    repeat (4) begin
      @(posedge clk);
      addr <= addr + 1;
      we <= ~we;
    end
    #1;
    $display("DONE");
    $finish;
  end
endmodule

// CHECK: DONE
// CHECK: Coverage Report
// CHECK: Covergroup: cg
// CHECK: cp_addr: {{[1-9][0-9]*}} hits
// CHECK: cp_we: {{[1-9][0-9]*}} hits
