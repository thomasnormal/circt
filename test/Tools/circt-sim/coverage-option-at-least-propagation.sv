// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: coverpoint options (option.at_least) must be propagated to
// runtime coverage accounting.

// VERILOG-NOT: error

module top;
  int x;

  covergroup cg;
    cp: coverpoint x {
      option.at_least = 2;
      bins only = {7};
    }
  endgroup

  initial begin
    static cg cg_inst = new();

    x = 7;
    cg_inst.sample();  // One hit, below at_least=2.

    // Expected 0% because the only bin has 1 hit and requires 2.
    // CHECK: inst_cov=0
    $display("inst_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $finish;
  end
endmodule
