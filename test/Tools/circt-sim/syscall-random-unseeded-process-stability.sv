// RUN: split-file %s %t
// RUN: circt-verilog %t/base.sv --ir-llhd --no-uvm-auto-include -o %t/base.mlir 2>&1
// RUN: circt-verilog %t/with_observer.sv --ir-llhd --no-uvm-auto-include -o %t/obs.mlir 2>&1
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %t/base.mlir --top top 2>&1 | FileCheck %s --check-prefix=BASE
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %t/obs.mlir --top top 2>&1 | FileCheck %s --check-prefix=OBS
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %t/base.mlir --top top 2>&1 | grep '^A=' > %t/base.out
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %t/obs.mlir --top top 2>&1 | grep '^A=' > %t/obs.out
// RUN: diff %t/base.out %t/obs.out
//
// Unseeded $random should be stable for a process even when unrelated
// non-random processes are added.

//--- base.sv
module top;
  bit clk = 0;
  always #1 clk = ~clk;

  integer a;
  initial begin
    @(posedge clk);
    a = $random;
    // BASE: A={{-?[0-9]+}}
    $display("A=%0d", a);
    $finish;
  end
endmodule

//--- with_observer.sv
module top;
  bit clk = 0;
  always #1 clk = ~clk;

  // Passive observer process: does not touch RNG state.
  initial begin
    @(posedge clk);
    // OBS: OBS
    $display("OBS");
  end

  integer a;
  initial begin
    @(posedge clk);
    a = $random;
    // OBS: A={{-?[0-9]+}}
    $display("A=%0d", a);
    $finish;
  end
endmodule
