// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 --vcd %t.vcd > %t.out 2>&1
// RUN: /usr/bin/grep -q "Simulation completed" %t.out
// RUN: bash -eu -c '\
// RUN:   sig_id=$(/usr/bin/grep "__sva__cover_" %t.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.vcd; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.vcd'
//
// Runtime clocked covers should be surfaced as synthetic 1-bit VCD signals
// (`__sva__cover_*`) so waveform viewers can show coverage hits over time.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 0;
    @(posedge clk); // miss
    a = 1;
    @(posedge clk); // hit
    a = 0;
    @(posedge clk);
    $finish;
  end

  cover property (@(posedge clk) a);
endmodule
