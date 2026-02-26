// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=500000000 --vcd %t.vcd > %t.out 2>&1
// RUN: bash -eu -c '\
// RUN:   sig_id=$(/usr/bin/grep "__sva__" %t.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.vcd; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.vcd'
//
// Runtime clocked assertions should surface as synthetic 1-bit VCD signals so
// waveform viewers can show pass/fail transitions.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1;
    @(posedge clk); // pass
    a = 0;
    @(posedge clk); // fail
    a = 1;
    @(posedge clk); // pass
    $finish;
  end

  assert property (@(posedge clk) a);
endmodule
