// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 --vcd %t.vcd > %t.out 2>&1
// RUN: /usr/bin/grep -q "Simulation completed" %t.out
// RUN: bash -eu -c '\
// RUN:   sig_id=$(/usr/bin/grep "__sva__cover_" %t.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.vcd; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.vcd'
//
// Runtime semantics: async accept_on(c) for clocked cover properties must
// observe pulses that happen between sampled property-clock edges.

module top;
  reg clk;
  reg b, c;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    b = 1'b0;
    c = 1'b0;

    @(posedge clk);      // cycle 1
    #1 c = 1'b1;         // async pulse between sampled edges
    #1 c = 1'b0;

    @(posedge clk);      // cycle 2
    @(posedge clk);      // cycle 3
    $finish;
  end

  cover property (@(posedge clk) accept_on(c) (1'b1 |-> ##1 b));
endmodule
