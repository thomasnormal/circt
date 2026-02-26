// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=500000000 --vcd %t.vcd > %t.out 2>&1
// RUN: bash -eu -c '\
// RUN:   sig_id=$(/usr/bin/grep "__sva__a_must_hold" %t.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   ! /usr/bin/grep -q "__sva__assert_" %t.vcd; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.vcd; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.vcd'
//
// Labeled concurrent assertions should use the assertion label in synthetic
// VCD signal names.

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

  a_must_hold: assert property (@(posedge clk) a);
endmodule
