// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=500000000 --vcd %t.vcd --trace clk > %t.out 2>&1
// RUN: bash -eu -c '\
// RUN:   /usr/bin/grep -q "__sva__" %t.vcd; \
// RUN:   sig_id=$(/usr/bin/grep "__sva__" %t.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   /usr/bin/grep -q '\'' clk $end'\'' %t.vcd; \
// RUN:   ! /usr/bin/grep -q '\'' a $end'\'' %t.vcd; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.vcd; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.vcd'
//
// Runtime clocked assertions should still be traced even when --trace filters
// regular signals.

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
