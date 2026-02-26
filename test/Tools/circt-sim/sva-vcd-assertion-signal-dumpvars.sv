// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: cd %t.dir && not circt-sim %t.mlir --top top --max-time=500000000 > %t.out 2>&1
// RUN: bash -eu -c '\
// RUN:   test -f %t.dir/__sva_runtime_dumpvars__.vcd; \
// RUN:   /usr/bin/grep -q "__sva__" %t.dir/__sva_runtime_dumpvars__.vcd; \
// RUN:   sig_id=$(/usr/bin/grep "__sva__" %t.dir/__sva_runtime_dumpvars__.vcd | /usr/bin/head -n1 | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_id"; \
// RUN:   /usr/bin/grep -q "1$sig_id" %t.dir/__sva_runtime_dumpvars__.vcd; \
// RUN:   /usr/bin/grep -q "0$sig_id" %t.dir/__sva_runtime_dumpvars__.vcd'
//
// Runtime SVA status signals should be present when VCD is opened via
// $dumpfile/$dumpvars (without --vcd CLI).

module top;
  reg clk;
  reg a;

  initial begin
    $dumpfile("__sva_runtime_dumpvars__.vcd");
    $dumpvars(0, top);

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
