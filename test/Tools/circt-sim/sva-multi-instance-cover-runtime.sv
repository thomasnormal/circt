// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=60000000 --vcd %t.vcd > %t.out 2>&1
// RUN: /usr/bin/grep -q "Simulation completed" %t.out
// RUN: bash -eu -c '\
// RUN:   ids=$(/usr/bin/grep "__sva__inst.*__sva__cover_" %t.vcd | /usr/bin/awk '\''{print $4}'\''); \
// RUN:   test -n "$ids"; \
// RUN:   count=$(/usr/bin/echo "$ids" | /usr/bin/wc -w); \
// RUN:   test "$count" -ge 2; \
// RUN:   ones=0; zeros_only=0; \
// RUN:   for id in $ids; do \
// RUN:     if /usr/bin/grep -q "1$id" %t.vcd; then \
// RUN:       ones=$((ones + 1)); \
// RUN:     else \
// RUN:       zeros_only=$((zeros_only + 1)); \
// RUN:     fi; \
// RUN:   done; \
// RUN:   test "$ones" -ge 1; \
// RUN:   test "$zeros_only" -ge 1'

// Regression: clocked cover runtime state must be per-instance.
// One instance covers, the other does not; VCD should show distinct outcomes.

module child(input logic clk, input logic a);
  cover property (@(posedge clk) a);
endmodule

module top;
  logic clk;
  logic bad;
  logic good;

  child u_bad(.clk(clk), .a(bad));
  child u_good(.clk(clk), .a(good));

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    bad = 1'b0;
    good = 1'b1;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end
endmodule
