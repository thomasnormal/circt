// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s --check-prefix=PASS
// RUN: circt-verilog --no-uvm-auto-include --ir-hw -DFAIL %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s --check-prefix=FAIL
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3
// XFAIL: *

// This is a minimized parity lock for the Yosys SVA `counter.sv` case.
// Expected behavior: pass profile is UNSAT, fail profile is SAT.

module top (input clk, reset, up, down, output reg [7:0] cnt);
  always @(posedge clk) begin
    if (reset)
      cnt <= 0;
    else if (up)
      cnt <= cnt + 1;
    else if (down)
      cnt <= cnt - 1;
  end

  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  assert property (up |=> cnt == $past(cnt) + 8'd 1);
  assert property (up [*2] |=> cnt == $past(cnt, 2) + 8'd 2);
  assert property (up ##1 up |=> cnt == $past(cnt, 2) + 8'd 2);

`ifndef FAIL
  assume property (down |-> !up);
`endif

  assert property (up ##1 down |=> cnt == $past(cnt, 2));
  assert property (down |=> cnt == $past(cnt) - 8'd 1);

  property down_n(n);
    down [*n] |=> cnt == $past(cnt, n) - n;
  endproperty

  assert property (down_n(8'd 3));
  assert property (down_n(8'd 5));
endmodule

// PASS: BMC_RESULT=UNSAT
// FAIL: BMC_RESULT=SAT
