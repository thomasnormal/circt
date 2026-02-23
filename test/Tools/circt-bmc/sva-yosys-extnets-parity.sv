// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s --check-prefix=PASS
// RUN: circt-verilog --no-uvm-auto-include --ir-hw -DFAIL %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s --check-prefix=FAIL
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

// This is a parity lock for the Yosys SVA `extnets.sv` case.
// Expected behavior: pass profile is UNSAT, fail profile is SAT.

module top(input i, output o);
  A A();
  B B();
  assign A.i = i;
  assign o = B.o;
  always @* assert(o == i);
endmodule

module A;
  wire i, y;
`ifdef FAIL
  assign B.x = i;
`else
  assign B.x = !i;
`endif
  assign y = !B.y;
endmodule

module B;
  wire x, y, o;
  assign y = x, o = A.y;
endmodule

// PASS: BMC_RESULT=UNSAT
// FAIL: BMC_RESULT=SAT
