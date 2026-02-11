// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 8 --allow-multi-clock --assume-known-inputs --ignore-asserts-until=0 --module=sva_multiclock_assume_domains_sat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 8 --allow-multi-clock --assume-known-inputs --ignore-asserts-until=0 --module=sva_multiclock_assume_domains_sat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_multiclock_assume_domains_sat(
    input logic clk_a, input logic clk_b, input logic req, input logic ack);
  assume property (@(posedge clk_a) req);
  assume property (@(posedge clk_b) !ack);

  assert property (@(posedge clk_a) req);
  assert property (@(posedge clk_b) ack);
endmodule

// JIT: BMC_RESULT=SAT
// SMTLIB: BMC_RESULT=SAT
