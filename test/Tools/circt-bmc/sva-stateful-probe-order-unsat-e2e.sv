// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 8 --ignore-asserts-until=0 --module=sva_stateful_probe_order_unsat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 8 --ignore-asserts-until=0 --print-counterexample --module=sva_stateful_probe_order_unsat - | \
// RUN:   FileCheck %s --check-prefix=PRINTCE
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 8 --ignore-asserts-until=0 --module=sva_stateful_probe_order_unsat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

// Regression for stateful LLHD probe ordering:
// A probe that appears before the textual drive must observe the stateful
// current value (not the signal init), otherwise $changed and named-property
// checks become spuriously SAT.
module sva_stateful_probe_order_unsat(input logic clk);
  int cyc = 0;
  logic val = 0;

  always @(posedge clk) begin
    cyc <= cyc + 1;
    val = ~val;
  end

  assert property (@(posedge clk) cyc == 0 || $changed(val));

  property check(cyc_mod_2, expected);
    @(posedge clk) cyc % 2 == cyc_mod_2 |=> val == expected;
  endproperty
  assert property (check(0, 1));
  assert property (check(1, 0));
endmodule

// JIT: BMC_RESULT=UNSAT
// PRINTCE: BMC_RESULT=UNSAT
// SMTLIB: BMC_RESULT=UNSAT
