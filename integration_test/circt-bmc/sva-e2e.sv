// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module delay_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=DELAY-FAIL
// DELAY-FAIL: Assertion can be violated!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --ignore-asserts-until=1 --module delay_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=DELAY-PASS
// DELAY-PASS: Bound reached with no violations!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module range_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=RANGE-FAIL
// RANGE-FAIL: Assertion can be violated!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 4 --ignore-asserts-until=1 --module range_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=RANGE-PASS
// RANGE-PASS: Bound reached with no violations!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module unbounded_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=UNBOUNDED-FAIL
// UNBOUNDED-FAIL: Assertion can be violated!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 4 --ignore-asserts-until=1 --module unbounded_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=UNBOUNDED-PASS
// UNBOUNDED-PASS: Bound reached with no violations!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module cover_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=COVER-PASS
// COVER-PASS: Bound reached with no violations!

module delay_fail(input logic clk);
  // This should be violated once delay buffering is functional.
  assert property (@(posedge clk) 1'b1 |-> ##1 1'b0);
endmodule

module delay_pass(input logic clk, input logic b);
  // Constrain b high so ##1 b is always satisfied after warmup.
  assume property (@(posedge clk) b);
  assert property (@(posedge clk) b |-> ##1 b);
endmodule

module range_fail(input logic clk, input logic b);
  // Range delay uses ##[1:2] on a non-constant input.
  assert property (@(posedge clk) b |-> ##[1:2] b);
endmodule

module range_pass(input logic clk, input logic b);
  // Constrain b high so ##[1:2] b is always satisfied after warmup.
  assume property (@(posedge clk) b);
  assert property (@(posedge clk) b |-> ##[1:2] b);
endmodule

module unbounded_fail(input logic clk, input logic b);
  // Unbounded delay within the BMC bound should still find violations.
  assert property (@(posedge clk) b |-> ##[1:$] b);
endmodule

module unbounded_pass(input logic clk, input logic b);
  // Constrain b high so ##[1:$] b is satisfied within the bound after warmup.
  assume property (@(posedge clk) b);
  assert property (@(posedge clk) b |-> ##[1:$] b);
endmodule

module cover_pass(input logic clk, input logic b);
  // Cover should not trigger a violation.
  cover property (@(posedge clk) b);
endmodule
