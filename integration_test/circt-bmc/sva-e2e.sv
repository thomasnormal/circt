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
// RUN:   circt-bmc -b 4 --module repeat_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=REPEAT-FAIL
// REPEAT-FAIL: Assertion can be violated!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 5 --module repeat_range_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=REPEAT-RANGE-FAIL
// REPEAT-RANGE-FAIL: Assertion can be violated!

// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module cover_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=COVER-PASS
// COVER-PASS: Bound reached with no violations!
//
// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module nonoverlap_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=NONOVERLAP-FAIL
// NONOVERLAP-FAIL: Assertion can be violated!
//
// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --ignore-asserts-until=1 --module nonoverlap_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=NONOVERLAP-PASS
// NONOVERLAP-PASS: Bound reached with no violations!
//
// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module nonoverlap_disable_fail --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=NONOVERLAP-DISABLE-FAIL
// NONOVERLAP-DISABLE-FAIL: Assertion can be violated!
//
// RUN: circt-verilog --uvm-path %S/Inputs/uvm_stub --ir-hw %s | \
// RUN:   circt-bmc -b 3 --ignore-asserts-until=1 --module nonoverlap_disable_pass --shared-libs=%libz3 - | \
// RUN:   FileCheck %s --check-prefix=NONOVERLAP-DISABLE-PASS
// NONOVERLAP-DISABLE-PASS: Bound reached with no violations!

// TODO: Enable repeat_pass and repeat_range_pass once LTLToCore implication
// semantics for multi-cycle sequences are corrected.

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

module repeat_fail(input logic clk, input logic b);
  // Consecutive repeat should fail without constraining b.
  assert property (@(posedge clk) b |-> b[*3]);
endmodule

module repeat_pass(input logic clk, input logic b);
  // Constrain b high so b[*3] holds after warmup.
  assume property (@(posedge clk) b);
  assert property (@(posedge clk) b |-> b[*3]);
endmodule

module repeat_range_fail(input logic clk, input logic b);
  // Bounded repeat range should fail without constraining b.
  assert property (@(posedge clk) b |-> b[*2:3]);
endmodule

module repeat_range_pass(input logic clk, input logic b);
  // Constrain b high so b[*2:3] holds after warmup.
  assume property (@(posedge clk) b);
  assert property (@(posedge clk) b |-> b[*2:3]);
endmodule

module cover_pass(input logic clk, input logic b);
  // Cover should not trigger a violation.
  cover property (@(posedge clk) b);
endmodule

module nonoverlap_fail(input logic clk, input logic a);
  logic q;
  always_ff @(posedge clk)
    q <= 1'b0;
  assert property (@(posedge clk) a |=> q);
endmodule

module nonoverlap_pass(input logic clk, input logic a);
  logic q;
  always_ff @(posedge clk)
    q <= a;
  assert property (@(posedge clk) a |=> q);
endmodule

module nonoverlap_disable_fail(input logic clk, input logic reset, input logic a);
  logic q;
  always_ff @(posedge clk)
    q <= reset ? 1'b0 : 1'b0;
  assert property (@(posedge clk) disable iff (reset) a |=> q);
endmodule

module nonoverlap_disable_pass(input logic clk, input logic reset, input logic a);
  logic q;
  always_ff @(posedge clk)
    q <= reset ? 1'b0 : a;
  assert property (@(posedge clk) disable iff (reset) a |=> q);
endmodule
