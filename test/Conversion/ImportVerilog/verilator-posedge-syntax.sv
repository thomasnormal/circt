// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test Verilator-style @posedge (clk) syntax extension
// Verilator accepts @posedge (clk) while strict IEEE 1800 requires @(posedge clk)
// This test verifies that slang now accepts the Verilator-style syntax.

module VerilatorPosedgeSyntax(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @VerilatorPosedgeSyntax

  //===--------------------------------------------------------------------===//
  // Test @posedge (clk) in sequence declaration
  //===--------------------------------------------------------------------===//
  sequence write_seq;
    @posedge (clk)
    (a & b);
  endsequence

  //===--------------------------------------------------------------------===//
  // Test @negedge (clk) in sequence declaration
  //===--------------------------------------------------------------------===//
  sequence read_seq;
    @negedge (clk)
    a;
  endsequence

  //===--------------------------------------------------------------------===//
  // Test @posedge (clk) in property declaration
  //===--------------------------------------------------------------------===//
  property posedge_prop;
    @posedge (clk)
    a |-> b;
  endproperty
  assert property (posedge_prop);

  //===--------------------------------------------------------------------===//
  // Test @negedge (clk) in property declaration
  //===--------------------------------------------------------------------===//
  property negedge_prop;
    @negedge (clk)
    a |=> b;
  endproperty
  assert property (negedge_prop);

  //===--------------------------------------------------------------------===//
  // Test @edge (clk) in property declaration
  //===--------------------------------------------------------------------===//
  property any_edge_prop;
    @edge (clk)
    a |-> ##1 b;
  endproperty
  assert property (any_edge_prop);

  //===--------------------------------------------------------------------===//
  // Test @posedge (clk) in always block
  //===--------------------------------------------------------------------===//
  always @posedge (clk) begin
    if (rst) begin
      // nothing
    end
  end

  //===--------------------------------------------------------------------===//
  // Test standard syntax still works (control case)
  //===--------------------------------------------------------------------===//
  property standard_syntax;
    @(posedge clk)
    a |-> b;
  endproperty
  assert property (standard_syntax);

  // CHECK-DAG: ltl.clock {{.*}}, posedge
  // CHECK-DAG: ltl.clock {{.*}}, negedge
  // CHECK-DAG: ltl.clock {{.*}}, edge
  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.detect_event posedge
  // CHECK-DAG: verif.assert

endmodule
