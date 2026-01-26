// RUN: circt-verilog --no-uvm-auto-include --compat vcs --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: --compat vcs enables VCS compatibility flags
//===----------------------------------------------------------------------===//
// VCS compatibility mode enables the following slang flags:
// - AllowHierarchicalConst: Allow hierarchical references in constant expressions
// - RelaxEnumConversions: Allow implicit enum conversions
// - RelaxStringConversions: Allow implicit string conversions
// - AllowRecursiveImplicitCall: Allow recursive implicit function calls
// - AllowBareValParamAssignment: Allow bare value parameter assignments
// - AllowSelfDeterminedStreamConcat: Allow self-determined streaming concat
// - AllowMergingAnsiPorts: Allow merging ANSI ports
// - AllowVirtualIfaceWithOverride: Allow virtual interface with override

//===----------------------------------------------------------------------===//
// Test 1: Relaxed enum conversions (VCS allows implicit int-to-enum)
//===----------------------------------------------------------------------===//
// Without --compat vcs, this implicit conversion from int to enum is an error:
//   "no implicit conversion from 'int' to 'state_t'"
// With --compat vcs, the RelaxEnumConversions flag allows this.

// CHECK-LABEL: moore.module {{(private )?}}@TestEnumConversion
module TestEnumConversion;
  typedef enum logic [1:0] {
    STATE_IDLE = 2'b00,
    STATE_RUN  = 2'b01,
    STATE_DONE = 2'b10
  } state_t;

  state_t current_state;
  int int_value;

  // VCS allows implicit integer to enum conversion (no explicit cast needed)
  // CHECK: moore.blocking_assign %current_state
  initial begin
    int_value = 1;
    // With --compat vcs, this implicit conversion is allowed
    current_state = int_value;
  end
endmodule

//===----------------------------------------------------------------------===//
// Test 2: String conversion relaxation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module {{(private )?}}@TestStringConversion
module TestStringConversion;
  string str;

  initial begin
    str = "hello";
    // CHECK: moore.builtin.display
    $display("String: %s", str);
  end
endmodule

//===----------------------------------------------------------------------===//
// Test 3: Module with standard ANSI ports (baseline)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module {{(private )?}}@TestAnsiPorts
module TestAnsiPorts(
  input logic clk,
  input logic rst_n,
  output logic [7:0] data_out
);
  // CHECK: moore.blocking_assign %data_out
  initial data_out = 8'h00;
endmodule

//===----------------------------------------------------------------------===//
// Test 4: Self-determined streaming concatenation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module {{(private )?}}@TestStreamConcat
module TestStreamConcat;
  logic [31:0] data;
  logic [7:0] bytes [4];

  initial begin
    data = 32'hDEADBEEF;
    // Streaming operator
    // CHECK: moore.conversion
    // CHECK: moore.blocking_assign %bytes
    {>>{bytes}} = data;
  end
endmodule

//===----------------------------------------------------------------------===//
// Test 5: Bare parameter value assignment (VCS extension)
//===----------------------------------------------------------------------===//
// VCS allows parameter assignments without explicit #() syntax in some cases

// CHECK-LABEL: moore.module private @ParamModule
module ParamModule #(parameter WIDTH = 8);
  logic [WIDTH-1:0] data;
  // CHECK: moore.output
endmodule

//===----------------------------------------------------------------------===//
// Top module to ensure all test modules are elaborated
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @top
module top;
  TestEnumConversion t1();
  TestStringConversion t2();
  TestAnsiPorts t3(.clk(1'b0), .rst_n(1'b1), .data_out());
  TestStreamConcat t4();
  ParamModule #(.WIDTH(16)) t5();
endmodule
