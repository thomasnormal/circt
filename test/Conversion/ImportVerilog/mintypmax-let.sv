// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefixes=CHECK,TYP
// RUN: circt-verilog --no-uvm-auto-include --ir-moore --timing=min %s | FileCheck %s --check-prefixes=CHECK,MIN
// RUN: circt-verilog --no-uvm-auto-include --ir-moore --timing=max %s | FileCheck %s --check-prefixes=CHECK,MAX
// REQUIRES: slang

// Test for MinTypMax expression support (IEEE 1800-2017 Section 11.11)
// and let construct support (IEEE 1800-2017 Section 11.12)

// CHECK-LABEL: moore.module @MinTypMaxDelay
module MinTypMaxDelay;
  // MinTypMax expressions are used for delay specifications with min:typ:max values.
  // We evaluate to the selected value (typically 'typ' by default).
  // TYP: moore.constant_time 200000000 fs
  // MIN: moore.constant_time 100000000 fs
  // MAX: moore.constant_time 300000000 fs
  initial begin
    #(100:200:300) $display("Done");
  end
endmodule

// CHECK-LABEL: moore.module @LetConstruct
module LetConstruct;
  // Let declarations define expression macros that are expanded inline when used.
  // CHECK: %a = moore.variable
  // CHECK: %b = moore.variable
  // CHECK: %c = moore.variable
  // CHECK: %d = moore.variable
  logic [3:0] a = 12;
  logic [3:0] b = 15;
  logic [3:0] c = 7;
  logic d;

  // The let construct - this is just a declaration, no IR generated for it
  let op(x, y, z) = |((x | y) & z);

  // When the let is used, it's expanded inline
  // CHECK: moore.procedure initial
  initial begin
    d = op(.x(a), .y(b), .z(c));
  end
endmodule
