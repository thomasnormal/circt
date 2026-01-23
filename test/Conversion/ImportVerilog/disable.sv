// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @DisableTest()
module DisableTest;
  reg a = 0;
  reg b = 0;
  reg c = 0;

  // Test basic disable of a named block
  // CHECK: moore.procedure initial {
  initial begin: block1
    a = 1;
    // CHECK: moore.disable "block1"
    disable block1;
    b = 1;  // This should not be reached
  end

  // Test disable inside a conditional
  // CHECK: moore.procedure initial {
  initial begin: block2
    a = 1;
    if (a == 1) begin
      // CHECK: moore.disable "block2"
      disable block2;
    end
    b = 1;
  end

  // Test disable of another block from fork
  // CHECK: moore.procedure initial {
  initial fork
    begin: named_block
      #10 a = 1;
      #10 b = 1;
    end
    // CHECK: moore.disable "named_block"
    #15 disable named_block;
  join
endmodule
