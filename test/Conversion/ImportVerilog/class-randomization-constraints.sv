// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Class with randomization constraints
class constrained_rand;
  rand bit [7:0] value;
  rand int count;

  // Simple range constraint
  constraint c_value { value inside {[10:20]}; }

  // Expression constraint
  constraint c_count { count > 0; count < 100; }

  function new();
    value = 0;
    count = 0;
  endfunction
endclass

// CHECK-LABEL: moore.class @constrained_rand
// CHECK: moore.class.property @value : !moore.packed<range<i8>>
// CHECK-SAME: rand_mode = rand
// CHECK: moore.class.property @count : !moore.int
// CHECK-SAME: rand_mode = rand

module test;
  initial begin
    constrained_rand obj = new();
    int success;

    // Test basic randomization
    success = obj.randomize();
    $display("Randomize success=%0d", success);
    $display("value=%0d (should be 10-20)", obj.value);
    $display("count=%0d (should be 1-99)", obj.count);

    // Test multiple randomizations
    repeat(3) begin
      success = obj.randomize();
      $display("value=%0d count=%0d", obj.value, obj.count);
    end
  end
endmodule

// CHECK-LABEL: moore.procedure always
// CHECK: %[[OBJ:.+]] = moore.class.new @constrained_rand
// CHECK: %[[SUCCESS1:.+]] = moore.randomize %[[OBJ]]
// CHECK: %[[SUCCESS2:.+]] = moore.randomize %[[OBJ]]
