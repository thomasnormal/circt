// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Test for constraint_mode() and pre/post_randomize callbacks
// IEEE 1800-2017 Section 18.6.1, 18.8

// CHECK-LABEL: moore.class.classdecl @Transaction
class Transaction;
  rand bit [7:0] addr;
  rand bit [31:0] data;

  // Named constraint
  constraint c_addr { addr > 8'h10; }

  // pre_randomize callback - called before randomization
  // CHECK: moore.class.methoddecl @pre_randomize
  function void pre_randomize();
    $display("pre_randomize called");
  endfunction

  // post_randomize callback - called after successful randomization
  // CHECK: moore.class.methoddecl @post_randomize
  function void post_randomize();
    $display("post_randomize: addr=%h, data=%h", addr, data);
  endfunction
endclass

module test;
  Transaction tx;

  initial begin
    tx = new();

    // Test constraint_mode getter - returns current mode (1 = enabled)
    // CHECK: moore.constraint_mode
    $display("c_addr mode = %d", tx.c_addr.constraint_mode());

    // Test constraint_mode setter - disable constraint
    // CHECK: moore.constraint_mode
    tx.c_addr.constraint_mode(0);

    // Test randomize - should call pre_randomize and post_randomize
    // CHECK: moore.call_pre_randomize
    // CHECK: moore.randomize
    // CHECK: moore.call_post_randomize
    if (tx.randomize()) begin
      $display("Randomization succeeded");
    end

    // Re-enable constraint
    // CHECK: moore.constraint_mode
    tx.c_addr.constraint_mode(1);

    // Disable all constraints on object
    // CHECK: moore.constraint_mode
    tx.constraint_mode(0);

    // Randomize again
    // CHECK: moore.call_pre_randomize
    // CHECK: moore.randomize
    // CHECK: moore.call_post_randomize
    if (tx.randomize()) begin
      $display("Randomization with all constraints disabled");
    end
  end
endmodule
