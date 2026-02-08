// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test short-circuit evaluation of logical operators (IEEE 1800-2017 §11.4.7).
// RHS of && should not be evaluated when LHS is false.
// RHS of || should not be evaluated when LHS is true.
// RHS of -> (logical implication) should not be evaluated when LHS is false.

module top;
  int count;

  function automatic int inc_count();
    count = count + 1;
    return 1;
  endfunction

  initial begin
    // Test 1: && short-circuit - LHS false, RHS should NOT be called
    count = 0;
    if (0 && inc_count()) begin end
    // CHECK: sc_and_false=0
    $display("sc_and_false=%0d", count);

    // Test 2: && no short-circuit - LHS true, RHS IS called
    count = 0;
    if (1 && inc_count()) begin end
    // CHECK: sc_and_true=1
    $display("sc_and_true=%0d", count);

    // Test 3: || short-circuit - LHS true, RHS should NOT be called
    count = 0;
    if (1 || inc_count()) begin end
    // CHECK: sc_or_true=0
    $display("sc_or_true=%0d", count);

    // Test 4: || no short-circuit - LHS false, RHS IS called
    count = 0;
    if (0 || inc_count()) begin end
    // CHECK: sc_or_false=1
    $display("sc_or_false=%0d", count);

    // Test 5: -> (implication) short-circuit - LHS false, RHS should NOT be called
    count = 0;
    if (0 -> inc_count()) begin end
    // CHECK: sc_impl_false=0
    $display("sc_impl_false=%0d", count);

    // Test 6: -> no short-circuit - LHS true, RHS IS called
    count = 0;
    if (1 -> inc_count()) begin end
    // CHECK: sc_impl_true=1
    $display("sc_impl_true=%0d", count);

    // Test 7: Nested short-circuit - (0 && (1 && inc_count()))
    count = 0;
    if (0 && (1 && inc_count())) begin end
    // CHECK: sc_nested=0
    $display("sc_nested=%0d", count);

    // Test 8: Chained - (1 && 1 && inc_count()) should call inc_count
    count = 0;
    if (1 && 1 && inc_count()) begin end
    // CHECK: sc_chain=1
    $display("sc_chain=%0d", count);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
