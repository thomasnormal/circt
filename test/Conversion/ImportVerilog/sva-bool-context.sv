// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// Tests for SVA sampled value functions in boolean contexts

// Test: $changed in boolean OR context within assertion
// CHECK-LABEL: moore.module @test_changed_or
module test_changed_or(input logic clk);
    int cyc = 0;
    logic val = 0;
    // $changed returns moore value type, OR with boolean uses moore.or
    // CHECK: moore.or
    assert property(@(posedge clk) cyc == 0 || $changed(val));
endmodule

// Test: $stable with logical NOT in assertion
// CHECK-LABEL: moore.module @test_stable_not
module test_stable_not(input logic clk);
    int cyc = 0;
    logic val = 0;
    // !$stable uses moore.not on the result
    // CHECK: moore.not
    assert property(@(posedge clk) cyc == 0 || !$stable(val));
endmodule

// Test: $sampled in procedural context
// CHECK-LABEL: moore.module @test_sampled_procedural
module test_sampled_procedural(input logic clk);
    logic val = 0;
    always @(posedge clk) begin
        // $sampled should return original moore type for comparison
        // CHECK: moore.ne %{{.*}}, %{{.*}} : l1 -> l1
        if (val != $sampled(val))
            $display("changed");
    end
endmodule

// Test: $changed in boolean AND context
// CHECK-LABEL: moore.module @test_changed_and
module test_changed_and(input logic clk);
    logic a, b;
    // Both return moore values, AND uses moore.and
    // CHECK: moore.and
    assert property(@(posedge clk) $changed(a) && $changed(b));
endmodule

// Test: Boolean implication with LTL property
// CHECK-LABEL: moore.module @test_impl
module test_impl(input logic clk);
    logic a, b;
    // CHECK: ltl.implication
    assert property(@(posedge clk) a |-> $changed(b));
endmodule
