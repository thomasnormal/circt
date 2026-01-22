// RUN: circt-verilog --parse-only %s
// Test that mixing bit (two-valued) and logic (four-valued) types in conditional
// expressions works correctly by promoting operands to the same type.
// This test reproduces the bug where moore.and was created with mismatched types.

// This should compile without the error:
// "moore.and op requires the same type for all operands and results"

module test;
  // Variables with different domains
  bit b_cond;           // two-valued (i1)
  time t_val;           // four-valued 64-bit (like UVM's phase_timeout)
  int result;

  initial begin
    // The combination of:
    // 1. A bit comparison (b_cond == 1) returning i1
    // 2. A time comparison (t_val == 0) returning l1
    // Used to cause "moore.and op requires same type for all operands"
    if ((b_cond == 1) && (t_val == 0)) begin
      result = 1;
    end

    // Test with multiple conditions using &&& operator pattern in if statements
    // This is similar to what UVM does with phase_timeout checking
    if (b_cond &&& (t_val == 0)) begin
      result = 2;
    end
  end
endmodule
