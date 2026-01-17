// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test randsequence break and return statement handling.
// IEEE 1800-2017 Section 18.17 specifies that:
// - break exits the current production early
// - return exits the entire randsequence

module RandSequenceControlTest;
  int a, b, c;
  int x, y, z;
  int error;
  int data;

  // CHECK-LABEL: moore.module @RandSequenceControlTest

  // Test 1: break statement exits current production early
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0; c = 0;
    x = 10;
    randsequence(main)
      main : first second | third;
      first : { if (x > 5) break; a = 1; };  // break exits first, continues with second
      second : { b = 2; };
      third : { c = 3; };
    endsequence
    // When main chooses first|second: a should be 0 (break skips a=1), b should be 2
    // When main chooses third: c should be 3
  end

  // Test 2: return statement exits entire randsequence immediately
  // CHECK: moore.procedure initial
  initial begin
    x = 0; y = 0; z = 0;
    error = 1;
    randsequence(main)
      main : first second third;
      first : { if (error) return; x = 1; };  // return exits entire randsequence
      second : { y = 2; };
      third : { z = 3; };
    endsequence
    // x, y, z should all be 0 since return exits immediately
  end

  // Test 3: break in weighted production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    randsequence(weighted)
      weighted : opt_a | opt_b;
      opt_a : { if (data == 0) break; data = 10; } := 5;
      opt_b : { data = 20; } := 3;
    endsequence
    // If opt_a is selected and data==0, break skips data=10
    // If opt_b is selected, data=20
  end

  // Test 4: return in weighted production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    error = 1;
    randsequence(weighted_ret)
      weighted_ret : ret_a ret_b;
      ret_a : { if (error) return; data = 100; } := 2;
      ret_b : { data = 200; } := 8;
    endsequence
    // If error==1, return exits entire sequence, data stays 0
  end

  // Test 5: break in code block before production items
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0;
    x = 20;
    randsequence(code_block_break)
      code_block_break : { if (x > 10) break; a = 5; } next_prod;
      next_prod : { b = 10; };
    endsequence
    // break exits code block, skips a=5, but next_prod should NOT execute
  end

  // Test 6: return in code block before production items
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0;
    error = 1;
    randsequence(code_block_return)
      code_block_return : { if (error) return; a = 7; } next_prod2;
      next_prod2 : { b = 15; };
    endsequence
    // return exits entire randsequence, a and b stay 0
  end

  // Test 7: break inside if-else production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    x = 1;
    randsequence(if_break)
      if_break : if (x) then_prod else else_prod;
      then_prod : { if (x > 0) break; data = 100; };
      else_prod : { data = 200; };
    endsequence
    // then_prod selected, break exits early, data stays 0
  end

  // Test 8: return inside if-else production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    x = 1;
    error = 1;
    randsequence(if_return)
      if_return : if (x) then_ret else else_ret;
      then_ret : { if (error) return; data = 300; };
      else_ret : { data = 400; };
    endsequence
    // return exits entire randsequence, data stays 0
  end

  // Test 9: break inside repeat production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    randsequence(repeat_break)
      repeat_break : repeat(5) rep_item;
      rep_item : { data = data + 1; if (data >= 3) break; };
    endsequence
    // First iteration: data=1, no break
    // Second iteration: data=2, no break
    // Third iteration: data=3, break exits rep_item
    // Fourth iteration: data=4, break exits rep_item
    // Fifth iteration: data=5, break exits rep_item
    // Final data should be 5
  end

  // Test 10: return inside repeat production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    randsequence(repeat_return)
      repeat_return : repeat(5) ret_item;
      ret_item : { data = data + 1; if (data >= 2) return; };
    endsequence
    // First iteration: data=1, no return
    // Second iteration: data=2, return exits entire randsequence
    // Final data should be 2
  end

  // Test 11: break inside case production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    x = 1;
    randsequence(case_break)
      case_break : case (x)
        0: case_a;
        1: case_b;
        default: case_c;
      endcase;
      case_a : { data = 10; };
      case_b : { if (x > 0) break; data = 20; };
      case_c : { data = 30; };
    endsequence
    // case_b selected (x==1), break exits early, data stays 0
  end

  // Test 12: return inside case production
  // CHECK: moore.procedure initial
  initial begin
    data = 0;
    x = 2;
    error = 1;
    randsequence(case_return)
      case_return : case (x)
        0: ret_case_a;
        default: ret_case_b;
      endcase;
      ret_case_a : { data = 100; };
      ret_case_b : { if (error) return; data = 200; };
    endsequence
    // ret_case_b selected (x==2), return exits entire randsequence, data stays 0
  end

  // Test 13: nested break - break only exits innermost production
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0; c = 0;
    x = 1;
    randsequence(nested_break)
      nested_break : outer_prod;
      outer_prod : inner_prod { b = 2; };
      inner_prod : { if (x) break; a = 1; };
    endsequence
    // break exits inner_prod (a stays 0), but outer_prod continues (b=2)
  end

  // Test 14: multiple breaks in sequence
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0; c = 0;
    x = 1;
    randsequence(multi_break)
      multi_break : br1 br2 br3;
      br1 : { if (x) break; a = 1; };
      br2 : { b = 2; };
      br3 : { c = 3; };
    endsequence
    // br1 breaks (a stays 0), then br2 executes (b=2), then br3 executes (c=3)
  end

  // Test 15: break and return combination
  // CHECK: moore.procedure initial
  initial begin
    a = 0; b = 0;
    x = 0;
    error = 1;
    randsequence(break_return_combo)
      break_return_combo : combo1 combo2;
      combo1 : { if (x) break; if (error) return; a = 1; };
      combo2 : { b = 2; };
    endsequence
    // x==0 so no break, error==1 so return exits entire randsequence
    // a and b both stay 0
  end

endmodule
