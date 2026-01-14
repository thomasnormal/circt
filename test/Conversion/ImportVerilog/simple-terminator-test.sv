// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

// Test for block terminator generation in various control flow constructs.
// These patterns ensure that all blocks have proper terminators (return, cf.br, cf.cond_br).

module TerminatorTest(input int in1, input int in2);
  int result1, result2;

  initial begin
    result1 = try_get_pattern(in1);
    result2 = all_paths_return(in2);
  end
endmodule

// Test case from uvm_set_before_get_dap::try_get
// Both branches return, so exit block should be unreachable
// CHECK-LABEL: func.func private @try_get_pattern
// CHECK:         cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK:         return %{{.*}} : !moore.i1
// CHECK:       ^bb2:
// CHECK:         return %{{.*}} : !moore.i1
// CHECK:       }
function bit try_get_pattern(int flag);
  if (!flag) begin
    return 0;
  end
  else begin
    return 1;
  end
endfunction

// Pattern: multiple if/else if/else statements with returns
// CHECK-LABEL: func.func private @all_paths_return
// CHECK:         cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK:         return %{{.*}} : !moore.i32
// CHECK:       ^bb2:
// CHECK:         cf.cond_br %{{.*}}, ^bb3, ^bb4
// CHECK:       ^bb3:
// CHECK:         return %{{.*}} : !moore.i32
// CHECK:       ^bb4:
// CHECK:         cf.cond_br %{{.*}}, ^bb5, ^bb6
// CHECK:       ^bb5:
// CHECK:         return %{{.*}} : !moore.i32
// CHECK:       ^bb6:
// CHECK:         return %{{.*}} : !moore.i32
// CHECK:       }
function int all_paths_return(int x);
  if (x == 0) begin
    return 0;
  end
  else if (x == 1) begin
    return 1;
  end
  else if (x == 2) begin
    return 2;
  end
  else begin
    return -1;
  end
endfunction
