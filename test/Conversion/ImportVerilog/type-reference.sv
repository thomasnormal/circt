// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test type() operator expressions (IEEE 1800-2017 Section 6.23)

// CHECK-LABEL: moore.module @TypeCaseMatch
module TypeCaseMatch #(parameter type T = logic[11:0]);
  // CHECK: moore.procedure initial
  initial begin
    // When T matches logic[11:0], emit only the matching case
    case (type(T))
      type(logic[11:0]) : $display("match 12");
      type(logic[7:0])  : $display("match 8");
      default           : $stop;
    endcase
    // CHECK: moore.fmt.literal "match 12"
    // CHECK-NOT: match 8
    // CHECK-NOT: moore.builtin.stop
  end
endmodule

// CHECK-LABEL: moore.module @TypeCaseDefault
module TypeCaseDefault #(parameter type T = logic[31:0]);
  // CHECK: moore.procedure initial
  initial begin
    // When T doesn't match any case, emit the default
    case (type(T))
      type(logic[11:0]) : $display("match 12");
      type(logic[7:0])  : $display("match 8");
      default           : $display("default");
    endcase
    // CHECK: moore.fmt.literal "default"
    // CHECK-NOT: match 12
    // CHECK-NOT: match 8
  end
endmodule

// CHECK-LABEL: moore.module @TypeCompareIf
module TypeCompareIf #(parameter type T = logic[11:0]);
  // CHECK: moore.procedure initial
  initial begin
    // Type comparisons in if statements are evaluated at compile time
    if (type(T) == type(logic[11:0]))
      $display("types match");
    if (type(T) != type(logic[11:0]))
      $display("types differ");
    // CHECK: moore.fmt.literal "types match"
    // CHECK-NOT: types differ
  end
endmodule

// CHECK-LABEL: moore.module @TypeCaseStruct
module TypeCaseStruct;
  typedef struct packed { logic [7:0] a; logic [7:0] b; } my_struct_t;
  parameter type T = my_struct_t;
  // CHECK: moore.procedure initial
  initial begin
    case (type(T))
      type(my_struct_t) : $display("struct match");
      default           : $stop;
    endcase
    // CHECK: moore.fmt.literal "struct match"
  end
endmodule
