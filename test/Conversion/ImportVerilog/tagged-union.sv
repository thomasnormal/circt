// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Tagged union expressions

// CHECK-LABEL: moore.module @TaggedUnionBasic()
module TaggedUnionBasic;
  typedef union tagged { int i; logic [7:0] b; } u_t;

  // CHECK: %u = moore.variable : <ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>>
  u_t u;
  // CHECK: %x = moore.variable : <i32>
  int x;
  // CHECK: %y = moore.variable : <l8>
  logic [7:0] y;

  initial begin
    // CHECK: [[UNION0:%.+]] = moore.union_create %{{.+}} {fieldName = "i"} : i32 -> uunion<{i: i32, b: l8}>
    // CHECK: [[STRUCT0:%.+]] = moore.struct_create %{{.+}}, [[UNION0]] : !moore.i1, !moore.uunion<{i: i32, b: l8}> -> ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>
    // CHECK: moore.blocking_assign %u, [[STRUCT0]] : ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>
    u = tagged i 42;

    // CHECK: [[UREAD0:%.+]] = moore.read %u : <ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>>
    // CHECK: [[DATA0:%.+]] = moore.struct_extract [[UREAD0]], "data" : ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}> -> uunion<{i: i32, b: l8}>
    // CHECK: [[IVAL:%.+]] = moore.union_extract [[DATA0]], "i" : uunion<{i: i32, b: l8}> -> i32
    // CHECK: moore.blocking_assign %x, [[IVAL]] : i32
    x = u.i;

    // CHECK: [[UNION1:%.+]] = moore.union_create %{{.+}} {fieldName = "b"} : l8 -> uunion<{i: i32, b: l8}>
    // CHECK: [[STRUCT1:%.+]] = moore.struct_create %{{.+}}, [[UNION1]] : !moore.i1, !moore.uunion<{i: i32, b: l8}> -> ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>
    // CHECK: moore.blocking_assign %u, [[STRUCT1]] : ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>
    u = tagged b 8'hAA;

    // CHECK: [[UREAD1:%.+]] = moore.read %u : <ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}>>
    // CHECK: [[DATA1:%.+]] = moore.struct_extract [[UREAD1]], "data" : ustruct<{tag: i1, data: uunion<{i: i32, b: l8}>}> -> uunion<{i: i32, b: l8}>
    // CHECK: [[BVAL:%.+]] = moore.union_extract [[DATA1]], "b" : uunion<{i: i32, b: l8}> -> l8
    // CHECK: moore.blocking_assign %y, [[BVAL]] : l8
    y = u.b;
  end
endmodule

// CHECK-LABEL: moore.module @TaggedUnionPatternCase()
module TaggedUnionPatternCase;
  typedef union tagged { int i; logic [7:0] b; } u_t;
  u_t u;
  int x;

  initial begin
    // CHECK: [[UVAL:%.+]] = moore.read %u
    // CHECK: [[TAG:%.+]] = moore.struct_extract [[UVAL]], "tag"
    // CHECK: [[TAG_MATCH:%.+]] = moore.case_eq [[TAG]], %{{.+}}
    // CHECK: cf.cond_br
    case (u) matches
      tagged i: x = 1;
      tagged b: x = 2;
      default: x = 3;
    endcase
  end
endmodule

// CHECK-LABEL: moore.module @TaggedUnionPatternIf()
module TaggedUnionPatternIf;
  typedef union tagged { int i; logic [7:0] b; } u_t;
  u_t u;
  int x;

  initial begin
    // CHECK: [[UVAL:%.+]] = moore.read %u
    // CHECK: [[TAG:%.+]] = moore.struct_extract [[UVAL]], "tag"
    // CHECK: [[TAG_MATCH:%.+]] = moore.case_eq [[TAG]], %{{.+}}
    // CHECK: cf.cond_br
    if (u matches tagged i)
      x = 1;
    else
      x = 2;

  end
endmodule

// Test structure patterns with variable binding
// CHECK-LABEL: moore.module @StructurePatternWithVariables()
module StructurePatternWithVariables;
  typedef union tagged {
    struct {
      bit [3:0] val1, val2;
    } a;
    struct {
      bit [7:0] val1, val2;
    } b;
  } u_t;

  u_t u;
  int result;

  initial begin
    // Test case statement with structure pattern and variable binding
    // The pattern variables v1, v2 are extracted from the struct fields and
    // used directly - the optimizer inlines their use.
    // CHECK: [[UVAL:%.+]] = moore.read %u
    // CHECK: [[TAG:%.+]] = moore.struct_extract [[UVAL]], "tag"
    // CHECK: [[TAG_MATCH:%.+]] = moore.case_eq [[TAG]], %{{.+}}
    // CHECK: moore.struct_extract {{.*}}, "data"
    // CHECK: moore.union_extract {{.*}}, "a"
    // CHECK: moore.struct_extract {{.*}}, "val1"
    // CHECK: moore.struct_extract {{.*}}, "val2"
    // CHECK: cf.cond_br
    case (u) matches
      tagged a '{.v1, .v2}: result = v1 + v2;
      tagged b '{.v1, .v2}: result = v1 - v2;
    endcase
  end
endmodule

// Test conditional expression with pattern matching
// CHECK-LABEL: moore.module @ConditionalPatternMatching()
module ConditionalPatternMatching;
  typedef union tagged {
    struct { bit [3:0] val1, val2; } a;
    struct { bit [7:0] val1, val2; } b;
  } u_t;

  u_t u;
  bit [3:0] result;

  initial begin
    // Test conditional expression with pattern (matches operator)
    // Pattern variables become moore.variable ops when used in conditional true branch
    // CHECK: [[UVAL:%.+]] = moore.read %u
    // CHECK: moore.struct_extract {{.*}}, "tag"
    // CHECK: moore.union_extract
    // CHECK: moore.struct_extract {{.*}}, "val1"
    // CHECK: %v1 = moore.variable
    // CHECK: moore.blocking_assign %v1
    // CHECK: moore.conditional
    result = u matches tagged a '{.v1, .v2} ? v1 : 4'd2;
  end
endmodule

// Test if statement with structure pattern and variable binding
// CHECK-LABEL: moore.module @IfPatternWithVariables()
module IfPatternWithVariables;
  typedef union tagged {
    struct { bit [3:0] val1, val2; } a;
    struct { bit [7:0] val1, val2; } b;
  } u_t;

  u_t u;
  int result;

  initial begin
    // Pattern matching with constant and variable binding
    // CHECK: [[UVAL:%.+]] = moore.read %u
    // CHECK: [[TAG:%.+]] = moore.struct_extract [[UVAL]], "tag"
    // CHECK: moore.union_extract
    // CHECK: moore.struct_extract {{.*}}, "val1"
    // CHECK: moore.case_eq
    // CHECK: moore.struct_extract {{.*}}, "val2"
    // CHECK: cf.cond_br
    if (u matches tagged a '{4'd0, .v})
      result = v;
    else
      result = 0;
  end
endmodule
