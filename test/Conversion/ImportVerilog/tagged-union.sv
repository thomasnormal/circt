// RUN: circt-verilog --ir-moore %s | FileCheck %s
// XFAIL: *
// Tagged union expressions are not yet supported

// CHECK-LABEL: moore.module @TaggedUnionBasic()
module TaggedUnionBasic;
  typedef union tagged { int i; logic [7:0] b; } u_t;

  // CHECK: %u = moore.variable : <struct<{tag: i1, data: union<{i: i32, b: l8}>}>>
  u_t u;
  // CHECK: %x = moore.variable : <i32>
  int x;
  // CHECK: %y = moore.variable : <l8>
  logic [7:0] y;

  initial begin
    // CHECK: [[TAG0:%.+]] = moore.constant 0 : i1
    // CHECK: [[VAL0:%.+]] = moore.constant 42 : i32
    // CHECK: [[UNION0:%.+]] = moore.union_create [[VAL0]] {fieldName = "i"} : i32 -> union<{i: i32, b: l8}>
    // CHECK: [[STRUCT0:%.+]] = moore.struct_create [[TAG0]], [[UNION0]] : !moore.i1, !moore.union<{i: i32, b: l8}> -> struct<{tag: i1, data: union<{i: i32, b: l8}>}>
    // CHECK: moore.blocking_assign %u, [[STRUCT0]] : struct<{tag: i1, data: union<{i: i32, b: l8}>}>
    u = tagged i 42;

    // CHECK: [[UREAD0:%.+]] = moore.read %u : <struct<{tag: i1, data: union<{i: i32, b: l8}>}>>
    // CHECK: [[DATA0:%.+]] = moore.struct_extract [[UREAD0]], "data" : struct<{tag: i1, data: union<{i: i32, b: l8}>}> -> union<{i: i32, b: l8}>
    // CHECK: [[IVAL:%.+]] = moore.union_extract [[DATA0]], "i" : union<{i: i32, b: l8}> -> i32
    // CHECK: moore.blocking_assign %x, [[IVAL]] : i32
    x = u.i;

    // CHECK: [[TAG1:%.+]] = moore.constant 1 : i1
    // CHECK: [[VAL1:%.+]] = moore.constant 170 : l8
    // CHECK: [[UNION1:%.+]] = moore.union_create [[VAL1]] {fieldName = "b"} : l8 -> union<{i: i32, b: l8}>
    // CHECK: [[STRUCT1:%.+]] = moore.struct_create [[TAG1]], [[UNION1]] : !moore.i1, !moore.union<{i: i32, b: l8}> -> struct<{tag: i1, data: union<{i: i32, b: l8}>}>
    // CHECK: moore.blocking_assign %u, [[STRUCT1]] : struct<{tag: i1, data: union<{i: i32, b: l8}>}>
    u = tagged b 8'hAA;

    // CHECK: [[UREAD1:%.+]] = moore.read %u : <struct<{tag: i1, data: union<{i: i32, b: l8}>}>>
    // CHECK: [[DATA1:%.+]] = moore.struct_extract [[UREAD1]], "data" : struct<{tag: i1, data: union<{i: i32, b: l8}>}> -> union<{i: i32, b: l8}>
    // CHECK: [[BVAL:%.+]] = moore.union_extract [[DATA1]], "b" : union<{i: i32, b: l8}> -> l8
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
    // CHECK: [[TAG0:%.+]] = moore.constant 0
    // CHECK: [[TAG_MATCH:%.+]] = moore.case_eq [[TAG]], [[TAG0]]
    // CHECK: cf.cond_br [[TAG_MATCH]], ^{{.*}}, ^{{.*}}
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
    // CHECK: [[TAG0:%.+]] = moore.constant 0
    // CHECK: [[TAG_MATCH:%.+]] = moore.case_eq [[TAG]], [[TAG0]]
    // CHECK: cf.cond_br [[TAG_MATCH]], ^{{.*}}, ^{{.*}}
    if (u matches tagged i)
      x = 1;
    else
      x = 2;

  end
endmodule
