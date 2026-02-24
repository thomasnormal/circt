// RUN: circt-verilog --ir-moore %s | FileCheck %s

module TaggedUnionInvalidMemberAccess;
  typedef union tagged {
    void Invalid;
    int Valid;
  } u_int;

  u_int a;
  int c;

  initial begin
    a = tagged Invalid;
    // CHECK: [[READ:%.+]] = moore.read %a
    // CHECK: [[TAG:%.+]] = moore.struct_extract [[READ]], "tag"
    // CHECK: [[DATA:%.+]] = moore.struct_extract [[READ]], "data"
    // CHECK: [[MATCH:%.+]] = moore.eq [[TAG]], %{{.+}}
    // CHECK: [[GUARD:%.+]] = moore.conditional [[MATCH]]
    // CHECK: moore.builtin.severity fatal
    // CHECK: moore.union_extract [[DATA]], "Valid"
    c = a.Valid;
  end
endmodule
