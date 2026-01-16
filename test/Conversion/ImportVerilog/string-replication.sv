// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test string replication operator
module StringReplication;
  string s;
  int m;

  // CHECK: moore.module @StringReplication
  // CHECK: %s = moore.variable : <string>
  // CHECK: %m = moore.variable : <i32>
  // CHECK: moore.procedure initial
  // CHECK: [[STR:%.*]] = moore.constant_string "-"
  // CHECK: [[COUNT:%.*]] = moore.read %m
  // CHECK: [[RESULT:%.*]] = moore.string_replicate [[COUNT]], {{.*}}
  // CHECK: moore.blocking_assign %s, [[RESULT]] : string
  initial s = {m{"-"}};
endmodule
