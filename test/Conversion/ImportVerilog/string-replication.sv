// RUN: circt-translate --import-verilog --verify-diagnostics %s
// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that string replication emits a proper error instead of crashing
// (replication operator requires IntType operand)
module StringReplication;
  string s;
  int m;
  // expected-error @below {{expression of type '!moore.string' cannot be cast to a simple bit vector}}
  // CHECK: error: expression of type '!moore.string' cannot be cast to a simple bit vector
  initial s = {m{"-"}};
endmodule
