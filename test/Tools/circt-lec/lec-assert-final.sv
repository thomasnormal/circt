// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

module modA(input logic a);
  initial begin
    assert final (a);
  end
endmodule

module modB(input logic a);
  initial begin
    assert final (a);
  end
endmodule

// CHECK: unsat
