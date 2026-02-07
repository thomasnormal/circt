// RUN: circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage --max-udp-coverage-notes=2 %s 2>&1 | FileCheck %s --check-prefix=LIMIT2
// REQUIRES: slang

// DEFAULT: warning: primitive does not specify outputs for all edges of all inputs
// DEFAULT: remark: missed desired output for rows:
// DEFAULT: (0x) 0
// DEFAULT: (10) 0
// DEFAULT: (1x) 0
// DEFAULT: ...and more

// LIMIT2: warning: primitive does not specify outputs for all edges of all inputs
// LIMIT2: remark: missed desired output for rows:
// LIMIT2-NEXT: (0x) 0
// LIMIT2-NEXT: (10) 0
// LIMIT2-NEXT: ...and more

primitive p(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    (01) 0 : ? : 0;
    (01) 1 : ? : 1;
  endtable
endprimitive

module top;
  logic q, clk, d;
  p u(q, clk, d);
endmodule
