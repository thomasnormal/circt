// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

// DIAG-NOT: dropping user-defined primitive instance

primitive udp_r(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    r 0 : ? : 0;
    r 1 : ? : 1;
  endtable
endprimitive

primitive udp_f(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    f 0 : ? : 0;
    f 1 : ? : 1;
  endtable
endprimitive

primitive udp_p(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    p 0 : ? : 0;
    p 1 : ? : 1;
  endtable
endprimitive

primitive udp_n(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    n 0 : ? : 0;
    n 1 : ? : 1;
  endtable
endprimitive

module top(input logic clk, input logic d,
           output logic q_r, output logic q_f,
           output logic q_p, output logic q_n);
  udp_r u_r(q_r, clk, d);
  udp_f u_f(q_f, clk, d);
  udp_p u_p(q_p, clk, d);
  udp_n u_n(q_n, clk, d);
endmodule

// IR-LABEL: moore.module @top
// IR: moore.procedure always {
// IR: moore.wait_event {
// IR: moore.detect_event any
// IR: moore.detect_event any
// IR: }
// IR: moore.case_ne
// IR: moore.blocking_assign %q_r
// IR: moore.procedure always {
// IR: moore.wait_event {
// IR: moore.detect_event any
// IR: moore.detect_event any
// IR: }
// IR: moore.case_ne
// IR: moore.blocking_assign %q_f
// IR: moore.procedure always {
// IR: moore.blocking_assign %q_p
// IR: moore.procedure always {
// IR: moore.blocking_assign %q_n
