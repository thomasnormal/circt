// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

// DIAG-NOT: dropping user-defined primitive instance

primitive udp_init_dff(q, clk, d);
  output q;
  reg q;
  input clk, d;
  initial q = 1'b1;
  table
    (01) 0 : ? : 0;
    (01) 1 : ? : 1;
  endtable
endprimitive

module top(input logic clk, input logic d, output logic q);
  udp_init_dff u0(q, clk, d);
endmodule

// IR-LABEL: moore.module @top
// IR-LABEL: moore.procedure initial
// IR: moore.blocking_assign %q, %{{.+}} : l1
// IR-LABEL: moore.procedure always
// IR: moore.wait_event {
// IR: moore.detect_event any %{{.+}} : l1
// IR: moore.detect_event any %{{.+}} : l1
// IR: }
// IR: moore.case_ne %{{.+}}, %{{.+}} : l1
// IR: moore.blocking_assign %q, %{{.+}} : l1
