// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

primitive udp_seq(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    * 0 : ? : 0;
    * 1 : ? : 1;
  endtable
endprimitive

module top(input logic clk, input logic d, output logic q);
  udp_seq u0(q, clk, d);
endmodule

// DIAG-NOT: dropping user-defined primitive instance
// IR-LABEL: moore.module @top
// IR: moore.procedure always
// IR: moore.wait_event {
// IR: moore.detect_event any %{{.+}} : l1
// IR: moore.detect_event any %{{.+}} : l1
// IR: }
// IR: %{{.+}} = moore.case_ne %{{.+}}, %{{.+}} : l1
// IR: moore.blocking_assign %q, %{{.+}} : l1
