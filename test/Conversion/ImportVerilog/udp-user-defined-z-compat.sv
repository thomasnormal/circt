// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

// DIAG-NOT: invalid symbol 'z' in state table
// DIAG-NOT: dropping user-defined primitive instance

primitive udp_z_compat(q, clk, d);
  output q;
  reg q;
  input clk, d;
  table
    (0z) z : z : z;
    (01) 0 : ? : 0;
    (01) 1 : ? : 1;
  endtable
endprimitive

module top(input logic clk, input logic d, output logic q);
  udp_z_compat u0(q, clk, d);
endmodule

// IR-LABEL: moore.module @top
// IR-DAG: %[[ZERO:.+]] = moore.constant 0 : l1
// IR-DAG: %[[X:.+]] = moore.constant bX : l1
// IR: moore.procedure always
// IR: moore.wait_event {
// IR: moore.detect_event any %{{.+}} : l1
// IR: moore.detect_event any %{{.+}} : l1
// IR: }
// IR: moore.case_eq %{{.+}}, %[[X]] : l1
// IR: moore.case_eq %{{.+}}, %[[X]] : l1
// IR: moore.case_eq %{{.+}}, %[[X]] : l1
// IR: moore.case_eq %{{.+}}, %[[ZERO]] : l1
// IR: moore.blocking_assign %q, %{{.+}} : l1
