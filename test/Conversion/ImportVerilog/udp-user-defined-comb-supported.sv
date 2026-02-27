// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

primitive udp_xor(q, a, b);
  output q;
  input a, b;
  table
    0 0 : 0;
    0 1 : 1;
    1 0 : 1;
    1 1 : 0;
  endtable
endprimitive

module top(input logic a, input logic b, output logic q);
  udp_xor u0(q, a, b);
endmodule

// DIAG-NOT: dropping user-defined primitive instance
// IR-LABEL: moore.module @top
// IR: %[[X:.+]] = moore.constant bX : l1
// IR: %[[ONE:.+]] = moore.constant 1 : l1
// IR: %[[ZERO:.+]] = moore.constant 0 : l1
// IR-DAG: %[[A0:.+]] = moore.case_eq %a, %[[ZERO]] : l1
// IR-DAG: %[[B0:.+]] = moore.case_eq %b, %[[ZERO]] : l1
// IR-DAG: %[[A1:.+]] = moore.case_eq %a, %[[ONE]] : l1
// IR-DAG: %[[B1:.+]] = moore.case_eq %b, %[[ONE]] : l1
// IR: moore.conditional
// IR: moore.output
