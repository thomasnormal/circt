// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s --check-prefix=IR

// DIAG-NOT: dropping user-defined primitive instance

primitive udp_latch(q, en, d);
  output q;
  reg q;
  input en, d;
  table
    // en d : q : q+
       0  ? : ? : -;
       1  0 : ? : 0;
       1  1 : ? : 1;
  endtable
endprimitive

module top(input logic en, input logic d, output logic q);
  udp_latch u0(q, en, d);
endmodule

// IR-LABEL: moore.module @top
// IR: moore.procedure always
// IR: moore.wait_event {
// IR: moore.detect_event any %{{.+}} : l1
// IR: moore.detect_event any %{{.+}} : l1
// IR: }
// IR: %[[EN0:.+]] = moore.case_eq %{{.+}}, %{{.+}} : l1
// IR: %[[BODY:.+]] = moore.conditional %[[EN0]] : i1 -> l1
// IR: moore.blocking_assign %q, %[[BODY]] : l1
