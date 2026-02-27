// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

// Keep default compatibility with mainstream simulators that allow assigning
// interface instances targeted by bind/defparam to virtual interfaces.

interface J;
  parameter int q = 1;
  logic [7:0] data;
endinterface

interface I;
  J j();
endinterface

module top;
  I i1();
  virtual I vi1 = i1;
  defparam i1.j.q = 42;
endmodule

// DIAG-NOT: error:

// IR-LABEL: moore.interface @I
// IR: moore.interface.signal @j : !moore.virtual_interface<@J> {interface_instance}
// IR-LABEL: moore.module @top
// IR: %i1 = moore.interface.instance  @I : <virtual_interface<@I>>
// IR: attributes {vpi.interface_instances = {i1 = "I"}}
