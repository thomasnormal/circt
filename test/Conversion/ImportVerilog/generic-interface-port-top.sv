// RUN: not circt-verilog --ir-moore %s --no-uvm-auto-include 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --ir-moore %s --allow-top-level-iface-ports --no-uvm-auto-include | FileCheck %s --check-prefix=OK
// REQUIRES: slang

// By default top-level interface ports without concrete connections are
// rejected by Slang. With --allow-top-level-iface-ports we synthesize an
// opaque interface declaration for the unresolved generic port.
module top(interface bus);
endmodule

// ERR: top-level module 'top' has unconnected interface port 'bus'

// OK-LABEL: moore.module @top
// OK-SAME: in %bus : !moore.ref<virtual_interface<@__generic_interface_0>>
// OK: moore.interface @__generic_interface_0 {
