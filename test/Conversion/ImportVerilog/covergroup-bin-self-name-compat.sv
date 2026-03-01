// RUN: circt-verilog --no-uvm-auto-include %s --ir-moore 2>&1 | FileCheck %s
// CHECK-NOT: error:
// CHECK-LABEL: moore.covergroup.decl @cg
// CHECK: moore.coverbin.decl @__circt_bin_BAUD_4800 kind<bins> values [0]
// CHECK: moore.coverbin.decl @__circt_bin_BAUD_9600 kind<bins> values [1]

typedef enum int { BAUD_4800, BAUD_9600 } baud_e;

class c;
  baud_e baud;

  covergroup cg;
    cp: coverpoint baud {
      bins BAUD_4800 = {BAUD_4800};
      bins BAUD_9600 = {BAUD_9600};
    }
  endgroup

  function new();
    cg = new();
  endfunction
endclass

module top;
  c obj;
  initial obj = new();
endmodule
