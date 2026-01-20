// RUN: circt-verilog --ir-moore %s | FileCheck %s

module std_randomize_capture;
  logic val_o;

  // CHECK: [[VAL_O:%.+]] = moore.variable : <l1>
  // CHECK: func.func private @randomize_o(%arg0: !moore.ref<l1>) -> !moore.i1
  function bit randomize_o();
    bit success;
    // CHECK: moore.std_randomize %arg0 : !moore.ref<l1>
    success = std::randomize(val_o);
    return success;
  endfunction
endmodule
