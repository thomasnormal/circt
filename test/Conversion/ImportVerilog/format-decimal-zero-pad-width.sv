// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module format_decimal_zero_pad_width;
  logic [7:0] v;

  initial begin
    v = 8'h30;
    $display("z=%011d s=%11d m=%0d", v, v, v);
  end
endmodule

// CHECK-LABEL: moore.module @format_decimal_zero_pad_width
// CHECK: moore.fmt.int decimal %{{.+}}, align right, pad zero width 11
// CHECK: moore.fmt.int decimal %{{.+}}, align right, pad space width 11
// CHECK: moore.fmt.int decimal %{{.+}}, align right, pad zero width 0
