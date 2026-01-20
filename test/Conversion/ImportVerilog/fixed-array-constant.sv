// RUN: circt-verilog --ir-moore %s | FileCheck %s

module fixed_array_const;
  localparam int arr[2] = '{1, 2};
  int x;
  initial begin
    x = arr[0];
  end
endmodule

// CHECK: moore.array_create
