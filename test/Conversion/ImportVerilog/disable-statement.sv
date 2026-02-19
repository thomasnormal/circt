// RUN: circt-verilog %s --ir-moore --no-uvm-auto-include | FileCheck %s

module disable_named_block;
  int i;

  initial begin
    begin : my_block
      for (i = 0; i < 10; i = i + 1) begin
        if (i == 2)
          disable my_block;
      end
    end
  end
endmodule

// CHECK-LABEL: moore.module @disable_named_block
// CHECK: moore.procedure initial
// CHECK-NOT: moore.disable
// CHECK: moore.return
