// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @ForkJoinDecl
module ForkJoinDecl;
  int i;
  initial begin
    i = 7;
    fork
      automatic int k = i;
      begin
        // CHECK: moore.fork join_none {
        // CHECK:   %k = moore.variable
        // CHECK:   moore.read %k
        // CHECK: }
        #1;
        $display(k);
      end
    join_none
  end
endmodule
