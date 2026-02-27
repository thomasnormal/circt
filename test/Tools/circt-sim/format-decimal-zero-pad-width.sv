// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [7:0] v;
  initial begin
    v = 8'h30;
    $display("z<%011d>", v);
    $display("s<%11d>", v);
    // CHECK: z<00000000048>
    // CHECK: s<         48>
    $finish;
  end
endmodule
