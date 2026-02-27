// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s --check-prefix=OUT
// REQUIRES: slang

module top;
  class par_cls #(int a = 25);
    parameter int b = 23;
  endclass

  par_cls #(15) inst;

  initial begin
    inst = new;
    if (par_cls::b != 23) begin
      $display("FAIL b=%0d", par_cls::b);
      $fatal(1);
    end
    $display("PASS b=%0d", par_cls::b);
    $finish;
  end
endmodule

// OUT: PASS b=23
