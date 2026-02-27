// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module ParametrizedClassScopeResolutionDefaultCompat;
  class par_cls #(int a = 25);
    parameter int b = 23;
  endclass

  par_cls #(15) inst;

  initial begin
    inst = new;
    if (par_cls::b != 23)
      $fatal(1, "unexpected class param value");
    $display("PASS b=%0d", par_cls::b);
  end
endmodule

// IR-LABEL: moore.module @ParametrizedClassScopeResolutionDefaultCompat
// IR: moore.constant 23 : i32
// IR: moore.builtin.display
