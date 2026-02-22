// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaProceduralExplicitClockPrecedence(input logic clk_proc,
                                            input logic clk_prop,
                                            input logic en, input logic a);
  // CHECK: %[[CLKPROPREAD:.+]] = moore.read %clk_prop_{{.+}} : <l1>
  // CHECK: %[[CLKPROPBOOL:.+]] = moore.to_builtin_bool %[[CLKPROPREAD]] : l1
  // CHECK: verif.clocked_assert %{{.+}} if %{{.+}}, posedge %[[CLKPROPBOOL]] : i1
  always @(posedge clk_proc) begin
    if (en)
      assert property (@(posedge clk_prop) a);
  end
endmodule
