// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-hidden-vars-test.c -ldl
// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top vpi_hidden_vars --max-time=100000 --vpi=%t.vpi.so 2>&1 | FileCheck %s
//
// CHECK: VPI_HIDDEN: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_HIDDEN: FINAL: {{[0-9]+}} passed, 0 failed

module vpi_hidden_vars;
  reg _underscore_name;
  reg a, b;
  reg [3:0] \weird.signal[1] ;
  reg [3:0] \weird.signal[2] ;

  always begin
    a <= 1'b0;
    b <= 1'b0;
    #10;
    a <= 1'b1;
    b <= 1'b1;
    #10;
    \weird.signal[1] <= 4'h0;
  end
endmodule
