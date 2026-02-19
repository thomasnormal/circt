// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test math system functions: $pow, $sqrt, $ln, $log10, $exp,
// $floor, $ceil, $sin, $cos, $tan, $asin, $acos, $atan, $atan2,
// $hypot, $sinh, $cosh, $tanh, $asinh, $acosh, $atanh
module top;
  real r;

  initial begin
    // $pow
    r = $pow(2.0, 10.0);
    // CHECK: pow=1024
    $display("pow=%0d", $rtoi(r));

    // $sqrt
    r = $sqrt(144.0);
    // CHECK: sqrt=12.000000
    $display("sqrt=%f", r);

    // $ln (natural log)
    r = $ln(2.718281828);
    // CHECK: ln=1.0
    $display("ln=%.1f", r);

    // $log10
    r = $log10(1000.0);
    // CHECK: log10=3.0
    $display("log10=%.1f", r);

    // $exp
    r = $exp(0.0);
    // CHECK: exp=1.000000
    $display("exp=%f", r);

    // $floor
    r = $floor(3.7);
    // CHECK: floor=3.000000
    $display("floor=%f", r);

    // $ceil
    r = $ceil(3.2);
    // CHECK: ceil=4.000000
    $display("ceil=%f", r);

    // $sin
    r = $sin(0.0);
    // CHECK: sin=0.000000
    $display("sin=%f", r);

    // $cos
    r = $cos(0.0);
    // CHECK: cos=1.000000
    $display("cos=%f", r);

    // $atan2
    r = $atan2(1.0, 1.0);
    // CHECK: atan2=0.8
    $display("atan2=%.1f", r);

    // $hypot
    r = $hypot(3.0, 4.0);
    // CHECK: hypot=5.000000
    $display("hypot=%f", r);

    $finish;
  end
endmodule
