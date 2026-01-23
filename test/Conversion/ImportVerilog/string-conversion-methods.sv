// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test string conversion methods (IEEE 1800-2017 Section 6.16.9)

// CHECK-LABEL: moore.module @StringConversionMethods()
module StringConversionMethods();
  string str;
  real r;
  int i;

  initial begin
    // atoreal: string to real
    // CHECK: moore.string.atoreal
    r = str.atoreal();

    // hextoa: integer to hex string
    // CHECK: moore.string.hextoa %str, %{{.+}} : <string>, l32
    str.hextoa(255);

    // octtoa: integer to octal string
    // CHECK: moore.string.octtoa %str, %{{.+}} : <string>, l32
    str.octtoa(63);

    // bintoa: integer to binary string
    // CHECK: moore.string.bintoa %str, %{{.+}} : <string>, l32
    str.bintoa(15);

    // realtoa: real to string
    // CHECK: moore.string.realtoa %str, %{{.+}} : <string>, f64
    str.realtoa(2.71828);

    // putc: set character at index
    // CHECK: moore.string.putc %str[%{{.+}}], %{{.+}} : <string>
    str.putc(0, 8'h41);
  end
endmodule
