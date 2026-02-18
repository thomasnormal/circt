// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test string methods: len, putc, getc, toupper, tolower, compare, icompare, substr, atoi, atohex, atoreal, itoa
module top;
  string s, s2;
  integer i;
  real r;

  initial begin
    s = "Hello World";

    // len
    // CHECK: len=11
    $display("len=%0d", s.len());

    // getc
    // CHECK: getc=72
    $display("getc=%0d", s.getc(0));  // 'H' = 72

    // toupper
    s2 = s.toupper();
    // CHECK: toupper=HELLO WORLD
    $display("toupper=%s", s2);

    // tolower
    s2 = s.tolower();
    // CHECK: tolower=hello world
    $display("tolower=%s", s2);

    // substr
    s2 = s.substr(0, 4);
    // CHECK: substr=Hello
    $display("substr=%s", s2);

    // compare
    // CHECK: compare=0
    $display("compare=%0d", s.compare("Hello World"));

    // icompare (case-insensitive)
    // CHECK: icompare=0
    $display("icompare=%0d", s.icompare("HELLO WORLD"));

    // atoi
    s = "42";
    i = s.atoi();
    // CHECK: atoi=42
    $display("atoi=%0d", i);

    // atohex
    s = "FF";
    i = s.atohex();
    // CHECK: atohex=255
    $display("atohex=%0d", i);

    // atoreal
    s = "3.14";
    r = s.atoreal();
    // CHECK: atoreal=3.14
    $display("atoreal=%.2f", r);

    // itoa
    s.itoa(99);
    // CHECK: itoa=99
    $display("itoa=%s", s);

    $finish;
  end
endmodule
