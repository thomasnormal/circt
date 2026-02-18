// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test string comparison and methods: len, getc, toupper, tolower, compare, icompare, substr
module top;
  string s1, s2, s3;
  int c;

  initial begin
    s1 = "Hello";

    // len — string length
    // CHECK: len=5
    $display("len=%0d", s1.len());

    // getc — get character at index (returns byte value)
    c = s1.getc(0);
    // 'H' = 72
    // CHECK: getc_0=72
    $display("getc_0=%0d", c);

    c = s1.getc(4);
    // 'o' = 111
    // CHECK: getc_4=111
    $display("getc_4=%0d", c);

    // toupper
    s2 = s1.toupper();
    // CHECK: upper=HELLO
    $display("upper=%s", s2);

    // tolower
    s3 = s1.tolower();
    // CHECK: lower=hello
    $display("lower=%s", s3);

    // compare — case-sensitive
    s1 = "abc";
    s2 = "abc";
    // CHECK: compare_eq=0
    $display("compare_eq=%0d", s1.compare(s2));

    s2 = "abd";
    // compare returns negative if s1 < s2
    // CHECK: compare_lt=1
    $display("compare_lt=%0d", s1.compare(s2) < 0);

    // icompare — case-insensitive
    s1 = "Hello";
    s2 = "hello";
    // CHECK: icompare_eq=0
    $display("icompare_eq=%0d", s1.icompare(s2));

    // substr
    s1 = "Hello World";
    s2 = s1.substr(6, 10);
    // CHECK: substr=World
    $display("substr=%s", s2);

    $finish;
  end
endmodule
