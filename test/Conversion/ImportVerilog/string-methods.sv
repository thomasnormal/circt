// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Tests for string comparison methods: compare() and icompare()

module StringMethods;
  string s1, s2;
  int result;

  initial begin
    s1 = "Hello";
    s2 = "hello";

    // CHECK: moore.string.compare
    result = s1.compare(s2);

    // CHECK: moore.string.icompare
    result = s1.icompare(s2);

    // Test with same case strings
    s1 = "abc";
    s2 = "abc";

    // CHECK: moore.string.compare
    result = s1.compare(s2);

    // CHECK: moore.string.icompare
    result = s1.icompare(s2);

    // Test ordering
    s1 = "aaa";
    s2 = "bbb";

    // CHECK: moore.string.compare
    result = s1.compare(s2);
  end
endmodule
