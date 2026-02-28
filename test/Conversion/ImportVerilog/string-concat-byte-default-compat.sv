// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Default behavior should allow mixed string+byte concatenation to match
// mainstream simulator compatibility, with strict mode still available via
// --relax-string-conversions=false.

module string_concat_byte_default_compat;
  string s;
  byte b;

  initial begin
    s = "ab";
    b = s[0];
    s = {s, string'(b)};
  end
endmodule

// CHECK-LABEL: moore.module @string_concat_byte_default_compat
// CHECK: %[[STRINIT_RAW:.+]] = moore.constant_string "ab" : i16
// CHECK: %[[STRINIT:.+]] = moore.int_to_string %[[STRINIT_RAW]] : i16
// CHECK: moore.blocking_assign %s, %[[STRINIT]] : string
// CHECK: %[[S0:.+]] = moore.read %s : <string>
// CHECK: %[[CHAR:.+]] = moore.string.getc %[[S0]][%{{.+}}]
// CHECK: moore.blocking_assign %b, %[[CHAR]] : i8
// CHECK: %[[S1:.+]] = moore.read %s : <string>
// CHECK: %[[B:.+]] = moore.read %b : <i8>
// CHECK: %[[BSTR:.+]] = moore.int_to_string %[[B]] : i8
// CHECK: %[[CAT:.+]] = moore.string_concat (%[[S1]], %[[BSTR]]) : string
// CHECK: moore.blocking_assign %s, %[[CAT]] : string
