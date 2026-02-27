// RUN: not circt-verilog --no-uvm-auto-include --parse-only --relax-enum-conversions=false %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --parse-only %s | FileCheck %s --check-prefix=OK
// RUN: circt-verilog --no-uvm-auto-include --parse-only --relax-enum-conversions %s | FileCheck %s --check-prefix=OK
// REQUIRES: slang

// ERR: no implicit conversion
// OK: module {

module top;
  typedef enum logic [1:0] {
    S0 = 2'b00,
    S1 = 2'b01
  } state_t;

  state_t s;
  int i;

  initial begin
    i = 1;
    s = i;
  end
endmodule
