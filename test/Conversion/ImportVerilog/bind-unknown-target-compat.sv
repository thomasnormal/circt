// RUN: circt-verilog --no-uvm-auto-include --verify-diagnostics %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module bind_unknown_target_checker;
endmodule

module bind_unknown_target_dummy_top;
  bind_unknown_target_checker u_checker();
endmodule

// expected-warning @below {{unknown module 'missing_target'}}
bind missing_target bind_unknown_target_checker u_bind_unknown();

bind
  missing_target_multiline // expected-warning {{unknown module 'missing_target_multiline'}}
  bind_unknown_target_checker u_bind_unknown_multiline();

// CHECK-LABEL: moore.module private @bind_unknown_target_checker
