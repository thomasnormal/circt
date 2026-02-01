// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s -o - | FileCheck %s

interface TargetIf;
endinterface

interface BindIf;
endinterface

module Top;
  TargetIf t();

  interface BindIf;
  endinterface

  bind t BindIf bind_inst();
endmodule

// CHECK: moore.interface @TargetIf
// CHECK: moore.module @Top
