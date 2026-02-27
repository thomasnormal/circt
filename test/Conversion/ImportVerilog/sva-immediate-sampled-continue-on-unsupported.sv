// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

interface ifc;
  logic s;
endinterface

module top;
  virtual ifc vif;

  initial begin
    assert ($stable(vif));
    assert ($rose(vif));
    assert ($fell(vif));
  end
endmodule

// STRICT-NOT: error: unsupported sampled value type for $stable
// STRICT-NOT: error: unsupported sampled value type for $rose
// STRICT-NOT: error: unsupported sampled value type for $fell
// WARN-NOT: warning: $stable has unsupported sampled value type
// WARN-NOT: warning: $rose has unsupported sampled value type
// WARN-NOT: warning: $fell has unsupported sampled value type

// IR-LABEL: moore.module @top
// IR: moore.assert
