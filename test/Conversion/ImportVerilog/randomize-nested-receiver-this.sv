// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

class AgentCfg;
  rand int spiMode;
  rand int shiftDirection;
endclass

class EnvCfg;
  AgentCfg spiMasterAgentConfig;
  function new();
    spiMasterAgentConfig = new;
  endfunction
endclass

class Test;
  EnvCfg spiEnvConfig;
  int operationModes;
  int shiftDirection;

  function new();
    spiEnvConfig = new;
  endfunction

  function void setup();
    if (!spiEnvConfig.spiMasterAgentConfig.randomize() with {
          this.spiMode == operationModes;
          this.shiftDirection == shiftDirection;
        }) begin
    end
  endfunction
endclass

module top;
  Test t;
  initial begin
    t = new;
    t.setup();
  end
endmodule

// CHECK: moore.randomize
// CHECK: moore.class.property_ref %{{.*}}[@spiMode]
// CHECK: moore.class.property_ref %{{.*}}[@shiftDirection]
// CHECK: moore.constraint.expr
