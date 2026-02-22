// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=2000000000 2>&1 | FileCheck %s --check-prefix=FUNC
// RUN: env CIRCT_SIM_TRACE_CONT_ASSIGN=1 CIRCT_SIM_TRACE_CONT_ASSIGN_FILTER=scl circt-sim %t.mlir --top top --max-time=2000000000 2>&1 | FileCheck %s --check-prefix=TRACE

// Regression for distinct-driver continuous-assign sensitivity on interface
// pointer sources with pullups. The strong drivers should be sensitive to
// interface pointer fields (via deferred expansion), not self-target fallback.

interface i3c_like_if(inout wire scl);
  logic scl_i;
  logic scl_o;
  logic scl_oen;

  assign scl = scl_oen ? scl_o : 1'bz;
  assign scl_i = scl;
endinterface

module driver(i3c_like_if vif);
  initial begin
    vif.scl_o = 1'b0;
    vif.scl_oen = 1'b0;

    #2;
    $display("DRV_REL_0 scl_i=%b", vif.scl_i);

    vif.scl_o = 1'b0;
    vif.scl_oen = 1'b1;
    #2;
    $display("DRV_LOW scl_i=%b", vif.scl_i);

    vif.scl_o = 1'b1;
    vif.scl_oen = 1'b0;
    #2;
    $display("DRV_REL_1 scl_i=%b", vif.scl_i);

    $finish;
  end
endmodule

module top;
  wire scl;
  pullup p1(scl);

  i3c_like_if c_if(scl);
  i3c_like_if t_if(scl);
  driver d(c_if);
endmodule

// FUNC: DRV_REL_0 scl_i=1
// FUNC: DRV_LOW scl_i=0
// FUNC: DRV_REL_1 scl_i=1

// TRACE: [CONT-DRV] sig={{[0-9]+}} name=scl enc=4state drives=3 multi=1 distinct=1
// TRACE: [CONT-DRV] #1 src=1 proc=0 strength(strong, strong) srcIds={{[0-9]+}}(sig_0,2state){{ *$}}
// TRACE: [CONT-DRV] #2 src=1 proc=0 strength(strong, strong) srcIds={{[0-9]+}}(sig_1,2state){{ *$}}
