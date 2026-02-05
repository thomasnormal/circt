// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

class C;
  int a;
  function void pre_randomize();
    a = 1;
  endfunction
  function void post_randomize();
    a = 2;
  endfunction
endclass

module top;
  initial begin
    C c = new();
    void'(c.randomize());
  end
endmodule

// CHECK-LABEL: moore.procedure initial
// CHECK: %[[CREF:.*]] = moore.variable : <class<@C>>
// CHECK: moore.blocking_assign %[[CREF]], %{{.*}} : class<@C>
// CHECK: %[[CVAL:.*]] = moore.read %[[CREF]] : <class<@C>>
// CHECK: moore.call_pre_randomize %[[CVAL]]
// CHECK: %{{.*}} = moore.randomize %[[CVAL]]
// CHECK: moore.call_post_randomize %[[CVAL]]
