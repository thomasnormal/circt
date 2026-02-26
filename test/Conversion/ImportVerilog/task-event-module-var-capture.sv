// RUN: circt-verilog %s --ir-moore | FileCheck %s

module TaskEventModuleVarCapture;
  bit clk;

  task automatic wait_for_clk();
    @(posedge clk);
  endtask

  initial begin
    wait_for_clk();
  end
endmodule

// CHECK: moore.module @TaskEventModuleVarCapture() {
// CHECK:   %clk = moore.variable : <i1>
// CHECK:   moore.procedure initial {
// CHECK:     func.call @wait_for_clk(%clk) : (!moore.ref<i1>) -> ()
// CHECK:   }
// CHECK: }
// CHECK: func.func private @wait_for_clk(%arg0: !moore.ref<i1>) {
// CHECK:   moore.wait_event {
// CHECK:     %[[READ:.*]] = moore.read %arg0 : <i1>
// CHECK:     moore.detect_event posedge %[[READ]] : i1
// CHECK:   }
// CHECK:   return
// CHECK: }
