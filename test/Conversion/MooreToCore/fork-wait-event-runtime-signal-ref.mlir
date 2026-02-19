// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// Verify that wait_event lowered inside fork passes a traced signal pointer to
// __moore_wait_event instead of a null pointer fallback.
// CHECK-LABEL: hw.module @ForkWaitEventRuntimeSignalRef
// CHECK: %[[CLK:.*]] = llhd.sig
// CHECK: llhd.process {
// CHECK: sim.fork join_type "join_none" {
// CHECK: %[[CLK_PTR:.*]] = builtin.unrealized_conversion_cast %[[CLK]] : !llhd.ref<i1> to !llvm.ptr
// CHECK: llvm.call @__moore_wait_event(%{{.*}}, %[[CLK_PTR]]) : (i32, !llvm.ptr) -> ()
moore.module @ForkWaitEventRuntimeSignalRef() {
  %false = moore.constant 0 : i1
  %clk = moore.variable %false : <i1>

  moore.procedure initial {
    moore.fork join_none {
      moore.wait_event {
        %clk_now = moore.read %clk : !moore.ref<i1>
        moore.detect_event negedge %clk_now : i1
      }
      moore.fork.terminator
    }

    moore.return
  }

  moore.output
}
