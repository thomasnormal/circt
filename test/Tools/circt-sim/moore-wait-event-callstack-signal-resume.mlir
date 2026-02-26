// RUN: circt-sim %s --max-time=1000000000 2>&1 | FileCheck %s
//
// Regression: moore.wait_event on a signal inside func.func context must
// resume the caller even with an active interpreter call stack.
//
// CHECK: CALL_WAIT_BEGIN
// CHECK: CALL_WAIT_END

module {
  func.func @wait_for_posedge(%clkRef: !llhd.ref<i1>) {
    %begin = sim.fmt.literal "CALL_WAIT_BEGIN\0A"
    %end = sim.fmt.literal "CALL_WAIT_END\0A"
    sim.proc.print %begin
    moore.wait_event {
      %clk = llhd.prb %clkRef : i1
      %clk_m = builtin.unrealized_conversion_cast %clk : i1 to !moore.l1
      moore.detect_event posedge %clk_m : l1
    }
    sim.proc.print %end
    return
  }

  hw.module @test() {
    %false = hw.constant false
    %true = hw.constant true
    %c1000_i64 = arith.constant 1000 : i64
    %eps = llhd.constant_time <0ns, 0d, 1e>

    %clk = llhd.sig %false : i1

    // Toggle clk once at 1ps and halt.
    llhd.process {
      %delay = llhd.int_to_time %c1000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      llhd.drv %clk, %true after %eps : i1
      llhd.halt
    }

    // Wait for posedge inside func.func context and terminate.
    llhd.process {
      func.call @wait_for_posedge(%clk) : (!llhd.ref<i1>) -> ()
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
