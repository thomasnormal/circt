// RUN: circt-sim --max-time=200000000 %s 2>&1 | FileCheck %s

// Test that fork branches with forever loops and __moore_delay properly yield
// and don't cause PROCESS_STEP_OVERFLOW. This is critical for UVM phase
// schedulers that use fork/join_none with forever loops containing blocking
// operations.

// The fork branch prints "tick" every 30ns. The main process waits 100ns and
// then prints "main done" and terminates. We expect multiple ticks (at least 3)
// and the simulation to complete successfully without PROCESS_STEP_OVERFLOW.

// CHECK-DAG: tick
// CHECK-DAG: tick
// CHECK-DAG: tick
// CHECK-DAG: main done
// CHECK-NOT: ERROR(PROCESS_STEP_OVERFLOW)
// CHECK: Simulation finished successfully

llvm.func @__moore_delay(i64)

hw.module @test() {
  %main_done = sim.fmt.literal "main done\0A"
  %tick = sim.fmt.literal "tick\0A"
  %space = sim.fmt.literal " "
  %c100000000_i64 = hw.constant 100000000 : i64  // 100ns
  %c30000000_i64 = hw.constant 30000000 : i64    // 30ns (tick interval)

  llhd.process {
    // Fork with forever loop - must yield properly during each delay
    %handle = sim.fork join_type "join_none" {
      sim.proc.print %space
      cf.br ^loop
    ^loop:
      llvm.call @__moore_delay(%c30000000_i64) : (i64) -> ()
      sim.proc.print %tick
      cf.br ^loop
    }

    // Main thread waits 100ns then exits
    %delay = llhd.int_to_time %c100000000_i64
    llhd.wait delay %delay, ^done
  ^done:
    sim.proc.print %main_done
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
