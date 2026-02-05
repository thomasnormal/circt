// RUN: circt-opt %s --strip-sim | FileCheck %s

// CHECK-LABEL: hw.module @strip_sim
// CHECK: verif.assume
// CHECK-SAME: bmc.finish
// CHECK-NOT: sim.
// CHECK: hw.output
hw.module @strip_sim(in %clock: !seq.clock, in %enable: i1) {
  %lit = sim.fmt.literal "hello"
  sim.print %lit on %clock if %enable
  sim.terminate success, quiet
  hw.output
}

// CHECK-LABEL: hw.module @strip_pause
// CHECK: verif.assert
// CHECK-NOT: sim.
hw.module @strip_pause() {
  sim.pause quiet
  hw.output
}

// CHECK-LABEL: func.func @strip_pause_guard
// CHECK: verif.assert %{{.*}} if %{{.*}} : i1
// CHECK-NOT: sim.
func.func @strip_pause_guard(%cond: i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  sim.pause quiet
  cf.br ^bb2
^bb2:
  func.return
}

// CHECK-LABEL: hw.module @strip_terminate_failure
// CHECK: verif.assert
// CHECK-NOT: sim.
hw.module @strip_terminate_failure() {
  sim.terminate failure, quiet
  hw.output
}

// CHECK-LABEL: hw.module @strip_clocked_pause
// CHECK: verif.clocked_assert
// CHECK-NOT: sim.
hw.module @strip_clocked_pause(in %clock: !seq.clock, in %enable: i1) {
  sim.clocked_pause %clock, %enable, quiet
  hw.output
}

// CHECK-LABEL: hw.module @strip_clocked_terminate_failure
// CHECK: verif.clocked_assert
// CHECK-NOT: sim.
hw.module @strip_clocked_terminate_failure(in %clock: !seq.clock, in %enable: i1) {
  sim.clocked_terminate %clock, %enable, failure, quiet
  hw.output
}

// CHECK-LABEL: hw.module @strip_clocked_terminate_success
// CHECK: verif.assume
// CHECK-SAME: bmc.finish
// CHECK-NOT: sim.
hw.module @strip_clocked_terminate_success(in %clock: !seq.clock, in %enable: i1) {
  sim.clocked_terminate %clock, %enable, success, quiet
  hw.output
}

// CHECK-LABEL: hw.module @strip_sim_llhd_process
// CHECK: llhd.constant_time <0ns, 0d, 0e>
// CHECK-NOT: llhd.current_time
// CHECK-NOT: sim.terminate
// CHECK-NOT: bmc.finish
hw.module @strip_sim_llhd_process() {
  %c0_i1 = hw.constant false
  %c1_i1 = hw.constant true
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %c0_i1 : i1
  %0:1 = llhd.process -> i1 {
    llhd.wait yield (%c1_i1 : i1), (%c1_i1 : i1), ^bb1
  ^bb1(%arg0: i1):
    %now = llhd.current_time
    %_ = llhd.time_to_int %now
    sim.terminate success, quiet
    llhd.halt %arg0 : i1
  }
  llhd.drv %sig, %0#0 after %t0 if %c1_i1 : i1
  hw.output
}
