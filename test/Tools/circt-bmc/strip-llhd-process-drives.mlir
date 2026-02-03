// RUN: circt-opt %s --strip-llhd-processes | FileCheck %s

// CHECK-LABEL: hw.module @drive_init_only
// CHECK-NOT: llhd.process
// CHECK: llhd.drv %{{.*}}, %{{.*}} after %{{.*}} : i1
hw.module @drive_init_only(out out: i1) {
  %c0_i1 = hw.constant false
  %c1_i1 = hw.constant true
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig name "sig" %c0_i1 : i1
  llhd.process {
    llhd.drv %sig, %c1_i1 after %t0 : i1
    llhd.halt
  }
  %0 = llhd.prb %sig : i1
  hw.output %0 : i1
}

// CHECK: hw.module @drive_with_wait
// CHECK-SAME: in %[[SIG_IN:[a-zA-Z0-9_]+]]{{ *}}: i1
// CHECK-NOT: llhd.process
// CHECK: llhd.drv %{{.*}}, %[[SIG_IN]] after %{{.*}} : i1
hw.module @drive_with_wait(out out: i1) {
  %c0_i1 = hw.constant false
  %c1_i1 = hw.constant true
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig name "sig" %c0_i1 : i1
  llhd.process {
    llhd.wait delay %t0, ^bb1
  ^bb1:
    llhd.drv %sig, %c1_i1 after %t0 : i1
    llhd.halt
  }
  %0 = llhd.prb %sig : i1
  hw.output %0 : i1
}

// CHECK: hw.module @drive_with_wait_from_input
// CHECK-SAME: in %[[IN:[a-zA-Z0-9_]+]]{{ *}}: i1
// CHECK-NOT: llhd.process
// CHECK: llhd.drv %{{.*}}, %[[IN]] after %{{.*}} : i1
hw.module @drive_with_wait_from_input(in %in: i1, out out: i1) {
  %c0_i1 = hw.constant false
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig name "sig" %c0_i1 : i1
  llhd.process {
    llhd.wait delay %t0, ^bb1
  ^bb1:
    llhd.drv %sig, %in after %t0 : i1
    llhd.halt
  }
  %0 = llhd.prb %sig : i1
  hw.output %0 : i1
}
