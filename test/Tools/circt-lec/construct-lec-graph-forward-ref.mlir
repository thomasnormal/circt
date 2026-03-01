// RUN: circt-opt --construct-lec="first-module=modA second-module=modB insert-mode=none" --verify-each %s | FileCheck %s

// `hw.module` bodies are graph regions and may contain forward references.
// Constructing `verif.lec` must preserve that legality.
hw.module @modA(out out: i1) {
  %sig = llhd.sig %v : i1
  %v = hw.constant false
  %p = llhd.prb %sig : i1
  hw.output %p : i1
}

hw.module @modB(out out: i1) {
  %sig = llhd.sig %v : i1
  %v = hw.constant false
  %p = llhd.prb %sig : i1
  hw.output %p : i1
}

// CHECK: verif.lec
// CHECK: first {
// CHECK: %[[SIGA:.*]] = llhd.sig %[[VA:[^ ]+]] : i1
// CHECK-NEXT: %[[VA]] = hw.constant false
// CHECK: second {
// CHECK: %[[SIGB:.*]] = llhd.sig %[[VB:[^ ]+]] : i1
// CHECK-NEXT: %[[VB]] = hw.constant false
