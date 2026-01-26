// RUN: circt-opt --llhd-inline-calls %s | FileCheck %s
// RUN: circt-opt --llhd-inline-calls --symbol-dce %s | FileCheck %s --check-prefixes=CHECK,CHECK-DCE

// CHECK-LABEL: @Basic
hw.module @Basic(in %a: i42, in %b: i42, out u: i42, out v: i42) {
  // CHECK: llhd.combinational
  %0:2 = llhd.combinational -> i42, i42 {
    // CHECK-NOT: call @foo
    %1:2 = func.call @foo(%a, %b) : (i42, i42) -> (i42, i42)
    // CHECK-NEXT:   cf.br [[BB1:\^.+]]
    // CHECK-NEXT: [[BB1]]:
    // CHECK-NEXT:   [[TMP1:%.+]] = comb.add %a, %b :
    // CHECK-NOT:    call @bar
    // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
    // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor %a, [[TMP2]] :
    // CHECK-NEXT:   cf.br [[BB2:\^.+]]([[TMP3]] : i42)
    // CHECK-NEXT: [[BB2]]([[TMP3B:%.+]]: i42):
    // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP3B]] : i42)
    // CHECK-NEXT: [[BB3]]([[TMP3C:%.+]]: i42):
    // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3C]], %b :
    // CHECK-NEXT:   cf.br [[BB4:\^.+]]
    // CHECK-NEXT: [[BB4]]:
    // CHECK-NEXT:   cf.br [[BB5:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
    // CHECK-NEXT: [[BB5]]([[CALLRES0:%.+]]: i42, [[CALLRES1:%.+]]: i42):

    // CHECK-NEXT: scf.execute_region
    %2:2 = scf.execute_region -> (i42, i42) {
      // CHECK-NOT: call @foo
      %3:2 = func.call @foo(%1#0, %1#1) : (i42, i42) -> (i42, i42)
      // CHECK-NEXT:   cf.br [[BB1:\^.+]]
      // CHECK-NEXT: [[BB1]]:
      // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[CALLRES0]], [[CALLRES1]] :
      // CHECK-NOT:    call @bar
      // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
      // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor [[CALLRES0]], [[TMP2]] :
      // CHECK-NEXT:   cf.br [[BB2:\^.+]]([[TMP3]] : i42)
      // CHECK-NEXT: [[BB2]]([[TMP3B:%.+]]: i42):
      // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP3B]] : i42)
      // CHECK-NEXT: [[BB3]]([[TMP3C:%.+]]: i42):
      // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3C]], [[CALLRES1]] :
      // CHECK-NEXT:   cf.br [[BB4:\^.+]]
      // CHECK-NEXT: [[BB4]]:
      // CHECK-NEXT:   cf.br [[BB5:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
      // CHECK-NEXT: [[BB5]]([[CALLRES2:%.+]]: i42, [[CALLRES3:%.+]]: i42):

      // CHECK-NEXT: scf.yield [[CALLRES2]], [[CALLRES3]]
      scf.yield %3#0, %3#1 : i42, i42
    }

    llhd.yield %2#0, %2#1 : i42, i42
  }
  hw.output %0#0, %0#1 : i42, i42
}

// CHECK-DCE-NOT: @foo
func.func private @foo(%arg0: i42, %arg1: i42) -> (i42, i42) {
  cf.br ^bb1
^bb1:
  %0 = comb.add %arg0, %arg1 : i42
  %1 = call @bar_wrapper(%arg0) : (i42) -> i42
  %2 = comb.mul %1, %arg1 : i42
  cf.br ^bb2
^bb2:
  return %0, %2 : i42, i42
}

// CHECK-DCE-NOT: @bar
func.func private @bar(%arg0: i42) -> i42 {
  %0 = hw.constant 42 : i42
  %1 = comb.xor %arg0, %0 : i42
  return %1 : i42
}

// CHECK-DCE-NOT: @bar_wrapper
func.func private @bar_wrapper(%arg0: i42) -> i42 {
  %0 = call @bar(%arg0) : (i42) -> i42
  return %0 : i42
}

// CHECK-LABEL: @Init
hw.module @Init() {
  // CHECK: seq.initial
  seq.initial() {
    %c1_i32 = hw.constant 1 : i32
    // CHECK-NOT: call @dummy
    func.call @dummy(%c1_i32) : (i32) -> ()
  } : () -> ()
  hw.output
}

func.func private @dummy(%arg0: i32) {
  return
}

// CHECK-LABEL: @TopLevelCall
hw.module @TopLevelCall(out out0 : i1) {
  %false = hw.constant false
  %sig = llhd.sig %false : i1
  // CHECK-NOT: call @readSig
  %val = func.call @readSig(%sig) : (!llhd.ref<i1>) -> i1
  hw.output %val : i1
}

func.func private @readSig(%arg0: !llhd.ref<i1>) -> i1 {
  %0 = llhd.prb %arg0 : i1
  return %0 : i1
}

// Test that UVM initialization functions are not inlined to avoid infinite
// recursion. These functions have guarded recursion at runtime.
// CHECK-LABEL: @UvmInitGuardedRecursion
hw.module @UvmInitGuardedRecursion(out out0 : i32) {
  %c0 = hw.constant 0 : i32
  %sig = llhd.sig %c0 : i32
  // CHECK: call @"uvm_pkg::uvm_init"
  // The UVM init function should not be inlined
  %val = func.call @"uvm_pkg::uvm_init"(%sig) : (!llhd.ref<i32>) -> i32
  hw.output %val : i32
}

// This function simulates the UVM initialization pattern where there's
// guarded recursion (the recursion is safe at runtime due to state checks).
func.func private @"uvm_pkg::uvm_init"(%arg0: !llhd.ref<i32>) -> i32 {
  %c1 = hw.constant 1 : i32
  // In real UVM, this would call back to uvm_get_report_object which calls
  // uvm_coreservice_t::get which calls uvm_init, but with runtime guards.
  return %c1 : i32
}

// Test that uvm_coreservice_t::get is not inlined
// CHECK-LABEL: @UvmCoreserviceGet
hw.module @UvmCoreserviceGet(out out0 : i32) {
  // CHECK: call @"uvm_pkg::uvm_coreservice_t::get"
  %val = func.call @"uvm_pkg::uvm_coreservice_t::get"() : () -> i32
  hw.output %val : i32
}

func.func private @"uvm_pkg::uvm_coreservice_t::get"() -> i32 {
  %c1 = hw.constant 1 : i32
  return %c1 : i32
}

// Test that uvm_get_report_object is not inlined
// CHECK-LABEL: @UvmGetReportObject
hw.module @UvmGetReportObject(out out0 : i32) {
  // CHECK: call @"uvm_pkg::uvm_get_report_object"
  %val = func.call @"uvm_pkg::uvm_get_report_object"() : () -> i32
  hw.output %val : i32
}

func.func private @"uvm_pkg::uvm_get_report_object"() -> i32 {
  %c1 = hw.constant 1 : i32
  return %c1 : i32
}

// Test that UVM constructors (`::new`) are not inlined to avoid recursion.
// CHECK-LABEL: @UvmPhaseNew
hw.module @UvmPhaseNew(out out0 : i32) {
  // CHECK: call @"uvm_pkg::uvm_phase::new"
  %val = func.call @"uvm_pkg::uvm_phase::new"() : () -> i32
  hw.output %val : i32
}

func.func private @"uvm_pkg::uvm_phase::new"() -> i32 {
  %c1 = hw.constant 1 : i32
  return %c1 : i32
}

// Test that external functions are not inlined (no body to inline).
// CHECK-LABEL: @ExternalCall
hw.module @ExternalCall(out out0 : i32) {
  %c0 = hw.constant 0 : i32
  // CHECK: call @ext
  %val = func.call @ext(%c0) : (i32) -> i32
  hw.output %val : i32
}

func.func private @ext(%arg0: i32) -> i32
