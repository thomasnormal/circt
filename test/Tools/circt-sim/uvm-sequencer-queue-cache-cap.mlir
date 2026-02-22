// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 CIRCT_SIM_UVM_SEQ_QUEUE_CACHE_MAX_ENTRIES=1 circt-sim %s 2>&1 | FileCheck %s --check-prefix=CHECK-CAP
// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 CIRCT_SIM_UVM_SEQ_QUEUE_CACHE_MAX_ENTRIES=1 CIRCT_SIM_UVM_SEQ_QUEUE_CACHE_EVICT_ON_CAP=1 circt-sim %s 2>&1 | FileCheck %s --check-prefix=CHECK-EVICT

// CHECK-CAP: [circt-sim] UVM sequencer queue cache: hits=0 misses=2 installs=0 entries=0 capacity_skips=0 evictions=0
// CHECK-CAP: [circt-sim] UVM sequencer queue cache limits: max_entries=1 capacity_skips=0 evictions=0 evict_on_cap=0

// CHECK-EVICT: [circt-sim] UVM sequencer queue cache: hits=0 misses=2 installs=0 entries=0 capacity_skips=0 evictions=0
// CHECK-EVICT: [circt-sim] UVM sequencer queue cache limits: max_entries=1 capacity_skips=0 evictions=0 evict_on_cap=1

module {
  llvm.mlir.global internal @port1_obj(0 : i8) : i8
  llvm.mlir.global internal @port2_obj(0 : i8) : i8
  llvm.mlir.global internal @seqr1_obj(0 : i8) : i8
  llvm.mlir.global internal @seqr2_obj(0 : i8) : i8

  func.func @"uvm_pkg::uvm_port_base::connect"(%self: !llvm.ptr, %provider: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(%port: !llvm.ptr, %out: !llvm.ptr) {
    return
  }

  hw.module @top() {
    llhd.process {
      cf.br ^start
    ^start:
      %port1 = llvm.mlir.addressof @port1_obj : !llvm.ptr
      %port2 = llvm.mlir.addressof @port2_obj : !llvm.ptr
      %seqr1 = llvm.mlir.addressof @seqr1_obj : !llvm.ptr
      %seqr2 = llvm.mlir.addressof @seqr2_obj : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref1 = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %ref2 = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr

      func.call @"uvm_pkg::uvm_port_base::connect"(%port1, %seqr1) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call @"uvm_pkg::uvm_port_base::connect"(%port2, %seqr2) : (!llvm.ptr, !llvm.ptr) -> ()

      func.call @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(%port1, %ref1) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(%port2, %ref2) : (!llvm.ptr, !llvm.ptr) -> ()
      llhd.halt
    }

    hw.output
  }
}
