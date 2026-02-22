// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s --max-time 1 2>&1 | FileCheck %s
//
// Check native analysis-port terminal cache behavior:
// 1) first size call installs cache entry (miss)
// 2) second size call reuses it (hit)
// 3) connect mutation invalidates the cache
// 4) next size call misses and rebuilds
//
// CHECK: [circt-sim] analysis port terminal cache: entries=1 hits=1 misses=2 invalidations=1

module {
  llvm.mlir.global internal @port_obj(0 : i64) : i64
  llvm.mlir.global internal @provider_a(0 : i64) : i64
  llvm.mlir.global internal @provider_b(0 : i64) : i64

  func.func @"uvm_pkg::uvm_port_base::connect"(%self: !llvm.ptr, %provider: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_port_base::size"(%self: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }

  hw.module @top() {
    llhd.process {
      cf.br ^entry
    ^entry:
      %port = llvm.mlir.addressof @port_obj : !llvm.ptr
      %pa = llvm.mlir.addressof @provider_a : !llvm.ptr
      %pb = llvm.mlir.addressof @provider_b : !llvm.ptr

      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %pa) : (!llvm.ptr, !llvm.ptr) -> ()
      %s0 = func.call @"uvm_pkg::uvm_port_base::size"(%port) : (!llvm.ptr) -> i32
      %s1 = func.call @"uvm_pkg::uvm_port_base::size"(%port) : (!llvm.ptr) -> i32

      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %pb) : (!llvm.ptr, !llvm.ptr) -> ()
      %s2 = func.call @"uvm_pkg::uvm_port_base::size"(%port) : (!llvm.ptr) -> i32

      llhd.halt
    }
    hw.output
  }
}
