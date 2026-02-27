// RUN: circt-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll

// Regression: ABI v5 all_func_flags must carry MAY_YIELD bits per FuncId.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] Collected 2 vtable FuncIds
//
// The first vtable function has moore.wait_event and must be flagged MAY_YIELD.
// The second function is pure arithmetic and must remain unflagged.
// LLVM: @__circt_sim_all_func_flags = private constant [2 x i8] c"\01\00"

llvm.mlir.global internal @g_evt(false) : i1

func.func private @"uvm_pkg::wait_like"() -> i32 {
  %ptr = llvm.mlir.addressof @g_evt : !llvm.ptr
  moore.wait_event {
    %v = llvm.load %ptr : !llvm.ptr -> i1
    %evt = builtin.unrealized_conversion_cast %v : i1 to !moore.event
    moore.detect_event any %evt : event
  }
  %one = hw.constant 1 : i32
  return %one : i32
}

func.func private @"uvm_pkg::keep_alive"(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::wait_like"],
    [1, @"uvm_pkg::keep_alive"]
  ]
} : !llvm.array<2 x ptr>
