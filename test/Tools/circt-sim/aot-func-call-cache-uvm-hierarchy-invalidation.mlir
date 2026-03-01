// RUN: circt-compile -v %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_NO_FUNC_DISPATCH=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: `uvm_component::get_child` must NOT be function-result cached.
// In UVM this call commonly takes a pointer to mutable string storage. A cache
// keyed only by pointer bits can return stale lookup results after in-place
// updates to the pointed-to key bytes.
//
// RUNTIME: first=11 second=22 count=2

llvm.mlir.global internal @key(11 : i64) : i64
llvm.mlir.global internal @counter(0 : i32) : i32

func.func private @"uvm_pkg::uvm_component::get_child"(%this: i64,
                                                       %key_ptr: i64) -> i64 {
  %counter_addr = llvm.mlir.addressof @counter : !llvm.ptr
  %old = llvm.load %counter_addr : !llvm.ptr -> i32
  %one = arith.constant 1 : i32
  %new = arith.addi %old, %one : i32
  llvm.store %new, %counter_addr : i32, !llvm.ptr

  %key_addr = llvm.inttoptr %key_ptr : i64 to !llvm.ptr
  %key = llvm.load %key_addr : !llvm.ptr -> i64
  return %key : i64
}

hw.module @top() {
  %this = hw.constant 1 : i64
  %new_key = hw.constant 22 : i64
  %key_addr = llvm.mlir.addressof @key : !llvm.ptr
  %key_ptr = llvm.ptrtoint %key_addr : !llvm.ptr to i64

  %first_prefix = sim.fmt.literal "first="
  %second_prefix = sim.fmt.literal " second="
  %count_prefix = sim.fmt.literal " count="
  %nl = sim.fmt.literal "\0A"

  llhd.process {
    %first = func.call @"uvm_pkg::uvm_component::get_child"(%this, %key_ptr) : (i64, i64) -> i64
    llvm.store %new_key, %key_addr : i64, !llvm.ptr
    %second = func.call @"uvm_pkg::uvm_component::get_child"(%this, %key_ptr) : (i64, i64) -> i64
    %counter_addr = llvm.mlir.addressof @counter : !llvm.ptr
    %count = llvm.load %counter_addr : !llvm.ptr -> i32

    %first_s = sim.fmt.dec %first signed : i64
    %second_s = sim.fmt.dec %second signed : i64
    %count_s = sim.fmt.dec %count signed : i32
    %msg = sim.fmt.concat (%first_prefix, %first_s, %second_prefix, %second_s, %count_prefix, %count_s, %nl)
    sim.proc.print %msg
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
