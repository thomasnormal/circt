// RUN: circt-compile -v %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_next_child CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_UVM_CHILD_ITER_UNSAFE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s

// Regression: in unsafe child-iterator opt-in mode, direct native dispatch
// must still demote when the iterator key pointer comes from interpreter-local
// process memory. Native __moore_assoc_next expects host/runtime-managed key
// storage and can crash on interpreter-owned buffers.
//
// CHECK: Compiled function calls:          0
// CHECK: Interpreted function calls:       1
// CHECK: direct_calls_native:              0
// CHECK: direct_calls_interpreted:         1
// CHECK: Top interpreted func.call fallback reasons (top 50):
// CHECK: uvm_pkg::uvm_component::get_next_child [pointer-safety=1]
// CHECK: safe=1

llvm.mlir.global internal @dummy_this(0 : i8) : i8

func.func private @"uvm_pkg::uvm_component::get_next_child"(%this: !llvm.ptr,
                                                            %key: !llvm.ptr) -> i32 {
  %v = llvm.load %key : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %one = hw.constant 1 : i64
  %prefix = sim.fmt.literal "safe="
  %nl = sim.fmt.literal "\0A"

  %this = llvm.mlir.addressof @dummy_this : !llvm.ptr

  llhd.process {
    %key = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %one32 = hw.constant 1 : i32
    llvm.store %one32, %key : i32, !llvm.ptr
    %r = func.call @"uvm_pkg::uvm_component::get_next_child"(%this, %key) : (!llvm.ptr, !llvm.ptr) -> i32
    %vr = sim.fmt.dec %r signed : i32
    %msg = sim.fmt.concat (%prefix, %vr, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %one
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
