// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_first_child,uvm_pkg::uvm_component::get_next_child circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=SAFE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_first_child,uvm_pkg::uvm_component::get_next_child CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_UVM_CHILD_ITER_UNSAFE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=UNSAFE

// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
//
// SAFE: Unmapped native func.call policy: default deny uvm_pkg::* and pointer-typed get_/set_/create_/m_initialize* (allow others){{.*}}with UVM child iterator safety deny
// SAFE: Compiled function calls:          0
// SAFE: Interpreted function calls:       2
// SAFE: direct_calls_native:              0
// SAFE: direct_calls_interpreted:         2
// SAFE: Top interpreted func.call fallback reasons (top 50):
// SAFE: uvm_pkg::uvm_component::get_first_child [unmapped-policy=1]
// SAFE: uvm_pkg::uvm_component::get_next_child [unmapped-policy=1]
// SAFE: sum=3
//
// UNSAFE: Unmapped native func.call policy: default deny uvm_pkg::* and pointer-typed get_/set_/create_/m_initialize* (allow others){{.*}}with allow list 'uvm_pkg::uvm_component::get_first_child,uvm_pkg::uvm_component::get_next_child'
// UNSAFE: Compiled function calls:          0
// UNSAFE: Interpreted function calls:       2
// UNSAFE: direct_calls_native:              0
// UNSAFE: direct_calls_interpreted:         2
// UNSAFE: Top interpreted func.call fallback reasons (top 50):
// UNSAFE: uvm_pkg::uvm_component::get_first_child [pointer-safety=1]
// UNSAFE: uvm_pkg::uvm_component::get_next_child [pointer-safety=1]
// UNSAFE: sum=3

func.func private @"uvm_pkg::uvm_component::get_first_child"(%this: !llvm.ptr,
                                                             %iter: !llvm.ptr) -> i32 {
  %c1 = arith.constant 1 : i32
  return %c1 : i32
}

func.func private @"uvm_pkg::uvm_component::get_next_child"(%this: !llvm.ptr,
                                                            %iter: !llvm.ptr) -> i32 {
  %c2 = arith.constant 2 : i32
  return %c2 : i32
}

llvm.mlir.global internal @dummy(0 : i8) : i8

hw.module @top() {
  %prefix = sim.fmt.literal "sum="
  %nl = sim.fmt.literal "\0A"
  %one = hw.constant 1 : i64
  %p = llvm.mlir.addressof @dummy : !llvm.ptr

  llhd.process {
    %a = func.call @"uvm_pkg::uvm_component::get_first_child"(%p, %p) : (!llvm.ptr, !llvm.ptr) -> i32
    %b = func.call @"uvm_pkg::uvm_component::get_next_child"(%p, %p) : (!llvm.ptr, !llvm.ptr) -> i32
    %sum = arith.addi %a, %b : i32
    %v = sim.fmt.dec %sum signed : i32
    %msg = sim.fmt.concat (%prefix, %v, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %one
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
