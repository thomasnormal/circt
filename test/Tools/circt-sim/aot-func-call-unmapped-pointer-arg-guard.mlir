// RUN: circt-compile -v %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_next_child circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: even with unmapped-name allowlist, native direct func.call must
// demote when pointer-typed UVM args are unknown/unmapped, instead of
// dereferencing invalid host pointers and crashing.
//
// RUNTIME: Compiled function calls:          0
// RUNTIME: Interpreted function calls:       1
// RUNTIME: direct_calls_native:              0
// RUNTIME: direct_calls_interpreted:         1
// RUNTIME: Top interpreted func.call fallback reasons (top 50):
// RUNTIME: uvm_pkg::uvm_component::get_next_child [unmapped-policy=1]
// RUNTIME: safe=1

llvm.mlir.global internal @dummy_this(0 : i8) : i8

func.func private @"uvm_pkg::uvm_component::get_next_child"(%this: !llvm.ptr,
                                                            %key: !llvm.ptr) -> i32 {
  // Native dispatch would segfault on invalid %key pointer payloads.
  %v = llvm.load %key : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %one = hw.constant 1 : i64
  %ok = hw.constant 1 : i32
  %prefix = sim.fmt.literal "safe="
  %nl = sim.fmt.literal "\0A"

  %this = llvm.mlir.addressof @dummy_this : !llvm.ptr
  %bad = llvm.inttoptr %one : i64 to !llvm.ptr

  llhd.process {
    %r = func.call @"uvm_pkg::uvm_component::get_next_child"(%this, %bad) : (!llvm.ptr, !llvm.ptr) -> i32
    %vr = sim.fmt.dec %ok signed : i32
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
