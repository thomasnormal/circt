// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// RUNTIME: Unmapped native func.call policy: default deny uvm_pkg::* and pointer-typed get_/set_/create_/m_initialize* (allow others)
// RUNTIME: Compiled function calls:          1
// RUNTIME: Interpreted function calls:       0
// RUNTIME: direct_calls_native:              1
// RUNTIME: direct_calls_interpreted:         0
// RUNTIME: out=2

llvm.mlir.global internal @"uvm_pkg::uvm_pkg::uvm_build_phase::m_inst"(#llvm.zero) {addr_space = 0 : i32} : !llvm.ptr

func.func @get_2160() -> !llvm.ptr {
  %g = llvm.mlir.addressof @"uvm_pkg::uvm_pkg::uvm_build_phase::m_inst" : !llvm.ptr
  %v = llvm.load %g : !llvm.ptr -> !llvm.ptr
  return %v : !llvm.ptr
}

func.func @add1(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %r = arith.addi %x, %c1 : i32
  return %r : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64
  %null = llvm.mlir.zero : !llvm.ptr

  llhd.process {
    %p = func.call @get_2160() : () -> !llvm.ptr
    %isNull = llvm.icmp "eq" %p, %null : !llvm.ptr
    %i = arith.extui %isNull : i1 to i32
    %o = func.call @add1(%i) : (i32) -> i32
    %vfmt = sim.fmt.dec %o signed : i32
    %msg = sim.fmt.concat (%prefix, %vfmt, %nl)
    sim.proc.print %msg
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
