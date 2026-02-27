// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=6
//
// COMPILED: out=6

func.func @bump_struct_field() -> i32 {
  %one = llvm.mlir.constant(1 : i64) : i64
  %base = llvm.mlir.undef : !llvm.struct<(i32, i64)>
  %c5 = hw.constant 5 : i32
  %c9 = hw.constant 9 : i64
  %c1 = hw.constant 1 : i32
  %t0 = llhd.constant_time <0ns, 0d, 1e>

  %slot = llvm.alloca %one x !llvm.struct<(i32, i64)> : (i64) -> !llvm.ptr
  %v0 = llvm.insertvalue %c5, %base[0] : !llvm.struct<(i32, i64)>
  %v1 = llvm.insertvalue %c9, %v0[1] : !llvm.struct<(i32, i64)>
  llvm.store %v1, %slot : !llvm.struct<(i32, i64)>, !llvm.ptr

  %ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!hw.struct<a: i32, b: i64>>
  %a_ref = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: i32, b: i64>>
  %a = llhd.prb %a_ref : i32
  %next = comb.add %a, %c1 : i32
  llhd.drv %a_ref, %next after %t0 : i32

  %loaded = llvm.load %slot : !llvm.ptr -> !llvm.struct<(i32, i64)>
  %out = llvm.extractvalue %loaded[0] : !llvm.struct<(i32, i64)>
  return %out : i32
}

hw.module @top() {
  llhd.process {
    %x = func.call @bump_struct_field() : () -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
