// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=4
//
// COMPILED: out=4

func.func @set_struct_field_bit2() -> i8 {
  %one = llvm.mlir.constant(1 : i64) : i64
  %z8 = hw.constant 0 : i8
  %slot = llvm.alloca %one x !llvm.struct<(i8, i8)> : (i64) -> !llvm.ptr
  %base = llvm.mlir.undef : !llvm.struct<(i8, i8)>
  %v0 = llvm.insertvalue %z8, %base[0] : !llvm.struct<(i8, i8)>
  %v1 = llvm.insertvalue %z8, %v0[1] : !llvm.struct<(i8, i8)>
  llvm.store %v1, %slot : !llvm.struct<(i8, i8)>, !llvm.ptr

  %ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!hw.struct<a: i8, b: i8>>
  %a_ref = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: i8, b: i8>>
  %idx = hw.constant 2 : i3
  %bit_ref = llhd.sig.extract %a_ref from %idx : <i8> -> <i1>
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  llhd.drv %bit_ref, %true after %t0 : i1

  %loaded = llvm.load %slot : !llvm.ptr -> !llvm.struct<(i8, i8)>
  %out = llvm.extractvalue %loaded[0] : !llvm.struct<(i8, i8)>
  return %out : i8
}

hw.module @top() {
  llhd.process {
    %x = func.call @set_struct_field_bit2() : () -> i8
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i8
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
