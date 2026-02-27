// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: {{.*}} 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=5
//
// COMPILED: out=5

func.func @sig_bitcast_init_read() -> i32 {
  %packed = hw.constant 21474836489 : i64
  %init = hw.bitcast %packed : (i64) -> !hw.struct<value: i32, unknown: i32>
  %sig = llhd.sig %init : !hw.struct<value: i32, unknown: i32>
  %ptr = builtin.unrealized_conversion_cast %sig : !llhd.ref<!hw.struct<value: i32, unknown: i32>> to !llvm.ptr
  %loaded = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i32, i32)>
  %out = llvm.extractvalue %loaded[0] : !llvm.struct<(i32, i32)>
  return %out : i32
}

hw.module @top() {
  llhd.process {
    %x = func.call @sig_bitcast_init_read() : () -> i32
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
