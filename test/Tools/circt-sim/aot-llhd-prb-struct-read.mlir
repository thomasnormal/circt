// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=5
//
// COMPILED: out=5

func.func @prb_struct_value() -> i32 {
  %v = hw.constant 5 : i32
  %u = hw.constant 9 : i32
  %init = hw.struct_create (%v, %u) : !hw.struct<value: i32, unknown: i32>
  %sig = llhd.sig %init : !hw.struct<value: i32, unknown: i32>
  %snap = llhd.prb %sig : !hw.struct<value: i32, unknown: i32>
  %out = hw.struct_extract %snap["value"] : !hw.struct<value: i32, unknown: i32>
  return %out : i32
}

hw.module @top() {
  llhd.process {
    %x = func.call @prb_struct_value() : () -> i32
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
