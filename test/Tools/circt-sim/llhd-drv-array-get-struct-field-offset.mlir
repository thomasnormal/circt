// RUN: circt-sim %s | FileCheck %s

// Regression: driving an array element through llhd.sig.array_get on a struct
// field must include the struct field bit offset within the parent signal.
// CHECK: bits0=42 tail=0

hw.module @top() {
  %c0_i8 = arith.constant 0 : i8
  %c42_i8 = arith.constant 42 : i8
  %idx0 = arith.constant 0 : i1

  // bits is not the least-significant field; tail is below it.
  %arr_init = hw.array_create %c0_i8, %c0_i8 : i8
  %s_init = hw.struct_create (%arr_init, %c0_i8) : !hw.struct<bits: !hw.array<2xi8>, tail: i8>
  %s = llhd.sig %s_init : !hw.struct<bits: !hw.array<2xi8>, tail: i8>

  %fmtBits0 = sim.fmt.literal "bits0="
  %fmtTail = sim.fmt.literal " tail="
  %fmtNl = sim.fmt.literal "\0A"

  llhd.process {
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %startFs = hw.constant 1000 : i64
    %startDelay = llhd.int_to_time %startFs
    %nextFs = hw.constant 1 : i64
    %nextDelay = llhd.int_to_time %nextFs

    llhd.wait delay %startDelay, ^bb1
  ^bb1:
    %bitsRef = llhd.sig.struct_extract %s["bits"] : <!hw.struct<bits: !hw.array<2xi8>, tail: i8>>
    %elemRef = llhd.sig.array_get %bitsRef[%idx0] : <!hw.array<2xi8>>
    llhd.drv %elemRef, %c42_i8 after %eps : i8

    llhd.wait delay %nextDelay, ^bb2
  ^bb2:
    %sVal = llhd.prb %s : !hw.struct<bits: !hw.array<2xi8>, tail: i8>
    %bitsVal = hw.struct_extract %sVal["bits"] : !hw.struct<bits: !hw.array<2xi8>, tail: i8>
    %tailVal = hw.struct_extract %sVal["tail"] : !hw.struct<bits: !hw.array<2xi8>, tail: i8>
    %bits0 = hw.array_get %bitsVal[%idx0] : !hw.array<2xi8>, i1

    %fmtBits0Val = sim.fmt.dec %bits0 : i8
    %fmtTailVal = sim.fmt.dec %tailVal : i8
    %out = sim.fmt.concat (%fmtBits0, %fmtBits0Val, %fmtTail, %fmtTailVal, %fmtNl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
