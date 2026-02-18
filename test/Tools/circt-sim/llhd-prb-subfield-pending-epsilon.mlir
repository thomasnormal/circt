// RUN: circt-sim %s | FileCheck %s

// Regression: probing llhd.sig.struct_extract / llhd.sig.array_get refs must
// not bypass subfield extraction when parent pending epsilon drives exist.
// CHECK: field_a=13 elem0=42

hw.module @top() {
  %c0_i8 = arith.constant 0 : i8
  %c13_i8 = arith.constant 13 : i8
  %c42_i8 = arith.constant 42 : i8
  %idx0 = arith.constant 0 : i1
  %arr_init = hw.array_create %c0_i8, %c0_i8 : i8
  %s_init = hw.struct_create (%c0_i8, %arr_init) : !hw.struct<a: i8, bits: !hw.array<2xi8>>
  %s = llhd.sig %s_init : !hw.struct<a: i8, bits: !hw.array<2xi8>>

  %fmtA = sim.fmt.literal "field_a="
  %fmtB = sim.fmt.literal " elem0="
  %fmtNl = sim.fmt.literal "\0A"

  llhd.process {
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %startFs = hw.constant 1000 : i64
    %startDelay = llhd.int_to_time %startFs
    llhd.wait delay %startDelay, ^bb1
  ^bb1:
    %aRef = llhd.sig.struct_extract %s["a"] : <!hw.struct<a: i8, bits: !hw.array<2xi8>>>
    llhd.drv %aRef, %c13_i8 after %eps : i8

    %bitsRef = llhd.sig.struct_extract %s["bits"] : <!hw.struct<a: i8, bits: !hw.array<2xi8>>>
    %elemRef = llhd.sig.array_get %bitsRef[%idx0] : <!hw.array<2xi8>>
    llhd.drv %elemRef, %c42_i8 after %eps : i8

    %aProbeRef = llhd.sig.struct_extract %s["a"] : <!hw.struct<a: i8, bits: !hw.array<2xi8>>>
    %aVal = llhd.prb %aProbeRef : i8
    %bitsProbeRef = llhd.sig.struct_extract %s["bits"] : <!hw.struct<a: i8, bits: !hw.array<2xi8>>>
    %elemProbeRef = llhd.sig.array_get %bitsProbeRef[%idx0] : <!hw.array<2xi8>>
    %elemVal = llhd.prb %elemProbeRef : i8

    %fmtAVal = sim.fmt.dec %aVal : i8
    %fmtElemVal = sim.fmt.dec %elemVal : i8
    %out = sim.fmt.concat (%fmtA, %fmtAVal, %fmtB, %fmtElemVal, %fmtNl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
