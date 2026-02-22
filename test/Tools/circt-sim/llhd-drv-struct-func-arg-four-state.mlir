// RUN: circt-sim %s --top top | FileCheck %s

// Regression: driving a struct-typed field through llhd.sig.struct_extract on
// a memory-backed !llhd.ref function argument must preserve LLVM/HW bit layout.
// Without conversion in the function-arg fallback path, LOAD_A reads as 0.

// CHECK: PRB_A=1
// CHECK: LOAD_A=1 B=0

module {
  func.func private @set_a(%arg0: !llhd.ref<!hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>>) {
    %time = llhd.constant_time <0ns, 0d, 1e>
    %a_val = hw.aggregate_constant [1 : i2, 0 : i2] : !hw.struct<value: i2, unknown: i2>
    %a_ref = llhd.sig.struct_extract %arg0["a"] : <!hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>>
    llhd.drv %a_ref, %a_val after %time : !hw.struct<value: i2, unknown: i2>
    return
  }

  hw.module @top() {
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c0_i2 = llvm.mlir.constant(0 : i2) : i2

    %fmt_prb = sim.fmt.literal "PRB_A="
    %fmt_load = sim.fmt.literal "LOAD_A="
    %fmt_sep = sim.fmt.literal " B="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %alloca = llvm.alloca %c1_i64 x !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)> : (i64) -> !llvm.ptr

      %undef_pair = llvm.mlir.undef : !llvm.struct<(i2, i2)>
      %pair0 = llvm.insertvalue %c0_i2, %undef_pair[0] : !llvm.struct<(i2, i2)>
      %pair1 = llvm.insertvalue %c0_i2, %pair0[1] : !llvm.struct<(i2, i2)>
      %undef_outer = llvm.mlir.undef : !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)>
      %s0 = llvm.insertvalue %pair1, %undef_outer[0] : !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)>
      %s1 = llvm.insertvalue %pair1, %s0[1] : !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)>
      llvm.store %s1, %alloca : !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)>, !llvm.ptr

      %ref = builtin.unrealized_conversion_cast %alloca : !llvm.ptr to !llhd.ref<!hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>>
      func.call @set_a(%ref) : (!llhd.ref<!hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>>) -> ()

      // Probe via llhd.prb on the extracted field ref.
      %a_ref = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>>
      %a_probe = llhd.prb %a_ref : !hw.struct<value: i2, unknown: i2>
      %a_probe_value = hw.struct_extract %a_probe["value"] : !hw.struct<value: i2, unknown: i2>
      %a_probe_fmt = sim.fmt.dec %a_probe_value specifierWidth 0 : i2
      %prb_out = sim.fmt.concat (%fmt_prb, %a_probe_fmt, %fmt_nl)
      sim.proc.print %prb_out

      // Also read back via llvm.load + cast to verify memory layout consistency.
      %raw = llvm.load %alloca : !llvm.ptr -> !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)>
      %val = builtin.unrealized_conversion_cast %raw : !llvm.struct<(struct<(i2, i2)>, struct<(i2, i2)>)> to !hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>
      %a = hw.struct_extract %val["a"] : !hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>
      %a_value = hw.struct_extract %a["value"] : !hw.struct<value: i2, unknown: i2>
      %b = hw.struct_extract %val["b"] : !hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.struct<value: i2, unknown: i2>>
      %b_value = hw.struct_extract %b["value"] : !hw.struct<value: i2, unknown: i2>
      %a_fmt = sim.fmt.dec %a_value specifierWidth 0 : i2
      %b_fmt = sim.fmt.dec %b_value specifierWidth 0 : i2
      %load_out = sim.fmt.concat (%fmt_load, %a_fmt, %fmt_sep, %b_fmt, %fmt_nl)
      sim.proc.print %load_out

      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
