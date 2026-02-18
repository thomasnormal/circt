// RUN: circt-sim %s | FileCheck %s

// Regression: casting a sub-reference (sig.struct_extract result) to !llvm.ptr
// inside a callee must preserve the subfield byte offset. Otherwise llvm.store
// writes through the base struct pointer and corrupts unrelated fields.

// CHECK: Before: a=104, b=0, c=0
// CHECK: After: a=104, b=0, c=1

module {
  func.func private @store_ref_bit_via_ptr(%bit_ref: !llhd.ref<i1>) {
    %true = hw.constant true
    %ptr = builtin.unrealized_conversion_cast %bit_ref : !llhd.ref<i1> to !llvm.ptr
    llvm.store %true, %ptr : i1, !llvm.ptr
    return
  }

  hw.module @test() {
    %one_i64 = llvm.mlir.constant(1 : i64) : i64
    %delay_fs = hw.constant 1000000 : i64

    %c104_i7 = hw.constant 104 : i7
    %c0_i1 = hw.constant false
    %fmt_before = sim.fmt.literal "Before: a="
    %fmt_after = sim.fmt.literal "After: a="
    %fmt_mid_b = sim.fmt.literal ", b="
    %fmt_mid_c = sim.fmt.literal ", c="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %delay = llhd.int_to_time %delay_fs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      %alloc = llvm.alloca %one_i64 x !llvm.struct<(i7, i1, i1)> : (i64) -> !llvm.ptr

      // Initialize the struct to a=104, b=0, c=0.
      %undef = llvm.mlir.undef : !llvm.struct<(i7, i1, i1)>
      %s0 = llvm.insertvalue %c104_i7, %undef[0] : !llvm.struct<(i7, i1, i1)>
      %s1 = llvm.insertvalue %c0_i1, %s0[1] : !llvm.struct<(i7, i1, i1)>
      %s2 = llvm.insertvalue %c0_i1, %s1[2] : !llvm.struct<(i7, i1, i1)>
      llvm.store %s2, %alloc : !llvm.struct<(i7, i1, i1)>, !llvm.ptr

      %ref = builtin.unrealized_conversion_cast %alloc : !llvm.ptr to !llhd.ref<!hw.struct<a: i7, b: i1, c: i1>>

      %before = llhd.prb %ref : !hw.struct<a: i7, b: i1, c: i1>
      %a0 = hw.struct_extract %before["a"] : !hw.struct<a: i7, b: i1, c: i1>
      %b0 = hw.struct_extract %before["b"] : !hw.struct<a: i7, b: i1, c: i1>
      %c0 = hw.struct_extract %before["c"] : !hw.struct<a: i7, b: i1, c: i1>
      %fmt_a0 = sim.fmt.dec %a0 : i7
      %fmt_b0 = sim.fmt.dec %b0 : i1
      %fmt_c0 = sim.fmt.dec %c0 : i1
      %line0 = sim.fmt.concat (%fmt_before, %fmt_a0, %fmt_mid_b, %fmt_b0, %fmt_mid_c, %fmt_c0, %fmt_nl)
      sim.proc.print %line0

      // Pass only field "c" by ref to the callee. The store must not touch "a".
      %c_ref = llhd.sig.struct_extract %ref["c"] : <!hw.struct<a: i7, b: i1, c: i1>>
      func.call @store_ref_bit_via_ptr(%c_ref) : (!llhd.ref<i1>) -> ()

      %after = llhd.prb %ref : !hw.struct<a: i7, b: i1, c: i1>
      %a1 = hw.struct_extract %after["a"] : !hw.struct<a: i7, b: i1, c: i1>
      %b1 = hw.struct_extract %after["b"] : !hw.struct<a: i7, b: i1, c: i1>
      %c1 = hw.struct_extract %after["c"] : !hw.struct<a: i7, b: i1, c: i1>
      %fmt_a1 = sim.fmt.dec %a1 : i7
      %fmt_b1 = sim.fmt.dec %b1 : i1
      %fmt_c1 = sim.fmt.dec %c1 : i1
      %line1 = sim.fmt.concat (%fmt_after, %fmt_a1, %fmt_mid_b, %fmt_b1, %fmt_mid_c, %fmt_c1, %fmt_nl)
      sim.proc.print %line1

      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
