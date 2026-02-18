// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression: driving a struct-field array element through a function-argument
// ref must map HW array indices to LLVM memory indices. Without this, memory-
// backed writes to arr[i] are reversed.

// CHECK: head=99 arr0=17 arr1=34 tail=77

module {
  func.func private @set_array_elem(
      %obj: !llhd.ref<!hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>>,
      %idx: i1, %val: i8) {
    %t = llhd.constant_time <0ns, 0d, 1e>
    %arr_ref = llhd.sig.struct_extract %obj["arr"] : <!hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>>
    %elem_ref = llhd.sig.array_get %arr_ref[%idx] : <!hw.array<2xi8>>
    llhd.drv %elem_ref, %val after %t : i8
    return
  }

  hw.module @test() {
    %one_i64 = llvm.mlir.constant(1 : i64) : i64
    %delay_fs = hw.constant 1000000 : i64

    %c99_i8 = hw.constant 99 : i8
    %c77_i8 = hw.constant 77 : i8
    %c0_i8 = hw.constant 0 : i8
    %c17_i8 = hw.constant 17 : i8
    %c34_i8 = hw.constant 34 : i8
    %c0_i1 = hw.constant false
    %c1_i1 = hw.constant true

    %fmt_prefix = sim.fmt.literal "head="
    %fmt_mid0 = sim.fmt.literal " arr0="
    %fmt_mid1 = sim.fmt.literal " arr1="
    %fmt_mid2 = sim.fmt.literal " tail="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %delay = llhd.int_to_time %delay_fs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      %alloc = llvm.alloca %one_i64 x !llvm.struct<(i8, array<2 x i8>, i8)> : (i64) -> !llvm.ptr

      %arr_undef = llvm.mlir.undef : !llvm.array<2 x i8>
      %arr0 = llvm.insertvalue %c0_i8, %arr_undef[0] : !llvm.array<2 x i8>
      %arr1 = llvm.insertvalue %c0_i8, %arr0[1] : !llvm.array<2 x i8>

      %s_undef = llvm.mlir.undef : !llvm.struct<(i8, array<2 x i8>, i8)>
      %s0 = llvm.insertvalue %c99_i8, %s_undef[0] : !llvm.struct<(i8, array<2 x i8>, i8)>
      %s1 = llvm.insertvalue %arr1, %s0[1] : !llvm.struct<(i8, array<2 x i8>, i8)>
      %s2 = llvm.insertvalue %c77_i8, %s1[2] : !llvm.struct<(i8, array<2 x i8>, i8)>
      llvm.store %s2, %alloc : !llvm.struct<(i8, array<2 x i8>, i8)>, !llvm.ptr

      %obj_ref = builtin.unrealized_conversion_cast %alloc : !llvm.ptr to !llhd.ref<!hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>>

      func.call @set_array_elem(%obj_ref, %c0_i1, %c17_i8) : (!llhd.ref<!hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>>, i1, i8) -> ()
      func.call @set_array_elem(%obj_ref, %c1_i1, %c34_i8) : (!llhd.ref<!hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>>, i1, i8) -> ()

      %loaded = llvm.load %alloc : !llvm.ptr -> !llvm.struct<(i8, array<2 x i8>, i8)>
      %as_hw = builtin.unrealized_conversion_cast %loaded : !llvm.struct<(i8, array<2 x i8>, i8)> to !hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>
      %head = hw.struct_extract %as_hw["head"] : !hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>
      %arr = hw.struct_extract %as_hw["arr"] : !hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>
      %tail = hw.struct_extract %as_hw["tail"] : !hw.struct<head: i8, arr: !hw.array<2xi8>, tail: i8>
      %arr0_v = hw.array_get %arr[%c0_i1] : !hw.array<2xi8>, i1
      %arr1_v = hw.array_get %arr[%c1_i1] : !hw.array<2xi8>, i1

      %f_head = sim.fmt.dec %head : i8
      %f_arr0 = sim.fmt.dec %arr0_v : i8
      %f_arr1 = sim.fmt.dec %arr1_v : i8
      %f_tail = sim.fmt.dec %tail : i8
      %out = sim.fmt.concat (%fmt_prefix, %f_head, %fmt_mid0, %f_arr0, %fmt_mid1, %f_arr1, %fmt_mid2, %f_tail, %fmt_nl)
      sim.proc.print %out

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
