// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression: probing llhd.sig.struct_extract from a function-argument ref
// backed by memory must observe same-process field drives immediately.
// This exercises block-argument remapping through a call chain:
//   top -> forward -> bump_twice
// where bump_twice performs prb/drv/prb/drv/prb on a struct subref.
//
// CHECK: inside_count=2
// CHECK: final_count=2

module {
  func.func private @bump_twice(%obj: !llhd.ref<!hw.struct<count: i8>>) {
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %one = hw.constant 1 : i8

    %count_ref = llhd.sig.struct_extract %obj["count"] : <!hw.struct<count: i8>>

    %v0 = llhd.prb %count_ref : i8
    %v1 = comb.add %v0, %one : i8
    llhd.drv %count_ref, %v1 after %eps : i8

    // The second probe must observe the first drive (value 1), not stale 0.
    %p1 = llhd.prb %count_ref : i8
    %v2 = comb.add %p1, %one : i8
    llhd.drv %count_ref, %v2 after %eps : i8

    %p2 = llhd.prb %count_ref : i8
    %fmtInside = sim.fmt.literal "inside_count="
    %fmtNl = sim.fmt.literal "\0A"
    %fmtP2 = sim.fmt.dec %p2 : i8
    %outInside = sim.fmt.concat (%fmtInside, %fmtP2, %fmtNl)
    sim.proc.print %outInside
    return
  }

  func.func private @forward(%obj: !llhd.ref<!hw.struct<count: i8>>) {
    func.call @bump_twice(%obj) : (!llhd.ref<!hw.struct<count: i8>>) -> ()
    return
  }

  hw.module @test() {
    %one_i64 = llvm.mlir.constant(1 : i64) : i64
    %zero_i8 = hw.constant 0 : i8
    %start_fs = hw.constant 1000000 : i64

    llhd.process {
      %delay = llhd.int_to_time %start_fs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      %alloc = llvm.alloca %one_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr

      %init_undef = llvm.mlir.undef : !llvm.struct<(i8)>
      %init = llvm.insertvalue %zero_i8, %init_undef[0] : !llvm.struct<(i8)>
      llvm.store %init, %alloc : !llvm.struct<(i8)>, !llvm.ptr

      %obj = builtin.unrealized_conversion_cast %alloc : !llvm.ptr to !llhd.ref<!hw.struct<count: i8>>
      func.call @forward(%obj) : (!llhd.ref<!hw.struct<count: i8>>) -> ()

      %count_ref = llhd.sig.struct_extract %obj["count"] : <!hw.struct<count: i8>>
      %count = llhd.prb %count_ref : i8

      %fmtFinal = sim.fmt.literal "final_count="
      %fmtNl = sim.fmt.literal "\0A"
      %fmtCount = sim.fmt.dec %count : i8
      %outFinal = sim.fmt.concat (%fmtFinal, %fmtCount, %fmtNl)
      sim.proc.print %outFinal

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
