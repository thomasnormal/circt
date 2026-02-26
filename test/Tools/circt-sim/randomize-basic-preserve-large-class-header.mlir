// RUN: circt-sim %s | FileCheck %s

// REPRO (pre-fix):
// __moore_randomize_basic byte-fills whole class objects. For large UVM-style
// classes this can corrupt non-rand control fields in the class header when
// restore code is missing, breaking sequencer/phase bookkeeping.
//
// This test ensures pointer identity in the large-class header is preserved.

// CHECK: ptr_preserved=1

llvm.func @__moore_class_srandom(!llvm.ptr, i32)
llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

hw.module @test() {
  llhd.process {
    %one = llvm.mlir.constant(1 : i64) : i64
    %class_size = llvm.mlir.constant(136 : i64) : i64
    %seed = llvm.mlir.constant(1 : i32) : i32

    // 15x i64 (120 bytes) + ptr at offset 120 + trailing i64.
    %obj = llvm.alloca %one x !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ptr, i64)> : (i64) -> !llvm.ptr
    %sent = llvm.alloca %one x i8 : (i64) -> !llvm.ptr

    %ptr_field = llvm.getelementptr %obj[0, 15] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ptr, i64)>
    llvm.store %sent, %ptr_field : !llvm.ptr, !llvm.ptr

    llvm.call @__moore_class_srandom(%obj, %seed) : (!llvm.ptr, i32) -> ()
    %rc = llvm.call @__moore_randomize_basic(%obj, %class_size) : (!llvm.ptr, i64) -> i32

    %after = llvm.load %ptr_field : !llvm.ptr -> !llvm.ptr
    %after_i = llvm.ptrtoint %after : !llvm.ptr to i64
    %sent_i = llvm.ptrtoint %sent : !llvm.ptr to i64
    %eq = llvm.icmp "eq" %after_i, %sent_i : i64
    %eq_i64 = llvm.zext %eq : i1 to i64

    %fmt_prefix = sim.fmt.literal "ptr_preserved="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %eq_i64 : i64
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out

    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
