// RUN: circt-sim %s | FileCheck %s

// REPRO (pre-fix):
// __moore_randomize_basic byte-fills class storage and can leave rand dynamic
// array descriptors ({ptr,len}) as garbage. That creates huge bogus lengths and
// stalls AVIP sequences in interpreted mode at time 0.
//
// This test seeds RNG deterministically, randomizes a class object that
// contains a dynamic-array descriptor field, and checks that the descriptor is
// sanitized to a small valid allocation.

// CHECK: len_after_randomize=1

llvm.func @__moore_class_srandom(!llvm.ptr, i32)
llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

hw.module @test() {
  llhd.process {
    %one = llvm.mlir.constant(1 : i64) : i64
    %class_size = llvm.mlir.constant(16 : i64) : i64
    %seed = llvm.mlir.constant(1 : i32) : i32

    // class { rand bit [7:0] dyn[]; }
    // Lowered representation includes a dynamic-array descriptor field.
    %obj = llvm.alloca %one x !llvm.struct<(!llvm.struct<(ptr, i64)>)> : (i64) -> !llvm.ptr

    // Keep a GEP user so randomize_basic discovers the descriptor field.
    %field_ptr = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.struct<(ptr, i64)>)>

    llvm.call @__moore_class_srandom(%obj, %seed) : (!llvm.ptr, i32) -> ()
    %rc = llvm.call @__moore_randomize_basic(%obj, %class_size) : (!llvm.ptr, i64) -> i32

    %desc = llvm.load %field_ptr : !llvm.ptr -> !llvm.struct<(ptr, i64)>
    %len = llvm.extractvalue %desc[1] : !llvm.struct<(ptr, i64)>

    %fmt_prefix = sim.fmt.literal "len_after_randomize="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %len : i64
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out

    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
