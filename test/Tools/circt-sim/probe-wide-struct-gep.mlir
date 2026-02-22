// RUN: circt-sim %s --top top | FileCheck %s

// Regression: probing a struct wider than 64 bits via a GEP-backed ref must
// use APInt (not uint64_t) to avoid silent truncation of upper bits.
// The struct is {i64, i32} = 96 bits total. Without the fix, only the lower
// 64 bits were read and the i32 field was lost.

// CHECK: A=42 B=99

module {
  hw.module @top() {
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c42_i64 = llvm.mlir.constant(42 : i64) : i64
    %c99_i32 = llvm.mlir.constant(99 : i32) : i32

    %fmt_a = sim.fmt.literal "A="
    %fmt_b = sim.fmt.literal " B="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Allocate a struct with two fields: i64 + i32 = 96 bits (>64 bits)
      %alloca = llvm.alloca %c1_i64 x !llvm.struct<(i64, i32)> : (i64) -> !llvm.ptr

      // Store values into the struct
      %undef = llvm.mlir.undef : !llvm.struct<(i64, i32)>
      %s0 = llvm.insertvalue %c42_i64, %undef[0] : !llvm.struct<(i64, i32)>
      %s1 = llvm.insertvalue %c99_i32, %s0[1] : !llvm.struct<(i64, i32)>
      llvm.store %s1, %alloca : !llvm.struct<(i64, i32)>, !llvm.ptr

      // Create a GEP to the struct (field 0 of the outer allocation)
      %gep = llvm.getelementptr %alloca[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i32)>

      // Cast GEP result to !llhd.ref and probe it
      %ref = builtin.unrealized_conversion_cast %gep : !llvm.ptr to !llhd.ref<!hw.struct<a: i64, b: i32>>
      %prb = llhd.prb %ref : !hw.struct<a: i64, b: i32>

      // Extract both fields and print
      %a = hw.struct_extract %prb["a"] : !hw.struct<a: i64, b: i32>
      %b = hw.struct_extract %prb["b"] : !hw.struct<a: i64, b: i32>

      %a_fmt = sim.fmt.dec %a specifierWidth 0 : i64
      %b_fmt = sim.fmt.dec %b specifierWidth 0 : i32
      %out = sim.fmt.concat (%fmt_a, %a_fmt, %fmt_b, %b_fmt, %fmt_nl)
      sim.proc.print %out

      llhd.halt
    }
  }
}
