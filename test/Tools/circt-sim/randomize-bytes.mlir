// RUN: circt-sim %s --top test 2>&1 | FileCheck %s

// CHECK: rc1 = 1
// CHECK: rc2 = 1
// CHECK: same_seed_equal = 1
// CHECK: nonzero = 1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @__moore_urandom_seeded(i32) -> i32
  llvm.func @__moore_randomize_bytes(!llvm.ptr, i64) -> i32

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i64 = arith.constant 5 : i64
    %seed_i32 = arith.constant 1234 : i32

    llhd.process {
      %ptr1 = llvm.call @malloc(%c5_i64) : (i64) -> !llvm.ptr
      %ptr2 = llvm.call @malloc(%c5_i64) : (i64) -> !llvm.ptr

      %_seed0 = llvm.call @__moore_urandom_seeded(%seed_i32) : (i32) -> i32
      %rc1 = llvm.call @__moore_randomize_bytes(%ptr1, %c5_i64) : (!llvm.ptr, i64) -> i32
      %_seed1 = llvm.call @__moore_urandom_seeded(%seed_i32) : (i32) -> i32
      %rc2 = llvm.call @__moore_randomize_bytes(%ptr2, %c5_i64) : (!llvm.ptr, i64) -> i32

      %p1_0 = llvm.getelementptr %ptr1[0] : (!llvm.ptr) -> !llvm.ptr, i8
      %p1_1 = llvm.getelementptr %ptr1[1] : (!llvm.ptr) -> !llvm.ptr, i8
      %p1_2 = llvm.getelementptr %ptr1[2] : (!llvm.ptr) -> !llvm.ptr, i8
      %p1_3 = llvm.getelementptr %ptr1[3] : (!llvm.ptr) -> !llvm.ptr, i8
      %p1_4 = llvm.getelementptr %ptr1[4] : (!llvm.ptr) -> !llvm.ptr, i8
      %p2_0 = llvm.getelementptr %ptr2[0] : (!llvm.ptr) -> !llvm.ptr, i8
      %p2_1 = llvm.getelementptr %ptr2[1] : (!llvm.ptr) -> !llvm.ptr, i8
      %p2_2 = llvm.getelementptr %ptr2[2] : (!llvm.ptr) -> !llvm.ptr, i8
      %p2_3 = llvm.getelementptr %ptr2[3] : (!llvm.ptr) -> !llvm.ptr, i8
      %p2_4 = llvm.getelementptr %ptr2[4] : (!llvm.ptr) -> !llvm.ptr, i8

      %b1_0 = llvm.load %p1_0 : !llvm.ptr -> i8
      %b1_1 = llvm.load %p1_1 : !llvm.ptr -> i8
      %b1_2 = llvm.load %p1_2 : !llvm.ptr -> i8
      %b1_3 = llvm.load %p1_3 : !llvm.ptr -> i8
      %b1_4 = llvm.load %p1_4 : !llvm.ptr -> i8
      %b2_0 = llvm.load %p2_0 : !llvm.ptr -> i8
      %b2_1 = llvm.load %p2_1 : !llvm.ptr -> i8
      %b2_2 = llvm.load %p2_2 : !llvm.ptr -> i8
      %b2_3 = llvm.load %p2_3 : !llvm.ptr -> i8
      %b2_4 = llvm.load %p2_4 : !llvm.ptr -> i8

      %eq0 = comb.icmp eq %b1_0, %b2_0 : i8
      %eq1 = comb.icmp eq %b1_1, %b2_1 : i8
      %eq2 = comb.icmp eq %b1_2, %b2_2 : i8
      %eq3 = comb.icmp eq %b1_3, %b2_3 : i8
      %eq4 = comb.icmp eq %b1_4, %b2_4 : i8
      %eq01 = comb.and %eq0, %eq1 : i1
      %eq23 = comb.and %eq2, %eq3 : i1
      %eq0123 = comb.and %eq01, %eq23 : i1
      %allEq = comb.and %eq0123, %eq4 : i1

      %nz0 = comb.icmp ne %b1_0, %c0_i8 : i8
      %nz1 = comb.icmp ne %b1_1, %c0_i8 : i8
      %nz2 = comb.icmp ne %b1_2, %c0_i8 : i8
      %nz3 = comb.icmp ne %b1_3, %c0_i8 : i8
      %nz4 = comb.icmp ne %b1_4, %c0_i8 : i8
      %nz01 = comb.or %nz0, %nz1 : i1
      %nz23 = comb.or %nz2, %nz3 : i1
      %nz0123 = comb.or %nz01, %nz23 : i1
      %anyNZ = comb.or %nz0123, %nz4 : i1

      %same_i32 = arith.select %allEq, %c1_i32, %c0_i32 : i32
      %nz_i32 = arith.select %anyNZ, %c1_i32, %c0_i32 : i32

      %nl = sim.fmt.literal "\0A"
      %lit_rc1 = sim.fmt.literal "rc1 = "
      %dec_rc1 = sim.fmt.dec %rc1 signed : i32
      %fmt_rc1 = sim.fmt.concat (%lit_rc1, %dec_rc1, %nl)
      sim.proc.print %fmt_rc1

      %lit_rc2 = sim.fmt.literal "rc2 = "
      %dec_rc2 = sim.fmt.dec %rc2 signed : i32
      %fmt_rc2 = sim.fmt.concat (%lit_rc2, %dec_rc2, %nl)
      sim.proc.print %fmt_rc2

      %lit_same = sim.fmt.literal "same_seed_equal = "
      %dec_same = sim.fmt.dec %same_i32 signed : i32
      %fmt_same = sim.fmt.concat (%lit_same, %dec_same, %nl)
      sim.proc.print %fmt_same

      %lit_nz = sim.fmt.literal "nonzero = "
      %dec_nz = sim.fmt.dec %nz_i32 signed : i32
      %fmt_nz = sim.fmt.concat (%lit_nz, %dec_nz, %nl)
      sim.proc.print %fmt_nz

      llvm.call @free(%ptr1) : (!llvm.ptr) -> ()
      llvm.call @free(%ptr2) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
