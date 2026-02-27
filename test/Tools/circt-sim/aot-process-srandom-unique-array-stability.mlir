// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --top test 2>&1 | FileCheck %s --check-prefix=INTERP
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | FileCheck %s --check-prefix=AOT

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// INTERP: EQ=1
// AOT: EQ=1

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @__moore_process_self() -> i64
llvm.func @__moore_process_srandom(i64, i32)
llvm.func @__moore_randomize_unique_array(!llvm.ptr, i64, i64, i64, i64) -> i32

func.func @run() -> i32 {
  %c20 = arith.constant 20 : i64
  %num = arith.constant 5 : i64
  %elem = arith.constant 4 : i64
  %min = arith.constant 0 : i64
  %max = arith.constant 31 : i64
  %seed = arith.constant 2026 : i32
  %c1 = arith.constant 1 : i32

  %h = llvm.call @__moore_process_self() : () -> i64
  %data = llvm.call @malloc(%c20) : (i64) -> !llvm.ptr
  %saved = llvm.call @malloc(%c20) : (i64) -> !llvm.ptr

  // First randomization with fixed seed.
  llvm.call @__moore_process_srandom(%h, %seed) : (i64, i32) -> ()
  %rc1 = llvm.call @__moore_randomize_unique_array(%data, %num, %elem, %min, %max)
      : (!llvm.ptr, i64, i64, i64, i64) -> i32

  // Save first randomized payload.
  %d0 = llvm.getelementptr %data[0, 0]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %d1 = llvm.getelementptr %data[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %d2 = llvm.getelementptr %data[0, 2]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %d3 = llvm.getelementptr %data[0, 3]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %d4 = llvm.getelementptr %data[0, 4]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %s0 = llvm.getelementptr %saved[0, 0]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %s1 = llvm.getelementptr %saved[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %s2 = llvm.getelementptr %saved[0, 2]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %s3 = llvm.getelementptr %saved[0, 3]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %s4 = llvm.getelementptr %saved[0, 4]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i32)>
  %v0 = llvm.load %d0 : !llvm.ptr -> i32
  %v1 = llvm.load %d1 : !llvm.ptr -> i32
  %v2 = llvm.load %d2 : !llvm.ptr -> i32
  %v3 = llvm.load %d3 : !llvm.ptr -> i32
  %v4 = llvm.load %d4 : !llvm.ptr -> i32
  llvm.store %v0, %s0 : i32, !llvm.ptr
  llvm.store %v1, %s1 : i32, !llvm.ptr
  llvm.store %v2, %s2 : i32, !llvm.ptr
  llvm.store %v3, %s3 : i32, !llvm.ptr
  llvm.store %v4, %s4 : i32, !llvm.ptr

  // Reseed and randomize again; should replay same payload.
  llvm.call @__moore_process_srandom(%h, %seed) : (i64, i32) -> ()
  %rc2 = llvm.call @__moore_randomize_unique_array(%data, %num, %elem, %min, %max)
      : (!llvm.ptr, i64, i64, i64, i64) -> i32
  %w0 = llvm.load %d0 : !llvm.ptr -> i32
  %w1 = llvm.load %d1 : !llvm.ptr -> i32
  %w2 = llvm.load %d2 : !llvm.ptr -> i32
  %w3 = llvm.load %d3 : !llvm.ptr -> i32
  %w4 = llvm.load %d4 : !llvm.ptr -> i32
  %u0 = llvm.load %s0 : !llvm.ptr -> i32
  %u1 = llvm.load %s1 : !llvm.ptr -> i32
  %u2 = llvm.load %s2 : !llvm.ptr -> i32
  %u3 = llvm.load %s3 : !llvm.ptr -> i32
  %u4 = llvm.load %s4 : !llvm.ptr -> i32

  %rc1_ok = arith.cmpi eq, %rc1, %c1 : i32
  %rc2_ok = arith.cmpi eq, %rc2, %c1 : i32
  %e0 = arith.cmpi eq, %w0, %u0 : i32
  %e1 = arith.cmpi eq, %w1, %u1 : i32
  %e2 = arith.cmpi eq, %w2, %u2 : i32
  %e3 = arith.cmpi eq, %w3, %u3 : i32
  %e4 = arith.cmpi eq, %w4, %u4 : i32
  %a01 = arith.andi %e0, %e1 : i1
  %a23 = arith.andi %e2, %e3 : i1
  %a4r = arith.andi %e4, %rc1_ok : i1
  %a0123 = arith.andi %a01, %a23 : i1
  %a_payload = arith.andi %a0123, %a4r : i1
  %all = arith.andi %a_payload, %rc2_ok : i1
  %eq_i32 = arith.extui %all : i1 to i32
  return %eq_i32 : i32
}

hw.module @test() {
  llhd.process {
    %eq = func.call @run() : () -> i32
    %lit = sim.fmt.literal "EQ="
    %d = sim.fmt.dec %eq signed : i32
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%lit, %d, %nl)
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
