// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --top test 2>&1 | FileCheck %s --check-prefix=INTERP
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | FileCheck %s --check-prefix=AOT

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// INTERP: OK=1
// AOT: OK=1

llvm.func @__moore_process_self() -> i64
llvm.func @__moore_process_srandom(i64, i32)
llvm.func @__moore_randomize_with_modulo(i64, i64) -> i64

func.func @run() -> i32 {
  %h = llvm.call @__moore_process_self() : () -> i64
  %seed = arith.constant 2026 : i32
  %mod = arith.constant 13 : i64
  %rem = arith.constant 7 : i64

  llvm.call @__moore_process_srandom(%h, %seed) : (i64, i32) -> ()
  %v1 = llvm.call @__moore_randomize_with_modulo(%mod, %rem) : (i64, i64) -> i64
  llvm.call @__moore_process_srandom(%h, %seed) : (i64, i32) -> ()
  %v2 = llvm.call @__moore_randomize_with_modulo(%mod, %rem) : (i64, i64) -> i64

  %same = arith.cmpi eq, %v1, %v2 : i64
  %r1 = arith.remsi %v1, %mod : i64
  %okRem = arith.cmpi eq, %r1, %rem : i64
  %all = arith.andi %same, %okRem : i1
  %ok = arith.extui %all : i1 to i32
  return %ok : i32
}

hw.module @test() {
  llhd.process {
    %ok = func.call @run() : () -> i32
    %lit = sim.fmt.literal "OK="
    %d = sim.fmt.dec %ok signed : i32
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%lit, %d, %nl)
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
