// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=STATS
//
// AOT microbenchmark: measures compiled vs interpreted callback throughput.
//
// A single-fire worker process waits 5 ns then runs an arithmetic accumulation
// loop (10 000 iterations of addi, accumulating 0+1+...+9999 = 49 995 000).
// All computation happens within one callback activation, so in-process
// arithmetic dominates over scheduling overhead.  The process has one wait
// and drives an i32 signal (≤64 bit) — eligible for AOT compilation.
//
// To compare wall-clock performance manually:
//   # interpreted
//   time circt-sim aot-microbench.mlir
//   # compiled (AOT)
//   circt-compile aot-microbench.mlir -o /tmp/mb.so
//   time circt-sim aot-microbench.mlir --compiled=/tmp/mb.so
//
// COMPILE: [circt-compile] Compiled 1 process bodies
//
// SIM: result=49995000
//
// STATS: Compiled callback invocations: {{[1-9][0-9]*}}
// STATS: result=49995000

hw.module @test() {
  %c0_i32      = hw.constant 0 : i32
  %c1_i32      = hw.constant 1 : i32
  %c10000_i32  = hw.constant 10000 : i32
  %c5_i64      = hw.constant   5000000 : i64
  %c10_i64     = hw.constant  10000000 : i64
  %c20_i64     = hw.constant  20000000 : i64

  %fmt_res = sim.fmt.literal "result="
  %fmt_nl  = sim.fmt.literal "\0A"

  // Result signal (32-bit accumulator output, starts at 0).
  %result = llhd.sig %c0_i32 : i32

  // Worker: wait 5 ns, accumulate sum(0..9999) = 49 995 000, drive result, halt.
  // Single wait + i32 drive (≤64 bit) → eligible for AOT callback compilation.
  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^entry
  ^entry:
    cf.br ^loop(%c0_i32, %c0_i32 : i32, i32)
  ^loop(%i : i32, %acc : i32):
    %sum    = arith.addi %acc, %i      : i32
    %next_i = arith.addi %i, %c1_i32   : i32
    %done   = arith.cmpi sge, %next_i, %c10000_i32 : i32
    cf.cond_br %done, ^finish(%sum : i32), ^loop(%next_i, %sum : i32, i32)
  ^finish(%final : i32):
    %eps = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %result, %final after %eps : i32
    llhd.halt
  }

  // Reader: print result at t=10 ns (worker fired at t=5 ns + epsilon delay).
  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^print
  ^print:
    %v   = llhd.prb %result : i32
    %fv  = sim.fmt.dec %v : i32
    %out = sim.fmt.concat (%fmt_res, %fv, %fmt_nl)
    sim.proc.print %out
    llhd.halt
  }

  // Terminator at t=20 ns.
  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
