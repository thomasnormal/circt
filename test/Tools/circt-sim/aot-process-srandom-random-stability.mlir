// RUN: circt-sim %s --top test 2>&1 | FileCheck %s --check-prefix=INTERP
// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | FileCheck %s --check-prefix=AOT

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// INTERP: R_EQ=0
// INTERP: U_EQ=1
// AOT: R_EQ=0
// AOT: U_EQ=1

llvm.func @__moore_process_self() -> i64
llvm.func @__moore_process_srandom(i64, i32)
llvm.func @__moore_random() -> i32
llvm.func @__moore_urandom() -> i32

func.func @run() -> i32 {
  %seed = arith.constant 111 : i32
  %handle = llvm.call @__moore_process_self() : () -> i64
  llvm.call @__moore_process_srandom(%handle, %seed) : (i64, i32) -> ()
  %r1 = llvm.call @__moore_random() : () -> i32
  llvm.call @__moore_process_srandom(%handle, %seed) : (i64, i32) -> ()
  %r2 = llvm.call @__moore_random() : () -> i32
  %r_eq = arith.cmpi eq, %r1, %r2 : i32
  %r_eq_i32 = arith.extui %r_eq : i1 to i32

  llvm.call @__moore_process_srandom(%handle, %seed) : (i64, i32) -> ()
  %u1 = llvm.call @__moore_urandom() : () -> i32
  llvm.call @__moore_process_srandom(%handle, %seed) : (i64, i32) -> ()
  %u2 = llvm.call @__moore_urandom() : () -> i32
  %u_eq = arith.cmpi eq, %u1, %u2 : i32
  %u_eq_i32 = arith.extui %u_eq : i1 to i32
  // Pack two flags into one i32 result:
  // bit0 = random_equal, bit1 = urandom_equal.
  %c1 = arith.constant 1 : i32
  %u_shift = arith.shli %u_eq_i32, %c1 : i32
  %result = arith.ori %r_eq_i32, %u_shift : i32
  return %result : i32
}

hw.module @test() {
  llhd.process {
    %packed = func.call @run() : () -> i32
    %c1 = arith.constant 1 : i32
    %r_eq = arith.andi %packed, %c1 : i32
    %u_sh = arith.shrui %packed, %c1 : i32
    %u_eq = arith.andi %u_sh, %c1 : i32

    %lit_r = sim.fmt.literal "R_EQ="
    %dr = sim.fmt.dec %r_eq signed : i32
    %nl = sim.fmt.literal "\0A"
    %fmt_r = sim.fmt.concat (%lit_r, %dr, %nl)
    sim.proc.print %fmt_r

    %lit_u = sim.fmt.literal "U_EQ="
    %du = sim.fmt.dec %u_eq signed : i32
    %fmt_u = sim.fmt.concat (%lit_u, %du, %nl)
    sim.proc.print %fmt_u
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
