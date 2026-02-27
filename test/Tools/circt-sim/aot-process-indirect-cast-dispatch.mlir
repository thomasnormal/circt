// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Ensure callback process eligibility accepts safe
// builtin.unrealized_conversion_cast patterns used by func.call_indirect.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
// COMPILE: [circt-compile] Processes: 3 total, 1 callback-eligible, 2 rejected
// RUNTIME-NOT: FATAL: trampoline dispatch
// RUNTIME: a=1

llvm.func private @set_true() -> i1 {
  %true = llvm.mlir.constant(true) : i1
  llvm.return %true : i1
}

hw.module @test() {
  %false = hw.constant false
  %c5_i64 = hw.constant 5000000 : i64
  %c7_i64 = hw.constant 7000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_a = sim.fmt.literal "a="
  %fmt_nl = sim.fmt.literal "\0A"

  %a = llhd.sig %false : i1

  // Driver: invokes call_indirect through a ptr->function unrealized cast.
  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^set
  ^set:
    %fn_ptr = llvm.mlir.addressof @set_true : !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fn_ptr : !llvm.ptr to () -> i1
    %v = func.call_indirect %fn() : () -> i1
    llhd.drv %a, %v after %eps : i1
    llhd.halt
  }

  // Reader at t=7ns (kept to ensure there is still a rejected process).
  llhd.process {
    %d = llhd.int_to_time %c7_i64
    llhd.wait delay %d, ^read
  ^read:
    %a1 = llhd.prb %a : i1
    %v = sim.fmt.dec %a1 : i1
    %o = sim.fmt.concat (%fmt_a, %v, %fmt_nl)
    sim.proc.print %o
    llhd.halt
  }

  // Terminator at t=20ns.
  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
