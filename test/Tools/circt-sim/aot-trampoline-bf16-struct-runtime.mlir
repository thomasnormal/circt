// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: trampoline struct packing must use exact float bit widths.
// bf16 fields in LLVM structs were previously treated as 64-bit in
// dispatchTrampoline packing, corrupting field layout in compiled mode.
//
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
// SIM: ok=-1
// COMPILED: ok=-1

module {
  // Force trampoline by making function non-compilable.
  func.func @"uvm_pkg::uvm_demo::bf16_pair"() -> !llvm.struct<(bf16, bf16)> {
    %true = arith.constant true
    verif.assume %true : i1

    %f0 = arith.constant 1.0 : bf16
    %f1 = arith.constant 2.0 : bf16
    %undef = llvm.mlir.undef : !llvm.struct<(bf16, bf16)>
    %s0 = llvm.insertvalue %f0, %undef[0] : !llvm.struct<(bf16, bf16)>
    %s1 = llvm.insertvalue %f1, %s0[1] : !llvm.struct<(bf16, bf16)>
    return %s1 : !llvm.struct<(bf16, bf16)>
  }

  // Compile this function; it calls demoted callee via generated trampoline.
  func.func @caller() -> i1 {
    %s = func.call @"uvm_pkg::uvm_demo::bf16_pair"() : () -> !llvm.struct<(bf16, bf16)>
    %f = llvm.extractvalue %s[1] : !llvm.struct<(bf16, bf16)>
    %bits = llvm.bitcast %f : bf16 to i16
    %exp = arith.constant 16384 : i16 // 0x4000 == bf16(2.0)
    %ok = arith.cmpi eq, %bits, %exp : i16
    return %ok : i1
  }

  hw.module @test() {
    %fmt_ok = sim.fmt.literal "ok="
    %fmt_nl = sim.fmt.literal "\0A"

    %c10_i64 = hw.constant 10000000 : i64

    llhd.process {
      %ok = func.call @caller() : () -> i1
      %fok = sim.fmt.dec %ok signed : i1
      %msg = sim.fmt.concat (%fmt_ok, %fok, %fmt_nl)
      sim.proc.print %msg
      llhd.halt
    }

    llhd.process {
      %d = llhd.int_to_time %c10_i64
      llhd.wait delay %d, ^done
    ^done:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
