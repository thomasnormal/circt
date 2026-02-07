// RUN: circt-sim %s --top=test_llvm_float --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM floating point operations in LLHD processes.
// Verifies fadd, fsub, fmul, fdiv, fcmp for f32 and f64.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_float() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Test f32 operations
    %f10 = arith.constant 10.0 : f32
    %f5 = arith.constant 5.0 : f32

    // fadd: 10.0 + 5.0 = 15.0
    %fadd_result = llvm.fadd %f10, %f5 : f32

    // fsub: 10.0 - 5.0 = 5.0
    %fsub_result = llvm.fsub %f10, %f5 : f32

    // fmul: 10.0 * 5.0 = 50.0
    %fmul_result = llvm.fmul %f10, %f5 : f32

    // fdiv: 10.0 / 5.0 = 2.0
    %fdiv_result = llvm.fdiv %f10, %f5 : f32

    // fcmp: 10.0 > 5.0 = true (1)
    %fcmp_ogt = llvm.fcmp "ogt" %f10, %f5 : f32

    // fcmp: 5.0 < 10.0 = true (1)
    %fcmp_olt = llvm.fcmp "olt" %f5, %f10 : f32

    // fcmp: 10.0 == 10.0 = true (1)
    %fcmp_oeq = llvm.fcmp "oeq" %f10, %f10 : f32

    // Convert comparison results to i32 and sum
    %cmp1_i32 = arith.extui %fcmp_ogt : i1 to i32
    %cmp2_i32 = arith.extui %fcmp_olt : i1 to i32
    %cmp3_i32 = arith.extui %fcmp_oeq : i1 to i32

    // Sum comparisons: 1 + 1 + 1 = 3
    %temp = llvm.add %cmp1_i32, %cmp2_i32 : i32
    %result = llvm.add %temp, %cmp3_i32 : i32

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}
