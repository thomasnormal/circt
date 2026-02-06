// RUN: circt-sim %s | FileCheck %s

// Test llhd.sig.extract + llhd.drv on alloca-backed local variables.
// This pattern occurs in UVM's uvm_oneway_hash function where individual
// bits of a local i32 variable are driven through !llhd.ref operations.

// CHECK: result=79764919

hw.module @test() {
  llhd.process {
    // Allocate a local i32 variable (mimics uvm_oneway_hash's hash value)
    %one = llvm.mlir.constant(1 : i64) : i64
    %c0_i32 = hw.constant 0 : i32
    %alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %c0_i32, %alloca : i32, !llvm.ptr

    // Cast to !llhd.ref<i32> (same pattern as MooreToCore local variables)
    %ref = builtin.unrealized_conversion_cast %alloca : !llvm.ptr to !llhd.ref<i32>

    // XOR the stored value with 79764918 (same constant as uvm_oneway_hash)
    %c79764918 = hw.constant 79764918 : i32
    %loaded = llvm.load %alloca : !llvm.ptr -> i32
    %xored = comb.xor %loaded, %c79764918 : i32
    llvm.store %xored, %alloca : i32, !llvm.ptr

    // Now use llhd.sig.extract + llhd.drv to set bit 0 to 1
    // This is the critical pattern that was failing in uvm_oneway_hash
    %c0_i5 = hw.constant 0 : i5
    %true = hw.constant true
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %bit0_ref = llhd.sig.extract %ref from %c0_i5 : <i32> -> <i1>
    llhd.drv %bit0_ref, %true after %eps : i1

    // Read back the result - should be 79764918 | 1 = 79764919
    // (79764918 in binary ends in 0, so setting bit 0 gives 79764919)
    %result = llvm.load %alloca : !llvm.ptr -> i32

    // Print the result
    %fmt_prefix = sim.fmt.literal "result="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %result : i32
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
