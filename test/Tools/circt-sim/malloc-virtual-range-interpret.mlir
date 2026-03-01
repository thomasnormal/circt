// RUN: circt-sim --mode=interpret %s | FileCheck %s
// RUN: circt-compile %s -o %t.so
// RUN: env CIRCT_AOT_NO_FUNC_DISPATCH=1 CIRCT_AOT_DISABLE_COMPILED_PROCESSES=1 circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=AOT-NODISPATCH

// In interpreted mode, malloc-backed class/object pointers must stay in the
// interpreter virtual range. UVM phase/domain containers often store pointer
// payloads in 32-bit slots, so host pointers can corrupt lookups.
//
// The same invariant must hold when a compiled module is loaded but function/
// process dispatch is effectively interpreted-only (no native dispatch).

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  func.func private @alloc_and_check_range() -> i1 {
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %lo = llvm.mlir.constant(268435456 : i64) : i64      // 0x1000_0000
    %hi = llvm.mlir.constant(4026531840 : i64) : i64     // 0xF000_0000

    %ptr = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr
    %ptr_i64 = llvm.ptrtoint %ptr : !llvm.ptr to i64

    %ge_lo = comb.icmp uge %ptr_i64, %lo : i64
    %lt_hi = comb.icmp ult %ptr_i64, %hi : i64
    %ok = comb.and %ge_lo, %lt_hi : i1
    return %ok : i1
  }

  hw.module @main() {
    %fmtPrefix = sim.fmt.literal "virtual_malloc_ok="
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %ok = func.call @alloc_and_check_range() : () -> i1
      %ok_i32 = arith.extui %ok : i1 to i32
      %okFmt = sim.fmt.dec %ok_i32 : i32
      %line = sim.fmt.concat (%fmtPrefix, %okFmt, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }

    hw.output
  }
}

// CHECK: virtual_malloc_ok=1
// AOT-NODISPATCH: Compiled function calls:          0
// AOT-NODISPATCH: Entry-table native calls:         0
// AOT-NODISPATCH: Entry-table trampoline calls:     0
// AOT-NODISPATCH: virtual_malloc_ok=1
