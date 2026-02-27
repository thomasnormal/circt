// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=COMPILED

// Test vtable dispatch through AOT-compiled functions.
//
// Two pure-arithmetic functions are registered as vtable entries and compiled
// by circt-compile. The simulation dispatches via call_indirect through
// the vtable. Output must be identical in interpreted and compiled modes.
//
// Vtable layout: !llvm.array<2 x ptr>
//   slot 0: @"math::add42"  -> arg + 42
//   slot 1: @"math::mul100" -> arg * 100
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// SIM: add42(5) = 47
// SIM: mul100(3) = 300
//
// COMPILED: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// COMPILED: Entry table: 2 entries for tagged-FuncId dispatch (2 native, 0 non-native)
// COMPILED: Entry-table native calls:         2
// COMPILED: Entry-table trampoline calls:     0
// COMPILED: add42(5) = 47
// COMPILED: mul100(3) = 300

// Pure arithmetic — compilable by circt-compile.
func.func private @"math::add42"(%a: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %a, %c42 : i32
  return %r : i32
}

func.func private @"math::mul100"(%a: i32) -> i32 {
  %c100 = arith.constant 100 : i32
  %r = arith.muli %a, %c100 : i32
  return %r : i32
}

// Vtable global with circt.vtable_entries attribute.
llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::add42"],
    [1, @"math::mul100"]
  ]
} : !llvm.array<2 x ptr>

hw.module @test() {
  %fmt_add_prefix = sim.fmt.literal "add42(5) = "
  %fmt_mul_prefix = sim.fmt.literal "mul100(3) = "
  %fmt_nl = sim.fmt.literal "\0A"
  %c10_i64 = hw.constant 10000000 : i64

  // Main process: dispatch through vtable and print results.
  llhd.process {
    %c5 = arith.constant 5 : i32
    %c3 = arith.constant 3 : i32

    // Get vtable address
    %vtable_addr = llvm.mlir.addressof @"math::__vtable__" : !llvm.ptr

    // --- Dispatch slot 0: add42(5) = 5 + 42 = 47 ---
    %slot0_addr = llvm.getelementptr %vtable_addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %fptr0 = llvm.load %slot0_addr : !llvm.ptr -> !llvm.ptr
    %add_fn = builtin.unrealized_conversion_cast %fptr0 : !llvm.ptr to (i32) -> i32
    %result1 = func.call_indirect %add_fn(%c5) : (i32) -> i32

    %fmt_val1 = sim.fmt.dec %result1 signed : i32
    %fmt_out1 = sim.fmt.concat (%fmt_add_prefix, %fmt_val1, %fmt_nl)
    sim.proc.print %fmt_out1

    // --- Dispatch slot 1: mul100(3) = 3 * 100 = 300 ---
    %slot1_addr = llvm.getelementptr %vtable_addr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %fptr1 = llvm.load %slot1_addr : !llvm.ptr -> !llvm.ptr
    %mul_fn = builtin.unrealized_conversion_cast %fptr1 : !llvm.ptr to (i32) -> i32
    %result2 = func.call_indirect %mul_fn(%c3) : (i32) -> i32

    %fmt_val2 = sim.fmt.dec %result2 signed : i32
    %fmt_out2 = sim.fmt.concat (%fmt_mul_prefix, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_out2

    llhd.halt
  }

  // Terminator at t=10ns.
  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
