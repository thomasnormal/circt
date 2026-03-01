// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE=0 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=GUARDED

// Regression: unsafe MAY_YIELD fid overrides must not force native dispatch
// for pointer-ABI MAY_YIELD wrappers, even when static suspend analysis would
// otherwise allow bypass.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// GUARDED: [circt-sim] AOT unsafe MAY_YIELD allow list: 1 fids
// GUARDED: [circt-sim] Ignoring unsafe MAY_YIELD fid override for fid=0 name=uvm_pkg::wrapper_ptr_may_yield (pointer ABI)
// GUARDED: [circt-sim] func.call skipped (yield):        1
// GUARDED: [circt-sim] direct_skipped_yield_default:            1
// GUARDED: out=42{{$}}

func.func private @"uvm_pkg::inner_add_one"(%x: i32) -> i32 {
  %one = hw.constant 1 : i32
  %r = arith.addi %x, %one : i32
  return %r : i32
}

func.func private @"uvm_pkg::wrapper_ptr_may_yield"(%p: !llvm.ptr, %x: i32) -> i32 {
  // call_indirect marks this wrapper MAY_YIELD.
  // Pointer ABI should still block unsafe FID override bypass.
  %never = hw.constant false
  cf.cond_br %never, ^do_indirect, ^ret
^do_indirect:
  %zero = llvm.mlir.constant(0 : i64) : i64
  %null = llvm.inttoptr %zero : i64 to !llvm.ptr
  %fn = builtin.unrealized_conversion_cast %null : !llvm.ptr to (!llvm.ptr, i32) -> i32
  %dead = func.call_indirect %fn(%p, %x) : (!llvm.ptr, i32) -> i32
  cf.br ^ret
^ret:
  %r = func.call @"uvm_pkg::inner_add_one"(%x) : (i32) -> i32
  return %r : i32
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::wrapper_ptr_may_yield"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %x = hw.constant 41 : i32
  %zero = llvm.mlir.constant(0 : i64) : i64
  %null = llvm.inttoptr %zero : i64 to !llvm.ptr
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @"uvm_pkg::wrapper_ptr_may_yield"(%null, %x) : (!llvm.ptr, i32) -> i32
    %vf = sim.fmt.dec %r signed : i32
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
