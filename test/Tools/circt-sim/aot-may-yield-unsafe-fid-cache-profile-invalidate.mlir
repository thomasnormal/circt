// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE=0 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: unsafe MAY_YIELD decision caching must be invalidated when
// runtime call_indirect profile shape changes. The first call to
// uvm_pkg::profiled_wrapper is interpreted (no profile yet), then runtime
// profiling proves the call_indirect target set is non-suspending, allowing
// native dispatch on subsequent calls.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 2 vtable FuncIds
// COMPILE: [circt-compile] Tagged 1/2 FuncIds as MAY_YIELD
//
// RUNTIME: [circt-sim] Runtime call_indirect profiling: enabled
// RUNTIME: [circt-sim] AOT unsafe MAY_YIELD allow list: 1 fids
// RUNTIME: [circt-sim] Ignoring unsafe MAY_YIELD fid override for fid=0 name=uvm_pkg::profiled_wrapper (body may suspend)
// RUNTIME: [circt-sim] func.call skipped (yield):{{ *}}1
// RUNTIME: [circt-sim] indirect_calls_total:{{ *}}1
// RUNTIME: [circt-sim] direct_calls_native:{{ *}}2
// RUNTIME: [circt-sim] direct_calls_interpreted:{{ *}}1
// RUNTIME: out=43{{$}}

func.func private @safe_add_one(%x: i32) -> i32 {
  %one = hw.constant 1 : i32
  %r = arith.addi %x, %one : i32
  return %r : i32
}

func.func private @"uvm_pkg::profiled_wrapper"(%x: i32) -> i32 {
  // Tagged pointer 0xF0000001 selects vtable FuncId 1 at runtime.
  %fid1_i64 = hw.constant 4026531841 : i64
  %tag1 = llvm.inttoptr %fid1_i64 : i64 to !llvm.ptr
  %fn = builtin.unrealized_conversion_cast %tag1 : !llvm.ptr to (i32) -> i32
  %r = func.call_indirect %fn(%x) : (i32) -> i32
  return %r : i32
}

llvm.mlir.global internal @"tb::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::profiled_wrapper"],
    [1, @safe_add_one]
  ]
} : !llvm.array<2 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %x0 = hw.constant 40 : i32
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %r0 = func.call @"uvm_pkg::profiled_wrapper"(%x0) : (i32) -> i32
    %r1 = func.call @"uvm_pkg::profiled_wrapper"(%r0) : (i32) -> i32
    %r2 = func.call @"uvm_pkg::profiled_wrapper"(%r1) : (i32) -> i32
    %vf = sim.fmt.dec %r2 signed : i32
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
