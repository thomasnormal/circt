// RUN: env CIRCT_AOT_STATS=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: keep uvm_sequence_base::clear_response_queue interpreted on
// call_indirect dispatch. Native dispatch of this method is currently unsafe
// on real UVM workloads and can corrupt queue pointers.
//
// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
//
// RUNTIME: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// RUNTIME: Entry-table native calls:         0
// RUNTIME: Entry-table trampoline calls:     0
// RUNTIME: indirect_calls_total:             0
// RUNTIME: indirect_calls_native:            0
// RUNTIME: indirect_calls_trampoline:        0
// RUNTIME: Top interpreted callees (candidates for compilation):
// RUNTIME: 1x  uvm_pkg::uvm_sequence_base::clear_response_queue
// RUNTIME: out=77

func.func @"uvm_pkg::uvm_sequence_base::clear_response_queue"(%self: i64) -> i32 {
  %c77 = arith.constant 77 : i32
  return %c77 : i32
}

llvm.mlir.global internal @"uvm_pkg::__seq_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_sequence_base::clear_response_queue"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %self = llvm.mlir.constant(0 : i64) : i64
    %vt = llvm.mlir.addressof @"uvm_pkg::__seq_vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64) -> i32
    %v = func.call_indirect %fn(%self) : (i64) -> i32
    %vf = sim.fmt.dec %v signed : i32
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
