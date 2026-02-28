// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_HIERARCHY=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_HIERARCHY=1 CIRCT_AOT_STATS=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: keep uvm_component_proxy::get_immediate_children on interpreted
// call_indirect dispatch even when hierarchy opt-in makes it native-eligible.
// Native execution of this method currently crashes in real UVM topology paths.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// RUNTIME: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// RUNTIME: Entry-table native calls:         0
// RUNTIME: Entry-table trampoline calls:     0
// RUNTIME: indirect_calls_total:             0
// RUNTIME: indirect_calls_native:            0
// RUNTIME: indirect_calls_trampoline:        0
// RUNTIME: Top interpreted callees (candidates for compilation):
// RUNTIME: 1x  uvm_pkg::uvm_component_proxy::get_immediate_children
// RUNTIME: out=77

func.func @"uvm_pkg::uvm_component_proxy::get_immediate_children"(
    %self: i64, %a: i64, %b: i64) -> i64 {
  %c77 = arith.constant 77 : i64
  return %c77 : i64
}

// Keep one always-native function so native entry-table setup is non-empty.
func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"uvm_pkg::__comp_proxy_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_component_proxy::get_immediate_children"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %self = llvm.mlir.constant(0 : i64) : i64
    %a = llvm.mlir.constant(11 : i64) : i64
    %b = llvm.mlir.constant(22 : i64) : i64
    %vt = llvm.mlir.addressof @"uvm_pkg::__comp_proxy_vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64, i64, i64) -> i64
    %r = func.call_indirect %fn(%self, %a, %b) : (i64, i64, i64) -> i64
    %rv = sim.fmt.dec %r signed : i64
    %msg = sim.fmt.concat (%prefix, %rv, %nl)
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
