// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression: do not apply the generic UVM get_full_name fast-path to
// uvm_port_* classes in call_indirect dispatch.
//
// For uvm_port_base/uvm_port_component the full_name string participates in
// connection-key bookkeeping. Returning an empty fast-path value can break
// resolve_bindings and report spurious connection errors.
//
// CHECK: GET_FULL_NAME_BODY
// CHECK: done

func.func private @"uvm_pkg::uvm_port_base_9999::get_full_name"(%self: i64) -> !llvm.struct<(ptr, i64)> {
  %fmt = sim.fmt.literal "GET_FULL_NAME_BODY\0A"
  sim.proc.print %fmt
  %zero = llvm.mlir.zero : !llvm.struct<(ptr, i64)>
  return %zero : !llvm.struct<(ptr, i64)>
}

func.func @call_get_full_name_indirect(%fptr: !llvm.ptr, %self: i64) -> !llvm.struct<(ptr, i64)> {
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64) -> !llvm.struct<(ptr, i64)>
  %r = func.call_indirect %fn(%self) : (i64) -> !llvm.struct<(ptr, i64)>
  return %r : !llvm.struct<(ptr, i64)>
}

func.func @invoke_get_full_name_from_vtable(%self: i64) {
  %vtable = llvm.mlir.addressof @"uvm_pkg::__port_vtable__" : !llvm.ptr
  %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
  %ignored = func.call @call_get_full_name_indirect(%fptr, %self) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, i64)>
  return
}

llvm.mlir.global internal @"uvm_pkg::__port_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_port_base_9999::get_full_name"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %c_self = hw.constant 4660 : i64
  %fmt_done = sim.fmt.literal "done\0A"
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    func.call @invoke_get_full_name_from_vtable(%c_self) : (i64) -> ()
    sim.proc.print %fmt_done
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
