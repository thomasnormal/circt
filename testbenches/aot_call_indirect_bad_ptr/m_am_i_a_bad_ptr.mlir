llvm.func @malloc(i64) -> !llvm.ptr

func.func @"uvm_pkg::uvm_typed_callbacks_1650::m_am_i_a"(
    %self: !llvm.ptr, %rhs: !llvm.ptr) -> i1 {
  %true = hw.constant true
  %false = hw.constant false
  %null = llvm.mlir.zero : !llvm.ptr
  %is_null = llvm.icmp "eq" %rhs, %null : !llvm.ptr
  cf.cond_br %is_null, ^bb_null, ^bb_deref
^bb_null:
  return %true : i1
^bb_deref:
  %class_ptr = llvm.getelementptr %rhs[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_pkg::uvm_object", (struct<"uvm_pkg::uvm_void", (i32, ptr)>, struct<(ptr, i64)>, i32)>
  %class_id = llvm.load %class_ptr : !llvm.ptr -> i32
  %zero = hw.constant 0 : i32
  %eq = comb.icmp eq %class_id, %zero : i32
  return %eq : i1
}

func.func @caller_indirect(%fptr: !llvm.ptr, %self: !llvm.ptr, %rhs: !llvm.ptr) -> i1 {
  %run = hw.constant true
  cf.cond_br %run, ^live, ^dead
^dead:
  sim.pause quiet
  cf.br ^live
^live:
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> i1
  %r = func.call_indirect %fn(%self, %rhs) : (!llvm.ptr, !llvm.ptr) -> i1
  return %r : i1
}

llvm.mlir.global internal @"uvm_pkg::__callbacks_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_typed_callbacks_1650::m_am_i_a"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %self_i64 = llvm.mlir.constant(0 : i64) : i64
    %self = llvm.inttoptr %self_i64 : i64 to !llvm.ptr
    // Corrupted non-null pointer observed in MAX_NATIVE stress traces.
    %rhs_i64 = llvm.mlir.constant(15450032513024 : i64) : i64
    %rhs = llvm.inttoptr %rhs_i64 : i64 to !llvm.ptr
    %vt = llvm.mlir.addressof @"uvm_pkg::__callbacks_vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %r = func.call @caller_indirect(%fptr, %self, %rhs) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i1
    %ri32 = arith.extui %r : i1 to i32
    %vf = sim.fmt.dec %ri32 signed : i32
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
