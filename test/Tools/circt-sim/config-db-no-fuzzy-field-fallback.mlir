// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Canonical config_db lookup should not apply field-name fuzzy fallback
// (e.g. bfm_x matching bfm_0). Only exact field names should match.
//
// CHECK: get_ok=0
// CHECK: dst_is_null=1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)

  llvm.mlir.global internal constant @inst_scope("scope\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @field_bfm0("bfm_0\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @field_bfmx("bfm_x\00") {addr_space = 0 : i32}

  func.func @"cfg::config_db_default_implementation_t::set"(
      %self: !llvm.ptr, %cntxt_name: !llvm.struct<(ptr, i64)>,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %value: !llvm.ptr) {
    return
  }

  func.func @"cfg::config_db_default_implementation_t::get"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %output_ref: !llhd.ref<!llvm.ptr>) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  llvm.mlir.global internal @"cfg::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"cfg::config_db_default_implementation_t::set"],
      [1, @"cfg::config_db_default_implementation_t::get"]
    ]
  } : !llvm.array<2 x ptr>

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c5_i64 = llvm.mlir.constant(5 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c16_i64 = llvm.mlir.constant(16 : i64) : i64

    %scope_ptr = llvm.mlir.addressof @inst_scope : !llvm.ptr
    %bfm0_ptr = llvm.mlir.addressof @field_bfm0 : !llvm.ptr
    %bfmx_ptr = llvm.mlir.addressof @field_bfmx : !llvm.ptr

    %empty_s0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_s0[1] : !llvm.struct<(ptr, i64)>

    %inst_s0 = llvm.insertvalue %scope_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %inst_scope = llvm.insertvalue %c5_i64, %inst_s0[1] : !llvm.struct<(ptr, i64)>

    %bfm0_s0 = llvm.insertvalue %bfm0_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %field_b0 = llvm.insertvalue %c5_i64, %bfm0_s0[1] : !llvm.struct<(ptr, i64)>

    %bfmx_s0 = llvm.insertvalue %bfmx_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %field_bx = llvm.insertvalue %c5_i64, %bfmx_s0[1] : !llvm.struct<(ptr, i64)>

    %lit_ok = sim.fmt.literal "get_ok="
    %lit_null = sim.fmt.literal "dst_is_null="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      cf.br ^start
    ^start:
      %obj = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      %vtable = llvm.mlir.addressof @"cfg::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cfg_impl", (i32, ptr)>
      llvm.store %vtable, %vptr_field : !llvm.ptr, !llvm.ptr

      %src = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %dst = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      llvm.store %null, %dst : !llvm.ptr, !llvm.ptr

      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr

      %set_slot = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %set_fp = llvm.load %set_slot : !llvm.ptr -> !llvm.ptr
      %set_fn = builtin.unrealized_conversion_cast %set_fp : !llvm.ptr to (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()
      func.call_indirect %set_fn(%obj, %empty_str, %inst_scope, %field_b0, %src) : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %get_slot = llvm.getelementptr %vptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %get_fp = llvm.load %get_slot : !llvm.ptr -> !llvm.ptr
      %get_fn = builtin.unrealized_conversion_cast %get_fp : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1
      %dst_ref = builtin.unrealized_conversion_cast %dst : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call_indirect %get_fn(%obj, %null, %inst_scope, %field_bx, %dst_ref) : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

      %loaded_dst = llvm.load %dst : !llvm.ptr -> !llvm.ptr
      %dst_is_null = llvm.icmp "eq" %loaded_dst, %null : !llvm.ptr

      %fmt_ok_v = sim.fmt.dec %ok : i1
      %fmt_ok = sim.fmt.concat (%lit_ok, %fmt_ok_v, %nl)
      sim.proc.print %fmt_ok

      %fmt_null_v = sim.fmt.dec %dst_is_null : i1
      %fmt_null = sim.fmt.concat (%lit_null, %fmt_null_v, %nl)
      sim.proc.print %fmt_null

      llvm.call @free(%dst) : (!llvm.ptr) -> ()
      llvm.call @free(%src) : (!llvm.ptr) -> ()
      llvm.call @free(%obj) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
