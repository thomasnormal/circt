// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: canonical config_db key lookup must tolerate overlong packed
// string lengths that include trailing NUL + garbage bytes. UVM names are
// C-string based; lookup should treat first NUL as terminator.
//
// CHECK: get_ok=1
// CHECK: dst_eq_src=1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)

  llvm.mlir.global internal constant @inst_scope_clean("scope\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @inst_scope_corrupt("scope\00garbage\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @field_name("k\00") {addr_space = 0 : i32}

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
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c5_i64 = llvm.mlir.constant(5 : i64) : i64
    %c13_i64 = llvm.mlir.constant(13 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c16_i64 = llvm.mlir.constant(16 : i64) : i64

    %inst_clean_ptr = llvm.mlir.addressof @inst_scope_clean : !llvm.ptr
    %inst_corrupt_ptr = llvm.mlir.addressof @inst_scope_corrupt : !llvm.ptr
    %field_ptr = llvm.mlir.addressof @field_name : !llvm.ptr

    %empty_s0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_s0[1] : !llvm.struct<(ptr, i64)>

    %inst_clean_s0 = llvm.insertvalue %inst_clean_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %inst_clean = llvm.insertvalue %c5_i64, %inst_clean_s0[1] : !llvm.struct<(ptr, i64)>

    %inst_corrupt_s0 = llvm.insertvalue %inst_corrupt_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %inst_corrupt = llvm.insertvalue %c13_i64, %inst_corrupt_s0[1] : !llvm.struct<(ptr, i64)>

    %field_s0 = llvm.insertvalue %field_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %field = llvm.insertvalue %c1_i64, %field_s0[1] : !llvm.struct<(ptr, i64)>

    %lit_ok = sim.fmt.literal "get_ok="
    %lit_eq = sim.fmt.literal "dst_eq_src="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      cf.br ^start
    ^start:
      %obj = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      %vtable_addr = llvm.mlir.addressof @"cfg::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cfg_impl", (i32, ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

      %src = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %dst = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      llvm.store %null, %dst : !llvm.ptr, !llvm.ptr

      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %set_slot = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %set_fp = llvm.load %set_slot : !llvm.ptr -> !llvm.ptr
      %set_fn = builtin.unrealized_conversion_cast %set_fp : !llvm.ptr to (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()
      func.call_indirect %set_fn(%obj, %empty_str, %inst_clean, %field, %src)
          : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %get_slot = llvm.getelementptr %vptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %get_fp = llvm.load %get_slot : !llvm.ptr -> !llvm.ptr
      %get_fn = builtin.unrealized_conversion_cast %get_fp : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1
      %dst_ref = builtin.unrealized_conversion_cast %dst : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call_indirect %get_fn(%obj, %null, %inst_corrupt, %field, %dst_ref)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

      %loaded_dst = llvm.load %dst : !llvm.ptr -> !llvm.ptr
      %dst_eq_src = llvm.icmp "eq" %loaded_dst, %src : !llvm.ptr

      %fmt_ok_v = sim.fmt.dec %ok : i1
      %fmt_ok = sim.fmt.concat (%lit_ok, %fmt_ok_v, %nl)
      sim.proc.print %fmt_ok

      %fmt_eq_v = sim.fmt.dec %dst_eq_src : i1
      %fmt_eq = sim.fmt.concat (%lit_eq, %fmt_eq_v, %nl)
      sim.proc.print %fmt_eq

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
