// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: config_db calls routed through helper wrappers should still hit
// canonical call_indirect interception and write through native pointer refs.
//
// CHECK: exists_ok=1
// CHECK: get_ok=1
// CHECK: ptr_eq=1
// CHECK: ptr_is_null=0
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @field_name("k\00") {addr_space = 0 : i32}

  func.func @"uvm::config_db_default_implementation_t::set"(
      %self: !llvm.ptr, %cntxt_name: !llvm.struct<(ptr, i64)>,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %value: !llvm.ptr) {
    return
  }

  func.func @"uvm::config_db_default_implementation_t::get"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %output_ref: !llhd.ref<!llvm.ptr>) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func @"uvm::config_db_default_implementation_t::exists"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %spell_chk: i1) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func private @get_9001(%obj: !llvm.ptr, %inst: !llvm.struct<(ptr, i64)>,
      %field: !llvm.struct<(ptr, i64)>, %out_ref: !llhd.ref<!llvm.ptr>) -> i1 {
    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cfg_impl", (i32, ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
    %slot = llvm.getelementptr %vptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
    %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1
    %null = llvm.mlir.zero : !llvm.ptr
    %ok = func.call_indirect %fn(%obj, %null, %inst, %field, %out_ref)
        : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1
    return %ok : i1
  }

  llvm.mlir.global internal @"uvm::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"uvm::config_db_default_implementation_t::set"],
      [1, @"uvm::config_db_default_implementation_t::get"],
      [2, @"uvm::config_db_default_implementation_t::exists"]
    ]
  } : !llvm.array<3 x ptr>

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c16_i64 = llvm.mlir.constant(16 : i64) : i64

    %field_ptr = llvm.mlir.addressof @field_name : !llvm.ptr

    %empty_s0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_s0[1] : !llvm.struct<(ptr, i64)>
    %field_s0 = llvm.insertvalue %field_ptr, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %field = llvm.insertvalue %c1_i64, %field_s0[1] : !llvm.struct<(ptr, i64)>

    %lit_exists = sim.fmt.literal "exists_ok="
    %lit_get = sim.fmt.literal "get_ok="
    %lit_eq = sim.fmt.literal "ptr_eq="
    %lit_null = sim.fmt.literal "ptr_is_null="
    %nl = sim.fmt.literal "\0A"
    %false = hw.constant false

    llhd.process {
      cf.br ^start
    ^start:
      %obj = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      %vtable_addr = llvm.mlir.addressof @"uvm::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cfg_impl", (i32, ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

      %src = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %dst = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      llvm.store %null, %dst : !llvm.ptr, !llvm.ptr

      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %set_slot = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %set_fp = llvm.load %set_slot : !llvm.ptr -> !llvm.ptr
      %set_fn = builtin.unrealized_conversion_cast %set_fp : !llvm.ptr to (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()
      func.call_indirect %set_fn(%obj, %empty_str, %empty_str, %field, %src)
          : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %exists_slot = llvm.getelementptr %vptr[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %exists_fp = llvm.load %exists_slot : !llvm.ptr -> !llvm.ptr
      %exists_fn = builtin.unrealized_conversion_cast %exists_fp : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i1) -> i1
      %exists = func.call_indirect %exists_fn(%obj, %null, %empty_str, %field, %false)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i1) -> i1

      %dst_ref = builtin.unrealized_conversion_cast %dst : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call @get_9001(%obj, %empty_str, %field, %dst_ref)
          : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>,
             !llhd.ref<!llvm.ptr>) -> i1

      %loaded = llvm.load %dst : !llvm.ptr -> !llvm.ptr
      %ptr_eq = llvm.icmp "eq" %loaded, %src : !llvm.ptr
      %ptr_is_null = llvm.icmp "eq" %loaded, %null : !llvm.ptr

      %fmt_exists_v = sim.fmt.dec %exists : i1
      %fmt_exists = sim.fmt.concat (%lit_exists, %fmt_exists_v, %nl)
      sim.proc.print %fmt_exists

      %fmt_ok_v = sim.fmt.dec %ok : i1
      %fmt_ok = sim.fmt.concat (%lit_get, %fmt_ok_v, %nl)
      sim.proc.print %fmt_ok

      %fmt_eq_v = sim.fmt.dec %ptr_eq : i1
      %fmt_eq = sim.fmt.concat (%lit_eq, %fmt_eq_v, %nl)
      sim.proc.print %fmt_eq

      %fmt_null_v = sim.fmt.dec %ptr_is_null : i1
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
