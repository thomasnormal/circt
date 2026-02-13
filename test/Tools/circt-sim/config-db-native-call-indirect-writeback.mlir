// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: call_indirect config_db::get interception must write through
// refs that point into native malloc-backed memory.

// CHECK: indirect_ok=1
// CHECK: ptr_eq=1
// CHECK: ptr_is_null=0
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)

  func.func @"cfg::config_db_default_implementation_t::set"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
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
      [0, @"cfg::config_db_default_implementation_t::get"]
    ]
  } : !llvm.array<1 x ptr>

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c16_i64 = llvm.mlir.constant(16 : i64) : i64

    %empty_str_0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_str_0[1] : !llvm.struct<(ptr, i64)>

    %lit_ok = sim.fmt.literal "indirect_ok="
    %lit_eq = sim.fmt.literal "ptr_eq="
    %lit_null = sim.fmt.literal "ptr_is_null="
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

      func.call @"cfg::config_db_default_implementation_t::set"(
          %obj, %null, %empty_str, %empty_str, %src)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>,
             !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot_addr = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

      %dst_ref = builtin.unrealized_conversion_cast %dst : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call_indirect %fn(%obj, %null, %empty_str, %empty_str, %dst_ref)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

      %loaded = llvm.load %dst : !llvm.ptr -> !llvm.ptr
      %ptr_eq = llvm.icmp "eq" %loaded, %src : !llvm.ptr
      %ptr_is_null = llvm.icmp "eq" %loaded, %null : !llvm.ptr

      %fmt_ok_val = sim.fmt.dec %ok : i1
      %fmt_ok = sim.fmt.concat (%lit_ok, %fmt_ok_val, %nl)
      sim.proc.print %fmt_ok

      %fmt_eq_val = sim.fmt.dec %ptr_eq : i1
      %fmt_eq = sim.fmt.concat (%lit_eq, %fmt_eq_val, %nl)
      sim.proc.print %fmt_eq

      %fmt_null_val = sim.fmt.dec %ptr_is_null : i1
      %fmt_null = sim.fmt.concat (%lit_null, %fmt_null_val, %nl)
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
