// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: config_db::get wrapper interception must write through output
// refs that point into native malloc-backed memory.
//
// This bypasses circt-verilog/UVM frontend lowering and directly exercises
// interpreter interceptors by:
// 1) storing a pointer value via config_db implementation set(),
// 2) reading it back via a get_NNNN wrapper into a native pointer slot.

// CHECK: exists_ok=1
// CHECK: get_ok=1
// CHECK: ptr_eq=1
// CHECK: ptr_is_null=0
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)

  func.func @"uvm::config_db_default_implementation_t::set"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %value: !llvm.ptr) {
    return
  }

  func.func @"uvm::config_db_default_implementation_t::exists"(
      %self: !llvm.ptr, %cntxt: !llvm.ptr,
      %inst: !llvm.struct<(ptr, i64)>, %field: !llvm.struct<(ptr, i64)>,
      %spell_chk: i1) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func private @get_9001(!llvm.ptr, !llvm.struct<(ptr, i64)>,
      !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64

    %empty_str_0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_str_0[1] : !llvm.struct<(ptr, i64)>

    %lit_exists = sim.fmt.literal "exists_ok="
    %lit_get = sim.fmt.literal "get_ok="
    %lit_eq = sim.fmt.literal "ptr_eq="
    %lit_null = sim.fmt.literal "ptr_is_null="
    %nl = sim.fmt.literal "\0A"
    %false = hw.constant false

    llhd.process {
      cf.br ^start
    ^start:
      %src = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %dst = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      llvm.store %null, %dst : !llvm.ptr, !llvm.ptr

      func.call @"uvm::config_db_default_implementation_t::set"(
          %null, %null, %empty_str, %empty_str, %src)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>,
             !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %exists = func.call @"uvm::config_db_default_implementation_t::exists"(
          %null, %null, %empty_str, %empty_str, %false)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>,
             !llvm.struct<(ptr, i64)>, i1) -> i1

      %dst_ref = builtin.unrealized_conversion_cast %dst : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call @get_9001(%null, %empty_str, %empty_str, %dst_ref)
          : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>,
             !llhd.ref<!llvm.ptr>) -> i1

      %loaded = llvm.load %dst : !llvm.ptr -> !llvm.ptr
      %ptr_eq = llvm.icmp "eq" %loaded, %src : !llvm.ptr
      %ptr_is_null = llvm.icmp "eq" %loaded, %null : !llvm.ptr

      %fmt_exists_val = sim.fmt.dec %exists : i1
      %fmt_exists = sim.fmt.concat (%lit_exists, %fmt_exists_val, %nl)
      sim.proc.print %fmt_exists

      %fmt_ok_val = sim.fmt.dec %ok : i1
      %fmt_ok = sim.fmt.concat (%lit_get, %fmt_ok_val, %nl)
      sim.proc.print %fmt_ok

      %fmt_eq_val = sim.fmt.dec %ptr_eq : i1
      %fmt_eq = sim.fmt.concat (%lit_eq, %fmt_eq_val, %nl)
      sim.proc.print %fmt_eq

      %fmt_null_val = sim.fmt.dec %ptr_is_null : i1
      %fmt_null = sim.fmt.concat (%lit_null, %fmt_null_val, %nl)
      sim.proc.print %fmt_null

      llvm.call @free(%dst) : (!llvm.ptr) -> ()
      llvm.call @free(%src) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
