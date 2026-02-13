// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: config_db::get wrapper interception must write through output
// refs that point to non-zero offsets inside native malloc-backed memory
// (e.g. dynamic-array element slots).
//
// CHECK: get_ok=1
// CHECK: slot1_eq_src=1
// CHECK: slot1_is_null=0
// CHECK: slot0_is_null=1
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

  func.func private @get_9010(!llvm.ptr, !llvm.struct<(ptr, i64)>,
      !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c16_i64 = llvm.mlir.constant(16 : i64) : i64

    %empty_str_0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_str_0[1] : !llvm.struct<(ptr, i64)>

    %lit_get = sim.fmt.literal "get_ok="
    %lit_eq = sim.fmt.literal "slot1_eq_src="
    %lit_null1 = sim.fmt.literal "slot1_is_null="
    %lit_null0 = sim.fmt.literal "slot0_is_null="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      cf.br ^start
    ^start:
      %src = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %slots = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      %slot0 = llvm.getelementptr %slots[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
      %slot1 = llvm.getelementptr %slots[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

      llvm.store %null, %slot0 : !llvm.ptr, !llvm.ptr
      llvm.store %null, %slot1 : !llvm.ptr, !llvm.ptr

      // Store the source pointer in config_db and read it back into slot1.
      func.call @"uvm::config_db_default_implementation_t::set"(
          %null, %null, %empty_str, %empty_str, %src)
          : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>,
             !llvm.struct<(ptr, i64)>, !llvm.ptr) -> ()

      %slot1_ref = builtin.unrealized_conversion_cast %slot1 : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %ok = func.call @get_9010(%null, %empty_str, %empty_str, %slot1_ref)
          : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>,
             !llhd.ref<!llvm.ptr>) -> i1

      %loaded0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %loaded1 = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
      %slot1_eq_src = llvm.icmp "eq" %loaded1, %src : !llvm.ptr
      %slot1_is_null = llvm.icmp "eq" %loaded1, %null : !llvm.ptr
      %slot0_is_null = llvm.icmp "eq" %loaded0, %null : !llvm.ptr

      %fmt_get_val = sim.fmt.dec %ok : i1
      %fmt_get = sim.fmt.concat (%lit_get, %fmt_get_val, %nl)
      sim.proc.print %fmt_get

      %fmt_eq_val = sim.fmt.dec %slot1_eq_src : i1
      %fmt_eq = sim.fmt.concat (%lit_eq, %fmt_eq_val, %nl)
      sim.proc.print %fmt_eq

      %fmt_null1_val = sim.fmt.dec %slot1_is_null : i1
      %fmt_null1 = sim.fmt.concat (%lit_null1, %fmt_null1_val, %nl)
      sim.proc.print %fmt_null1

      %fmt_null0_val = sim.fmt.dec %slot0_is_null : i1
      %fmt_null0 = sim.fmt.concat (%lit_null0, %fmt_null0_val, %nl)
      sim.proc.print %fmt_null0

      llvm.call @free(%slots) : (!llvm.ptr) -> ()
      llvm.call @free(%src) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
