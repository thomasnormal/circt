// RUN: circt-sim %s --top top --max-time=100000000 | FileCheck %s

// Regression test for module-level initialization with scf.if + struct
// extraction feeding llvm.store. The module-level pre-execution pass must
// evaluate this chain so the later continuous drive reads the initialized value.
//
// Before the fix, module-level scf.if/struct ops were skipped, store operands
// stayed unknown, and llvm.store became a no-op. The field remained zero and
// the signal printed "V=0 U=0". The expected initialized value is V=1 U=0.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  hw.module @top() {
    %c2_i64 = llvm.mlir.constant(2 : i64) : i64
    %time0 = llhd.constant_time <0ns, 0d, 1e>
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %false_i1 = hw.constant false
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %known_one = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

    %malloc = llvm.call @malloc(%c2_i64) : (i64) -> !llvm.ptr
    %iface_sig = llhd.sig %malloc : !llvm.ptr
    %iface_ptr = llhd.prb %iface_sig : !llvm.ptr
    %field_ptr = llvm.getelementptr %iface_ptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.test_if", (struct<(i1, i1)>)>

    %sel = scf.if %false_i1 -> (!hw.struct<value: i1, unknown: i1>) {
      scf.yield %sig_init : !hw.struct<value: i1, unknown: i1>
    } else {
      scf.yield %known_one : !hw.struct<value: i1, unknown: i1>
    }
    %sel_val = hw.struct_extract %sel["value"] : !hw.struct<value: i1, unknown: i1>
    %tmp0 = llvm.insertvalue %sel_val, %undef[0] : !llvm.struct<(i1, i1)>
    %sel_unk = hw.struct_extract %sel["unknown"] : !hw.struct<value: i1, unknown: i1>
    %tmp1 = llvm.insertvalue %sel_unk, %tmp0[1] : !llvm.struct<(i1, i1)>
    llvm.store %tmp1, %field_ptr : !llvm.struct<(i1, i1)>, !llvm.ptr

    %s = llhd.sig %sig_init : !hw.struct<value: i1, unknown: i1>
    %loaded = llvm.load %field_ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
    %loaded_v = llvm.extractvalue %loaded[0] : !llvm.struct<(i1, i1)>
    %loaded_u = llvm.extractvalue %loaded[1] : !llvm.struct<(i1, i1)>
    %drive_val = hw.struct_create (%loaded_v, %loaded_u) : !hw.struct<value: i1, unknown: i1>
    llhd.drv %s, %drive_val after %time0 : !hw.struct<value: i1, unknown: i1>

    %fmt_pref = sim.fmt.literal "V="
    %fmt_sep = sim.fmt.literal " U="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %d = llhd.int_to_time %c2_i64
      llhd.wait delay %d, ^bb1
    ^bb1:
      %cur = llhd.prb %s : !hw.struct<value: i1, unknown: i1>
      %cur_v = hw.struct_extract %cur["value"] : !hw.struct<value: i1, unknown: i1>
      %cur_u = hw.struct_extract %cur["unknown"] : !hw.struct<value: i1, unknown: i1>
      %fmt_v = sim.fmt.dec %cur_v : i1
      %fmt_u = sim.fmt.dec %cur_u : i1
      %out = sim.fmt.concat (%fmt_pref, %fmt_v, %fmt_sep, %fmt_u, %fmt_nl)
      sim.proc.print %out
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}

// CHECK: V=1 U=0
