// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression: driving through an out-of-range llhd.sig.extract on a
// memory-backed ref (e.g. sentinel low-bit patterns) must be treated as a
// guarded no-op, not a hard interpreter failure.

// CHECK-NOT: interpretOperation failed
// CHECK: gbyte=0

module {
  llvm.mlir.global internal @gbyte(0 : i8) : i8

  hw.module @test() {
    %delay_fs = hw.constant 1000000 : i64
    %c63_i6 = hw.constant 63 : i6
    %c170_i8 = hw.constant 170 : i8
    %fmt = sim.fmt.literal "gbyte="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      %delay = llhd.int_to_time %delay_fs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      %addr = llvm.mlir.addressof @gbyte : !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %addr : !llvm.ptr to !llhd.ref<i64>
      %byte_ref = llhd.sig.extract %ref from %c63_i6 : <i64> -> <i8>
      %t0 = llhd.constant_time <0ns, 0d, 1e>
      llhd.drv %byte_ref, %c170_i8 after %t0 : i8

      %loaded = llvm.load %addr : !llvm.ptr -> i8
      %f = sim.fmt.dec %loaded : i8
      %out = sim.fmt.concat (%fmt, %f, %nl)
      sim.proc.print %out

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
