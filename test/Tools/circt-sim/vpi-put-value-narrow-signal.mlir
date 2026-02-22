// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-put-value-narrow-signal.c -ldl
// RUN: circt-sim %s --top narrow_test --max-time=1000 --vpi=%t.vpi.so 2>&1 | FileCheck %s

// Test that vpi_put_value properly masks values wider than the signal width.
// Without masking, writing 0xFF to a 2-bit signal would crash with an
// APInt assertion: "Value is not an N-bit unsigned value".

// CHECK: VPI_NARROW: signal width=2
// CHECK: VPI_NARROW: wrote=0xFF read=3
// CHECK: VPI_NARROW: wrote=-1 read=3
// CHECK: VPI_NARROW: wrote=2 read=2
// CHECK: VPI_NARROW: vecval=0xDEADBEEF read=3
// CHECK: VPI_NARROW: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_NARROW: FINAL: {{[0-9]+}} passed, 0 failed

// Module with a 2-bit signal to test narrow put_value masking.
hw.module @narrow_test(in %data : !hw.struct<value: i2, unknown: i2>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %narrow_sel = llhd.sig name "narrow_sel" %data : !hw.struct<value: i2, unknown: i2>
  llhd.drv %narrow_sel, %data after %0 : !hw.struct<value: i2, unknown: i2>
}
