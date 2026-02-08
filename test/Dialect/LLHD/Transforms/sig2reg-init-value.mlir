// RUN: circt-opt --llhd-sig2reg -cse %s | FileCheck %s

// Test that Sig2Reg uses the initial value for reads when there's a circular
// dependency (write value depends on probe result), to break the cycle.

// CHECK-LABEL: hw.module @circular_dep_breaks_cycle
hw.module @circular_dep_breaks_cycle(out out: i8) {
  %init = hw.constant 42 : i8
  %c7_i8 = hw.constant 7 : i8
  %time = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %init : i8
  %prb = llhd.prb %sig : i8
  // Drive value depends on probe (circular dependency)
  %new_val = comb.add %prb, %c7_i8 : i8
  llhd.drv %sig, %new_val after %time : i8

  // Probe should return init (42) to break the circular dependency
  // CHECK: %c42_i8 = hw.constant 42 : i8
  // CHECK: hw.output %c42_i8 : i8
  hw.output %prb : i8
}

// CHECK-LABEL: hw.module @struct_circular_dep
hw.module @struct_circular_dep(out lo: i4, out hi: i4) {
  %c5_i4 = hw.constant 5 : i4
  %c0_i4 = hw.constant 0 : i4
  %init = hw.struct_create (%c5_i4, %c0_i4) : !hw.struct<lo: i4, hi: i4>
  %time = llhd.constant_time <0ns, 0d, 1e>
  %c-6_i4 = hw.constant -6 : i4

  %sig = llhd.sig %init : !hw.struct<lo: i4, hi: i4>
  %prb = llhd.prb %sig : !hw.struct<lo: i4, hi: i4>

  // Inject hi field using probe result (circular dependency through %prb)
  %new_val = hw.struct_inject %prb["hi"], %c-6_i4 : !hw.struct<lo: i4, hi: i4>
  llhd.drv %sig, %new_val after %time : !hw.struct<lo: i4, hi: i4>

  // Extract fields from probe - should get init values (lo=5, hi=0)
  %lo = hw.struct_extract %prb["lo"] : !hw.struct<lo: i4, hi: i4>
  %hi = hw.struct_extract %prb["hi"] : !hw.struct<lo: i4, hi: i4>

  // CHECK: %[[INIT:.+]] = hw.struct_create
  // CHECK: %[[INITBC:.+]] = hw.bitcast %[[INIT]]
  // CHECK: %[[READVAL:.+]] = hw.bitcast %[[INITBC]]
  // CHECK-DAG: %[[LO:.+]] = hw.struct_extract %[[READVAL]]["lo"]
  // CHECK-DAG: %[[HI:.+]] = hw.struct_extract %[[READVAL]]["hi"]
  // CHECK: hw.output %[[LO]], %[[HI]] : i4, i4
  hw.output %lo, %hi : i4, i4
}
