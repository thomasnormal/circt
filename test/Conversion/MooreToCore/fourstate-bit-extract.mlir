// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test dynamic bit extraction from 4-state signals.
// This tests the fix for the UART AVIP issue where extracting bits from
// 4-state signals failed with "expected same element type as slice".

// CHECK-LABEL: hw.module @dyn_extract_ref_fourstate
// CHECK-SAME: (in %idx_in : i4, out out : !hw.struct<value: i1, unknown: i1>)
moore.module @dyn_extract_ref_fourstate(in %idx_in : !moore.i4, out out : !moore.l1) {
  // Create a 4-state signal
  %sig = moore.variable : !moore.ref<l8>

  // Dynamic bit extraction from 4-state ref - this was failing before the fix
  // CHECK: [[SIG:%.+]] = llhd.sig
  // CHECK: llhd.sig.struct_extract [[SIG]]["value"]
  // CHECK: llhd.sig.struct_extract [[SIG]]["unknown"]
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i8> -> <i1>
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i8> -> <i1>
  // CHECK: llhd.prb
  // CHECK: llhd.prb
  // CHECK: hw.struct_create
  // CHECK: llhd.sig
  %bit_ref = moore.dyn_extract_ref %sig from %idx_in : !moore.ref<l8>, !moore.i4 -> !moore.ref<l1>

  // Read the extracted bit
  // CHECK: llhd.prb
  %bit = moore.read %bit_ref : !moore.ref<l1>

  // CHECK: hw.output
  moore.output %bit : !moore.l1
}

// Test with wider extraction (multiple bits)
// CHECK-LABEL: hw.module @dyn_extract_ref_fourstate_slice
// CHECK-SAME: (in %idx : i5, out out : !hw.struct<value: i4, unknown: i4>)
moore.module @dyn_extract_ref_fourstate_slice(in %idx : !moore.i5, out out : !moore.l4) {
  // Create a 4-state signal
  %sig = moore.variable : !moore.ref<l16>

  // Dynamic slice extraction from 4-state ref
  // CHECK: llhd.sig.struct_extract {{%.+}}["value"]
  // CHECK: llhd.sig.struct_extract {{%.+}}["unknown"]
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i16> -> <i4>
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i16> -> <i4>
  %slice_ref = moore.dyn_extract_ref %sig from %idx : !moore.ref<l16>, !moore.i5 -> !moore.ref<l4>

  // Read the extracted slice
  %slice = moore.read %slice_ref : !moore.ref<l4>

  moore.output %slice : !moore.l4
}

// Test with 4-state index
// CHECK-LABEL: hw.module @dyn_extract_ref_fourstate_idx
// CHECK-SAME: (in %idx : !hw.struct<value: i4, unknown: i4>, out out : !hw.struct<value: i1, unknown: i1>)
moore.module @dyn_extract_ref_fourstate_idx(in %idx : !moore.l4, out out : !moore.l1) {
  // Create a 4-state signal
  %sig = moore.variable : !moore.ref<l8>

  // Dynamic bit extraction with 4-state index
  // The index should have its value component extracted
  // CHECK: hw.struct_extract %idx["value"]
  // CHECK: llhd.sig.struct_extract {{%.+}}["value"]
  // CHECK: llhd.sig.struct_extract {{%.+}}["unknown"]
  %bit_ref = moore.dyn_extract_ref %sig from %idx : !moore.ref<l8>, !moore.l4 -> !moore.ref<l1>

  %bit = moore.read %bit_ref : !moore.ref<l1>

  moore.output %bit : !moore.l1
}

// Test static bit extraction from 4-state signal (ExtractRefOp)
// CHECK-LABEL: hw.module @extract_ref_fourstate
// CHECK-SAME: (out out : !hw.struct<value: i1, unknown: i1>)
moore.module @extract_ref_fourstate(out out : !moore.l1) {
  // Create a 4-state signal
  %sig = moore.variable : !moore.ref<l8>

  // Static bit extraction from 4-state ref
  // CHECK: llhd.sig.struct_extract {{%.+}}["value"]
  // CHECK: llhd.sig.struct_extract {{%.+}}["unknown"]
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i8> -> <i1>
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i8> -> <i1>
  %bit_ref = moore.extract_ref %sig from 3 : !moore.ref<l8> -> !moore.ref<l1>

  // Read the extracted bit
  %bit = moore.read %bit_ref : !moore.ref<l1>

  moore.output %bit : !moore.l1
}

// Test static slice extraction from 4-state signal
// CHECK-LABEL: hw.module @extract_ref_fourstate_slice
// CHECK-SAME: (out out : !hw.struct<value: i4, unknown: i4>)
moore.module @extract_ref_fourstate_slice(out out : !moore.l4) {
  // Create a 4-state signal
  %sig = moore.variable : !moore.ref<l16>

  // Static slice extraction from 4-state ref
  // CHECK: llhd.sig.struct_extract {{%.+}}["value"]
  // CHECK: llhd.sig.struct_extract {{%.+}}["unknown"]
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i16> -> <i4>
  // CHECK: llhd.sig.extract {{%.+}} from {{%.+}} : <i16> -> <i4>
  %slice_ref = moore.extract_ref %sig from 4 : !moore.ref<l16> -> !moore.ref<l4>

  // Read the extracted slice
  %slice = moore.read %slice_ref : !moore.ref<l4>

  moore.output %slice : !moore.l4
}
