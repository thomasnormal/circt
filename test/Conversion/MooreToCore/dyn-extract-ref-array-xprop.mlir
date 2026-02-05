// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @dyn_extract_ref_array_unknown_idx
// CHECK-SAME: (in %idx : !hw.struct<value: i2, unknown: i2>, out out : !hw.struct<value: i4, unknown: i4>)
// CHECK: hw.struct_extract %idx["unknown"]
// CHECK: comb.icmp ne
// CHECK: comb.icmp ugt
// CHECK: comb.or
// CHECK: scf.if
// CHECK: llhd.prb
// CHECK: llhd.sig
// CHECK: } else {
// CHECK: llhd.sig.array_get
moore.module @dyn_extract_ref_array_unknown_idx(in %idx : !moore.l2, out out : !moore.l4) {
  %arr = moore.variable : <array<4 x l4>>
  %elem_ref = moore.dyn_extract_ref %arr from %idx : !moore.ref<array<4 x l4>>, !moore.l2 -> !moore.ref<l4>
  %val = moore.read %elem_ref : !moore.ref<l4>
  moore.output %val : !moore.l4
}

// CHECK-LABEL: hw.module @dyn_extract_ref_array_slice_unknown_idx
// CHECK-SAME: (in %idx : !hw.struct<value: i3, unknown: i3>, out out : !hw.array<2xstruct<value: i4, unknown: i4>>)
// CHECK: hw.struct_extract %idx["unknown"]
// CHECK: comb.icmp ne
// CHECK: comb.icmp ugt
// CHECK: comb.or
// CHECK: scf.if
// CHECK: } else {
// CHECK: llhd.sig.array_slice
moore.module @dyn_extract_ref_array_slice_unknown_idx(in %idx : !moore.l3, out out : !moore.array<2 x l4>) {
  %arr = moore.variable : <array<8 x l4>>
  %slice_ref = moore.dyn_extract_ref %arr from %idx : !moore.ref<array<8 x l4>>, !moore.l3 -> !moore.ref<array<2 x l4>>
  %slice = moore.read %slice_ref : !moore.ref<array<2 x l4>>
  moore.output %slice : !moore.array<2 x l4>
}
