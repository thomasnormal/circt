// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Ensure we compute a consensus value for unknown indices on small *non-constant*
// arrays, instead of pessimizing to all-unknown.

// CHECK-LABEL: func.func @RuntimeArrayUnknownIndex
// Unknown index condition:
// CHECK: %[[IDXUNK:.*]] = comb.icmp ne %{{.*}}, %{{.*}} : i2
// The unknown-index result is computed symbolically (not just all-unknown):
// CHECK: %[[CVAL:.*]] = comb.concat
// CHECK: %[[CUNK:.*]] = comb.concat
// CHECK: %[[CONS:.*]] = hw.struct_create (%[[CVAL]], %[[CUNK]]) : !hw.struct<value: i2, unknown: i2>
// CHECK: comb.mux %[[IDXUNK]], %[[CONS]], %{{.*}} : !hw.struct<value: i2, unknown: i2>
func.func @RuntimeArrayUnknownIndex(%x: !moore.l2, %idx: !moore.l2) -> !moore.l2 {
  // Non-constant array: element comes from an argument.
  %arr = moore.array_create %x, %x, %x, %x
    : !moore.l2, !moore.l2, !moore.l2, !moore.l2 -> !moore.array<4 x l2>
  %val = moore.dyn_extract %arr from %idx
    : !moore.array<4 x l2>, !moore.l2 -> !moore.l2
  return %val : !moore.l2
}
