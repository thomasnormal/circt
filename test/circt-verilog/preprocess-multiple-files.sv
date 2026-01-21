// RUN: circt-verilog %S/Inputs/preprocess-a.sv %S/Inputs/preprocess-b.sv -E | FileCheck %s --check-prefixes=CHECK-MULTI-UNIT
// RUN: circt-verilog %S/Inputs/preprocess-a.sv %S/Inputs/preprocess-b.sv -E --single-unit | FileCheck %s --check-prefixes=CHECK-SINGLE-UNIT
// REQUIRES: slang

// CHECK-MULTI-UNIT: import hello::undefined;
// CHECK-SINGLE-UNIT: import hello::defined;
