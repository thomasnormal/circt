// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @StringAtoI
func.func @StringAtoI(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_atoi
  %val = moore.string.atoi %str
  return %val : !moore.i32
}

// CHECK-LABEL: func @StringAtoHex
func.func @StringAtoHex(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_atohex
  %val = moore.string.atohex %str
  return %val : !moore.i32
}

// CHECK-LABEL: func @StringAtoOct
func.func @StringAtoOct(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_atooct
  %val = moore.string.atooct %str
  return %val : !moore.i32
}

// CHECK-LABEL: func @StringAtoBin
func.func @StringAtoBin(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_atobin
  %val = moore.string.atobin %str
  return %val : !moore.i32
}
