// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @StringLen
func.func @StringLen(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_len
  %len = moore.string.len %str
  return %len : !moore.i32
}

// CHECK-LABEL: func @StringToUpper
func.func @StringToUpper(%str: !moore.string) -> !moore.string {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_toupper
  %upper = moore.string.toupper %str
  return %upper : !moore.string
}

// CHECK-LABEL: func @StringToLower
func.func @StringToLower(%str: !moore.string) -> !moore.string {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_tolower
  %lower = moore.string.tolower %str
  return %lower : !moore.string
}

// CHECK-LABEL: func @StringGetC
func.func @StringGetC(%str: !moore.string, %index: !moore.i32) -> !moore.i8 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_getc
  %ch = moore.string.getc %str[%index]
  return %ch : !moore.i8
}

// CHECK-LABEL: func @StringPutC
func.func @StringPutC(%str: !moore.ref<string>, %index: !moore.i32, %ch: !moore.i8) {
  // CHECK: llvm.call @__moore_string_putc
  moore.string.putc %str[%index], %ch : !moore.ref<string>
  return
}

// CHECK-LABEL: func @StringSubstr
func.func @StringSubstr(%str: !moore.string, %start: !moore.i32, %len: !moore.i32) -> !moore.string {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_substr
  %sub = moore.string.substr %str[%start, %len]
  return %sub : !moore.string
}

// CHECK-LABEL: func @StringItoa
func.func @StringItoa(%dest: !moore.ref<string>, %val: !moore.i32) {
  // CHECK: llvm.call @__moore_string_itoa
  // CHECK: llhd.drv
  moore.string.itoa %dest, %val : !moore.ref<string>, !moore.i32
  return
}

// CHECK-LABEL: func @StringConcat
func.func @StringConcat(%a: !moore.string, %b: !moore.string) -> !moore.string {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_concat
  %result = moore.string_concat (%a, %b) : !moore.string
  return %result : !moore.string
}

// CHECK-LABEL: func @StringCmpEq
func.func @StringCmpEq(%a: !moore.string, %b: !moore.string) -> !moore.i1 {
  // CHECK: llvm.call @__moore_string_cmp
  // CHECK: arith.cmpi eq
  %result = moore.string_cmp eq %a, %b : !moore.string -> !moore.i1
  return %result : !moore.i1
}

// CHECK-LABEL: func @StringCmpNe
func.func @StringCmpNe(%a: !moore.string, %b: !moore.string) -> !moore.i1 {
  // CHECK: llvm.call @__moore_string_cmp
  // CHECK: arith.cmpi ne
  %result = moore.string_cmp ne %a, %b : !moore.string -> !moore.i1
  return %result : !moore.i1
}

// CHECK-LABEL: func @IntToString
func.func @IntToString(%val: !moore.i32) -> !moore.string {
  // CHECK: arith.extui
  // CHECK: llvm.call @__moore_int_to_string
  %str = moore.int_to_string %val : i32
  return %str : !moore.string
}

// CHECK-LABEL: func @StringToInt
func.func @StringToInt(%str: !moore.string) -> !moore.i32 {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_to_int
  // CHECK: arith.trunci
  %val = moore.string_to_int %str : i32
  return %val : !moore.i32
}

// CHECK-LABEL: func @IsUnknown
func.func @IsUnknown(%val: !moore.i32) -> !moore.i1 {
  // IsUnknown always returns false in two-valued lowering
  // CHECK: hw.constant false
  %result = moore.builtin.isunknown %val : !moore.i32
  return %result : !moore.i1
}
