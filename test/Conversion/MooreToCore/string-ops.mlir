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
  // CHECK: llvm.store
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

// CHECK-LABEL: func @StringReplicate
func.func @StringReplicate(%str: !moore.string, %count: !moore.i32) -> !moore.string {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.call @__moore_string_replicate
  %result = moore.string_replicate %count, %str
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
  // CHECK: llvm.call @__moore_packed_string_to_string
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

// CHECK-LABEL: func @CountOnes
func.func @CountOnes(%val: !moore.i16) -> !moore.i16 {
  // CountOnes is lowered to llvm.intr.ctpop (count population)
  // CHECK: llvm.intr.ctpop(%{{.*}}) : (i16) -> i16
  %result = moore.builtin.countones %val : !moore.i16
  return %result : !moore.i16
}

// CHECK-LABEL: func @OneHot
func.func @OneHot(%val: !moore.i8) -> !moore.i1 {
  // OneHot is lowered to ctpop(x) == 1
  // CHECK-DAG: %[[ONE:.*]] = hw.constant 1 : i8
  // CHECK: %[[CTPOP:.*]] = llvm.intr.ctpop(%{{.*}}) : (i8) -> i8
  // CHECK: comb.icmp eq %[[CTPOP]], %[[ONE]] : i8
  %result = moore.builtin.onehot %val : !moore.i8
  return %result : !moore.i1
}

// CHECK-LABEL: func @OneHot0
func.func @OneHot0(%val: !moore.i8) -> !moore.i1 {
  // OneHot0 is lowered to ctpop(x) <= 1
  // CHECK-DAG: %[[ONE:.*]] = hw.constant 1 : i8
  // CHECK: %[[CTPOP:.*]] = llvm.intr.ctpop(%{{.*}}) : (i8) -> i8
  // CHECK: comb.icmp ule %[[CTPOP]], %[[ONE]] : i8
  %result = moore.builtin.onehot0 %val : !moore.i8
  return %result : !moore.i1
}

// CHECK-LABEL: func @CountBitsOnes
func.func @CountBitsOnes(%val: !moore.i16) -> !moore.i16 {
  // $countbits(x, 1) is lowered to ctpop(x)
  // CHECK: llvm.intr.ctpop(%{{.*}}) : (i16) -> i16
  %result = moore.builtin.countbits %val, 2 : !moore.i16
  return %result : !moore.i16
}

// CHECK-LABEL: func @CountBitsZeros
func.func @CountBitsZeros(%val: !moore.i8) -> !moore.i8 {
  // $countbits(x, 0) is lowered to bitwidth - ctpop(x)
  // CHECK-DAG: %[[WIDTH:.*]] = hw.constant 8 : i8
  // CHECK: %[[CTPOP:.*]] = llvm.intr.ctpop(%{{.*}}) : (i8) -> i8
  // CHECK: comb.sub %[[WIDTH]], %[[CTPOP]] : i8
  %result = moore.builtin.countbits %val, 1 : !moore.i8
  return %result : !moore.i8
}

// CHECK-LABEL: func @CountBitsBoth
func.func @CountBitsBoth(%val: !moore.i8) -> !moore.i8 {
  // $countbits(x, 0, 1) is lowered to bitwidth (all bits are 0 or 1)
  // CHECK: hw.constant 8 : i8
  %result = moore.builtin.countbits %val, 3 : !moore.i8
  return %result : !moore.i8
}

// CHECK-LABEL: hw.module @FormatStringTest
moore.module @FormatStringTest() {
  moore.procedure initial {
    %0 = moore.constant_string "IDLE" : i32
    %1 = moore.int_to_string %0 : i32
    // CHECK: sim.fmt.dyn_string %{{.*}} : !llvm.struct<(ptr, i64)>
    %2 = moore.fmt.string %1
    moore.builtin.display %2
    moore.return
  }
  moore.output
}

// Test fstring_to_string conversion with literal input
// CHECK-LABEL: func @FStringToStringLiteral
func.func @FStringToStringLiteral() -> !moore.string {
  // The global is hoisted to module level, so just check the addressof reference.
  // CHECK: [[ADDR:%.+]] = llvm.mlir.addressof @__moore_str_{{.*}} : !llvm.ptr
  // CHECK: [[LEN:%.+]] = arith.constant 5 : i64
  // CHECK: [[UNDEF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: [[S1:%.+]] = llvm.insertvalue [[ADDR]], [[UNDEF]][0]
  // CHECK: [[S2:%.+]] = llvm.insertvalue [[LEN]], [[S1]][1]
  // CHECK: return [[S2]]
  %0 = moore.fmt.literal "hello"
  %1 = moore.fstring_to_string %0
  return %1 : !moore.string
}

// Test fstring_to_string conversion with empty literal
// CHECK-LABEL: func @FStringToStringEmptyLiteral
func.func @FStringToStringEmptyLiteral() -> !moore.string {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[ZERO:%.+]] = arith.constant 0 : i64
  // CHECK: [[UNDEF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: [[S1:%.+]] = llvm.insertvalue [[NULL]], [[UNDEF]][0]
  // CHECK: [[S2:%.+]] = llvm.insertvalue [[ZERO]], [[S1]][1]
  // CHECK: return [[S2]]
  %0 = moore.fmt.literal ""
  %1 = moore.fstring_to_string %0
  return %1 : !moore.string
}

// Test fstring_to_string conversion with dynamic string input (round-trip)
// CHECK-LABEL: func @FStringToStringDynamic
func.func @FStringToStringDynamic(%str: !moore.string) -> !moore.string {
  // When input is fmt.string (which converts string to format string),
  // fstring_to_string should return the original string
  // CHECK: return %arg0
  %0 = moore.fmt.string %str
  %1 = moore.fstring_to_string %0
  return %1 : !moore.string
}

// Test fstring_to_string conversion with formatted integer
// CHECK-LABEL: func @FStringToStringFormattedInt
func.func @FStringToStringFormattedInt(%val: !moore.i32) -> !moore.string {
  // CHECK: arith.extsi %arg0 : i32 to i64
  // CHECK: llvm.call @__moore_int_to_string
  %0 = moore.fmt.int decimal %val, align right, pad space signed : i32
  %1 = moore.fstring_to_string %0
  return %1 : !moore.string
}

// Test fstring_to_string conversion with concatenation
// CHECK-LABEL: func @FStringToStringConcat
func.func @FStringToStringConcat(%str: !moore.string) -> !moore.string {
  // The global is hoisted to module level, so just check the concat call.
  // CHECK: llvm.call @__moore_string_concat
  %0 = moore.fmt.literal "prefix: "
  %1 = moore.fmt.string %str
  %2 = moore.fmt.concat (%0, %1)
  %3 = moore.fstring_to_string %2
  return %3 : !moore.string
}
