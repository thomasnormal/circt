// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @FuncArgsAndReturns
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @FuncArgsAndReturns(%arg0: !moore.i8, %arg1: !moore.i32, %arg2: !moore.i1) -> !moore.i8 {
  // CHECK-NEXT: return %arg0 : i8
  return %arg0 : !moore.i8
}

// CHECK-LABEL: func @ControlFlow
// CHECK-SAME: (%arg0: i32, %arg1: i1)
func.func @ControlFlow(%arg0: !moore.i32, %arg1: i1) {
  // CHECK:   cf.br ^bb1
  // CHECK: ^bb1:
  // CHECK:   cf.cond_br %arg1, ^bb1, ^bb2
  // CHECK: ^bb2:
  // CHECK:   return
  cf.br ^bb1(%arg0: !moore.i32)
^bb1(%0: !moore.i32):
  cf.cond_br %arg1, ^bb1(%0 : !moore.i32), ^bb2(%arg0 : !moore.i32)
^bb2(%1: !moore.i32):
  return
}

// CHECK-LABEL: func @Calls
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @Calls(%arg0: !moore.i8, %arg1: !moore.i32, %arg2: !moore.i1) -> !moore.i8 {
  // CHECK-NEXT: %true =
  // CHECK-NEXT: call @ControlFlow(%arg1, %true) : (i32, i1) -> ()
  // CHECK-NEXT: [[TMP:%.+]] = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (i8, i32, i1) -> i8
  // CHECK-NEXT: return [[TMP]] : i8
  %true = hw.constant true
  call @ControlFlow(%arg1, %true) : (!moore.i32, i1) -> ()
  %0 = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (!moore.i8, !moore.i32, !moore.i1) -> !moore.i8
  return %0 : !moore.i8
}

// CHECK-LABEL: func @UnrealizedConversionCast
func.func @UnrealizedConversionCast(%arg0: !moore.i8) -> !moore.i16 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %arg0, %arg0 : i8, i8
  // CHECK-NEXT: return [[TMP]] : i16
  %0 = builtin.unrealized_conversion_cast %arg0 : !moore.i8 to i8
  %1 = comb.concat %0, %0 : i8, i8
  %2 = builtin.unrealized_conversion_cast %1 : i16 to !moore.i16
  return %2 : !moore.i16
}

// CHECK-LABEL: func @Expressions
// CHECK-SAME: (%arg0: i1, %arg1: !hw.struct<value: i1, unknown: i1>, %arg2: i6, %arg3: i5, %arg4: i1, %arg5: !hw.array<5xi32>, %arg6: !llhd.ref<i1>, %arg7: !llhd.ref<!hw.array<5xi32>>)
func.func @Expressions(%arg0: !moore.i1, %arg1: !moore.l1, %arg2: !moore.i6, %arg3: !moore.i5, %arg4: !moore.i1, %arg5: !moore.array<5 x i32>, %arg6: !moore.ref<i1>, %arg7: !moore.ref<array<5 x i32>>) {
  // CHECK: hw.aggregate_constant
  // CHECK: hw.aggregate_constant
  // Local variables in functions use llvm.alloca instead of llhd.sig
  // CHECK: llvm.mlir.constant(1 : i64)
  // CHECK: hw.constant 0 : i12
  moore.concat %arg0, %arg0 : (!moore.i1, !moore.i1) -> !moore.i2
  moore.concat %arg1, %arg1 : (!moore.l1, !moore.l1) -> !moore.l2

  moore.replicate %arg0 : i1 -> i2
  moore.replicate %arg1 : l1 -> l2

  // CHECK: %name = hw.wire %arg0 : i1
  %name = moore.assigned_variable %arg0 : !moore.i1

  moore.constant 12 : !moore.i32
  moore.constant 3 : !moore.i6

  moore.shl %arg2, %arg0 : !moore.i6, !moore.i1

  moore.shl %arg3, %arg2 : !moore.i5, !moore.i6

  moore.shr %arg2, %arg0 : !moore.i6, !moore.i1

  moore.ashr %arg2, %arg2 : !moore.i6, !moore.i6

  moore.ashr %arg3, %arg2 : !moore.i5, !moore.i6

  %2 = moore.constant 2 : !moore.i32
  %c0 = moore.constant 0 : !moore.i32

  moore.extract %arg2 from 2 : !moore.i6 -> !moore.i2
  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> !moore.array<2 x i32>
  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> i32

  moore.extract %arg2 from -2 : !moore.i6 -> !moore.i10

  moore.extract %arg2 from 4 : !moore.i6 -> !moore.i4

  moore.extract %arg2 from -2 : !moore.i6 -> !moore.i4

  moore.extract %arg2 from -6 : !moore.i6 -> !moore.i4

  moore.extract %arg2 from 6 : !moore.i6 -> !moore.i4

  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> !moore.array<9 x i32>

  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> !moore.array<4 x i32>

  moore.extract %arg5 from -1 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  moore.extract %arg5 from 5 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> i32
  moore.extract %arg5 from 6 : !moore.array<5 x i32> -> i32

  moore.extract_ref %arg6 from 0 : !moore.ref<i1> -> !moore.ref<i1>
  moore.extract_ref %arg7 from 2 : !moore.ref<array<5 x i32>> -> !moore.ref<array<2 x i32>>
  moore.extract_ref %arg7 from 2 : !moore.ref<array<5 x i32>> -> !moore.ref<i32>

  moore.dyn_extract %arg2 from %2 : !moore.i6, !moore.i32 -> !moore.i1
  moore.dyn_extract %arg5 from %2 : !moore.array<5 x i32>, !moore.i32 -> !moore.array<2 x i32>
  moore.dyn_extract %arg5 from %2 : !moore.array<5 x i32>, !moore.i32 -> !moore.i32

  moore.dyn_extract_ref %arg6 from %c0 : !moore.ref<i1>, !moore.i32 -> !moore.ref<i1>
  moore.dyn_extract_ref %arg7 from %2 : !moore.ref<array<5 x i32>>, !moore.i32 -> !moore.ref<array<2 x i32>>
  moore.dyn_extract_ref %arg7 from %2 : !moore.ref<array<5 x i32>>, !moore.i32 -> !moore.ref<i32>

  moore.reduce_and %arg2 : !moore.i6 -> !moore.i1

  moore.reduce_or %arg0 : !moore.i1 -> !moore.i1

  moore.reduce_xor %arg1 : !moore.l1 -> !moore.l1

  moore.bool_cast %arg2 : !moore.i6 -> !moore.i1

  moore.not %arg2 : !moore.i6

  moore.neg %arg2 : !moore.i6

  moore.add %arg1, %arg1 : !moore.l1
  moore.sub %arg1, %arg1 : !moore.l1
  moore.mul %arg1, %arg1 : !moore.l1
  moore.divu %arg0, %arg0 : !moore.i1
  moore.divs %arg4, %arg4 : !moore.i1
  moore.modu %arg0, %arg0 : !moore.i1
  moore.mods %arg4, %arg4 : !moore.i1
  moore.and %arg0, %arg0 : !moore.i1
  moore.or %arg0, %arg0 : !moore.i1
  moore.xor %arg0, %arg0 : !moore.i1

  moore.ult %arg1, %arg1 : !moore.l1 -> !moore.l1
  moore.ule %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.ugt %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.uge %arg0, %arg0 : !moore.i1 -> !moore.i1

  moore.slt %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sle %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sgt %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sge %arg4, %arg4 : !moore.i1 -> !moore.i1

  moore.eq %arg1, %arg1 : !moore.l1 -> !moore.l1
  moore.ne %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.case_eq %arg0, %arg0 : !moore.i1
  moore.case_ne %arg0, %arg0 : !moore.i1
  moore.wildcard_eq %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.wildcard_ne %arg0, %arg0 : !moore.i1 -> !moore.i1

  %k0 = moore.conditional %arg0 : i1 -> i6 {
    moore.yield %arg2 : i6
  } {
    %0 = moore.constant 19 : i6
    moore.yield %0 : i6
  }
  moore.reduce_xor %k0 : i6 -> i1

  // CHECK: hw.struct_extract %arg1["value"]
  // CHECK: scf.if
  // Local variable in function uses llvm.alloca + llvm.store
  // CHECK:   llvm.alloca
  // CHECK:   llvm.store
  // CHECK:   scf.yield
  // CHECK: scf.yield
  %k1 = moore.conditional %arg1 : l1 -> l6 {
    %0 = moore.constant bXXXXXX : l6
    %var_l6 = moore.variable : !moore.ref<l6>
    moore.blocking_assign %var_l6, %0 : l6
    moore.yield %0 : l6
  } {
    %0 = moore.constant 19 : l6
    moore.yield %0 : l6
  }
  moore.reduce_xor %k1 : l6 -> l1

  // CHECK: return
  return
}

// CHECK-LABEL: func @ConvertReal
// CHECK-SAME: (%arg0: f32, %arg1: f64) -> f32
func.func @ConvertReal(%arg0: !moore.f32, %arg1: !moore.f64) -> !moore.f32 {
  // CHECK: arith.truncf %arg1 : f64 to f32
  %0 = moore.convert_real %arg0 : !moore.f32 -> !moore.f64
  %1 = moore.convert_real %arg1 : !moore.f64 -> !moore.f32
  // CHECK: return
  return %1 : !moore.f32
}

// CHECK-LABEL: ExtractRefArrayElement
func.func @ExtractRefArrayElement(%j: !moore.ref<array<1 x array<1 x l3>>>) -> (!moore.ref<array<1 x l3>>) {
  // CHECK: llhd.sig.array_get
  %0 = moore.extract_ref %j from 0 : !moore.ref<array<1 x array<1 x l3>>> -> !moore.ref<array<1 x l3>>
  return %0 : !moore.ref<array<1 x l3>>
}

// CHECK-LABEL: DynExtractArrayElement
func.func @DynExtractArrayElement(%j: !moore.array<2 x array<1 x l3>>, %idx: !moore.l1) -> (!moore.array<1 x l3>) {
  // CHECK: hw.array_get
  %0 = moore.dyn_extract %j from %idx : !moore.array<2 x array<1 x l3>>, !moore.l1 -> !moore.array<1 x l3>
  return %0 : !moore.array<1 x l3>
}

// CHECK-LABEL: DynExtractRefArrayElement
func.func @DynExtractRefArrayElement(%j: !moore.ref<array<2 x array<1 x l3>>>, %idx: !moore.l1) -> (!moore.ref<array<1 x l3>>) {
  // CHECK: llhd.sig.array_get
  %0 = moore.dyn_extract_ref %j from %idx : !moore.ref<array<2 x array<1 x l3>>>, !moore.l1 -> !moore.ref<array<1 x l3>>
  return %0 : !moore.ref<array<1 x l3>>
}

// CHECK-LABEL: func @AdvancedConversion
func.func @AdvancedConversion(%arg0: !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>) -> (!moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320) {
  // CHECK: [[V0:%.+]] = hw.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  %0 = moore.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  // CHECK: [[V1:%.+]] = hw.bitcast [[V0]] : (i320) -> !hw.array<5xstruct<exp_bits: i32, man_bits: i32>> 
  %1 = moore.sbv_to_packed %0 : array<5 x struct<{exp_bits: i32, man_bits: i32}>>
  // CHECK: [[V2:%.+]] = hw.bitcast %arg0 : (!hw.array<5xstruct<exp_bits: i32, man_bits: i32>>) -> i320
  %2 = moore.packed_to_sbv %arg0 : array<5 x struct<{exp_bits: i32, man_bits: i32}>>
  // CHECK: return [[V1]], [[V2]]
  return %1, %2 : !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320
}

// CHECK-LABEL: func @Statements
func.func @Statements(%arg0: !moore.i42) {
  // Local variables in functions use llvm.alloca for immediate memory semantics
  // CHECK: llvm.alloca
  %x = moore.variable : !moore.ref<i42>
  // CHECK: llvm.store %arg0
  moore.blocking_assign %x, %arg0 : i42
  // CHECK: llvm.store %arg0
  moore.nonblocking_assign %x, %arg0 : i42
  // CHECK: return
  return
}

// CHECK-LABEL: func @FormatStrings
func.func @FormatStrings(%arg0: !moore.i42, %arg1: !moore.f32, %arg2: !moore.f64) {
  // CHECK: sim.fmt.literal "hello"
  %0 = moore.fmt.literal "hello"
  %1 = moore.fmt.concat (%0, %0)
  moore.fmt.int decimal %arg0, align right, pad space width 42 : i42
  moore.fmt.int decimal %arg0, align left, pad zero : i42
  moore.fmt.int decimal %arg0, align right, pad space signed : i42
  moore.fmt.int binary %arg0, align right, pad space width 42 : i42
  moore.fmt.int binary %arg0, align left, pad zero : i42
  moore.fmt.int octal %arg0, align right, pad space width 42 : i42
  moore.fmt.int octal %arg0, align right, pad zero width 42 : i42
  moore.fmt.int hex_lower %arg0, align right, pad space width 42 : i42
  moore.fmt.int hex_lower %arg0, align right, pad zero : i42
  moore.fmt.int hex_upper %arg0, align right, pad space width 42 : i42

  moore.fmt.real float %arg1, align left : f32
  moore.fmt.real exponential %arg2, align left : f64
  moore.fmt.real general %arg1, align left fracDigits 6 : f32
  moore.fmt.real float %arg2, align left fracDigits 10 : f64
  moore.fmt.real exponential %arg1, align right fieldWidth 9 fracDigits 8 : f32
  moore.fmt.real general %arg2, align right : f64
  moore.fmt.real float %arg1, align right fieldWidth 15 : f32
  // CHECK: sim.proc.print
  moore.builtin.display %0
  return
}

// CHECK-LABEL: hw.module @InstanceNull() {
moore.module @InstanceNull() {

  // CHECK-NEXT: hw.instance "null_instance" @Null() -> ()
  moore.instance "null_instance" @Null() -> ()

  // CHECK-NEXT: hw.output
  moore.output
}

// CHECK-LABEL: hw.module private @Null() {
moore.module private @Null() {

  // CHECK-NEXT: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Top(in
// CHECK-SAME: !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: out out0 : !hw.struct<value: i1, unknown: i1>
moore.module @Top(in %arg0 : !moore.l1, in %arg1 : !moore.l1, out out0 : !moore.l1) {
  // CHECK: hw.instance "inst_0" @SubModule_0
  %inst_0.c = moore.instance "inst_0" @SubModule_0(a: %arg0 : !moore.l1, b: %arg1 : !moore.l1) -> (c: !moore.l1)

  // CHECK: hw.instance "inst_1" @SubModule_0
  %inst_1.c = moore.instance "inst_1" @SubModule_0(a: %inst_0.c : !moore.l1, b: %arg1 : !moore.l1) -> (c: !moore.l1)

  // CHECK: hw.output
  moore.output %inst_1.c : !moore.l1
}

// CHECK-LABEL: hw.module private @SubModule_0(in
// CHECK-SAME: !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: out c : !hw.struct<value: i1, unknown: i1>
moore.module private @SubModule_0(in %a : !moore.l1, in %b : !moore.l1, out c : !moore.l1) {
  // CHECK: hw.struct_extract
  // CHECK: hw.struct_extract
  // CHECK: hw.struct_create
  %0 = moore.and %a, %b : !moore.l1

  // CHECK: hw.output
  moore.output %0 : !moore.l1
}

// CHECK-LABEL: hw.module @PreservePortOrderTop(
// CHECK-SAME:    out a : i42,
// CHECK-SAME:    in %b : i42
// CHECK-SAME:  ) {
moore.module @PreservePortOrderTop(out a: !moore.i42, in %b: !moore.i42) {
  // CHECK: [[TMP:%.+]] = hw.instance "inst" @PreservePortOrder(x: %b: i42, z: %b: i42) -> (y: i42)
  // CHECK: hw.output [[TMP]] : i42
  %0 = moore.instance "inst" @PreservePortOrder(x: %b: !moore.i42, z: %b: !moore.i42) -> (y: !moore.i42)
  moore.output %0 : !moore.i42
}

// CHECK-LABEL: hw.module private @PreservePortOrder(
// CHECK-SAME:    in %x : i42,
// CHECK-SAME:    out y : i42,
// CHECK-SAME:    in %z : i42
// CHECK-SAME:  ) {
moore.module private @PreservePortOrder(in %x: !moore.i42, out y: !moore.i42, in %z: !moore.i42) {
  moore.output %x : !moore.i42
}

// CHECK-LABEL: hw.module @Variable
moore.module @Variable() {
  // CHECK: %a = llhd.sig
  %a = moore.variable : !moore.ref<i32>

  %b1 = moore.variable : !moore.ref<i8>

  %0 = moore.read %b1 : !moore.ref<i8>
  %b2 = moore.variable %0 : !moore.ref<i8>

  %1 = moore.constant 1 : l1
  %l = moore.variable %1 : !moore.ref<l1>
  %m = moore.variable : !moore.ref<l19>

  %3 = moore.constant 10 : i32

  // CHECK: llhd.drv %a,
  moore.assign %a, %3 : i32

  moore.variable : <chandle>

  moore.variable : <time>

  %c42_fs = moore.constant_time 42 fs
  moore.variable %c42_fs : <time>

  moore.variable : <f32>

  moore.variable : <f64>

  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Net
moore.module @Net() {
  // CHECK: %a = llhd.sig
  %a = moore.net wire : !moore.ref<i32>

  // CHECK: llhd.prb %a
  %0 = moore.read %a : !moore.ref<i32>

  // CHECK: %b = llhd.sig
  // CHECK: llhd.drv %b,
  %b = moore.net wire %0 : !moore.ref<i32>

  %3 = moore.constant 10 : i32
  // CHECK: llhd.drv %a,
  moore.assign %a, %3 : i32

  // Test supply0 net (always driven to 0)
  // Note: unused supply nets are removed by DCE
  %supply0_net = moore.net name "supply0_net" supply0 : !moore.ref<i8>

  // Test supply1 net (always driven to all 1s)
  // Note: unused supply nets are removed by DCE
  %supply1_net = moore.net name "supply1_net" supply1 : !moore.ref<i8>
}

// CHECK-LABEL: hw.module @NetLogic
// CHECK-DAG: [[SUP1_INIT:%.+]] = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
// CHECK-DAG: [[SUP0_INIT:%.+]] = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
// CHECK-DAG: [[WIRE_INIT:%.+]] = hw.aggregate_constant [false, true] : !hw.struct<value: i1, unknown: i1>
// CHECK: llhd.sig [[WIRE_INIT]] : !hw.struct<value: i1, unknown: i1>
// CHECK: llhd.sig [[SUP0_INIT]] : !hw.struct<value: i1, unknown: i1>
// CHECK: llhd.sig [[SUP1_INIT]] : !hw.struct<value: i1, unknown: i1>
moore.module @NetLogic(out a : !moore.l1, out b : !moore.l1, out c : !moore.l1) {
  %wire = moore.net wire : !moore.ref<l1>
  %supply0 = moore.net supply0 : !moore.ref<l1>
  %supply1 = moore.net supply1 : !moore.ref<l1>
  %0 = moore.read %wire : !moore.ref<l1>
  %1 = moore.read %supply0 : !moore.ref<l1>
  %2 = moore.read %supply1 : !moore.ref<l1>
  moore.output %0, %1, %2 : !moore.l1, !moore.l1, !moore.l1
}

// CHECK-LABEL: hw.module @UnpackedArray
moore.module @UnpackedArray(in %arr : !moore.uarray<2 x i32>, in %sel : !moore.i1, out c : !moore.i32) {
  // CHECK: hw.array_get %arr[%sel] : !hw.array<2xi32>, i1
  %0 = moore.dyn_extract %arr from %sel : !moore.uarray<2 x i32>, !moore.i1 -> !moore.i32

  %1 = moore.extract %arr from 1 : !moore.uarray<2 x i32> -> !moore.i32

  %2 = moore.variable : !moore.ref<uarray<4 x i32>>

  // CHECK: llhd.sig.array_get
  %3 = moore.extract_ref %2 from 1 : !moore.ref<uarray<4 x i32>> -> !moore.ref<i32>
  moore.assign %3, %0 : i32

  %4 = moore.variable : !moore.ref<uarray<4 x uarray<8 x array<8 x i4>>>>

  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @Struct
moore.module @Struct(in %a : !moore.i32, in %b : !moore.i32, in %arg0 : !moore.struct<{exp_bits: i32, man_bits: i32}>, in %arg1 : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>>, out a : !moore.i32, out b : !moore.struct<{exp_bits: i32, man_bits: i32}>, out c : !moore.struct<{exp_bits: i32, man_bits: i32}>) {
  // CHECK: hw.struct_extract %arg0["exp_bits"] : !hw.struct<exp_bits: i32, man_bits: i32>
  %0 = moore.struct_extract %arg0, "exp_bits" : !moore.struct<{exp_bits: i32, man_bits: i32}> -> i32

  // CHECK: llhd.sig.struct_extract %arg1["exp_bits"] : <!hw.struct<exp_bits: i32, man_bits: i32>>
  %ref = moore.struct_extract_ref %arg1, "exp_bits" : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>> -> !moore.ref<i32>
  moore.assign %ref, %0 : !moore.i32

  // CHECK: llhd.sig
  // CHECK: llhd.sig %arg0 : !hw.struct<exp_bits: i32, man_bits: i32>
  %1 = moore.variable : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>>
  %2 = moore.variable %arg0 : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>>

  %3 = moore.read %1 : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>>
  %4 = moore.read %2 : !moore.ref<struct<{exp_bits: i32, man_bits: i32}>>

  moore.struct_create %a, %b : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>

  moore.output %0, %3, %4 : !moore.i32, !moore.struct<{exp_bits: i32, man_bits: i32}>, !moore.struct<{exp_bits: i32, man_bits: i32}>
}

// CHECK-LABEL: func @ArrayCreate
// CHECK-SAME: () ->  !hw.array<2xi8>
func.func @ArrayCreate() -> !moore.array<2x!moore.i8> {
  %c0 = moore.constant 42 : !moore.i8
  // CHECK: hw.aggregate_constant
  %arr = moore.array_create %c0, %c0 : !moore.i8, !moore.i8 -> !moore.array<2x!moore.i8>
  return %arr : !moore.array<2x!moore.i8>
}

// CHECK-LABEL: func @UnpackedArrayCreate
// CHECK-SAME: () ->  !hw.array<2xi8>
func.func @UnpackedArrayCreate() -> !moore.uarray<2x!moore.i8> {
  %a = moore.constant 7 : !moore.i8
  // CHECK: hw.aggregate_constant
  %arr = moore.array_create %a, %a : !moore.i8, !moore.i8 -> !moore.uarray<2x!moore.i8>
  return %arr : !moore.uarray<2x!moore.i8>
}

// CHECK-LABEL:   hw.module @UnpackedStruct
moore.module @UnpackedStruct() {
  %0 = moore.constant 1 : i32
  %1 = moore.constant 0 : i32

  // CHECK: %ms = llhd.sig
  %ms = moore.variable : !moore.ref<ustruct<{a: i32, b: i32}>>

  // CHECK: llhd.process {
  moore.procedure initial {
    %2 = moore.struct_create %1, %0 : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>

    // CHECK: llhd.drv %ms,
    moore.blocking_assign %ms, %2 : ustruct<{a: i32, b: i32}>

    %3 = moore.struct_create %0, %0 : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>

    // CHECK: llhd.drv %ms,
    moore.blocking_assign %ms, %3 : ustruct<{a: i32, b: i32}>

    // CHECK: llhd.drv %ms,
    moore.blocking_assign %ms, %3 : ustruct<{a: i32, b: i32}>

    // CHECK: llhd.halt
    moore.return
  }
  moore.output
}

// CHECK-LABEL: func.func @CaseXZ
func.func @CaseXZ(%arg0: !moore.l8, %arg1: !moore.l8) {
  %0 = moore.constant b10XX01ZZ : l8
  %1 = moore.constant b1XX01ZZ0 : l8

  moore.casez_eq %arg0, %arg1 : l8
  moore.casez_eq %arg0, %1 : l8
  moore.casez_eq %0, %arg1 : l8
  moore.casez_eq %0, %1 : l8

  moore.casexz_eq %arg0, %arg1 : l8
  moore.casexz_eq %arg0, %1 : l8
  moore.casexz_eq %0, %arg1 : l8
  moore.casexz_eq %0, %1 : l8

  return
}

// CHECK-LABEL: func.func @CmpReal
func.func @CmpReal(%arg0: !moore.f32, %arg1: !moore.f32) {
  moore.fne %arg0, %arg1 : f32 -> i1
  moore.flt %arg0, %arg1 : f32 -> i1
  moore.fle %arg0, %arg1 : f32 -> i1
  moore.fgt %arg0, %arg1 : f32 -> i1
  moore.fge %arg0, %arg1 : f32 -> i1
  moore.feq %arg0, %arg1 : f32 -> i1

  return
}

// CHECK-LABEL: func.func @BinaryRealOps
func.func @BinaryRealOps(%arg0: !moore.f32, %arg1: !moore.f32) {
  moore.fadd %arg0, %arg1 : f32
  moore.fsub %arg0, %arg1 : f32
  moore.fdiv %arg0, %arg1 : f32
  moore.fmul %arg0, %arg1 : f32
  moore.fpow %arg0, %arg1 : f32

  return
}

// CHECK-LABEL: hw.module @Procedures
moore.module @Procedures() {
  // CHECK: seq.initial() {
  // CHECK:   func.call @dummyA()
  // CHECK: } : () -> ()
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.final {
  // CHECK:   func.call @dummyA()
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure final {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB:.+]]
  // CHECK: ^[[BB]]:
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[BB]]
  // CHECK: }
  moore.procedure always {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB:.+]]
  // CHECK: ^[[BB]]:
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[BB]]
  // CHECK: }
  moore.procedure always_ff {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // TODO: moore.procedure always_comb
  // TODO: moore.procedure always_latch
}

func.func private @dummyA() -> ()
func.func private @dummyB() -> ()
func.func private @dummyC() -> ()

// CHECK-LABEL: hw.module @WaitEvent
moore.module @WaitEvent() {
  // CHECK: %a = llhd.sig
  // CHECK: [[PRB_A6:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A5:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A4:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A3:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A2:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A1:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A0:%.+]] = llhd.prb %a
  // CHECK: %b = llhd.sig
  // CHECK: [[PRB_B2:%.+]] = llhd.prb %b
  // CHECK: [[PRB_B1:%.+]] = llhd.prb %b
  // CHECK: [[PRB_B0:%.+]] = llhd.prb %b
  // CHECK: %c = llhd.sig
  // CHECK: [[PRB_C:%.+]] = llhd.prb %c
  // CHECK: %d = llhd.sig
  // CHECK: [[PRB_D4:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D3:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D2:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D1:%.+]] = llhd.prb %d
  %a = moore.variable : !moore.ref<i1>
  %b = moore.variable : !moore.ref<i1>
  %c = moore.variable : !moore.ref<i1>
  %d = moore.variable : !moore.ref<i4>

  // CHECK: llhd.process {
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[WAIT:.+]]
  // CHECK: ^[[WAIT]]:
  // CHECK:   func.call @dummyB()
  // CHECK:   llhd.wait ^[[CHECK:.+]]
  // CHECK: ^[[CHECK]]:
  // CHECK:   func.call @dummyB()
  // CHECK:   cf.br
  // CHECK:   func.call @dummyC()
  // Unused moore.read is dropped (no llhd.prb generated)
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.wait_event {
      func.call @dummyB() : () -> ()
    }
    func.call @dummyC() : () -> ()
    moore.read %a : !moore.ref<i1>
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      moore.detect_event any %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      %1 = moore.read %b : !moore.ref<i1>
      moore.detect_event any %0 if %1 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      %1 = moore.read %b : !moore.ref<i1>
      %2 = moore.read %c : !moore.ref<i1>
      %3 = moore.read %d : !moore.ref<i4>
      moore.detect_event any %0 : i1
      moore.detect_event any %1 : i1
      moore.detect_event any %2 : i1
      moore.detect_event any %3 : i4
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      moore.detect_event posedge %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      moore.detect_event negedge %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      moore.detect_event edge %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %d : !moore.ref<i4>
      moore.detect_event posedge %0 : i4
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %d : !moore.ref<i4>
      moore.detect_event negedge %0 : i4
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait
    moore.wait_event {
      %0 = moore.read %d : !moore.ref<i4>
      moore.detect_event edge %0 : i4
    }
    moore.return
  }

  %cond = moore.constant 0 : i1
  // CHECK: llhd.process {
  moore.procedure always_comb {
    // CHECK: llhd.wait
    %1 = moore.conditional %cond : i1 -> i1 {
      %2 = moore.read %a : !moore.ref<i1>
      moore.yield %2 : !moore.i1
    } {
      %3 = moore.read %b : !moore.ref<i1>
      moore.yield %3 : !moore.i1
    }
    moore.return
  }

  %e = moore.variable : !moore.ref<i1>

  // CHECK: llhd.halt
  moore.procedure always_latch {
    %3 = moore.read %e : !moore.ref<i1>
    moore.return
  }

  moore.procedure initial {
    moore.wait_event {
      %0 = moore.constant 0 : i1
      %1 = moore.conditional %0 : i1 -> i1 {
        %2 = moore.read %a : !moore.ref<i1>
        moore.yield %2 : !moore.i1
      } {
        %3 = moore.read %b : !moore.ref<i1>
        moore.yield %3 : !moore.i1
      }
      moore.detect_event any %1 : i1
    }
    moore.return
  }
  // CHECK: hw.output
}

// CHECK-LABEL: hw.module @EmptyWaitEvent(
moore.module @EmptyWaitEvent(out out : !moore.l32) {
  // CHECK: %out = llhd.sig
  // CHECK: llhd.process {
  %0 = moore.constant 0 : l32
  %out = moore.variable : !moore.ref<l32>
  moore.procedure always {
    moore.wait_event {
    }
    moore.blocking_assign %out, %0 : l32
    moore.return
  }
  %1 = moore.read %out : !moore.ref<l32>
  moore.output %1 : !moore.l32
}


// CHECK-LABEL: hw.module @WaitDelay
moore.module @WaitDelay(in %d: !moore.time) {
  // CHECK: llhd.process {
  // CHECK:   llhd.wait delay
  moore.procedure initial {
    %0 = moore.constant_time 1000000 fs
    func.call @dummyA() : () -> ()
    moore.wait_delay %0
    func.call @dummyB() : () -> ()
    moore.wait_delay %d
    func.call @dummyC() : () -> ()
    moore.return
  }
}

// Just check that block without predecessors are handled without crashing
// CHECK-LABEL: @NoPredecessorBlockErasure
moore.module @NoPredecessorBlockErasure(in %clk_i : !moore.l1, in %raddr_i : !moore.array<2 x l5>, out rdata_o : !moore.array<2 x l32>, in %waddr_i : !moore.array<1 x l5>, in %wdata_i : !moore.array<1 x l32>, in %we_i : !moore.l1) {
  %0 = moore.constant 0 : l32
  %1 = moore.constant 1 : i32
  %2 = moore.constant 0 : i32
  %rdata_o = moore.variable : !moore.ref<array<2 x l32>>
  %mem = moore.variable : !moore.ref<array<32 x l32>>
  moore.procedure always_ff {
    cf.br ^bb1(%2 : !moore.i32)
  ^bb1(%4: !moore.i32):  // 2 preds: ^bb0, ^bb8
    moore.return
  ^bb2:  // no predecessors
    cf.br ^bb3(%2 : !moore.i32)
  ^bb3(%5: !moore.i32):  // 2 preds: ^bb2, ^bb6
    cf.br ^bb8
  ^bb4:  // no predecessors
    cf.br ^bb6
  ^bb5:  // no predecessors
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %6 = moore.add %5, %1 : i32
    cf.br ^bb3(%6 : !moore.i32)
  ^bb7:  // no predecessors
    %7 = moore.extract_ref %mem from 0 : !moore.ref<array<32 x l32>> -> !moore.ref<l32>
    moore.nonblocking_assign %7, %0 : l32
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb3, ^bb7
    %8 = moore.add %4, %1 : i32
    cf.br ^bb1(%8 : !moore.i32)
  }
  %3 = moore.read %rdata_o : !moore.ref<array<2 x l32>>
  moore.output %3 : !moore.array<2 x l32>
}

%dbg0 = moore.constant 42 : l32
dbg.variable "a", %dbg0 : !moore.l32
%dbg1 = dbg.scope "foo", "bar"
dbg.variable "b", %dbg0 scope %dbg1 : !moore.l32
dbg.array [%dbg0] : !moore.l32
dbg.struct {"q": %dbg0} : !moore.l32

// CHECK-LABEL: hw.module @Assert
moore.module @Assert(in %cond : !moore.l1)  {
  moore.procedure always {
  // CHECK: verif.assert
  moore.assert immediate %cond label "cond" : l1
  // CHECK: verif.assume
  moore.assume observed %cond  : l1
  // CHECK: verif.cover
  moore.cover final %cond : l1
  moore.return
  }
}

// CHECK-LABEL: func.func @ConstantString
func.func @ConstantString() {
  %str = moore.constant_string "Test" : i32
  %str1 = moore.constant_string "Test" : i36
  %str2 = moore.constant_string "Test" : i8
  %str_trunc = moore.constant_string "Test" : i7
  %str_trunc1 = moore.constant_string "Test" : i17
  %str_empty = moore.constant_string "" : i0
  %str_empty_zext = moore.constant_string "" : i8
  return
}

// CHECK-LABEL: func.func @RecurciveConditional
func.func @RecurciveConditional(%arg0 : !moore.l1, %arg1 : !moore.l1) {
  %c_2 = moore.constant -2 : l2
  %c_1 = moore.constant 1 : l2
  %c_0 = moore.constant 0 : l2

  %0 = moore.conditional %arg0 : l1 -> l2 {
    %1 = moore.conditional %arg1 : l1 -> l2 {
      moore.yield %c_0 : l2
    } {
      moore.yield %c_1 : l2
    }
    moore.yield %1 : l2
  } {
    moore.yield %c_2 : l2
  }

  return
}

// CHECK-LABEL: func.func @Conversions
func.func @Conversions(%arg0: !moore.i16, %arg1: !moore.l16, %arg2: !moore.l1) {
  // CHECK: comb.extract %arg0 from 0 : (i16) -> i8
  // CHECK: dbg.variable "trunc"
  %0 = moore.trunc %arg0 : i16 -> i8
  dbg.variable "trunc", %0 : !moore.i8

  // CHECK: dbg.variable "zext"
  %1 = moore.zext %arg0 : i16 -> i32
  dbg.variable "zext", %1 : !moore.i32

  // CHECK: dbg.variable "sext"
  %2 = moore.sext %arg0 : i16 -> i32
  dbg.variable "sext", %2 : !moore.i32

  %3 = moore.int_to_logic %arg0 : i16
  dbg.variable "i2l", %3 : !moore.l16

  %4 = moore.logic_to_int %arg1 : l16
  dbg.variable "l2i", %4 : !moore.i16

  %5 = moore.to_builtin_bool %arg2 : l1
  dbg.variable "builtin_bool", %5 : i1

  return
}

// CHECK-LABEL: func.func @PowUOp
func.func @PowUOp(%arg0: !moore.i32, %arg1: !moore.i32) {
  %0 = moore.powu %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: func.func @PowSOp
func.func @PowSOp(%arg0: !moore.i32, %arg1: !moore.i32) {
  %0 = moore.pows %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @scfInsideProcess
moore.module @scfInsideProcess(in %in0: !moore.i32, in %in1: !moore.i32) {
  %var = moore.variable : <i32>
  // CHECK: llhd.process
  // CHECK-NOT: scf.for
  moore.procedure initial {
    %0 = moore.pows %in0, %in1 : !moore.i32
    moore.blocking_assign %var, %0 : !moore.i32
    moore.return
  }
}

// CHECK-LABEL: @blockArgAsObservedValue
moore.module @blockArgAsObservedValue(in %in0: !moore.i32, in %in1: !moore.i32) {
  %var = moore.variable : <i32>
  // CHECK: llhd.process
  moore.procedure always_comb {
      %0 = moore.add %in0, %in1 : !moore.i32
      moore.blocking_assign %var, %0 : !moore.i32
      // CHECK:   llhd.wait (%in0, %in1 : i32, i32), ^bb1
      moore.return
  }
}

// CHECK-LABEL: @Time
// CHECK-SAME: (%arg0: i64) -> (i64, i64)
func.func @Time(%arg0: !moore.time) -> (!moore.time, !moore.time) {
  // CHECK-NEXT: [[TMP:%.+]] = hw.constant 1234000 : i64
  %0 = moore.constant_time 1234000 fs
  // CHECK-NEXT: return %arg0, [[TMP]] : i64, i64
  return %arg0, %0 : !moore.time, !moore.time
}

// CHECK-LABEL: @Unreachable
// Initial blocks with unreachable (from $finish) now use seq.initial
// The unreachable is converted to seq.yield
moore.module @Unreachable() {
  moore.procedure initial {
    moore.unreachable
  }
}

// CHECK-LABEL: @SimulationControl
func.func @SimulationControl() {
  // CHECK-NOT: moore.builtin.finish_message
  moore.builtin.finish_message false
  moore.builtin.finish_message true

  // CHECK-NEXT: sim.pause quiet
  moore.builtin.stop

  // CHECK-NEXT: sim.terminate success, quiet
  moore.builtin.finish 0
  // CHECK-NEXT: sim.terminate failure, quiet
  moore.builtin.finish 42

  return
}

// CHECK-LABEL: @SeverityToPrint
func.func @SeverityToPrint() {
  // CHECK: sim.proc.print
  %0 = moore.fmt.literal "Fatal condition met!"
  moore.builtin.severity fatal %0

  // CHECK: sim.proc.print
  %1 = moore.fmt.literal "Error condition met!"
  moore.builtin.severity error %1

  // CHECK: sim.proc.print
  %2 = moore.fmt.literal "Warning condition met!"
  moore.builtin.severity warning %2

  return
}

// CHECK-LABEL: func.func @CHandle(%arg0: !llvm.ptr)
func.func @CHandle(%arg0: !moore.chandle) {
    return
}

// CHECK-LABEL: @MultiDimensionalSlice
moore.module @MultiDimensionalSlice(in %in : !moore.array<2 x array<2 x l2>>, out out : !moore.array<2 x l2>) {
  // CHECK: hw.array_get %in
  %0 = moore.extract %in from 0 : array<2 x array<2 x l2>> -> array<2 x l2>
  moore.output %0 : !moore.array<2 x l2>
}

// CHECK-LABEL: hw.module @ContinuousAssignment
// CHECK-SAME: in %a : !llhd.ref<i42>
// CHECK-SAME: in %b : i42
// CHECK-SAME: in %c : i64
moore.module @ContinuousAssignment(in %a: !moore.ref<i42>, in %b: !moore.i42, in %c: !moore.time) {
  // For continuous assignments from block arguments, use zero epsilon delay
  // to fix initialization order issues (signals read correct value at t=0).
  // CHECK-NEXT: [[DELTA:%.+]] = llhd.constant_time <0ns, 0d, 0e>
  // CHECK-NEXT: llhd.drv %a, %b after [[DELTA]]
  moore.assign %a, %b : i42
  // CHECK-NEXT: [[TIME:%.+]] = llhd.int_to_time %c
  // CHECK-NEXT: llhd.drv %a, %b after [[TIME]]
  moore.delayed_assign %a, %b, %c : i42
}

// CHECK-LABEL: func.func @NonBlockingAssignment
// CHECK-SAME: %arg0: !llhd.ref<i42>
// CHECK-SAME: %arg1: i42
// CHECK-SAME: %arg2: i64
func.func @NonBlockingAssignment(%arg0: !moore.ref<i42>, %arg1: !moore.i42, %arg2: !moore.time) {
  // For function ref parameters, use llvm.store instead of llhd.drv
  // CHECK-NEXT: [[PTR:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<i42> to !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, [[PTR]] : i42, !llvm.ptr
  moore.nonblocking_assign %arg0, %arg1 : i42
  // Delayed nonblocking assign to ref parameter also uses llvm.store
  // CHECK-NEXT: [[PTR2:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<i42> to !llvm.ptr
  // CHECK-NEXT: llvm.store %arg1, [[PTR2]] : i42, !llvm.ptr
  moore.delayed_nonblocking_assign %arg0, %arg1, %arg2 : i42
  return
}

// CHECK-LABEL: func.func @ConstantReals
func.func @ConstantReals() {
  moore.constant_real 1.234500e+00 : f32
  moore.constant_real 1.234500e+00 : f64
  return
}

// CHECK-LABEL: func.func @IntToRealLowering
func.func @IntToRealLowering(%arg0: !moore.i32, %arg1: !moore.i42) {
  %0 = moore.sint_to_real %arg0 : i32 -> f32
  %1 = moore.uint_to_real %arg1 : i42 -> f64
  return
}

// CHECK-LABEL: func.func @RealToIntLowering
func.func @RealToIntLowering(%arg0: !moore.f32, %arg1: !moore.f64) {
  %0 = moore.real_to_int %arg0 : f32 -> i42
  %1 = moore.real_to_int %arg1 : f64 -> i42
  return
}

// CHECK-LABEL: func.func @RealToBitsLowering
func.func @RealToBitsLowering(%arg0: !moore.f64, %arg1: !moore.f32) {
  %0 = moore.builtin.realtobits %arg0
  %1 = moore.builtin.shortrealtobits %arg1
  return
}

// CHECK-LABEL: func.func @BitsToRealLowering
func.func @BitsToRealLowering(%arg0: !moore.i64, %arg1: !moore.i32) {
  %0 = moore.builtin.bitstoreal %arg0 : i64
  %1 = moore.builtin.bitstoshortreal %arg1 : i32
  return
}

// CHECK-LABEL: func.func @CurrentTime
// CHECK-SAME: () -> i64
func.func @CurrentTime() -> !moore.time {
  // CHECK-NEXT: [[TMP:%.+]] = llhd.current_time
  // CHECK-NEXT: [[TMP2:%.+]] = llhd.time_to_int [[TMP]]
  %0 = moore.builtin.time
  // CHECK-NEXT: return [[TMP2]] : i64
  return %0 : !moore.time
}

// CHECK-LABEL: func.func @TimeConversion
// CHECK-SAME: (%arg0: !hw.struct<value: i64, unknown: i64>, %arg1: i64) -> (i64, !hw.struct<value: i64, unknown: i64>)
func.func @TimeConversion(%arg0: !moore.l64, %arg1: !moore.time) -> (!moore.time, !moore.l64) {
  // Note: The hw.constant 0 is hoisted before the struct_extract due to CSE.
  // CHECK-DAG: hw.struct_extract
  %0 = moore.logic_to_time %arg0
  // CHECK-DAG: hw.constant 0 : i64
  // CHECK-DAG: hw.struct_create
  %1 = moore.time_to_logic %arg1
  // CHECK: return
  return %0, %1 : !moore.time, !moore.l64
}

// CHECK-LABEL: func.func @RefToRefConversion
func.func @RefToRefConversion(%arg0: !moore.ref<uarray<16 x l1>>) -> !moore.ref<l16> {
  // CHECK: llhd.prb
  // CHECK: hw.bitcast
  // CHECK: llhd.sig
  // CHECK: return
  %0 = moore.conversion %arg0 : !moore.ref<uarray<16 x l1>> -> !moore.ref<l16>
  return %0 : !moore.ref<l16>
}

// CHECK-LABEL: func.func @RefToRefConversionWidthChange
func.func @RefToRefConversionWidthChange(%arg0: !moore.ref<i8>) -> !moore.ref<i16> {
  // CHECK: llhd.prb
  // CHECK: comb.concat
  // CHECK: llhd.sig
  // CHECK: return
  %0 = moore.conversion %arg0 : !moore.ref<i8> -> !moore.ref<i16>
  return %0 : !moore.ref<i16>
}

// CHECK-LABEL: func.func @ValueToRefConversion
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @ValueToRefConversion(%arg0: !moore.i32) -> !moore.ref<i32> {
  // CHECK: [[SIG:%.+]] = llhd.sig %[[ARG0]] : i32
  // CHECK: return [[SIG]] : !llhd.ref<i32>
  %0 = moore.conversion %arg0 : !moore.i32 -> !moore.ref<i32>
  return %0 : !moore.ref<i32>
}

// CHECK-LABEL: func.func @Extract4StateToFourState
// CHECK-SAME: (%[[ARG0:.*]]: !hw.struct<value: i8, unknown: i8>) -> !hw.struct<value: i2, unknown: i2>
func.func @Extract4StateToFourState(%arg0: !moore.l8) -> !moore.l2 {
  // CHECK: %[[VALUE:.*]] = hw.struct_extract %[[ARG0]]["value"]
  // CHECK: %[[UNKNOWN:.*]] = hw.struct_extract %[[ARG0]]["unknown"]
  // CHECK: %[[VEXT:.*]] = comb.extract %[[VALUE]] from 3
  // CHECK: %[[UEXT:.*]] = comb.extract %[[UNKNOWN]] from 3
  // CHECK: %[[RESULT:.*]] = hw.struct_create (%[[VEXT]], %[[UEXT]])
  // CHECK: return %[[RESULT]]
  %0 = moore.extract %arg0 from 3 : !moore.l8 -> !moore.l2
  return %0 : !moore.l2
}

// CHECK-LABEL: func.func @Extract4StateTo2State
// CHECK-SAME: (%[[ARG0:.*]]: !hw.struct<value: i8, unknown: i8>) -> i2
func.func @Extract4StateTo2State(%arg0: !moore.l8) -> !moore.i2 {
  // CHECK: %[[VALUE:.*]] = hw.struct_extract %[[ARG0]]["value"]
  // CHECK: %[[VEXT:.*]] = comb.extract %[[VALUE]] from 4
  // CHECK: return %[[VEXT]]
  %0 = moore.extract %arg0 from 4 : !moore.l8 -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @Extract4StateOutOfBounds
// CHECK-SAME: (%[[ARG0:.*]]: !hw.struct<value: i4, unknown: i4>) -> !hw.struct<value: i4, unknown: i4>
func.func @Extract4StateOutOfBounds(%arg0: !moore.l4) -> !moore.l4 {
  // Test out-of-bounds high (extracting from bit 2 with width 4 from a 4-bit value)
  // CHECK: hw.struct_extract
  // CHECK: hw.struct_extract
  // CHECK: comb.extract
  // CHECK: comb.concat
  // CHECK: hw.struct_create
  %0 = moore.extract %arg0 from 2 : !moore.l4 -> !moore.l4
  return %0 : !moore.l4
}

// CHECK-LABEL: func.func @Extract4State1BitFrom1Bit
// CHECK-SAME: (%[[ARG0:.*]]: !hw.struct<value: i1, unknown: i1>) -> !hw.struct<value: i1, unknown: i1>
func.func @Extract4State1BitFrom1Bit(%arg0: !moore.l1) -> !moore.l1 {
  // This is the pattern that was failing in hdl_top.sv
  // CHECK: %[[VALUE:.*]] = hw.struct_extract %[[ARG0]]["value"]
  // CHECK: %[[UNKNOWN:.*]] = hw.struct_extract %[[ARG0]]["unknown"]
  // CHECK: hw.struct_create (%[[VALUE]], %[[UNKNOWN]])
  %0 = moore.extract %arg0 from 0 : !moore.l1 -> !moore.l1
  return %0 : !moore.l1
}
