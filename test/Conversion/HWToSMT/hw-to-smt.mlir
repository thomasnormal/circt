// RUN: circt-opt %s --convert-hw-to-smt | FileCheck %s

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: smt.bv.constant #smt.bv<42> : !smt.bv<32>
  %c42_i32 = hw.constant 42 : i32
  // CHECK: smt.bv.constant #smt.bv<-1> : !smt.bv<3>
  %c-1_i32 = hw.constant -1 : i3
  // CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %false = hw.constant false

  return
}

// CHECK-LABEL: func.func @modA(%{{.*}}: !smt.bv<32>) -> !smt.bv<32>
hw.module @modA(in %in: i32, out out: i32) {
  // CHECK-NEXT: return
  hw.output %in : i32
}

// CHECK-LABEL: func.func @modB(%{{.*}}: !smt.bv<32>) -> !smt.bv<32>
hw.module @modB(in %in: i32, out out: i32) {
  // CHECK-NEXT: [[V:%.+]] = call @modA(%{{.*}}) : (!smt.bv<32>) -> !smt.bv<32>
  %0 = hw.instance "inst" @modA(in: %in: i32) -> (out: i32)
  // CHECK-NEXT: return [[V]] : !smt.bv<32>
  hw.output %0 : i32
}

// CHECK-LABEL: func.func @inject
// CHECK-SAME: (%[[ARR:.+]]: !smt.array<[!smt.bv<2> -> !smt.bv<8>]>, %[[IDX:.+]]: !smt.bv<2>, %[[VAL:.+]]: !smt.bv<8>)
hw.module @inject(in %arr: !hw.array<3xi8>, in %index: i2, in %v: i8, out out: !hw.array<3xi8>) {
  // CHECK-NEXT: %[[OOB:.+]] = smt.declare_fun : !smt.array<[!smt.bv<2> -> !smt.bv<8>]>
  // CHECK-NEXT: %[[C2:.+]] = smt.bv.constant #smt.bv<-2> : !smt.bv<2>
  // CHECK-NEXT: %[[CMP:.+]] = smt.bv.cmp ule %[[IDX]], %[[C2]] : !smt.bv<2>
  // CHECK-NEXT: %[[STORED:.+]] = smt.array.store %[[ARR]][%[[IDX]]], %[[VAL]] : !smt.array<[!smt.bv<2> -> !smt.bv<8>]>
  // CHECK-NEXT: %[[RESULT:.+]] = smt.ite %[[CMP]], %[[STORED]], %[[OOB]] : !smt.array<[!smt.bv<2> -> !smt.bv<8>]>
  // CHECK-NEXT: return %[[RESULT]] : !smt.array<[!smt.bv<2> -> !smt.bv<8>]>
  %arr_injected = hw.array_inject %arr[%index], %v : !hw.array<3xi8>, i2
  hw.output %arr_injected : !hw.array<3xi8>
}

// CHECK-LABEL: func.func @struct_create
// CHECK-SAME: (%{{.+}}: !smt.bv<1>, %{{.+}}: !smt.bv<1>) -> !smt.bv<2>
hw.module @struct_create(in %a: i1, in %b: i1, out out: !hw.struct<value: i1, unknown: i1>) {
  // CHECK-NEXT: %[[CONCAT:.+]] = smt.bv.concat %{{.+}}, %{{.+}} : !smt.bv<1>, !smt.bv<1>
  // CHECK-NEXT: return %[[CONCAT]] : !smt.bv<2>
  %s = hw.struct_create (%a, %b) : !hw.struct<value: i1, unknown: i1>
  hw.output %s : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: func.func @struct_extract
// CHECK-SAME: (%{{.+}}: !smt.bv<2>) -> !smt.bv<1>
hw.module @struct_extract(in %s: !hw.struct<value: i1, unknown: i1>, out out: i1) {
  // CHECK-NEXT: %[[EXTRACT:.+]] = smt.bv.extract %{{.+}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
  // CHECK-NEXT: return %[[EXTRACT]] : !smt.bv<1>
  %v = hw.struct_extract %s["value"] : !hw.struct<value: i1, unknown: i1>
  hw.output %v : i1
}

// CHECK-LABEL: func.func @struct_explode
// CHECK-SAME: (%{{.+}}: !smt.bv<2>) -> !smt.bv<1>
hw.module @struct_explode(in %s: !hw.struct<value: i1, unknown: i1>, out out: i1) {
  // CHECK-NEXT: %[[HI:.+]] = smt.bv.extract %{{.+}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
  // CHECK-NEXT: %[[LO:.+]] = smt.bv.extract %{{.+}} from 0 : (!smt.bv<2>) -> !smt.bv<1>
  // CHECK-NEXT: return %[[LO]] : !smt.bv<1>
  %value, %unknown = hw.struct_explode %s : !hw.struct<value: i1, unknown: i1>
  hw.output %unknown : i1
}
