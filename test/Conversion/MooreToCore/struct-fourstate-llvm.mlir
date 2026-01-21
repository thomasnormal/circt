// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test for unpacked structs with 4-state fields that require LLVM struct handling.
// When a struct has dynamic fields (strings, queues, etc.) AND 4-state logic fields,
// the struct converts to LLVM struct. The 4-state fields (hw.struct<value, unknown>)
// must be properly converted to LLVM struct for store/load/extract operations.

// CHECK-LABEL: hw.module @FourStateInLLVMStruct
moore.module @FourStateInLLVMStruct() {
  // Unpacked struct with string (forces LLVM struct) and 4-state logic field
  // CHECK-DAG: %[[CONST:.*]] = hw.aggregate_constant [-1 : i8, 0 : i8] : !hw.struct<value: i8, unknown: i8>
  // CHECK-DAG: %[[ALLOC:.*]] = llvm.alloca {{.*}} x !llvm.struct<(struct<(ptr, i64)>, struct<(i8, i8)>)>
  // CHECK: llvm.store {{.*}}, %[[ALLOC]]
  %data = moore.variable : !moore.ref<ustruct<{name: string, value: l8}>>

  moore.procedure initial {
    // Store 4-state value to struct field - converts hw.struct to llvm.struct
    // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOC]][0, 1]
    // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CONST]] : !hw.struct<value: i8, unknown: i8> to !llvm.struct<(i8, i8)>
    // CHECK: llvm.store %[[CAST]], %[[GEP]] : !llvm.struct<(i8, i8)>, !llvm.ptr
    %ref = moore.struct_extract_ref %data, "value" : !moore.ref<ustruct<{name: string, value: l8}>> -> !moore.ref<l8>
    %val = moore.constant -1 : l8
    moore.blocking_assign %ref, %val : l8

    // Read entire struct and extract 4-state field - converts llvm.struct back to hw.struct
    // CHECK: %[[LOADED:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr -> !llvm.struct<(struct<(ptr, i64)>, struct<(i8, i8)>)>
    // CHECK: %[[FIELD:.*]] = llvm.extractvalue %[[LOADED]][1] : !llvm.struct<(struct<(ptr, i64)>, struct<(i8, i8)>)>
    // CHECK: %[[HWSTRUCT:.*]] = builtin.unrealized_conversion_cast %[[FIELD]] : !llvm.struct<(i8, i8)> to !hw.struct<value: i8, unknown: i8>
    // CHECK: hw.struct_extract %[[HWSTRUCT]]["value"]
    %read = moore.read %data : !moore.ref<ustruct<{name: string, value: l8}>>
    %extracted = moore.struct_extract %read, "value" : ustruct<{name: string, value: l8}> -> l8
    %fmt = moore.fmt.int hex_lower %extracted, align right, pad zero : l8
    moore.builtin.display %fmt

    moore.return
  }

  moore.output
}

// CHECK-LABEL: hw.module @FourStateArrayInLLVMStruct
moore.module @FourStateArrayInLLVMStruct() {
  // Unpacked struct with string and 4-state field with X bits
  // CHECK-DAG: %[[XCONST:.*]] = hw.aggregate_constant [-8 : i4, 3 : i4] : !hw.struct<value: i4, unknown: i4>
  // CHECK-DAG: %[[ALLOC2:.*]] = llvm.alloca {{.*}} x !llvm.struct<(struct<(ptr, i64)>, struct<(i4, i4)>)>
  %data = moore.variable : !moore.ref<ustruct<{tag: string, flags: l4}>>

  moore.procedure initial {
    // Store 4-state value with X bits - unknown bits are preserved in the llvm.struct
    // CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[ALLOC2]][0, 1]
    // CHECK: %[[XCAST:.*]] = builtin.unrealized_conversion_cast %[[XCONST]] : !hw.struct<value: i4, unknown: i4> to !llvm.struct<(i4, i4)>
    // CHECK: llvm.store %[[XCAST]], %[[GEP2]] : !llvm.struct<(i4, i4)>, !llvm.ptr
    %ref = moore.struct_extract_ref %data, "flags" : !moore.ref<ustruct<{tag: string, flags: l4}>> -> !moore.ref<l4>
    %val = moore.constant b10XX : l4
    moore.blocking_assign %ref, %val : l4

    moore.return
  }

  moore.output
}
