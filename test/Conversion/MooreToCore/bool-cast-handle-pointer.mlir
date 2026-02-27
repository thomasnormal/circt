// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

moore.class.classdecl @C {
}

func.func @bool_cast_class_handle(%x: !moore.class<@C>) -> !moore.i1 {
  %b = moore.bool_cast %x : !moore.class<@C> -> !moore.i1
  return %b : !moore.i1
}

// CHECK-LABEL: func.func @bool_cast_class_handle
// CHECK-SAME: (%[[X:.*]]: !llvm.ptr) -> i1
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[CMP:.*]] = llvm.icmp "ne" %[[X]], %[[NULL]] : !llvm.ptr
// CHECK: return %[[CMP]] : i1
