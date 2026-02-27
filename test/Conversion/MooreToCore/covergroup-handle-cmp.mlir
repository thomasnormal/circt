// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

moore.covergroup.decl @cg {
}

func.func @covergroup_cmp_eq(%a: !moore.covergroup<@cg>, %b: !moore.covergroup<@cg>) -> !moore.i1 {
  %eq = moore.covergroup_handle_cmp eq %a, %b : !moore.covergroup<@cg> -> !moore.i1
  return %eq : !moore.i1
}

func.func @covergroup_cmp_ne(%a: !moore.covergroup<@cg>, %b: !moore.covergroup<@cg>) -> !moore.i1 {
  %ne = moore.covergroup_handle_cmp ne %a, %b : !moore.covergroup<@cg> -> !moore.i1
  return %ne : !moore.i1
}

// CHECK-LABEL: func.func @covergroup_cmp_eq
// CHECK-SAME: (%[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr) -> i1
// CHECK: %[[EQ:.*]] = llvm.icmp "eq" %[[A]], %[[B]] : !llvm.ptr
// CHECK: return %[[EQ]] : i1

// CHECK-LABEL: func.func @covergroup_cmp_ne
// CHECK-SAME: (%[[A2:.*]]: !llvm.ptr, %[[B2:.*]]: !llvm.ptr) -> i1
// CHECK: %[[NE:.*]] = llvm.icmp "ne" %[[A2]], %[[B2]] : !llvm.ptr
// CHECK: return %[[NE]] : i1
