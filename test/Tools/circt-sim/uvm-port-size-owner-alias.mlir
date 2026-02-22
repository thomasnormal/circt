// RUN: circt-sim %s 2>&1 | FileCheck %s
//
// Regression: uvm_port_base::size must resolve owner-alias pointers.
// A connect() recorded on one subobject pointer should be visible when size()
// is called on another pointer into the same owning object.
//
// CHECK: alias_size_ok
// CHECK-NOT: alias_size_bad_func_call
// CHECK-NOT: alias_size_bad_call_indirect

module {
  llvm.mlir.global internal @port_owner(#llvm.zero) : !llvm.array<32 x i8>
  llvm.mlir.global internal @provider_obj(0 : i64) : i64

  llvm.mlir.global internal @size_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_port_base::size"]]
  } : !llvm.array<1 x ptr>

  func.func @"uvm_pkg::uvm_port_base::connect"(%self: !llvm.ptr, %provider: !llvm.ptr) {
    return
  }

  // This body should be bypassed by native interception.
  func.func @"uvm_pkg::uvm_port_base::size"(%self: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }

  hw.module @top() {
    %msgOk = sim.fmt.literal "alias_size_ok\0A"
    %msgBadFC = sim.fmt.literal "alias_size_bad_func_call\0A"
    %msgBadCI = sim.fmt.literal "alias_size_bad_call_indirect\0A"

    llhd.process {
      %base = llvm.mlir.addressof @port_owner : !llvm.ptr
      %provider = llvm.mlir.addressof @provider_obj : !llvm.ptr

      // Two aliases into the same owning allocation.
      %portConnect = llvm.getelementptr %base[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %portSize = llvm.getelementptr %base[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>

      func.call @"uvm_pkg::uvm_port_base::connect"(%portConnect, %provider) : (!llvm.ptr, !llvm.ptr) -> ()

      // func.call path
      %sizeFC = func.call @"uvm_pkg::uvm_port_base::size"(%portSize) : (!llvm.ptr) -> i32
      %c1 = arith.constant 1 : i32
      %okFC = arith.cmpi eq, %sizeFC, %c1 : i32
      cf.cond_br %okFC, ^check_ci, ^bad_fc

    ^bad_fc:
      sim.proc.print %msgBadFC
      llhd.halt

    ^check_ci:
      // func.call_indirect path
      %vt = llvm.mlir.addressof @size_vtable : !llvm.ptr
      %fnAddr = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fnPtr = llvm.load %fnAddr : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fnPtr : !llvm.ptr to (!llvm.ptr) -> i32
      %sizeCI = func.call_indirect %fn(%portSize) : (!llvm.ptr) -> i32
      %okCI = arith.cmpi eq, %sizeCI, %c1 : i32
      cf.cond_br %okCI, ^good, ^bad_ci

    ^bad_ci:
      sim.proc.print %msgBadCI
      llhd.halt

    ^good:
      sim.proc.print %msgOk
      llhd.halt
    }

    hw.output
  }
}
