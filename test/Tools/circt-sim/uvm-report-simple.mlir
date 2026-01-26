// RUN: circt-sim %s --max-time 1000000 2>&1 | FileCheck %s
// Simple test for UVM report function dispatching

// CHECK: UVM_INFO
// CHECK: Simulation

module {
  // Global string constant
  llvm.mlir.global private constant @msg_id("TEST\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_text("Hello UVM\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_file("test.sv\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_ctx("ctx\00") {addr_space = 0 : i32}

  // Declare external UVM report function
  llvm.func @__moore_uvm_report_info(!llvm.ptr, i64, !llvm.ptr, i64, i32, !llvm.ptr, i64, i32, !llvm.ptr, i64)

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %false = hw.constant false
    %true = hw.constant true

    // Signal to prevent canonicalization of process
    %sig = llhd.sig %false : i1

    llhd.process {
      cf.br ^start

    ^start:
      // Drive signal to prevent canonicalization
      llhd.drv %sig, %true after %eps : i1

      %id_ptr = llvm.mlir.addressof @msg_id : !llvm.ptr
      %msg_ptr = llvm.mlir.addressof @msg_text : !llvm.ptr
      %file_ptr = llvm.mlir.addressof @msg_file : !llvm.ptr
      %ctx_ptr = llvm.mlir.addressof @msg_ctx : !llvm.ptr

      %id_len = llvm.mlir.constant(4 : i64) : i64
      %msg_len = llvm.mlir.constant(9 : i64) : i64
      %file_len = llvm.mlir.constant(7 : i64) : i64
      %ctx_len = llvm.mlir.constant(3 : i64) : i64
      %verbosity = llvm.mlir.constant(200 : i32) : i32
      %line = llvm.mlir.constant(10 : i32) : i32

      llvm.call @__moore_uvm_report_info(%id_ptr, %id_len, %msg_ptr, %msg_len,
                                          %verbosity, %file_ptr, %file_len,
                                          %line, %ctx_ptr, %ctx_len) :
        (!llvm.ptr, i64, !llvm.ptr, i64, i32, !llvm.ptr, i64, i32, !llvm.ptr, i64) -> ()

      llhd.wait delay %t1, ^done

    ^done:
      llhd.halt
    }

    hw.output
  }
}
