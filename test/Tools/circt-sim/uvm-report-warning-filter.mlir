// RUN: circt-sim %s --max-time 1000000 2>&1 | FileCheck %s
// CHECK: UVM_WARNING
// CHECK: keep warning
// CHECK-NOT: filtered warning

module {
  llvm.mlir.global private constant @id_filtered("UVM/COMP/NAME\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @id_keep("KEEP\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_filtered("filtered warning\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_keep("keep warning\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_file("test.sv\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @msg_ctx("ctx\00") {addr_space = 0 : i32}

  llvm.func @__moore_uvm_report_warning(!llvm.ptr, i64, !llvm.ptr, i64, i32, !llvm.ptr, i64, i32, !llvm.ptr, i64)

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %false = hw.constant false
    %true = hw.constant true

    %sig = llhd.sig %false : i1

    llhd.process {
      cf.br ^start

    ^start:
      llhd.drv %sig, %true after %eps : i1

      %id_filtered_ptr = llvm.mlir.addressof @id_filtered : !llvm.ptr
      %id_keep_ptr = llvm.mlir.addressof @id_keep : !llvm.ptr
      %msg_filtered_ptr = llvm.mlir.addressof @msg_filtered : !llvm.ptr
      %msg_keep_ptr = llvm.mlir.addressof @msg_keep : !llvm.ptr
      %file_ptr = llvm.mlir.addressof @msg_file : !llvm.ptr
      %ctx_ptr = llvm.mlir.addressof @msg_ctx : !llvm.ptr

      %id_filtered_len = llvm.mlir.constant(13 : i64) : i64
      %id_keep_len = llvm.mlir.constant(4 : i64) : i64
      %msg_filtered_len = llvm.mlir.constant(16 : i64) : i64
      %msg_keep_len = llvm.mlir.constant(12 : i64) : i64
      %file_len = llvm.mlir.constant(7 : i64) : i64
      %ctx_len = llvm.mlir.constant(3 : i64) : i64
      %verbosity = llvm.mlir.constant(200 : i32) : i32
      %line = llvm.mlir.constant(42 : i32) : i32

      llvm.call @__moore_uvm_report_warning(%id_filtered_ptr, %id_filtered_len,
                                            %msg_filtered_ptr, %msg_filtered_len,
                                            %verbosity, %file_ptr, %file_len,
                                            %line, %ctx_ptr, %ctx_len) :
          (!llvm.ptr, i64, !llvm.ptr, i64, i32, !llvm.ptr, i64, i32, !llvm.ptr, i64) -> ()

      llvm.call @__moore_uvm_report_warning(%id_keep_ptr, %id_keep_len,
                                            %msg_keep_ptr, %msg_keep_len,
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
