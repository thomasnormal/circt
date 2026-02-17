// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s

// CHECK: [circt-sim] Memory state: global_blocks={{[1-9][0-9]*}} global_bytes={{[1-9][0-9]*}} malloc_blocks={{[0-9]+}} malloc_bytes={{[0-9]+}} native_blocks={{[0-9]+}} native_bytes={{[0-9]+}} process_blocks={{[0-9]+}} process_bytes={{[0-9]+}} dynamic_strings={{[0-9]+}} dynamic_string_bytes={{[0-9]+}} config_db_entries={{[0-9]+}} config_db_bytes={{[0-9]+}} analysis_conn_ports={{[0-9]+}} analysis_conn_edges={{[0-9]+}} seq_fifo_maps={{[0-9]+}} seq_fifo_items={{[0-9]+}} largest_process={{[0-9]+}} largest_process_bytes={{[0-9]+}}
// CHECK: [circt-sim] Memory process top[0]: proc={{[0-9]+}} bytes={{[0-9]+}} name={{[^ ]+}} func={{[^ ]+}}

module {
  llvm.mlir.global internal @g0(0 : i64) : i64

  hw.module @top() {
    llhd.process {
      cf.br ^start
    ^start:
      %one = llvm.mlir.constant(1 : i64) : i64
      %ptr = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
      %v = llvm.mlir.constant(7 : i32) : i32
      llvm.store %v, %ptr : i32, !llvm.ptr
      llhd.halt
    }
    hw.output
  }
}
