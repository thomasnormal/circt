// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 CIRCT_SIM_PROFILE_MEMORY_SAMPLE_INTERVAL=1 circt-sim %s 2>&1 | FileCheck %s

// CHECK: [circt-sim] Memory peak: samples={{[1-9][0-9]*}} sample_interval_steps=1 peak_step={{[1-9][0-9]*}} peak_total_bytes={{[1-9][0-9]*}} global_bytes={{[1-9][0-9]*}} malloc_bytes={{[0-9]+}} native_bytes={{[0-9]+}} process_bytes={{[0-9]+}} dynamic_string_bytes={{[0-9]+}} config_db_bytes={{[0-9]+}} analysis_conn_edges={{[0-9]+}} seq_fifo_items={{[0-9]+}} largest_process={{[0-9]+}} largest_process_bytes={{[0-9]+}} largest_process_func={{[^ ]+}}
// CHECK: [circt-sim] Memory process top[0]: proc={{[0-9]+}} bytes={{[0-9]+}} name={{[^ ]+}} func={{[^ ]+}}

module {
  llvm.mlir.global internal @g0(0 : i64) : i64

  hw.module @top() {
    llhd.process {
      cf.br ^start
    ^start:
      %one = llvm.mlir.constant(1 : i64) : i64
      %ptr = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
      %v = llvm.mlir.constant(13 : i32) : i32
      llvm.store %v, %ptr : i32, !llvm.ptr
      llhd.halt
    }
    hw.output
  }
}
