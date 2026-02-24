set(libs
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTImportVerilog
  CIRCTLLHD
  CIRCTLLHDTransforms
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTMooreTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSim
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRArithDialect
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
)

add_circt_tool(circt-verilog circt-verilog.cpp DEPENDS ${libs})
target_link_libraries(circt-verilog PRIVATE ${libs})

# For wasm/node builds, map POSIX paths directly to the host filesystem.
# This keeps `--uvm-path` and host input file behavior consistent with native.
option(CIRCT_VERILOG_WASM_ENABLE_NODERAWFS
  "Enable Emscripten Node.js raw filesystem bridge (-sNODERAWFS=1) for circt-verilog"
  ON
)
if(EMSCRIPTEN AND CIRCT_VERILOG_WASM_ENABLE_NODERAWFS)
  target_link_options(circt-verilog PRIVATE "SHELL:-sNODERAWFS=1")
endif()

option(CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH
  "Allow Emscripten wasm heap growth (-sALLOW_MEMORY_GROWTH=1) for circt-verilog"
  ON
)
if(EMSCRIPTEN AND CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH)
  target_link_options(circt-verilog PRIVATE "SHELL:-sALLOW_MEMORY_GROWTH=1")
endif()

set(CIRCT_VERILOG_WASM_STACK_SIZE "33554432" CACHE STRING
  "Emscripten wasm stack size in bytes (-sSTACK_SIZE=...) for circt-verilog"
)
if(EMSCRIPTEN)
  if(NOT CIRCT_VERILOG_WASM_STACK_SIZE MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "CIRCT_VERILOG_WASM_STACK_SIZE must be a positive integer in bytes (got: "
      "${CIRCT_VERILOG_WASM_STACK_SIZE})")
  endif()
  if(CIRCT_VERILOG_WASM_STACK_SIZE LESS 1)
    message(FATAL_ERROR
      "CIRCT_VERILOG_WASM_STACK_SIZE must be >= 1 (got: "
      "${CIRCT_VERILOG_WASM_STACK_SIZE})")
  endif()
  target_link_options(circt-verilog PRIVATE
    "SHELL:-sSTACK_SIZE=${CIRCT_VERILOG_WASM_STACK_SIZE}")
endif()

llvm_update_compile_flags(circt-verilog)
mlir_check_all_link_libraries(circt-verilog)
