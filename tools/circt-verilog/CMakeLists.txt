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

llvm_update_compile_flags(circt-verilog)
mlir_check_all_link_libraries(circt-verilog)
