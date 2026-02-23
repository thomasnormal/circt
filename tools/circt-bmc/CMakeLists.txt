set(LLVM_LINK_COMPONENTS Support)
set(CIRCT_BMC_IMPORT_VERILOG_DEPS)
if(TARGET CIRCTImportVerilog)
  list(APPEND CIRCT_BMC_IMPORT_VERILOG_DEPS CIRCTImportVerilog)
endif()

add_circt_tool(circt-bmc circt-bmc.cpp)
target_link_libraries(circt-bmc
  PRIVATE
  CIRCTBMCTransforms
  CIRCTComb
  CIRCTCombToSMT
  CIRCTEmitTransforms
  CIRCTHW
  CIRCTHWTransforms
  CIRCTHWToSMT
  ${CIRCT_BMC_IMPORT_VERILOG_DEPS}
  CIRCTLECTransforms
  CIRCTLLHDTransforms
  CIRCTLTL
  CIRCTLTLToCore
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTOMTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSim
  CIRCTSimTransforms
  CIRCTSMTToZ3LLVM
  CIRCTSVAToLTL
  CIRCTTransforms
  CIRCTSupport
  CIRCTVerif
  CIRCTVerifToSMT
  CIRCTVerifTransforms
  LLVMSupport
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSMT
  MLIRExportSMTLIB
  MLIRTargetLLVMIRExport
)

llvm_update_compile_flags(circt-bmc)
mlir_check_all_link_libraries(circt-bmc)
