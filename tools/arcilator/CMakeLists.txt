if(ARCILATOR_JIT_ENABLED)
  add_compile_definitions(ARCILATOR_ENABLE_JIT)
  set(ARCILATOR_JIT_LLVM_COMPONENTS native)
  set(ARCILATOR_JIT_DEPS
    CIRCTArcJITRuntime
    LLVMOrcJIT
    MLIRExecutionEngine
  )
endif()

set(LLVM_LINK_COMPONENTS Support ${ARCILATOR_JIT_LLVM_COMPONENTS})

set(libs
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTArcilator
  CIRCTCombToArith
  CIRCTCombToLLVM
  CIRCTConvertToArcs
  CIRCTEmitTransforms
  CIRCTHWToLLVM
  CIRCTExportArc
  CIRCTHWTransforms
  CIRCTLLHD
  CIRCTLLHDTransforms
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSimTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRFuncToLLVM
  MLIRIndexDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
)

add_circt_tool(arcilator arcilator.cpp BehavioralLowering.cpp DEPENDS ${libs})
target_link_libraries(arcilator
  PRIVATE
  ${libs}
  ${ARCILATOR_JIT_DEPS}
)

llvm_update_compile_flags(arcilator)
mlir_check_all_link_libraries(arcilator)

configure_file(arcilator-header-cpp.py
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)
add_custom_target(arcilator-header-cpp SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)

configure_file(arcilator-runtime.h
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)
add_custom_target(arcilator-runtime-header SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)
