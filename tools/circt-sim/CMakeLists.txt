set(LLVM_LINK_COMPONENTS Support native)

set(libs
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTEmit
  CIRCTHW
  CIRCTHWTransforms
  CIRCTLLHD
  CIRCTMoore
  CIRCTOM
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSim
  CIRCTSimTransforms
  CIRCTSV
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRTransforms
)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  list(APPEND libs
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
  )
endif()

add_circt_tool(circt-sim
  PARTIAL_SOURCES_INTENDED
  AOTProcessCompiler.cpp
  CompiledModuleLoader.cpp
  circt-sim.cpp
  LLHDProcessInterpreter.cpp
  LLHDProcessInterpreterBytecode.cpp
  LLHDProcessInterpreterCallIndirect.cpp
  LLHDProcessInterpreterDrive.cpp
  LLHDProcessInterpreterGlobals.cpp
  LLHDProcessInterpreterMemory.cpp
  LLHDProcessInterpreterModuleLevelInit.cpp
  LLHDProcessInterpreterNativeThunkExec.cpp
  LLHDProcessInterpreterNativeThunkPolicy.cpp
  LLHDProcessInterpreterTrace.cpp
  LLHDProcessInterpreterUvm.cpp
  LLHDProcessInterpreterWaitCondition.cpp
  UcontextProcess.cpp
  UVMFastPaths.cpp
  DEPENDS ${libs}
)
target_link_libraries(circt-sim
  PRIVATE
  ${libs}
  MooreRuntime
)

# For wasm/node builds, map POSIX paths directly to the host filesystem.
# This keeps CLI behavior (e.g. --vcd /tmp/test.vcd) consistent with native.
option(CIRCT_SIM_WASM_ENABLE_NODERAWFS
  "Enable Emscripten Node.js raw filesystem bridge (-sNODERAWFS=1) for circt-sim"
  ON
)
if(EMSCRIPTEN AND CIRCT_SIM_WASM_ENABLE_NODERAWFS)
  target_link_options(circt-sim PRIVATE "SHELL:-sNODERAWFS=1")
endif()

option(CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH
  "Enable Emscripten heap growth (-sALLOW_MEMORY_GROWTH=1) for circt-sim"
  ON
)
set(CIRCT_SIM_WASM_STACK_SIZE "33554432" CACHE STRING
  "Emscripten wasm stack size in bytes for circt-sim (-sSTACK_SIZE=...)"
)

if(EMSCRIPTEN)
  if(NOT CIRCT_SIM_WASM_STACK_SIZE MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "CIRCT_SIM_WASM_STACK_SIZE must be a numeric integer, got '${CIRCT_SIM_WASM_STACK_SIZE}'"
    )
  endif()
  if(CIRCT_SIM_WASM_STACK_SIZE LESS 1)
    message(FATAL_ERROR
      "CIRCT_SIM_WASM_STACK_SIZE must be >= 1, got '${CIRCT_SIM_WASM_STACK_SIZE}'"
    )
  endif()

  if(CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH)
    target_link_options(circt-sim PRIVATE "SHELL:-sALLOW_MEMORY_GROWTH=1")
  endif()
  target_link_options(circt-sim PRIVATE
    "SHELL:-sSTACK_SIZE=${CIRCT_SIM_WASM_STACK_SIZE}"
  )

  target_link_options(circt-sim PRIVATE
    "SHELL:--js-library=${CMAKE_CURRENT_SOURCE_DIR}/circt-sim-vpi-wasm.js"
    "SHELL:-sASYNCIFY=1"
    "SHELL:-sASYNCIFY_IMPORTS=['circt_vpi_wasm_yield']"
  )

  set(circt_sim_wasm_vpi_exports
    vpi_startup_register
    vpi_register_cb
    vpi_remove_cb
    vpi_handle
    vpi_handle_by_name
    vpi_handle_by_index
    vpi_iterate
    vpi_scan
    vpi_free_object
    vpi_get
    vpi_get_str
    vpi_get_time
    vpi_get_value
    vpi_put_value
    vpi_chk_error
    vpi_get_vlog_info
    vpi_control
    vpi_release_handle
  )
  foreach(sym IN LISTS circt_sim_wasm_vpi_exports)
    target_link_options(circt-sim PRIVATE "SHELL:-Wl,--export=${sym}")
  endforeach()
endif()

llvm_update_compile_flags(circt-sim)
mlir_check_all_link_libraries(circt-sim)

# Export symbols from the executable so that dlopen'd VPI libraries (e.g., cocotb)
# can resolve vpi_* C API functions. Without this, the VPI functions defined in
# VPIRuntime.cpp and MooreRuntime.cpp are invisible to loaded shared libraries.
set_target_properties(circt-sim PROPERTIES ENABLE_EXPORTS 1)
