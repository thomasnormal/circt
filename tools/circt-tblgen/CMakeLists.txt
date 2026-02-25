set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(circt-tblgen CIRCT
  circt-tblgen.cpp
  FIRRTLAnnotationsGen.cpp
  FIRRTLIntrinsicsGen.cpp
)

target_link_libraries(circt-tblgen
  PRIVATE
  MLIRTableGen
  MLIRTblgenLib
)

# For wasm/node builds, map POSIX paths directly to the host filesystem so
# host-path .td inputs behave like native command-line usage.
option(CIRCT_TBLGEN_WASM_ENABLE_NODERAWFS
  "Enable Emscripten Node.js raw filesystem bridge (-sNODERAWFS=1) for circt-tblgen"
  ON
)
if(EMSCRIPTEN AND CIRCT_TBLGEN_WASM_ENABLE_NODERAWFS)
  target_link_options(circt-tblgen PRIVATE "SHELL:-sNODERAWFS=1")
endif()

llvm_update_compile_flags(circt-tblgen)
mlir_check_all_link_libraries(circt-tblgen)
