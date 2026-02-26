set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  Support
)

add_circt_unittest(CIRCTSimCompileToolTests
  LowerTaggedIndirectCallsTest.cpp
  ${CIRCT_SOURCE_DIR}/tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp
)

target_include_directories(CIRCTSimCompileToolTests
  PRIVATE
  ${CIRCT_SOURCE_DIR}/tools/circt-sim-compile
)

target_link_libraries(CIRCTSimCompileToolTests
  PRIVATE
  LLVMCore
  LLVMAsmParser
  LLVMSupport
)
