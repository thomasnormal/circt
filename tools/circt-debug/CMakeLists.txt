set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-debug
  circt-debug.cpp

  DEPENDS
  CIRCTDebug
)

target_link_libraries(circt-debug
  PRIVATE
  CIRCTDebug
  CIRCTSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)
