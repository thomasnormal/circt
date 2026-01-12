set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-debug
  circt-debug.cpp

  DEPENDS
  CIRCTDebugTool
)

target_link_libraries(circt-debug
  PRIVATE
  CIRCTDebugTool
  CIRCTSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)
