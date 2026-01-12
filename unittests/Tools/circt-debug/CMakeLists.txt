add_circt_unittest(CIRCTDebugTests
  DebugTest.cpp
)

target_link_libraries(CIRCTDebugTests
  PRIVATE
  CIRCTDebugTool
  CIRCTSupport
  MLIRIR
  MLIRSupport
)
