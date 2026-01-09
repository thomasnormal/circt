add_circt_unittest(CIRCTDebugTests
  DebugTest.cpp
)

target_link_libraries(CIRCTDebugTests
  PRIVATE
  CIRCTDebug
  CIRCTSupport
  MLIRIR
  MLIRSupport
)
