add_circt_unittest(CIRCTSupportTests
  DiagnosticsTest.cpp
  FVIntTest.cpp
  JSONTest.cpp
  NPNClassTest.cpp
  PrettyPrinterTest.cpp
  ProjectConfigTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
  MLIRIR
)
