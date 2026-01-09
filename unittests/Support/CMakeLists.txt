add_circt_unittest(CIRCTSupportTests
  CoverageDatabaseTest.cpp
  FVIntTest.cpp
  JSONTest.cpp
  NPNClassTest.cpp
  PrettyPrinterTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
)
