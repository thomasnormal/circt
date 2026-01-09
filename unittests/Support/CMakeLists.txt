add_circt_unittest(CIRCTSupportTests
  CoverageDatabaseTest.cpp
  CoverageReportGeneratorTest.cpp
  DiagnosticsTest.cpp
  FVIntTest.cpp
  JSONTest.cpp
  NPNClassTest.cpp
  PrettyPrinterTest.cpp
  ProjectConfigTest.cpp
  TestReportingTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
  MLIRIR
)
