add_circt_unittest(CIRCTLintingTests
  LintConfigTest.cpp
)

target_link_libraries(CIRCTLintingTests
  PRIVATE
  CIRCTLinting
)
