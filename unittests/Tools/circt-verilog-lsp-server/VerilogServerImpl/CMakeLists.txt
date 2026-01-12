add_circt_unittest(CIRCTVerilogLSPSemanticTokensTests
  SemanticTokensTest.cpp
)

target_link_libraries(CIRCTVerilogLSPSemanticTokensTests
  PRIVATE
  CIRCTVerilogLspServerImpl
)
