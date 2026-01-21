add_circt_unittest(CIRCTVerilogLSPSemanticTokensTests
  SemanticTokensTest.cpp
)

target_include_directories(CIRCTVerilogLSPSemanticTokensTests
  PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl
)

target_link_libraries(CIRCTVerilogLSPSemanticTokensTests
  PRIVATE
  CIRCTVerilogLspServerImpl
)
