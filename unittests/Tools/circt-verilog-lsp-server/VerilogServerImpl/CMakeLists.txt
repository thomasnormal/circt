add_circt_unittest(CIRCTVerilogLSPVerilogServerImplTests
  PARTIAL_SOURCES_INTENDED
  SemanticTokensTest.cpp
  SlangPositionConversionTest.cpp
)

target_include_directories(CIRCTVerilogLSPVerilogServerImplTests
  PRIVATE
  ${CIRCT_SOURCE_DIR}/lib/Tools/circt-verilog-lsp-server/VerilogServerImpl
)

target_link_libraries(CIRCTVerilogLSPVerilogServerImplTests
  PRIVATE
  CIRCTVerilogLspServerImpl
)
