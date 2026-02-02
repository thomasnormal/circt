add_circt_unittest(CIRCTMooreTests
  ConversionTest.cpp
  TypesTest.cpp
)

target_link_libraries(CIRCTMooreTests
  PRIVATE
  CIRCTMoore
  CIRCTMooreToCore
)
