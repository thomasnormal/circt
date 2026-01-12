set(LLVM_OPTIONAL_SOURCES
  LintConfig.cpp
  LintRules.cpp
)

add_circt_library(CIRCTLinting
  LintConfig.cpp
  LintRules.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  LINK_LIBS PRIVATE
  slang::slang
)

target_include_directories(CIRCTLinting
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)
