include(SlangCompilerOptions)

set(LLVM_OPTIONAL_SOURCES
  LintConfig.cpp
  LintRules.cpp
)

add_circt_library(CIRCTLinting
  LintConfig.cpp
  LintRules.cpp

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  LINK_LIBS PRIVATE
  slang_slang
)
