add_circt_dialect(Coverage coverage)
add_circt_doc(CoverageOps Dialects/CoverageOps -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CoveragePasses.td)
mlir_tablegen(CoveragePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCoverageTransformsIncGen)
add_circt_doc(CoveragePasses CoveragePasses -gen-pass-doc)

add_dependencies(circt-headers MLIRCoverageIncGen)
