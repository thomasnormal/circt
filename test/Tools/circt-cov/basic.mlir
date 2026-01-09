// RUN: circt-cov --help 2>&1 | FileCheck %s --check-prefix=HELP

// Basic test for circt-cov tool help message
// HELP: USAGE: circt-cov [options] <input files>
// HELP: --merge
// HELP: --report
// HELP: --diff
// HELP: --exclude
// HELP: --trend
// HELP: --convert
