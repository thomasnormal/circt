// RUN: circt-verilog --help | FileCheck %s

// CHECK: OVERVIEW: Verilog and SystemVerilog frontend
// CHECK-DAG: --language-version
// CHECK-DAG: --max-parse-depth
// CHECK-DAG: --max-lexer-errors
// CHECK-DAG: --num-threads
// CHECK-DAG: --max-instance-depth
// CHECK-DAG: --max-generate-steps
// CHECK-DAG: --max-constexpr-depth
// CHECK-DAG: --max-constexpr-steps
// CHECK-DAG: --max-constexpr-backtrace
// CHECK-DAG: --max-instance-array
