// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Just test the type conversion
func.func @SimpleString(%str: !moore.string) -> !moore.string {
  return %str : !moore.string
}
