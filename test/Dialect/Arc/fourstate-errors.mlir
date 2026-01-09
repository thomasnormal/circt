// RUN: circt-opt --verify-diagnostics %s --split-input-file

//===----------------------------------------------------------------------===//
// FourState Type Errors
//===----------------------------------------------------------------------===//

// Test zero-width type
// expected-error @+1 {{4-state logic type must have a positive width}}
func.func @zero_width_type(%arg0: !arc.logic<0>) {
  return
}

// -----

//===----------------------------------------------------------------------===//
// Constant Errors
//===----------------------------------------------------------------------===//

// Test constant width mismatch
func.func @constant_width_mismatch() {
  // expected-error @+1 {{constant width 16 does not match result type width 8}}
  %c = arc.fourstate.constant 1234 : !arc.logic<8>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Conversion Errors
//===----------------------------------------------------------------------===//

// Test to_fourstate width mismatch
func.func @to_fourstate_width_mismatch(%arg0: i16) {
  // expected-error @+1 {{input width 16 does not match result width 8}}
  %result = arc.to_fourstate %arg0 : (i16) -> !arc.logic<8>
  return
}

// -----

// Test from_fourstate width mismatch
func.func @from_fourstate_width_mismatch(%arg0: !arc.logic<16>) {
  // expected-error @+1 {{input width 16 does not match result width 8}}
  %result = arc.from_fourstate %arg0 : (!arc.logic<16>) -> i8
  return
}

// -----

//===----------------------------------------------------------------------===//
// Concat Errors
//===----------------------------------------------------------------------===//

// Test concat width mismatch
func.func @concat_width_mismatch(%a: !arc.logic<4>, %b: !arc.logic<4>) {
  // expected-error @+1 {{concatenated input widths 8 do not match result width 16}}
  %result = arc.fourstate.concat %a, %b : !arc.logic<4>, !arc.logic<4> -> !arc.logic<16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Extract Errors
//===----------------------------------------------------------------------===//

// Test extract out of bounds
func.func @extract_out_of_bounds(%input: !arc.logic<8>) {
  // expected-error @+1 {{extract range [4, 12) exceeds input width 8}}
  %result = arc.fourstate.extract %input from 4 : !arc.logic<8> -> !arc.logic<8>
  return
}

// -----

// Test extract range exceeds width
func.func @extract_exceeds_width(%input: !arc.logic<8>) {
  // expected-error @+1 {{extract range [6, 10) exceeds input width 8}}
  %result = arc.fourstate.extract %input from 6 : !arc.logic<8> -> !arc.logic<4>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Replicate Errors
//===----------------------------------------------------------------------===//

// Test replicate non-multiple width
func.func @replicate_non_multiple(%input: !arc.logic<3>) {
  // expected-error @+1 {{result width 8 must be a multiple of input width 3}}
  %result = arc.fourstate.replicate %input : (!arc.logic<3>) -> !arc.logic<8>
  return
}

// -----

// Test replicate shrinking
func.func @replicate_shrinking(%input: !arc.logic<8>) {
  // expected-error @+1 {{result width must be >= input width}}
  %result = arc.fourstate.replicate %input : (!arc.logic<8>) -> !arc.logic<4>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Mux Errors
//===----------------------------------------------------------------------===//

// Test mux condition not 1-bit
func.func @mux_condition_width(%cond: !arc.logic<2>, %a: !arc.logic<8>, %b: !arc.logic<8>) {
  // expected-error @+1 {{condition must be 1-bit 4-state, got width 2}}
  %result = arc.fourstate.mux %cond, %a, %b : !arc.logic<8>
  return
}
