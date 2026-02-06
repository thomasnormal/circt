//===- SVAOps.td - SVA dialect operations -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the SVA (SystemVerilog Assertions)
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SVA_SVAOPS_TD
#define CIRCT_DIALECT_SVA_SVAOPS_TD

include "circt/Dialect/SVA/SVADialect.td"
include "circt/Dialect/SVA/SVATypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SVAOp<string mnemonic, list<Trait> traits = []> :
  Op<SVADialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sequence Operations
//===----------------------------------------------------------------------===//

def SequenceDelayOp : SVAOp<"seq.delay", [Pure]> {
  let summary = "Delay a sequence by a number of cycles (##n, ##[m:n])";
  let description = [{
    Delays the input sequence by a specified number of cycles. This corresponds
    to the SVA `##n` (exact delay) and `##[m:n]` (range delay) operators.

    The `delay` attribute specifies the minimum number of cycles to delay.
    The optional `length` attribute specifies the additional cycles in the
    range (so `##[m:n]` has delay=m and length=n-m).

    Examples:
    ```mlir
    // ##2 seq - delay by exactly 2 cycles
    %0 = sva.seq.delay %seq, 2 : !sva.sequence

    // ##[1:3] seq - delay by 1 to 3 cycles
    %1 = sva.seq.delay %seq, 1, 2 : !sva.sequence

    // ##[0:$] seq - delay by 0 or more cycles (unbounded)
    %2 = sva.seq.delay %seq, 0 : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input,
    I64Attr:$delay,
    OptionalAttr<I64Attr>:$length
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input `,` $delay (`,` $length^)? attr-dict `:` type($input)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "int64_t":$delay), [{
      build($_builder, $_state,
            $_builder.getType<SequenceType>(),
            input, $_builder.getI64IntegerAttr(delay),
            ::mlir::IntegerAttr());
    }]>,
    OpBuilder<(ins "::mlir::Value":$input, "int64_t":$delay, "int64_t":$length), [{
      build($_builder, $_state,
            $_builder.getType<SequenceType>(),
            input, $_builder.getI64IntegerAttr(delay),
            $_builder.getI64IntegerAttr(length));
    }]>
  ];
}

def SequenceRepeatOp : SVAOp<"seq.repeat", [Pure]> {
  let summary = "Repeat a sequence (consecutive repetition: [*n], [*m:n])";
  let description = [{
    Repeats the input sequence a specified number of times consecutively. This
    corresponds to the SVA `[*n]` (exact repetition) and `[*m:n]` (range
    repetition) operators.

    The `base` attribute specifies the minimum number of repetitions.
    The optional `more` attribute specifies the additional repetitions in the
    range (so `[*m:n]` has base=m and more=n-m).

    Examples:
    ```mlir
    // seq[*3] - repeat exactly 3 times
    %0 = sva.seq.repeat %seq, 3 : !sva.sequence

    // seq[*2:4] - repeat 2 to 4 times
    %1 = sva.seq.repeat %seq, 2, 2 : !sva.sequence

    // seq[*1:$] - repeat 1 or more times (unbounded)
    %2 = sva.seq.repeat %seq, 1 : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input,
    I64Attr:$base,
    OptionalAttr<I64Attr>:$more
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input `,` $base (`,` $more^)? attr-dict `:` type($input)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "int64_t":$base), [{
      build($_builder, $_state,
            $_builder.getType<SequenceType>(),
            input, $_builder.getI64IntegerAttr(base),
            ::mlir::IntegerAttr());
    }]>,
    OpBuilder<(ins "::mlir::Value":$input, "int64_t":$base, "int64_t":$more), [{
      build($_builder, $_state,
            $_builder.getType<SequenceType>(),
            input, $_builder.getI64IntegerAttr(base),
            $_builder.getI64IntegerAttr(more));
    }]>
  ];
}

def SequenceGotoRepeatOp : SVAOp<"seq.goto_repeat", [Pure]> {
  let summary = "Non-consecutive goto repetition ([->n], [->m:n])";
  let description = [{
    Non-consecutive goto repetition of the input sequence. The sequence must
    match the specified number of times, but not necessarily in consecutive
    cycles. The match ends when the final repetition completes.

    This corresponds to the SVA `[->n]`, `[->m:n]`, and `[->m:$]` operators.

    Example:
    ```mlir
    // seq[->2:3] - match 2 to 3 times non-consecutively
    %0 = sva.seq.goto_repeat %seq, 2, 1 : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input,
    I64Attr:$base,
    OptionalAttr<I64Attr>:$more
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input `,` $base (`,` $more^)? attr-dict `:` type($input)
  }];
}

def SequenceNonConsecutiveRepeatOp : SVAOp<"seq.non_consecutive_repeat", [Pure]> {
  let summary = "Non-consecutive repetition ([=n], [=m:n])";
  let description = [{
    Non-consecutive repetition of the input sequence. The sequence must match
    the specified number of times, but not necessarily in consecutive cycles.
    Unlike goto repetition, the match can continue after the final repetition.

    This corresponds to the SVA `[=n]`, `[=m:n]`, and `[=m:$]` operators.

    Example:
    ```mlir
    // seq[=2:3] - match 2 to 3 times non-consecutively
    %0 = sva.seq.non_consecutive_repeat %seq, 2, 1 : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input,
    I64Attr:$base,
    OptionalAttr<I64Attr>:$more
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input `,` $base (`,` $more^)? attr-dict `:` type($input)
  }];
}

def SequenceConcatOp : SVAOp<"seq.concat", [Pure]> {
  let summary = "Concatenate sequences";
  let description = [{
    Concatenates multiple sequences together. The end time of each sequence
    coincides with the start time of the next sequence (no implicit delay).

    To express "A followed by B in the next cycle" (A ##1 B), use:
    ```mlir
    %delayed_b = sva.seq.delay %b, 1 : !sva.sequence
    %result = sva.seq.concat %a, %delayed_b : !sva.sequence, !sva.sequence
    ```

    Example:
    ```mlir
    %0 = sva.seq.concat %seq1, %seq2, %seq3 : !sva.sequence, !sva.sequence, !sva.sequence
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnySequenceType>:$inputs
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def SequenceOrOp : SVAOp<"seq.or", [Pure, Commutative]> {
  let summary = "Disjunction of sequences (or)";
  let description = [{
    Creates a disjunction of sequences. The resulting sequence matches if any
    of the input sequences matches.

    This corresponds to the SVA `or` operator for sequences.

    Example:
    ```mlir
    // (seq1) or (seq2) or (seq3)
    %0 = sva.seq.or %seq1, %seq2, %seq3 : !sva.sequence, !sva.sequence, !sva.sequence
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnySequenceType>:$inputs
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def SequenceAndOp : SVAOp<"seq.and", [Pure, Commutative]> {
  let summary = "Conjunction of sequences (and)";
  let description = [{
    Creates a conjunction of sequences. The resulting sequence matches if all
    of the input sequences match with the same start time.

    This corresponds to the SVA `and` operator for sequences.

    Example:
    ```mlir
    // (seq1) and (seq2)
    %0 = sva.seq.and %seq1, %seq2 : !sva.sequence, !sva.sequence
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnySequenceType>:$inputs
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def SequenceIntersectOp : SVAOp<"seq.intersect", [Pure, Commutative]> {
  let summary = "Intersection of sequences (intersect)";
  let description = [{
    Creates an intersection of sequences. The resulting sequence matches if all
    of the input sequences match with the same start and end times.

    This corresponds to the SVA `intersect` operator for sequences.

    Example:
    ```mlir
    // (seq1) intersect (seq2)
    %0 = sva.seq.intersect %seq1, %seq2 : !sva.sequence, !sva.sequence
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnySequenceType>:$inputs
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def SequenceFirstMatchOp : SVAOp<"seq.first_match", [Pure]> {
  let summary = "First match of a sequence";
  let description = [{
    Matches only the first (earliest) successful match of the input sequence.
    This is useful for sequences with variable-length matches.

    This corresponds to the SVA `first_match` operator.

    Example:
    ```mlir
    %0 = sva.seq.first_match %seq : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def SequenceWithinOp : SVAOp<"seq.within", [Pure]> {
  let summary = "Sequence containment (within)";
  let description = [{
    Checks if the first sequence completes within the duration of the second
    sequence. The start of the first sequence can be delayed.

    This corresponds to the SVA `within` operator.

    Example:
    ```mlir
    // seq1 within seq2
    %0 = sva.seq.within %seq1, %seq2 : !sva.sequence, !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$inner,
    SVAAnySequenceType:$outer
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $inner `,` $outer attr-dict `:` type($inner) `,` type($outer)
  }];
}

def SequenceThroughoutOp : SVAOp<"seq.throughout", [Pure]> {
  let summary = "Signal throughout sequence";
  let description = [{
    Checks if a signal holds true throughout the duration of a sequence.

    This corresponds to the SVA `throughout` operator.

    Example:
    ```mlir
    // enable throughout (req ##1 gnt)
    %0 = sva.seq.throughout %enable, %seq : i1, !sva.sequence
    ```
  }];

  let arguments = (ins
    I1:$signal,
    SVAAnySequenceType:$sequence
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $signal `,` $sequence attr-dict `:` type($signal) `,` type($sequence)
  }];
}

//===----------------------------------------------------------------------===//
// Clocking
//===----------------------------------------------------------------------===//

// Clock edge enum
def SVAPosEdge : I32EnumAttrCase<"Pos", 0, "posedge">;
def SVANegEdge : I32EnumAttrCase<"Neg", 1, "negedge">;
def SVABothEdges : I32EnumAttrCase<"Both", 2, "edge">;

def SVAClockEdgeAttr : I32EnumAttr<"ClockEdge", "clock edge",
                                    [SVAPosEdge, SVANegEdge, SVABothEdges]> {
  let cppNamespace = "circt::sva";
}

def SequenceClockOp : SVAOp<"seq.clock", [Pure]> {
  let summary = "Clock a sequence";
  let description = [{
    Associates a clock with a sequence. All cycle delays in the input sequence
    refer to ticks of this clock.

    This corresponds to the SVA `@(edge clock)` operator.

    Example:
    ```mlir
    // @(posedge clk) (req ##1 gnt)
    %0 = sva.seq.clock %seq, posedge %clk : !sva.sequence
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$input,
    SVAClockEdgeAttr:$edge,
    I1:$clock
  );

  let results = (outs SVASequenceType:$result);

  let assemblyFormat = [{
    $input `,` $edge $clock attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Property Operations
//===----------------------------------------------------------------------===//

def PropertyNotOp : SVAOp<"prop.not", [Pure]> {
  let summary = "Negation of a property (not)";
  let description = [{
    Negates a property. The resulting property holds if and only if the input
    property does not hold.

    This corresponds to the SVA `not` operator.

    Example:
    ```mlir
    %0 = sva.prop.not %prop : !sva.property
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def PropertyAndOp : SVAOp<"prop.and", [Pure, Commutative]> {
  let summary = "Conjunction of properties (and)";
  let description = [{
    Creates a conjunction of properties. The resulting property holds if all
    input properties hold.

    This corresponds to the SVA `and` operator for properties.

    Example:
    ```mlir
    %0 = sva.prop.and %prop1, %prop2 : !sva.property, !sva.property
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnyPropertyType>:$inputs
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def PropertyOrOp : SVAOp<"prop.or", [Pure, Commutative]> {
  let summary = "Disjunction of properties (or)";
  let description = [{
    Creates a disjunction of properties. The resulting property holds if any
    input property holds.

    This corresponds to the SVA `or` operator for properties.

    Example:
    ```mlir
    %0 = sva.prop.or %prop1, %prop2 : !sva.property, !sva.property
    ```
  }];

  let arguments = (ins
    Variadic<SVAAnyPropertyType>:$inputs
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let hasVerifier = 1;
}

def PropertyImplicationOp : SVAOp<"prop.implication", [Pure]> {
  let summary = "Property implication (|->, |=>)";
  let description = [{
    Creates an implication property. If the antecedent sequence matches, the
    consequent property must hold.

    The `overlapping` attribute determines the implication type:
    - `overlapping = true`: Overlapping implication (`|->`). The consequent
      is checked starting at the same cycle where the antecedent ends.
    - `overlapping = false`: Non-overlapping implication (`|=>`). The consequent
      is checked starting one cycle after the antecedent ends.

    Examples:
    ```mlir
    // req |-> gnt (overlapping)
    %0 = sva.prop.implication %req, %gnt overlapping : i1, i1

    // req |=> gnt (non-overlapping)
    %1 = sva.prop.implication %req, %gnt : i1, i1
    ```
  }];

  let arguments = (ins
    SVAAnySequenceType:$antecedent,
    SVAAnyPropertyType:$consequent,
    UnitAttr:$overlapping
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $antecedent `,` $consequent (`overlapping` $overlapping^)? attr-dict `:` type($antecedent) `,` type($consequent)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$antecedent, "::mlir::Value":$consequent, "bool":$overlapping), [{
      build($_builder, $_state,
            $_builder.getType<PropertyType>(),
            antecedent, consequent,
            overlapping ? $_builder.getUnitAttr() : ::mlir::UnitAttr());
    }]>
  ];
}

def PropertyIfElseOp : SVAOp<"prop.if", [Pure]> {
  let summary = "Conditional property (if-else)";
  let description = [{
    Creates a conditional property. If the condition is true, the then-property
    must hold; otherwise the else-property (if provided) must hold.

    This corresponds to the SVA `if (cond) prop1 else prop2` operator.

    Example:
    ```mlir
    // if (mode) prop1 else prop2
    %0 = sva.prop.if %mode, %prop1, %prop2 : i1, !sva.property, !sva.property
    ```
  }];

  let arguments = (ins
    I1:$condition,
    SVAAnyPropertyType:$thenProperty,
    Optional<SVAAnyPropertyType>:$elseProperty
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $condition `,` $thenProperty (`,` $elseProperty^)? attr-dict `:` type($condition) `,` type($thenProperty) (`,` type($elseProperty)^)?
  }];
}

def PropertyEventuallyOp : SVAOp<"prop.eventually", [Pure]> {
  let summary = "Eventual property (s_eventually)";
  let description = [{
    The property must hold at some point in the future. This is a strong
    operator that requires the property to eventually become true.

    This corresponds to the SVA `s_eventually` operator.

    Example:
    ```mlir
    // s_eventually done
    %0 = sva.prop.eventually %done : i1
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def PropertyAlwaysOp : SVAOp<"prop.always", [Pure]> {
  let summary = "Always property (always)";
  let description = [{
    The property must hold at all times from the current cycle onwards.

    This corresponds to the SVA `always` operator.

    Example:
    ```mlir
    // always (req |-> gnt)
    %0 = sva.prop.always %prop : !sva.property
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def PropertyUntilOp : SVAOp<"prop.until", [Pure]> {
  let summary = "Until property (until, s_until)";
  let description = [{
    The first property must hold until the second property holds. The `strong`
    attribute determines whether the second property must eventually hold
    (strong until) or may never hold (weak until).

    This corresponds to the SVA `until` (weak) and `s_until` (strong) operators.

    Example:
    ```mlir
    // req s_until gnt
    %0 = sva.prop.until %req, %gnt strong : i1, i1
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input,
    SVAAnyPropertyType:$condition,
    UnitAttr:$strong
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input `,` $condition (`strong` $strong^)? attr-dict `:` type($input) `,` type($condition)
  }];
}

def PropertyNextTimeOp : SVAOp<"prop.nexttime", [Pure]> {
  let summary = "Next time property (nexttime)";
  let description = [{
    The property must hold at the next clock tick. An optional count specifies
    how many clock ticks to look ahead.

    This corresponds to the SVA `nexttime` and `nexttime[n]` operators.

    Example:
    ```mlir
    // nexttime[2] prop
    %0 = sva.prop.nexttime %prop, 2 : !sva.property
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input,
    OptionalAttr<I64Attr>:$count
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input (`,` $count^)? attr-dict `:` type($input)
  }];
}

def PropertyClockOp : SVAOp<"prop.clock", [Pure]> {
  let summary = "Clock a property";
  let description = [{
    Associates a clock with a property. All temporal operators in the property
    refer to ticks of this clock.

    This corresponds to the SVA `@(edge clock)` operator for properties.

    Example:
    ```mlir
    // @(posedge clk) property
    %0 = sva.prop.clock %prop, posedge %clk : !sva.property
    ```
  }];

  let arguments = (ins
    SVAAnyPropertyType:$input,
    SVAClockEdgeAttr:$edge,
    I1:$clock
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $input `,` $edge $clock attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Assertion Directives
//===----------------------------------------------------------------------===//

class AssertLikeOp<string mnemonic, list<Trait> traits = []> :
  SVAOp<mnemonic, traits> {
  let arguments = (ins
    SVAAnyPropertyType:$property,
    Optional<I1>:$enable,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $property (`if` $enable^)? (`label` $label^)? (`message` $message^)? attr-dict `:` type($property)
  }];
}

def AssertPropertyOp : AssertLikeOp<"assert"> {
  let summary = "Assert that a property holds";
  let description = [{
    Asserts that the given property holds. During simulation, if the property
    fails, an error is reported with the optional message. During formal
    verification, this becomes a proof obligation.

    Example:
    ```mlir
    sva.assert %prop label "req_gnt" message "request not granted" : !sva.property
    ```
  }];
}

def AssumePropertyOp : AssertLikeOp<"assume"> {
  let summary = "Assume that a property holds";
  let description = [{
    Assumes that the given property holds. During formal verification, this
    constrains the input space. During simulation, this may be checked like
    an assertion or ignored.

    Example:
    ```mlir
    sva.assume %prop label "valid_input" : !sva.property
    ```
  }];
}

def CoverPropertyOp : AssertLikeOp<"cover"> {
  let summary = "Cover that a property can hold";
  let description = [{
    Records coverage when the given property holds. This is used to verify that
    interesting scenarios are exercised during simulation.

    Example:
    ```mlir
    sva.cover %prop label "corner_case" : !sva.property
    ```
  }];
}

def ExpectPropertyOp : AssertLikeOp<"expect"> {
  let summary = "Expect a property to hold (for procedural contexts)";
  let description = [{
    Similar to assert, but for use in procedural contexts. The property is
    evaluated immediately rather than being continuously monitored.

    Example:
    ```mlir
    sva.expect %prop label "immediate_check" : !sva.property
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Clocked Assertion Directives
//===----------------------------------------------------------------------===//

class ClockedAssertLikeOp<string mnemonic, list<Trait> traits = []> :
  SVAOp<mnemonic, traits> {
  let arguments = (ins
    SVAAnyPropertyType:$property,
    SVAClockEdgeAttr:$edge,
    I1:$clock,
    Optional<I1>:$enable,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $property `,` $edge $clock (`if` $enable^)? (`label` $label^)? (`message` $message^)? attr-dict `:` type($property)
  }];
}

def ClockedAssertPropertyOp : ClockedAssertLikeOp<"clocked_assert"> {
  let summary = "Assert a property on clock edges";
  let description = [{
    Asserts that the given property holds, checked on each clock edge.
    This is equivalent to wrapping the property in a clock operator.

    Example:
    ```mlir
    sva.clocked_assert %prop, posedge %clk label "req_gnt" : !sva.property
    ```
  }];
}

def ClockedAssumePropertyOp : ClockedAssertLikeOp<"clocked_assume"> {
  let summary = "Assume a property on clock edges";
}

def ClockedCoverPropertyOp : ClockedAssertLikeOp<"clocked_cover"> {
  let summary = "Cover a property on clock edges";
}

//===----------------------------------------------------------------------===//
// Disable Conditions
//===----------------------------------------------------------------------===//

def DisableIffOp : SVAOp<"disable_iff", [Pure]> {
  let summary = "Disable condition for a property";
  let description = [{
    Disables evaluation of a property when the condition is true. This is
    typically used to disable assertions during reset.

    This corresponds to the SVA `disable iff (cond)` construct.

    Example:
    ```mlir
    // disable iff (reset) property
    %0 = sva.disable_iff %reset, %prop : i1, !sva.property
    ```
  }];

  let arguments = (ins
    I1:$condition,
    SVAAnyPropertyType:$property
  );

  let results = (outs SVAPropertyType:$result);

  let assemblyFormat = [{
    $condition `,` $property attr-dict `:` type($condition) `,` type($property)
  }];
}

//===----------------------------------------------------------------------===//
// Local Variables and Sampling
//===----------------------------------------------------------------------===//

def SequenceMatchedOp : SVAOp<"seq.matched", [Pure]> {
  let summary = "Check if a sequence has matched";
  let description = [{
    Returns true if the sequence matched, used for detecting sequence endpoints
    in procedural code.

    This corresponds to the SVA `.matched` sequence method.

    Example:
    ```mlir
    %0 = sva.seq.matched %seq : !sva.sequence -> i1
    ```
  }];

  let arguments = (ins
    SVASequenceType:$sequence
  );

  let results = (outs I1:$result);

  let assemblyFormat = [{
    $sequence attr-dict `:` type($sequence) `->` type($result)
  }];
}

def SequenceTriggeredOp : SVAOp<"seq.triggered", [Pure]> {
  let summary = "Check if a sequence was triggered";
  let description = [{
    Returns true if the sequence was triggered (started matching) at the
    previous clock tick.

    This corresponds to the SVA `.triggered` sequence method.

    Example:
    ```mlir
    %0 = sva.seq.triggered %seq : !sva.sequence -> i1
    ```
  }];

  let arguments = (ins
    SVASequenceType:$sequence
  );

  let results = (outs I1:$result);

  let assemblyFormat = [{
    $sequence attr-dict `:` type($sequence) `->` type($result)
  }];
}

#endif // CIRCT_DIALECT_SVA_SVAOPS_TD
