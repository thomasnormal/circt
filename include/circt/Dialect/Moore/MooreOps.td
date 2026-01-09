//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreAttributes.td"
include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

class ResultIsSingleBitMatchingInputDomain<string result, string input> :
  TypesMatchWith<"result is single bit matching input domain",
    input, result, [{
    IntType::get($_self.getContext(), 1,
      llvm::cast<UnpackedType>($_self).getDomain())
  }]>;

class TypeDomainsMatch<list<string> values> :
  AllMatchSameOperatorTrait<values, [{
    cast<moore::UnpackedType>($_self.getType()).getDomain()
  }], "domain">;

class TypeIsSimpleBitVectorOf<string sbvType, string packedType> :
  TypesMatchWith<
    sbvType # " type is simple bit vector equivalent of " # packedType,
    sbvType, packedType, [{
      cast<PackedType>($_self).getSimpleBitVector()
    }]>;

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def SVModuleOp : MooreOp<"module", [
  IsolatedFromAbove,
  RegionKindInterface,
  Symbol,
  SingleBlockImplicitTerminator<"OutputOp">,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "A module definition";
  let description = [{
    The `moore.module` operation represents a SystemVerilog module, including
    its name, port list, and the constituent parts that make up its body. The
    module's body is a graph region.

    See IEEE 1800-2017 § 3.3 "Modules" and § 23.2 "Module definitions".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$bodyRegion);
  let hasCustomAssemblyFormat = 1;

  let builders = [OpBuilder<
    (ins "StringRef":$name, "hw::ModuleType":$type)>];
  let extraClassDeclaration = [{
    /// Return the `moore.output` op terminator of this module.
    OutputOp getOutputOp();
    /// Return the list of values assigned to output ports.
    OperandRange getOutputs();
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }
  }];
}

def OutputOp : MooreOp<"output", [
  Terminator, HasParent<"SVModuleOp">, Pure, ReturnLike
]> {
  let summary = "Assign module outputs";
  let description = [{
    The `moore.output` operation marks the end of a `moore.module` body region
    and specifies the values to present for the module's output ports.
  }];

  let arguments = (ins Variadic<AnyType>:$outputs);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, mlir::ValueRange()); }]>
  ];
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` type($outputs))?
  }];
  let hasVerifier = 1;
}

def InstanceOp : MooreOp<"instance", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Create an instance of a module";
  let description = [{
    The `moore.instance` operation instantiates a `moore.module` operation.

    See IEEE 1800-2017 § 23.3 "Module instances".
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$moduleName,
    Variadic<AnyType>:$inputs,
    StrArrayAttr:$inputNames,
    StrArrayAttr:$outputNames
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
}

def Initial: I32EnumAttrCase<"Initial", 0, "initial">;
def Final: I32EnumAttrCase<"Final", 1, "final">;
def Always: I32EnumAttrCase<"Always", 2, "always">;
def AlwaysComb: I32EnumAttrCase<"AlwaysComb", 3, "always_comb">;
def AlwaysLatch: I32EnumAttrCase<"AlwaysLatch", 4, "always_latch">;
def AlwaysFF: I32EnumAttrCase<"AlwaysFF", 5, "always_ff">;

def ProcedureKindAttr: I32EnumAttr<"ProcedureKind", "Procedure kind",
            [Initial, Final, Always, AlwaysComb, AlwaysLatch, AlwaysFF]>{
  let cppNamespace = "circt::moore";
}

def ProcedureOp : MooreOp<"procedure", [
  NoRegionArguments,
  RecursiveMemoryEffects,
  RecursivelySpeculatable
]> {
  let summary = "A procedure executed at different points in time";
  let description = [{
    The `moore.procedure` operation represents the SystemVerilog `initial`,
    `final`, `always`, `always_comb`, `always_latch`, and `always_ff`
    procedures.

    Execution times of the various procedures:

    - An `initial` procedure is executed once at the start of a design's
      lifetime, before any other procedures are executed.

    - A `final` procedure is executed once at the end of a design's lifetime,
      after all other procedures have stopped execution.

    - An `always` or `always_ff` procedure is repeatedly executed during a
      design's lifetime. Timing and event control inside the procedure can
      suspend its execution, for example to wait for a signal to change. If no
      such timing or event control is present, the procedure repeats infinitely
      at the current timestep, effectively deadlocking the design.

    - An `always_comb` or `always_latch` procedure is executed once at the start
      of a design's lifetime, after any `initial` procedures, and throughout the
      lifetime of the design whenever any of the variables read by the body of
      the procedure changes. Since the procedure is only executed when its
      change, and not repeatedly, the body generally does not contain any timing
      or event control. This behavior mitigates a shortcoming of `always`
      procedures, which commonly have an event control like `@*` that blocks
      and waits for a change of any input signals. This prevents the body from
      executing when the design is initialized and properly reacting to the
      initial values of signals. In contrast, `always_comb` and `always_latch`
      procedures have an implicit unconditional execution at design start-up.

    See IEEE 1800-2017 § 9.2 "Structured procedures".
  }];

  let arguments = (ins ProcedureKindAttr:$kind);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $kind attr-dict-with-keyword $body
  }];
}

def ReturnOp : MooreOp<"return", [
  Pure, Terminator, HasParent<"ProcedureOp">
]> {
  let summary = "Return from a procedure";
  let assemblyFormat = [{ attr-dict }];
}

def UnreachableOp : MooreOp<"unreachable", [Terminator]> {
  let summary = "Terminates a block as unreachable";
  let description = [{
    The `moore.unreachable` op is used to indicate that control flow never
    reaches the end of a block. This is useful for operations such as `$fatal`
    which never return as they cause the simulator to shut down. Behavior is
    undefined if control actually _does_ reach this terminator, but should
    probably crash the process with a useful error message.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def VariableOp : MooreOp<"variable", [
  DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"initial value and variable types match",
    "result", "initial", "cast<RefType>($_self).getNestedType()">
]> {
  let summary = "A variable declaration";
  let description = [{
    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    Optional<UnpackedType>:$initial
  );
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) ($initial^)? attr-dict
    `:` type($result)
  }];
  let hasCanonicalizeMethod = true;
}

def NetKindAttr : I32EnumAttr<"NetKind", "Net type kind", [
    I32EnumAttrCase<"Supply0", 0, "supply0">,
    I32EnumAttrCase<"Supply1", 1, "supply1">,
    I32EnumAttrCase<"Tri", 2, "tri">,
    I32EnumAttrCase<"TriAnd", 3, "triand">,
    I32EnumAttrCase<"TriOr", 4, "trior">,
    I32EnumAttrCase<"TriReg", 5, "trireg">,
    I32EnumAttrCase<"Tri0", 6, "tri0">,
    I32EnumAttrCase<"Tri1", 7, "tri1">,
    I32EnumAttrCase<"UWire", 8, "uwire">,
    I32EnumAttrCase<"Wire", 9, "wire">,
    I32EnumAttrCase<"WAnd", 10, "wand">,
    I32EnumAttrCase<"WOr", 11, "wor">,
    I32EnumAttrCase<"Interconnect", 12, "interconnect">,
    I32EnumAttrCase<"UserDefined", 13, "userdefined">,
    I32EnumAttrCase<"Unknown", 14, "unknown">,
  ]> {
  let cppNamespace = "::circt::moore";
}

def NetOp : MooreOp<"net", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"assigned value and variable types match",
    "result", "assignment", "cast<RefType>($_self).getNestedType()">,
]> {
  let summary = "A net declaration";
  let description = [{
    The `moore.net` operation is a net declaration. Net types defines different
    types of net connection in SV. There are twelve built-in net types defined
    in the official standard construct of the operation:
    `supply0`, `supply1`, `tri`, `triand`, `trior`, `trireg`, `tri0`, `tri1`,
    `uwire`, `wire`, `wand`, `wor`.
    Optional assignment argument allows net operation to be initialized with
    specific values as soon as it is created. Only one net declaration
    assignment can be made for a particular net. See IEEE 1800-2017 § 10.3.1
    "The net declaration assignment" for the differences between net declaration
    assignments and continuous assign statements. It has some features that are
    not supported: declaring an interconnect net and using user-defined types in
    the net operation.
    
    See IEEE 1800-2017 § 6.7 "Net declarations".
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    NetKindAttr:$kind,
    Optional<UnpackedType>:$assignment
  );
  let results = (outs Res<RefType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
    `:` type($result)
  }];
  let hasCanonicalizeMethod = true;
}

def AssignedVariableOp : MooreOp<"assigned_variable", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  SameOperandsAndResultType
]> {
  let summary = "A variable with a unique continuously assigned value";
  let arguments = (ins OptionalAttr<StrAttr>:$name, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $input attr-dict `:` type($input)
  }];
  let hasCanonicalizeMethod = true;
}

def ReadOp : MooreOp<"read", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
  TypesMatchWith<"input and result types match",
    "input", "result", "cast<RefType>($_self).getNestedType()">
]> {
  let summary = "Read the current value of a declaration";
  let description = [{
    Samples the current value of a declaration. This is a helper to capture the
    exact point at which declarations that can be targeted by all possible 
    expressions are read. It's similar to llvm.load.
  }];
  let arguments = (ins Arg<RefType, "", [MemRead]>:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def GlobalVariableOp : MooreOp<"global_variable", [
  IsolatedFromAbove,
  NoRegionArguments,
  SingleBlock,
  Symbol,
]> {
  let summary = "A global variable declaration";
  let description = [{
    Defines a global or package variable.

    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<UnpackedType>:$type
  );
  let regions = (region MaxSizedRegion<1>:$initRegion);
  let assemblyFormat = [{
    $sym_name attr-dict `:` $type (`init` $initRegion^)?
  }];
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    Block *getInitBlock();
  }];
}

def GetGlobalVariableOp : MooreOp<"get_global_variable", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure,
]> {
  let summary = "Get a reference to a global variable";
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $global_name attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "moore::GlobalVariableOp":$global), [{
      build($_builder, $_state,
        RefType::get(global.getType()), global.getSymName());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Assignments
//===----------------------------------------------------------------------===//

class AssignOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [TypesMatchWith<"src and dst types match",
    "src", "dst", "RefType::get(cast<UnpackedType>($_self))">]> {
  let arguments = (ins RefType:$dst, UnpackedType:$src);
  let assemblyFormat = [{
    $dst `,` $src attr-dict `:` type($src)
  }];
}

class DelayedAssignOpBase<string mnemonic, list<Trait> traits = []> :
    AssignOpBase<mnemonic, traits> {
  let arguments = (ins RefType:$dst, UnpackedType:$src, TimeType:$delay);
  let assemblyFormat = [{
    $dst `,` $src `,` $delay attr-dict `:` type($src)
  }];
}

// Continuous assignment

def ContinuousAssignOp : AssignOpBase<"assign", [HasParent<"SVModuleOp">]> {
  let summary = "Continuous assignment within a module";
  let description = [{
    A continuous assignment in module scope, such as `assign x = y;`, which
    continuously drives the value on the right-hand side onto the left-hand
    side.

    See IEEE 1800-2017 § 10.3 "Continuous assignments".
  }];
}

def DelayedContinuousAssignOp :
    DelayedAssignOpBase<"delayed_assign", [HasParent<"SVModuleOp">]> {
  let summary = "Delayed continuous assignment within a module";
  let description = [{
    A continuous assignment with a delay.

    See the `moore.assign` op.
    See IEEE 1800-2017 § 10.3 "Continuous assignments".
  }];
}

// Blocking assignment

def BlockingAssignOp : AssignOpBase<"blocking_assign", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>
]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A blocking procedural assignment in a sequential block, such as `x = y`. The
    effects of the assignment are visible to any subsequent operations in the
    block.

    See IEEE 1800-2017 § 10.4.1 "Blocking procedural assignments".
  }];
  let arguments = (ins
    Arg<RefType, "", [MemWrite]>:$dst,
    UnpackedType:$src
  );
}

// Non-blocking assignment

def NonBlockingAssignOp : AssignOpBase<"nonblocking_assign"> {
  let summary = "Nonblocking procedural assignment";
  let description = [{
    A nonblocking procedural assignment in a sequential block, such as `x <= y;`
    or `x <= @(posedge y) z` or `x <= #1ns y`. The assignment does not take
    effect immediately. Subsequent operations in the block do not see the
    effects of this assignment. Instead, the assignment is scheduled to happen
    in a subsequent time step as dictated by the delay or event control.

    See IEEE 1800-2017 § 10.4.2 "Nonblocking procedural assignments".
  }];
}

def DelayedNonBlockingAssignOp :
    DelayedAssignOpBase<"delayed_nonblocking_assign"> {
  let summary = "Delayed nonblocking procedural assignment";
  let description = [{
    A nonblocking procedural assignment with an intra-assignment delay control.

    See the `moore.nonblocking_assign` op.
    See IEEE 1800-2017 § 9.4.5 "Intra-assignment timing controls".
  }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

// Any change on the input.
def AnyChange: I32EnumAttrCase<"AnyChange", 0, "any">;
// A transition from 0 to X/Z/1, or from X/Z to 1.
def PosEdge: I32EnumAttrCase<"PosEdge", 1, "posedge">;
// A transition from 1 to X/Z/0, or from X/Z to 0.
def NegEdge: I32EnumAttrCase<"NegEdge", 2, "negedge">;
// The combination of `PosEdge` and `NegEdge`.
def BothEdges: I32EnumAttrCase<"BothEdges", 3, "edge">;

def EdgeAttr: I32EnumAttr<"Edge", "Edge kind",
                          [AnyChange, PosEdge, NegEdge, BothEdges]> {
  let cppNamespace = "circt::moore";
}

def WaitEventOp : MooreOp<"wait_event", [
  NoRegionArguments,
  SingleBlock,
  NoTerminator
]> {
  let summary = "Suspend execution until an event occurs";
  let description = [{
    The `moore.wait_event` op suspends execution of the current process until
    its body signals that an event has been the detected. Conceptually, the body
    of this op is executed whenever any potentially relevant signal has changed.
    If one of the contained `moore.detect_event` ops detect an event, execution
    resumes after the `moore.wait_event` operation. If no event is detected, the
    current process remains suspended.

    Example corresponding to the SystemVerilog `@(posedge x, negedge y iff z)`:
    ```
    moore.wait_event {
      %0 = moore.read %x : <i1>
      %1 = moore.read %y : <i1>
      %2 = moore.read %z : <i1>
      moore.detect_event posedge %0 : i1
      moore.detect_event negedge %1 if %2 : i1
    }
    ```

    The body may also contain any operations necessary to evaluate the event
    conditions. For example, the SV `@(posedge ~x iff i == 42)`:
    ```
    moore.wait_event {
      %0 = moore.read %x : <i1>
      %1 = moore.not %0 : i1
      %2 = moore.read %i : <i19>
      %3 = moore.constant 42 : i19
      %4 = moore.eq %2, %3 : i19
      moore.detect_event posedge %0 if %4 : i1
    }
    ```

    See IEEE 1800-2017 § 9.4.2 "Event control".
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ attr-dict-with-keyword $body }];
}

def DetectEventOp : MooreOp<"detect_event", [
  HasParent<"WaitEventOp">
]> {
  let summary = "Check if an event occured within a `wait_event` op";
  let description = [{
    The `moore.detect_event` op is used inside the body of a `moore.wait_event`
    to check if an interesting value change has occurred on its operand. The
    `moore.detect_event` op implicitly stores the previous value of its operand
    and compares it against the current value to detect an interesting edge:

    - `posedge` checks for a low-to-high transition
    - `negedge` checks for a high-to-low transition
    - `edge` checks for either a `posedge` or a `negedge`
    - `any` checks for any value change (including e.g. X to Z)

    The edges are detected as follows:

    - `0` to `1 X Z`: `posedge`
    - `1` to `0 X Z`: `negedge`
    - `X Z` to `1`: `posedge`
    - `X Z` to `0`: `negedge`

    | From  | To 0    | To 1    | To X    | To Z    |
    |-------|---------|---------|---------|---------|
    | 0     | -       | posedge | posedge | posedge |
    | 1     | negedge | -       | negedge | negedge |
    | X     | negedge | posedge | -       | -       |
    | Z     | negedge | posedge | -       | -       |

    See IEEE 1800-2017 § 9.4.2 "Event control".
  }];
  let arguments = (ins
    EdgeAttr:$edge,
    UnpackedType:$input,
    Optional<BitType>:$condition
  );
  let assemblyFormat = [{
    $edge $input (`if` $condition^)? attr-dict `:` type($input)
  }];
}

def WaitDelayOp : MooreOp<"wait_delay"> {
  let summary = "Suspend execution for a given amount of time";
  let description = [{
    The `moore.wait_delay` op suspends execution of the current process for the
    amount of time specified by its operand. Corresponds to the `#` delay
    control in SystemVerilog.

    See IEEE 1800-2017 § 9.4.1 "Delay control".
  }];
  let arguments = (ins TimeType:$delay);
  let assemblyFormat = [{ $delay attr-dict }];
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ConstantOp : MooreOp<"constant", [Pure, ConstantLike]> {
  let summary = "A constant integer value";
  let arguments = (ins FVIntegerAttr:$value);
  let results = (outs IntType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "IntType":$type, "const FVInt &":$value)>,
    OpBuilder<(ins "IntType":$type, "const APInt &":$value)>,
    OpBuilder<(
      ins "IntType":$type, "int64_t":$value, CArg<"bool", "true">:$isSigned)>,
  ];
}

def ConstantTimeOp : MooreOp<"constant_time", [Pure, ConstantLike]> {
  let summary = "A constant time value in femtoseconds";
  let arguments = (ins UI64Attr:$value);
  let results = (outs TimeType:$result);
  let assemblyFormat = "$value `fs` attr-dict";
  let hasFolder = 1;
}

def ConstantStringOp : MooreOp<"constant_string", [Pure]> {
  let summary = "A constant integer value defined by a string of bytes";
  let description = [{
    Defines a constant integer value based on the bytes of a string. The
    resulting integer contains 8 bits for each byte in the string. The first,
    left-most character is placed in the most significnat bits; the last,
    right-most character is placed in the least significant bits.

    Constant strings are represented as integers since they have a known width
    and bit-pattern, which is often used in SV to assign strings to parameters,
    or store strings in bit vectors. In contrast, the `string` type is a dynamic
    string with runtime-known length. To obtain a constant `string` value, the
    resulting integer should be converted to a `string`.

    See IEEE 1800-2017 § 5.9 "String literals".
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def APFloatAttr : Attr<CPred<"isa<FloatAttr>($_self)">,
                        "arbitrary float attribute"> {
  let storageType = [{ mlir::FloatAttr }];
  let returnType = [{ llvm::APFloat }];
}

def ConstantRealOp : MooreOp<"constant_real", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  Pure,
]> {
  let summary = "A constant real value";
  let description = [{
    Produces a constant value of a real type.

    See IEEE 1800-2017 § 5.7.2 "Real literal constants".
  }];
  let arguments = (ins APFloatAttr:$value);
  let results = (outs RealType:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// Casting
//===----------------------------------------------------------------------===//

def ConversionOp : MooreOp<"conversion", [Pure]> {
  let summary = "A type conversion";
  let description = [{
    An explicit or implicit type conversion. These are either generated
    automatically in order to make assignments compatible:

    ```
    int a;
    shortint b;
    a = b;  // generates an implicit cast from shortint to int
    ```

    Or explicitly by the user through a type, sign, or const cast expression:

    ```
    byte'(a)
    unsigned'(a)
    signed'(a)
    42'(a)
    ```

    See IEEE 1800-2017 § 6.24 "Casting".
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

def PackedToSBVOp : MooreOp<"packed_to_sbv", [
  Pure,
  TypeIsSimpleBitVectorOf<"input", "result">,
]> {
  let summary = "Convert a packed type to its simple bit vector equivalent";
  let arguments = (ins PackedTypeWithKnownSize:$input);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def SBVToPackedOp : MooreOp<"sbv_to_packed", [
  Pure,
  TypeIsSimpleBitVectorOf<"result", "input">,
]> {
  let summary = "Convert a simple bit vector to an equivalent packed type";
  let arguments = (ins IntType:$input);
  let results = (outs PackedTypeWithKnownSize:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

def LogicToIntOp : MooreOp<"logic_to_int", [
  Pure,
  TypesMatchWith<"result matches input with two-valued domain",
    "input", "result", "cast<IntType>($_self).getTwoValued()">
]> {
  let summary = "Convert a four-valued to a two-valued integer";
  let arguments = (ins FourValuedIntType:$input);
  let results = (outs TwoValuedIntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let hasFolder = 1;
}

def IntToLogicOp : MooreOp<"int_to_logic", [
  Pure,
  TypesMatchWith<"result matches input with four-valued domain",
    "input", "result", "cast<IntType>($_self).getFourValued()">
]> {
  let summary = "Convert a two-valued to a four-valued integer";
  let arguments = (ins TwoValuedIntType:$input);
  let results = (outs FourValuedIntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let hasFolder = 1;
}

def ToBuiltinBoolOp : MooreOp<"to_builtin_bool", [Pure]> {
  let summary = "Convert a `!moore.l1` or `!moore.i1` to a builtin `i1`";
  let description = [{
    Maps `X` and `Z` to 0, and passes through 1 and 0 unchanged.
  }];
  let arguments = (ins AnySingleBitType:$input);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def TimeToLogicOp : MooreOp<"time_to_logic", [Pure]> {
  let summary = "Convert a time type to an integer number of femtoseconds";
  let arguments = (ins TimeType:$input);
  let results = (outs FourValuedI64:$result);
  let assemblyFormat = "$input attr-dict";
  let hasFolder = 1;
}

def LogicToTimeOp : MooreOp<"logic_to_time", [Pure]> {
  let summary = "Convert an integer number of femtoseconds to a time type";
  let arguments = (ins FourValuedI64:$input);
  let results = (outs TimeType:$result);
  let assemblyFormat = "$input attr-dict";
  let hasFolder = 1;
}

def RealToIntOp : MooreOp<"real_to_int", [
  Pure
]> {
  let summary = "Convert a real value to a two-valued integer";
  let description = [{
    See IEEE 1800-2023 Section 6.24.1: "Cast operator" and 6.12:
    "Real, shortreal and realtime data types". Accordingly,
    output values are rounded.
  }];
  let arguments = (ins RealType:$input);
  let results = (outs TwoValuedIntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

class IntToRealOpBase<string name> : MooreOp<name, [
  Pure
]> {
  let summary = "Convert an integer value to a real";
  let description = [{
    See IEEE 1800-2023 Section 6.24.1: "Cast operator" and 6.12:
    "Real, shortreal and realtime data types".
  }];
  let arguments = (ins TwoValuedIntType:$input);
  let results = (outs RealType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def SIntToRealOp : IntToRealOpBase<"sint_to_real">;
def UIntToRealOp : IntToRealOpBase<"uint_to_real">;

def IntToStringOp : MooreOp<"int_to_string", [
  Pure
]> {
  let summary = "Convert an integer to a string";
  let arguments = (ins TwoValuedIntType:$input);
  let results = (outs StringType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def StringToIntOp : MooreOp<"string_to_int", [
  Pure
]> {
  let summary = "Convert a string to an integer";
  let description = [{
    If the width of the result type is smaller than the input string, it is
    truncated. If it is larger, it is appended with 0s.
  }];
  
  let arguments = (ins StringType:$input);
  let results = (outs TwoValuedIntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

def ConvertRealOp : MooreOp<"convert_real", [
  Pure
]> {
  let summary = "Convert a real to a different bitwidth";
  let description = [{
    The value is rounded to the nearest value representable by the target type.
    See IEEE 1800-2023 Section 6.24.1.
  }];
  
  let arguments = (ins RealType:$input);
  let results = (outs RealType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Resizing
//===----------------------------------------------------------------------===//

def TruncOp : MooreOp<"trunc", [
  Pure,
  TypeDomainsMatch<["input", "result"]>,
  PredOpTrait<"result width must be smaller than input width", CPred<[{
    cast<moore::IntType>($result.getType()).getBitSize() <
      cast<moore::IntType>($input.getType()).getBitSize()
  }]>>,
]> {
  let summary = "Truncate a value";
  let description = [{
    Reduce the bit width of a value by removing some of its most significant
    bits. This can only change the bit width of an integer type.
  }];
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

class ExtOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  TypeDomainsMatch<["input", "result"]>,
  PredOpTrait<"result width must be larger than input width", CPred<[{
    cast<moore::IntType>($result.getType()).getBitSize() >
      cast<moore::IntType>($input.getType()).getBitSize()
  }]>>,
]> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

def ZExtOp : ExtOpBase<"zext"> {
  let summary = "Zero-extend a value";
  let description = [{
    Increase the bit width of a value by inserting additional zero most
    significant bits. This keeps the unsigned value constant.
  }];
}

def SExtOp : ExtOpBase<"sext"> {
  let summary = "Sign-extend a value";
  let description = [{
    Increase the bit width of a value by replicating its most significant bit.
    This keeps the signed value constant.
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def UArrayCmpPredicateAttr : I64EnumAttr<
    "UArrayCmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>
    ]> {
  let cppNamespace = "circt::moore";
}

def StringCmpPredicateAttr : I64EnumAttr<
    "StringCmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>
    ]> {
  let cppNamespace = "circt::moore";
}

def NegOp : MooreOp<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Arithmetic negation";
  let description = [{
    Negate a value to its two's complement form. If any bit in the input is Z or
    X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def NegRealOp : MooreOp<"fneg", [Pure, SameOperandsAndResultType]> {
  let summary = "Arithmetic negation";
  let description = [{
    Negate a real value.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators" and § 11.3.1 "Operators
    with real operands"
  }];
  let arguments = (ins RealType:$input);
  let results = (outs RealType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def NotOp : MooreOp<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise unary negation";
  let description = [{
    Applies the boolean NOT operation to each bit in the input. Corresponds to
    the `~` operator, as well as the negation in the `~&`, `~|`, `^~`, and `~^`
    reduction operators.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    | Input | Result |
    |-------|--------|
    | 0     | 1      |
    | 1     | 0      |
    | X     | X      |
    | Z     | X      |
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

class ReduceOpBase<string mnemonic, string operatorName> : MooreOp<mnemonic, [
  Pure,
  ResultIsSingleBitMatchingInputDomain<"result", "input">
]> {
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let summary = !subst("$op", operatorName, "Reduction $op operator");
  let description = !subst("$op", operatorName, [{
    Reduces all bits in the input to a single result bit by iteratively applying
    the boolean $op operator. If the input has only a single bit, that bit is
    returned.

    See IEEE 1800-2017 § 11.4.9 "Reduction operators". See the corresponding
    `and`, `or`, and `xor` operations for the truth table.
  }]);
}

def ReduceAndOp : ReduceOpBase<"reduce_and", "AND">;
def ReduceOrOp : ReduceOpBase<"reduce_or", "OR">;
def ReduceXorOp : ReduceOpBase<"reduce_xor", "XOR">;

def BoolCastOp : MooreOp<"bool_cast", [
  Pure,
  ResultIsSingleBitMatchingInputDomain<"result", "input">
]> {
  let summary = "Cast a value to a single bit boolean";
  let description = [{
    Convert a nonzero or true value into 1, a zero or false value into 0, and
    any value containing Z or X bits into a X. This conversion is useful in
    combination with the logical and, or, implication, equivalence, and negation
    operators.

    See IEEE 1800-2017 § 11.4.7 "Logical operators".
  }];
  let arguments = (ins UnpackedType:$input);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

class BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def AddOp : BinaryOpBase<"add", [Commutative]> {
  let summary = "Addition";
  let description = [{
    Add the operands. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def SubOp : BinaryOpBase<"sub"> {
  let summary = "Subtraction";
  let description = [{
    Subtract the right-hand side from the left-hand side operand. If any bit in
    the two operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let hasFolder = 1;
}

def MulOp : BinaryOpBase<"mul", [Commutative]> {
  let summary = "Multiplication";
  let description = [{
    Multiply the operands. If any bit in the two operands is Z or X, all bits in
    the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let hasFolder = 1;
}

class DivOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
  let summary = "Division";
  let description = [{
    Divide the left-hand side by the right-hand side operand. Any fractional
    part is truncated toward zero. If the right-hand side is zero, all bits of
    the result are X. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let hasFolder = 1;
}

def DivUOp : DivOpBase<"divu">;
def DivSOp : DivOpBase<"divs">;

class ModOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
  let summary = "Remainder";
  let description = [{
    Compute the remainder of the left-hand side divided by the right-hand side
    operand. If the right-hand side is zero, all bits of the result are X. The
    sign of the result is the sign of the left-hand side. If any bit in the two
    operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Consider the following examples:

    | LHS | RHS | Result |
    |-----|-----|--------|
    |  11 |   3 |      2 |
    | -11 |   3 |     -2 |
    |  11 |  -3 |      2 |
    | -11 |  -3 |     -2 |
  }];
}

def ModUOp : ModOpBase<"modu">;
def ModSOp : ModOpBase<"mods">;

class PowOpBase<string mnemonic> : BinaryOpBase<mnemonic> {
  let summary = "Power";
  let description = [{
    Raise the left-hand side to the power of the right-hand side. `powu` treats
    its operands as unsigned numbers, while `pows` treats them as signed
    numbers.

    Evaluation rules for `a ** b`:

    |       | a < -1 | a = -1         | a = 0 | a = 1 | a > 1  |
    |-------|--------|----------------|-------|-------|--------|
    | b > 0 | a ** b | b odd ? -1 : 1 |     0 |     1 | a ** b |
    | b = 0 |      1 |              1 |     1 |     1 |      1 |
    | b < 0 |      0 | b odd ? -1 : 1 |     X |     1 |      0 |

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def PowUOp : PowOpBase<"powu">;
def PowSOp : PowOpBase<"pows">;

def AndOp : BinaryOpBase<"and", [Commutative]> {
  let summary = "Bitwise AND operation";
  let description = [{
    Applies the boolean AND operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `&` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 0 | 0 | 0 |
    | 1 | 0 | 1 | X | X |
    | X | 0 | X | X | X |
    | Z | 0 | X | X | X |
  }];
}

def OrOp : BinaryOpBase<"or", [Commutative]> {
  let summary = "Bitwise OR operation";
  let description = [{
    Applies the boolean OR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `|` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 1 | 1 | 1 |
    | X | X | 1 | X | X |
    | Z | X | 1 | X | X |
  }];
}

def XorOp : BinaryOpBase<"xor", [Commutative]> {
  let summary = "Bitwise XOR operation";
  let description = [{
    Applies the boolean XOR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `^` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 0 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |
  }];
}

class ShiftOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  TypesMatchWith<
    "value and result types must match", "value", "result", "$_self">
]> {
  let description = [{
    Shifts the `value` to the left or right by `amount` number of bits. The
    result has the same type as the input value. The amount is always treated as
    an unsigned number and has no effect on the signedness of the result. X or
    Z bits in the input value are simply shifted left or right the same way 0 or
    1 bits are. If the amount contains X or Z bits, all result bits are X.

    `shl` shifts bits to the left, filling in 0 for the vacated least
    significant bits. `shr` and `ashr` shift bits to the right; `shr` fills in
    0 for the vacated most significant bits, and `ashr` copies the input's sign
    bit into the vacated most significant bits. Note that in contrast to the SV
    spec, the `ashr` _always_ fills in the sign bit regardless of the signedness
    of the input.

    `shl` corresponds to the `<<` and `<<<` operators. `shr` corresponds to the
    `>>` operator, and the `>>>` operator applied to an unsigned value. `ashr`
    corresponds to the `>>>` operator applied to a signed value.

    See IEEE 1800-2017 § 11.4.10 "Shift operators".
  }];
  let arguments = (ins SimpleBitVectorType:$value, SimpleBitVectorType:$amount);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $value `,` $amount attr-dict `:` type($value) `,` type($amount)
  }];
}

def ShlOp : ShiftOpBase<"shl"> { let summary = "Logical left shift"; }
def ShrOp : ShiftOpBase<"shr"> { let summary = "Logical right shift"; }
def AShrOp : ShiftOpBase<"ashr"> { let summary = "Arithmetic right shift"; }

def UArrayCmpOp : MooreOp<"uarray_cmp", [
  Pure,
  Commutative,
  SameTypeOperands,
]> {
  let description = [{
    Performs an elementwise comparison of two unpacked arrays
    using the specified predicate (for example, "eq" for equality or "ne" for inequality)
    and returns a single bit result.
    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)

    The result is `1` if the comparison is true and `0` otherwise.
  }];
  let arguments = (ins 
    UArrayCmpPredicateAttr:$predicate,
    UnpackedArrayType:$lhs,
    UnpackedArrayType:$rhs
  );
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];  
}

def StringCmpOp : MooreOp<"string_cmp", [
  Pure,
  SameTypeOperands
]> {
  let description = [{
    Compares two string operands using the specified
    predicate and returns a single bit result. Supported predicates:
    - eq : equal
    - ne : not equal
    - lt : less than
    - le : less or equal
    - gt : greater than
    - ge : greater or equal

    The equality operator yields 1 if its operands are equal and 0 otherwise.
    Relational operators compare two strings lexicographically,
    returning 1 when the specified condition holds and 0 when it does not.
  }];
  let arguments = (ins
    StringCmpPredicateAttr:$predicate,
    StringType:$lhs,
    StringType:$rhs
  );
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

class LogicalEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands,
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If all corresponding bits in the left- and
    right-hand side are equal, and all are 0 or 1 (not X or Z), the two operands
    are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
    equal, but all are 0 or 1, the two operands are considered not equal (`eq`
    returns 0, `ne` returns 1). If any bit in the two operands is Z or X,
    returns X. `eq` corresponds to the `==` operator and `ne` to the `!=`
    operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def EqOp : LogicalEqOpBase<"eq"> { let summary = "Logical equality"; }
def NeOp : LogicalEqOpBase<"ne"> { let summary = "Logical inequality"; }

class CaseEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0 or 1 result. If all corresponding bits in the left- and
    right-hand side are equal (both 0, 1, X, or Z), the two operands are
    considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
    not equal, the two operands are considered not equal (`case_eq` returns 0,
    `case_ne` returns 1). `case_eq` corresponds to the `===` operator and
    `case_ne` to the `!==` operator.

    `casez_eq` treats Z bits in either operand as wildcards and skips them
    during the comparison. `casexz_eq` treats X and Z bits as wildcards. These
    are different from the `wildcard_eq` operation, which only considers X/Z in
    the right-hand operand as wildcards.

    Case statements use this operation to perform case comparisons:
    - `case` statements use `case_eq`
    - `casez` statements use `casez_eq`
    - `casex` statements use `casexz_eq`

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

def CaseEqOp : CaseEqOpBase<"case_eq"> { let summary = "Case equality"; }
def CaseNeOp : CaseEqOpBase<"case_ne"> { let summary = "Case inequality"; }
def CaseZEqOp : CaseEqOpBase<"casez_eq"> {
  let summary = "Case equality with Z as wildcard";
}
def CaseXZEqOp : CaseEqOpBase<"casexz_eq"> {
  let summary = "Case equality with X and Z as wildcard";
}

class WildcardEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If any bit in the left-hand side is Z or X,
    returns X. Performs the same comparison as the `eq` and `ne` operations, but
    all right-hand side bits that are X or Z are skipped. Therefore, X and Z in
    the right-hand side act as wildcards or "don't care" values. `wildcard_eq`
    corresponds to the `==?` operator and `wildcard_ne` to the `!=?` operator.

    See IEEE 1800-2017 § 11.4.6 "Wildcard equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def WildcardEqOp : WildcardEqOpBase<"wildcard_eq"> {
  let summary = "Wildcard equality";
}
def WildcardNeOp : WildcardEqOpBase<"wildcard_ne"> {
  let summary = "Wildcard inequality";
}

class RelationalOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the left- and right-hand side operand and returns a single bit 0,
    1, or X result. If any bit in the two operands is Z or X, returns X.
    Otherwise, if all bits are 0 or 1, `ult/slt`, `ule/sle`, `ugt/sgt`, and
    `uge/sge` return whether the left-hand side is less than, less than or equal
    to, greater than, or greater than or equal to the right-hand side,
    respectively. `ult/slt` corresponds to the `<` operator, `ule/sle` to `<=`,
    `ugt/sgt` to `>`, and `uge/sge` to `>=`.

    See IEEE 1800-2017 § 11.4.4 "Relational operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs AnySingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def UltOp : RelationalOpBase<"ult"> {
  let summary = "Unsigned less than comparison";
}
def UleOp : RelationalOpBase<"ule"> {
  let summary = "Unsigned less than or equal comparison";
}
def UgtOp : RelationalOpBase<"ugt"> {
  let summary = "Unsigned greater than comparison";
}
def UgeOp : RelationalOpBase<"uge"> {
  let summary = "Unsigned greater than or equal comparison";
}

def SltOp : RelationalOpBase<"slt"> {
  let summary = "Signed less than comparison";
}
def SleOp : RelationalOpBase<"sle"> {
  let summary = "Signed less than or equal comparison";
}
def SgtOp : RelationalOpBase<"sgt"> {
  let summary = "Signed greater than comparison";
}
def SgeOp : RelationalOpBase<"sge"> {
  let summary = "Signed greater than or equal comparison";
}

def ConcatOp : MooreOp<"concat", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "A concatenation of expressions";
  let description = [{
    This operation represents the SystemVerilog concatenation expression
    `{x, y, z}`. See IEEE 1800-2017 §11.4.12 "Concatenation operators".

    All operands must be simple bit vector types.

    The concatenation result is a simple bit vector type. The result is unsigned
    regardless of the sign of the operands (see concatenation-specific rules in
    IEEE 1800-2017 §11.8.1 "Rules for expression types"). The size of the result
    is the sum of the sizes of all operands. If any of the operands is
    four-valued, the result is four-valued; otherwise it is two-valued.
  }];
  let arguments = (ins Variadic<IntType>:$values);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` `(` type($values) `)` `->` type($result)
  }];
}

def ConcatRefOp : MooreOp<"concat_ref", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "The copy of concat that explicitly works on the ref type.";
  let arguments = (ins Variadic<RefType>:$values);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` `(` type($values) `)` `->` type($result)
  }];
}

def ReplicateOp : MooreOp<"replicate", [
  Pure
]> {
  let summary = "Multiple concatenation of expressions";
  let description = [{
    This operation indicates a joining together of that many copies of the 
    concatenation `{constant{w}}`. Which enclosed together within brace.
    The 'constant' must a non-negative, non-x, and non-z constant expression.
    The 'constant' may be a value of zero, but it only exists in parameterized
    code, and it will be ignored(type is changed to the void).

    Example:
    ```
      {0{w}}   // empty! ignore it.
      {4{w}}   // the same as {w, w, w, w}
    ```
    See IEEE 1800-2017 §11.4.12 "Concatenation operators".
  }];
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Real Binary Operations
//===----------------------------------------------------------------------===//

class BinaryRealOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins RealType:$lhs, RealType:$rhs);
  let description = [{
    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators" and § 11.3.1 "Operators
    with real operands"
  }];
  let results = (outs RealType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def AddRealOp : BinaryRealOpBase<"fadd"> {
  let summary = "Addition; add two real operands.";
}

def SubRealOp : BinaryRealOpBase<"fsub"> {
  let summary = [{Subtraction; subtract the RHS real operand from the LHS
  real operand.}];
}

def DivRealOp : BinaryRealOpBase<"fdiv"> {
  let summary = [{Division; Divide the LHS real operand by the LHS
  real operand.}];
}

def MulRealOp : BinaryRealOpBase<"fmul"> {
  let summary = [{Multiplication; Multiply the RHS real operand with the LHS
  real operand.}];
}

def PowRealOp : BinaryRealOpBase<"fpow"> {
  let summary = [{Power; Exponentiate the LHS real base with the RHS real
  exponent.}];
}

class LogicalEqRealOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands,
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0 or 1. If all corresponding bits in the left- and
    right-hand side are equal, and all are 0 or 1, the two operands
    are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
    equal, but all are 0 or 1, the two operands are considered not equal (`eq`
    returns 0, `ne` returns 1). `eq` corresponds to the `==` operator and `ne`
    to the `!=` operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators" and § 11.3.1 "Operators
    with real operands".
  }];
  let arguments = (ins RealType:$lhs, RealType:$rhs);
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def EqRealOp : LogicalEqRealOpBase<"feq"> { let summary = "Logical equality"; }
def NeRealOp : LogicalEqRealOpBase<"fne"> { let summary = "Logical inequality"; }


class RelationalRealOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  ResultIsSingleBitMatchingInputDomain<"result", "lhs">
]> {
  let description = [{
    Compares the left- and right-hand side operand and returns a single bit 0,
    or 1 result. If all bits are 0 or 1, `flt`, `fle`, `fgt`, and
    `fge` return whether the left-hand side is less than, less than or equal
    to, greater than, or greater than or equal to the right-hand side,
    respectively. `flt` corresponds to the `<` operator, `fle` to `<=`,
    `fgt` to `>`, and `fge` to `>=`.

    See IEEE 1800-2017 § 11.4.4 "Relational operators" and § 11.3.1 "Operators
    with real operands".
  }];
  let arguments = (ins RealType:$lhs, RealType:$rhs);
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}


def FltOp : RelationalRealOpBase<"flt"> {
  let summary = "Real-valued less than comparison";
}
def FleOp : RelationalRealOpBase<"fle"> {
  let summary = "Real-valued less than or equal comparison";
}
def FgtOp : RelationalRealOpBase<"fgt"> {
  let summary = "Real-valued greater than comparison";
}
def FgeOp : RelationalRealOpBase<"fge"> {
  let summary = "Real-valued greater than or equal comparison";
}

//===----------------------------------------------------------------------===//
// Bit/Element Extraction
//===----------------------------------------------------------------------===//

def ExtractOp : MooreOp<"extract", [Pure]> {
  let summary = "Extract a range or single bits from a value";
  let description = [{
    It's used to select from a value with a constant low bit.
    This operation includes the vector bit/part-select, array, and memory 
    addressing.If the address is invalid--out of bounds or has x or z bit--
    then it will produce x for 4-state or 0 for 2-state.
    Bit-select results are unsigned, regardless of the operands.
    Part-select results are unsigned, regardless of the operands even if 
    the part-select specifies the entire vector.
    See IEEE 1800-2017 § 11.8.1 "Rules for expression types"

    Example:
    ```
    logic v [7:0];
    v[1];                      // the bit-select addressing
    v[3:0];                    // the part-select addressing
    v[3-:4];  v[0+:4];         // They are equivalent to v[3:0]
    ```
    See IEEE 1800-2017 § 11.5.1 "Vector bit-select and part-select addressing".

    Example:
    ```
    // an array of 256-by-256 8-bit elements
    logic [7:0] twod_array [0:255][0:255];
    logic [7:0] mem_name [0:1023];      // a memory of 1024 8-bit words
    ```
    See IEEE 1800-2017 § 11.5.2 "Array and memory addressing".
  }];
  let arguments = (ins UnpackedType:$input, I32Attr:$lowBit);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
  }];
}

def DynExtractOp : MooreOp<"dyn_extract", [Pure]> {
  let description = [{
    It's similar with extract, but it's used to select from a value
    with a dynamic low bit.
  }];
  let arguments = (ins UnpackedType:$input, UnpackedType:$lowBit);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

def ExtractRefOp : MooreOp<"extract_ref", [Pure]> {
  let description = [{
  The copy of extract that explicitly works on the ref type.
  }];
  let arguments = (ins RefType:$input, I32Attr:$lowBit);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` type($input) `->` type($result)
  }];
}

def DynExtractRefOp : MooreOp<"dyn_extract_ref", [Pure]> {
  let description = [{
  The copy of dyn_extract that explicitly works on the ref type.
  }];
  let arguments = (ins RefType:$input, UnpackedType:$lowBit);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:`
    type($input) `,` type($lowBit) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Array Manipulation
//===----------------------------------------------------------------------===//

def ArrayCreateOp : MooreOp<"array_create", [Pure, SameTypeOperands]> {
  let summary = "Create an array value from individual elements";
  let arguments = (ins Variadic<UnpackedType>:$elements);
  let results = (outs AnyStaticArrayType:$result);
  let assemblyFormat = [{
    $elements attr-dict `:` type($elements) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Struct Manipulation
//===----------------------------------------------------------------------===//

def StructCreateOp : MooreOp<"struct_create", [Pure]> {
  let summary = "Create a struct value from individual fields";
  let arguments = (ins Variadic<UnpackedType>:$fields);
  let results = (outs AnyStructType:$result);
  let assemblyFormat = [{
    $fields attr-dict `:` type($fields) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def StructExtractOp : MooreOp<"struct_extract", [Pure]> {
  let summary = "Obtain the value of a struct field";
  let arguments = (ins StrAttr:$fieldName, AnyStructType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def StructExtractRefOp : MooreOp<"struct_extract_ref", [
  Pure,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>
]> {
  let summary = "Create a reference to a struct field";
  let arguments = (ins StrAttr:$fieldName, AnyStructRefType:$input);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def StructInjectOp : MooreOp<"struct_inject", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Update the value of a struct field";
  let description = [{
    Takes an existing struct value, sets one of its fields to a new value, and
    returns the resulting struct value.
  }];
  let arguments = (ins
    AnyStructType:$input,
    StrAttr:$fieldName,
    UnpackedType:$newValue
  );
  let results = (outs AnyStructType:$result);
  let assemblyFormat = [{
    $input `,` $fieldName `,` $newValue attr-dict
    `:` type($input) `,` type($newValue)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// Union Manipulation
//===----------------------------------------------------------------------===//

def UnionCreateOp : MooreOp<"union_create", [Pure]> {
  let summary = "Union Create operation";
  let description = [{
    A union is a data type that represents a single piece
    of storage that can be accessed using one of
    the named member data types. Only one of the
    data types in the union can be used at a time.
    By default, a union is unpacked, meaning there
    is no required representation for how members
    of the union are stored. Dynamic types and chandle
    types can only be used in tagged unions.
    See IEEE 1800-2017 § 7.3 "Unions"

    Example:
    ```
    typedef union { int i; shortreal f; } num; // named union type
    num n;
    n.f = 0.0; // set n in floating point format
    typedef struct {
    bit isfloat;
    union { int i; shortreal f; } n;           // anonymous union type
    } tagged_st;                               // named structure
    ```
    See IEEE 1800-2017 § 7.3 "Unions"
  }];
  let arguments = (ins UnpackedType:$input, StrAttr:$fieldName);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def UnionExtractOp : MooreOp<"union_extract"> {
  let summary = "Union Extract operation";
  let description = [{
    With packed unions, writing one member and reading another is
    independent of the byte ordering of the machine,
    unlike an unpacked union of unpacked structures,
    which are C-compatible and have members in ascending address order.
    See IEEE 1800-2017 § 7.3.1 "Packed unions"

    Example:
    ```
    typedef union packed { // default unsigned
    s_atmcell acell;
    bit [423:0] bit_slice;
    bit [52:0][7:0] byte_slice;
    } u_atmcell;
    u_atmcell u1;
    byte b; bit [3:0] nib;
    b = u1.bit_slice[415:408]; // same as b = u1.byte_slice[51];
    nib = u1.bit_slice [423:420];
    ```
    See IEEE 1800-2017 § 7.3.1 "Packed unions"
  }];
  let arguments = (ins StrAttr:$fieldName, UnpackedType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `,`   $fieldName  attr-dict `:`
    type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def UnionExtractRefOp : MooreOp<"union_extract_ref"> {
  let summary = "Union Extract operation";
  let arguments = (ins StrAttr:$fieldName, RefType:$input);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $input `,`   $fieldName  attr-dict `:`
    type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def ConditionalOp : MooreOp<"conditional",[
  RecursiveMemoryEffects,
  NoRegionArguments,
]> {
  let summary = "Conditional operation";
  let description = [{
    If the condition is true, this op evaluates the first region and returns its
    result without evaluating the second region. If the the condition is false,
    this op evaluates the second region and returns its result without
    evaluating the first region.

    If the condition is unknown (X or Z), _both_ regions are evaluated. If both
    results are equal as per `case_eq`, one of the results is returned. If the
    results are not equal, this op returns a value based on the data types of
    the results.

    In case the results of the first and second region are of an integral type,
    they are merged by applying the following bit-wise truth table:

    |?: | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | X | X | X |
    | 1 | X | 1 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |

    Non-integral data types define other rules which are not yet implemented.
    See IEEE 1800-2017 § 11.4.11 "Conditional operator".
  }];
  let arguments = (ins AnySingleBitType:$condition);
  let results = (outs UnpackedType:$result);
  let regions = (region SizedRegion<1>:$trueRegion,
                        SizedRegion<1>:$falseRegion);
  let assemblyFormat = [{
    $condition attr-dict `:` type($condition) `->` type($result)
    $trueRegion $falseRegion
  }];
}

def YieldOp : MooreOp<"yield", [
  Pure,
  Terminator,
  ParentOneOf<["ConditionalOp", "GlobalVariableOp"]>,
]> {
  let summary = "conditional yield and termination operation";
  let description = [{
    "moore.yield" yields an SSA value from the Moore dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "moore.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "moore.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];
  let arguments = (ins UnpackedType:$result);
  let assemblyFormat = [{
    attr-dict $result `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

// Simple immediate assertions, like `assert`.
def ImmediateAssert: I32EnumAttrCase<"Immediate", 0, "immediate">;
// Observed deferred assertions, like `assert #0`.
def ObservedAssert: I32EnumAttrCase<"Observed", 1, "observed">;
// Final deferred assertions, like `assert final`.
def FinalAssert: I32EnumAttrCase<"Final", 2, "final">;

// A mode specifying how immediate/deferred assertions operate.
def DeferAssertAttr : I32EnumAttr<
  "DeferAssert", "assertion deferring mode",
  [ImmediateAssert, ObservedAssert, FinalAssert]>
{
  let cppNamespace = "circt::moore";
}

class ImmediateAssertOp<string mnemonic, list<Trait> traits = []> : 
    MooreOp<mnemonic, traits # [HasParent<"ProcedureOp">]>{
  let arguments = (ins
    DeferAssertAttr:$defer,
    AnySingleBitType:$cond,
    OptionalAttr<StrAttr>:$label
  );
  let assemblyFormat = [{
    $defer $cond (`label` $label^)? attr-dict `:` type($cond)
  }];
}

def AssertOp : ImmediateAssertOp<"assert">{
  let summary = "If cond is not true, an error should be thrown.";
}

def AssumeOp : ImmediateAssertOp<"assume">{
  let summary = "Verify the cond whether has the expected behavior.";
}

def CoverOp : ImmediateAssertOp<"cover">{
  let summary = "Monitor the coverage information.";
}

//===----------------------------------------------------------------------===//
// Format Strings
//===----------------------------------------------------------------------===//

def FormatLiteralOp : MooreOp<"fmt.literal", [Pure]> {
  let summary = "A constant string fragment";
  let description = [{
    Creates a constant string fragment to be used as a format string. The
    literal is printed as is, without any further escaping or processing of its
    characters.
  }];
  let arguments = (ins StrAttr:$literal);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = "$literal attr-dict";
}

def FormatStringToStringOp : MooreOp<"fstring_to_string", [Pure]> {
  let summary = "A dynamic string created through formatting";
  let description = [{
    Creates a dynamic string from a format string.
    Translates into a no-op.
  }];
  let arguments = (ins FormatStringType:$fmtstring);
  let results = (outs StringType:$result);
  let assemblyFormat = "$fmtstring attr-dict";
}

def FormatConcatOp : MooreOp<"fmt.concat", [Pure]> {
  let summary = "Concatenate string fragments";
  let description = [{
    Concatenates an arbitrary number of format string into one larger format
    string. The strings are concatenated from left to right, with the first
    operand appearing at the left start of the result string, and the last
    operand appearing at the right end. Produces an empty string if no inputs
    are provided.
  }];
  let arguments = (ins Variadic<FormatStringType>:$inputs);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = "` ` `(` $inputs `)` attr-dict";
}

def FmtDec : I32EnumAttrCase<"Decimal", 0, "decimal">;
def FmtBin : I32EnumAttrCase<"Binary", 1, "binary">;
def FmtOct : I32EnumAttrCase<"Octal", 2, "octal">;
def FmtHexL : I32EnumAttrCase<"HexLower", 3, "hex_lower">;
def FmtHexU : I32EnumAttrCase<"HexUpper", 4, "hex_upper">;
def IntFormatAttr : I32EnumAttr<"IntFormat", "Integer format",
                               [FmtDec, FmtBin, FmtOct, FmtHexL, FmtHexU]> {
  let cppNamespace = "circt::moore";
}

def AlignRight : I32EnumAttrCase<"Right", 0, "right">;
def AlignLeft : I32EnumAttrCase<"Left", 1, "left">;
def IntAlignAttr : I32EnumAttr<"IntAlign", "Integer alignment",
                              [AlignRight, AlignLeft]> {
  let cppNamespace = "circt::moore";
}

def PadSpace : I32EnumAttrCase<"Space", 0, "space">;
def PadZero : I32EnumAttrCase<"Zero", 1, "zero">;
def IntPaddingAttr : I32EnumAttr<"IntPadding", "Integer alignment",
                                 [PadSpace, PadZero]> {
  let cppNamespace = "circt::moore";
}

def FormatIntOp : MooreOp<"fmt.int", [Pure]> {
  let summary = "Format an integer value";
  let description = [{
    Format an integer value as a string according to the specified format.

    See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
  }];
  let arguments = (ins
    IntType:$value,
    IntFormatAttr:$format,
    IntAlignAttr:$alignment,
    IntPaddingAttr:$padding,
    OptionalAttr<I32Attr>:$specifierWidth,
    UnitAttr:$isSigned
  );
  let results = (outs FormatStringType:$result);
  let assemblyFormat = [{
    $format $value `,` 
    `align` $alignment `,`
    `pad` $padding
    (`width` $specifierWidth^)? 
    (`signed` $isSigned^)?
    attr-dict `:` type($value)
  }];
}

def FmtFlt:I32EnumAttrCase<"Float", 0, "float">;
def FmtExp:I32EnumAttrCase<"Exponential", 1, "exponential">;
def FmtGen:I32EnumAttrCase<"General", 2, "general">;

def RealFormatAttr : I32EnumAttr<"RealFormat", "Real format",
                               [FmtFlt, FmtExp, FmtGen]> {
  let cppNamespace = "circt::moore";
}

def FormatRealOp : MooreOp<"fmt.real", [Pure]> {
  let summary = "Format a real number value";
  let description = [{
  Format a real value as a string according to the specified format.

  See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
  }];
  let arguments = (ins
    RealType:$value,
    RealFormatAttr:$format,
    IntAlignAttr:$alignment,
    OptionalAttr<I32Attr>:$fieldWidth,
    OptionalAttr<I32Attr>:$fracDigits
  );
  let results = (outs FormatStringType:$result);
  let assemblyFormat = [{
    $format $value `,`
    `align` $alignment
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` type($value)
  }];
}

def FormatStringOp : MooreOp<"fmt.string", [Pure]> {
  let summary = "A dynamic string fragment";
  let description = [{
    Creates a dynamic string fragment to be used as a format string. The
    string is printed as is, without any further escaping or processing of its
    characters.

    Use fmt.literal over this operator for any constant strings / literals.
  }];
  let arguments = (ins
    StringType:$string,
    OptionalAttr<I32Attr>:$width,
    OptionalAttr<IntAlignAttr>:$alignment,
    OptionalAttr<IntPaddingAttr>:$padding);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = [{
    $string
    (`,` `width` $width^)?
    (`,` `alignment` $alignment^)?
    (`,` `padding` $padding^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Builtin System Tasks and Functions
//===----------------------------------------------------------------------===//

class Builtin<string mnemonic, list<Trait> traits = []> :
    MooreOp<"builtin." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Conversion Builtins
//===----------------------------------------------------------------------===//

def RealtobitsBIOp : Builtin<"realtobits"> {
  let summary = "Convert a real-valued number to its logic vector representation";
  let description = [{
    Corresponds to the `$realtobits` system function. Returns a 64-bit
    logic vector corresponding to the bit representation of the real number.
    Note that this does not correspond to a cast to another type, but rather a no-op.

    See IEEE 1800-2023 § 20.5 "Conversion functions".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins RealF64:$value);
  let results = (outs TwoValuedI64:$result);
  let assemblyFormat = "$value attr-dict";
}

def BitstorealBIOp : Builtin<"bitstoreal"> {
  let summary = "Convert a logic vector representation to its real-valued number";
  let description = [{
    Corresponds to the `$bitstoreal` system function. Returns a real number
    corresponding to the real number representation of the logic vector.
    Note that this does not correspond to a cast to another type, but rather a no-op.

    See IEEE 1800-2023 § 20.5 "Conversion functions".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins TwoValuedI64:$value);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$value attr-dict`:` type($value)";
}


def ShortrealtobitsBIOp : Builtin<"shortrealtobits"> {
  let summary = "Convert a real-valued number to its logic vector representation";
  let description = [{
    Corresponds to the `$shortrealtobits` system function. Returns a 64-bit
    logic vector corresponding to the bit representation of the real number.
    Note that this does not correspond to a cast to another type, but rather a no-op.

    See IEEE 1800-2023 § 20.5 "Conversion functions".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins RealF32:$value);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$value attr-dict";
}

def BitstoshortrealBIOp : Builtin<"bitstoshortreal"> {
  let summary = "Convert a logic vector representation to its real-valued number";
  let description = [{
    Corresponds to the `$bitstoshortreal` system function. Returns a real number
    corresponding to the real number representation of the logic vector.
    Note that this does not correspond to a cast to another type, but rather a no-op.

    See IEEE 1800-2023 § 20.5 "Conversion functions".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins TwoValuedI32:$value);
  let results = (outs RealF32:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// True-Random and Pseudo-Random Generators
//===----------------------------------------------------------------------===//

def RandomBIOp : Builtin<"random"> {
  let summary = "Generate a true random signed integer (optionally seeded)";
  let description = [{
    Corresponds to the `$random` system function. Returns a 32-bit
    true random integer. The seed is optional; when provided, it initializes
    the generator. If not provided, treat it as 0 in semantics/lowering.

    `$random` is largely considered to be deprecated since it leads to
    non-reproducible simulation results. Consider using `$urandom` instead.

    See IEEE 1800-2023 § 20.14 "Probablistic distribution functions".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins Optional<TwoValuedI32>:$seed);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "($seed^)? attr-dict";
}

def UrandomBIOp : Builtin<"urandom"> {
  let summary = "Generate a pseudo-random unsigned integer (optionally seeded)";
  let description = [{
    Corresponds to the `$urandom` system function. Returns a 32-bit
    pseudo-random integer. The seed is optional; when provided, it initializes
    the generator. If not provided, treat it as 0 in semantics/lowering.

    See IEEE 1800-2023 § 18.13 "Random number system functions and methods".
  }];

  // Optional positional attribute for the seed
  let arguments = (ins Optional<TwoValuedI32>:$seed);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "($seed^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Simulation Time Measurement Builtins
//===----------------------------------------------------------------------===//

def TimeBIOp : Builtin<"time"> {
  let summary = "Return the current simulation time";
  let description = [{
    Corresponds to the `$time` system function. Returns a int-rounded 64-bit
    integer corresponding to the elapsed simulation time scaled by the
    simulation's timescale.

    See IEEE 1800-2023 § 20.3 "Simulation time system functions".
  }];

  // Optional positional attribute for the seed
  let results = (outs TimeType:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Simulation Control Builtins
//===----------------------------------------------------------------------===//

def StopBIOp : Builtin<"stop"> {
  let summary = "Suspend simulation";
  let description = [{
    Corresponds to the `$stop` system task. Causes the simulation to be
    suspended but the simulator does not exit. Printing of the optional
    diagnostic message is handled by the `finish_message` op.

    See IEEE 1800-2017 § 20.2 "Simulation control system tasks".
  }];
  let assemblyFormat = "attr-dict";
}

def FinishBIOp : Builtin<"finish"> {
  let summary = "Exit simulation";
  let description = [{
    Corresponds to the `$finish` system task. Causes the simulator to exit and
    pass control back to the host operating system. Printing of the optional
    diagnostic message is handled by the `finish_message` op.

    The exit code argument of this op is not directly accessible from Verilog,
    but is used to distinguish between the implicit `$finish` call in `$fatal`
    and an explicit `$finish` called by the user.

    See IEEE 1800-2017 § 20.2 "Simulation control system tasks".
  }];
  let arguments = (ins I8Attr:$exitCode);
  let assemblyFormat = "$exitCode attr-dict";
}

def FinishMessageBIOp : Builtin<"finish_message"> {
  let summary = "Print diagnostic message for the finish system task";
  let description = [{
    Prints the diagnostic message for `$stop`, `$finish`, `$exit`, and `$fatal`
    mandated by the SystemVerilog standard. The exact message is controlled by
    the verbosity parameter as specified in the standard:

    - The absence of this op corresponds to `$finish(0)`.
    - `moore.builtin.finish_message false` corresponds to `$finish(1)`.
    - `moore.builtin.finish_message true` corresponds to `$finish(2)`.

    The `withStats` argument controls how detailed the printed message is:

    - **false**: Print simulation time and location.
    - **true**: Print simulation time, location, and statistics about the memory
      and CPU usage of the simulator.

    See IEEE 1800-2017 § 20.2 "Simulation control system tasks".
  }];
  let arguments = (ins I1Attr:$withStats);
  let assemblyFormat = "$withStats attr-dict";
}

//===----------------------------------------------------------------------===//
// Severity and Display Builtins
//===----------------------------------------------------------------------===//

def DisplayBIOp : Builtin<"display"> {
  let summary = "Print a text message";
  let description = [{
    Prints the given format string to the standard text output of the simulator.
    In most cases this should be stdout. This corresponds to the `$display` and
    `$write` system tasks. Message formatting is handled by `moore.fmt.*` ops.

    See IEEE 1800-2017 § 21.2 "Display system tasks".
  }];
  let arguments = (ins FormatStringType:$message);
  let assemblyFormat = "$message attr-dict";
}

def SeverityInfo : I32EnumAttrCase<"Info", 0, "info">;
def SeverityWarning : I32EnumAttrCase<"Warning", 1, "warning">;
def SeverityError : I32EnumAttrCase<"Error", 2, "error">;
def SeverityFatal : I32EnumAttrCase<"Fatal", 3, "fatal">;
def SeverityAttr : I32EnumAttr<"Severity", "Diagnostic severity", [
  SeverityInfo, SeverityWarning, SeverityError, SeverityFatal
]> {
  let cppNamespace = "circt::moore";
}

def SeverityBIOp : Builtin<"severity"> {
  let summary = "Print a diagnostic message";
  let description = [{
    Prints the given format string to the standard diagnostic output of the
    simulator. In most cases this should be stderr. This corresponds to the
    `$info`, `$warning`, `$error`, and `$fatal` system tasks. Message formatting
    is handled by `moore.fmt.*` ops. This only handles the message printing of
    `$fatal`; printing of the additional statistics and the call to `$finish`
    must be done through the `finish_message` and `finish` ops.

    See IEEE 1800-2017 § 20.10 "Severity tasks".
  }];
  let arguments = (ins
    SeverityAttr:$severity,
    FormatStringType:$message
  );
  let assemblyFormat = "$severity $message attr-dict";
}

//===----------------------------------------------------------------------===//
// Math Builtins
//===----------------------------------------------------------------------===//

class RealMathFunc<string mnemonic, list<Trait> traits = []> : 
    Builtin<mnemonic, traits # [SameOperandsAndResultType]> {
  let description = [{
    The system real math functions shall accept real value arguments and return
    a real result type. Their behavior shall match the equivalent C language
    standard math library function indicated.

    See IEEE 1800-2017 § 20.8.2 "Real math functions".
   }];
  let arguments = (ins RealF64:$value);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def Clog2BIOp : Builtin<"clog2", [SameOperandsAndResultType]> {
  let summary = "Compute ceil(log2(x)) of x";
  let description = [{
    Computes the ceiling of the base-2 logarithm of the argument. The argument
    is interpreted as unsigned. The result is 0 if the argument is 0. The result
    corresponds to the minimum address width necessary to address a given number
    of elements, or the number of bits necessary to represent a given number of
    states.

    If any of the bits in the argument are X or Z, the result is X.

    See IEEE 1800-2017 § 20.8.1 "Integer math functions".
  }];
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def LnBIOp : RealMathFunc<"ln"> {
  let summary = "Natural logarithm";
}

def Log10BIOp : RealMathFunc<"log10"> {
  let summary = "Decimal logarithm";
}

def ExpBIOp : RealMathFunc<"exp"> {
  let summary = "Exponential";
}

def SqrtBIOp : RealMathFunc<"sqrt"> {
  let summary = "Square root";
}

def FloorBIOp : RealMathFunc<"floor"> {
  let summary = "Floor";
}

def CeilBIOp : RealMathFunc<"ceil"> {
  let summary = "Ceiling";
}

def SinBIOp : RealMathFunc<"sin"> {
  let summary = "Sine";
}

def CosBIOp : RealMathFunc<"cos"> {
  let summary = "Cosine";
}

def TanBIOp : RealMathFunc<"tan"> {
  let summary = "Tangent";
}

def AsinBIOp : RealMathFunc<"asin"> {
  let summary = "Arc-sine";
}

def AcosBIOp : RealMathFunc<"acos"> {
  let summary = "Arc-cosine";
}

def AtanBIOp : RealMathFunc<"atan"> {
  let summary = "Arc-tangent";
}

def SinhBIOp : RealMathFunc<"sinh"> {
  let summary = "Hyperbolic sine";
}

def CoshBIOp : RealMathFunc<"cosh"> {
  let summary = "Hyperbolic cosine";
}

def TanhBIOp : RealMathFunc<"tanh"> {
  let summary = "Hyperbolic tangent";
}

def AsinhBIOp : RealMathFunc<"asinh"> {
  let summary = "Arc-hyperbolic sine";
}

def AcoshBIOp : RealMathFunc<"acosh"> {
  let summary = "Arc-hyperbolic cosine";
}

def AtanhBIOp : RealMathFunc<"atanh"> {
  let summary = "Arc-hyperbolic tangent";
}

//===----------------------------------------------------------------------===//
// Classes
//===----------------------------------------------------------------------===//

// Access visibility for class members (properties and methods)
// Named "MemberAccess" to avoid conflict with MLIR's Symbol visibility
def MemberAccessPublic: I32EnumAttrCase<"Public", 0, "public">;
def MemberAccessProtected: I32EnumAttrCase<"Protected", 1, "protected">;
def MemberAccessLocal: I32EnumAttrCase<"Local", 2, "local">;

def MemberAccessAttr: I32EnumAttr<"MemberAccess", "Member access visibility",
            [MemberAccessPublic, MemberAccessProtected, MemberAccessLocal]>{
  let cppNamespace = "circt::moore";
}

// Randomization mode for class properties and struct fields
def RandModeNone: I32EnumAttrCase<"None", 0, "none">;
def RandModeRand: I32EnumAttrCase<"Rand", 1, "rand">;
def RandModeRandC: I32EnumAttrCase<"RandC", 2, "randc">;

def RandModeAttr: I32EnumAttr<"RandMode", "Randomization mode",
            [RandModeNone, RandModeRand, RandModeRandC]>{
  let cppNamespace = "circt::moore";
}

def ClassPropertyDeclOp
    : MooreOp<"class.propertydecl", [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class property";
  let description = [{
  Declares a property within a class declaration. The `member_access` attribute
  specifies the access level:
  - `public`: Accessible from anywhere (default)
  - `protected`: Accessible from this class and derived classes
  - `local`: Accessible only from within this class

  The optional `rand_mode` attribute indicates whether this property
  participates in randomization:
  - `none`: Not randomizable (default)
  - `rand`: Standard random variable
  - `randc`: Random-cyclic variable (cycles through all values before repeating)
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type,
                       DefaultValuedAttr<MemberAccessAttr, "MemberAccess::Public">:$member_access,
                       DefaultValuedAttr<RandModeAttr, "RandMode::None">:$rand_mode);

  let results = (outs);
  let assemblyFormat = [{
     $sym_name `:` $type (`rand_mode` $rand_mode^)? attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Type getPropertyType() { return getTypeAttr().getValue(); }
    bool isRandomizable() { return getRandMode() != RandMode::None; }
    bool isMemberAccessPublic() { return getMemberAccess() == MemberAccess::Public; }
    bool isMemberAccessProtected() { return getMemberAccess() == MemberAccess::Protected; }
    bool isMemberAccessLocal() { return getMemberAccess() == MemberAccess::Local; }
  }];
}


def ClassMethodDeclOp
    : MooreOp<"class.methoddecl", [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class method";
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<SymbolRefAttr>:$impl);
  let results = (outs);
  let assemblyFormat = [{
      $sym_name (`->` $impl^)? `:` $function_type attr-dict
  }];
}

def ClassDeclOp
    : MooreOp<"class.classdecl", [Symbol, SymbolTable, IsolatedFromAbove,
    NoTerminator, SingleBlock]> {
  let summary = "Class declaration";
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<SymbolRefAttr>:$base,
      OptionalAttr<SymbolRefArrayAttr>:$implementedInterfaces);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name
    (`extends` $base^)?
    (`implements` $implementedInterfaces^)?
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def ConstraintBlockOp
    : MooreOp<"constraint.block", [Symbol, HasParent<"ClassDeclOp">,
    NoTerminator, SingleBlock]> {
  let summary = "Define a constraint block";
  let description = [{
    Represents a named constraint block within a class. Constraint blocks
    define relationships between randomizable variables that must be satisfied
    during randomization.

    Example:
    ```mlir
    moore.constraint.block @valid_range {
      // constraint expressions go here
    }
    ```

    See IEEE 1800-2017 Section 18.5 "Constraint blocks".
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       UnitAttr:$is_static,
                       UnitAttr:$is_pure);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`static` $is_static^)? (`pure` $is_pure^)? $sym_name
    attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// Constraint Expression Operations
//===----------------------------------------------------------------------===//

def ConstraintExprOp
    : MooreOp<"constraint.expr", []> {
  let summary = "A constraint expression";
  let description = [{
    Represents a constraint expression that must hold during randomization.
    The expression evaluates to a boolean condition that constrains the
    values of random variables.

    Example:
    ```mlir
    moore.constraint.block @c1 {
      %cond = moore.lt %a, %b : i32
      moore.constraint.expr %cond
    }
    ```

    See IEEE 1800-2017 Section 18.5.1 "Constraint expressions".
  }];

  let arguments = (ins I1:$condition);
  let results = (outs);
  let assemblyFormat = [{
    $condition attr-dict
  }];
}

def ConstraintImplicationOp
    : MooreOp<"constraint.implication", [SingleBlock, NoTerminator]> {
  let summary = "Constraint implication (expr -> constraint)";
  let description = [{
    Represents a constraint implication where if the antecedent expression
    is true, the constraints in the consequent block must be satisfied.
    If the antecedent is false, the consequent constraints are ignored.

    Example:
    ```mlir
    moore.constraint.block @c1 {
      moore.constraint.implication %mode_is_fast {
        // constraints that only apply when mode_is_fast is true
        moore.constraint.expr %speed_limit
      }
    }
    ```

    See IEEE 1800-2017 Section 18.5.6 "Implication constraints".
  }];

  let arguments = (ins I1:$antecedent);
  let results = (outs);
  let regions = (region SizedRegion<1>:$consequent);
  let assemblyFormat = [{
    $antecedent attr-dict-with-keyword $consequent
  }];
}

def ConstraintIfElseOp
    : MooreOp<"constraint.if_else", [SingleBlock, NoTerminator]> {
  let summary = "Conditional constraint (if-else)";
  let description = [{
    Represents a conditional constraint where different constraints apply
    based on a condition. If the condition is true, the constraints in
    the then_region apply; otherwise, the constraints in the else_region
    (if present) apply.

    Example:
    ```mlir
    moore.constraint.block @c1 {
      moore.constraint.if_else %is_write {
        // constraints for write mode
      } else {
        // constraints for read mode
      }
    }
    ```

    See IEEE 1800-2017 Section 18.5.7 "if-else constraints".
  }];

  let arguments = (ins I1:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$then_region, AnyRegion:$else_region);
  let assemblyFormat = [{
    $condition $then_region (`else` $else_region^)? attr-dict
  }];
}

def ConstraintForeachOp
    : MooreOp<"constraint.foreach", [SingleBlock, NoTerminator]> {
  let summary = "Foreach constraint over array elements";
  let description = [{
    Represents a foreach constraint that iterates over array elements
    and applies constraints involving each element. The loop variable
    is passed as a block argument.

    Example:
    ```mlir
    moore.constraint.block @c1 {
      moore.constraint.foreach %array : !moore.uarray<8 x i32> {
      ^bb0(%i: index, %elem: !moore.i32):
        // constraints involving %elem
      }
    }
    ```

    See IEEE 1800-2017 Section 18.5.8 "foreach constraints".
  }];

  let arguments = (ins AnyType:$array);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $array `:` type($array) $body attr-dict
  }];
}

def ConstraintDistOp
    : MooreOp<"constraint.dist", []> {
  let summary = "Distribution constraint";
  let description = [{
    Represents a distribution constraint that specifies weighted probability
    distributions for random variable values.

    The values array contains the distribution points: single values are
    represented once, ranges as [low, high] pairs. The range_markers array
    indicates which entries are range starts (1) vs single values (0).
    The weights array provides the relative probability weight for each entry.
    The per_range array indicates whether each weight applies per-item (0)
    or is divided across the range (1 = :/).

    Example:
    ```mlir
    // x dist { 0 := 10, [1:5] :/ 50, 6 := 40 }
    moore.constraint.dist %x, [0, 1, 5, 6], [10, 50, 40], [0, 1, 0] : i8
    // values: 0, range [1,5], 6
    // weights: 10, 50, 40
    // per_range: false, true, false (0=:=, 1=:/)
    ```

    See IEEE 1800-2017 Section 18.5.4 "Distribution constraints".
  }];

  let arguments = (ins AnyType:$variable,
                       DenseI64ArrayAttr:$values,
                       DenseI64ArrayAttr:$weights,
                       DenseI64ArrayAttr:$per_range);
  let results = (outs);
  let assemblyFormat = [{
    $variable `,` $values `,` $weights `,` $per_range `:` type($variable) attr-dict
  }];
  let hasVerifier = 1;
}

def ConstraintInsideOp
    : MooreOp<"constraint.inside", []> {
  let summary = "Inside constraint (set membership)";
  let description = [{
    Represents an inside constraint that requires a variable to be
    a member of a set of values or ranges.

    The ranges array contains pairs of [low, high] values. Single values
    are represented as [value, value].

    Example:
    ```mlir
    // x inside { 1, [3:5], 7 }
    moore.constraint.inside %x, [[1, 1], [3, 5], [7, 7]] : i8
    ```

    See IEEE 1800-2017 Section 18.5.3 "Set membership constraints".
  }];

  let arguments = (ins AnyType:$variable,
                       DenseI64ArrayAttr:$ranges);
  let results = (outs);
  let assemblyFormat = [{
    $variable `,` $ranges `:` type($variable) attr-dict
  }];
  let hasVerifier = 1;
}

def ConstraintSolveBeforeOp
    : MooreOp<"constraint.solve_before", []> {
  let summary = "Solve ordering constraint";
  let description = [{
    Specifies that certain random variables should be solved before others.
    This provides guidance to the constraint solver about the order in which
    to determine variable values.

    Example:
    ```mlir
    // solve mode before data, addr
    moore.constraint.solve_before [@mode], [@data, @addr]
    ```

    See IEEE 1800-2017 Section 18.5.10 "Constraint ordering".
  }];

  let arguments = (ins SymbolRefArrayAttr:$before,
                       SymbolRefArrayAttr:$after);
  let results = (outs);
  let assemblyFormat = [{
    $before `,` $after attr-dict
  }];
}

def ConstraintDisableOp
    : MooreOp<"constraint.disable", []> {
  let summary = "Disable soft constraint";
  let description = [{
    Disables a soft constraint by name. This allows overriding soft constraints
    defined in base classes or earlier in the constraint block.

    Example:
    ```mlir
    moore.constraint.disable @soft_limit
    ```

    See IEEE 1800-2017 Section 18.5.13 "Soft constraints".
  }];

  let arguments = (ins SymbolRefAttr:$constraint_name);
  let results = (outs);
  let assemblyFormat = [{
    $constraint_name attr-dict
  }];
}

def ConstraintUniqueOp
    : MooreOp<"constraint.unique", []> {
  let summary = "Unique constraint for array elements";
  let description = [{
    Specifies that all elements of an array must have unique values
    after randomization.

    Example:
    ```mlir
    // unique { arr }
    moore.constraint.unique %arr : !moore.uarray<8 x i32>
    ```

    See IEEE 1800-2017 Section 18.5.5 "Uniqueness constraints".
  }];

  let arguments = (ins Variadic<AnyType>:$variables);
  let results = (outs);
  let assemblyFormat = [{
    $variables `:` type($variables) attr-dict
  }];
}

def RandomizeOp
    : MooreOp<"randomize", []> {
  let summary = "Randomize an object";
  let description = [{
    Invokes the randomize() method on a class object. This operation
    assigns random values to all `rand` and `randc` properties while
    satisfying all active constraints.

    The result is a 1-bit value indicating success (1) or failure (0).

    Example:
    ```mlir
    %success = moore.randomize %obj : !moore.class.object<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.6 "Randomization methods".
  }];

  let arguments = (ins ClassHandleType:$object);
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $object `:` type($object) attr-dict
  }];
}

def ClassNewOp
    : MooreOp<
          "class.new", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Allocate a new class instance";
  let description = [{
    Allocates a new instance of class @C. This op does not call the constructor.
    The result is a non-null `!moore.class.object<@C>` handle.
  }];

  let arguments = (ins);
  let results = (outs ClassHandleType:$result);

  let assemblyFormat = [{
     `:` type($result) attr-dict
  }];
  let hasVerifier = 1;
}


def ClassPropertyRefOp
    : MooreOp<"class.property_ref", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,]> {
  let summary = "Get a !moore.ref to a class property";
  let description = [{
    Construct a reference to a class instance's property field. Translates
    to a fixed offset from the the class handle into its data struct.
  }];
  let arguments = (ins ClassHandleType:$instance,
      FlatSymbolRefAttr:$property);
  let results = (outs RefType:$propertyRef);
  let assemblyFormat = "$instance`[`$property`]` `:` type($instance) `->` "
                       "type($propertyRef) attr-dict";
}

def ClassUpcastOp
    : MooreOp<"class.upcast", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,]> {
  let summary = "Upcast a derived handle to a base handle (zero-cost)";
  let description = [{
    Casts an instance of a class to one of its base classes.
    The created instance points to the same underlying data struct as the original
    instance, and modifying it will modify the spawning instance's data as well.
  }];
  let arguments = (ins ClassHandleType:$instance);
  let results = (outs ClassHandleType:$result);
  let assemblyFormat =
      "$instance `:` type($instance) `to` type($result) attr-dict";
}

def VTableLoadMethodOp
    : MooreOp<"vtable.load_method", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Load a virtual method entry from a vtable.";
  let description = [{
    Loads a virtual method function pointer from a vtable.
    The verifier resolves the symbols in the symbol table and ensures that the
    result function type matches the erased ABI.
  }];

  let arguments = (ins
    ClassHandleType:$object,
    SymbolRefAttr:$methodSym
  );

  let results = (outs
      Type<CPred<"::llvm::isa<mlir::FunctionType>($_self)">>:$result
  );

  let assemblyFormat =
    "$object `:` $methodSym `of` type($object) `->` type($result) attr-dict";
}

def VTableOp : MooreOp<"vtable",
    [NoTerminator, SingleBlock, IsolatedFromAbove,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Virtual dispatch table";
  let description = [{
    Lives at module top-level. Contains `moore.vtable_entry` and nested
    `moore.vtable` ops for base segments. Holds the dispatch targets for every
    possible virtual method call of a class and its ancestors.
  }];
  let arguments = (ins
    SymbolRefAttr:$sym_name
  );

  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
  let hasRegionVerifier = 1;
}

def VTableEntryOp : MooreOp<"vtable_entry", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "One vtable slot pointing to the selected implementation";
  let description = [{
    Each entry references the resolved implementation with a SymbolRefAttr.
  }];

  let arguments = (ins
    SymbolNameAttr:$name,
    SymbolRefAttr:$target
  );

  let assemblyFormat = [{
    $name `->` $target attr-dict
  }];
}


//===----------------------------------------------------------------------===//
// Interfaces and Virtual Interfaces
//===----------------------------------------------------------------------===//

def InterfaceDeclOp : MooreOp<"interface", [
  Symbol,
  SymbolTable,
  IsolatedFromAbove,
  NoTerminator,
  SingleBlock
]> {
  let summary = "Interface declaration";
  let description = [{
    Declares a SystemVerilog interface. Interfaces are bundles of signals and
    modports that provide a standardized way to connect modules. They are
    essential for UVM testbenches where they define the communication protocol
    between the DUT and verification components.

    Example:
    ```mlir
    moore.interface @my_bus {
      moore.interface.signal @clk : !moore.l1
      moore.interface.signal @data : !moore.l32
      moore.interface.signal @valid : !moore.l1
      moore.interface.modport @driver (output @clk, output @data, output @valid)
      moore.interface.modport @monitor (input @clk, input @data, input @valid)
    }
    ```

    See IEEE 1800-2017 Section 25 "Interfaces".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

def InterfaceSignalDeclOp : MooreOp<"interface.signal", [
  Symbol,
  HasParent<"InterfaceDeclOp">
]> {
  let summary = "Declare a signal within an interface";
  let description = [{
    Declares a signal within an interface. Signals define the wires and
    registers that make up the interface's communication channels.

    Example:
    ```mlir
    moore.interface.signal @data : !moore.l32
    moore.interface.signal @valid : !moore.l1
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );
  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Type getSignalType() { return getTypeAttr().getValue(); }
  }];
}

def ModportDeclOp : MooreOp<"interface.modport", [
  Symbol,
  HasParent<"InterfaceDeclOp">
]> {
  let summary = "Declare a modport within an interface";
  let description = [{
    Declares a modport within an interface. Modports define directional views
    of an interface's signals, specifying which signals are inputs, outputs,
    or bidirectional from a particular perspective.

    Modports are crucial for UVM as they allow different verification
    components (drivers, monitors) to have appropriate access to interface
    signals.

    Example:
    ```mlir
    moore.interface.modport @driver (output @clk, output @data, input @ready)
    moore.interface.modport @monitor (input @clk, input @data, input @ready)
    ```

    See IEEE 1800-2017 Section 25.4 "Modports".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ModportPortArrayAttr:$ports
  );
  let assemblyFormat = [{
    $sym_name custom<ModportPorts>($ports) attr-dict
  }];
}

def InterfaceInstanceOp : MooreOp<"interface.instance", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Instantiate an interface";
  let description = [{
    Creates an instance of an interface. Interface instances are used to
    connect modules and provide the actual signals that virtual interfaces
    can reference.

    Example:
    ```mlir
    %bus = moore.interface.instance @my_bus : !moore.ref<!moore.virtual_interface<@my_bus>>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    FlatSymbolRefAttr:$interfaceName
  );
  let results = (outs VirtualInterfaceRefType:$result);
  let assemblyFormat = [{
    custom<ImplicitSSAName>($name) $interfaceName attr-dict `:` type($result)
  }];
}

def VirtualInterfaceGetOp : MooreOp<"virtual_interface.get", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Get a modport view from a virtual interface";
  let description = [{
    Extracts a modport view from a virtual interface. This is used when a
    UVM component needs to access interface signals through a specific
    modport perspective.

    Example:
    ```mlir
    %driver_view = moore.virtual_interface.get %vif @driver :
      !moore.virtual_interface<@my_bus> -> !moore.virtual_interface<@my_bus::@driver>
    ```
  }];

  let arguments = (ins
    VirtualInterfaceType:$vif,
    FlatSymbolRefAttr:$modport
  );
  let results = (outs VirtualInterfaceType:$result);
  let assemblyFormat = [{
    $vif $modport attr-dict `:` type($vif) `->` type($result)
  }];
}

def VirtualInterfaceSignalRefOp : MooreOp<"virtual_interface.signal_ref", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Get a reference to a signal in a virtual interface";
  let description = [{
    Returns a reference to a signal within a virtual interface. This allows
    reading from and writing to interface signals through the virtual
    interface handle.

    Example:
    ```mlir
    %data_ref = moore.virtual_interface.signal_ref %vif @data :
      !moore.virtual_interface<@my_bus> -> !moore.ref<!moore.l32>
    ```
  }];

  let arguments = (ins
    VirtualInterfaceType:$vif,
    FlatSymbolRefAttr:$signal
  );
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $vif `[` $signal `]` attr-dict `:` type($vif) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// UVM Factory Support
//===----------------------------------------------------------------------===//

// Base class for UVM operations
class UVMOp<string mnemonic, list<Trait> traits = []> :
    MooreOp<"uvm." # mnemonic, traits>;

// UVM Component/Object type kind
def UVMObjectKind: I32EnumAttrCase<"Object", 0, "object">;
def UVMComponentKind: I32EnumAttrCase<"Component", 1, "component">;

def UVMTypeKindAttr: I32EnumAttr<"UVMTypeKind", "UVM type kind",
            [UVMObjectKind, UVMComponentKind]>{
  let cppNamespace = "circt::moore";
}

def UVMTypeRegistrationOp : UVMOp<"type_register", [
  Symbol,
  IsolatedFromAbove,
  NoTerminator,
  SingleBlock
]> {
  let summary = "Register a class type with the UVM factory";
  let description = [{
    Registers a class type with the UVM factory system. This is equivalent
    to the `uvm_component_utils or `uvm_object_utils macros in SystemVerilog.

    The factory registration enables dynamic object creation through
    type names and supports type overrides for verification reuse.

    Example:
    ```mlir
    moore.uvm.type_register @MyComponent component {
      // Optional: field registration for automation
      moore.uvm.field_int @addr, UVM_DEFAULT
      moore.uvm.field_object @child, UVM_DEFAULT
    }
    ```

    See UVM 1.2 Reference Manual Section 8.2 "Factory Registration".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$class_type,
    UVMTypeKindAttr:$kind
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name $class_type $kind attr-dict-with-keyword $body
  }];
}

def UVMFieldIntOp : UVMOp<"field_int", [HasParent<"UVMTypeRegistrationOp">]> {
  let summary = "Register an integer field for UVM automation";
  let description = [{
    Registers an integer field for UVM automation. This enables automatic
    field operations like copy, compare, pack, unpack, print, and record.

    Example:
    ```mlir
    moore.uvm.field_int @addr, UVM_DEFAULT
    moore.uvm.field_int @data, UVM_NOPRINT | UVM_NOCOPY
    ```

    See UVM 1.2 Reference Manual Section 8.3 "Field Macros".
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$field_name,
    I32Attr:$flags
  );
  let assemblyFormat = [{
    $field_name `,` $flags attr-dict
  }];
}

def UVMFieldObjectOp : UVMOp<"field_object", [HasParent<"UVMTypeRegistrationOp">]> {
  let summary = "Register an object field for UVM automation";
  let description = [{
    Registers an object (class handle) field for UVM automation.

    Example:
    ```mlir
    moore.uvm.field_object @child, UVM_DEFAULT
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$field_name,
    I32Attr:$flags
  );
  let assemblyFormat = [{
    $field_name `,` $flags attr-dict
  }];
}

def UVMFieldStringOp : UVMOp<"field_string", [HasParent<"UVMTypeRegistrationOp">]> {
  let summary = "Register a string field for UVM automation";
  let description = [{
    Registers a string field for UVM automation.

    Example:
    ```mlir
    moore.uvm.field_string @name, UVM_DEFAULT
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$field_name,
    I32Attr:$flags
  );
  let assemblyFormat = [{
    $field_name `,` $flags attr-dict
  }];
}

def UVMFieldArrayIntOp : UVMOp<"field_array_int", [HasParent<"UVMTypeRegistrationOp">]> {
  let summary = "Register an integer array field for UVM automation";
  let description = [{
    Registers an integer array field for UVM automation.

    Example:
    ```mlir
    moore.uvm.field_array_int @data_array, UVM_DEFAULT
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$field_name,
    I32Attr:$flags
  );
  let assemblyFormat = [{
    $field_name `,` $flags attr-dict
  }];
}

def UVMObjectCreateOp : UVMOp<"create_object"> {
  let summary = "Create an object via the UVM factory";
  let description = [{
    Creates an object using the UVM factory. This is equivalent to
    factory.create_object_by_type() or type::type_id::create() in
    SystemVerilog UVM code.

    The factory lookup allows type overrides to substitute derived types
    at runtime without changing the calling code.

    Example:
    ```mlir
    %obj = moore.uvm.create_object "my_transaction" : !moore.class<@my_transaction>
    ```

    See UVM 1.2 Reference Manual Section 8.3 "Factory Methods".
  }];

  let arguments = (ins
    StrAttr:$type_name,
    OptionalAttr<StrAttr>:$inst_name,
    Optional<ClassHandleType>:$parent
  );
  let results = (outs ClassHandleType:$result);
  let assemblyFormat = [{
    $type_name (`,` $inst_name^)? (`parent` $parent^ `:` type($parent))?
    attr-dict `:` type($result)
  }];
}

def UVMComponentCreateOp : UVMOp<"create_component"> {
  let summary = "Create a component via the UVM factory";
  let description = [{
    Creates a component using the UVM factory. This is equivalent to
    factory.create_component_by_type() or type::type_id::create() in
    SystemVerilog UVM code.

    Components require a name and parent for the UVM hierarchy.

    Example:
    ```mlir
    %comp = moore.uvm.create_component "my_driver", "drv0", %parent :
      !moore.class<@my_driver>
    ```

    See UVM 1.2 Reference Manual Section 8.3 "Factory Methods".
  }];

  let arguments = (ins
    StrAttr:$type_name,
    StrAttr:$inst_name,
    ClassHandleType:$parent
  );
  let results = (outs ClassHandleType:$result);
  let assemblyFormat = [{
    $type_name `,` $inst_name `,` $parent `:` type($parent) attr-dict `->` type($result)
  }];
}

def UVMTypeOverrideOp : UVMOp<"type_override"> {
  let summary = "Override a type in the UVM factory";
  let description = [{
    Registers a type override in the UVM factory. When the original type
    is requested through create_object or create_component, the override
    type will be created instead.

    Example:
    ```mlir
    moore.uvm.type_override "base_driver", "extended_driver"
    ```

    See UVM 1.2 Reference Manual Section 8.3.2 "Type Overrides".
  }];

  let arguments = (ins
    StrAttr:$original_type,
    StrAttr:$override_type,
    UnitAttr:$replace
  );
  let assemblyFormat = [{
    $original_type `->` $override_type (`replace` $replace^)? attr-dict
  }];
}

def UVMInstanceOverrideOp : UVMOp<"instance_override"> {
  let summary = "Override a type for a specific instance path";
  let description = [{
    Registers an instance-specific type override in the UVM factory.
    Only instances matching the given path will use the override type.

    Example:
    ```mlir
    moore.uvm.instance_override "base_driver", "debug_driver", "env.agent.driver"
    ```

    See UVM 1.2 Reference Manual Section 8.3.2 "Instance Overrides".
  }];

  let arguments = (ins
    StrAttr:$original_type,
    StrAttr:$override_type,
    StrAttr:$inst_path
  );
  let assemblyFormat = [{
    $original_type `->` $override_type `at` $inst_path attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UVM Phase Support
//===----------------------------------------------------------------------===//

// UVM phase enumeration
def UVMBuildPhase: I32EnumAttrCase<"Build", 0, "build">;
def UVMConnectPhase: I32EnumAttrCase<"Connect", 1, "connect">;
def UVMEndOfElaborationPhase: I32EnumAttrCase<"EndOfElaboration", 2, "end_of_elaboration">;
def UVMStartOfSimulationPhase: I32EnumAttrCase<"StartOfSimulation", 3, "start_of_simulation">;
def UVMRunPhase: I32EnumAttrCase<"Run", 4, "run">;
def UVMExtractPhase: I32EnumAttrCase<"Extract", 5, "extract">;
def UVMCheckPhase: I32EnumAttrCase<"Check", 6, "check">;
def UVMReportPhase: I32EnumAttrCase<"Report", 7, "report">;
def UVMFinalPhase: I32EnumAttrCase<"Final", 8, "final">;
// Run-time sub-phases
def UVMPreResetPhase: I32EnumAttrCase<"PreReset", 9, "pre_reset">;
def UVMResetPhase: I32EnumAttrCase<"Reset", 10, "reset">;
def UVMPostResetPhase: I32EnumAttrCase<"PostReset", 11, "post_reset">;
def UVMPreConfigurePhase: I32EnumAttrCase<"PreConfigure", 12, "pre_configure">;
def UVMConfigurePhase: I32EnumAttrCase<"Configure", 13, "configure">;
def UVMPostConfigurePhase: I32EnumAttrCase<"PostConfigure", 14, "post_configure">;
def UVMPreMainPhase: I32EnumAttrCase<"PreMain", 15, "pre_main">;
def UVMMainPhase: I32EnumAttrCase<"Main", 16, "main">;
def UVMPostMainPhase: I32EnumAttrCase<"PostMain", 17, "post_main">;
def UVMPreShutdownPhase: I32EnumAttrCase<"PreShutdown", 18, "pre_shutdown">;
def UVMShutdownPhase: I32EnumAttrCase<"Shutdown", 19, "shutdown">;
def UVMPostShutdownPhase: I32EnumAttrCase<"PostShutdown", 20, "post_shutdown">;

def UVMPhaseAttr: I32EnumAttr<"UVMPhase", "UVM phase",
            [UVMBuildPhase, UVMConnectPhase, UVMEndOfElaborationPhase,
             UVMStartOfSimulationPhase, UVMRunPhase, UVMExtractPhase,
             UVMCheckPhase, UVMReportPhase, UVMFinalPhase,
             UVMPreResetPhase, UVMResetPhase, UVMPostResetPhase,
             UVMPreConfigurePhase, UVMConfigurePhase, UVMPostConfigurePhase,
             UVMPreMainPhase, UVMMainPhase, UVMPostMainPhase,
             UVMPreShutdownPhase, UVMShutdownPhase, UVMPostShutdownPhase]>{
  let cppNamespace = "circt::moore";
}

def UVMPhaseMethodOp : UVMOp<"phase_method", [
  HasParent<"ClassDeclOp">,
  NoTerminator,
  SingleBlock
]> {
  let summary = "Declare a UVM phase method";
  let description = [{
    Declares a UVM phase method implementation within a component class.
    Phase methods are called by the UVM phasing mechanism during simulation.

    Example:
    ```mlir
    moore.uvm.phase_method build %phase {
      // build_phase implementation
    }
    ```

    See UVM 1.2 Reference Manual Chapter 9 "Phasing".
  }];

  let arguments = (ins
    UVMPhaseAttr:$phase
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $phase attr-dict-with-keyword $body
  }];
}

def UVMRaiseObjectionOp : UVMOp<"raise_objection"> {
  let summary = "Raise an objection to phase advancement";
  let description = [{
    Raises an objection to prevent the current phase from ending.
    The phase will not complete until all objections are dropped.

    Example:
    ```mlir
    moore.uvm.raise_objection %phase
    ```

    See UVM 1.2 Reference Manual Section 9.3 "Objections".
  }];

  let arguments = (ins
    ClassHandleType:$phase,
    Optional<TwoValuedI32>:$count
  );
  let assemblyFormat = [{
    $phase (`,` $count^)? `:` type($phase) attr-dict
  }];
}

def UVMDropObjectionOp : UVMOp<"drop_objection"> {
  let summary = "Drop an objection to phase advancement";
  let description = [{
    Drops a previously raised objection, potentially allowing the phase
    to complete if all objections are dropped.

    Example:
    ```mlir
    moore.uvm.drop_objection %phase
    ```

    See UVM 1.2 Reference Manual Section 9.3 "Objections".
  }];

  let arguments = (ins
    ClassHandleType:$phase,
    Optional<TwoValuedI32>:$count
  );
  let assemblyFormat = [{
    $phase (`,` $count^)? `:` type($phase) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database
//===----------------------------------------------------------------------===//

def UVMConfigDbSetOp : UVMOp<"config_db.set"> {
  let summary = "Set a value in the UVM configuration database";
  let description = [{
    Sets a value in the UVM configuration database. The value can be
    retrieved by components matching the instance path using config_db.get.

    Example:
    ```mlir
    moore.uvm.config_db.set %null, "*.driver", "vif", %vif : !moore.virtual_interface<@my_if>
    ```

    See UVM 1.2 Reference Manual Section 7.3 "Configuration Database".
  }];

  let arguments = (ins
    Optional<ClassHandleType>:$context,
    StrAttr:$inst_name,
    StrAttr:$field_name,
    AnyType:$value
  );
  let assemblyFormat = [{
    ($context^ `:` type($context) `,`)? $inst_name `,` $field_name `,`
    $value `:` type($value) attr-dict
  }];
}

def UVMConfigDbGetOp : UVMOp<"config_db.get"> {
  let summary = "Get a value from the UVM configuration database";
  let description = [{
    Retrieves a value from the UVM configuration database. Returns true
    if the value was found, false otherwise.

    Example:
    ```mlir
    %found, %vif = moore.uvm.config_db.get %this, "", "vif" :
      !moore.class<@my_component> -> !moore.virtual_interface<@my_if>
    ```

    See UVM 1.2 Reference Manual Section 7.3 "Configuration Database".
  }];

  let arguments = (ins
    ClassHandleType:$context,
    StrAttr:$inst_name,
    StrAttr:$field_name
  );
  let results = (outs I1:$found, AnyType:$value);
  let assemblyFormat = [{
    $context `:` type($context) `,` $inst_name `,` $field_name attr-dict `->` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// UVM Sequence and TLM Support
//===----------------------------------------------------------------------===//

def UVMSequenceStartOp : UVMOp<"sequence.start"> {
  let summary = "Start a sequence on a sequencer";
  let description = [{
    Starts a UVM sequence on a sequencer. This is equivalent to
    sequence.start(sequencer) in SystemVerilog.

    The operation is a blocking call that completes when the sequence
    finishes executing.

    Example:
    ```mlir
    moore.uvm.sequence.start %seq, %sqr : !moore.class<@my_sequence>,
      !moore.class<@my_sequencer>
    ```

    See UVM 1.2 Reference Manual Section 5.4 "Sequence Execution".
  }];

  let arguments = (ins
    ClassHandleType:$sequence,
    ClassHandleType:$sequencer,
    Optional<ClassHandleType>:$parent_sequence
  );
  let assemblyFormat = [{
    $sequence `,` $sequencer (`,` `parent` $parent_sequence^ `:` type($parent_sequence))?
    `:` type($sequence) `,` type($sequencer) attr-dict
  }];
}

def UVMSequenceItemStartOp : UVMOp<"seq_item.start"> {
  let summary = "Start item phase for a sequence item";
  let description = [{
    Begins the start_item phase of sequence item execution.
    This requests arbitration from the sequencer.

    Example:
    ```mlir
    moore.uvm.seq_item.start %item, %sqr : !moore.class<@my_item>,
      !moore.class<@my_sequencer>
    ```

    See UVM 1.2 Reference Manual Section 5.5 "Sequence Items".
  }];

  let arguments = (ins
    ClassHandleType:$item,
    ClassHandleType:$sequencer
  );
  let assemblyFormat = [{
    $item `,` $sequencer `:` type($item) `,` type($sequencer) attr-dict
  }];
}

def UVMSequenceItemFinishOp : UVMOp<"seq_item.finish"> {
  let summary = "Finish item phase for a sequence item";
  let description = [{
    Completes the finish_item phase of sequence item execution.
    This sends the item to the driver and waits for the response.

    Example:
    ```mlir
    moore.uvm.seq_item.finish %item, %sqr : !moore.class<@my_item>,
      !moore.class<@my_sequencer>
    ```

    See UVM 1.2 Reference Manual Section 5.5 "Sequence Items".
  }];

  let arguments = (ins
    ClassHandleType:$item,
    ClassHandleType:$sequencer
  );
  let assemblyFormat = [{
    $item `,` $sequencer `:` type($item) `,` type($sequencer) attr-dict
  }];
}

def UVMGetResponseOp : UVMOp<"get_response"> {
  let summary = "Get response from driver";
  let description = [{
    Retrieves a response item from the driver. This is used in
    request-response protocols where the driver sends back a response
    for each transaction.

    Example:
    ```mlir
    %rsp = moore.uvm.get_response %sqr : !moore.class<@my_sequencer>
      -> !moore.class<@my_response>
    ```

    See UVM 1.2 Reference Manual Section 5.6 "Response Handling".
  }];

  let arguments = (ins
    ClassHandleType:$sequencer
  );
  let results = (outs ClassHandleType:$response);
  let assemblyFormat = [{
    $sequencer `:` type($sequencer) attr-dict `->` type($response)
  }];
}

//===----------------------------------------------------------------------===//
// UVM TLM Port Operations
//===----------------------------------------------------------------------===//

def UVMTLMPutOp : UVMOp<"tlm.put"> {
  let summary = "Put a transaction on a TLM port";
  let description = [{
    Puts a transaction on a TLM port. This is a blocking operation
    that waits for the target to accept the transaction.

    Example:
    ```mlir
    moore.uvm.tlm.put %port, %txn : !moore.class<@port_type>,
      !moore.class<@transaction>
    ```

    See UVM 1.2 Reference Manual Section 12.2 "TLM Ports".
  }];

  let arguments = (ins
    ClassHandleType:$port,
    ClassHandleType:$transaction
  );
  let assemblyFormat = [{
    $port `,` $transaction `:` type($port) `,` type($transaction) attr-dict
  }];
}

def UVMTLMGetOp : UVMOp<"tlm.get"> {
  let summary = "Get a transaction from a TLM port";
  let description = [{
    Gets a transaction from a TLM port. This is a blocking operation
    that waits for a transaction to be available.

    Example:
    ```mlir
    %txn = moore.uvm.tlm.get %port : !moore.class<@port_type>
      -> !moore.class<@transaction>
    ```

    See UVM 1.2 Reference Manual Section 12.2 "TLM Ports".
  }];

  let arguments = (ins
    ClassHandleType:$port
  );
  let results = (outs ClassHandleType:$transaction);
  let assemblyFormat = [{
    $port `:` type($port) attr-dict `->` type($transaction)
  }];
}

def UVMTLMTryPutOp : UVMOp<"tlm.try_put"> {
  let summary = "Try to put a transaction on a TLM port (non-blocking)";
  let description = [{
    Attempts to put a transaction on a TLM port without blocking.
    Returns true if successful, false if the port cannot accept.

    Example:
    ```mlir
    %ok = moore.uvm.tlm.try_put %port, %txn : !moore.class<@port_type>,
      !moore.class<@transaction>
    ```
  }];

  let arguments = (ins
    ClassHandleType:$port,
    ClassHandleType:$transaction
  );
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $port `,` $transaction `:` type($port) `,` type($transaction) attr-dict
  }];
}

def UVMTLMTryGetOp : UVMOp<"tlm.try_get"> {
  let summary = "Try to get a transaction from a TLM port (non-blocking)";
  let description = [{
    Attempts to get a transaction from a TLM port without blocking.
    Returns true if successful, false if no transaction available.

    Example:
    ```mlir
    %ok, %txn = moore.uvm.tlm.try_get %port : !moore.class<@port_type>
      -> !moore.class<@transaction>
    ```
  }];

  let arguments = (ins
    ClassHandleType:$port
  );
  let results = (outs I1:$success, ClassHandleType:$transaction);
  let assemblyFormat = [{
    $port `:` type($port) attr-dict `->` type($transaction)
  }];
}

def UVMTLMAnalysisWriteOp : UVMOp<"tlm.analysis_write"> {
  let summary = "Write to an analysis port (broadcast)";
  let description = [{
    Writes a transaction to an analysis port, broadcasting it to all
    connected subscribers. This is a non-blocking operation.

    Example:
    ```mlir
    moore.uvm.tlm.analysis_write %port, %txn : !moore.class<@analysis_port>,
      !moore.class<@transaction>
    ```

    See UVM 1.2 Reference Manual Section 12.3 "Analysis Ports".
  }];

  let arguments = (ins
    ClassHandleType:$port,
    ClassHandleType:$transaction
  );
  let assemblyFormat = [{
    $port `,` $transaction `:` type($port) `,` type($transaction) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UVM Reporting
//===----------------------------------------------------------------------===//

// UVM severity levels
def UVMInfoSeverity: I32EnumAttrCase<"Info", 0, "info">;
def UVMWarningSeverity: I32EnumAttrCase<"Warning", 1, "warning">;
def UVMErrorSeverity: I32EnumAttrCase<"Error", 2, "error">;
def UVMFatalSeverity: I32EnumAttrCase<"Fatal", 3, "fatal">;

def UVMSeverityAttr: I32EnumAttr<"UVMSeverity", "UVM message severity",
            [UVMInfoSeverity, UVMWarningSeverity, UVMErrorSeverity, UVMFatalSeverity]>{
  let cppNamespace = "circt::moore";
}

def UVMReportOp : UVMOp<"report"> {
  let summary = "Issue a UVM report message";
  let description = [{
    Issues a UVM report message with the specified severity, ID, and message.
    This is equivalent to `uvm_info, `uvm_warning, `uvm_error, and `uvm_fatal.

    Example:
    ```mlir
    moore.uvm.report info, "MY_ID", "Transaction completed" : !moore.string
    ```

    See UVM 1.2 Reference Manual Section 6.2 "Reporting".
  }];

  let arguments = (ins
    UVMSeverityAttr:$severity,
    StrAttr:$id,
    StringType:$message,
    I32Attr:$verbosity
  );
  let assemblyFormat = [{
    $severity `,` $id `,` $message `:` type($message) `,` $verbosity attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UVM Base Class Operations
//===----------------------------------------------------------------------===//

def UVMClassDeclOp : UVMOp<"class.decl", [
  Symbol,
  SymbolTable,
  IsolatedFromAbove,
  NoTerminator,
  SingleBlock
]> {
  let summary = "Declare a UVM class extending a base class";
  let description = [{
    Declares a class that extends one of the UVM base classes. This operation
    captures the UVM hierarchy relationship and enables proper phase method
    resolution and factory registration.

    Example:
    ```mlir
    moore.uvm.class.decl @my_driver extends uvm_driver {
      moore.class.propertydecl @req : !moore.class<@my_item>
      moore.uvm.phase_method build { ... }
    }
    ```

    See UVM 1.2 Reference Manual Chapter 5 "UVM Class Reference".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    UVMBaseClassAttr:$base_class,
    OptionalAttr<SymbolRefAttr>:$parent_class
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `extends` $base_class (`from` $parent_class^)?
    attr-dict-with-keyword $body
  }];
}

def UVMTLMPortDeclOp : UVMOp<"tlm.port_decl", [
  Symbol,
  HasParent<"UVMClassDeclOp, ClassDeclOp">
]> {
  let summary = "Declare a TLM port in a UVM component";
  let description = [{
    Declares a TLM port within a UVM component. TLM ports provide
    transaction-level communication between components.

    Example:
    ```mlir
    moore.uvm.tlm.port_decl @seq_item_port put : !moore.class<@my_item>
    moore.uvm.tlm.port_decl @analysis_port analysis : !moore.class<@my_item>
    ```

    See UVM 1.2 Reference Manual Section 12.2 "TLM Ports".
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TLMPortKindAttr:$port_kind,
    TypeAttr:$transaction_type
  );
  let assemblyFormat = [{
    $sym_name $port_kind `:` $transaction_type attr-dict
  }];
}

def UVMTLMPortConnectOp : UVMOp<"tlm.connect"> {
  let summary = "Connect two TLM ports";
  let description = [{
    Connects two TLM ports together. This is typically done in the
    connect_phase of a UVM component.

    Example:
    ```mlir
    moore.uvm.tlm.connect %driver.seq_item_port, %sequencer.seq_item_export
      : !moore.class<@driver_type>, !moore.class<@sequencer_type>
    ```

    See UVM 1.2 Reference Manual Section 12.2.2 "Port Connections".
  }];

  let arguments = (ins
    ClassHandleType:$port,
    ClassHandleType:$target
  );
  let assemblyFormat = [{
    $port `,` $target `:` type($port) `,` type($target) attr-dict
  }];
}

def UVMAnalysisPortConnectOp : UVMOp<"analysis.connect"> {
  let summary = "Connect an analysis port to a subscriber";
  let description = [{
    Connects an analysis port to a subscriber's analysis export.
    Analysis ports broadcast transactions to all connected subscribers.

    Example:
    ```mlir
    moore.uvm.analysis.connect %monitor.ap, %scoreboard.analysis_export
      : !moore.class<@monitor_type>, !moore.class<@scoreboard_type>
    ```

    See UVM 1.2 Reference Manual Section 12.3 "Analysis Ports".
  }];

  let arguments = (ins
    ClassHandleType:$port,
    ClassHandleType:$subscriber
  );
  let assemblyFormat = [{
    $port `,` $subscriber `:` type($port) `,` type($subscriber) attr-dict
  }];
}

def UVMGetParentOp : UVMOp<"get_parent"> {
  let summary = "Get parent component in UVM hierarchy";
  let description = [{
    Returns the parent component of a UVM component. This is essential
    for navigating the component hierarchy.

    Example:
    ```mlir
    %parent = moore.uvm.get_parent %this : !moore.class<@my_component>
      -> !moore.class<@uvm_component>
    ```

    See UVM 1.2 Reference Manual Section 13.1.3 "Hierarchy Information".
  }];

  let arguments = (ins ClassHandleType:$component);
  let results = (outs ClassHandleType:$parent);
  let assemblyFormat = [{
    $component `:` type($component) attr-dict `->` type($parent)
  }];
}

def UVMGetFullNameOp : UVMOp<"get_full_name"> {
  let summary = "Get full hierarchical name of a component";
  let description = [{
    Returns the full hierarchical name of a UVM component as a string.
    This includes all ancestor component names separated by dots.

    Example:
    ```mlir
    %name = moore.uvm.get_full_name %this : !moore.class<@my_component>
    ```

    See UVM 1.2 Reference Manual Section 13.1.3 "Hierarchy Information".
  }];

  let arguments = (ins ClassHandleType:$component);
  let results = (outs StringType:$name);
  let assemblyFormat = [{
    $component `:` type($component) attr-dict
  }];
}

def UVMFindByNameOp : UVMOp<"find_by_name"> {
  let summary = "Find a component by hierarchical path";
  let description = [{
    Searches for a component by its hierarchical path name.
    Returns null if no matching component is found.

    Example:
    ```mlir
    %comp = moore.uvm.find_by_name "env.agent.driver", %root
      : !moore.class<@uvm_component> -> !moore.class<@my_driver>
    ```

    See UVM 1.2 Reference Manual Section 13.1.3.3 "find".
  }];

  let arguments = (ins
    StrAttr:$path,
    Optional<ClassHandleType>:$context
  );
  let results = (outs ClassHandleType:$component);
  let assemblyFormat = [{
    $path (`,` $context^ `:` type($context))? attr-dict `->` type($component)
  }];
}

//===----------------------------------------------------------------------===//
// String Builtins
//===----------------------------------------------------------------------===//

class StringBuiltin<string mnemonic, list<Trait> traits = []> :
    MooreOp<"string." # mnemonic, traits>;

def StringLenOp : StringBuiltin<"len"> {
  let summary = "Get the length of a string";
  let description = [{
    Returns the number of characters in the given string.
  }];
  let arguments = (ins StringType:$str);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringToUpperOp : StringBuiltin<"toupper"> {
  let summary = "Convert a string to uppercase";
  let description = [{
    Converts all characters in the given string to uppercase.
  }];
  let arguments = (ins StringType:$str);
  let results = (outs StringType:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringToLowerOp : StringBuiltin<"tolower"> {
  let summary = "Convert a string to lowercase";
  let description = [{
    Converts all characters in the given string to lowercase.
  }];
  let arguments = (ins StringType:$str);
  let results = (outs StringType:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringGetCOp : StringBuiltin<"getc"> {
  let summary = "Get a character from a string";
  let description = [{
    Returns the character at the specified index in the given string.
  }];
  let arguments = (ins StringType:$str, TwoValuedI32:$index);
  let results = (outs TwoValuedI8:$result);
  let assemblyFormat = "$str `[` $index `]` attr-dict";
}

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
