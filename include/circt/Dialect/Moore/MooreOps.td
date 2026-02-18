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

// Drive strength for continuous assignments and net declarations.
// See IEEE 1800-2017 § 7.9 "Strength reduction".
def DriveStrengthAttr : I32EnumAttr<"DriveStrength", "Signal drive strength", [
    I32EnumAttrCase<"Supply", 0, "supply">,
    I32EnumAttrCase<"Strong", 1, "strong">,
    I32EnumAttrCase<"Pull", 2, "pull">,
    I32EnumAttrCase<"Weak", 3, "weak">,
    I32EnumAttrCase<"HighZ", 4, "highz">,
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

def ContinuousAssignOp : MooreOp<"assign", [
  HasParent<"SVModuleOp">,
  TypesMatchWith<"src and dst types match",
    "src", "dst", "RefType::get(cast<UnpackedType>($_self))">
]> {
  let summary = "Continuous assignment within a module";
  let description = [{
    A continuous assignment in module scope, such as `assign x = y;`, which
    continuously drives the value on the right-hand side onto the left-hand
    side.

    An optional drive strength pair (strength0, strength1) can be specified
    for the assignment to control how the driven value resolves when multiple
    drivers exist. The strength0 value applies when driving 0, and strength1
    when driving 1.

    See IEEE 1800-2017 § 10.3 "Continuous assignments".
    See IEEE 1800-2017 § 7.9 "Strength reduction".
  }];
  let arguments = (ins
    RefType:$dst,
    UnpackedType:$src,
    OptionalAttr<DriveStrengthAttr>:$strength0,
    OptionalAttr<DriveStrengthAttr>:$strength1
  );
  let assemblyFormat = [{
    $dst `,` $src (`strength` `(` $strength0^ `,` $strength1 `)`)? attr-dict `:` type($src)
  }];
  let builders = [
    // Builder without strength (for backward compatibility)
    OpBuilder<(ins "Value":$dst, "Value":$src), [{
      build($_builder, $_state, dst, src, DriveStrengthAttr{}, DriveStrengthAttr{});
    }]>
  ];
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

def EventTriggeredOp : MooreOp<"event_triggered", [Pure]> {
  let summary = "Check if an event was triggered in the current time slot";
  let description = [{
    The `moore.event_triggered` op checks if the given event has been triggered
    in the current time slot. Returns 1 if the event was triggered, 0 otherwise.
    This corresponds to the SystemVerilog `.triggered` property on events.

    Example:
    ```mlir
    %triggered = moore.event_triggered %event : !moore.event
    ```

    See IEEE 1800-2017 § 15.5.3 "Event sequencing: triggered".
  }];
  let arguments = (ins EventType:$event);
  let results = (outs BitType:$result);
  let assemblyFormat = [{ $event attr-dict `:` type($event) }];
}

def EventTriggerOp : MooreOp<"event_trigger"> {
  let summary = "Trigger an event";
  let description = [{
    The `moore.event_trigger` op triggers the given event, waking up any
    processes waiting on it. This corresponds to the SystemVerilog `->event`
    named event trigger statement.

    Example:
    ```mlir
    moore.event_trigger %event : !moore.ref<!moore.event>
    ```

    See IEEE 1800-2017 Section 15.5 "Event trigger".
  }];
  let arguments = (ins EventRefType:$event);
  let assemblyFormat = [{ $event attr-dict `:` type($event) }];
}

def WaitConditionOp : MooreOp<"wait_condition"> {
  let summary = "Suspend execution until a condition becomes true";
  let description = [{
    The `moore.wait_condition` op suspends execution of the current process
    until the given condition evaluates to true. Corresponds to the SystemVerilog
    `wait(condition)` statement.

    Example:
    ```mlir
    moore.wait_condition %cond : i1
    ```

    See IEEE 1800-2017 § 9.4.3 "Level-sensitive event control".
  }];
  let arguments = (ins BitType:$condition);
  let assemblyFormat = [{ $condition attr-dict `:` type($condition) }];
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
      ins "IntType":$type, "int64_t":$value, CArg<"bool", "false">:$isSigned)>,
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

def ClassHandleCmpPredicateAttr : I64EnumAttr<
    "ClassHandleCmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>
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

def ClassNullOp : MooreOp<"class.null", [Pure]> {
  let summary = "Create a null class handle";
  let description = [{
    Creates a null class handle value. In SystemVerilog, `null` represents
    the absence of a valid class instance reference. Comparing a class handle
    to `null` checks whether the handle points to a valid object.

    Example:
    ```mlir
    %null = moore.class.null : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 8.4 "Class handle operations".
  }];

  let arguments = (ins);
  let results = (outs ClassHandleType:$result);

  let assemblyFormat = [{
    `:` type($result) attr-dict
  }];
}

def ClassHandleCmpOp : MooreOp<"class_handle_cmp", [
  Pure,
  Commutative,
  SameTypeOperands
]> {
  let summary = "Compare class handles for reference equality";
  let description = [{
    Compares two class handles using reference equality. This checks whether
    two handles point to the same object instance (or are both null).
    Supported predicates:
    - eq : equal (handles point to same object or both null)
    - ne : not equal (handles point to different objects)

    Note: This is different from comparing the contents of the objects.
    Two different objects with identical data will compare as not equal.

    Example:
    ```mlir
    %result = moore.class_handle_cmp eq %a, %b : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 8.4 "Class handle operations".
  }];
  let arguments = (ins
    ClassHandleCmpPredicateAttr:$predicate,
    ClassHandleType:$lhs,
    ClassHandleType:$rhs
  );
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def StringConcatOp : MooreOp<"string_concat", [Pure]> {
  let summary = "Concatenate strings";
  let description = [{
    Concatenates an arbitrary number of strings into one larger string.
    The strings are concatenated from left to right, with the first
    operand appearing at the left start of the result string, and the last
    operand appearing at the right end. Produces an empty string if no inputs
    are provided.

    Example:
    ```mlir
    %result = moore.string_concat (%a, %b, %c) : !moore.string
    ```
  }];
  let arguments = (ins Variadic<StringType>:$inputs);
  let results = (outs StringType:$result);
  let assemblyFormat = "` ` `(` $inputs `)` attr-dict `:` type($result)";
}

def StringReplicateOp : MooreOp<"string_replicate", [Pure]> {
  let summary = "Replicate a string a specified number of times";
  let description = [{
    Replicates a string a specified number of times, concatenating the
    copies together. This corresponds to the SystemVerilog string
    replication operator `{n{str}}`.

    Example:
    ```mlir
    %result = moore.string_replicate %count, %str : i32, !moore.string
    ```
    If `%count` is 3 and `%str` is "ab", `%result` will be "ababab".
  }];
  let arguments = (ins TwoValuedI32:$count, StringType:$string);
  let results = (outs StringType:$result);
  let assemblyFormat = "$count `,` $string attr-dict";
}

def StreamConcatOp : MooreOp<"stream_concat", [Pure]> {
  let summary = "Streaming concatenation of a dynamic array or queue";
  let description = [{
    Performs streaming concatenation on a dynamic array or queue, iterating
    over all elements at runtime. This corresponds to the SystemVerilog
    streaming concatenation operator `{>>{array}}` or `{<<{array}}` when
    applied to dynamic-sized types like queues or dynamic arrays.

    For queues/arrays of strings, this concatenates all string elements into
    a single string. For queues/arrays of other types, this packs the element
    bits into a single bit vector.

    The `isRightToLeft` attribute controls the direction:
    - false (default): left-to-right streaming (`{>>{array}}`)
    - true: right-to-left streaming (`{<<{array}}`)

    Example:
    ```mlir
    // String queue streaming: concatenates all strings
    %result = moore.stream_concat %str_queue : !moore.queue<string, 0> -> !moore.string

    // Integer queue streaming: packs bits into vector
    %bits = moore.stream_concat %int_queue : !moore.queue<i32, 0> -> !moore.i32
    ```

    See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".
  }];
  let arguments = (ins
    UnpackedType:$input,
    DefaultValuedAttr<BoolAttr, "false">:$isRightToLeft
  );
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input (`right_to_left` $isRightToLeft^)? attr-dict `:` type($input) `->` type($result)
  }];
}

def StreamUnpackOp : MooreOp<"stream_unpack", []> {
  let summary = "Streaming unpack into a dynamic array or queue";
  let description = [{
    Performs streaming unpacking from a bit vector into a dynamic array or queue
    reference. This is the inverse of `moore.stream_concat` and corresponds to
    using streaming operators on the left-hand side of an assignment:
    `{<<{array}} = source` or `{>>{array}} = source`.

    The source value is unpacked bit-by-bit into the elements of the target
    dynamic array. The array is resized as needed to hold all the bits.

    The `isRightToLeft` attribute controls the direction:
    - false (default): left-to-right streaming (`{>>{array}} = src`)
    - true: right-to-left streaming (`{<<{array}} = src`)

    Example:
    ```mlir
    // Unpack bits into a dynamic array
    moore.stream_unpack %array_ref, %source : !moore.ref<!moore.open_uarray<i1>>, !moore.i64
    ```

    See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".
  }];
  let arguments = (ins
    RefType:$dst,
    UnpackedType:$src,
    DefaultValuedAttr<BoolAttr, "false">:$isRightToLeft
  );
  let assemblyFormat = [{
    $dst `,` $src (`right_to_left` $isRightToLeft^)? attr-dict `:` type($dst) `,` type($src)
  }];
}

def StreamConcatMixedOp : MooreOp<"stream_concat_mixed", [
    Pure, AttrSizedOperandSegments]> {
  let summary = "Mixed static/dynamic streaming concatenation";
  let description = [{
    Performs streaming concatenation with a mix of static bit vectors and a
    single dynamic array or queue. This is used when streaming operators like
    `{<<{a, b, dyn_arr, c}}` mix static and dynamic operands.

    The operation takes:
    - `staticPrefix`: Static bit vectors that appear before the dynamic array
    - `dynamicArray`: A dynamic array or queue
    - `staticSuffix`: Static bit vectors that appear after the dynamic array

    The result is a queue containing all the bits, with the streaming direction
    applied.

    Example:
    ```mlir
    // Mixed streaming: {<< 8 {header, data[], crc}}
    %result = moore.stream_concat_mixed [%header] %data [%crc] right_to_left slice 8
        : (!moore.i32, !moore.open_uarray<i8>, !moore.i32) -> !moore.queue<i8, 0>
    ```

    See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".
  }];
  let arguments = (ins
    Variadic<UnpackedType>:$staticPrefix,
    UnpackedType:$dynamicArray,
    Variadic<UnpackedType>:$staticSuffix,
    I32Attr:$sliceSize,
    DefaultValuedAttr<BoolAttr, "false">:$isRightToLeft
  );
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `[` $staticPrefix `]` $dynamicArray `[` $staticSuffix `]`
    (`right_to_left` $isRightToLeft^)? `slice` $sliceSize
    attr-dict `:` functional-type(operands, $result)
  }];
}

def StreamUnpackMixedOp : MooreOp<"stream_unpack_mixed", [AttrSizedOperandSegments]> {
  let summary = "Mixed static/dynamic streaming unpack";
  let description = [{
    Performs streaming unpacking into a mix of static lvalue references and a
    single dynamic array or queue reference. This is used when streaming lvalue
    operators like `{<<{a, b, dyn_arr, c}} = src` mix static and dynamic targets.

    The operation takes:
    - `staticPrefixRefs`: References to static lvalues before the dynamic array
    - `dynamicArrayRef`: Reference to a dynamic array or queue
    - `staticSuffixRefs`: References to static lvalues after the dynamic array
    - `src`: The source value to unpack from

    The bits from src are distributed to the static prefix, then remaining bits
    go to the dynamic array (resized as needed), then the static suffix gets
    the trailing bits.

    Example:
    ```mlir
    // Mixed streaming unpack: {<< 8 {header, data[], crc}} = pkt
    moore.stream_unpack_mixed [%header_ref] %data_ref [%crc_ref] %pkt right_to_left slice 8
        : (!moore.ref<i32>, !moore.ref<open_uarray<i8>>, !moore.ref<i32>, !moore.queue<i8, 0>)
    ```

    See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".
  }];
  let arguments = (ins
    Variadic<RefType>:$staticPrefixRefs,
    RefType:$dynamicArrayRef,
    Variadic<RefType>:$staticSuffixRefs,
    UnpackedType:$src,
    I32Attr:$sliceSize,
    DefaultValuedAttr<BoolAttr, "false">:$isRightToLeft
  );
  let assemblyFormat = [{
    `[` $staticPrefixRefs `]` $dynamicArrayRef `[` $staticSuffixRefs `]` $src
    (`right_to_left` $isRightToLeft^)? `slice` $sliceSize
    attr-dict `:` functional-type(operands, results)
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

// Immediate assertions can appear in procedures (always, initial, etc.) or
// in class methods (which lower to func::FuncOp). We don't restrict the parent
// since the lowering to verif ops works regardless of the enclosing region.
class ImmediateAssertOp<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits>{
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

def PastOp : MooreOp<"past", [Pure, SameOperandsAndResultType]> {
  let summary = "Get the value from a previous clock cycle";
  let description = [{
    The `moore.past` operation returns the value of an expression from a previous
    clock cycle. This is used to implement the SystemVerilog `$past` system
    function when the value needs to preserve its type for comparisons.

    The `delay` attribute specifies how many clock cycles in the past to look.
    A delay of 1 (the default) returns the value from the previous cycle.

    Example:
    ```mlir
    %past_val = moore.past %val delay 1 : !moore.l32
    ```

    See IEEE 1800-2017 Section 16.9.3 "Past sampled value functions".
  }];
  let arguments = (ins UnpackedType:$input, I64Attr:$delay);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input `delay` $delay attr-dict `:` type($input)
  }];
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

def FormatCharOp : MooreOp<"fmt.char", [Pure]> {
  let summary = "Format an integer value as a character";
  let description = [{
    Format an integer value as a single ASCII character. For values up to 127,
    ASCII encoding is used. For larger values, behavior is unspecified.

    See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
  }];
  let arguments = (ins IntType:$value);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
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

def FormatClassOp : MooreOp<"fmt.class", [Pure]> {
  let summary = "Format a class handle value";
  let description = [{
    Format a class handle (object reference) as a string. Since actual
    object addresses are simulator-specific and meaningless in MLIR,
    this operation produces a placeholder representation.
    See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
  }];
  let arguments = (ins ClassHandleType:$value);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
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
  let assemblyFormat = "(`seed` $seed^)? attr-dict";
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
  let assemblyFormat = "(`seed` $seed^)? attr-dict";
}

def UrandomRangeBIOp : Builtin<"urandom_range"> {
  let summary = "Generate a pseudo-random unsigned integer within a range";
  let description = [{
    Corresponds to the `$urandom_range` system function. Returns a 32-bit
    unsigned random number in the range [min, max]. If min is omitted or
    greater than max, min is treated as 0.

    According to IEEE 1800-2017 Section 18.13.3:
    - $urandom_range(max) returns a value in [0, max]
    - $urandom_range(max, min) returns a value in [min, max]
    - If min > max, the arguments are automatically swapped

    See IEEE 1800-2017 § 18.13.3 "$urandom_range".
  }];

  let arguments = (ins TwoValuedI32:$maxval, Optional<TwoValuedI32>:$minval);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$maxval (`,` $minval^)? attr-dict";
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
// Process Control: Fork/Join Operations
//===----------------------------------------------------------------------===//

// Join type enum for fork operations
def JoinAll : I32EnumAttrCase<"JoinAll", 0, "join">;
def JoinAny : I32EnumAttrCase<"JoinAny", 1, "join_any">;
def JoinNone : I32EnumAttrCase<"JoinNone", 2, "join_none">;

def JoinTypeAttr : I32EnumAttr<"JoinType", "Fork join type",
                               [JoinAll, JoinAny, JoinNone]> {
  let cppNamespace = "circt::moore";
}

def ForkOp : MooreOp<"fork", [
  RecursiveMemoryEffects,
  NoRegionArguments,
]> {
  let summary = "Spawn concurrent processes (SystemVerilog fork)";
  let description = [{
    The `moore.fork` operation spawns multiple concurrent processes that execute
    in parallel. This corresponds to the SystemVerilog `fork...join`,
    `fork...join_any`, and `fork...join_none` constructs.

    Each region within the fork represents a separate concurrent process.
    The `join_type` attribute specifies the synchronization behavior:
    - `join`: Wait for all spawned processes to complete (default)
    - `join_any`: Wait for any one process to complete
    - `join_none`: Don't wait, continue immediately

    Example:
    ```
    // fork...join (wait for all)
    moore.fork join {
      // Process 1
      moore.wait_delay %d1
    }, {
      // Process 2
      moore.wait_delay %d2
    }

    // fork...join_any (wait for first)
    moore.fork join_any {
      // Process 1
    }, {
      // Process 2
    }

    // fork...join_none (continue immediately)
    moore.fork join_none {
      // Background process
    }
    ```

    See IEEE 1800-2017 Section 9.3.2 "Parallel blocks".
  }];
  let arguments = (ins
    JoinTypeAttr:$joinType,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs);
  let regions = (region VariadicRegion<AnyRegion>:$branches);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ForkTerminatorOp : MooreOp<"fork.terminator", [
  Pure,
  Terminator,
  ParentOneOf<["ForkOp"]>,
]> {
  let summary = "Terminator for fork branch regions";
  let description = [{
    `moore.fork.terminator` marks the end of a branch within a `moore.fork`
    operation. This is an implicit terminator that is automatically inserted.
  }];
  let assemblyFormat = "attr-dict";
}

def WaitForkOp : MooreOp<"wait_fork"> {
  let summary = "Wait for all child processes to complete";
  let description = [{
    The `moore.wait_fork` operation waits for all processes spawned by
    `fork...join_none` in the current scope to complete. This is equivalent
    to the SystemVerilog `wait fork` statement.

    Unlike an explicit join which operates on a specific fork, `wait_fork`
    waits for ALL outstanding child processes.

    Example:
    ```
    moore.fork join_none { ... }
    moore.fork join_none { ... }
    moore.fork join_none { ... }

    // Wait for all three forks to complete
    moore.wait_fork
    ```

    See IEEE 1800-2017 Section 9.6.1 "Wait fork statement".
  }];
  let assemblyFormat = "attr-dict";
}

def DisableForkOp : MooreOp<"disable_fork"> {
  let summary = "Disable all child processes";
  let description = [{
    The `moore.disable_fork` operation terminates all processes spawned by
    `fork...join_none` in the current scope. This corresponds to the
    SystemVerilog `disable fork` statement.

    All child processes (and their descendants) are immediately terminated.

    Example:
    ```
    moore.fork join_none { ... }
    moore.fork join_none { ... }

    // Terminate all child processes
    moore.disable_fork
    ```

    See IEEE 1800-2017 Section 9.6.2 "Disable fork statement".
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Process Control: Named Blocks and Disable
//===----------------------------------------------------------------------===//

def NamedBlockOp : MooreOp<"named_block", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"NamedBlockTerminatorOp">,
]> {
  let summary = "A named block that can be disabled";
  let description = [{
    The `moore.named_block` operation defines a named scope that can be targeted
    by `moore.disable`. This corresponds to named `begin...end` blocks in
    SystemVerilog.

    Named blocks can be nested, and disabling an outer block also terminates
    all nested blocks.

    Example:
    ```
    moore.named_block "search_loop" {
      // ... loop body ...
      scf.if %found {
        moore.disable "search_loop"
      }
    }
    ```

    See IEEE 1800-2017 Section 9.3.4 "Block names".
  }];
  let arguments = (ins StrAttr:$blockName);
  let regions = (region MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def NamedBlockTerminatorOp : MooreOp<"named_block.terminator", [
  Pure,
  Terminator,
  ParentOneOf<["NamedBlockOp"]>,
]> {
  let summary = "Terminator for named block regions";
  let description = [{
    `moore.named_block.terminator` marks the end of a `moore.named_block`.
    This is an implicit terminator.
  }];
  let assemblyFormat = "attr-dict";
}

def DisableOp : MooreOp<"disable"> {
  let summary = "Disable a named block or task";
  let description = [{
    The `moore.disable` operation terminates execution of a named block or task.
    This corresponds to the SystemVerilog `disable` statement.

    When a block/task is disabled:
    - Execution jumps to the statement following the disabled block
    - All nested blocks within the disabled block are also terminated
    - Variables retain their values at the time of disable

    The target must be the name of a `moore.named_block` operation.

    Example:
    ```
    moore.named_block "outer_loop" {
      moore.named_block "inner_loop" {
        // ... some condition ...
        moore.disable "outer_loop"  // Exit both loops
      }
    }
    ```

    See IEEE 1800-2017 Section 9.6.3 "Disable".
  }];
  let arguments = (ins StrAttr:$target);
  let assemblyFormat = "$target attr-dict";
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

def FWriteBIOp : Builtin<"fwrite"> {
  let summary = "Write a formatted message to a file";
  let description = [{
    Writes the given format string to the file identified by the file descriptor.
    This corresponds to the `$fwrite` and `$fdisplay` system tasks. Message
    formatting is handled by `moore.fmt.*` ops. The file descriptor is typically
    obtained from `$fopen`.

    See IEEE 1800-2017 Section 21.3 "File I/O tasks and functions".
  }];
  let arguments = (ins UnpackedType:$fd, FormatStringType:$message);
  let assemblyFormat = "$fd `,` $message attr-dict `:` type($fd)";
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

// Binary real math functions
def Atan2BIOp : Builtin<"atan2", [SameOperandsAndResultType]> {
  let summary = "Two-argument arc-tangent";
  let description = [{
    The `$atan2(y, x)` system function returns the arc-tangent of y/x in radians,
    using the signs of both arguments to determine the quadrant of the return value.

    See IEEE 1800-2017 section 20.8.2 "Real math functions".
  }];
  let arguments = (ins RealF64:$y, RealF64:$x);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$y `,` $x attr-dict `:` type($result)";
}

def HypotBIOp : Builtin<"hypot", [SameOperandsAndResultType]> {
  let summary = "Hypotenuse function";
  let description = [{
    The `$hypot(x, y)` system function returns `sqrt(x^2 + y^2)`, the length
    of the hypotenuse of a right triangle with sides of length x and y.

    See IEEE 1800-2017 section 20.8.2 "Real math functions".
  }];
  let arguments = (ins RealF64:$x, RealF64:$y);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$x `,` $y attr-dict `:` type($result)";
}

def PowBIOp : Builtin<"pow", [SameOperandsAndResultType]> {
  let summary = "Power function";
  let description = [{
    The `$pow(x, y)` system function returns x raised to the power y.

    See IEEE 1800-2017 section 20.8.2 "Real math functions".
  }];
  let arguments = (ins RealF64:$x, RealF64:$y);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$x `,` $y attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Bit Vector System Functions
//===----------------------------------------------------------------------===//

def IsUnknownBIOp : Builtin<"isunknown"> {
  let summary = "Check if any bit is X or Z";
  let description = [{
    Returns 1 if any bit in the input expression is X or Z, otherwise returns 0.
    This is commonly used for checking for unknown values in verification.

    See IEEE 1800-2017 Section 20.9 "Bit vector system functions".
  }];
  let arguments = (ins SimpleBitVectorType:$value);
  let results = (outs TwoValuedI1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def CountOnesBIOp : Builtin<"countones", [SameOperandsAndResultType]> {
  let summary = "Count the number of 1 bits in a bit vector";
  let description = [{
    Returns the count of the number of bits set to 1 in the input expression.
    If any bit in the expression is X or Z, the result is X.

    This function is commonly used in constraint expressions for one-hot
    encoding validation (e.g., `$countones(sel) == 1`).

    See IEEE 1800-2017 Section 20.9 "Bit vector system functions".
  }];
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def OneHotBIOp : Builtin<"onehot"> {
  let summary = "Check if exactly one bit is set in a bit vector";
  let description = [{
    Returns 1 if exactly one bit in the input expression is set to 1,
    otherwise returns 0. If any bit is X or Z, the result is X.

    This function is commonly used in assertions to verify one-hot encoding
    of state machines and other signals.

    Equivalent to `$countones(expr) == 1`.

    See IEEE 1800-2017 Section 20.9 "Bit vector system functions".
  }];
  let arguments = (ins SimpleBitVectorType:$value);
  let results = (outs TwoValuedI1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def OneHot0BIOp : Builtin<"onehot0"> {
  let summary = "Check if at most one bit is set in a bit vector";
  let description = [{
    Returns 1 if at most one bit (zero or one) in the input expression is
    set to 1, otherwise returns 0. If any bit is X or Z, the result is X.

    This function is commonly used in assertions to verify one-hot or
    zero encoding of signals where idle states are represented by all zeros.

    Equivalent to `$countones(expr) <= 1`.

    See IEEE 1800-2017 Section 20.9 "Bit vector system functions".
  }];
  let arguments = (ins SimpleBitVectorType:$value);
  let results = (outs TwoValuedI1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def CountBitsBIOp : Builtin<"countbits", [SameOperandsAndResultType]> {
  let summary = "Count the number of bits matching a control value";
  let description = [{
    Returns the count of the number of bits in the input expression that
    match any of the specified control_bit values (0, 1, x, z).

    For the two-valued (no X/Z) lowering:
    - `$countbits(x, 1)` is equivalent to `$countones(x)`
    - `$countbits(x, 0)` is equivalent to `bitwidth - $countones(x)`

    The control_bits attribute specifies which bit values to count:
    - 0b0001 (1): count zeros
    - 0b0010 (2): count ones
    - 0b0100 (4): count X values (always 0 in two-valued)
    - 0b1000 (8): count Z values (always 0 in two-valued)

    Multiple values can be OR'd together to count multiple bit types.

    See IEEE 1800-2017 Section 20.9 "Bit vector system functions".
  }];
  let arguments = (ins IntType:$value, I32Attr:$control_bits);
  let results = (outs IntType:$result);
  let assemblyFormat = "$value `,` $control_bits attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// File I/O System Functions
//===----------------------------------------------------------------------===//

def FOpenBIOp : Builtin<"fopen"> {
  let summary = "Open a file and return a file descriptor";
  let description = [{
    Corresponds to the `$fopen` system function. Opens a file with the specified
    filename and mode, and returns a 32-bit integer file descriptor (also known
    as a multichannel descriptor or MCD).

    The mode argument specifies the file access mode:
    - "r" or "rb": Open for reading
    - "w" or "wb": Open for writing (truncate existing file)
    - "a" or "ab": Open for appending
    - "r+", "r+b", "rb+": Open for reading and writing
    - "w+", "w+b", "wb+": Open for reading and writing (truncate)
    - "a+", "a+b", "ab+": Open for reading and appending

    Returns 0 if the file could not be opened, otherwise returns a non-zero
    file descriptor that can be used with other file I/O functions.

    See IEEE 1800-2017 Section 21.3 "Opening and closing files".
  }];
  let arguments = (ins StringType:$filename, Optional<StringType>:$mode);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$filename (`,` $mode^)? attr-dict";
}

def FCloseBIOp : Builtin<"fclose"> {
  let summary = "Close a file";
  let description = [{
    Corresponds to the `$fclose` system task. Closes the file associated with
    the given file descriptor (fd). After closing, the file descriptor becomes
    invalid and should not be used for further file I/O operations.

    The fd argument is a 32-bit integer file descriptor previously returned
    by `$fopen`.

    See IEEE 1800-2017 Section 21.3.2 "Closing files".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let assemblyFormat = "$fd attr-dict";
}

def SScanfBIOp : Builtin<"sscanf"> {
  let summary = "Read formatted data from a string";
  let description = [{
    Corresponds to the `$sscanf` system function. Reads data from the input
    string according to the format specification and stores the results in
    the provided output arguments.

    The format string supports the following format specifiers:
    - %d: Signed decimal integer
    - %h, %x: Hexadecimal integer
    - %o: Octal integer
    - %b: Binary integer
    - %s: String (whitespace-delimited)
    - %c: Single character

    Returns the number of items successfully read and assigned, or EOF (-1)
    if an input failure occurs before any conversion.

    The `args` are references to variables where the parsed values will be
    stored. The number of format specifiers in the format string should match
    the number of output arguments.

    Example:
    ```
    string s = "42 hello";
    int x;
    string str;
    int n = $sscanf(s, "%d %s", x, str);
    // n = 2, x = 42, str = "hello"
    ```

    See IEEE 1800-2017 Section 21.3.4 "Reading data from a string".
  }];
  let arguments = (ins StringType:$input, StrAttr:$format, Variadic<RefType>:$args);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$input `,` $format `,` $args attr-dict `:` type($args)";
}

def FScanfBIOp : Builtin<"fscanf"> {
  let summary = "Read formatted data from a file";
  let description = [{
    Corresponds to the `$fscanf` system function. Reads data from the file
    associated with the given file descriptor according to the format
    specification and stores the results in the provided output arguments.

    The format string supports the same format specifiers as `$sscanf`:
    - %d: Signed decimal integer
    - %h, %x: Hexadecimal integer
    - %o: Octal integer
    - %b: Binary integer
    - %s: String (whitespace-delimited)
    - %c: Single character

    Returns the number of items successfully read and assigned, or EOF (-1)
    if an input failure occurs before any conversion.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins TwoValuedI32:$fd, StrAttr:$format, Variadic<RefType>:$args);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd `,` $format `,` $args attr-dict `:` type($args)";
}

def FGetCBIOp : Builtin<"fgetc"> {
  let summary = "Read a single character from a file";
  let description = [{
    Corresponds to the `$fgetc` system function. Reads and returns a single
    character (byte) from the file specified by the file descriptor.

    Returns the character read as an integer, or EOF (-1) if end-of-file
    is reached or an error occurs.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd attr-dict";
}

def FGetSBIOp : Builtin<"fgets"> {
  let summary = "Read a line from a file into a string";
  let description = [{
    Corresponds to the `$fgets` system function. Reads characters from the
    file specified by the file descriptor into the string variable until
    a newline is encountered, end-of-file is reached, or the string is full.

    The str argument must be a reference to a string variable where the
    read data will be stored. Returns the number of characters read, or
    0 if end-of-file is reached or an error occurs.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins RefType:$str, TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str `,` $fd attr-dict `:` type($str)";
}

def FEofBIOp : Builtin<"feof"> {
  let summary = "Check if end-of-file has been reached";
  let description = [{
    Corresponds to the `$feof` system function. Tests whether end-of-file
    has been reached on the file specified by the file descriptor.

    Returns a non-zero value if end-of-file has been reached, 0 otherwise.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd attr-dict";
}

def FFlushBIOp : Builtin<"fflush"> {
  let summary = "Flush file buffer";
  let description = [{
    Corresponds to the `$fflush` system task. Flushes the output buffer
    for the file specified by the file descriptor. If fd is 0, all open
    files are flushed.

    See IEEE 1800-2017 Section 21.3.2 "Closing files".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let assemblyFormat = "$fd attr-dict";
}

def FTellBIOp : Builtin<"ftell"> {
  let summary = "Get current file position";
  let description = [{
    Corresponds to the `$ftell` system function. Returns the current value
    of the file position indicator for the file specified by the file
    descriptor.

    Returns the current file position, or -1 if an error occurs.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd attr-dict";
}

def FErrorBIOp : Builtin<"ferror"> {
  let summary = "Get file error status and message";
  let description = [{
    Corresponds to the `$ferror` system function. Returns the error code
    for the specified file descriptor and writes the error message to
    the output string reference.

    The function returns 0 if no error occurred, or a non-zero error code
    otherwise.

    See IEEE 1800-2017 Section 21.3.1 "Opening and closing files".
  }];
  let arguments = (ins TwoValuedI32:$fd, RefType:$strRef);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd `,` $strRef attr-dict `:` type($strRef)";
}

def StrobeBIOp : Builtin<"strobe"> {
  let summary = "Display values at the end of the current time step";
  let description = [{
    Corresponds to the `$strobe` system task. Similar to `$display`, but
    the output is delayed until the end of the current simulation time
    step, ensuring that all events scheduled for the current time have
    been processed.

    See IEEE 1800-2017 Section 21.2.4 "Strobed monitoring".
  }];
  let arguments = (ins FormatStringType:$message);
  let assemblyFormat = "$message attr-dict";
}

def MonitorBIOp : Builtin<"monitor"> {
  let summary = "Enable continuous monitoring of variable changes";
  let description = [{
    Corresponds to the `$monitor` system task. Enables monitoring of the
    specified variables. Whenever any of the monitored variables changes
    value during simulation, the formatted message is automatically
    displayed at the end of the current time step.

    Only one $monitor task can be active at a time. Subsequent calls to
    $monitor replace the previous monitoring specification.

    See IEEE 1800-2017 Section 21.2.3 "Monitoring".
  }];
  let arguments = (ins FormatStringType:$message);
  let assemblyFormat = "$message attr-dict";
}

def FStrobeBIOp : Builtin<"fstrobe"> {
  let summary = "Write strobe output to a file";
  let description = [{
    Corresponds to the `$fstrobe` system task. Like `$strobe`, but writes
    the output to the file specified by the file descriptor instead of
    standard output.

    See IEEE 1800-2017 Section 21.3.1 "File output tasks".
  }];
  let arguments = (ins TwoValuedI32:$fd, FormatStringType:$message);
  let assemblyFormat = "$fd `,` $message attr-dict";
}

def FMonitorBIOp : Builtin<"fmonitor"> {
  let summary = "Enable continuous monitoring to a file";
  let description = [{
    Corresponds to the `$fmonitor` system task. Like `$monitor`, but writes
    the output to the file specified by the file descriptor instead of
    standard output.

    See IEEE 1800-2017 Section 21.3.1 "File output tasks".
  }];
  let arguments = (ins TwoValuedI32:$fd, FormatStringType:$message);
  let assemblyFormat = "$fd `,` $message attr-dict";
}

def MonitorOnBIOp : Builtin<"monitoron"> {
  let summary = "Enable monitoring";
  let description = [{
    Corresponds to the `$monitoron` system task. Re-enables monitoring.
    See IEEE 1800-2017 Section 21.2.3 "Monitoring".
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def MonitorOffBIOp : Builtin<"monitoroff"> {
  let summary = "Disable monitoring";
  let description = [{
    Corresponds to the `$monitoroff` system task. Disables monitoring.
    See IEEE 1800-2017 Section 21.2.3 "Monitoring".
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def PrintTimescaleBIOp : Builtin<"printtimescale"> {
  let summary = "Print the timescale of a module";
  let description = [{
    Corresponds to the `$printtimescale` system task.
    See IEEE 1800-2017 Section 20.3 "Timescale system tasks".
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def UngetCBIOp : Builtin<"ungetc"> {
  let summary = "Push a character back into the file stream";
  let description = [{
    Corresponds to the `$ungetc` system task.
    See IEEE 1800-2017 Section 21.3.4 "File positioning functions".
  }];
  let arguments = (ins TwoValuedI32:$c, TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$c `,` $fd attr-dict";
}

def FSeekBIOp : Builtin<"fseek"> {
  let summary = "Set file position";
  let description = [{
    Corresponds to the `$fseek` system function. Sets the file position for
    the file specified by the file descriptor. The offset is relative to
    the position specified by the whence argument:
      0 = SEEK_SET (beginning of file)
      1 = SEEK_CUR (current position)
      2 = SEEK_END (end of file)

    Returns 0 on success, or -1 on error.

    See IEEE 1800-2017 Section 21.3.3 "File positioning functions".
  }];
  let arguments = (ins TwoValuedI32:$fd, TwoValuedI32:$offset, TwoValuedI32:$whence);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$fd `,` $offset `,` $whence attr-dict";
}

def RewindBIOp : Builtin<"rewind"> {
  let summary = "Reset file position to beginning";
  let description = [{
    Corresponds to the `$rewind` system task. Sets the file position for
    the file specified by the file descriptor to the beginning of the file.

    Equivalent to `$fseek(fd, 0, 0)`.

    See IEEE 1800-2017 Section 21.3.3 "File positioning functions".
  }];
  let arguments = (ins TwoValuedI32:$fd);
  let assemblyFormat = "$fd attr-dict";
}

def FReadBIOp : Builtin<"fread"> {
  let summary = "Read binary data from a file";
  let description = [{
    Corresponds to the `$fread` system function. Reads binary data from the
    file specified by the file descriptor into the destination variable.

    Returns the number of bytes read, or 0 if end-of-file or error.

    See IEEE 1800-2017 Section 21.3.3 "File input functions".
  }];
  let arguments = (ins RefType:$dest, TwoValuedI32:$fd);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$dest `,` $fd attr-dict `:` type($dest)";
}

def ReadMemBBIOp : Builtin<"readmemb"> {
  let summary = "Load memory from file in binary format";
  let description = [{
    Corresponds to the `$readmemb` system task. Reads binary data from the
    specified file and loads it into the memory array. The file contains
    whitespace-separated binary values (0, 1, x, z).

    See IEEE 1800-2017 Section 21.4 "Loading memory array data from a file".
  }];
  let arguments = (ins StringType:$filename, RefType:$mem);
  let assemblyFormat = "$filename `,` $mem attr-dict `:` type($mem)";
}

def ReadMemHBIOp : Builtin<"readmemh"> {
  let summary = "Load memory from file in hexadecimal format";
  let description = [{
    Corresponds to the `$readmemh` system task. Reads hexadecimal data from
    the specified file and loads it into the memory array. The file contains
    whitespace-separated hexadecimal values.

    See IEEE 1800-2017 Section 21.4 "Loading memory array data from a file".
  }];
  let arguments = (ins StringType:$filename, RefType:$mem);
  let assemblyFormat = "$filename `,` $mem attr-dict `:` type($mem)";
}

def WriteMemBBIOp : Builtin<"writememb"> {
  let summary = "Write memory to file in binary format";
  let description = [{
    Corresponds to the `$writememb` system task. Writes the contents of
    a memory array to the specified file in binary format, one value per line.

    See IEEE 1800-2017 Section 21.4 "Loading memory array data from a file".
  }];
  let arguments = (ins StringType:$filename, RefType:$mem);
  let assemblyFormat = "$filename `,` $mem attr-dict `:` type($mem)";
}

def WriteMemHBIOp : Builtin<"writememh"> {
  let summary = "Write memory to file in hexadecimal format";
  let description = [{
    Corresponds to the `$writememh` system task. Writes the contents of
    a memory array to the specified file in hexadecimal format, one value
    per line.

    See IEEE 1800-2017 Section 21.4 "Loading memory array data from a file".
  }];
  let arguments = (ins StringType:$filename, RefType:$mem);
  let assemblyFormat = "$filename `,` $mem attr-dict `:` type($mem)";
}

//===----------------------------------------------------------------------===//
// OS System Functions
//===----------------------------------------------------------------------===//

def SystemBIOp : Builtin<"system"> {
  let summary = "Execute an OS command via the system shell";
  let description = [{
    Corresponds to the `$system` system function (IEEE 1800-2017 Section 20.18.1).
    Passes the command string to the host OS command interpreter (equivalent to
    the C `system()` function).

    Returns the exit status of the command as a 32-bit integer. When called
    without arguments, the behavior is implementation-defined.

    Example:
    ```
    int result = $system("echo hello");
    $system("ls -la");
    ```
  }];
  let arguments = (ins Optional<StringType>:$command);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "($command^)? attr-dict";
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
    : MooreOp<"class.classdecl", [Symbol, SymbolTable,
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
// Coverage Operations
//===----------------------------------------------------------------------===//

def CovergroupDeclOp
    : MooreOp<"covergroup.decl", [Symbol, SymbolTable, IsolatedFromAbove,
                                  NoTerminator, SingleBlock]> {
  let summary = "Covergroup declaration";
  let description = [{
    Declares a covergroup type. A covergroup defines a set of coverpoints
    and cross coverage items that track functional coverage during simulation.

    The optional `sampling_event` attribute describes the event that triggers
    coverage sampling (e.g., "@(posedge clk)").

    Coverage options (IEEE 1800-2017 Section 19.7):
    - `weight`: Relative importance for coverage calculation (default: 1)
    - `goal`: Target coverage percentage (default: 100)
    - `comment`: Documentation string
    - `per_instance`: When true, track coverage per instance (default: false)
    - `at_least`: Minimum hits for a bin to be covered (default: 1)
    - `cross_num_print_missing`: Max missing cross products to print (default: 0)
    - `strobe`: When true, sample at the end of the time slot (default: false)

    Type options (set via type_option.*) affect all instances:
    - `type_weight`, `type_goal`, `type_comment`

    Example:
    ```mlir
    moore.covergroup.decl @my_cg sampling_event<"@(posedge clk)">
        per_instance<true> goal<90> {
      moore.coverpoint.decl @data_cp : !moore.l8 {}
    }
    ```

    See IEEE 1800-2017 Section 19 "Functional coverage".
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sampling_event,
                       OptionalAttr<I64Attr>:$weight,
                       OptionalAttr<I64Attr>:$goal,
                       OptionalAttr<StrAttr>:$comment,
                       UnitAttr:$per_instance,
                       OptionalAttr<I64Attr>:$at_least,
                       OptionalAttr<I64Attr>:$cross_num_print_missing,
                       UnitAttr:$strobe,
                       OptionalAttr<I64Attr>:$type_weight,
                       OptionalAttr<I64Attr>:$type_goal,
                       OptionalAttr<StrAttr>:$type_comment);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name (`sampling_event` `<` $sampling_event^ `>`)?
    (`weight` `<` $weight^ `>`)?
    (`goal` `<` $goal^ `>`)?
    (`comment` `<` $comment^ `>`)?
    (`per_instance` $per_instance^)?
    (`at_least` `<` $at_least^ `>`)?
    (`cross_num_print_missing` `<` $cross_num_print_missing^ `>`)?
    (`strobe` $strobe^)?
    (`type_weight` `<` $type_weight^ `>`)?
    (`type_goal` `<` $type_goal^ `>`)?
    (`type_comment` `<` $type_comment^ `>`)?
    attr-dict-with-keyword $body
  }];
}

def CoverpointDeclOp
    : MooreOp<"coverpoint.decl", [Symbol, SymbolTable,
                                  HasParent<"CovergroupDeclOp">,
                                  NoTerminator, SingleBlock]> {
  let summary = "Coverpoint declaration";
  let description = [{
    Declares a coverpoint within a covergroup. A coverpoint tracks coverage
    of values or expressions. It may optionally contain bins that specify
    how values should be grouped for coverage tracking.

    The optional `iff` attribute specifies a condition that must be true for
    coverage to be recorded at a sample point. This implements the
    "coverpoint expr iff (condition)" syntax from IEEE 1800-2017 Section 19.5.

    Coverage options (IEEE 1800-2017 Section 19.7):
    - `weight`: Relative importance for coverage calculation (default: 1)
    - `goal`: Target coverage percentage (default: 100)
    - `comment`: Documentation string
    - `at_least`: Minimum hits for a bin to be covered (default: 1)
    - `auto_bin_max`: Maximum auto-generated bins (default: 64, 0 = one per value)
    - `detect_overlap`: Report overlapping bins (default: false)
    - `cross_auto_bin_max`: Max auto bins for cross (default: 64)

    Example without bins:
    ```mlir
    moore.coverpoint.decl @data_cp : !moore.l8 {}
    ```

    Example with auto_bin_max:
    ```mlir
    moore.coverpoint.decl @data_cp : !moore.l8 auto_bin_max<16> {}
    ```

    Example with iff condition:
    ```mlir
    moore.coverpoint.decl @data_cp : !moore.l8 iff<"enable"> {}
    ```

    Example with options:
    ```mlir
    moore.coverpoint.decl @data_cp : !moore.l8
        weight<2> goal<95> comment<"Data coverage"> at_least<5> {
      moore.coverbin.decl @low kind<bins> values [0, 1, 2, 3]
      moore.coverbin.decl @high kind<bins> values [4, 5, 6, 7]
    }
    ```

    See IEEE 1800-2017 Section 19.5 "Defining coverage points".
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<StrAttr>:$iff,
                       OptionalAttr<I64Attr>:$weight,
                       OptionalAttr<I64Attr>:$goal,
                       OptionalAttr<StrAttr>:$comment,
                       OptionalAttr<I64Attr>:$at_least,
                       OptionalAttr<I64Attr>:$auto_bin_max,
                       UnitAttr:$detect_overlap,
                       OptionalAttr<I64Attr>:$cross_auto_bin_max);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `:` $type
    (`iff` `<` $iff^ `>`)?
    (`weight` `<` $weight^ `>`)?
    (`goal` `<` $goal^ `>`)?
    (`comment` `<` $comment^ `>`)?
    (`at_least` `<` $at_least^ `>`)?
    (`auto_bin_max` `<` $auto_bin_max^ `>`)?
    (`detect_overlap` $detect_overlap^)?
    (`cross_auto_bin_max` `<` $cross_auto_bin_max^ `>`)?
    attr-dict-with-keyword $body
  }];
}

def CoverageBinKind : I32EnumAttr<"CoverageBinKind", "coverage bin kind", [
  I32EnumAttrCase<"Bins", 0, "bins">,
  I32EnumAttrCase<"IllegalBins", 1, "illegal_bins">,
  I32EnumAttrCase<"IgnoreBins", 2, "ignore_bins">
]> {
  let cppNamespace = "::circt::moore";
}

def TransitionRepeatKind : I32EnumAttr<"TransitionRepeatKind",
    "transition repeat kind in coverage bins", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"Consecutive", 1, "consecutive">,
  I32EnumAttrCase<"Nonconsecutive", 2, "nonconsecutive">,
  I32EnumAttrCase<"GoTo", 3, "goto">
]> {
  let cppNamespace = "::circt::moore";
}

def CoverageBinDeclOp
    : MooreOp<"coverbin.decl", [Symbol, HasParent<"CoverpointDeclOp">]> {
  let summary = "Coverage bin declaration";
  let description = [{
    Declares a coverage bin within a coverpoint. Bins specify how values
    should be grouped for coverage tracking.

    The `kind` attribute specifies the bin type:
    - `bins`: Normal coverage bins that count hits
    - `illegal_bins`: Values that should not occur (violations are reported)
    - `ignore_bins`: Values to exclude from coverage calculation

    The `values` attribute contains an array of integer values or ranges
    that belong to this bin.

    Array bin syntax:
    - `is_array` with no `num_bins`: Creates one bin per value (e.g., `bins x[] = {[0:7]}`)
    - `is_array` with `num_bins`: Creates N bins covering the range (e.g., `bins x[4] = {[0:255]}`)

    Transition bins track state transitions rather than individual values.
    The `transitions` attribute contains an array of transition sequences.
    Each sequence is an array of [value, repeatKind, repeatFrom, repeatTo].
    - repeatKind: 0=none, 1=consecutive[*n], 2=nonconsecutive[=n], 3=goto[->n]
    - repeatFrom, repeatTo: repeat count range (0 if not applicable)

    Example value bins:
    ```mlir
    moore.coverbin.decl @low kind<bins> values [0, 1, 2, 3]
    moore.coverbin.decl @high kind<bins> values [4, 5, 6, 7]
    moore.coverbin.decl @bad kind<illegal_bins> values [255]
    moore.coverbin.decl @auto_bins kind<bins> array values [0, 1, 2, 3, 4, 5, 6, 7]
    moore.coverbin.decl @fixed_bins kind<bins> array num_bins<4> values [0, 1, 2, 3, 4, 5, 6, 7]
    ```

    Example transition bins:
    ```mlir
    // bins idle_to_run = (IDLE => RUN);
    moore.coverbin.decl @idle_to_run kind<bins> transitions [[[0, 0, 0, 0], [1, 0, 0, 0]]]
    // bins cycle = (IDLE => RUN => DONE => IDLE);
    moore.coverbin.decl @cycle kind<bins> transitions [[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]]
    // bins repeat3 = (IDLE [*3] => RUN);
    moore.coverbin.decl @repeat3 kind<bins> transitions [[[0, 1, 3, 3], [1, 0, 0, 0]]]
    ```

    See IEEE 1800-2017 Section 19.5.1 "Defining bins for values".
    See IEEE 1800-2017 Section 19.5.4 "Defining transition bins".
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       CoverageBinKind:$kind,
                       UnitAttr:$is_wildcard,
                       UnitAttr:$is_default,
                       UnitAttr:$is_array,
                       UnitAttr:$is_default_sequence,
                       OptionalAttr<I64Attr>:$num_bins,
                       OptionalAttr<ArrayAttr>:$values,
                       OptionalAttr<ArrayAttr>:$transitions);
  let assemblyFormat = [{
    $sym_name `kind` `<` $kind `>`
    (`wildcard` $is_wildcard^)?
    (`default` $is_default^)?
    (`default_sequence` $is_default_sequence^)?
    (`array` $is_array^)?
    (`num_bins` `<` $num_bins^ `>`)?
    (`values` $values^)?
    (`transitions` $transitions^)?
    attr-dict
  }];
}

def CoverCrossDeclOp
    : MooreOp<"covercross.decl", [Symbol, SymbolTable,
                                  HasParent<"CovergroupDeclOp">,
                                  NoTerminator, SingleBlock]> {
  let summary = "Cross coverage declaration";
  let description = [{
    Declares a cross coverage item within a covergroup. Cross coverage tracks
    the Cartesian product of multiple coverpoints. It may optionally contain
    bins that use binsof/intersect expressions to select specific cross products.

    Coverage options (IEEE 1800-2017 Section 19.7):
    - `weight`: Relative importance for coverage calculation (default: 1)
    - `goal`: Target coverage percentage (default: 100)
    - `comment`: Documentation string
    - `at_least`: Minimum hits for a bin to be covered (default: 1)
    - `cross_num_print_missing`: Max missing cross products to print (default: 0)
    - `cross_auto_bin_max`: Maximum auto-generated cross bins (default: 64)

    Example without bins (automatic bins):
    ```mlir
    moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp] {}
    ```

    Example with bins using binsof/intersect:
    ```mlir
    moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp]
        weight<2> goal<90> {
      moore.crossbin.decl @valid kind<bins> {
        moore.binsof @addr_cp intersect [0, 1, 2, 3]
      }
      moore.crossbin.decl @ignore_high kind<ignore_bins> {
        moore.binsof @addr_cp intersect [250, 251, 252, 253, 254, 255]
      }
    }
    ```

    See IEEE 1800-2017 Section 19.6 "Defining cross coverage".
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       SymbolRefArrayAttr:$targets,
                       OptionalAttr<I64Attr>:$weight,
                       OptionalAttr<I64Attr>:$goal,
                       OptionalAttr<StrAttr>:$comment,
                       OptionalAttr<I64Attr>:$at_least,
                       OptionalAttr<I64Attr>:$cross_num_print_missing,
                       OptionalAttr<I64Attr>:$cross_auto_bin_max);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `targets` $targets
    (`weight` `<` $weight^ `>`)?
    (`goal` `<` $goal^ `>`)?
    (`comment` `<` $comment^ `>`)?
    (`at_least` `<` $at_least^ `>`)?
    (`cross_num_print_missing` `<` $cross_num_print_missing^ `>`)?
    (`cross_auto_bin_max` `<` $cross_auto_bin_max^ `>`)?
    attr-dict-with-keyword $body
  }];
}

def CrossBinDeclOp
    : MooreOp<"crossbin.decl", [Symbol, HasParent<"CoverCrossDeclOp">,
                                NoTerminator, SingleBlock]> {
  let summary = "Cross coverage bin declaration";
  let description = [{
    Declares a bin within a cross coverage item. Cross bins use binsof/intersect
    expressions to select which cross products to include in the bin.

    The `kind` attribute specifies the bin type:
    - `bins`: Normal coverage bins that count hits
    - `illegal_bins`: Values that should not occur (violations are reported)
    - `ignore_bins`: Values to exclude from coverage calculation

    The body contains binsof expressions that define the selection criteria.

    Example:
    ```mlir
    moore.crossbin.decl @valid kind<bins> {
      moore.binsof @addr_cp intersect [0, 1, 2, 3]
    }
    ```

    See IEEE 1800-2017 Section 19.6.1 "Specifying cross coverage bins".
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, CoverageBinKind:$kind);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ $sym_name `kind` `<` $kind `>` attr-dict-with-keyword $body }];
}

def BinsOfOp : MooreOp<"binsof", [HasParent<"CrossBinDeclOp">]> {
  let summary = "Binsof select expression for cross coverage";
  let description = [{
    Represents a binsof expression used in cross coverage bin selection.
    The binsof expression references a coverpoint or coverpoint bin and
    optionally specifies an intersect clause to filter values.

    The optional `negate` attribute implements the `!binsof(...)` syntax,
    which inverts the selection to match values NOT in the specified set.

    Example without intersect (selects all bins of the coverpoint):
    ```mlir
    moore.binsof @addr_cp
    ```

    Example with intersect (selects specific values):
    ```mlir
    moore.binsof @addr_cp intersect [0, 1, 2, 3]
    ```

    Example with negation (selects values NOT in the set):
    ```mlir
    moore.binsof @addr_cp intersect [0, 1, 2, 3] negate
    ```

    Example referencing a specific bin:
    ```mlir
    moore.binsof @addr_cp::@low_bins intersect [0, 1]
    ```

    See IEEE 1800-2017 Section 19.6.1.1 "select_expression".
  }];
  let arguments = (ins SymbolRefAttr:$target,
                       OptionalAttr<ArrayAttr>:$intersect_values,
                       UnitAttr:$negate);
  let assemblyFormat = [{
    $target (`intersect` $intersect_values^)? (`negate` $negate^)? attr-dict
  }];
}

def CovergroupInstOp : MooreOp<"covergroup.inst"> {
  let summary = "Covergroup instantiation";
  let description = [{
    Instantiates a covergroup, allocating runtime data structures for
    tracking coverage. Returns a handle to the covergroup instance.

    Example:
    ```mlir
    %cg = moore.covergroup.inst @my_covergroup : !moore.covergroup<@my_covergroup>
    ```

    This corresponds to SystemVerilog `covergroup_type cg_inst = new();`

    See IEEE 1800-2017 Section 19.3 "Using covergroup in classes".
  }];
  let arguments = (ins FlatSymbolRefAttr:$covergroup);
  let results = (outs CovergroupHandleType:$result);
  let assemblyFormat = [{ $covergroup attr-dict `:` type($result) }];
}

def CovergroupSampleOp : MooreOp<"covergroup.sample",
    [AttrSizedOperandSegments]> {
  let summary = "Sample a covergroup";
  let description = [{
    Triggers sampling of a covergroup instance. Each coverpoint in the
    covergroup will capture its current value for coverage tracking.

    The values operand provides the values to sample for each coverpoint
    in the order they are declared in the covergroup.

    The optional iff_conditions operand provides per-coverpoint guard
    conditions. When present, the iff condition at index i corresponds to
    the coverpoint value at index i. A coverpoint is only sampled when
    its iff condition is nonzero (or when no iff condition is provided).

    Example:
    ```mlir
    moore.covergroup.sample %cg_inst(%val1, %val2) : !moore.covergroup<@cg>(!moore.l8, !moore.l4)
    ```

    Example with iff conditions:
    ```mlir
    moore.covergroup.sample %cg_inst(%val1, %val2) iff(%cond1, %cond2) : !moore.covergroup<@cg>(!moore.l8, !moore.l4) iff(!moore.i1, !moore.i1)
    ```

    This corresponds to SystemVerilog `cg_inst.sample(args)` or implicit
    sampling triggered by the covergroup's sampling event.

    See IEEE 1800-2017 Section 19.8 "Covergroup methods".
  }];
  let arguments = (ins CovergroupHandleType:$covergroup,
                       Variadic<AnyType>:$values,
                       Variadic<AnyType>:$iff_conditions);
  let assemblyFormat = [{
    $covergroup `(` $values `)` (`iff` `(` $iff_conditions^ `)`)? attr-dict
    `:` type($covergroup) `(` type($values) `)`
    (`iff` `(` type($iff_conditions)^ `)`)?
  }];
}

def CovergroupGetCoverageOp : MooreOp<"covergroup.get_coverage", [Pure]> {
  let summary = "Get coverage percentage of a covergroup";
  let description = [{
    Returns the current coverage percentage (0.0 to 100.0) of a covergroup
    instance.

    Example:
    ```mlir
    %coverage = moore.covergroup.get_coverage %cg_inst : !moore.covergroup<@cg> -> !moore.real
    ```

    This corresponds to SystemVerilog `cg_inst.get_coverage()`.

    See IEEE 1800-2017 Section 19.8.1 "Coverage methods".
  }];
  let arguments = (ins CovergroupHandleType:$covergroup);
  let results = (outs RealType:$result);
  let assemblyFormat = [{
    $covergroup attr-dict `:` type($covergroup) `->` type($result)
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

    If `is_soft` is set, this is a soft constraint that provides a default
    value which can be overridden by hard constraints. Soft constraints are
    only applied when no conflicting hard constraint exists.

    Example:
    ```mlir
    moore.constraint.block @c1 {
      %cond = moore.lt %a, %b : i32
      moore.constraint.expr %cond
    }

    // Soft constraint example:
    moore.constraint.block @soft_c {
      %eq = moore.eq %value, %const : i32
      moore.constraint.expr %eq soft
    }
    ```

    See IEEE 1800-2017 Section 18.5.1 "Constraint expressions".
    See IEEE 1800-2017 Section 18.5.13 "Soft constraints".
  }];

  let arguments = (ins AnySingleBitType:$condition, UnitAttr:$is_soft);
  let results = (outs);
  let assemblyFormat = [{
    $condition attr-dict `:` type($condition) (`soft` $is_soft^)?
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
      moore.constraint.implication %mode_is_fast : i1 {
        // constraints that only apply when mode_is_fast is true
        moore.constraint.expr %speed_limit : i1
      }
    }
    ```

    See IEEE 1800-2017 Section 18.5.6 "Implication constraints".
  }];

  let arguments = (ins AnySingleBitType:$antecedent);
  let results = (outs);
  let regions = (region SizedRegion<1>:$consequent);
  let assemblyFormat = [{
    $antecedent `:` type($antecedent) attr-dict-with-keyword $consequent
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
      moore.constraint.if_else %is_write : i1 {
        // constraints for write mode
      } else {
        // constraints for read mode
      }
    }
    ```

    See IEEE 1800-2017 Section 18.5.7 "if-else constraints".
  }];

  let arguments = (ins AnySingleBitType:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$then_region, AnyRegion:$else_region);
  let assemblyFormat = [{
    $condition `:` type($condition) $then_region (`else` $else_region^)? attr-dict
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
                       DenseI64ArrayAttr:$per_range,
                       OptionalAttr<UnitAttr>:$isSigned);
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

    If `is_soft` is set, this is a soft constraint that provides default
    value ranges which can be overridden by hard constraints.

    Example:
    ```mlir
    // x inside { 1, [3:5], 7 }
    moore.constraint.inside %x, [[1, 1], [3, 5], [7, 7]] : i8

    // Soft inside constraint (default range, can be overridden)
    moore.constraint.inside %x, [0, 100] : i32 soft
    ```

    See IEEE 1800-2017 Section 18.5.3 "Set membership constraints".
    See IEEE 1800-2017 Section 18.5.13 "Soft constraints".
  }];

  let arguments = (ins AnyType:$variable,
                       DenseI64ArrayAttr:$ranges,
                       UnitAttr:$is_soft);
  let results = (outs);
  let assemblyFormat = [{
    $variable `,` $ranges `:` type($variable) (`soft` $is_soft^)? attr-dict
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

def ConstraintDisableSoftOp
    : MooreOp<"constraint.disable_soft", []> {
  let summary = "Disable soft constraints on a variable";
  let description = [{
    Disables all soft constraints on the specified random variable within the
    current constraint block. This allows derived classes or later constraints
    to override soft constraints defined earlier or in base classes.

    The target operand should be a reference to a random variable (rand/randc).

    Example:
    ```mlir
    // disable soft b;
    %b_ref = moore.class.property_ref %this[@b] : <@MyClass> -> <i32>
    %b_val = moore.read %b_ref : <i32>
    moore.constraint.disable_soft %b_val : i32
    ```

    See IEEE 1800-2017 Section 18.5.14.2 "Discarding soft constraints".
  }];

  let arguments = (ins AnyType:$target);
  let results = (outs);
  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];
}

def ConstraintDisableOp
    : MooreOp<"constraint.disable", [HasParent<"ConstraintBlockOp">]> {
  let summary = "Disable a named constraint block";
  let description = [{
    Disables a named constraint block by symbol reference. This is used within
    a constraint block to disable another constraint block (typically to override
    soft constraints or base class constraints).

    Example:
    ```mlir
    // Disable the soft_limit constraint block
    moore.constraint.block @override {
      moore.constraint.disable @soft_limit
    }
    ```

    See IEEE 1800-2017 Section 18.5.14 "Disabling constraints".
  }];

  let arguments = (ins SymbolRefAttr:$constraint_name);
  let results = (outs);
  let assemblyFormat = [{
    $constraint_name attr-dict
  }];
}

def ConstraintUniqueOp
    : MooreOp<"constraint.unique", []> {
  let summary = "Unique constraint for variables or array elements";
  let description = [{
    Specifies that all provided variables/elements must have unique values
    after randomization. Can be used with:
    - Individual variables: unique { a, b, c } ensures a, b, c are all different
    - Arrays: unique { arr } ensures all elements of arr are different

    Example:
    ```mlir
    // unique { a, b, c } - individual variables
    moore.constraint.unique %a, %b, %c : i8, i8, i8

    // unique { arr } - array elements
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

def ConstraintMethodCallOp
    : MooreOp<"constraint.method_call", [Pure]> {
  let summary = "Method call inside a constraint expression";
  let description = [{
    Represents a method call within a constraint expression. Unlike func.call,
    this operation uses a qualified method symbol that can be resolved across
    class boundaries. The method is specified by a symbol reference to the
    containing class and the method name.

    The object argument is the class instance on which the method is called.
    Arguments are the method's input parameters.

    Example:
    ```mlir
    // Call method F on class instance with one argument
    %result = moore.constraint.method_call @"a::F"(%this, %arg0) :
              (!moore.class<@a>, !moore.i32) -> !moore.i32
    ```

    See IEEE 1800-2017 Section 18.5.12 "Functions in constraints".
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$method,
    Variadic<AnyType>:$operands
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $method `(` $operands `)` `:` functional-type($operands, $result) attr-dict
  }];
}

def RandomizeOp
    : MooreOp<"randomize", [SingleBlock, NoTerminator]> {
  let summary = "Randomize an object";
  let description = [{
    Invokes the randomize() method on a class object. This operation
    assigns random values to all `rand` and `randc` properties while
    satisfying all active constraints.

    The result is a 1-bit value indicating success (1) or failure (0).

    An optional `inline_constraints` region can contain inline constraints
    specified with a `with` clause. These constraints are combined with
    the class's declared constraints during randomization.

    Optional modes:
    - `check_only`: When true, only checks if constraints can be satisfied
      without actually randomizing. Used for `randomize(null)`.
      See IEEE 1800-2017 Section 18.11.1.
    - `variable_list`: When specified, only these variables are treated as
      random for this call. Used for `randomize(v, w)`.
      See IEEE 1800-2017 Section 18.11.

    Example:
    ```mlir
    // Without inline constraints:
    %success = moore.randomize %obj : !moore.class<@MyClass>

    // With inline constraints (randomize() with { x > 0; }):
    %success = moore.randomize %obj : !moore.class<@MyClass> {
      %0 = moore.class.property_ref %obj[@x] : !moore.class<@MyClass> -> !moore.ref<i32>
      %1 = moore.read %0 : <i32>
      %2 = moore.constant 0 : i32
      %3 = moore.sgt %1, %2 : i32 -> i1
      moore.constraint.expr %3 : i1
    }

    // Check-only mode (randomize(null)):
    %success = moore.randomize %obj check_only : !moore.class<@MyClass>

    // With variable list (randomize(v, w)):
    %success = moore.randomize %obj variable_list([@v, @w]) : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.6 "Randomization methods".
    See IEEE 1800-2017 Section 18.7 "In-line constraints".
    See IEEE 1800-2017 Section 18.11 "In-line random variable control".
  }];

  let arguments = (ins ClassHandleType:$object,
                       UnitAttr:$check_only,
                       OptionalAttr<ArrayAttr>:$variable_list);
  let results = (outs I1:$success);
  let regions = (region AnyRegion:$inline_constraints);
  let assemblyFormat = [{
    $object (`check_only` $check_only^)?
            (`variable_list` `(` $variable_list^ `)`)?
            `:` type($object) attr-dict-with-keyword ($inline_constraints^)?
  }];

}

def StdRandomizeOp
    : MooreOp<"std_randomize", [SingleBlock, NoTerminator]> {
  let summary = "Scope randomize function (std::randomize)";
  let description = [{
    Randomizes one or more variables passed by reference. This is the
    standalone randomization function `std::randomize()` as opposed to
    the class method `obj.randomize()`.

    The result is a 1-bit value indicating success (1) or failure (0).

    An optional `inline_constraints` region can contain inline constraints
    specified with a `with` clause.

    Example:
    ```mlir
    // Without inline constraints:
    %success = moore.std_randomize %x, %y : !moore.ref<i32>, !moore.ref<i32>

    // With inline constraints (std::randomize(x, y) with { x < y; }):
    %success = moore.std_randomize %x, %y : !moore.ref<i32>, !moore.ref<i32> {
      %0 = moore.read %x : <i32>
      %1 = moore.read %y : <i32>
      %2 = moore.slt %0, %1 : i32 -> i1
      moore.constraint.expr %2 : i1
    }
    ```

    See IEEE 1800-2017 Section 18.12 "Scope randomize function".
  }];

  let arguments = (ins Variadic<RefType>:$variables);
  let results = (outs I1:$success);
  let regions = (region AnyRegion:$inline_constraints);
  let assemblyFormat = [{
    $variables `:` type($variables) attr-dict-with-keyword ($inline_constraints^)?
  }];

}

def ConstraintModeOp
    : MooreOp<"constraint_mode", []> {
  let summary = "Get or set constraint mode";
  let description = [{
    Controls whether a constraint is enabled or disabled during randomization.
    SystemVerilog supports two forms:

    1. Class-level: `obj.constraint_mode(0)` - Disable all constraints in the object
    2. Constraint-level: `obj.c_name.constraint_mode(0)` - Disable specific constraint

    When called with an argument (mode), sets the constraint mode:
    - 0: Disabled (constraint is ignored during randomization)
    - 1: Enabled (constraint is active during randomization)

    When called without an argument, returns the current constraint mode.

    The result is an integer:
    - When setting: Returns the previous mode (0 or 1)
    - When getting: Returns the current mode (0 or 1)

    Example:
    ```mlir
    // Disable all constraints on an object
    %prev = moore.constraint_mode set(%obj) mode(0) : !moore.class<@MyClass>

    // Disable a specific constraint
    %prev = moore.constraint_mode set(%obj, @c_valid) mode(0) : !moore.class<@MyClass>

    // Get the current mode of a specific constraint
    %mode = moore.constraint_mode get(%obj, @c_valid) : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.8 "Disabling random variables and constraints".
  }];

  let arguments = (ins ClassHandleType:$object,
                       OptionalAttr<FlatSymbolRefAttr>:$constraint,
                       Optional<IntType>:$mode);
  let results = (outs IntType:$result);
  // Use generic assembly format to avoid optional group restrictions.
}

def RandModeOp
    : MooreOp<"rand_mode", []> {
  let summary = "Get or set rand mode";
  let description = [{
    Controls whether random variables are enabled or disabled during
    randomization. SystemVerilog supports two forms:

    1. Class-level: `obj.rand_mode(0)` - Disable all random variables in the object
    2. Property-level: `obj.var.rand_mode(0)` - Disable a specific random variable

    When called with an argument (mode), sets the rand mode:
    - 0: Disabled (variable is not randomized)
    - 1: Enabled (variable is randomized)

    When called without an argument, returns the current rand mode.

    The result is an integer:
    - When setting: Returns the previous mode (0 or 1)
    - When getting: Returns the current mode (0 or 1)

    Example:
    ```mlir
    // Disable all random variables on an object
    %prev = moore.rand_mode set(%obj) mode(0) : !moore.class<@MyClass>

    // Disable a specific random variable
    %prev = moore.rand_mode set(%obj, @addr) mode(0) : !moore.class<@MyClass>

    // Get the current mode of a specific variable
    %mode = moore.rand_mode get(%obj, @addr) : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.8 "Disabling random variables and constraints".
  }];

  let arguments = (ins ClassHandleType:$object,
                       OptionalAttr<FlatSymbolRefAttr>:$property,
                       Optional<IntType>:$mode);
  let results = (outs IntType:$result);
  // Use generic assembly format to avoid optional group restrictions.
}

def CallPreRandomizeOp
    : MooreOp<"call_pre_randomize", []> {
  let summary = "Call pre_randomize callback on a class object";
  let description = [{
    Invokes the pre_randomize() method on a class object. This method is
    automatically called before randomization begins. User classes can
    override this method to perform setup before randomization.

    The pre_randomize() method is a virtual function that returns void
    by default. If a class overrides it with a different signature,
    slang will report an error.

    Example:
    ```mlir
    moore.call_pre_randomize %obj : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
  }];

  let arguments = (ins ClassHandleType:$object);
  let results = (outs);
  let assemblyFormat = [{
    $object `:` type($object) attr-dict
  }];
}

def CallPostRandomizeOp
    : MooreOp<"call_post_randomize", []> {
  let summary = "Call post_randomize callback on a class object";
  let description = [{
    Invokes the post_randomize() method on a class object. This method is
    automatically called after randomization succeeds. User classes can
    override this method to perform post-processing after randomization.

    The post_randomize() method is a virtual function that returns void
    by default. If a class overrides it with a different signature,
    slang will report an error.

    The `success` operand gates the call: post_randomize is only invoked
    when randomization succeeds (IEEE 1800-2017 Section 18.6.3).

    Example:
    ```mlir
    moore.call_post_randomize %obj, %success : !moore.class<@MyClass>
    ```

    See IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
  }];

  let arguments = (ins ClassHandleType:$object, I1:$success);
  let results = (outs);
  let assemblyFormat = [{
    $object `,` $success `:` type($object) attr-dict
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

def ClassCopyOp
    : MooreOp<
          "class.copy", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          TypesMatchWith<"source and result have the same type",
                        "result", "source", "$_self">]> {
  let summary = "Create a shallow copy of a class instance";
  let description = [{
    Creates a new instance of a class by shallow copying an existing instance.
    This implements the SystemVerilog `new <source>` syntax for class shallow
    copy (IEEE 1800-2017 Section 8.12).

    The new object has the same values for all class properties as the source
    object at the time of the copy. Nested class objects are not deep-copied;
    both the original and the copy will reference the same nested objects.

    Example:
    ```mlir
    %copy = moore.class.copy %source : <@MyClass>
    ```
  }];

  let arguments = (ins ClassHandleType:$source);
  let results = (outs ClassHandleType:$result);

  let assemblyFormat = [{
     $source `:` type($result) attr-dict
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

def ClassDynCastOp
    : MooreOp<"class.dyn_cast", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,]> {
  let summary = "Dynamic cast (downcast) a class handle with runtime type check";
  let description = [{
    Implements the SystemVerilog $cast system function for class handles.
    Attempts to downcast a class handle to a derived type at runtime.

    Returns two values:
    - `result`: The casted handle (valid only if cast succeeds)
    - `success`: An i1 indicating whether the cast succeeded (1) or failed (0)

    IEEE 1800-2017 Section 8.16: "$cast shall attempt to assign the source
    expression to the destination variable. When used as a function, $cast
    returns 1 if the cast is legal, and 0 if the cast fails."

    This operation is essential for UVM's dynamic type checking and factory
    pattern, where components need to be downcast from base handles to their
    actual derived types.

    Example:
    ```mlir
    %result, %success = moore.class.dyn_cast %src : !moore.class<@Base> to !moore.class<@Derived>
    ```
  }];
  let arguments = (ins ClassHandleType:$source);
  let results = (outs ClassHandleType:$result, I1:$success);
  let assemblyFormat =
      "$source `:` type($source) `to` type($result) attr-dict";
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
    TypeAttr:$type,
    OptionalAttr<UnitAttr>:$interface_instance
  );
  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Type getSignalType() { return getTypeAttr().getValue(); }
    bool isInterfaceInstance() { return getInterfaceInstanceAttr() != nullptr; }
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

def VirtualInterfaceCmpPredicateAttr : I64EnumAttr<
    "VirtualInterfaceCmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>
    ]> {
  let cppNamespace = "circt::moore";
}

def VirtualInterfaceNullOp : MooreOp<"virtual_interface.null", [Pure]> {
  let summary = "Create a null virtual interface handle";
  let description = [{
    Creates a null virtual interface value. In SystemVerilog, `null` represents
    the absence of a valid interface instance reference. Comparing a virtual
    interface to `null` checks whether the handle points to a valid interface.

    Example:
    ```mlir
    %null = moore.virtual_interface.null : !moore.virtual_interface<@my_bus>
    ```

    See IEEE 1800-2017 Section 25.9 "Virtual interfaces".
  }];

  let arguments = (ins);
  let results = (outs VirtualInterfaceType:$result);

  let assemblyFormat = [{
    `:` type($result) attr-dict
  }];
}

def VirtualInterfaceCmpOp : MooreOp<"virtual_interface_cmp", [
  Pure,
  Commutative,
  SameTypeOperands
]> {
  let summary = "Compare virtual interfaces for reference equality";
  let description = [{
    Compares two virtual interfaces using reference equality. This checks whether
    two handles point to the same interface instance (or are both null).
    Supported predicates:
    - eq : equal (handles point to same interface or both null)
    - ne : not equal (handles point to different interfaces)

    Example:
    ```mlir
    %result = moore.virtual_interface_cmp eq %a, %b : !moore.virtual_interface<@my_bus>
    ```

    See IEEE 1800-2017 Section 25.9 "Virtual interfaces".
  }];
  let arguments = (ins
    VirtualInterfaceCmpPredicateAttr:$predicate,
    VirtualInterfaceType:$lhs,
    VirtualInterfaceType:$rhs
  );
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def VirtualInterfaceBindOp : MooreOp<"virtual_interface.bind", []> {
  let summary = "Bind an interface instance to a virtual interface variable";
  let description = [{
    Associates a virtual interface variable (stored at the destination reference)
    with an actual interface instance. This is the runtime binding operation
    that enables UVM-style virtual interface patterns.

    The destination must be a reference to a virtual interface type (typically
    a class property), and the source must be either:
    - A virtual interface value (from reading another virtual interface)
    - A reference to an interface instance (from `moore.interface.instance`)

    When the source is a reference to an interface instance, the binding
    operation implicitly converts it to a virtual interface value by probing
    the reference to get the interface pointer.

    Example:
    ```mlir
    // Bind interface instance to class property
    %vif_ref = moore.class.property_ref %obj[@vif] : ...
    %intf = moore.interface.instance @my_bus : !moore.ref<virtual_interface<@my_bus>>
    moore.virtual_interface.bind %vif_ref, %intf : !moore.ref<virtual_interface<@my_bus>>, !moore.ref<virtual_interface<@my_bus>>

    // Bind virtual interface value to class property
    %vif_val = moore.read %other_vif : <virtual_interface<@my_bus>>
    moore.virtual_interface.bind %vif_ref, %vif_val : !moore.ref<virtual_interface<@my_bus>>, !moore.virtual_interface<@my_bus>
    ```

    See IEEE 1800-2017 Section 25.9 "Virtual interfaces".
  }];
  let arguments = (ins
    VirtualInterfaceRefType:$dest,
    AnyType:$source
  );
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $source attr-dict `:` type($dest) `,` type($source)
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
// Array Builtins
//===----------------------------------------------------------------------===//

class ArrayBuiltin<string mnemonic, list<Trait> traits = []> :
    MooreOp<"array." # mnemonic, traits>;

def ArraySizeOp : ArrayBuiltin<"size"> {
  let summary = "Get the size of a dynamic array, associative array, or queue";
  let description = [{
    Returns the number of elements in the given array. This corresponds to the
    SystemVerilog `.size()` built-in method on dynamic arrays, associative
    arrays, and queues.

    For associative arrays, this is equivalent to calling `.num()`.

    Example:
    ```mlir
    %size = moore.array.size %queue : !moore.queue<i32, 0> -> i32
    ```
  }];
  let arguments = (ins AnyDynamicArrayType:$array);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$array `:` type($array) attr-dict";
}

def ArrayContainsOp : ArrayBuiltin<"contains", [Pure]> {
  let summary = "Check if an unpacked array contains a value";
  let description = [{
    Returns true (1) if the given value is found in the array, false (0)
    otherwise. This corresponds to checking membership in the SystemVerilog
    `inside` operator when the right-hand side is an unpacked array.

    The comparison uses wildcard equality semantics (X/Z bits match anything).

    This operation supports static unpacked arrays, dynamic arrays, and queues.

    Example:
    ```mlir
    // Check if arr contains value 42
    %found = moore.array.contains %arr, %value : !moore.uarray<10 x i32>, i32
    %found = moore.array.contains %dyn_arr, %value : !moore.open_uarray<i32>, i32
    %found = moore.array.contains %queue, %value : !moore.queue<i32, 0>, i32
    ```
  }];
  let arguments = (ins AnyUnpackedArrayType:$array, UnpackedType:$value);
  let results = (outs TwoValuedI1:$result);
  let assemblyFormat = "$array `,` $value `:` type($array) `,` type($value) attr-dict";
}

def DynArrayNewOp : MooreOp<"dyn_array.new",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Create a new dynamic array with a given size";
  let description = [{
    Creates a new dynamic array with the specified number of elements. This
    corresponds to the SystemVerilog `new[size]` expression for dynamic arrays.

    If the optional `init` operand is provided, the existing elements from that
    array are copied into the new array, corresponding to `new[size](init)`.

    Example:
    ```mlir
    // arr = new[10];
    %arr = moore.dyn_array.new %size : i32 -> !moore.open_uarray<i32>

    // arr = new[20](existing);
    %arr2 = moore.dyn_array.new %size, %existing : i32, !moore.open_uarray<i32> -> !moore.open_uarray<i32>
    ```
  }];
  let arguments = (ins TwoValuedI32:$size,
                       Optional<OpenUnpackedArrayType>:$init);
  let results = (outs OpenUnpackedArrayType:$result);
  let assemblyFormat = [{
    $size (`,` $init^)? `:` type($size) (`,` type($init)^)? `->` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Queue Builtins
//===----------------------------------------------------------------------===//

class QueueBuiltin<string mnemonic, list<Trait> traits = []> :
    MooreOp<"queue." # mnemonic, traits>;

def QueuePushBackOp : QueueBuiltin<"push_back"> {
  let summary = "Push an element to the back of a queue";
  let description = [{
    Appends an element to the back of a queue. This corresponds to the
    SystemVerilog `.push_back()` built-in method on queues.

    Example:
    ```mlir
    moore.queue.push_back %queue, %elem : !moore.ref<!moore.queue<i32, 0>>, i32
    ```
  }];
  let arguments = (ins RefType:$queue, UnpackedType:$element);
  let assemblyFormat = "$queue `,` $element `:` type($queue) `,` type($element) attr-dict";
}

def QueuePushFrontOp : QueueBuiltin<"push_front"> {
  let summary = "Push an element to the front of a queue";
  let description = [{
    Prepends an element to the front of a queue. This corresponds to the
    SystemVerilog `.push_front()` built-in method on queues.

    Example:
    ```mlir
    moore.queue.push_front %queue, %elem : !moore.ref<!moore.queue<i32, 0>>, i32
    ```
  }];
  let arguments = (ins RefType:$queue, UnpackedType:$element);
  let assemblyFormat = "$queue `,` $element `:` type($queue) `,` type($element) attr-dict";
}

def QueueInsertOp : QueueBuiltin<"insert"> {
  let summary = "Insert an element into a queue";
  let description = [{
    Inserts an element into a queue at the given index, shifting any subsequent
    elements. This corresponds to the SystemVerilog `.insert(index, element)`
    built-in method on queues.

    Example:
    ```mlir
    moore.queue.insert %queue, %idx, %elem : !moore.ref<!moore.queue<i32, 0>>, i32, i32
    ```
  }];
  let arguments = (ins RefType:$queue, IntType:$index, UnpackedType:$element);
  let assemblyFormat = "$queue `,` $index `,` $element `:` type($queue) `,` type($index) `,` type($element) attr-dict";
}

def QueuePopBackOp : QueueBuiltin<"pop_back"> {
  let summary = "Pop an element from the back of a queue";
  let description = [{
    Removes and returns the last element from a queue. This corresponds to the
    SystemVerilog `.pop_back()` built-in method on queues.

    Example:
    ```mlir
    %elem = moore.queue.pop_back %queue : !moore.ref<!moore.queue<i32, 0>> -> i32
    ```
  }];
  let arguments = (ins RefType:$queue);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = "$queue `:` type($queue) `->` type($result) attr-dict";
}

def QueuePopFrontOp : QueueBuiltin<"pop_front"> {
  let summary = "Pop an element from the front of a queue";
  let description = [{
    Removes and returns the first element from a queue. This corresponds to the
    SystemVerilog `.pop_front()` built-in method on queues.

    Example:
    ```mlir
    %elem = moore.queue.pop_front %queue : !moore.ref<!moore.queue<i32, 0>> -> i32
    ```
  }];
  let arguments = (ins RefType:$queue);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = "$queue `:` type($queue) `->` type($result) attr-dict";
}

def QueueDeleteOp : QueueBuiltin<"delete"> {
  let summary = "Delete elements from a queue";
  let description = [{
    Removes elements from a queue. If an index is provided, removes only the
    element at that index (and shifts remaining elements). If no index is
    provided, removes all elements. This corresponds to the SystemVerilog
    `.delete()` and `.delete(index)` built-in methods on queues.

    Example:
    ```mlir
    // Delete all elements
    moore.queue.delete %queue : !moore.ref<!moore.queue<i32, 0>>
    // Delete element at index
    moore.queue.delete %queue[%idx] : !moore.ref<!moore.queue<i32, 0>>, i32
    ```
  }];
  let arguments = (ins RefType:$queue, Optional<IntType>:$index);
  let assemblyFormat = [{
    $queue (`[` $index^ `]`)? `:` type($queue) (`,` type($index)^)? attr-dict
  }];
}

def QueueSortOp : QueueBuiltin<"sort"> {
  let summary = "Sort the elements of a queue";
  let description = [{
    Sorts the elements of a queue in ascending order. This corresponds to the
    SystemVerilog `.sort()` built-in method on queues and dynamic arrays.

    Example:
    ```mlir
    moore.queue.sort %queue : !moore.ref<!moore.queue<i32, 0>>
    ```
  }];
  let arguments = (ins RefType:$queue);
  let assemblyFormat = "$queue `:` type($queue) attr-dict";
}

def QueueRSortOp : QueueBuiltin<"rsort"> {
  let summary = "Sort the elements of a queue in descending order";
  let description = [{
    Sorts the elements of a queue in descending order. This corresponds to the
    SystemVerilog `.rsort()` built-in method on queues and dynamic arrays.

    Example:
    ```mlir
    moore.queue.rsort %queue : !moore.ref<!moore.queue<i32, 0>>
    ```
  }];
  let arguments = (ins RefType:$queue);
  let assemblyFormat = "$queue `:` type($queue) attr-dict";
}

def QueueSortWithOp : QueueBuiltin<"sort.with", [
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Sort the elements of a queue with a custom key expression";
  let description = [{
    Sorts the elements of a queue in ascending order using a custom key
    expression. This corresponds to the SystemVerilog `.sort() with (expr)`
    built-in method on queues and dynamic arrays.

    The operation takes a queue reference and contains a single-block region
    that defines the key expression. The region has a block argument
    representing the current element being evaluated. The region must terminate
    with a `moore.queue.sort.key.yield` operation that yields the sort key value.

    Example:
    ```mlir
    // q.sort() with (item % 10) - sorts by the value modulo 10
    moore.queue.sort.with %queue : !moore.ref<!moore.queue<i32, 0>> {
    ^bb0(%item: !moore.i32):
      %ten = moore.constant 10 : i32
      %key = moore.smod %item, %ten : i32
      moore.queue.sort.key.yield %key : i32
    }
    ```

    See IEEE 1800-2017 Section 7.12.3 "Array ordering methods".
  }];
  let arguments = (ins RefType:$queue);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$queue `:` type($queue) $body attr-dict";
}

def QueueRSortWithOp : QueueBuiltin<"rsort.with", [
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Sort the elements of a queue in descending order with a custom key expression";
  let description = [{
    Sorts the elements of a queue in descending order using a custom key
    expression. This corresponds to the SystemVerilog `.rsort() with (expr)`
    built-in method on queues and dynamic arrays.

    The operation takes a queue reference and contains a single-block region
    that defines the key expression. The region has a block argument
    representing the current element being evaluated. The region must terminate
    with a `moore.queue.sort.key.yield` operation that yields the sort key value.

    Example:
    ```mlir
    // q.rsort() with (item % 10) - sorts descending by the value modulo 10
    moore.queue.rsort.with %queue : !moore.ref<!moore.queue<i32, 0>> {
    ^bb0(%item: !moore.i32):
      %ten = moore.constant 10 : i32
      %key = moore.smod %item, %ten : i32
      moore.queue.sort.key.yield %key : i32
    }
    ```

    See IEEE 1800-2017 Section 7.12.3 "Array ordering methods".
  }];
  let arguments = (ins RefType:$queue);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$queue `:` type($queue) $body attr-dict";
}

def QueueSortKeyYieldOp : MooreOp<"queue.sort.key.yield", [
  Pure,
  Terminator,
  HasParent<"QueueSortWithOp, QueueRSortWithOp">,
]> {
  let summary = "Yield a sort key value from a queue sort.with body";
  let description = [{
    Terminates the body region of a `moore.queue.sort.with` or
    `moore.queue.rsort.with` operation and yields the sort key value
    for the current element.

    Example:
    ```mlir
    moore.queue.sort.key.yield %key : i32
    ```
  }];
  let arguments = (ins UnpackedType:$key);
  let assemblyFormat = "$key `:` type($key) attr-dict";
}

def QueueShuffleOp : QueueBuiltin<"shuffle"> {
  let summary = "Shuffle the elements of a queue";
  let description = [{
    Randomly shuffles the elements of a queue. This corresponds to the
    SystemVerilog `.shuffle()` built-in method on queues and dynamic arrays.

    Example:
    ```mlir
    moore.queue.shuffle %queue : !moore.ref<!moore.queue<i32, 0>>
    ```
  }];
  let arguments = (ins RefType:$queue);
  let assemblyFormat = "$queue `:` type($queue) attr-dict";
}

def QueueReverseOp : QueueBuiltin<"reverse"> {
  let summary = "Reverse the elements of a queue";
  let description = [{
    Reverses the order of elements in a queue in-place. This corresponds to the
    SystemVerilog `.reverse()` built-in method on queues and dynamic arrays
    (IEEE 1800-2017 Section 7.12.2).

    Example:
    ```mlir
    moore.queue.reverse %queue : !moore.ref<!moore.queue<i32, 0>>
    ```
  }];
  let arguments = (ins RefType:$queue);
  let assemblyFormat = "$queue `:` type($queue) attr-dict";
}

def QueueMaxOp : QueueBuiltin<"max"> {
  let summary = "Find the maximum element(s) in a queue or array";
  let description = [{
    Returns a queue containing the element(s) with the maximum value.
    This corresponds to the SystemVerilog `.max()` built-in method on arrays
    and queues (IEEE 1800-2017 Section 7.12.3).

    Example:
    ```mlir
    %max_vals = moore.queue.max %array : !moore.open_uarray<i32> -> !moore.queue<i32, 0>
    ```
  }];
  let arguments = (ins UnpackedType:$array);
  let results = (outs QueueType:$result);
  let assemblyFormat = "$array `:` type($array) `->` type($result) attr-dict";
}

def QueueMinOp : QueueBuiltin<"min"> {
  let summary = "Find the minimum element(s) in a queue or array";
  let description = [{
    Returns a queue containing the element(s) with the minimum value.
    This corresponds to the SystemVerilog `.min()` built-in method on arrays
    and queues (IEEE 1800-2017 Section 7.12.3).

    Example:
    ```mlir
    %min_vals = moore.queue.min %array : !moore.open_uarray<i32> -> !moore.queue<i32, 0>
    ```
  }];
  let arguments = (ins UnpackedType:$array);
  let results = (outs QueueType:$result);
  let assemblyFormat = "$array `:` type($array) `->` type($result) attr-dict";
}

def QueueUniqueOp : QueueBuiltin<"unique"> {
  let summary = "Return unique elements from a queue or array";
  let description = [{
    Returns a queue containing only the unique elements from the input array
    or queue. Duplicate elements are removed, keeping only the first occurrence.
    This corresponds to the SystemVerilog `.unique()` built-in method on arrays
    and queues (IEEE 1800-2017 Section 7.12.1).

    Example:
    ```mlir
    %unique_vals = moore.queue.unique %array : !moore.open_uarray<i32> -> !moore.queue<i32, 0>
    ```
  }];
  let arguments = (ins UnpackedType:$array);
  let results = (outs QueueType:$result);
  let assemblyFormat = "$array `:` type($array) `->` type($result) attr-dict";
}

def QueueUniqueIndexOp : QueueBuiltin<"unique_index"> {
  let summary = "Return indices of unique elements from a queue or array";
  let description = [{
    Returns a queue containing the indices of the first occurrence of each
    unique element in the input array or queue. This corresponds to the
    SystemVerilog `.unique_index()` built-in method on arrays and queues
    (IEEE 1800-2017 Section 7.12.1).

    Example:
    ```mlir
    %idxs = moore.queue.unique_index %array : !moore.open_uarray<i32> -> !moore.queue<i64, 0>
    ```
  }];
  let arguments = (ins UnpackedType:$array);
  let results = (outs QueueType:$result);
  let assemblyFormat = "$array `:` type($array) `->` type($result) attr-dict";
}

def QueueReduceKindSum : I32EnumAttrCase<"Sum", 0, "sum">;
def QueueReduceKindProduct : I32EnumAttrCase<"Product", 1, "product">;
def QueueReduceKindAnd : I32EnumAttrCase<"And", 2, "and">;
def QueueReduceKindOr : I32EnumAttrCase<"Or", 3, "or">;
def QueueReduceKindXor : I32EnumAttrCase<"Xor", 4, "xor">;

def QueueReduceKindAttr
    : I32EnumAttr<"QueueReduceKind", "Queue reduction method",
                  [QueueReduceKindSum, QueueReduceKindProduct,
                   QueueReduceKindAnd, QueueReduceKindOr,
                   QueueReduceKindXor]> {
  let cppNamespace = "circt::moore";
}

def QueueReduceOp : QueueBuiltin<"reduce", [Pure]> {
  let summary = "Reduce a queue or array to a single value";
  let description = [{
    Reduces all elements of a queue or array into a single value using the
    specified reduction kind (sum, product, and, or, xor). This corresponds to
    SystemVerilog `.sum()`, `.product()`, `.and()`, `.or()`, and `.xor()`
    built-in methods on arrays and queues.

    Example:
    ```mlir
    %sum = moore.queue.reduce sum %array : !moore.open_uarray<i32> -> i32
    ```
  }];
  let arguments = (ins QueueReduceKindAttr:$kind, UnpackedType:$array);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind $array `:` type($array) `->` type($result) attr-dict";
}

def QueueConcatOp : QueueBuiltin<"concat", [Pure]> {
  let summary = "Concatenate queues";
  let description = [{
    Concatenates an arbitrary number of queues into one larger queue.
    The elements are concatenated from left to right, with the first
    operand's elements appearing at the beginning of the result queue, and the
    last operand's elements appearing at the end. This corresponds to the
    SystemVerilog queue concatenation syntax `{queue1, queue2}`.

    Example:
    ```mlir
    %result = moore.queue.concat(%a, %b) : !moore.queue<i32, 0>, !moore.queue<i32, 0> -> <i32, 0>
    ```

    See IEEE 1800-2017 Section 7.10.4 "Queue operators".
  }];
  let arguments = (ins Variadic<QueueType>:$inputs);
  let results = (outs QueueType:$result);
  let assemblyFormat = [{
    `(` $inputs `)` attr-dict (`:` type($inputs)^)? `->` type($result)
  }];
}

def QueueSliceOp : QueueBuiltin<"slice", [Pure]> {
  let summary = "Slice a queue by index range";
  let description = [{
    Returns a new queue containing elements in the inclusive range
    [start, end] from the input queue. If the range is invalid or empty,
    the result is an empty queue.

    Example:
    ```mlir
    %slice = moore.queue.slice %q, %start, %end : !moore.queue<i32, 0>
    ```
  }];
  let arguments = (ins QueueType:$queue, TwoValuedI32:$start, TwoValuedI32:$end);
  let results = (outs QueueType:$result);
  let assemblyFormat = "$queue `,` $start `,` $end attr-dict `:` type($queue) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Array Locator Methods
//===----------------------------------------------------------------------===//

// Locator method mode: All (find/find_index), First (find_first/find_first_index),
// or Last (find_last/find_last_index)
def LocatorAll : I32EnumAttrCase<"All", 0, "all">;
def LocatorFirst : I32EnumAttrCase<"First", 1, "first">;
def LocatorLast : I32EnumAttrCase<"Last", 2, "last">;

def LocatorModeAttr : I32EnumAttr<"LocatorMode", "Array locator method mode",
                               [LocatorAll, LocatorFirst, LocatorLast]> {
  let cppNamespace = "circt::moore";
}

def ArrayLocatorYieldOp : MooreOp<"array.locator.yield", [
  Pure,
  Terminator,
  ParentOneOf<["ArrayLocatorOp"]>,
]> {
  let summary = "Yield a predicate result from an array locator body";
  let description = [{
    `moore.array.locator.yield` terminates the predicate region of an
    `moore.array.locator` operation. It yields a boolean value that determines
    whether the current element matches the search criteria.
  }];
  let arguments = (ins AnySingleBitType:$condition);
  let assemblyFormat = "attr-dict $condition `:` type($condition)";
}

def ArrayLocatorOp : MooreOp<"array.locator", [
  RecursiveMemoryEffects,
]> {
  let summary = "Find elements in an array matching a predicate";
  let description = [{
    This operation implements the SystemVerilog array locator methods: `find`,
    `find_index`, `find_first`, `find_first_index`, `find_last`, and
    `find_last_index`. These methods search an array for elements that match
    a user-defined predicate.

    The operation takes an array input and contains a single-block region that
    defines the predicate. The region has two block arguments: the current element
    being evaluated, and the current index (as a 32-bit integer). The index
    argument enables use of `item.index` in the predicate expression. The
    predicate region must terminate with an `moore.array.locator.yield` operation
    that yields a boolean value.

    The `mode` attribute specifies the search behavior:
    - `all`: Return all matching elements/indices (find/find_index)
    - `first`: Return only the first matching element/index (find_first/find_first_index)
    - `last`: Return only the last matching element/index (find_last/find_last_index)

    The `indexed` attribute specifies what to return:
    - `false`: Return the matching elements themselves (find/find_first/find_last)
    - `true`: Return the indices of matching elements (find_index/find_first_index/find_last_index)

    Example:
    ```mlir
    // arr.find(x) with (x > 5) - returns queue of elements > 5
    %result = moore.array.locator all, elements %arr : !moore.queue<i32, 0> -> !moore.queue<i32, 0> {
    ^bb0(%item: !moore.i32, %index: !moore.i32):
      %five = moore.constant 5 : i32
      %cond = moore.sgt %item, %five : i32
      moore.array.locator.yield %cond : i1
    }

    // arr.find(x) with (x == x.index) - returns elements matching their index
    %result = moore.array.locator all, elements %arr : !moore.queue<i32, 0> -> !moore.queue<i32, 0> {
    ^bb0(%item: !moore.i32, %index: !moore.i32):
      %cond = moore.eq %item, %index : i32
      moore.array.locator.yield %cond : i1
    }
    ```

    See IEEE 1800-2017 Section 7.12.2 "Array locator methods".
  }];
  let arguments = (ins
    LocatorModeAttr:$mode,
    UnitAttr:$indexed,
    UnpackedType:$array
  );
  let results = (outs QueueType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $mode `,` (`indices` $indexed^) : (`elements`)? $array `:` type($array)
    `->` type($result) $body attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Associative Array Builtins
//===----------------------------------------------------------------------===//

class AssocArrayBuiltin<string mnemonic, list<Trait> traits = []> :
    MooreOp<"assoc." # mnemonic, traits>;

def AssocArrayFirstOp : AssocArrayBuiltin<"first"> {
  let summary = "Get the first key of an associative array";
  let description = [{
    Returns true and sets the key reference to the first key of the associative
    array if it is non-empty. Returns false if the array is empty.
    This corresponds to the SystemVerilog `.first(ref index)` built-in method.

    Example:
    ```mlir
    %found = moore.assoc.first %array, %key : !moore.ref<!moore.assoc<i32, string>>, !moore.ref<string>
    ```
  }];
  let arguments = (ins RefType:$array, RefType:$key);
  let results = (outs TwoValuedI1:$found);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayNextOp : AssocArrayBuiltin<"next"> {
  let summary = "Get the next key of an associative array";
  let description = [{
    Returns true and sets the key reference to the next key of the associative
    array after the current key. Returns false if there are no more keys.
    This corresponds to the SystemVerilog `.next(ref index)` built-in method.

    Example:
    ```mlir
    %found = moore.assoc.next %array, %key : !moore.ref<!moore.assoc<i32, string>>, !moore.ref<string>
    ```
  }];
  let arguments = (ins RefType:$array, RefType:$key);
  let results = (outs TwoValuedI1:$found);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayLastOp : AssocArrayBuiltin<"last"> {
  let summary = "Get the last key of an associative array";
  let description = [{
    Returns true and sets the key reference to the last key of the associative
    array if it is non-empty. Returns false if the array is empty.
    This corresponds to the SystemVerilog `.last(ref index)` built-in method.

    Example:
    ```mlir
    %found = moore.assoc.last %array, %key : !moore.ref<!moore.assoc<i32, string>>, !moore.ref<string>
    ```
  }];
  let arguments = (ins RefType:$array, RefType:$key);
  let results = (outs TwoValuedI1:$found);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayPrevOp : AssocArrayBuiltin<"prev"> {
  let summary = "Get the previous key of an associative array";
  let description = [{
    Returns true and sets the key reference to the previous key of the associative
    array before the current key. Returns false if there are no previous keys.
    This corresponds to the SystemVerilog `.prev(ref index)` built-in method.

    Example:
    ```mlir
    %found = moore.assoc.prev %array, %key : !moore.ref<!moore.assoc<i32, string>>, !moore.ref<string>
    ```
  }];
  let arguments = (ins RefType:$array, RefType:$key);
  let results = (outs TwoValuedI1:$found);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayExistsOp : AssocArrayBuiltin<"exists"> {
  let summary = "Check if a key exists in an associative array";
  let description = [{
    Returns 1 if the specified key exists in the associative array, 0 otherwise.
    This corresponds to the SystemVerilog `.exists(index)` built-in method.

    Example:
    ```mlir
    %found = moore.assoc.exists %array, %key : !moore.assoc_array<i32, string>, i32
    ```
  }];
  let arguments = (ins AssocArrayType:$array, UnpackedType:$key);
  let results = (outs TwoValuedI1:$result);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayDeleteOp : AssocArrayBuiltin<"delete"> {
  let summary = "Delete all elements from an associative array";
  let description = [{
    Removes all elements from an associative array, leaving it empty. This
    corresponds to the SystemVerilog `.delete()` built-in method on associative
    arrays when called with no arguments.

    Example:
    ```mlir
    moore.assoc.delete %array : !moore.ref<!moore.assoc<i32, string>>
    ```
  }];
  let arguments = (ins RefType:$array);
  let assemblyFormat = "$array `:` type($array) attr-dict";
}

def AssocArrayDeleteKeyOp : AssocArrayBuiltin<"delete_key"> {
  let summary = "Delete a specific key from an associative array";
  let description = [{
    Removes the element with the specified key from an associative array. This
    corresponds to the SystemVerilog `.delete(index)` built-in method on
    associative arrays when called with a key argument.

    Example:
    ```mlir
    moore.assoc.delete_key %array, %key : !moore.ref<!moore.assoc<i32, string>>, string
    ```
  }];
  let arguments = (ins RefType:$array, UnpackedType:$key);
  let assemblyFormat = "$array `,` $key `:` type($array) `,` type($key) attr-dict";
}

def AssocArrayCreateOp : AssocArrayBuiltin<"create", [Pure]> {
  let summary = "Create an empty associative array";
  let description = [{
    Creates an empty associative array. This operation is used to implement
    assignment patterns with default values for associative arrays, such as
    `'{default: value}` in SystemVerilog.

    Note: The default value is currently ignored during lowering. Accessing
    non-existent keys will return the type's default value (zero or empty
    string) rather than the specified default.

    Example:
    ```mlir
    %aa = moore.assoc.create : !moore.assoc_array<string, i32>
    ```

    See IEEE 1800-2017 Section 7.9.11 "Associative array assignment patterns".
  }];
  let arguments = (ins);
  let results = (outs AnyAssocArrayType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
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

def StringSubstrOp : StringBuiltin<"substr"> {
  let summary = "Extract a substring from a string";
  let description = [{
    Extracts a substring from the given string, starting at `start` with
    the specified `len` number of characters.
  }];
  let arguments = (ins StringType:$str, TwoValuedI32:$start, TwoValuedI32:$len);
  let results = (outs StringType:$result);
  let assemblyFormat = "$str `[` $start `,` $len `]` attr-dict";
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

def StringPutCOp : StringBuiltin<"putc"> {
  let summary = "Set a character in a string";
  let description = [{
    Sets the character at the specified index in the given string reference.
    This corresponds to the SystemVerilog `str.putc(i, c)` method and also
    supports the indexed assignment syntax `str[i] = c`.
  }];
  let arguments = (ins RefType:$str, TwoValuedI32:$index, TwoValuedI8:$character);
  let assemblyFormat = "$str `[` $index `]` `,` $character `:` type($str) attr-dict";
}

def StringItoaOp : StringBuiltin<"itoa"> {
  let summary = "Convert an integer to a decimal string";
  let description = [{
    Converts the given integer value to its decimal string representation
    and stores the result in the destination string reference.
    This corresponds to the SystemVerilog `str.itoa(i)` method.
  }];
  let arguments = (ins RefType:$dest, IntType:$value);
  let assemblyFormat = "$dest `,` $value `:` type($dest) `,` type($value) attr-dict";
}

def StringAtoIOp : StringBuiltin<"atoi"> {
  let summary = "Convert a string to a decimal integer";
  let description = [{
    Interprets the string as a decimal integer and returns the result.
    This corresponds to the SystemVerilog `str.atoi()` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$str);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringAtoHexOp : StringBuiltin<"atohex"> {
  let summary = "Convert a string to a hexadecimal integer";
  let description = [{
    Interprets the string as a hexadecimal integer and returns the result.
    This corresponds to the SystemVerilog `str.atohex()` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$str);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringAtoOctOp : StringBuiltin<"atooct"> {
  let summary = "Convert a string to an octal integer";
  let description = [{
    Interprets the string as an octal integer and returns the result.
    This corresponds to the SystemVerilog `str.atooct()` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$str);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringAtoBinOp : StringBuiltin<"atobin"> {
  let summary = "Convert a string to a binary integer";
  let description = [{
    Interprets the string as a binary integer and returns the result.
    This corresponds to the SystemVerilog `str.atobin()` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$str);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringAtoRealOp : StringBuiltin<"atoreal"> {
  let summary = "Convert a string to a real number";
  let description = [{
    Interprets the string as a real number and returns the result.
    This corresponds to the SystemVerilog `str.atoreal()` method
    (IEEE 1800-2017 Section 6.16.9).
  }];
  let arguments = (ins StringType:$str);
  let results = (outs RealF64:$result);
  let assemblyFormat = "$str attr-dict";
}

def StringHexToAOp : StringBuiltin<"hextoa"> {
  let summary = "Convert an integer to a hexadecimal string";
  let description = [{
    Converts the given integer value to its hexadecimal string representation
    and stores the result in the destination string reference.
    This corresponds to the SystemVerilog `str.hextoa(i)` method
    (IEEE 1800-2017 Section 6.16.9).
  }];
  let arguments = (ins RefType:$dest, IntType:$value);
  let assemblyFormat = "$dest `,` $value `:` type($dest) `,` type($value) attr-dict";
}

def StringOctToAOp : StringBuiltin<"octtoa"> {
  let summary = "Convert an integer to an octal string";
  let description = [{
    Converts the given integer value to its octal string representation
    and stores the result in the destination string reference.
    This corresponds to the SystemVerilog `str.octtoa(i)` method
    (IEEE 1800-2017 Section 6.16.9).
  }];
  let arguments = (ins RefType:$dest, IntType:$value);
  let assemblyFormat = "$dest `,` $value `:` type($dest) `,` type($value) attr-dict";
}

def StringBinToAOp : StringBuiltin<"bintoa"> {
  let summary = "Convert an integer to a binary string";
  let description = [{
    Converts the given integer value to its binary string representation
    and stores the result in the destination string reference.
    This corresponds to the SystemVerilog `str.bintoa(i)` method
    (IEEE 1800-2017 Section 6.16.9).
  }];
  let arguments = (ins RefType:$dest, IntType:$value);
  let assemblyFormat = "$dest `,` $value `:` type($dest) `,` type($value) attr-dict";
}

def StringRealToAOp : StringBuiltin<"realtoa"> {
  let summary = "Convert a real number to a string";
  let description = [{
    Converts the given real value to its string representation
    and stores the result in the destination string reference.
    This corresponds to the SystemVerilog `str.realtoa(r)` method
    (IEEE 1800-2017 Section 6.16.9).
  }];
  let arguments = (ins RefType:$dest, RealType:$value);
  let assemblyFormat = "$dest `,` $value `:` type($dest) `,` type($value) attr-dict";
}

def StringCompareOp : StringBuiltin<"compare", [Pure]> {
  let summary = "Lexicographically compare two strings";
  let description = [{
    Compares two strings lexicographically (case-sensitive).
    Returns a negative integer if `lhs` < `rhs`,
    zero if `lhs` == `rhs`,
    and a positive integer if `lhs` > `rhs`.
    This corresponds to the SystemVerilog `str.compare(s)` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$lhs, StringType:$rhs);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def StringICompareOp : StringBuiltin<"icompare", [Pure]> {
  let summary = "Lexicographically compare two strings (case-insensitive)";
  let description = [{
    Compares two strings lexicographically (case-insensitive).
    Returns a negative integer if `lhs` < `rhs`,
    zero if `lhs` == `rhs`,
    and a positive integer if `lhs` > `rhs`.
    This corresponds to the SystemVerilog `str.icompare(s)` method
    (IEEE 1800-2017 Section 6.16.8).
  }];
  let arguments = (ins StringType:$lhs, StringType:$rhs);
  let results = (outs TwoValuedI32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// Clocking Blocks
//===----------------------------------------------------------------------===//

def ClockingBlockDeclOp : MooreOp<"clocking_block", [
  NoTerminator,
  SingleBlock
]> {
  let summary = "Clocking block declaration";
  let description = [{
    Declares a SystemVerilog clocking block. Clocking blocks define a
    synchronization mechanism for signals relative to a clock event.
    They specify input and output skews for sampling and driving signals.

    Example:
    ```mlir
    moore.clocking_block "cb" {
      // Signals sampled/driven relative to the clocking event
    }
    ```

    See IEEE 1800-2017 Section 14 "Clocking Blocks".
  }];

  let arguments = (ins
    StrAttr:$name
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $name attr-dict-with-keyword $body
  }];
}

def ClockingSignalOp : MooreOp<"clocking_block.signal", [
  HasParent<"ClockingBlockDeclOp">
]> {
  let summary = "Declare a signal within a clocking block";
  let description = [{
    Declares a signal within a clocking block. The signal specifies the
    direction (input, output, or inout) and optional skew values for
    sampling and driving.

    Example:
    ```mlir
    moore.clocking_block.signal "data" : !moore.l32
    ```

    See IEEE 1800-2017 Section 14.3 "Clocking block declaration".
  }];

  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type
  );
  let assemblyFormat = [{
    $name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Type getSignalType() { return getTypeAttr().getValue(); }
  }];
}

#endif // CIRCT_DIALECT_MOORE_MOOREOPS

def FormatTimeOp : MooreOp<"fmt.time", [Pure]> {
  let summary = "Format a time value using $timeformat settings";
  let description = [{
    Format a simulation time value (in femtoseconds) using the global
    timeformat settings (units, precision, suffix, minimum width).
  }];
  let arguments = (ins IntType:$time);
  let results = (outs FormatStringType:$result);
  let assemblyFormat = "$time attr-dict `:` type($time)";
}

def TimeFormatBIOp : Builtin<"timeformat"> {
  let summary = "Set time display format";
  let description = [{
    Corresponds to the timeformat system task. Sets the formatting parameters
    used by the %t format specifier in display, write, etc.
  }];
  let arguments = (ins TwoValuedI32:$units, TwoValuedI32:$precision,
                       StrAttr:$suffix, TwoValuedI32:$min_width);
  let assemblyFormat = "$units `,` $precision `,` $suffix `,` $min_width attr-dict";
}

def DistBIOp : Builtin<"dist"> {
  let summary = "Distribution random number generation";
  let description = [{
    Implements IEEE 1800-2017 Section 20.15 probabilistic distribution functions:
    $dist_uniform, $dist_normal, $dist_exponential, $dist_poisson,
    $dist_chi_square, $dist_t, $dist_erlang.
    The seed argument is inout (ref) and updated by the runtime.
  }];
  let arguments = (ins StrAttr:$dist_name, RefType:$seed,
                       Variadic<IntType>:$params);
  let results = (outs IntType:$result);
  let assemblyFormat = [{
    $dist_name `,` $seed `,` $params attr-dict `:` type($seed) `,`
    type($params) `->` type($result)
  }];
}
