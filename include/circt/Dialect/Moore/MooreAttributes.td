//===- MooreAttributes.td - Moore attribute definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
#define CIRCT_DIALECT_MOORE_MOOREATTRIBUTES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"

def FVIntegerAttr : AttrDef<MooreDialect, "FVInteger"> {
  let mnemonic = "fvint";
  let summary = "An attribute containing a four-valued integer";
  let parameters = (ins "FVInt":$value);
  let hasCustomAssemblyFormat = 1;
  let convertFromStorage = "$_self.getValue()";
  let returnType = "circt::FVInt";
}

//===----------------------------------------------------------------------===//
// Modport Direction Enum
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

def ModportDirInput: I32EnumAttrCase<"Input", 0, "input">;
def ModportDirOutput: I32EnumAttrCase<"Output", 1, "output">;
def ModportDirInOut: I32EnumAttrCase<"InOut", 2, "inout">;
def ModportDirRef: I32EnumAttrCase<"Ref", 3, "ref">;

def ModportDirAttr: I32EnumAttr<"ModportDir", "Modport port direction",
            [ModportDirInput, ModportDirOutput, ModportDirInOut, ModportDirRef]>{
  let cppNamespace = "circt::moore";
}

//===----------------------------------------------------------------------===//
// Modport Port Attribute
//===----------------------------------------------------------------------===//

def ModportPortAttr : AttrDef<MooreDialect, "ModportPort"> {
  let mnemonic = "modport_port";
  let summary = "A port in a modport declaration";
  let description = [{
    Represents a single port within a modport declaration, combining a
    direction (input, output, inout, ref) with a signal reference.
  }];
  let parameters = (ins
    "ModportDirAttr":$direction,
    "mlir::FlatSymbolRefAttr":$signal
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def ModportPortArrayAttr : TypedArrayAttrBase<ModportPortAttr,
  "array of modport ports">;

//===----------------------------------------------------------------------===//
// UVM Base Class Kind
//===----------------------------------------------------------------------===//

// Enumeration of UVM base class types
def UVMBaseNone: I32EnumAttrCase<"None", 0, "none">;
def UVMBaseObject: I32EnumAttrCase<"UVMObject", 1, "uvm_object">;
def UVMBaseComponent: I32EnumAttrCase<"UVMComponent", 2, "uvm_component">;
def UVMBaseSequenceItem: I32EnumAttrCase<"UVMSequenceItem", 3, "uvm_sequence_item">;
def UVMBaseSequence: I32EnumAttrCase<"UVMSequence", 4, "uvm_sequence">;
def UVMBaseDriver: I32EnumAttrCase<"UVMDriver", 5, "uvm_driver">;
def UVMBaseMonitor: I32EnumAttrCase<"UVMMonitor", 6, "uvm_monitor">;
def UVMBaseAgent: I32EnumAttrCase<"UVMAgent", 7, "uvm_agent">;
def UVMBaseEnv: I32EnumAttrCase<"UVMEnv", 8, "uvm_env">;
def UVMBaseTest: I32EnumAttrCase<"UVMTest", 9, "uvm_test">;
def UVMBaseSequencer: I32EnumAttrCase<"UVMSequencer", 10, "uvm_sequencer">;
def UVMBaseScoreboard: I32EnumAttrCase<"UVMScoreboard", 11, "uvm_scoreboard">;
def UVMBaseSubscriber: I32EnumAttrCase<"UVMSubscriber", 12, "uvm_subscriber">;

def UVMBaseClassAttr: I32EnumAttr<"UVMBaseClass", "UVM base class kind",
            [UVMBaseNone, UVMBaseObject, UVMBaseComponent, UVMBaseSequenceItem,
             UVMBaseSequence, UVMBaseDriver, UVMBaseMonitor, UVMBaseAgent,
             UVMBaseEnv, UVMBaseTest, UVMBaseSequencer, UVMBaseScoreboard,
             UVMBaseSubscriber]>{
  let cppNamespace = "circt::moore";
}

//===----------------------------------------------------------------------===//
// UVM Port Kind
//===----------------------------------------------------------------------===//

// Enumeration of TLM port types
def TLMPortKindPut: I32EnumAttrCase<"Put", 0, "put">;
def TLMPortKindGet: I32EnumAttrCase<"Get", 1, "get">;
def TLMPortKindPeek: I32EnumAttrCase<"Peek", 2, "peek">;
def TLMPortKindGetPeek: I32EnumAttrCase<"GetPeek", 3, "get_peek">;
def TLMPortKindBlocking: I32EnumAttrCase<"Blocking", 4, "blocking">;
def TLMPortKindNonBlocking: I32EnumAttrCase<"NonBlocking", 5, "nonblocking">;
def TLMPortKindTransport: I32EnumAttrCase<"Transport", 6, "transport">;
def TLMPortKindAnalysis: I32EnumAttrCase<"Analysis", 7, "analysis">;

def TLMPortKindAttr: I32EnumAttr<"TLMPortKind", "TLM port kind",
            [TLMPortKindPut, TLMPortKindGet, TLMPortKindPeek, TLMPortKindGetPeek,
             TLMPortKindBlocking, TLMPortKindNonBlocking, TLMPortKindTransport,
             TLMPortKindAnalysis]>{
  let cppNamespace = "circt::moore";
}

#endif // CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
