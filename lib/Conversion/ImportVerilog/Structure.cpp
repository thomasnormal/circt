//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Constraints.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/TimingControl.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/symbols/ClassSymbols.h"
#include "slang/ast/symbols/CoverSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/types/NetType.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/ADT/ScopeExit.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void guessNamespacePrefix(const slang::ast::Symbol &symbol,
                                 SmallString<64> &prefix) {
  if (symbol.kind != slang::ast::SymbolKind::Package)
    return;
  guessNamespacePrefix(symbol.getParentScope()->asSymbol(), prefix);
  if (!symbol.name.empty()) {
    prefix += symbol.name;
    prefix += "::";
  }
}

static const slang::ast::InstanceBodySymbol *
getInstanceBodyParent(const slang::ast::InstanceSymbol &instSym) {
  auto *parentScope = instSym.getParentScope();
  if (!parentScope)
    return nullptr;
  if (auto *parentBody =
          parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>())
    return parentBody;
  if (auto *parentArray =
          parentScope->asSymbol().as_if<slang::ast::InstanceArraySymbol>()) {
    auto *arrayParent = parentArray->getParentScope();
    if (!arrayParent)
      return nullptr;
    return arrayParent->asSymbol().as_if<slang::ast::InstanceBodySymbol>();
  }
  return nullptr;
}

static std::optional<int64_t>
getInterfaceArrayIndex(const slang::ast::InstanceSymbol &instSym) {
  if (instSym.arrayPath.empty())
    return std::nullopt;
  slang::SmallVector<slang::ConstantRange, 4> dimensions;
  instSym.getArrayDimensions(dimensions);
  if (dimensions.empty())
    return std::nullopt;
  const auto &range = dimensions[0];
  int32_t relIndex = static_cast<int32_t>(instSym.arrayPath[0]);
  int32_t actualIndex =
      range.isLittleEndian() ? range.lower() + relIndex
                             : range.upper() - relIndex;
  return actualIndex;
}

static bool
getInterfaceArrayIndices(const slang::ast::InstanceSymbol &instSym,
                          SmallVectorImpl<int64_t> &indices) {
  if (instSym.arrayPath.empty())
    return false;
  slang::SmallVector<slang::ConstantRange, 4> dimensions;
  instSym.getArrayDimensions(dimensions);
  auto dimCount = std::min(dimensions.size(), instSym.arrayPath.size());
  if (dimCount == 0)
    return false;
  indices.clear();
  indices.reserve(dimCount);
  for (size_t i = 0; i < dimCount; ++i) {
    const auto &range = dimensions[i];
    int64_t relIndex = static_cast<int64_t>(instSym.arrayPath[i]);
    int64_t actualIndex =
        range.isLittleEndian() ? range.lower() + relIndex
                               : range.upper() - relIndex;
    indices.push_back(actualIndex);
  }
  return true;
}

static bool
matchInterfaceArrayIndices(const slang::ast::InstanceSymbol &lhs,
                            const slang::ast::InstanceSymbol &rhs) {
  SmallVector<int64_t, 4> lhsIndices;
  SmallVector<int64_t, 4> rhsIndices;
  bool lhsIsArray = getInterfaceArrayIndices(lhs, lhsIndices);
  bool rhsIsArray = getInterfaceArrayIndices(rhs, rhsIndices);
  if (lhsIsArray != rhsIsArray)
    return false;
  if (lhsIsArray && lhsIndices != rhsIndices)
    return false;
  return true;
}

static const slang::ast::InstanceSymbol *findInterfaceArrayElement(
    const slang::ast::InstanceArraySymbol &arraySym, int64_t index) {
  for (const auto *elementSym : arraySym.elements) {
    if (!elementSym)
      continue;
    auto *element = elementSym->as_if<slang::ast::InstanceSymbol>();
    if (!element)
      continue;
    auto actualIndex = getInterfaceArrayIndex(*element);
    if (actualIndex && *actualIndex == index)
      return element;
  }
  return nullptr;
}

static bool shouldCacheInterfaceInstance(
    const slang::ast::InstanceSymbol &instSym) {
  if (auto *parentBody = getInstanceBodyParent(instSym)) {
    if (parentBody->getDefinition().definitionKind ==
        slang::ast::DefinitionKind::Interface)
      return false;
  }
  return true;
}

static LogicalResult emitInterfacePortConnections(
    Context &context, Location loc, const slang::ast::InstanceSymbol &instNode,
    Value instRef) {
  using slang::ast::ArgumentDirection;
  using slang::ast::AssignmentExpression;
  using slang::ast::MultiPortSymbol;
  using slang::ast::PortSymbol;

  if (instNode.getPortConnections().empty())
    return success();

  OpBuilder &builder = context.builder;
  Value ifaceValue;
  auto getIfaceValue = [&]() -> Value {
    if (!ifaceValue)
      ifaceValue = moore::ReadOp::create(builder, loc, instRef);
    return ifaceValue;
  };

  auto getIfaceSignalRef = [&](StringRef name, Type portType) -> Value {
    auto unpackedType = dyn_cast<moore::UnpackedType>(portType);
    if (!unpackedType)
      return {};
    auto signalSym = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
    auto refTy = moore::RefType::get(unpackedType);
    return moore::VirtualInterfaceSignalRefOp::create(builder, loc, refTy,
                                                      getIfaceValue(),
                                                      signalSym);
  };

  auto connectInputPort = [&](const PortSymbol &port,
                              Value rhs) -> LogicalResult {
    auto portType = context.convertType(port.getType());
    if (!portType)
      return failure();
    auto signalRef = getIfaceSignalRef(port.name, portType);
    if (!signalRef)
      return mlir::emitError(loc)
             << "unsupported interface port type for `" << port.name << "`";
    rhs = context.materializeConversion(portType, rhs, false, loc);
    if (!rhs)
      return failure();
    moore::ContinuousAssignOp::create(builder, loc, signalRef, rhs);
    return success();
  };

  auto connectOutputPort = [&](const PortSymbol &port,
                               Value lhs) -> LogicalResult {
    auto portType = context.convertType(port.getType());
    if (!portType)
      return failure();
    auto signalRef = getIfaceSignalRef(port.name, portType);
    if (!signalRef)
      return mlir::emitError(loc)
             << "unsupported interface port type for `" << port.name << "`";
    Value rhs = moore::ReadOp::create(builder, loc, signalRef);
    auto dstType = cast<moore::RefType>(lhs.getType()).getNestedType();
    rhs = context.materializeConversion(dstType, rhs, false, loc);
    if (!rhs)
      return failure();
    moore::ContinuousAssignOp::create(builder, loc, lhs, rhs);
    return success();
  };

  // Connect an inout interface port by creating bidirectional assignments
  // between the external signal and the interface signal.
  auto connectInOutPort = [&](const PortSymbol &port,
                              Value externalRef) -> LogicalResult {
    auto portType = context.convertType(port.getType());
    if (!portType)
      return failure();
    auto signalRef = getIfaceSignalRef(port.name, portType);
    if (!signalRef)
      return mlir::emitError(loc)
             << "unsupported interface port type for `" << port.name << "`";
    // For inout ports, we connect the external signal to the interface signal.
    // Read the external value and assign to interface signal.
    Value externalValue = moore::ReadOp::create(builder, loc, externalRef);
    externalValue =
        context.materializeConversion(portType, externalValue, false, loc);
    if (!externalValue)
      return failure();
    moore::ContinuousAssignOp::create(builder, loc, signalRef, externalValue);
    // Also connect interface signal back to external (bidirectional).
    Value ifaceValue = moore::ReadOp::create(builder, loc, signalRef);
    auto dstType = cast<moore::RefType>(externalRef.getType()).getNestedType();
    ifaceValue = context.materializeConversion(dstType, ifaceValue, false, loc);
    if (!ifaceValue)
      return failure();
    moore::ContinuousAssignOp::create(builder, loc, externalRef, ifaceValue);
    return success();
  };

  for (const auto *con : instNode.getPortConnections()) {
    const auto *portSymbol = &con->port;
    const auto *expr = con->getExpression();

    if (!expr) {
      // Unconnected interface ports are allowed; leave them undriven.
      continue;
    }

    if (const auto *assign = expr->as_if<AssignmentExpression>())
      expr = &assign->left();

    if (auto *port = portSymbol->as_if<PortSymbol>()) {
      if (port->direction == ArgumentDirection::In) {
        auto rhs = context.convertRvalueExpression(*expr);
        if (!rhs || failed(connectInputPort(*port, rhs)))
          return failure();
        continue;
      }
      if (port->direction == ArgumentDirection::Out) {
        auto lhs = context.convertLvalueExpression(*expr);
        if (!lhs || failed(connectOutputPort(*port, lhs)))
          return failure();
        continue;
      }
      if (port->direction == ArgumentDirection::InOut) {
        auto lhs = context.convertLvalueExpression(*expr);
        if (!lhs || failed(connectInOutPort(*port, lhs)))
          return failure();
        continue;
      }
      return mlir::emitError(loc)
             << "unsupported interface port `" << port->name << "` ("
             << slang::ast::toString(port->direction) << ")";
    }

    if (const auto *multiPort = portSymbol->as_if<MultiPortSymbol>()) {
      auto value = context.convertLvalueExpression(*expr);
      if (!value)
        return failure();
      unsigned offset = 0;
      for (const auto *port : llvm::reverse(multiPort->ports)) {
        unsigned width = port->getType().getBitWidth();
        auto portType = context.convertType(port->getType());
        if (!portType)
          return failure();
        auto sliceType =
            moore::RefType::get(cast<moore::UnpackedType>(portType));
        Value slice =
            moore::ExtractRefOp::create(builder, loc, sliceType, value, offset);
        if (port->direction == ArgumentDirection::In) {
          Value rhs = moore::ReadOp::create(builder, loc, slice);
          if (failed(connectInputPort(*port, rhs)))
            return failure();
        } else if (port->direction == ArgumentDirection::Out) {
          if (failed(connectOutputPort(*port, slice)))
            return failure();
        } else if (port->direction == ArgumentDirection::InOut) {
          if (failed(connectInOutPort(*port, slice)))
            return failure();
        } else {
          return mlir::emitError(loc)
                 << "unsupported interface port `" << port->name << "` ("
                 << slang::ast::toString(port->direction) << ")";
        }
        offset += width;
      }
      continue;
    }

    return mlir::emitError(loc)
           << "unsupported interface port `" << portSymbol->name << "` ("
           << slang::ast::toString(portSymbol->kind) << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Base Visitor
//===----------------------------------------------------------------------===//

namespace {
/// Base visitor which ignores AST nodes that are handled by Slang's name
/// resolution and type checking.
struct BaseVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  BaseVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip semicolons.
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip members that are implicitly imported from some other scope for the
  // sake of name resolution, such as enum variant names.
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Handle classes without parameters or specialized generic classes
  LogicalResult visit(const slang::ast::ClassType &classdecl) {
    return context.convertClassDeclaration(classdecl);
  }

  // GenericClassDefSymbol represents parameterized (template) classes, which
  // per IEEE 1800-2023 §8.25 are abstract and not instantiable. Slang models
  // concrete specializations as ClassType, so we skip GenericClassDefSymbol
  // entirely.
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Skip typedefs.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }
  LogicalResult visit(const slang::ast::ForwardingTypedefSymbol &) {
    return success();
  }

  // Skip imports. The AST already has its names resolved.
  LogicalResult visit(const slang::ast::ExplicitImportSymbol &) {
    return success();
  }
  LogicalResult visit(const slang::ast::WildcardImportSymbol &) {
    return success();
  }

  // Skip type parameters. The Slang AST is already monomorphized.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Skip elaboration system tasks. These are reported directly by Slang.
  LogicalResult visit(const slang::ast::ElabSystemTaskSymbol &) {
    return success();
  }

  // Skip let declarations. These define expression macros that Slang expands
  // inline when they are used.
  LogicalResult visit(const slang::ast::LetDeclSymbol &) { return success(); }

  // Handle parameters.
  LogicalResult visit(const slang::ast::ParameterSymbol &param) {
    visitParameter(param);
    return success();
  }

  LogicalResult visit(const slang::ast::SpecparamSymbol &param) {
    visitParameter(param);
    return success();
  }

  template <class Node>
  void visitParameter(const Node &param) {
    // If debug info is enabled, try to materialize the parameter's constant
    // value on a best-effort basis and create a `dbg.variable` to track the
    // value.
    if (!context.options.debugInfo)
      return;
    auto value =
        context.materializeConstant(param.getValue(), param.getType(), loc);
    if (!value)
      return;
    if (builder.getInsertionBlock()->getParentOp() == context.intoModuleOp)
      context.orderedRootOps.insert({param.location, value.getDefiningOp()});

    // Prefix the parameter name with the surrounding namespace to create
    // somewhat sane names in the IR.
    SmallString<64> paramName;
    guessNamespacePrefix(param.getParentScope()->asSymbol(), paramName);
    paramName += param.name;

    debug::VariableOp::create(builder, loc, builder.getStringAttr(paramName),
                              value, Value{});
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Top-Level Item Conversion
//===----------------------------------------------------------------------===//

namespace {
struct RootVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Handle packages.
  LogicalResult visit(const slang::ast::PackageSymbol &package) {
    return context.convertPackage(package);
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  // Handle global variables.
  LogicalResult visit(const slang::ast::VariableSymbol &var) {
    return context.convertGlobalVariable(var);
  }

  // Handle interface definitions without instances.
  LogicalResult visit(const slang::ast::DefinitionSymbol &definition) {
    if (definition.definitionKind != slang::ast::DefinitionKind::Interface)
      return success();
    if (definition.getInstanceCount() != 0)
      return success();

    auto &body = slang::ast::InstanceBodySymbol::fromDefinition(
        context.compilation, definition, definition.location,
        slang::ast::InstanceFlags::Uninstantiated, nullptr, nullptr, nullptr);
    return context.convertInterfaceHeader(&body) ? success() : failure();
  }

  // Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Package Conversion
//===----------------------------------------------------------------------===//

namespace {
struct PackageVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  // Handle global variables.
  LogicalResult visit(const slang::ast::VariableSymbol &var) {
    return context.convertGlobalVariable(var);
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported package member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Module Conversion
//===----------------------------------------------------------------------===//

static moore::ProcedureKind
convertProcedureKind(slang::ast::ProceduralBlockKind kind) {
  switch (kind) {
  case slang::ast::ProceduralBlockKind::Always:
    return moore::ProcedureKind::Always;
  case slang::ast::ProceduralBlockKind::AlwaysComb:
    return moore::ProcedureKind::AlwaysComb;
  case slang::ast::ProceduralBlockKind::AlwaysLatch:
    return moore::ProcedureKind::AlwaysLatch;
  case slang::ast::ProceduralBlockKind::AlwaysFF:
    return moore::ProcedureKind::AlwaysFF;
  case slang::ast::ProceduralBlockKind::Initial:
    return moore::ProcedureKind::Initial;
  case slang::ast::ProceduralBlockKind::Final:
    return moore::ProcedureKind::Final;
  }
  llvm_unreachable("all procedure kinds handled");
}

static moore::NetKind convertNetKind(slang::ast::NetType::NetKind kind) {
  switch (kind) {
  case slang::ast::NetType::Supply0:
    return moore::NetKind::Supply0;
  case slang::ast::NetType::Supply1:
    return moore::NetKind::Supply1;
  case slang::ast::NetType::Tri:
    return moore::NetKind::Tri;
  case slang::ast::NetType::TriAnd:
    return moore::NetKind::TriAnd;
  case slang::ast::NetType::TriOr:
    return moore::NetKind::TriOr;
  case slang::ast::NetType::TriReg:
    return moore::NetKind::TriReg;
  case slang::ast::NetType::Tri0:
    return moore::NetKind::Tri0;
  case slang::ast::NetType::Tri1:
    return moore::NetKind::Tri1;
  case slang::ast::NetType::UWire:
    return moore::NetKind::UWire;
  case slang::ast::NetType::Wire:
    return moore::NetKind::Wire;
  case slang::ast::NetType::WAnd:
    return moore::NetKind::WAnd;
  case slang::ast::NetType::WOr:
    return moore::NetKind::WOr;
  case slang::ast::NetType::Interconnect:
    return moore::NetKind::Interconnect;
  case slang::ast::NetType::UserDefined:
    return moore::NetKind::UserDefined;
  case slang::ast::NetType::Unknown:
    return moore::NetKind::Unknown;
  }
  llvm_unreachable("all net kinds handled");
}

// Convert slang drive strength to Moore drive strength.
static std::optional<moore::DriveStrength>
convertDriveStrength(std::optional<slang::ast::DriveStrength> strength) {
  if (!strength)
    return std::nullopt;
  switch (*strength) {
  case slang::ast::DriveStrength::Supply:
    return moore::DriveStrength::Supply;
  case slang::ast::DriveStrength::Strong:
    return moore::DriveStrength::Strong;
  case slang::ast::DriveStrength::Pull:
    return moore::DriveStrength::Pull;
  case slang::ast::DriveStrength::Weak:
    return moore::DriveStrength::Weak;
  case slang::ast::DriveStrength::HighZ:
    return moore::DriveStrength::HighZ;
  }
  llvm_unreachable("all drive strengths handled");
}

namespace {
struct ModuleVisitor : public BaseVisitor {
  using BaseVisitor::visit;

  // A prefix of block names such as `foo.bar.` to put in front of variable and
  // instance names.
  StringRef blockNamePrefix;

  ModuleVisitor(Context &context, Location loc, StringRef blockNamePrefix = "")
      : BaseVisitor(context, loc), blockNamePrefix(blockNamePrefix) {}

  // Skip ports which are already handled by the module itself.
  LogicalResult visit(const slang::ast::PortSymbol &) { return success(); }
  LogicalResult visit(const slang::ast::MultiPortSymbol &) { return success(); }
  LogicalResult visit(const slang::ast::InterfacePortSymbol &) {
    return success();
  }

  // Skip genvars.
  LogicalResult visit(const slang::ast::GenvarSymbol &genvarNode) {
    return success();
  }

  // Skip defparams which have been handled by slang.
  LogicalResult visit(const slang::ast::DefParamSymbol &) { return success(); }

  // Ignore type parameters. These have already been handled by Slang's type
  // checking.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Skip user-defined nettype declarations. These define custom net types with
  // optional resolution functions, but the actual net declarations using these
  // types are handled separately.
  LogicalResult visit(const slang::ast::NetType &) { return success(); }

  SmallString<64>
  formatInstanceName(const slang::ast::InstanceSymbolBase &instNode) {
    SmallString<64> name(blockNamePrefix);
    if (!instNode.arrayPath.empty()) {
      slang::SmallVector<slang::ConstantRange, 4> dimensions;
      instNode.getArrayDimensions(dimensions);
      name += instNode.getArrayName();
      auto dimCount = std::min(dimensions.size(), instNode.arrayPath.size());
      for (size_t i = 0; i < dimCount; ++i) {
        auto &range = dimensions[i];
        int32_t relIndex = static_cast<int32_t>(instNode.arrayPath[i]);
        int32_t actualIndex =
            range.isLittleEndian() ? range.lower() + relIndex
                                   : range.upper() - relIndex;
        name += '_';
        Twine(actualIndex).toVector(name);
      }
      return name;
    }

    name += instNode.name;
    return name;
  }

  // Handle instance arrays.
  LogicalResult visit(const slang::ast::InstanceArraySymbol &arrayNode) {
    for (const auto *element : arrayNode.elements) {
      if (!element)
        continue;
      if (failed(element->visit(ModuleVisitor(context, loc, blockNamePrefix))))
        return failure();
    }
    return success();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    using slang::ast::ArgumentDirection;
    using slang::ast::AssignmentExpression;
    using slang::ast::InterfacePortSymbol;
    using slang::ast::MultiPortSymbol;
    using slang::ast::PortSymbol;

    auto instanceName = formatInstanceName(instNode);

    // Check if this is an interface instance
    auto kind = instNode.body.getDefinition().definitionKind;
    if (kind == slang::ast::DefinitionKind::Interface) {
      // Handle interface instantiation
      auto *ifaceLowering = context.convertInterfaceHeader(&instNode.body);
      if (!ifaceLowering)
        return failure();

      // Create a virtual interface type referencing this interface
      auto ifaceRef =
          mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                       ifaceLowering->op.getSymName());
      auto vifType = moore::VirtualInterfaceType::get(
          builder.getContext(), ifaceRef);
      auto vifRefType = moore::RefType::get(vifType);

      // Create the interface instance op and store it for later lookup
      auto instOp = moore::InterfaceInstanceOp::create(
          builder, loc, vifRefType,
          builder.getStringAttr(instanceName),
          ifaceRef);

      // Store the interface instance for lookup when it's referenced
      // (e.g., in virtual interface assignments like `vif = intf`)
      context.interfaceInstances[&instNode] = instOp.getResult();

      // Register nested interface instances inside this interface instance.
      if (auto parentRefTy =
              dyn_cast<moore::RefType>(instOp.getResult().getType())) {
        if (auto parentVifTy = dyn_cast<moore::VirtualInterfaceType>(
                parentRefTy.getNestedType())) {
          Value parentVif = moore::ConversionOp::create(builder, loc,
                                                        parentVifTy,
                                                        instOp.getResult());

          for (auto &member : instNode.body.members()) {
            if (auto *childInst =
                    member.as_if<slang::ast::InstanceSymbol>()) {
              if (childInst->getDefinition().definitionKind !=
                  slang::ast::DefinitionKind::Interface)
                continue;
              if (!shouldCacheInterfaceInstance(*childInst))
                continue;

              auto *ifaceLowering =
                  context.convertInterfaceHeader(&childInst->body);
              if (!ifaceLowering)
                return failure();

              auto childIfaceRef = mlir::FlatSymbolRefAttr::get(
                  builder.getContext(), ifaceLowering->op.getSymName());
              auto childVifTy = moore::VirtualInterfaceType::get(
                  builder.getContext(), childIfaceRef);
              auto childRefTy = moore::RefType::get(childVifTy);
              auto signalSym = mlir::FlatSymbolRefAttr::get(
                  builder.getContext(), childInst->name);

              Value childRef = moore::VirtualInterfaceSignalRefOp::create(
                  builder, loc, childRefTy, parentVif, signalSym);
              context.interfaceInstances[childInst] = childRef;
            }
          }
        }
      }

      if (!instNode.getPortConnections().empty())
        context.pendingInterfacePortConnections.push_back(
            {&instNode, instOp.getResult(), loc});

      return success();
    }

    auto *moduleLowering = context.convertModuleHeader(&instNode.body);
    if (!moduleLowering)
      return failure();
    auto module = moduleLowering->op;
    auto moduleType = module.getModuleType();

    // Set visibility attribute for instantiated module.
    SymbolTable::setSymbolVisibility(module, SymbolTable::Visibility::Private);

    // Prepare the values that are involved in port connections. This creates
    // rvalues for input ports and appropriate lvalues for output, inout, and
    // ref ports. We also separate multi-ports into the individual underlying
    // ports with their corresponding connection.
    SmallDenseMap<const slang::ast::Symbol *, Value> portValues;
    portValues.reserve(moduleType.getNumPorts());

    auto canonicalPortSymbol =
        [&](const slang::ast::Symbol &symbol) -> const slang::ast::Symbol * {
      if (auto *existing =
              moduleLowering->portsBySyntaxNode.lookup(symbol.getSyntax()))
        return existing;
      return &symbol;
    };

    for (const auto *con : instNode.getPortConnections()) {
      const auto *portSymbol = canonicalPortSymbol(con->port);

      if (auto *ifacePort = portSymbol->as_if<InterfacePortSymbol>()) {
        auto [ifaceConn, modportSym] = con->getIfaceConn();
        (void)modportSym;
        if (!ifaceConn) {
          return mlir::emitError(loc)
                 << "unsupported unconnected interface port `"
                 << ifacePort->name << "`";
        }
        Value ifaceValue;
        if (auto *expr = con->getExpression()) {
          if (auto symRef = expr->getSymbolReference()) {
            if (auto *instSym =
                    symRef->as_if<slang::ast::InstanceSymbol>()) {
              ifaceValue = context.resolveInterfaceInstance(instSym, loc);
            } else if (auto *ifacePortSym =
                           symRef
                               ->as_if<slang::ast::InterfacePortSymbol>()) {
              if (auto it = context.interfacePortValues.find(ifacePortSym);
                  it != context.interfacePortValues.end())
                ifaceValue = it->second;
            }
          }
          if (auto *arb =
                  expr->as_if<slang::ast::ArbitrarySymbolExpression>()) {
            ifaceValue = context.resolveInterfaceInstance(arb->hierRef, loc);
          } else if (auto *hier =
                         expr->as_if<
                             slang::ast::HierarchicalValueExpression>()) {
            ifaceValue = context.resolveInterfaceInstance(hier->ref, loc);
          } else if (auto *elemSel =
                         expr->as_if<slang::ast::ElementSelectExpression>()) {
            auto getArraySym =
                [&](const slang::ast::Expression &valueExpr)
                -> const slang::ast::InstanceArraySymbol * {
              if (auto symRef = valueExpr.getSymbolReference())
                return symRef->as_if<slang::ast::InstanceArraySymbol>();
              if (auto *hier =
                      valueExpr
                          .as_if<slang::ast::HierarchicalValueExpression>()) {
                if (!hier->ref.path.empty())
                  return hier->ref.path.back()
                      .symbol->as_if<slang::ast::InstanceArraySymbol>();
              }
              if (auto *arb =
                      valueExpr
                          .as_if<slang::ast::ArbitrarySymbolExpression>()) {
                if (!arb->hierRef.path.empty())
                  return arb->hierRef.path.back()
                      .symbol->as_if<slang::ast::InstanceArraySymbol>();
              }
              return nullptr;
            };

            if (auto *arraySym = getArraySym(elemSel->value())) {
              auto constIndex = context.evaluateConstant(elemSel->selector());
              if (constIndex.isInteger()) {
                auto index = constIndex.integer().as<int64_t>();
                if (index) {
                  if (auto *element =
                          findInterfaceArrayElement(*arraySym, *index)) {
                    ifaceValue =
                        context.resolveInterfaceInstance(element, loc);
                  }
                }
              }
            }
          }
        }
        if (!ifaceValue) {
          if (auto *instSym = ifaceConn->as_if<slang::ast::InstanceSymbol>()) {
            ifaceValue = context.resolveInterfaceInstance(instSym, loc);
          } else if (auto *ifacePortSym =
                         ifaceConn
                             ->as_if<slang::ast::InterfacePortSymbol>()) {
            if (auto it = context.interfacePortValues.find(ifacePortSym);
                it != context.interfacePortValues.end())
              ifaceValue = it->second;
          } else {
            return mlir::emitError(loc)
                   << "unsupported interface port connection for `"
                   << ifacePort->name << "`";
          }
        }
        if (ifaceValue && ifacePort->interfaceDef) {
          if (auto *connInst =
                  ifaceConn->as_if<slang::ast::InstanceSymbol>()) {
            const auto &connDef = connInst->getDefinition();
            if (connDef.definitionKind ==
                    slang::ast::DefinitionKind::Interface &&
                &connDef != ifacePort->interfaceDef) {
              const slang::ast::InstanceSymbol *match = nullptr;
              for (auto &member : connInst->body.members()) {
                auto *childInst =
                    member.as_if<slang::ast::InstanceSymbol>();
                if (!childInst)
                  continue;
                if (childInst->getDefinition().definitionKind !=
                    slang::ast::DefinitionKind::Interface)
                  continue;
                if (&childInst->getDefinition() != ifacePort->interfaceDef)
                  continue;
                if (match) {
                  return mlir::emitError(loc)
                         << "ambiguous nested interface instance for port `"
                         << ifacePort->name << "`";
                }
                match = childInst;
              }

              if (match) {
                auto parentRefTy =
                    dyn_cast<moore::RefType>(ifaceValue.getType());
                if (!parentRefTy)
                  return mlir::emitError(loc)
                         << "invalid interface connection type for `"
                         << ifacePort->name << "`";
                auto parentVifTy =
                    dyn_cast<moore::VirtualInterfaceType>(
                        parentRefTy.getNestedType());
                if (!parentVifTy)
                  return mlir::emitError(loc)
                         << "invalid interface connection type for `"
                         << ifacePort->name << "`";

                Value parentVif = moore::ConversionOp::create(
                    builder, loc, parentVifTy, ifaceValue);
                auto *ifaceLowering =
                    context.convertInterfaceHeader(&match->body);
                if (!ifaceLowering)
                  return failure();
                auto childIfaceRef = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), ifaceLowering->op.getSymName());
                auto childVifTy = moore::VirtualInterfaceType::get(
                    builder.getContext(), childIfaceRef);
                auto childRefTy = moore::RefType::get(childVifTy);
                auto signalSym = mlir::FlatSymbolRefAttr::get(
                    builder.getContext(), match->name);
                ifaceValue = moore::VirtualInterfaceSignalRefOp::create(
                    builder, loc, childRefTy, parentVif, signalSym);
              }
            }
          }
        }
        if (!ifaceValue) {
          return mlir::emitError(loc)
                 << "unknown interface instance for port `" << ifacePort->name
                 << "`";
        }
        portValues.insert({portSymbol, ifaceValue});
        continue;
      }

      const auto *expr = con->getExpression();

      // Handle unconnected behavior. The expression is null if it have no
      // connection for the port.
      if (!expr) {
        if (auto *port = portSymbol->as_if<PortSymbol>()) {
          switch (port->direction) {
          case ArgumentDirection::In: {
            auto refType = moore::RefType::get(
                cast<moore::UnpackedType>(context.convertType(port->getType())));

            if (const auto *net =
                    port->internalSymbol->as_if<slang::ast::NetSymbol>()) {
              auto netOp = moore::NetOp::create(
                  builder, loc, refType,
                  StringAttr::get(builder.getContext(), net->name),
                  convertNetKind(net->netType.netKind), nullptr);
              auto readOp = moore::ReadOp::create(builder, loc, netOp);
              portValues.insert({portSymbol, readOp});
            } else if (const auto *var =
                           port->internalSymbol
                               ->as_if<slang::ast::VariableSymbol>()) {
              auto varOp = moore::VariableOp::create(
                  builder, loc, refType,
                  StringAttr::get(builder.getContext(), var->name), nullptr);
              auto readOp = moore::ReadOp::create(builder, loc, varOp);
              portValues.insert({portSymbol, readOp});
            } else {
              return mlir::emitError(loc)
                     << "unsupported internal symbol for unconnected port `"
                     << port->name << "`";
            }
            continue;
          }

          // No need to express unconnected behavior for output port, skip to the
          // next iteration of the loop.
          case ArgumentDirection::Out:
            continue;

          // TODO: Mark Inout port as unsupported and it will be supported later.
          default:
            return mlir::emitError(loc)
                   << "unsupported port `" << port->name << "` ("
                   << slang::ast::toString(port->kind) << ")";
          }
        }

        return mlir::emitError(loc)
               << "unsupported port `" << portSymbol->name << "` ("
               << slang::ast::toString(portSymbol->kind) << ")";
      }

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for
      // output and inout ports.
      if (const auto *assign = expr->as_if<AssignmentExpression>())
        expr = &assign->left();

      // Regular ports lower the connected expression to an lvalue or rvalue and
      // either attach it to the instance as an operand (for input, inout, and
      // ref ports), or assign an instance output to it (for output ports).
      if (auto *port = portSymbol->as_if<PortSymbol>()) {
        // Convert as rvalue for inputs, lvalue for all others.
        auto value = (port->direction == ArgumentDirection::In)
                         ? context.convertRvalueExpression(*expr)
                         : context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        portValues.insert({portSymbol, value});
        continue;
      }

      // Multi-ports lower the connected expression to an lvalue and then slice
      // it up into multiple sub-values, one for each of the ports in the
      // multi-port.
      if (const auto *multiPort = portSymbol->as_if<MultiPortSymbol>()) {
        // Convert as lvalue.
        auto value = context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        unsigned offset = 0;
        for (const auto *port : llvm::reverse(multiPort->ports)) {
          unsigned width = port->getType().getBitWidth();
          auto sliceType = context.convertType(port->getType());
          if (!sliceType)
            return failure();
          Value slice = moore::ExtractRefOp::create(
              builder, loc,
              moore::RefType::get(cast<moore::UnpackedType>(sliceType)), value,
              offset);
          // Create the "ReadOp" for input ports.
          if (port->direction == ArgumentDirection::In)
            slice = moore::ReadOp::create(builder, loc, slice);
          if (const auto *slicePort =
                  canonicalPortSymbol(*port)->as_if<PortSymbol>())
            portValues.insert({slicePort, slice});
          offset += width;
        }
        continue;
      }

      mlir::emitError(loc) << "unsupported instance port `" << con->port.name
                           << "` (" << slang::ast::toString(con->port.kind)
                           << ")";
      return failure();
    }

    // Match the module's ports up with the port values determined above.
    SmallVector<Value> inputValues;
    SmallVector<Value> outputValues;
    inputValues.reserve(moduleType.getNumInputs());
    outputValues.reserve(moduleType.getNumOutputs());

    for (auto &port : moduleLowering->ports) {
      auto value = portValues.lookup(port.symbol);
      if (auto *portSym = port.symbol->as_if<PortSymbol>()) {
        if (portSym->direction == ArgumentDirection::Out)
          outputValues.push_back(value);
        else
          inputValues.push_back(value);
      } else {
        inputValues.push_back(value);
      }
    }

    // Insert conversions for input ports.
    for (auto [value, type] :
         llvm::zip(inputValues, moduleType.getInputTypes()))
      // TODO: This should honor signedness in the conversion.
      value = context.materializeConversion(type, value, false, value.getLoc());

    // Here we use the hierarchical value recorded in `Context::valueSymbols`.
    // Then we pass it as the input port with the ref<T> type of the instance.
    for (const auto &hierPath : context.hierPaths[&instNode.body]) {
      if (!hierPath.hierName ||
          hierPath.direction != ArgumentDirection::In)
        continue;
      auto hierValue = context.valueSymbols.lookup(hierPath.valueSym);
      if (!hierValue) {
        mlir::emitError(loc)
            << "missing hierarchical value for `" << hierPath.hierName.getValue()
            << "`";
        return failure();
      }
      inputValues.push_back(hierValue);
    }

    // Create the instance op itself.
    auto inputNames = builder.getArrayAttr(moduleType.getInputNames());
    auto outputNames = builder.getArrayAttr(moduleType.getOutputNames());
    auto inst = moore::InstanceOp::create(
        builder, loc, moduleType.getOutputTypes(),
        builder.getStringAttr(instanceName),
        FlatSymbolRefAttr::get(module.getSymNameAttr()), inputValues,
        inputNames, outputNames);

    // Record instance's results generated by hierarchical names.
    for (const auto &hierPath : context.hierPaths[&instNode.body]) {
      if (!hierPath.idx || hierPath.direction != ArgumentDirection::Out)
        continue;
      Value result = inst->getResult(*hierPath.idx);
      if (auto it = context.hierValuePlaceholders.find(hierPath.valueSym);
          it != context.hierValuePlaceholders.end()) {
        it->second.replaceAllUsesWith(result);
        if (auto *def = it->second.getDefiningOp())
          if (def->use_empty())
            def->erase();
        context.hierValuePlaceholders.erase(it);
      }
      context.valueSymbols.insert(hierPath.valueSym, result);
    }

    // Assign output values from the instance to the connected expression.
    for (auto [lvalue, output] : llvm::zip(outputValues, inst.getOutputs())) {
      if (!lvalue)
        continue;
      Value rvalue = output;
      auto dstType = cast<moore::RefType>(lvalue.getType()).getNestedType();
      // TODO: This should honor signedness in the conversion.
      rvalue = context.materializeConversion(dstType, rvalue, false, loc);
      moore::ContinuousAssignOp::create(builder, loc, lvalue, rvalue);
    }

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      // Special handling for dynamic array initialization with assignment
      // patterns. When the target is an OpenUnpackedArrayType (e.g., string[]),
      // slang may report the assignment pattern type as a packed integer
      // (concatenation), but we need to treat each element as a separate value.
      if (auto arrayType = dyn_cast<moore::OpenUnpackedArrayType>(loweredType)) {
        // Check for SimpleAssignmentPatternExpression
        const slang::ast::AssignmentPatternExpressionBase *pattern = nullptr;
        if (auto *simple =
                init->as_if<slang::ast::SimpleAssignmentPatternExpression>())
          pattern = simple;
        else if (auto *structured =
                     init->as_if<slang::ast::StructuredAssignmentPatternExpression>())
          pattern = structured;
        else if (auto *replicated =
                     init->as_if<slang::ast::ReplicatedAssignmentPatternExpression>())
          pattern = replicated;
        // Also check for ConcatenationExpression - slang may interpret
        // { "a", "b" } as a concatenation rather than an assignment pattern.
        if (!pattern) {
          if (auto *concat =
                  init->as_if<slang::ast::ConcatenationExpression>()) {
            auto elementType = arrayType.getElementType();
            SmallVector<Value> elements;
            bool success = true;
            for (const auto *operand : concat->operands()) {
              Value elem = context.convertRvalueExpression(*operand, elementType);
              if (!elem) {
                success = false;
                break;
              }
              elements.push_back(elem);
            }
            if (success && !elements.empty()) {
              // Create a queue, push elements, then convert to dynamic array.
              auto queueType =
                  moore::QueueType::get(elementType, /*bound=*/0);
              Value queueValue =
                  moore::QueueConcatOp::create(builder, loc, queueType, {});
              auto refTy = moore::RefType::get(queueType);
              auto tmpVar = moore::VariableOp::create(
                  builder, loc, refTy,
                  builder.getStringAttr("dyn_array_init_tmp"), queueValue);
              for (Value elem : elements)
                moore::QueuePushBackOp::create(builder, loc, tmpVar, elem);
              Value result = moore::ReadOp::create(builder, loc, tmpVar);
              initial =
                  moore::ConversionOp::create(builder, loc, arrayType, result);
            }
          }
        }

        if (pattern) {
          auto elementType = arrayType.getElementType();
          SmallVector<Value> elements;
          bool success = true;
          for (const auto *elemExpr : pattern->elements()) {
            Value elem = context.convertRvalueExpression(*elemExpr, elementType);
            if (!elem) {
              success = false;
              break;
            }
            elements.push_back(elem);
          }
          if (success && !elements.empty()) {
            // Create a queue, push elements, then convert to dynamic array.
            auto queueType =
                moore::QueueType::get(elementType, /*bound=*/0);
            Value queueValue =
                moore::QueueConcatOp::create(builder, loc, queueType, {});
            auto refTy = moore::RefType::get(queueType);
            auto tmpVar = moore::VariableOp::create(
                builder, loc, refTy,
                builder.getStringAttr("dyn_array_init_tmp"), queueValue);
            for (Value elem : elements)
              moore::QueuePushBackOp::create(builder, loc, tmpVar, elem);
            Value result = moore::ReadOp::create(builder, loc, tmpVar);
            initial =
                moore::ConversionOp::create(builder, loc, arrayType, result);
          }
        }
      }
      // Fall back to standard conversion if special handling didn't work.
      if (!initial) {
        initial = context.convertRvalueExpression(*init, loweredType);
        if (!initial)
          return failure();
      }
    }

    auto varOp = moore::VariableOp::create(
        builder, loc,
        moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(Twine(blockNamePrefix) + varNode.name), initial);
    context.valueSymbols.insert(&varNode, varOp);
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto loweredType = context.convertType(*netNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (const auto *init = netNode.getInitializer()) {
      assignment = context.convertRvalueExpression(*init, loweredType);
      if (!assignment)
        return failure();
    }

    auto netkind = convertNetKind(netNode.netType.netKind);
    if (netkind == moore::NetKind::UserDefined ||
        netkind == moore::NetKind::Unknown)
      return mlir::emitError(loc, "unsupported net kind `")
             << netNode.netType.name << "`";

    auto netOp = moore::NetOp::create(
        builder, loc,
        moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(Twine(blockNamePrefix) + netNode.name), netkind,
        assignment);
    context.valueSymbols.insert(&netNode, netOp);
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    const auto *expr =
        assignNode.getAssignment().as_if<slang::ast::AssignmentExpression>();
    if (!expr) {
      if (context.options.allowNonProceduralDynamic.value_or(false)) {
        // When DynamicNotProcedural is downgraded to a warning, slang wraps
        // the expression in an InvalidExpression. The problem is that slang
        // wraps just the base expression (e.g., "obj"), not the full member
        // access ("obj.val"). To recover the full expression, we re-bind from
        // syntax in a procedural context (without NonProcedural flag).
        const auto *syntax = assignNode.getSyntax();
        if (syntax && context.currentScope) {
          // Get the expression syntax from the continuous assign
          const auto *exprSyntax =
              syntax->as_if<slang::syntax::ExpressionSyntax>();
          if (exprSyntax) {
            // Create an AST context without NonProcedural flag - this allows
            // dynamic type access to succeed
            slang::ast::ASTContext astContext(
                *context.currentScope, slang::ast::LookupLocation::max);

            // Re-bind the expression in the new (procedural) context
            const auto &reboundExpr = slang::ast::Expression::bind(
                *exprSyntax, astContext, slang::ast::ASTFlags::AssignmentAllowed);
            if (!reboundExpr.bad()) {
              // The re-bound expression should now be a valid
              // AssignmentExpression
              const auto *assignExpr =
                  reboundExpr.as_if<slang::ast::AssignmentExpression>();
              if (assignExpr) {
                // Convert to always_comb for procedural context
                auto procOp = moore::ProcedureOp::create(
                    builder, loc, moore::ProcedureKind::AlwaysComb);
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToEnd(
                    &procOp.getBody().emplaceBlock());
                Context::ValueSymbolScope scope(context.valueSymbols);

                auto lhs = context.convertLvalueExpression(assignExpr->left());
                if (!lhs) {
                  procOp.erase();
                  mlir::emitWarning(loc)
                      << "skipping continuous assignment: failed to convert "
                         "LHS in always_comb fallback";
                  return success();
                }

                // Get the nested type from the lvalue
                Type lhsNestedType;
                if (auto refType = dyn_cast<moore::RefType>(lhs.getType()))
                  lhsNestedType = refType.getNestedType();
                else if (isa<moore::ClassHandleType>(lhs.getType()))
                  lhsNestedType = lhs.getType();
                else {
                  procOp.erase();
                  mlir::emitWarning(loc)
                      << "skipping continuous assignment: unsupported LHS "
                         "type in always_comb fallback";
                  return success();
                }

                auto rhs = context.convertRvalueExpression(assignExpr->right(),
                                                           lhsNestedType);
                if (!rhs) {
                  procOp.erase();
                  mlir::emitWarning(loc)
                      << "skipping continuous assignment: failed to convert "
                         "RHS in always_comb fallback";
                  return success();
                }

                moore::BlockingAssignOp::create(builder, loc, lhs, rhs);
                moore::ReturnOp::create(builder, loc);

                mlir::emitRemark(loc)
                    << "converted continuous assignment with dynamic type "
                       "access to always_comb block";
                return success();
              }
            }
          }
        }
        mlir::emitWarning(loc)
            << "skipping continuous assignment without an assignment "
               "expression after DynamicNotProcedural downgrade";
        return success();
      }
      mlir::emitError(loc)
          << "expected assignment expression in continuous assignment";
      return failure();
    }
    auto lhs = context.convertLvalueExpression(expr->left());
    if (!lhs)
      return failure();

    // Get the nested type from the lvalue. This handles both RefType (for
    // regular variables) and ClassHandleType (for class handles).
    Type lhsNestedType;
    if (auto refType = dyn_cast<moore::RefType>(lhs.getType()))
      lhsNestedType = refType.getNestedType();
    else if (isa<moore::ClassHandleType>(lhs.getType()))
      lhsNestedType = lhs.getType();
    else {
      mlir::emitError(loc) << "unsupported lvalue type in continuous assign: "
                           << lhs.getType();
      return failure();
    }

    auto rhs = context.convertRvalueExpression(expr->right(), lhsNestedType);
    if (!rhs)
      return failure();

    // Handle delayed assignments.
    if (auto *timingCtrl = assignNode.getDelay()) {
      auto *ctrl = timingCtrl->as_if<slang::ast::DelayControl>();
      assert(ctrl && "slang guarantees this to be a simple delay");
      auto delay = context.convertRvalueExpression(
          ctrl->expr, moore::TimeType::get(builder.getContext()));
      if (!delay)
        return failure();
      moore::DelayedContinuousAssignOp::create(builder, loc, lhs, rhs, delay);
      return success();
    }

    // Extract drive strength from the assignment.
    auto [strength0, strength1] = assignNode.getDriveStrength();
    auto str0 = convertDriveStrength(strength0);
    auto str1 = convertDriveStrength(strength1);

    // Create attribute values for the optional strengths.
    moore::DriveStrengthAttr str0Attr;
    moore::DriveStrengthAttr str1Attr;
    if (str0)
      str0Attr = moore::DriveStrengthAttr::get(builder.getContext(), *str0);
    if (str1)
      str1Attr = moore::DriveStrengthAttr::get(builder.getContext(), *str1);

    // Otherwise this is a regular assignment.
    moore::ContinuousAssignOp::create(builder, loc, lhs, rhs, str0Attr,
                                      str1Attr);
    return success();
  }

  // Handle procedures.
  LogicalResult convertProcedure(moore::ProcedureKind kind,
                                 const slang::ast::Statement &body) {
    if (body.as_if<slang::ast::ConcurrentAssertionStatement>())
      return context.convertStatement(body);
    auto procOp = moore::ProcedureOp::create(builder, loc, kind);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&procOp.getBody().emplaceBlock());
    Context::ValueSymbolScope scope(context.valueSymbols);
    if (failed(context.convertStatement(body)))
      return failure();
    if (builder.getBlock())
      moore::ReturnOp::create(builder, loc);
    return success();
  }

  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto isConcurrentAssertionBlock =
        [](const slang::ast::Statement &stmt,
           const auto &self) -> bool {
      if (stmt.as_if<slang::ast::ConcurrentAssertionStatement>())
        return true;
      if (auto *block = stmt.as_if<slang::ast::BlockStatement>())
        return self(block->body, self);
      if (auto *list = stmt.as_if<slang::ast::StatementList>()) {
        if (list->list.size() == 1)
          return self(*list->list[0], self);
      }
      return false;
    };

    if (procNode.isFromAssertion &&
        isConcurrentAssertionBlock(procNode.getBody(),
                                   isConcurrentAssertionBlock))
      return context.convertStatement(procNode.getBody());
    // Detect `always @(*) <stmt>` and convert to `always_comb <stmt>` if
    // requested by the user.
    if (context.options.lowerAlwaysAtStarAsComb) {
      auto *stmt = procNode.getBody().as_if<slang::ast::TimedStatement>();
      if (procNode.procedureKind == slang::ast::ProceduralBlockKind::Always &&
          stmt &&
          stmt->timing.kind == slang::ast::TimingControlKind::ImplicitEvent)
        return convertProcedure(moore::ProcedureKind::AlwaysComb, stmt->stmt);
    }

    return convertProcedure(convertProcedureKind(procNode.procedureKind),
                            procNode.getBody());
  }

  // Handle generate block.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    // Ignore uninstantiated blocks.
    if (genNode.isUninstantiated)
      return success();

    // If the block has a name, add it to the list of block name prefices.
    SmallString<64> prefix = blockNamePrefix;
    if (!genNode.name.empty() ||
        genNode.getParentScope()->asSymbol().kind !=
            slang::ast::SymbolKind::GenerateBlockArray) {
      prefix += genNode.getExternalName();
      prefix += '.';
    }

    // Visit each member of the generate block.
    for (auto &member : genNode.members())
      if (failed(member.visit(ModuleVisitor(context, loc, prefix))))
        return failure();
    return success();
  }

  // Handle generate block array.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArrNode) {
    // If the block has a name, add it to the list of block name prefices and
    // prepare to append the array index and a `.` in each iteration.
    SmallString<64> prefix = blockNamePrefix;
    prefix += genArrNode.getExternalName();
    prefix += '_';
    auto prefixBaseLen = prefix.size();

    // Visit each iteration entry of the generate block.
    for (const auto *entry : genArrNode.entries) {
      // Append the index to the prefix.
      prefix.resize(prefixBaseLen);
      if (entry->arrayIndex)
        prefix += entry->arrayIndex->toString();
      else
        Twine(entry->constructIndex).toVector(prefix);
      prefix += '.';

      // Visit this iteration entry.
      if (failed(entry->asSymbol().visit(ModuleVisitor(context, loc, prefix))))
        return failure();
    }
    return success();
  }

  // Ignore statement block symbols. These get generated by Slang for blocks
  // with variables and other declarations. For example, having an initial
  // procedure with a variable declaration, such as `initial begin int x;
  // end`, will create the procedure with a block and variable declaration as
  // expected, but will also create a `StatementBlockSymbol` with just the
  // variable layout _next to_ the initial procedure.
  LogicalResult visit(const slang::ast::StatementBlockSymbol &) {
    return success();
  }

  // Ignore sequence declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::SequenceSymbol &seqNode) {
    return success();
  }

  // Ignore property declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::PropertySymbol &propNode) {
    return success();
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  // Handle covergroup type definitions
  LogicalResult visit(const slang::ast::CovergroupType &cg) {
    return context.convertCovergroup(cg);
  }

  // Skip covergroup body - handled by convertCovergroup
  LogicalResult visit(const slang::ast::CovergroupBodySymbol &) {
    return success();
  }

  // Skip individual coverage symbols - handled by convertCovergroup
  LogicalResult visit(const slang::ast::CoverpointSymbol &) {
    return success();
  }

  LogicalResult visit(const slang::ast::CoverCrossSymbol &) {
    return success();
  }

  LogicalResult visit(const slang::ast::CoverCrossBodySymbol &) {
    return success();
  }

  // Handle clocking blocks
  LogicalResult visit(const slang::ast::ClockingBlockSymbol &clockingBlock) {
    // Create the ClockingBlockDeclOp
    auto clockingOp = moore::ClockingBlockDeclOp::create(
        builder, loc, clockingBlock.name);

    // Create the body block for the clocking block
    clockingOp.getBody().emplaceBlock();

    // Set insertion point to inside the clocking block body
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&clockingOp.getBody().front());

    // Iterate over clocking block members (ClockVarSymbol)
    for (auto &member : clockingBlock.members()) {
      if (auto *clockVar =
              member.as_if<slang::ast::ClockVarSymbol>()) {
        auto memberLoc = context.convertLocation(clockVar->location);
        auto type = context.convertType(clockVar->getType());
        if (!type)
          return failure();
        moore::ClockingSignalOp::create(builder, memberLoc, clockVar->name,
                                        type);
      }
    }

    return success();
  }

  // Handle primitive instances (pullup, pulldown, gate primitives).
  LogicalResult visit(const slang::ast::PrimitiveInstanceSymbol &primNode) {
    auto primName = primNode.primitiveType.name;

    // Get the port connections for this primitive instance.
    auto portConnections = primNode.getPortConnections();
    if (portConnections.empty()) {
      mlir::emitError(loc) << "primitive `" << primName
                           << "` has no port connections";
      return failure();
    }

    // Helper lambda to unpack the `<expr> = EmptyArgument` pattern emitted
    // by Slang for output ports.
    auto unpackOutputExpr = [](const slang::ast::Expression *expr)
        -> const slang::ast::Expression * {
      using slang::ast::AssignmentExpression;
      if (const auto *assign = expr->as_if<AssignmentExpression>())
        return &assign->left();
      return expr;
    };

    // Handle pullup and pulldown primitives.
    // These drive a constant 1 or 0 onto the connected net.
    if (primName == "pullup" || primName == "pulldown") {
      // pullup/pulldown have exactly one output port.
      if (portConnections.size() != 1) {
        mlir::emitError(loc) << primName << " primitive expects 1 port, got "
                             << portConnections.size();
        return failure();
      }

      // Get the target net expression - unpack AssignmentExpression wrapper.
      const auto *portExpr = unpackOutputExpr(portConnections[0]);
      auto target = context.convertLvalueExpression(*portExpr);
      if (!target)
        return failure();

      // Get the type of the target to create an appropriately-sized constant.
      auto refType = dyn_cast<moore::RefType>(target.getType());
      if (!refType) {
        mlir::emitError(loc) << primName
                             << " target must be a reference type";
        return failure();
      }
      auto targetType = refType.getNestedType();

      // Create the constant value: 1 for pullup, 0 for pulldown.
      // Note: In a full 4-state simulation, pullup drives a weak 1 and
      // pulldown drives a weak 0. For now, we model this as a continuous
      // assignment of a constant value.
      bool isPullup = (primName == "pullup");
      Value constVal;

      // Try to get the bit size for the target type. This works for both
      // IntType and packed array types.
      auto packedType = dyn_cast<moore::PackedType>(targetType);
      if (!packedType) {
        mlir::emitError(loc) << primName
                             << " target must be a packed type, got "
                             << targetType;
        return failure();
      }

      auto bitSize = packedType.getBitSize();
      if (!bitSize) {
        mlir::emitError(loc) << primName
                             << " target must have a known bit size";
        return failure();
      }

      // Get the simple bit vector type for this packed type.
      auto sbvType = packedType.getSimpleBitVector();
      if (!sbvType) {
        mlir::emitError(loc) << primName
                             << " target cannot be converted to a simple bit "
                             << "vector type";
        return failure();
      }

      // Create the constant with the appropriate value.
      APInt value =
          isPullup ? APInt::getAllOnes(*bitSize) : APInt::getZero(*bitSize);
      constVal = moore::ConstantOp::create(builder, loc, sbvType, value);

      // If the target type is different from the simple bit vector type,
      // we need a conversion.
      if (sbvType != targetType) {
        constVal =
            moore::ConversionOp::create(builder, loc, targetType, constVal);
      }

      // Extract drive strength from the primitive instance.
      // Pullup/pulldown primitives can have an optional strength modifier.
      // The syntax is: pullup (strength1) (net) or pulldown (strength0) (net)
      // Default is Pull strength if not specified.
      auto [strength0, strength1] = primNode.getDriveStrength();
      moore::DriveStrengthAttr str0Attr;
      moore::DriveStrengthAttr str1Attr;

      if (isPullup) {
        // For pullup: strength1 is the relevant one (driving 1).
        // strength0 should be HighZ (not driving 0).
        str0Attr = moore::DriveStrengthAttr::get(builder.getContext(),
                                                  moore::DriveStrength::HighZ);
        auto str1 = convertDriveStrength(strength1);
        str1Attr = moore::DriveStrengthAttr::get(
            builder.getContext(), str1.value_or(moore::DriveStrength::Pull));
      } else {
        // For pulldown: strength0 is the relevant one (driving 0).
        // strength1 should be HighZ (not driving 1).
        auto str0 = convertDriveStrength(strength0);
        str0Attr = moore::DriveStrengthAttr::get(
            builder.getContext(), str0.value_or(moore::DriveStrength::Pull));
        str1Attr = moore::DriveStrengthAttr::get(builder.getContext(),
                                                  moore::DriveStrength::HighZ);
      }

      // Create a continuous assignment to drive the constant onto the net.
      moore::ContinuousAssignOp::create(builder, loc, target, constVal, str0Attr,
                                        str1Attr);
      return success();
    }

    // Handle basic gate primitives: and, or, nand, nor, xor, xnor.
    // These gates have one output followed by two or more inputs.
    // Output = f(input0, input1, ..., inputN)
    if (primName == "and" || primName == "or" || primName == "nand" ||
        primName == "nor" || primName == "xor" || primName == "xnor") {
      // Gate primitives need at least 3 ports: 1 output + 2+ inputs
      if (portConnections.size() < 3) {
        mlir::emitError(loc) << primName << " primitive expects at least 3 ports "
                             << "(1 output + 2 inputs), got "
                             << portConnections.size();
        return failure();
      }

      // First port is the output - unpack the AssignmentExpression wrapper
      const auto *outputExpr = unpackOutputExpr(portConnections[0]);
      auto output = context.convertLvalueExpression(*outputExpr);
      if (!output)
        return failure();

      auto refType = dyn_cast<moore::RefType>(output.getType());
      if (!refType) {
        mlir::emitError(loc) << primName << " output must be a reference type";
        return failure();
      }
      auto outputType = refType.getNestedType();

      // Collect all inputs (starting from index 1)
      SmallVector<Value, 4> inputs;
      for (size_t i = 1; i < portConnections.size(); ++i) {
        const auto *inputExpr = portConnections[i];
        auto input = context.convertRvalueExpression(*inputExpr);
        if (!input)
          return failure();
        // Convert to simple bit vector if needed
        input = context.convertToSimpleBitVector(input);
        if (!input)
          return failure();
        inputs.push_back(input);
      }

      // Compute the gate output by reducing all inputs
      Value result = inputs[0];
      for (size_t i = 1; i < inputs.size(); ++i) {
        if (primName == "and" || primName == "nand") {
          result = moore::AndOp::create(builder, loc, result, inputs[i]);
        } else if (primName == "or" || primName == "nor") {
          result = moore::OrOp::create(builder, loc, result, inputs[i]);
        } else { // xor, xnor
          result = moore::XorOp::create(builder, loc, result, inputs[i]);
        }
      }

      // Apply inversion for nand, nor, xnor
      if (primName == "nand" || primName == "nor" || primName == "xnor") {
        result = moore::NotOp::create(builder, loc, result);
      }

      // Convert result to output type if needed
      if (result.getType() != outputType) {
        result = moore::ConversionOp::create(builder, loc, outputType, result);
      }

      // Assign the result to the output
      moore::ContinuousAssignOp::create(builder, loc, output, result);
      return success();
    }

    // Handle buf and not primitives.
    // These have one or more outputs followed by exactly one input.
    // Each output = input (for buf) or ~input (for not)
    if (primName == "buf" || primName == "not") {
      // Need at least 2 ports: 1+ outputs + 1 input
      if (portConnections.size() < 2) {
        mlir::emitError(loc) << primName << " primitive expects at least 2 ports "
                             << "(1+ outputs + 1 input), got "
                             << portConnections.size();
        return failure();
      }

      // Last port is the input
      const auto *inputExpr = portConnections.back();
      auto input = context.convertRvalueExpression(*inputExpr);
      if (!input)
        return failure();
      input = context.convertToSimpleBitVector(input);
      if (!input)
        return failure();

      // Apply not for the "not" primitive
      Value result = input;
      if (primName == "not") {
        result = moore::NotOp::create(builder, loc, input);
      }

      // Assign to all output ports (all but the last)
      for (size_t i = 0; i < portConnections.size() - 1; ++i) {
        const auto *outputExpr = unpackOutputExpr(portConnections[i]);
        auto output = context.convertLvalueExpression(*outputExpr);
        if (!output)
          return failure();

        auto refType = dyn_cast<moore::RefType>(output.getType());
        if (!refType) {
          mlir::emitError(loc) << primName << " output must be a reference type";
          return failure();
        }
        auto outputType = refType.getNestedType();

        Value assignVal = result;
        if (result.getType() != outputType) {
          assignVal = moore::ConversionOp::create(builder, loc, outputType, result);
        }
        moore::ContinuousAssignOp::create(builder, loc, output, assignVal);
      }
      return success();
    }

    // Handle three-state buffers: bufif0, bufif1, notif0, notif1
    // These have: output, input, enable
    // bufif0: output = enable ? Z : input
    // bufif1: output = enable ? input : Z
    // notif0: output = enable ? Z : ~input
    // notif1: output = enable ? ~input : Z
    if (primName == "bufif0" || primName == "bufif1" ||
        primName == "notif0" || primName == "notif1") {
      if (portConnections.size() != 3) {
        mlir::emitError(loc) << primName << " primitive expects exactly 3 ports "
                             << "(output, input, enable), got "
                             << portConnections.size();
        return failure();
      }

      // Get output, input, and enable - unpack AssignmentExpression for output
      const auto *outputExpr = unpackOutputExpr(portConnections[0]);
      auto output = context.convertLvalueExpression(*outputExpr);
      if (!output)
        return failure();

      const auto *inputExpr = portConnections[1];
      auto input = context.convertRvalueExpression(*inputExpr);
      if (!input)
        return failure();
      input = context.convertToSimpleBitVector(input);
      if (!input)
        return failure();

      const auto *enableExpr = portConnections[2];
      auto enable = context.convertRvalueExpression(*enableExpr);
      if (!enable)
        return failure();
      enable = context.convertToSimpleBitVector(enable);
      if (!enable)
        return failure();

      auto refType = dyn_cast<moore::RefType>(output.getType());
      if (!refType) {
        mlir::emitError(loc) << primName << " output must be a reference type";
        return failure();
      }
      auto outputType = refType.getNestedType();

      // Apply inversion for notif gates
      Value dataVal = input;
      if (primName == "notif0" || primName == "notif1") {
        dataVal = moore::NotOp::create(builder, loc, input);
      }

      // For simplicity, we model the tristate behavior:
      // For bufif1/notif1: when enable is high, drive data; when low, high-Z
      // For bufif0/notif0: when enable is low, drive data; when high, high-Z
      // Since we don't have a proper tristate model yet, we'll use a conditional
      // assignment. The output is driven when the enable condition is met.
      // TODO: Implement proper tristate modeling with high-Z support.
      // For now, we just assign the data value when enabled, ignoring high-Z.
      // This is a simplification that works for basic simulation.

      if (dataVal.getType() != outputType) {
        dataVal = moore::ConversionOp::create(builder, loc, outputType, dataVal);
      }
      moore::ContinuousAssignOp::create(builder, loc, output, dataVal);
      return success();
    }

    // Handle MOS switch primitives: nmos, pmos, rnmos, rpmos
    // These have: output, input, control
    // nmos/rnmos: conducts when control is high (output = control ? input : Z)
    // pmos/rpmos: conducts when control is low (output = ~control ? input : Z)
    // The 'r' prefix means resistive (weaker drive strength) - we model the
    // same as non-resistive for simulation purposes.
    if (primName == "nmos" || primName == "pmos" ||
        primName == "rnmos" || primName == "rpmos") {
      if (portConnections.size() != 3) {
        mlir::emitError(loc) << primName << " primitive expects exactly 3 ports "
                             << "(output, input, control), got "
                             << portConnections.size();
        return failure();
      }

      // Get output, input, and control
      const auto *outputExpr = unpackOutputExpr(portConnections[0]);
      auto output = context.convertLvalueExpression(*outputExpr);
      if (!output)
        return failure();

      const auto *inputExpr = portConnections[1];
      auto input = context.convertRvalueExpression(*inputExpr);
      if (!input)
        return failure();
      input = context.convertToSimpleBitVector(input);
      if (!input)
        return failure();

      const auto *controlExpr = portConnections[2];
      auto control = context.convertRvalueExpression(*controlExpr);
      if (!control)
        return failure();
      control = context.convertToSimpleBitVector(control);
      if (!control)
        return failure();

      auto refType = dyn_cast<moore::RefType>(output.getType());
      if (!refType) {
        mlir::emitError(loc) << primName << " output must be a reference type";
        return failure();
      }
      auto outputType = refType.getNestedType();

      // For simulation, we model MOS switches as pass-through when conducting.
      // nmos/rnmos conduct when control is high, pmos/rpmos when control is low.
      // TODO: Implement proper tristate/high-Z modeling.
      // For now, we just assign the input value, ignoring the control signal.
      // This is a simplification that works for basic simulation.
      Value dataVal = input;
      if (dataVal.getType() != outputType) {
        dataVal = moore::ConversionOp::create(builder, loc, outputType, dataVal);
      }
      moore::ContinuousAssignOp::create(builder, loc, output, dataVal);
      return success();
    }

    // Handle complementary MOS primitives: cmos, rcmos
    // These have: output, input, ncontrol, pcontrol
    // cmos/rcmos: combines nmos and pmos behavior
    // Conducts when ncontrol is high AND pcontrol is low
    if (primName == "cmos" || primName == "rcmos") {
      if (portConnections.size() != 4) {
        mlir::emitError(loc) << primName << " primitive expects exactly 4 ports "
                             << "(output, input, ncontrol, pcontrol), got "
                             << portConnections.size();
        return failure();
      }

      // Get output, input, ncontrol, pcontrol
      const auto *outputExpr = unpackOutputExpr(portConnections[0]);
      auto output = context.convertLvalueExpression(*outputExpr);
      if (!output)
        return failure();

      const auto *inputExpr = portConnections[1];
      auto input = context.convertRvalueExpression(*inputExpr);
      if (!input)
        return failure();
      input = context.convertToSimpleBitVector(input);
      if (!input)
        return failure();

      // ncontrol and pcontrol are read but not used in simplified model
      const auto *ncontrolExpr = portConnections[2];
      auto ncontrol = context.convertRvalueExpression(*ncontrolExpr);
      if (!ncontrol)
        return failure();

      const auto *pcontrolExpr = portConnections[3];
      auto pcontrol = context.convertRvalueExpression(*pcontrolExpr);
      if (!pcontrol)
        return failure();

      auto refType = dyn_cast<moore::RefType>(output.getType());
      if (!refType) {
        mlir::emitError(loc) << primName << " output must be a reference type";
        return failure();
      }
      auto outputType = refType.getNestedType();

      // For simulation, we model CMOS as pass-through.
      // TODO: Implement proper complementary MOS modeling with tristate.
      Value dataVal = input;
      if (dataVal.getType() != outputType) {
        dataVal = moore::ConversionOp::create(builder, loc, outputType, dataVal);
      }
      moore::ContinuousAssignOp::create(builder, loc, output, dataVal);
      return success();
    }

    // Handle bidirectional switch primitives: tran, rtran
    // These have: inout1, inout2 (bidirectional connection)
    // For simulation, we model as a simple connection between the two ports.
    if (primName == "tran" || primName == "rtran") {
      if (portConnections.size() != 2) {
        mlir::emitError(loc) << primName << " primitive expects exactly 2 ports "
                             << "(inout1, inout2), got "
                             << portConnections.size();
        return failure();
      }

      // Get both inout ports - unpack AssignmentExpression wrappers
      const auto *port1Expr = unpackOutputExpr(portConnections[0]);
      auto port1 = context.convertLvalueExpression(*port1Expr);
      if (!port1)
        return failure();

      const auto *port2Expr = unpackOutputExpr(portConnections[1]);
      auto port2 = context.convertLvalueExpression(*port2Expr);
      if (!port2)
        return failure();

      auto refType1 = dyn_cast<moore::RefType>(port1.getType());
      auto refType2 = dyn_cast<moore::RefType>(port2.getType());
      if (!refType1 || !refType2) {
        mlir::emitError(loc) << primName << " ports must be reference types";
        return failure();
      }

      // For simulation, we model bidirectional switches as continuous
      // assignments in both directions. This is a simplification.
      // TODO: Implement proper bidirectional modeling.
      auto val1 = moore::ReadOp::create(builder, loc, port1);
      auto val2 = moore::ReadOp::create(builder, loc, port2);

      // Convert types if needed
      auto type1 = refType1.getNestedType();
      auto type2 = refType2.getNestedType();

      Value assignVal1 = val2;
      Value assignVal2 = val1;
      if (val2.getType() != type1) {
        assignVal1 = moore::ConversionOp::create(builder, loc, type1, val2);
      }
      if (val1.getType() != type2) {
        assignVal2 = moore::ConversionOp::create(builder, loc, type2, val1);
      }

      moore::ContinuousAssignOp::create(builder, loc, port1, assignVal1);
      moore::ContinuousAssignOp::create(builder, loc, port2, assignVal2);
      return success();
    }

    // Handle controlled bidirectional switch primitives:
    // tranif0, tranif1, rtranif0, rtranif1
    // These have: inout1, inout2, control
    // tranif0/rtranif0: conducts when control is low
    // tranif1/rtranif1: conducts when control is high
    if (primName == "tranif0" || primName == "tranif1" ||
        primName == "rtranif0" || primName == "rtranif1") {
      if (portConnections.size() != 3) {
        mlir::emitError(loc) << primName << " primitive expects exactly 3 ports "
                             << "(inout1, inout2, control), got "
                             << portConnections.size();
        return failure();
      }

      // Get both inout ports and control
      const auto *port1Expr = unpackOutputExpr(portConnections[0]);
      auto port1 = context.convertLvalueExpression(*port1Expr);
      if (!port1)
        return failure();

      const auto *port2Expr = unpackOutputExpr(portConnections[1]);
      auto port2 = context.convertLvalueExpression(*port2Expr);
      if (!port2)
        return failure();

      const auto *controlExpr = portConnections[2];
      auto control = context.convertRvalueExpression(*controlExpr);
      if (!control)
        return failure();

      auto refType1 = dyn_cast<moore::RefType>(port1.getType());
      auto refType2 = dyn_cast<moore::RefType>(port2.getType());
      if (!refType1 || !refType2) {
        mlir::emitError(loc) << primName << " ports must be reference types";
        return failure();
      }

      // For simulation, we model controlled bidirectional switches as
      // continuous assignments in both directions, ignoring the control.
      // TODO: Implement proper conditional bidirectional modeling.
      auto val1 = moore::ReadOp::create(builder, loc, port1);
      auto val2 = moore::ReadOp::create(builder, loc, port2);

      auto type1 = refType1.getNestedType();
      auto type2 = refType2.getNestedType();

      Value assignVal1 = val2;
      Value assignVal2 = val1;
      if (val2.getType() != type1) {
        assignVal1 = moore::ConversionOp::create(builder, loc, type1, val2);
      }
      if (val1.getType() != type2) {
        assignVal2 = moore::ConversionOp::create(builder, loc, type2, val1);
      }

      moore::ContinuousAssignOp::create(builder, loc, port1, assignVal1);
      moore::ContinuousAssignOp::create(builder, loc, port2, assignVal2);
      return success();
    }

    // For other gate primitives, emit an error.
    mlir::emitError(loc) << "unsupported primitive type: " << primName;
    return failure();
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported module member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult Context::convertCompilation() {
  const auto &root = compilation.getRoot();

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = root.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  LLVM_DEBUG(llvm::dbgs() << "=== convertCompilation: traversing topInstances ===\n");
  // First only to visit the whole AST to collect the hierarchical names without
  // any operation creating.
  for (auto *inst : root.topInstances)
    if (failed(traverseInstanceBody(inst->body))) {
      LLVM_DEBUG(llvm::dbgs() << "=== convertCompilation: FAILED at traverseInstanceBody ===\n");
      return failure();
    }

  LLVM_DEBUG(llvm::dbgs() << "=== convertCompilation: visiting compilationUnits ===\n");
  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(RootVisitor(*this, loc)))) {
        LLVM_DEBUG(llvm::dbgs() << "=== convertCompilation: FAILED at RootVisitor member: "
                                << member.name << " (kind: " << slang::ast::toString(member.kind) << ")\n");
        return failure();
      }
    }
  }

  // Prime the root definition worklist by adding all the top-level modules
  // and interfaces.
  SmallVector<const slang::ast::InstanceSymbol *> topInstances;
  for (auto *inst : root.topInstances) {
    auto kind = inst->body.getDefinition().definitionKind;
    if (kind == slang::ast::DefinitionKind::Interface) {
      // Handle interfaces separately
      if (!convertInterfaceHeader(&inst->body))
        return failure();
    } else {
      // Handle modules and programs
      if (!convertModuleHeader(&inst->body))
        return failure();
    }
  }

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  // Convert all the interface bodies.
  while (!interfaceWorklist.empty()) {
    auto *iface = interfaceWorklist.front();
    interfaceWorklist.pop();
    if (failed(convertInterfaceBody(iface)))
      return failure();
  }

  // Convert the initializers of global variables.
  for (auto *var : globalVariableWorklist) {
    auto varOp = globalVariables.at(var);
    auto &block = varOp.getInitRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);
    auto value =
        convertRvalueExpression(*var->getInitializer(), varOp.getType());
    if (!value)
      return failure();
    moore::YieldOp::create(builder, varOp.getLoc(), value);
  }
  globalVariableWorklist.clear();

  return success();
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
ModuleLowering *
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  using slang::ast::ArgumentDirection;
  using slang::ast::InterfacePortSymbol;
  using slang::ast::MultiPortSymbol;
  using slang::ast::ParameterSymbol;
  using slang::ast::PortSymbol;
  using slang::ast::TypeParameterSymbol;

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  auto parameters = module->getParameters();
  bool hasModuleSame = false;
  // If there is already exist a module that has the same name with this
  // module, has the same parent scope, has the same parameters, and is not
  // targeted by instance-specific bind directives, we can define this module
  // is a duplicate module.
  //
  // Instance-specific bind directives (e.g., "bind top.inst1 monitor mon()")
  // create different bound instances in different instance bodies, so we must
  // treat them as unique modules even if they have the same definition and
  // parameters.
  for (auto const &existingModule : modules) {
    if (module->getDeclaringDefinition() ==
        existingModule.getFirst()->getDeclaringDefinition()) {
      // Check if either instance body has instance-specific binds by comparing
      // their hierarchyOverrideNode pointers. If they differ, the instances
      // have different bound modules and must be kept separate.
      if (module->hierarchyOverrideNode !=
          existingModule.getFirst()->hierarchyOverrideNode)
        continue;

      auto moduleParameters = existingModule.getFirst()->getParameters();
      hasModuleSame = true;
      for (auto it1 = parameters.begin(), it2 = moduleParameters.begin();
           it1 != parameters.end() && it2 != moduleParameters.end();
           it1++, it2++) {
        // Parameters size different
        if (it1 == parameters.end() || it2 == moduleParameters.end()) {
          hasModuleSame = false;
          break;
        }
        const auto *para1 = (*it1)->symbol.as_if<ParameterSymbol>();
        const auto *para2 = (*it2)->symbol.as_if<ParameterSymbol>();
        // Parameters kind different
        if ((para1 == nullptr) ^ (para2 == nullptr)) {
          hasModuleSame = false;
          break;
        }
        // Compare ParameterSymbol
        if (para1 != nullptr) {
          hasModuleSame = para1->getValue() == para2->getValue();
        }
        // Compare TypeParameterSymbol
        if (para1 == nullptr) {
          auto para1Type = convertType(
              (*it1)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          auto para2Type = convertType(
              (*it2)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          hasModuleSame = para1Type == para2Type;
        }
        if (!hasModuleSame)
          break;
      }
      if (hasModuleSame) {
        module = existingModule.first;
        break;
      }
    }
  }

  auto &slot = modules[module];
  if (slot)
    return slot.get();
  slot = std::make_unique<ModuleLowering>();
  auto &lowering = *slot;

  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules and programs for now. Interfaces are handled
  // separately via convertInterfaceHeader.
  auto kind = module->getDefinition().definitionKind;
  if (kind != slang::ast::DefinitionKind::Module &&
      kind != slang::ast::DefinitionKind::Program) {
    mlir::emitError(loc) << "unsupported definition: "
                         << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  auto block = std::make_unique<Block>();
  SmallVector<hw::ModulePort> modulePorts;

  // It's used to tag where a hierarchical name is on the port list.
  unsigned int outputIdx = 0, inputIdx = 0;
  for (auto *symbol : module->getPortList()) {
    auto handlePort = [&](const PortSymbol &port) {
      auto portLoc = convertLocation(port.location);
      auto type = convertType(port.getType());
      if (!type)
        return failure();
      auto portName = builder.getStringAttr(port.name);
      BlockArgument arg;
      if (port.direction == ArgumentDirection::Out) {
        modulePorts.push_back({portName, type, hw::ModulePort::Output});
        outputIdx++;
      } else {
        // Only the ref type wrapper exists for the time being, the net type
        // wrapper for inout may be introduced later if necessary.
        if (port.direction != ArgumentDirection::In)
          type = moore::RefType::get(cast<moore::UnpackedType>(type));
        modulePorts.push_back({portName, type, hw::ModulePort::Input});
        arg = block->addArgument(type, portLoc);
        inputIdx++;
      }
      lowering.ports.push_back({&port, portLoc, arg});
      return success();
    };

    auto handleInterfacePort = [&](const InterfacePortSymbol &port) {
      auto portLoc = convertLocation(port.location);
      if (!port.interfaceDef) {
        mlir::emitError(portLoc)
            << "unsupported generic interface port `" << port.name << "`";
        return failure();
      }

      auto &ifaceBody = slang::ast::InstanceBodySymbol::fromDefinition(
          compilation, *port.interfaceDef, port.location,
          slang::ast::InstanceFlags::None, nullptr, nullptr, nullptr);
      auto *ifaceLowering = convertInterfaceHeader(&ifaceBody);
      if (!ifaceLowering)
        return failure();

      auto ifaceName = ifaceLowering->op.getSymName();
      mlir::SymbolRefAttr ifaceRef;
      if (!port.modport.empty()) {
        ifaceRef = mlir::SymbolRefAttr::get(
            getContext(),
            ifaceName,
            {mlir::FlatSymbolRefAttr::get(getContext(),
                                          port.modport)});
      } else {
        ifaceRef = mlir::FlatSymbolRefAttr::get(getContext(), ifaceName);
      }

      auto vifType = moore::VirtualInterfaceType::get(getContext(), ifaceRef);
      auto portType = moore::RefType::get(vifType);
      auto portName = builder.getStringAttr(port.name);
      modulePorts.push_back({portName, portType, hw::ModulePort::Input});
      auto arg = block->addArgument(portType, portLoc);
      lowering.ports.push_back({&port, portLoc, arg});
      inputIdx++;
      return success();
    };

    if (const auto *port = symbol->as_if<PortSymbol>()) {
      if (failed(handlePort(*port)))
        return {};
    } else if (const auto *multiPort = symbol->as_if<MultiPortSymbol>()) {
      for (auto *port : multiPort->ports)
        if (failed(handlePort(*port)))
          return {};
    } else if (const auto *ifacePort = symbol->as_if<InterfacePortSymbol>()) {
      if (failed(handleInterfacePort(*ifacePort)))
        return {};
    } else {
      mlir::emitError(convertLocation(symbol->location))
          << "unsupported module port `" << symbol->name << "` ("
          << slang::ast::toString(symbol->kind) << ")";
      return {};
    }
  }

  // Mapping hierarchical names into the module's ports.
  for (auto &hierPath : hierPaths[module]) {
    auto hierType = convertType(hierPath.valueSym->getType());
    if (!hierType)
      return {};

    if (auto hierName = hierPath.hierName) {
      // The type of all hierarchical names are marked as the "RefType".
      hierType = moore::RefType::get(cast<moore::UnpackedType>(hierType));
      if (hierPath.direction == ArgumentDirection::Out) {
        hierPath.idx = outputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Output});
      } else {
        hierPath.idx = inputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Input});
        auto hierLoc = convertLocation(hierPath.valueSym->location);
        block->addArgument(hierType, hierLoc);
      }
    }
  }
  auto moduleType = hw::ModuleType::get(getContext(), modulePorts);

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.upper_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp =
      moore::SVModuleOp::create(builder, loc, module->name, moduleType);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().push_back(block.release());
  lowering.op = moduleOp;

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);

  // Map duplicate port by Syntax
  for (const auto &port : lowering.ports)
    if (auto *syntax = port.symbol->getSyntax())
      lowering.portsBySyntaxNode.insert({syntax, port.symbol});

  return &lowering;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  hierValuePlaceholders.clear();
  auto &lowering = *modules[module];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(lowering.op.getBody());

  auto prevInterfaceInstances = std::move(interfaceInstances);
  interfaceInstances.clear();
  auto interfaceInstancesGuard = llvm::make_scope_exit(
      [&] { interfaceInstances = std::move(prevInterfaceInstances); });
  auto prevPendingInterfacePortConnections =
      std::move(pendingInterfacePortConnections);
  pendingInterfacePortConnections.clear();
  auto pendingInterfacePortConnectionsGuard = llvm::make_scope_exit([&] {
    pendingInterfacePortConnections =
        std::move(prevPendingInterfacePortConnections);
  });

  ValueSymbolScope scope(valueSymbols);
  SmallVector<const slang::ast::InterfacePortSymbol *> ifacePortSymbols;
  auto ifacePortGuard = llvm::make_scope_exit([&] {
    for (auto *ifacePort : ifacePortSymbols)
      interfacePortValues.erase(ifacePort);
  });

  for (auto &port : lowering.ports) {
    if (auto *ifacePort =
            port.symbol->as_if<slang::ast::InterfacePortSymbol>()) {
      interfacePortValues[ifacePort] = port.arg;
      ifacePortSymbols.push_back(ifacePort);
    }
  }

  // Keep track of the current scope for %m format specifier.
  auto prevScope = currentScope;
  currentScope = module;
  auto scopeGuard = llvm::make_scope_exit([&] { currentScope = prevScope; });

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Collect downward hierarchical names. Such as,
  // module SubA; int x = Top.y; endmodule. The "Top" module is the parent of
  // the "SubA", so "Top.y" is the downward hierarchical name.
  for (auto &hierPath : hierPaths[module])
    if (hierPath.direction == slang::ast::ArgumentDirection::In && hierPath.idx)
      valueSymbols.insert(hierPath.valueSym,
                          lowering.op.getBody()->getArgument(*hierPath.idx));

  // First pass: collect instance members and process interface instances early.
  // Interface instances need to be registered before procedural blocks that
  // might reference them (e.g., in virtual interface assignments).
  SmallVector<const slang::ast::Symbol *> instanceMembers;
  SmallVector<const slang::ast::Symbol *> preInstanceMembers;
  SmallVector<const slang::ast::Symbol *> postInstanceMembers;
  for (auto &member : module->members()) {
    if (member.kind == slang::ast::SymbolKind::Instance) {
      auto &instNode = member.as<slang::ast::InstanceSymbol>();
      auto kind = instNode.body.getDefinition().definitionKind;
      if (kind == slang::ast::DefinitionKind::Interface) {
        // Process interface instances immediately so they're available
        // for virtual interface assignments in procedural blocks.
        auto loc = convertLocation(member.location);
        if (failed(member.visit(ModuleVisitor(*this, loc))))
          return failure();
      } else {
        instanceMembers.push_back(&member);
      }
      continue;
    }
    if (member.kind == slang::ast::SymbolKind::ContinuousAssign ||
        member.kind == slang::ast::SymbolKind::ProceduralBlock) {
      postInstanceMembers.push_back(&member);
    } else {
      preInstanceMembers.push_back(&member);
    }
  }

  // Second pass: convert non-instance declarations to populate value symbols
  // used in instance connections.
  for (auto *member : preInstanceMembers) {
    auto memberLoc = convertLocation(member->location);
    if (failed(member->visit(ModuleVisitor(*this, memberLoc))))
      return failure();
  }

  auto canConvertInstance = [&](const slang::ast::InstanceSymbol &instNode) {
    for (const auto &hierPath : hierPaths[&instNode.body]) {
      if (hierPath.direction != slang::ast::ArgumentDirection::In)
        continue;
      if (!valueSymbols.lookup(hierPath.valueSym)) {
        auto placeholder = getOrCreateHierarchicalPlaceholder(
            hierPath.valueSym, convertLocation(instNode.location));
        if (!placeholder)
          return false;
        valueSymbols.insert(hierPath.valueSym, placeholder);
      }
    }
    return true;
  };

  SmallVector<const slang::ast::Symbol *> pending = instanceMembers;
  bool progress = true;
  while (progress && !pending.empty()) {
    progress = false;
    for (size_t i = 0; i < pending.size();) {
      auto *member = pending[i];
      auto &instNode = member->as<slang::ast::InstanceSymbol>();
      if (!canConvertInstance(instNode)) {
        ++i;
        continue;
      }
      auto loc = convertLocation(member->location);
      if (failed(member->visit(ModuleVisitor(*this, loc))))
        return failure();
      pending.erase(pending.begin() + i);
      progress = true;
    }
  }

  if (!pending.empty()) {
    auto loc = convertLocation(pending.front()->location);
    auto &instNode = pending.front()->as<slang::ast::InstanceSymbol>();
    for (const auto &hierPath : hierPaths[&instNode.body]) {
      if (hierPath.direction != slang::ast::ArgumentDirection::In)
        continue;
      if (!valueSymbols.lookup(hierPath.valueSym)) {
        mlir::emitError(loc)
            << "missing hierarchical value for `"
            << hierPath.hierName.getValue() << "`";
        break;
      }
    }
    return failure();
  }

  for (const auto &pendingConn : pendingInterfacePortConnections) {
    if (failed(emitInterfacePortConnections(*this, pendingConn.loc,
                                            *pendingConn.instSym,
                                            pendingConn.instRef)))
      return failure();
  }
  pendingInterfacePortConnections.clear();

  // Final pass: convert procedural blocks and continuous assigns after
  // instances so hierarchical outputs are available.
  for (auto *member : postInstanceMembers) {
    auto memberLoc = convertLocation(member->location);
    if (failed(member->visit(ModuleVisitor(*this, memberLoc))))
      return failure();
  }

  // Create additional ops to drive input port values onto the corresponding
  // internal variables and nets, and to collect output port values for the
  // terminator.
  SmallVector<Value> outputs;
  for (auto &port : lowering.ports) {
    if (auto *ifacePort =
            port.symbol->as_if<slang::ast::InterfacePortSymbol>()) {
      auto [ifaceConn, modportSym] = ifacePort->getConnection();
      (void)modportSym;
      if (auto *instSym =
              ifaceConn ? ifaceConn->as_if<slang::ast::InstanceSymbol>()
                        : nullptr)
        interfaceInstances[instSym] = port.arg;
      continue;
    }

    auto *portSym = port.symbol->as_if<slang::ast::PortSymbol>();
    if (!portSym)
      return mlir::emitError(port.loc, "unsupported port: `")
             << port.symbol->name << "` has no PortSymbol mapping";
    Value value;
    if (auto *expr = portSym->getInternalExpr()) {
      value = convertLvalueExpression(*expr);
    } else if (portSym->internalSymbol) {
      if (const auto *sym =
              portSym->internalSymbol->as_if<slang::ast::ValueSymbol>())
        value = valueSymbols.lookup(sym);
    }
    if (!value)
      return mlir::emitError(port.loc, "unsupported port: `")
             << portSym->name
             << "` does not map to an internal symbol or expression";

    // Collect output port values to be returned in the terminator.
    if (portSym->direction == slang::ast::ArgumentDirection::Out) {
      if (isa<moore::RefType>(value.getType()))
        value = moore::ReadOp::create(builder, value.getLoc(), value);
      outputs.push_back(value);
      continue;
    }

    // Assign the value coming in through the port to the internal net or symbol
    // of that port.
    Value portArg = port.arg;
    if (portSym->direction != slang::ast::ArgumentDirection::In)
      portArg = moore::ReadOp::create(builder, port.loc, port.arg);
    moore::ContinuousAssignOp::create(builder, port.loc, value, portArg);
  }

  // Ensure the number of operands of this module's terminator and the number of
  // its(the current module) output ports remain consistent.
  for (auto &hierPath : hierPaths[module])
    if (auto hierValue = valueSymbols.lookup(hierPath.valueSym))
      if (hierPath.direction == slang::ast::ArgumentDirection::Out)
        outputs.push_back(hierValue);

  moore::OutputOp::create(builder, lowering.op.getLoc(), outputs);
  if (!hierValuePlaceholders.empty()) {
    auto it = hierValuePlaceholders.begin();
    mlir::emitError(lowering.op.getLoc())
        << "missing hierarchical value for `" << it->first->name << "`";
    return failure();
  }
  return success();
}

Value Context::getOrCreateHierarchicalPlaceholder(
    const slang::ast::ValueSymbol *sym, Location loc) {
  if (auto it = hierValuePlaceholders.find(sym);
      it != hierValuePlaceholders.end())
    return it->second;

  auto loweredType = convertType(sym->getType());
  if (!loweredType)
    return {};
  auto refTy = moore::RefType::get(cast<moore::UnpackedType>(loweredType));
  auto placeholderName = builder.getStringAttr(
      Twine("__hier_placeholder") + Twine(nextHierPlaceholderId++));

  Value placeholder;
  if (sym->kind == slang::ast::SymbolKind::Net) {
    placeholder = moore::NetOp::create(builder, loc, refTy, placeholderName,
                                       moore::NetKind::Wire, Value{});
  } else {
    placeholder = moore::VariableOp::create(builder, loc, refTy,
                                            placeholderName, Value{});
  }

  hierValuePlaceholders.insert({sym, placeholder});
  return placeholder;
}

/// Convert a package and its contents.
LogicalResult
Context::convertPackage(const slang::ast::PackageSymbol &package) {
  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = package.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(intoModuleOp.getBody());
  ValueSymbolScope scope(valueSymbols);

  // Two-pass conversion:
  // Pass 1: Convert global variables first so they're available for classes.
  // Note: Some variables may not appear in members() due to how Slang handles
  // symbols declared after forward typedefs. Those are handled via on-demand
  // conversion in Expressions.cpp when they're first referenced.
  LLVM_DEBUG(llvm::dbgs() << "=== convertPackage Pass 1: Variables ===\n");
  for (auto &member : package.members()) {
    if (member.kind == slang::ast::SymbolKind::Variable) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(PackageVisitor(*this, loc)))) {
        LLVM_DEBUG(llvm::dbgs() << "=== convertPackage FAILED at Variable: " << member.name << "\n");
        return failure();
      }
    }
  }

  // Pass 2: Convert remaining members (classes, functions, etc.)
  LLVM_DEBUG(llvm::dbgs() << "=== convertPackage Pass 2: Non-variables ===\n");
  for (auto &member : package.members()) {
    if (member.kind != slang::ast::SymbolKind::Variable) {
      auto loc = convertLocation(member.location);
      LLVM_DEBUG(llvm::dbgs() << "  Processing member: " << member.name << " (kind: " << slang::ast::toString(member.kind) << ")\n");
      if (failed(member.visit(PackageVisitor(*this, loc)))) {
        LLVM_DEBUG(llvm::dbgs() << "=== convertPackage FAILED at member: " << member.name << " (kind: " << slang::ast::toString(member.kind) << ")\n");
        return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interface Conversion
//===----------------------------------------------------------------------===//

/// Convert an interface declaration header. Creates the moore.interface op
/// and schedules the body for conversion.
///
/// This function deduplicates interface declarations: if an interface with the
/// same DefinitionSymbol has already been created (e.g., from a different
/// virtual interface variable referencing the same interface type), we reuse
/// that existing declaration rather than creating a duplicate.
InterfaceLowering *
Context::convertInterfaceHeader(const slang::ast::InstanceBodySymbol *iface) {
  // Check if we've already processed this specific instance body.
  auto &slot = interfaces[iface];
  if (slot)
    return slot.get();

  // Check if an interface with the same definition has already been created.
  // This handles the case where multiple virtual interface variables reference
  // the same interface type, which would otherwise create duplicate interface
  // declarations with mangled names (e.g., @my_if, @my_if_0, @my_if_1).
  const auto &definition = iface->getDefinition();
  auto defIt = interfacesByDefinition.find(&definition);
  if (defIt != interfacesByDefinition.end()) {
    // Reuse the existing interface lowering. We don't need to create a new
    // unique_ptr; instead, we create a new one that points to the same op.
    slot = std::make_unique<InterfaceLowering>();
    slot->op = defIt->second->op;
    return slot.get();
  }

  // Create a new interface lowering.
  slot = std::make_unique<InterfaceLowering>();
  auto &lowering = *slot;

  auto loc = convertLocation(iface->location);
  OpBuilder::InsertionGuard g(builder);

  // Pick an insertion point according to the source file location.
  auto it = orderedRootOps.upper_bound(iface->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create the interface op
  auto ifaceOp = moore::InterfaceDeclOp::create(
      builder, loc, definition.name);
  orderedRootOps.insert(it, {iface->location, ifaceOp});

  // Create the body block for the interface
  ifaceOp.getBody().emplaceBlock();

  lowering.op = ifaceOp;

  // Register in the definition-based map for deduplication.
  interfacesByDefinition[&definition] = &lowering;

  // Add the interface to the symbol table
  symbolTable.insert(ifaceOp);

  // Schedule the body to be lowered
  interfaceWorklist.push(iface);

  return &lowering;
}

/// Convert an interface body - signals, modports, ports, and procedural code.
LogicalResult
Context::convertInterfaceBody(const slang::ast::InstanceBodySymbol *iface) {
  auto &lowering = *interfaces[iface];

  // Check if the body has already been converted to avoid duplicate conversion.
  if (lowering.bodyConverted)
    return success();
  lowering.bodyConverted = true;

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&lowering.op.getBody().front());

  // Set up interface context for signal access tracking.
  // Save and restore the previous interface context.
  auto prevInterfaceBody = currentInterfaceBody;
  currentInterfaceBody = iface;
  auto interfaceGuard =
      llvm::make_scope_exit([&] { currentInterfaceBody = prevInterfaceBody; });

  // Clear and restore interface signal names map.
  auto prevSignalNames = std::move(interfaceSignalNames);
  interfaceSignalNames.clear();
  auto signalNamesGuard =
      llvm::make_scope_exit([&] { interfaceSignalNames = std::move(prevSignalNames); });

  // Track internal symbols from interface ports so we can skip them when
  // iterating through members (they would otherwise appear as duplicate
  // variables/nets).
  llvm::DenseSet<const slang::ast::Symbol *> portInternalSymbols;

  // Handle interface ports as signals.
  // Interface ports (like `interface foo(input clk, input reset)`) are
  // essentially external signals that get connected when the interface is
  // instantiated.
  for (auto *symbol : iface->getPortList()) {
    if (const auto *port = symbol->as_if<slang::ast::PortSymbol>()) {
      auto portLoc = convertLocation(port->location);
      auto type = convertType(port->getType());
      if (!type)
        return failure();
      moore::InterfaceSignalDeclOp::create(builder, portLoc, port->name, type,
                                           mlir::UnitAttr());
      // Track the internal symbol so we skip it in member iteration
      if (port->internalSymbol) {
        portInternalSymbols.insert(port->internalSymbol);
        // Map the internal symbol to the signal name for rvalue lookup
        interfaceSignalNames[port->internalSymbol] = port->name;
      }
      // Also map the port symbol itself
      interfaceSignalNames[port] = port->name;
    } else if (const auto *multiPort =
                   symbol->as_if<slang::ast::MultiPortSymbol>()) {
      for (auto *port : multiPort->ports) {
        auto portLoc = convertLocation(port->location);
        auto type = convertType(port->getType());
        if (!type)
          return failure();
        moore::InterfaceSignalDeclOp::create(builder, portLoc, port->name, type,
                                             mlir::UnitAttr());
        if (port->internalSymbol) {
          portInternalSymbols.insert(port->internalSymbol);
          interfaceSignalNames[port->internalSymbol] = port->name;
        }
        interfaceSignalNames[port] = port->name;
      }
    }
  }

  // First pass: Convert signal declarations (variables/nets) and modports.
  // We need to build the signal name map before converting procedural code.
  for (auto &member : iface->members()) {
    auto loc = convertLocation(member.location);

    // Skip ports - they are already handled above through getPortList()
    if (member.as_if<slang::ast::PortSymbol>() ||
        member.as_if<slang::ast::MultiPortSymbol>())
      continue;

    // Skip internal symbols that correspond to interface ports
    if (portInternalSymbols.count(&member))
      continue;

    // Handle variables/nets as interface signals
    if (auto *var = member.as_if<slang::ast::VariableSymbol>()) {
      auto type = convertType(var->getType());
      if (!type)
        return failure();
      moore::InterfaceSignalDeclOp::create(builder, loc, var->name, type,
                                           mlir::UnitAttr());
      // Map the variable symbol to its signal name
      interfaceSignalNames[var] = var->name;
      continue;
    }

    if (auto *net = member.as_if<slang::ast::NetSymbol>()) {
      auto type = convertType(net->getType());
      if (!type)
        return failure();
      moore::InterfaceSignalDeclOp::create(builder, loc, net->name, type,
                                           mlir::UnitAttr());
      // Map the net symbol to its signal name
      interfaceSignalNames[net] = net->name;
      continue;
    }

    // Handle nested interface instances by modeling them as interface signals
    // that carry a virtual interface handle.
    if (auto *inst = member.as_if<slang::ast::InstanceSymbol>()) {
      if (inst->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        auto *ifaceLowering = convertInterfaceHeader(&inst->body);
        if (!ifaceLowering)
          return failure();

        auto ifaceRef = mlir::FlatSymbolRefAttr::get(
            getContext(), ifaceLowering->op.getSymName());
        auto vifType =
            moore::VirtualInterfaceType::get(getContext(), ifaceRef);

        moore::InterfaceSignalDeclOp::create(
            builder, loc, inst->name, vifType, builder.getUnitAttr());
        interfaceSignalNames[inst] = inst->name;
        continue;
      }
    }

    // Handle modports
    if (auto *modport = member.as_if<slang::ast::ModportSymbol>()) {
      SmallVector<mlir::Attribute> ports;

      for (auto &portMember : modport->members()) {
        if (auto *port =
                portMember.as_if<slang::ast::ModportPortSymbol>()) {
          // Map slang direction to Moore ModportDir
          moore::ModportDir dir;
          switch (port->direction) {
          case slang::ast::ArgumentDirection::In:
            dir = moore::ModportDir::Input;
            break;
          case slang::ast::ArgumentDirection::Out:
            dir = moore::ModportDir::Output;
            break;
          case slang::ast::ArgumentDirection::InOut:
            dir = moore::ModportDir::InOut;
            break;
          case slang::ast::ArgumentDirection::Ref:
            dir = moore::ModportDir::Ref;
            break;
          default:
            dir = moore::ModportDir::InOut;
            break;
          }

          // Get the signal name from the internal symbol
          StringRef signalName;
          if (port->internalSymbol) {
            signalName = port->internalSymbol->name;
          } else {
            signalName = port->name;
          }

          auto dirAttr = moore::ModportDirAttr::get(getContext(), dir);
          auto signalRef =
              mlir::FlatSymbolRefAttr::get(getContext(), signalName);
          ports.push_back(
              moore::ModportPortAttr::get(getContext(), dirAttr, signalRef));
        }
      }

      moore::ModportDeclOp::create(builder, loc, modport->name,
                                   builder.getArrayAttr(ports));
      continue;
    }
  }

  // Second pass: Convert procedural code (tasks, functions).
  // These need the signal name map to be fully built first.
  for (auto &member : iface->members()) {
    // Handle tasks and functions
    if (auto *subroutine = member.as_if<slang::ast::SubroutineSymbol>()) {
      if (failed(convertFunction(*subroutine)))
        return failure();
      continue;
    }

    // Note: always blocks and other procedural code in interfaces would need
    // additional support. For now, we focus on tasks/functions which are the
    // primary use case for BFMs.
  }

  return success();
}

Value Context::resolveInterfaceInstance(
    const slang::ast::InstanceSymbol *instSym, Location loc) {
  if (!instSym)
    return {};

  auto findArrayAlias = [&]() -> Value {
    Value match;
    auto *targetParent = getInstanceBodyParent(*instSym);
    for (const auto &entry : interfaceInstances) {
      const auto *candidate = entry.first;
      if (candidate->name != instSym->name)
        continue;
      if (&candidate->getDefinition() != &instSym->getDefinition())
        continue;
      if (!matchInterfaceArrayIndices(*candidate, *instSym))
        continue;
      if (targetParent &&
          getInstanceBodyParent(*candidate) != targetParent)
        continue;
      if (match)
        return {};
      match = entry.second;
    }
    return match;
  };

  if (shouldCacheInterfaceInstance(*instSym)) {
    if (auto it = interfaceInstances.find(instSym);
        it != interfaceInstances.end()) {
      if (!currentScope)
        return it->second;
      if (auto *body = getInstanceBodyParent(*instSym)) {
        if (body == currentScope)
          return it->second;
      }
    }
    if (auto alias = findArrayAlias()) {
      if (!currentScope)
        return alias;
      if (auto *body = getInstanceBodyParent(*instSym)) {
        if (body == currentScope)
          return alias;
      }
    }
  }

  auto findPortBase =
      [&](const slang::ast::InstanceSymbol *target) -> Value {
    Value candidate;
    bool ambiguous = false;
    const auto &targetDef = target->getDefinition();
    for (const auto &entry : interfacePortValues) {
      auto *ifacePort = entry.first;
      auto *scope = ifacePort->getParentScope();
      if (!scope)
        continue;
      auto *body =
          scope->asSymbol().as_if<slang::ast::InstanceBodySymbol>();
      if (body && body->parentInstance) {
        auto [ifaceConn, modport] = ifacePort->getConnection();
        (void)modport;
        if (ifaceConn == target)
          return entry.second;
        continue;
      }
      if (!ifacePort->isGeneric && ifacePort->interfaceDef == &targetDef) {
        if (candidate) {
          ambiguous = true;
          continue;
        }
        candidate = entry.second;
      }
    }
    if (ambiguous)
      return {};
    return candidate;
  };

  auto findScopedInstance =
      [&](const slang::ast::InstanceSymbol *target) -> Value {
    if (!currentScope)
      return {};
    if (auto *sym = currentScope->find(target->name)) {
      if (auto *inst = sym->as_if<slang::ast::InstanceSymbol>()) {
        if (&inst->getDefinition() == &target->getDefinition() &&
            matchInterfaceArrayIndices(*inst, *target)) {
          if (auto it = interfaceInstances.find(inst);
              it != interfaceInstances.end())
            return it->second;
        }
      }
    }
    for (const auto &entry : interfaceInstances) {
      const auto *candidate = entry.first;
      if (candidate->name != target->name)
        continue;
      if (&candidate->getDefinition() != &target->getDefinition())
        continue;
      if (!matchInterfaceArrayIndices(*candidate, *target))
        continue;
      auto *body = getInstanceBodyParent(*candidate);
      if (body != currentScope)
        continue;
      return entry.second;
    }
    // Also search in parent scopes - interface instances defined in an
    // enclosing module should be visible from within generate blocks.
    for (const auto &entry : interfaceInstances) {
      const auto *candidate = entry.first;
      if (candidate->name != target->name)
        continue;
      if (&candidate->getDefinition() != &target->getDefinition())
        continue;
      if (!matchInterfaceArrayIndices(*candidate, *target))
        continue;
      // Check if the candidate is in an ancestor scope of currentScope.
      auto *candidateBody = getInstanceBodyParent(*candidate);
      if (!candidateBody)
        continue;
      // Walk up from currentScope looking for candidateBody.
      const slang::ast::Scope *walkScope = currentScope;
      while (walkScope) {
        auto *walkBody =
            walkScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>();
        if (walkBody == candidateBody)
          return entry.second;
        if (walkBody && walkBody->parentInstance) {
          walkScope = walkBody->parentInstance->getParentScope();
          continue;
        }
        break;
      }
    }
    return {};
  };

  SmallVector<const slang::ast::InstanceSymbol *, 4> chain;
  llvm::SmallPtrSet<const slang::ast::InstanceSymbol *, 8> visited;

  const slang::ast::InstanceSymbol *cursor = instSym;
  while (cursor) {
    if (!visited.insert(cursor).second)
      return {};
    chain.push_back(cursor);
    cursor = cursor->body.parentInstance;
  }

  int baseIndex = -1;
  Value currentRef;

  for (size_t i = 0; i < chain.size(); ++i) {
    if (auto base = findPortBase(chain[i])) {
      baseIndex = static_cast<int>(i);
      currentRef = base;
      break;
    }
  }

  if (!currentRef) {
    for (size_t i = 0; i < chain.size(); ++i) {
      if (!shouldCacheInterfaceInstance(*chain[i]))
        continue;
      if (auto it = interfaceInstances.find(chain[i]);
          it != interfaceInstances.end()) {
        baseIndex = static_cast<int>(i);
        currentRef = it->second;
        break;
      }
      if (auto scoped = findScopedInstance(chain[i])) {
        baseIndex = static_cast<int>(i);
        currentRef = scoped;
        break;
      }
    }
  }

  if (!currentRef)
    return {};

  // Walk from the resolved base down to the requested instance.
  for (int idx = baseIndex - 1; idx >= 0; --idx) {
    const auto *childInst = chain[static_cast<size_t>(idx)];

    auto parentRefTy = dyn_cast<moore::RefType>(currentRef.getType());
    if (!parentRefTy)
      return {};

    auto parentVifTy =
        dyn_cast<moore::VirtualInterfaceType>(parentRefTy.getNestedType());
    if (!parentVifTy)
      return {};

    Value parentVif =
        moore::ConversionOp::create(builder, loc, parentVifTy, currentRef);

    auto *ifaceLowering = convertInterfaceHeader(&childInst->body);
    if (!ifaceLowering)
      return {};

    auto ifaceRef = mlir::FlatSymbolRefAttr::get(
        getContext(), ifaceLowering->op.getSymName());
    auto childVifTy =
        moore::VirtualInterfaceType::get(getContext(), ifaceRef);
    auto childRefTy = moore::RefType::get(childVifTy);
    auto signalSym =
        mlir::FlatSymbolRefAttr::get(getContext(), childInst->name);

    currentRef = moore::VirtualInterfaceSignalRefOp::create(
        builder, loc, childRefTy, parentVif, signalSym);
    if (shouldCacheInterfaceInstance(*childInst))
      interfaceInstances[childInst] = currentRef;
  }

  return currentRef;
}

Value Context::resolveInterfaceInstance(
    const slang::ast::HierarchicalReference &ref, Location loc) {
  Value currentRef;

  for (const auto &elem : ref.path) {
    if (!currentRef) {
      if (auto *instSym =
              elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
        currentRef = resolveInterfaceInstance(instSym, loc);
        if (currentRef)
          continue;
      }
      if (auto *ifacePort =
              elem.symbol->as_if<slang::ast::InterfacePortSymbol>()) {
        if (auto it = interfacePortValues.find(ifacePort);
            it != interfacePortValues.end())
          currentRef = it->second;
        continue;
      }
      continue;
    }

    auto *childInst = elem.symbol->as_if<slang::ast::InstanceSymbol>();
    if (!childInst ||
        childInst->getDefinition().definitionKind !=
            slang::ast::DefinitionKind::Interface)
      continue;

    auto parentRefTy = dyn_cast<moore::RefType>(currentRef.getType());
    if (!parentRefTy)
      return {};
    auto parentVifTy =
        dyn_cast<moore::VirtualInterfaceType>(parentRefTy.getNestedType());
    if (!parentVifTy)
      return {};

    Value parentVif =
        moore::ConversionOp::create(builder, loc, parentVifTy, currentRef);

    auto *ifaceLowering = convertInterfaceHeader(&childInst->body);
    if (!ifaceLowering)
      return {};

    auto ifaceRef = mlir::FlatSymbolRefAttr::get(
        getContext(), ifaceLowering->op.getSymName());
    auto childVifTy = moore::VirtualInterfaceType::get(getContext(), ifaceRef);
    auto childRefTy = moore::RefType::get(childVifTy);
    auto signalSym =
        mlir::FlatSymbolRefAttr::get(getContext(), childInst->name);

    currentRef = moore::VirtualInterfaceSignalRefOp::create(
        builder, loc, childRefTy, parentVif, signalSym);
    if (shouldCacheInterfaceInstance(*childInst))
      interfaceInstances[childInst] = currentRef;
  }

  return currentRef;
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareFunction(const slang::ast::SubroutineSymbol &subroutine) {
  // Check if there already is a declaration for this function.
  auto it = functions.find(&subroutine);
  if (it != functions.end() && it->second) {
    if (!it->second->op)
      return {};
    return it->second.get();
  }

  // Check if this is a task/function inside an interface.
  // Interface methods need an implicit first argument for the interface instance.
  const auto &parentSym = subroutine.getParentScope()->asSymbol();
  if (parentSym.kind == slang::ast::SymbolKind::InstanceBody) {
    if (auto *instBody = parentSym.as_if<slang::ast::InstanceBodySymbol>()) {
      if (instBody->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        // This is a task/function inside an interface.
        // Build qualified name: @"InterfaceName"::subroutine
        SmallString<64> qualName;
        qualName += instBody->getDefinition().name;
        qualName += "::";
        qualName += subroutine.name;

        // Add implicit interface argument: %iface : !moore.virtual_interface<@I>
        SmallVector<Type, 1> extraParams;
        auto ifaceSym = mlir::FlatSymbolRefAttr::get(
            getContext(), instBody->getDefinition().name);
        auto vifType = moore::VirtualInterfaceType::get(getContext(), ifaceSym);
        extraParams.push_back(vifType);

        return declareCallableImpl(subroutine, qualName, extraParams);
      }
    }
  }

  if (!subroutine.thisVar) {
    SmallString<64> name;

    // DPI-C imports should use just the function name (for C linkage),
    // not a namespace-prefixed name.
    if (subroutine.flags & slang::ast::MethodFlags::DPIImport) {
      name = subroutine.name;
    } else {
      guessNamespacePrefix(subroutine.getParentScope()->asSymbol(), name);
      name += subroutine.name;
    }

    SmallVector<Type, 1> noThis = {};
    return declareCallableImpl(subroutine, name, noThis);
  }

  auto loc = convertLocation(subroutine.location);

  // Extract 'this' type and ensure it's a class.
  // Use getCanonicalType() to unwrap type aliases so we get the same ClassType
  // pointer that's used in the 'classes' map.
  const slang::ast::Type &thisTy = subroutine.thisVar->getType().getCanonicalType();
  moore::ClassDeclOp ownerDecl;

  if (auto *classTy = thisTy.as_if<slang::ast::ClassType>()) {
    // If the class is not yet in the map, convert it now. This handles cases
    // where a class method references its own class type (self-reference) before
    // the class declaration is fully processed.
    // Note: We call convertClassDeclaration (not just declareClass) to ensure
    // the class body is populated with methods. This is important for
    // parameterized class specializations where the specialization might be
    // encountered via a method's 'this' type before being explicitly converted.
    // Failures are tolerated (e.g., for forward references or recursive calls
    // that will be resolved later), but the class declaration must exist.
    (void)convertClassDeclaration(*classTy);

    // Re-check if this function was already created during convertClassDeclaration.
    // This can happen when converting a class method triggers class body conversion,
    // which in turn converts all class members including this same method.
    auto recheck = functions.find(&subroutine);
    if (recheck != functions.end() && recheck->second && recheck->second->op)
      return recheck->second.get();

    auto *lowering = declareClass(*classTy);
    if (!lowering || !lowering->op) {
      mlir::emitError(loc) << "class '" << classTy->name
                           << "' has not been lowered yet";
      return {};
    }
    ownerDecl = lowering->op;
  } else {
    mlir::emitError(loc) << "expected 'this' to be a class type, got "
                         << thisTy.toString();
    return {};
  }

  // Build qualified name: @"Pkg::Class"::subroutine
  SmallString<64> qualName;
  qualName += ownerDecl.getSymName(); // already qualified
  qualName += "::";
  qualName += subroutine.name;

  // %this : class<@C>
  SmallVector<Type, 1> extraParams;
  {
    auto classSym = mlir::FlatSymbolRefAttr::get(ownerDecl.getSymNameAttr());
    auto handleTy = moore::ClassHandleType::get(getContext(), classSym);
    extraParams.push_back(handleTy);
  }

  auto *fLowering = declareCallableImpl(subroutine, qualName, extraParams);
  return fLowering;
}

/// Helper function to generate the function signature from a SubroutineSymbol
/// and optional extra arguments (used for %this argument)
static FunctionType
getFunctionSignature(Context &context,
                     const slang::ast::SubroutineSymbol &subroutine,
                     llvm::SmallVectorImpl<Type> &extraParams) {
  using slang::ast::ArgumentDirection;

  SmallVector<Type> inputTypes;
  inputTypes.append(extraParams.begin(), extraParams.end());
  SmallVector<Type, 1> outputTypes;

  for (const auto *arg : subroutine.getArguments()) {
    auto type = context.convertType(arg->getType());
    if (!type)
      return {};
    if (arg->direction == ArgumentDirection::In) {
      inputTypes.push_back(type);
    } else {
      auto unpackedType = dyn_cast<moore::UnpackedType>(type);
      if (!unpackedType) {
        mlir::emitError(context.convertLocation(arg->location))
            << "argument type " << type << " is not an unpacked type";
        return {};
      }
      inputTypes.push_back(moore::RefType::get(unpackedType));
    }
  }

  const auto &returnType = subroutine.getReturnType();
  if (!returnType.isVoid()) {
    auto type = context.convertType(returnType);
    if (!type)
      return {};
    outputTypes.push_back(type);
  }

  auto funcType =
      FunctionType::get(context.getContext(), inputTypes, outputTypes);

  // Create a function declaration.
  return funcType;
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareCallableImpl(const slang::ast::SubroutineSymbol &subroutine,
                             mlir::StringRef qualifiedName,
                             llvm::SmallVectorImpl<Type> &extraParams) {
  auto loc = convertLocation(subroutine.location);
  std::unique_ptr<FunctionLowering> lowering =
      std::make_unique<FunctionLowering>();

  // Pick an insertion point for this function according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(subroutine.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  auto funcTy = getFunctionSignature(*this, subroutine, extraParams);
  if (!funcTy)
    return nullptr;

  // Re-check if the function was already created during getFunctionSignature.
  // This can happen when converting argument types triggers class conversions
  // that recursively lead to declaring this same function.
  auto recheckIt = functions.find(&subroutine);
  if (recheckIt != functions.end() && recheckIt->second &&
      recheckIt->second->op)
    return recheckIt->second.get();

  auto funcOp = mlir::func::FuncOp::create(builder, loc, qualifiedName, funcTy);

  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  orderedRootOps.insert(it, {subroutine.location, funcOp});
  lowering->op = funcOp;

  // Add the function to the symbol table of the MLIR module, which uniquifies
  // its name.
  symbolTable.insert(funcOp);
  functions[&subroutine] = std::move(lowering);

  return functions[&subroutine].get();
}

/// Special case handling for recursive functions with captures;
/// this function fixes the in-body call of the recursive function with
/// the captured arguments.
static LogicalResult rewriteCallSitesToPassCaptures(mlir::func::FuncOp callee,
                                                    ArrayRef<Value> captures) {
  if (captures.empty())
    return success();

  mlir::ModuleOp module = callee->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return callee.emitError("expected callee to be nested under ModuleOp");

  auto usesOpt = mlir::SymbolTable::getSymbolUses(callee, module);
  if (!usesOpt)
    return callee.emitError("failed to compute symbol uses");

  // Snapshot the relevant users before we mutate IR.
  SmallVector<mlir::func::CallOp, 8> callSites;
  callSites.reserve(std::distance(usesOpt->begin(), usesOpt->end()));
  for (const mlir::SymbolTable::SymbolUse &use : *usesOpt) {
    if (auto call = llvm::dyn_cast<mlir::func::CallOp>(use.getUser()))
      callSites.push_back(call);
  }
  if (callSites.empty())
    return success();

  Block &entry = callee.getBody().front();
  const unsigned numCaps = captures.size();
  const unsigned numEntryArgs = entry.getNumArguments();
  if (numEntryArgs < numCaps)
    return callee.emitError("entry block has fewer args than captures");
  const unsigned capArgStart = numEntryArgs - numCaps;

  // Current (finalized) function type.
  auto fTy = callee.getFunctionType();

  for (auto call : callSites) {
    SmallVector<Value> newOperands(call.getArgOperands().begin(),
                                   call.getArgOperands().end());

    const bool inSameFunc = callee->isProperAncestor(call);
    if (inSameFunc) {
      // Append the function’s *capture block arguments* in order.
      for (unsigned i = 0; i < numCaps; ++i)
        newOperands.push_back(entry.getArgument(capArgStart + i));
    } else {
      // External call site: pass the captured SSA values.
      newOperands.append(captures.begin(), captures.end());
    }

    OpBuilder b(call);
    auto flatRef = mlir::FlatSymbolRefAttr::get(callee);
    auto newCall = mlir::func::CallOp::create(
        b, call.getLoc(), fTy.getResults(), flatRef, newOperands);
    call->replaceAllUsesWith(newCall.getOperation());
    call->erase();
  }

  return success();
}

/// Convert a function.
LogicalResult
Context::convertFunction(const slang::ast::SubroutineSymbol &subroutine) {
  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = subroutine.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Keep track of the current scope for virtual interface member access.
  // This allows us to bind syntax to expressions when accessing interface
  // members through virtual interfaces inside class methods.
  auto prevScope = currentScope;
  currentScope = &subroutine;
  auto scopeGuard = llvm::make_scope_exit([&] { currentScope = prevScope; });

  // First get or create the function declaration.
  auto *lowering = declareFunction(subroutine);
  if (!lowering)
    return failure();

  // If function already has been finalized, or is already being converted
  // (recursive/re-entrant calls) stop here.
  if (lowering->capturesFinalized || lowering->isConverting)
    return success();

  // Helper to construct a default value for a given type (used for stubbing
  // pure virtual functions).
  auto getDefaultValue = [&](Type ty, Location loc) -> Value {
    if (auto intTy = dyn_cast<moore::IntType>(ty))
      return moore::ConstantOp::create(builder, loc, intTy, 0);
    if (isa<moore::StringType>(ty)) {
      // Create an empty string by converting a 0-width integer to string
      auto intTy = moore::IntType::getInt(getContext(), 8);
      auto emptyInt = moore::ConstantStringOp::create(builder, loc, intTy, "");
      return moore::IntToStringOp::create(builder, loc, emptyInt);
    }
    if (isa<moore::FormatStringType>(ty)) {
      // Create empty format string from empty string
      auto intTy = moore::IntType::getInt(getContext(), 8);
      auto emptyInt = moore::ConstantStringOp::create(builder, loc, intTy, "");
      auto empty = moore::IntToStringOp::create(builder, loc, emptyInt);
      return moore::FormatStringOp::create(builder, loc, ty, empty,
                                           IntegerAttr(), nullptr, nullptr);
    }
    if (auto classTy = dyn_cast<moore::ClassHandleType>(ty))
      return moore::ClassNullOp::create(builder, loc, classTy);
    if (auto queueTy = dyn_cast<moore::QueueType>(ty))
      return moore::QueueConcatOp::create(builder, loc, queueTy, {});
    // Fallback: unrealized cast placeholder.
    return mlir::UnrealizedConversionCastOp::create(builder, loc, ty,
                                                    ValueRange{})
        .getResult(0);
  };

  // DPI-imported functions have no body to convert; just mark them as finalized.
  if (subroutine.flags & slang::ast::MethodFlags::DPIImport) {
    lowering->capturesFinalized = true;
    return success();
  }

  const bool isMethod = (subroutine.thisVar != nullptr);

  ValueSymbolScope scope(valueSymbols);

  // Create a function body block and populate it with block arguments.
  SmallVector<moore::VariableOp> argVariables;
  auto &block = lowering->op.getBody().emplaceBlock();

  LLVM_DEBUG(llvm::dbgs() << "convertFunction: "
                          << lowering->op.getSymName() << "\n");

  // Pure virtual methods: stub out with a default return value.
  // Do this BEFORE full argument processing to avoid issues with pure virtual
  // methods that may have incomplete argument info, but we still need to add
  // block arguments to match the function signature.
  if (subroutine.flags & slang::ast::MethodFlags::Pure) {
    // Add block arguments to match function signature
    auto funcTy = lowering->op.getFunctionType();
    auto loc = lowering->op.getLoc();
    for (auto inputTy : funcTy.getInputs()) {
      block.addArgument(inputTy, loc);
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    if (funcTy.getNumResults() == 0) {
      mlir::func::ReturnOp::create(builder, loc);
    } else {
      auto retTy = funcTy.getResult(0);
      auto retVal = getDefaultValue(retTy, loc);
      mlir::func::ReturnOp::create(builder, loc, retVal);
    }
    lowering->capturesFinalized = true;
    return success();
  }

  // Check if this is an interface method (task/function inside an interface).
  // Interface methods have an implicit first argument for the interface instance.
  bool isInterfaceMethod = false;
  const slang::ast::InstanceBodySymbol *ifaceBody = nullptr;
  const auto &parentSym = subroutine.getParentScope()->asSymbol();
  if (parentSym.kind == slang::ast::SymbolKind::InstanceBody) {
    if (auto *instBody = parentSym.as_if<slang::ast::InstanceBodySymbol>()) {
      if (instBody->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        isInterfaceMethod = true;
        ifaceBody = instBody;
      }
    }
  }

  // If this is a class method, the first input is %this :
  // !moore.class<@C>
  if (isMethod) {
    auto thisLoc = convertLocation(subroutine.location);
    auto thisType = lowering->op.getFunctionType().getInput(0);
    auto thisArg = block.addArgument(thisType, thisLoc);

    // Bind `this` so NamedValue/MemberAccess can find it.
    valueSymbols.insert(subroutine.thisVar, thisArg);
  }

  // If this is an interface method, the first input is the interface instance.
  // Set up the interface argument for signal access within the method body.
  auto prevInterfaceArg = currentInterfaceArg;
  auto prevInterfaceBody = currentInterfaceBody;
  auto prevInterfaceSignalNames = std::move(interfaceSignalNames);
  auto interfaceArgGuard = llvm::make_scope_exit([&] {
    currentInterfaceArg = prevInterfaceArg;
    currentInterfaceBody = prevInterfaceBody;
    interfaceSignalNames = std::move(prevInterfaceSignalNames);
  });

  if (isInterfaceMethod) {
    auto ifaceLoc = convertLocation(subroutine.location);
    auto ifaceType = lowering->op.getFunctionType().getInput(0);
    auto ifaceArg = block.addArgument(ifaceType, ifaceLoc);

    // Store the interface argument and body for signal access in expressions
    currentInterfaceArg = ifaceArg;
    currentInterfaceBody = ifaceBody;

    // Build the signal name map for this interface.
    // We always clear and rebuild the map because we may be processing a
    // different interface than before (e.g., when converting virtual interface
    // method calls from within a class, multiple interfaces may have methods
    // called in sequence).
    interfaceSignalNames.clear();
    for (auto *symbol : ifaceBody->getPortList()) {
      if (const auto *port = symbol->as_if<slang::ast::PortSymbol>()) {
        if (port->internalSymbol)
          interfaceSignalNames[port->internalSymbol] = port->name;
        interfaceSignalNames[port] = port->name;
      } else if (const auto *multiPort =
                     symbol->as_if<slang::ast::MultiPortSymbol>()) {
        for (auto *port : multiPort->ports) {
          if (port->internalSymbol)
            interfaceSignalNames[port->internalSymbol] = port->name;
          interfaceSignalNames[port] = port->name;
        }
      }
    }
    for (auto &member : ifaceBody->members()) {
      if (auto *var = member.as_if<slang::ast::VariableSymbol>())
        interfaceSignalNames[var] = var->name;
      if (auto *net = member.as_if<slang::ast::NetSymbol>())
        interfaceSignalNames[net] = net->name;
    }
  }

  // Add user-defined block arguments
  auto inputs = lowering->op.getFunctionType().getInputs();
  auto astArgs = subroutine.getArguments();
  // Drop the implicit first argument (this for methods, interface for interface methods)
  unsigned numImplicitArgs = (isMethod || isInterfaceMethod) ? 1 : 0;
  auto valInputs = llvm::ArrayRef<Type>(inputs).drop_front(numImplicitArgs);

  for (auto [astArg, type] : llvm::zip(astArgs, valInputs)) {
    auto loc = convertLocation(astArg->location);
    auto blockArg = block.addArgument(type, loc);

    if (isa<moore::RefType>(type)) {
      valueSymbols.insert(astArg, blockArg);
    } else {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&block);

      auto shadowArg = moore::VariableOp::create(
          builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
          StringAttr{}, blockArg);
      valueSymbols.insert(astArg, shadowArg);
      argVariables.push_back(shadowArg);
    }
  }

  // Convert the body of the function.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);

  // Note: Pure virtual methods are handled above (before argument processing).

  Value returnVar;
  if (subroutine.returnValVar) {
    auto type = convertType(*subroutine.returnValVar->getDeclaredType());
    if (!type)
      return failure();
    returnVar = moore::VariableOp::create(
        builder, lowering->op.getLoc(),
        moore::RefType::get(cast<moore::UnpackedType>(type)), StringAttr{},
        Value{});
    valueSymbols.insert(subroutine.returnValVar, returnVar);
  }

  // Save previous callbacks
  auto prevRCb = rvalueReadCallback;
  auto prevWCb = variableAssignCallback;
  auto prevRCbGuard = llvm::make_scope_exit([&] {
    rvalueReadCallback = prevRCb;
    variableAssignCallback = prevWCb;
  });

  // Capture this function's captured context directly
  rvalueReadCallback = [lowering, prevRCb](moore::ReadOp rop) {
    mlir::Value ref = rop.getInput();

    // Don't capture anything that's not a reference
    mlir::Type ty = ref.getType();
    if (!ty || !(isa<moore::RefType>(ty)))
      return;

    // Don't capture anything that's a local reference
    mlir::Region *defReg = ref.getParentRegion();
    if (defReg && lowering->op.getBody().isAncestor(defReg))
      return;

    // If we've already recorded this capture, skip.
    if (lowering->captureIndex.count(ref))
      return;

    // Only capture refs defined outside this function’s region
    auto [it, inserted] =
        lowering->captureIndex.try_emplace(ref, lowering->captures.size());
    if (inserted) {
      lowering->captures.push_back(ref);
      // Propagate over outer scope
      if (prevRCb)
        prevRCb(rop); // chain previous callback
    }
  };
  // Capture this function's captured context directly
  variableAssignCallback = [lowering, prevWCb](mlir::Operation *op) {
    mlir::Value dstRef =
        llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
            .Case<moore::BlockingAssignOp, moore::NonBlockingAssignOp,
                  moore::DelayedNonBlockingAssignOp>(
                [](auto op) { return op.getDst(); })
            .Default([](auto) -> mlir::Value { return {}; });

    // Don't capture anything that's not a reference
    mlir::Type ty = dstRef.getType();
    if (!ty || !(isa<moore::RefType>(ty)))
      return;

    // Don't capture anything that's a local reference
    mlir::Region *defReg = dstRef.getParentRegion();
    if (defReg && lowering->op.getBody().isAncestor(defReg))
      return;

    // If we've already recorded this capture, skip.
    if (lowering->captureIndex.count(dstRef))
      return;

    // Only capture refs defined outside this function’s region
    auto [it, inserted] =
        lowering->captureIndex.try_emplace(dstRef, lowering->captures.size());
    if (inserted) {
      lowering->captures.push_back(dstRef);
      // Propagate over outer scope
      if (prevWCb)
        prevWCb(op); // chain previous callback
    }
  };

  auto savedThis = currentThisRef;
  currentThisRef = valueSymbols.lookup(subroutine.thisVar);
  auto restoreThis = llvm::make_scope_exit([&] { currentThisRef = savedThis; });

  // Reset assertion guard when entering an isolated function region.
  // Values from the outer scope cannot be used inside a func::FuncOp due to
  // IsolatedFromAbove, so we must clear any guard that was set in outer scope.
  auto savedAssertionGuard = currentAssertionGuard;
  currentAssertionGuard = {};
  auto restoreGuard =
      llvm::make_scope_exit([&] { currentAssertionGuard = savedAssertionGuard; });

  // Track current function lowering for downstream helpers (returns, captures).
  auto *savedLowering = currentFunctionLowering;
  currentFunctionLowering = lowering;
  auto restoreLowering =
      llvm::make_scope_exit([&] { currentFunctionLowering = savedLowering; });

  lowering->isConverting = true;
  auto convertingGuard = llvm::make_scope_exit([&] { lowering->isConverting = false; });

  if (failed(convertStatement(subroutine.getBody())))
    return failure();

  // Plumb captures into the function as extra block arguments
  if (failed(finalizeFunctionBodyCaptures(*lowering)))
    return failure();

  // For the special case of recursive functions, fix the call sites within the
  // body
  if (failed(rewriteCallSitesToPassCaptures(lowering->op, lowering->captures)))
    return failure();

  // If there was no explicit return statement provided by the user, insert a
  // default one.
  if (builder.getBlock()) {
    if (!subroutine.getReturnType().isVoid()) {
      Value retVal;
      if (returnVar) {
        // Read the return variable that was populated by the function body.
        retVal = moore::ReadOp::create(builder, returnVar.getLoc(), returnVar);
      } else {
        // No return variable was created (e.g., some built-in functions).
        // Create a default value based on the function's return type.
        auto funcTy = lowering->op.getFunctionType();
        auto retTy = funcTy.getResult(0);
        retVal = getDefaultValue(retTy, lowering->op.getLoc());
      }
      mlir::func::ReturnOp::create(builder, lowering->op.getLoc(), retVal);
    } else {
      mlir::func::ReturnOp::create(builder, lowering->op.getLoc(),
                                   ValueRange{});
    }
  }
  if (returnVar && returnVar.use_empty())
    returnVar.getDefiningOp()->erase();

  for (auto var : argVariables) {
    if (llvm::all_of(var->getUsers(),
                     [](auto *user) { return isa<moore::ReadOp>(user); })) {
      for (auto *user : llvm::make_early_inc_range(var->getUsers())) {
        user->getResult(0).replaceAllUsesWith(var.getInitial());
        user->erase();
      }
      var->erase();
    }
  }

  lowering->capturesFinalized = true;
  return success();
}

LogicalResult
Context::finalizeFunctionBodyCaptures(FunctionLowering &lowering) {
  if (lowering.captures.empty())
    return success();

  MLIRContext *ctx = getContext();

  // Build new input type list: existing inputs + capture ref types.
  SmallVector<Type> newInputs(lowering.op.getFunctionType().getInputs().begin(),
                              lowering.op.getFunctionType().getInputs().end());

  for (Value cap : lowering.captures) {
    // Expect captures to be refs.
    Type capTy = cap.getType();
    if (!isa<moore::RefType>(capTy)) {
      return lowering.op.emitError(
          "expected captured value to be a ref-like type");
    }
    newInputs.push_back(capTy);
  }

  // Results unchanged.
  auto newFuncTy = FunctionType::get(
      ctx, newInputs, lowering.op.getFunctionType().getResults());
  lowering.op.setFunctionType(newFuncTy);

  // Add the new block arguments to the entry block.
  Block &entry = lowering.op.getBody().front();
  SmallVector<Value> capArgs;
  capArgs.reserve(lowering.captures.size());
  for (Type t :
       llvm::ArrayRef<Type>(newInputs).take_back(lowering.captures.size())) {
    capArgs.push_back(entry.addArgument(t, lowering.op.getLoc()));
  }

  // Replace uses of each captured Value *inside the function body* with the new
  // arg. Keep uses outside untouched (e.g., in callers).
  for (auto [cap, idx] : lowering.captureIndex) {
    Value arg = capArgs[idx];
    cap.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return lowering.op->isProperAncestor(use.getOwner());
    });
  }

  return success();
}

namespace {

/// Helper function to construct the classes fully qualified base class name
/// and the name of all implemented interface classes
std::pair<mlir::SymbolRefAttr, mlir::ArrayAttr>
buildBaseAndImplementsAttrs(Context &context,
                            const slang::ast::ClassType &cls) {
  mlir::MLIRContext *ctx = context.getContext();

  // Base class (if any)
  // Look up the actual ClassDeclOp to get the correct symbol name, since
  // symbolTable.insert() may have renamed it to avoid conflicts.
  mlir::SymbolRefAttr base;
  if (const auto *b = cls.getBaseClass()) {
    const auto &canonicalBase = b->getCanonicalType();
    if (const auto *baseClass = canonicalBase.as_if<slang::ast::ClassType>()) {
      auto it = context.classes.find(baseClass);
      if (it != context.classes.end() && it->second && it->second->op) {
        base = mlir::SymbolRefAttr::get(it->second->op.getSymNameAttr());
      } else {
        // Fallback to computing the name if the class isn't in the map yet.
        base = mlir::SymbolRefAttr::get(fullyQualifiedClassName(context, *b));
      }
    } else {
      // Not a ClassType, fall back to name-based lookup
      base = mlir::SymbolRefAttr::get(fullyQualifiedClassName(context, *b));
    }
  }

  // Implemented interfaces (if any)
  SmallVector<mlir::Attribute> impls;
  if (auto ifaces = cls.getDeclaredInterfaces(); !ifaces.empty()) {
    impls.reserve(ifaces.size());
    for (const auto *iface : ifaces)
      impls.push_back(mlir::FlatSymbolRefAttr::get(
          fullyQualifiedClassName(context, *iface)));
  }

  mlir::ArrayAttr implArr =
      impls.empty() ? mlir::ArrayAttr() : mlir::ArrayAttr::get(ctx, impls);

  return {base, implArr};
}

/// Visit a slang::ast::ClassType and populate the body of an existing
/// moore::ClassDeclOp with field/method decls.
struct ClassDeclVisitor {
  Context &context;
  OpBuilder &builder;
  ClassLowering &classLowering;

  ClassDeclVisitor(Context &ctx, ClassLowering &lowering)
      : context(ctx), builder(ctx.builder), classLowering(lowering) {}

  LogicalResult run(const slang::ast::ClassType &classAST) {
    LLVM_DEBUG(llvm::dbgs() << "=== ClassDeclVisitor::run: " << classAST.name
                            << " ===\n");

    // Check if the body has already been converted (or is being converted).
    if (classLowering.bodyConverted) {
      LLVM_DEBUG(llvm::dbgs() << "  Body already converted, skipping\n");
      return success();
    }
    classLowering.bodyConverted = true;

    // Save and clear currentThisRef during class body conversion.
    // This prevents stale values from a surrounding function conversion from
    // being used when converting constraint expressions or other class members.
    // Without this, if class A's method conversion triggers conversion of
    // class B, and class B has constraints that access properties, the
    // constraints would incorrectly try to upcast class A's 'this' to class B.
    auto savedThisRef = context.currentThisRef;
    context.currentThisRef = {};
    auto restoreThisRef =
        llvm::make_scope_exit([&] { context.currentThisRef = savedThisRef; });

    // The block is created in declareClass() to satisfy SingleBlock trait.
    Block &body = classLowering.op.getBody().front();

    // Log base class if present
    if (classAST.getBaseClass()) {
      LLVM_DEBUG(llvm::dbgs() << "  Base class: "
                              << classAST.getBaseClass()->name << "\n");
    }

    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPointToEnd(&body);

    // Three-pass conversion:
    // 1. Properties first (so constraints can reference them)
    // 2. Constraints (may reference properties and call methods)
    // 3. Methods (may reference properties)

    // Pass 1a: Convert properties, parameters, and type aliases
    LLVM_DEBUG(llvm::dbgs() << "  Pass 1a: Properties, parameters, aliases\n");
    for (const auto &mem : classAST.members()) {
      if (mem.kind == slang::ast::SymbolKind::ClassProperty ||
          mem.kind == slang::ast::SymbolKind::Parameter ||
          mem.kind == slang::ast::SymbolKind::TypeAlias ||
          mem.kind == slang::ast::SymbolKind::TypeParameter) {
        LLVM_DEBUG(llvm::dbgs() << "    Processing member: " << mem.name
                                << " (kind: " << slang::ast::toString(mem.kind)
                                << ")\n");
        if (failed(mem.visit(*this))) {
          LLVM_DEBUG(llvm::dbgs() << "    FAILED at member: " << mem.name
                                  << " (kind: "
                                  << slang::ast::toString(mem.kind) << ")\n");
          return failure();
        }
      }
    }

    // Pass 1b: Convert constraints (after properties are declared)
    LLVM_DEBUG(llvm::dbgs() << "  Pass 1b: Constraints\n");
    for (const auto &mem : classAST.members()) {
      if (mem.kind == slang::ast::SymbolKind::ConstraintBlock) {
        LLVM_DEBUG(llvm::dbgs() << "    Processing member: " << mem.name
                                << " (kind: " << slang::ast::toString(mem.kind)
                                << ")\n");
        if (failed(mem.visit(*this))) {
          LLVM_DEBUG(llvm::dbgs() << "    FAILED at member: " << mem.name
                                  << " (kind: "
                                  << slang::ast::toString(mem.kind) << ")\n");
          return failure();
        }
      }
    }

    // Pass 2: Convert methods and other members
    LLVM_DEBUG(llvm::dbgs() << "  Pass 2: Methods and other members\n");
    for (const auto &mem : classAST.members()) {
      if (mem.kind != slang::ast::SymbolKind::ClassProperty &&
          mem.kind != slang::ast::SymbolKind::Parameter &&
          mem.kind != slang::ast::SymbolKind::TypeAlias &&
          mem.kind != slang::ast::SymbolKind::TypeParameter &&
          mem.kind != slang::ast::SymbolKind::ConstraintBlock) {
        LLVM_DEBUG(llvm::dbgs() << "    Processing member: " << mem.name
                                << " (kind: " << slang::ast::toString(mem.kind)
                                << ")\n");
        if (failed(mem.visit(*this))) {
          LLVM_DEBUG(llvm::dbgs() << "    FAILED at member: " << mem.name
                                  << " (kind: "
                                  << slang::ast::toString(mem.kind) << ")\n");
          return failure();
        }
      }
    }

    // Pass 3: Register inherited virtual methods in the vtable.
    // When a derived class inherits virtual methods from a base class without
    // overriding them, those methods must still be registered in the derived
    // class's vtable so that virtual dispatch works correctly.
    // classAST.members() only returns explicitly defined members, not inherited
    // ones, so we need to walk up the inheritance chain manually.
    LLVM_DEBUG(llvm::dbgs() << "  Pass 3: Inherited virtual methods\n");

    // Collect names of methods explicitly defined in this class to detect
    // overrides.
    llvm::StringSet<> definedMethods;
    for (const auto &mem : classAST.members()) {
      if (mem.kind == slang::ast::SymbolKind::Subroutine ||
          mem.kind == slang::ast::SymbolKind::MethodPrototype) {
        definedMethods.insert(mem.name);
      }
    }

    // Walk up the inheritance chain.
    const slang::ast::Type *baseType = classAST.getBaseClass();
    while (baseType) {
      const auto &canonicalBase = baseType->getCanonicalType();
      const auto *baseClass = canonicalBase.as_if<slang::ast::ClassType>();
      if (!baseClass)
        break;

      LLVM_DEBUG(llvm::dbgs() << "    Checking base class: " << baseClass->name
                              << "\n");

      // Iterate over base class members looking for virtual methods.
      for (const auto &mem : baseClass->members()) {
        // Check for SubroutineSymbol (method implementation).
        if (auto *fn = mem.as_if<slang::ast::SubroutineSymbol>()) {
          // Skip if not virtual.
          if (!(fn->flags & slang::ast::MethodFlags::Virtual))
            continue;
          // Skip if overridden in derived class.
          if (definedMethods.contains(fn->name))
            continue;
          // Skip pure virtual methods (no implementation).
          if (fn->flags & slang::ast::MethodFlags::Pure)
            continue;
          // Skip builtin methods.
          if (fn->flags & slang::ast::MethodFlags::BuiltIn)
            continue;

          LLVM_DEBUG(llvm::dbgs()
                     << "      Registering inherited virtual method: "
                     << fn->name << "\n");

          // Look up the function lowering from the base class.
          auto it = context.functions.find(fn);
          if (it == context.functions.end() || !it->second || !it->second->op) {
            // The base class function wasn't converted yet - try to convert it.
            if (failed(context.convertFunction(*fn))) {
              LLVM_DEBUG(llvm::dbgs()
                         << "        FAILED: Could not convert base method\n");
              return failure();
            }
            it = context.functions.find(fn);
            if (it == context.functions.end() || !it->second ||
                !it->second->op) {
              LLVM_DEBUG(llvm::dbgs()
                         << "        FAILED: Function not in map after "
                            "conversion\n");
              return failure();
            }
          }

          auto *lowering = it->second.get();
          auto loc = convertLocation(fn->location);
          FunctionType fnTy = lowering->op.getFunctionType();

          // Emit the method decl pointing to the base class's function.
          moore::ClassMethodDeclOp::create(builder, loc, fn->name, fnTy,
                                           SymbolRefAttr::get(lowering->op));

          // Add to definedMethods so we don't re-register this method if it
          // appears in an even more ancestral base class.
          definedMethods.insert(fn->name);
        }

        // Also check for MethodPrototypeSymbol (extern method declarations).
        // These are important when an intermediate base class has an extern
        // virtual method that overrides an ancestor's method.
        if (auto *proto = mem.as_if<slang::ast::MethodPrototypeSymbol>()) {
          // Skip if not virtual.
          if (!(proto->flags & slang::ast::MethodFlags::Virtual))
            continue;
          // Skip if already defined in derived class or earlier in chain.
          if (definedMethods.contains(proto->name))
            continue;
          // Get the implementation.
          const auto *impl = proto->getSubroutine();
          if (!impl)
            continue;
          // Skip builtin methods.
          if (impl->flags & slang::ast::MethodFlags::BuiltIn)
            continue;

          LLVM_DEBUG(llvm::dbgs()
                     << "      Registering inherited extern virtual method: "
                     << proto->name << "\n");

          // Look up the function lowering for the implementation.
          auto it = context.functions.find(impl);
          if (it == context.functions.end() || !it->second || !it->second->op) {
            if (failed(context.convertFunction(*impl))) {
              LLVM_DEBUG(llvm::dbgs()
                         << "        FAILED: Could not convert base method\n");
              return failure();
            }
            it = context.functions.find(impl);
            if (it == context.functions.end() || !it->second ||
                !it->second->op) {
              LLVM_DEBUG(llvm::dbgs()
                         << "        FAILED: Function not in map after "
                            "conversion\n");
              return failure();
            }
          }

          auto *lowering = it->second.get();
          auto loc = convertLocation(proto->location);
          FunctionType fnTy = lowering->op.getFunctionType();

          moore::ClassMethodDeclOp::create(builder, loc, proto->name, fnTy,
                                           SymbolRefAttr::get(lowering->op));

          definedMethods.insert(proto->name);
        }
      }

      // Move to the next base class.
      baseType = baseClass->getBaseClass();
    }

    LLVM_DEBUG(llvm::dbgs() << "  ClassDeclVisitor::run completed successfully\n");
    return success();
  }

  // Properties: ClassPropertySymbol
  LogicalResult visit(const slang::ast::ClassPropertySymbol &prop) {
    LLVM_DEBUG(llvm::dbgs() << "      ClassPropertySymbol: " << prop.name
                            << "\n");
    auto loc = convertLocation(prop.location);
    auto ty = context.convertType(prop.getType());
    if (!ty) {
      LLVM_DEBUG(llvm::dbgs() << "        FAILED: Could not convert type\n");
      return failure();
    }

    // Check if this is a static property.
    bool isStatic = prop.lifetime == slang::ast::VariableLifetime::Static;

    // Static properties are stored as global variables, not instance fields.
    if (isStatic) {
      // Check if already converted (for on-demand conversion).
      if (context.globalVariables.count(&prop))
        return success();

      // Check by fully qualified name (handles multiple specializations of
      // parameterized classes that share the same static member).
      auto symName = fullyQualifiedSymbolName(context, prop);
      if (context.globalVariablesByName.count(symName)) {
        // Reuse the existing global variable for this symbol pointer.
        context.globalVariables[&prop] = context.globalVariablesByName[symName];
        return success();
      }

      // Pick an insertion point for this variable at the module level.
      OpBuilder::InsertionGuard g(builder);
      auto it = context.orderedRootOps.upper_bound(prop.location);
      if (it == context.orderedRootOps.end())
        builder.setInsertionPointToEnd(context.intoModuleOp.getBody());
      else
        builder.setInsertionPoint(it->second);

      // Create the global variable op (symName already computed above).
      auto varOp = moore::GlobalVariableOp::create(
          builder, loc, symName, cast<moore::UnpackedType>(ty));
      context.orderedRootOps.insert({prop.location, varOp});
      context.globalVariables[&prop] = varOp;
      context.globalVariablesByName[symName] = varOp;

      // If the property has an initializer expression, remember it for later.
      if (prop.getInitializer())
        context.globalVariableWorklist.push_back(&prop);

      return success();
    }

    // Convert slang's Visibility to Moore's MemberAccess
    moore::MemberAccess memberAccess;
    switch (prop.visibility) {
    case slang::ast::Visibility::Public:
      memberAccess = moore::MemberAccess::Public;
      break;
    case slang::ast::Visibility::Protected:
      memberAccess = moore::MemberAccess::Protected;
      break;
    case slang::ast::Visibility::Local:
      memberAccess = moore::MemberAccess::Local;
      break;
    default:
      memberAccess = moore::MemberAccess::Public;
      break;
    }

    // Convert slang's RandMode to Moore's RandMode
    moore::RandMode randMode;
    switch (prop.randMode) {
    case slang::ast::RandMode::None:
      randMode = moore::RandMode::None;
      break;
    case slang::ast::RandMode::Rand:
      randMode = moore::RandMode::Rand;
      break;
    case slang::ast::RandMode::RandC:
      randMode = moore::RandMode::RandC;
      break;
    default:
      randMode = moore::RandMode::None;
      break;
    }

    moore::ClassPropertyDeclOp::create(builder, loc, prop.name, ty, memberAccess,
                                       randMode);
    return success();
  }

  // Parameters in specialized classes hold no further information; slang
  // already elaborates them in all relevant places.
  LogicalResult visit(const slang::ast::ParameterSymbol &) { return success(); }

  // Parameters in specialized classes hold no further information; slang
  // already elaborates them in all relevant places.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Type aliases in specialized classes hold no further information; slang
  // already elaborates them in all relevant places.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }

  // Fully-fledged functions - SubroutineSymbol
  LogicalResult visit(const slang::ast::SubroutineSymbol &fn) {
    LLVM_DEBUG(llvm::dbgs() << "      SubroutineSymbol: " << fn.name << "\n");
    if (fn.flags & slang::ast::MethodFlags::BuiltIn) {
      LLVM_DEBUG(llvm::dbgs() << "        Skipping builtin method\n");
      static bool remarkEmitted = false;
      if (remarkEmitted)
        return success();

      mlir::emitRemark(classLowering.op.getLoc())
          << "Class builtin functions (needed for randomization, constraints, "
             "and covergroups) are not yet supported and will be dropped "
             "during lowering.";
      remarkEmitted = true;
      return success();
    }

    const mlir::UnitAttr isVirtual =
        (fn.flags & slang::ast::MethodFlags::Virtual)
            ? UnitAttr::get(context.getContext())
            : nullptr;

    auto loc = convertLocation(fn.location);
    // Pure virtual functions regulate inheritance rules during parsing.
    // They don't emit any code, so we don't need to convert them, we only need
    // to register them for the purpose of stable VTable construction.
    if (fn.flags & slang::ast::MethodFlags::Pure) {
      LLVM_DEBUG(llvm::dbgs() << "        Pure virtual method\n");
      // Add an extra %this argument.
      SmallVector<Type, 1> extraParams;
      auto classSym =
          mlir::FlatSymbolRefAttr::get(classLowering.op.getSymNameAttr());
      auto handleTy =
          moore::ClassHandleType::get(context.getContext(), classSym);
      extraParams.push_back(handleTy);

      auto funcTy = getFunctionSignature(context, fn, extraParams);
      if (!funcTy) {
        LLVM_DEBUG(llvm::dbgs()
                   << "        FAILED: Could not get function signature\n");
        return failure();
      }
      moore::ClassMethodDeclOp::create(builder, loc, fn.name, funcTy, nullptr);
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "        Declaring function\n");
    auto *lowering = context.declareFunction(fn);
    if (!lowering) {
      LLVM_DEBUG(llvm::dbgs() << "        FAILED: declareFunction returned null\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "        Converting function body\n");
    if (failed(context.convertFunction(fn))) {
      LLVM_DEBUG(llvm::dbgs() << "        FAILED: convertFunction failed\n");
      return failure();
    }

    // If the function is still being converted (recursive call scenario),
    // capturesFinalized will be false but this is expected - the conversion
    // will complete when the recursion unwinds. Only fail if the function
    // is not being converted and captures still aren't finalized.
    if (!lowering->capturesFinalized && !lowering->isConverting) {
      mlir::emitError(loc)
          << "function '" << fn.name << "' conversion did not complete";
      return failure();
    }

    // We only emit methoddecls for virtual methods.
    if (!isVirtual)
      return success();

    // Grab the finalized function type from the lowered func.op.
    FunctionType fnTy = lowering->op.getFunctionType();
    // Emit the method decl into the class body, preserving source order.
    moore::ClassMethodDeclOp::create(builder, loc, fn.name, fnTy,
                                     SymbolRefAttr::get(lowering->op));

    return success();
  }

  // A method prototype corresponds to the forward declaration of a concrete
  // method, the forward declaration of a virtual method, or the defintion of an
  // interface method meant to be implemented by classes implementing the
  // interface class.
  // In the first two cases, the best thing to do is to look up the actual
  // implementation and translate it when reading the method prototype, so we
  // can insert the MethodDeclOp in the correct order in the ClassDeclOp.
  // The latter case requires support for virtual interface methods, which is
  // currently not implemented. Since forward declarations of non-interface
  // methods must be followed by an implementation within the same compilation
  // unit, we can simply return a failure if we can't find a unique
  // implementation until we implement support for interface methods.
  LogicalResult visit(const slang::ast::MethodPrototypeSymbol &fn) {
    LLVM_DEBUG(llvm::dbgs() << "      MethodPrototypeSymbol: " << fn.name
                            << "\n");
    const auto *externImpl = fn.getSubroutine();
    // We needn't convert a forward declaration without a unique implementation.
    if (!externImpl) {
      LLVM_DEBUG(llvm::dbgs()
                 << "      FAILED: No implementation for forward declaration\n");
      mlir::emitError(convertLocation(fn.location))
          << "Didn't find an implementation matching the forward declaration "
             "of "
          << fn.name;
      return failure();
    }

    // Check if the PROTOTYPE is virtual. The implementation may not have the
    // Virtual flag for extern methods - only the prototype carries it.
    bool protoIsVirtual =
        (fn.flags & slang::ast::MethodFlags::Virtual) ? true : false;

    LLVM_DEBUG(llvm::dbgs() << "      Found implementation, visiting"
                            << (protoIsVirtual ? " (virtual)" : "") << "\n");

    // Skip builtin methods.
    if (externImpl->flags & slang::ast::MethodFlags::BuiltIn) {
      LLVM_DEBUG(llvm::dbgs() << "        Skipping builtin method\n");
      return success();
    }

    // Convert the function body.
    auto *lowering = context.declareFunction(*externImpl);
    if (!lowering) {
      LLVM_DEBUG(llvm::dbgs()
                 << "        FAILED: declareFunction returned null\n");
      return failure();
    }

    if (failed(context.convertFunction(*externImpl))) {
      LLVM_DEBUG(llvm::dbgs() << "        FAILED: convertFunction failed\n");
      return failure();
    }

    // If the function is still being converted (recursive call scenario),
    // capturesFinalized will be false but this is expected.
    if (!lowering->capturesFinalized && !lowering->isConverting) {
      mlir::emitError(convertLocation(fn.location))
          << "function '" << fn.name << "' conversion did not complete";
      return failure();
    }

    // Create ClassMethodDeclOp if the PROTOTYPE is virtual.
    // This is the key fix: we use the prototype's virtual flag, not the
    // implementation's, because extern method implementations don't carry
    // the virtual flag - only their prototypes do.
    if (protoIsVirtual) {
      auto loc = convertLocation(fn.location);
      FunctionType fnTy = lowering->op.getFunctionType();
      moore::ClassMethodDeclOp::create(builder, loc, fn.name, fnTy,
                                       SymbolRefAttr::get(lowering->op));
    }

    return success();
  }

  // Nested class definition, skip
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Nested class definition, convert
  LogicalResult visit(const slang::ast::ClassType &cls) {
    LLVM_DEBUG(llvm::dbgs() << "      Nested ClassType: " << cls.name << "\n");
    auto result = context.convertClassDeclaration(cls);
    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "        FAILED: convertClassDeclaration failed\n");
    }
    return result;
  }

  // Transparent members: ignore (inherited names pulled in by slang)
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Empty members: ignore
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Covergroups declared inside a class: ensure the covergroup is converted.
  // Note: The ClassPropertySymbol that references this covergroup is processed
  // in Pass 1, which already triggers covergroup conversion through type
  // conversion and creates the class property. This handler just ensures the
  // covergroup is converted if it hasn't been already (which shouldn't happen
  // in normal cases).
  LogicalResult visit(const slang::ast::CovergroupType &covergroup) {
    LLVM_DEBUG(llvm::dbgs() << "      CovergroupType: " << covergroup.name
                            << "\n");

    // The covergroup should already be converted by the ClassPropertySymbol
    // handler through type conversion. If not, convert it now.
    if (!context.covergroups.count(&covergroup)) {
      if (failed(context.convertCovergroup(covergroup)))
        return failure();
    }

    // Don't create a property here - the ClassPropertySymbol handler already
    // created it.
    return success();
  }

  // Constraint blocks: convert to moore.constraint.block
  LogicalResult visit(const slang::ast::ConstraintBlockSymbol &constraint) {
    auto loc = convertLocation(constraint.location);

    // Check for static and pure flags
    bool isStatic =
        (constraint.flags & slang::ast::ConstraintBlockFlags::Static) ==
        slang::ast::ConstraintBlockFlags::Static;
    bool isPure =
        (constraint.flags & slang::ast::ConstraintBlockFlags::Pure) ==
        slang::ast::ConstraintBlockFlags::Pure;

    // Create the constraint block operation
    auto constraintOp = moore::ConstraintBlockOp::create(
        builder, loc, constraint.name, isStatic, isPure);

    // Create the body block with a 'this' argument for non-static constraints.
    // This allows constraint expressions to call methods and access properties
    // on the class instance.
    Block &bodyBlock = constraintOp.getBody().emplaceBlock();
    Value thisArg;
    if (!isStatic) {
      auto classSymRef =
          mlir::FlatSymbolRefAttr::get(classLowering.op.getSymNameAttr());
      auto classHandleType =
          moore::ClassHandleType::get(context.getContext(), classSymRef);
      thisArg = bodyBlock.addArgument(classHandleType, loc);
    }

    // Convert the constraint expressions inside the block
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPointToEnd(&bodyBlock);

    // Get the constraint body from slang
    const auto &constraintBody = constraint.getConstraints();
    // Handle implicit constraint blocks with no body (Invalid constraint).
    // These are valid declarations that just have no constraints defined yet.
    if (constraintBody.kind != slang::ast::ConstraintKind::Invalid) {
      // Set up the 'this' reference for constraint expression conversion.
      // This allows method calls and property accesses within the constraint
      // to resolve against the class instance.
      auto savedThisRef = context.currentThisRef;
      auto savedInConstraint = context.inConstraintExpr;
      if (thisArg)
        context.currentThisRef = thisArg;
      context.inConstraintExpr = true;
      auto restoreContext = llvm::make_scope_exit([&] {
        context.currentThisRef = savedThisRef;
        context.inConstraintExpr = savedInConstraint;
      });

      if (failed(context.convertConstraint(constraintBody, loc)))
        return failure();
    }

    return success();
  }

  // Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    Location loc = UnknownLoc::get(context.getContext());
    if constexpr (requires { node.location; })
      loc = convertLocation(node.location);
    mlir::emitError(loc) << "unsupported construct in ClassType members: "
                         << slang::ast::toString(node.kind);
    return failure();
  }

private:
  Location convertLocation(const slang::SourceLocation &sloc) {
    return context.convertLocation(sloc);
  }
};
} // namespace

ClassLowering *Context::declareClass(const slang::ast::ClassType &cls) {
  // Check if there already is a declaration for this class.
  // IMPORTANT: Do NOT hold a reference to classes[&cls] across operations that
  // may modify the map (like recursive calls to convertClassDeclaration),
  // because DenseMap can rehash and invalidate references.
  bool isNewDecl = !classes.contains(&cls);
  if (isNewDecl) {
    // Create the ClassLowering entry first.
    classes[&cls] = std::make_unique<ClassLowering>();
    auto loc = convertLocation(cls.location);

    // Pick an insertion point for this function according to the source file
    // location.
    OpBuilder::InsertionGuard g(builder);
    auto it = orderedRootOps.upper_bound(cls.location);
    if (it == orderedRootOps.end())
      builder.setInsertionPointToEnd(intoModuleOp.getBody());
    else
      builder.setInsertionPoint(it->second);

    auto symName = fullyQualifiedClassName(*this, cls);

    // Create the ClassDeclOp first with empty base/implements attrs.
    // We need to do this before processing base classes because base class
    // processing may recursively reference this class (e.g., through method
    // parameter types), and we need lowering->op to be valid.
    auto classDeclOp =
        moore::ClassDeclOp::create(builder, loc, symName, nullptr, nullptr);
    // Emplace a block immediately to satisfy the SingleBlock/SymbolTable
    // requirements. ClassDeclVisitor::run() will populate this block later.
    classDeclOp.getBody().emplaceBlock();

    SymbolTable::setSymbolVisibility(classDeclOp,
                                     SymbolTable::Visibility::Public);
    orderedRootOps.insert(it, {cls.location, classDeclOp});
    // Re-fetch from map after potential modifications.
    classes[&cls]->op = classDeclOp;
    // insert() may rename the symbol if there's a conflict
    mlir::StringAttr actualSymName = symbolTable.insert(classDeclOp);

    // If this class is a specialization of a generic class, record the mapping
    // from specialized name to generic class name. This allows us to recognize
    // when two class types (e.g., uvm_pool_18 and uvm_pool) are related through
    // the same generic class template. Use actualSymName since insert() may
    // have renamed the symbol.
    if (cls.genericClass) {
      auto genericSymName = fullyQualifiedSymbolName(*this, *cls.genericClass);
      classSpecializationToGeneric[actualSymName] = genericSymName;
    }
  }

  // Always ensure base class is declared if present.
  // This ensures recursive type references find a valid lowering->op.
  // Use getCanonicalType() to unwrap type aliases (e.g., typedef class foo)
  // so we get the same ClassType pointer that's used in the 'classes' map.
  // Note: We don't fail if the base class body conversion fails - the class
  // declaration will still exist and can be referenced. Body conversion
  // failures are handled separately during member conversion.
  if (const auto *maybeBaseClass = cls.getBaseClass()) {
    const auto &canonicalBase = maybeBaseClass->getCanonicalType();
    if (const auto *baseClass = canonicalBase.as_if<slang::ast::ClassType>()) {
      if (!classes.contains(baseClass)) {
        // Trigger conversion - this will add baseClass to the classes map
        // even if body conversion fails.
        (void)convertClassDeclaration(*baseClass);
      }
    }
  }

  // Re-fetch the lowering pointer after potential map modifications from
  // recursive base class processing.
  auto &lowering = classes[&cls];
  if (!lowering || !lowering->op) {
    // This should not happen - the class should have been fully initialized
    // above. Return null to signal an error.
    return nullptr;
  }

  // Always update the base and implements attributes.
  // This handles the case where the class was forward-declared during type
  // conversion and the base attributes weren't set.
  auto [base, impls] = buildBaseAndImplementsAttrs(*this, cls);
  if (base && !lowering->op.getBaseAttr())
    lowering->op.setBaseAttr(base);
  if (impls && !lowering->op.getImplementedInterfacesAttr())
    lowering->op.setImplementedInterfacesAttr(impls);

  return lowering.get();
}

LogicalResult
Context::convertClassDeclaration(const slang::ast::ClassType &classdecl) {

  // Keep track of local time scale.
  auto prevTimeScale = timeScale;
  timeScale = classdecl.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard = llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Get or create the class declaration.
  auto *lowering = declareClass(classdecl);
  if (!lowering)
    return failure();

  // If the body has already been converted (or is being converted), skip.
  // ClassDeclVisitor::run checks if body is empty and populates it.
  if (failed(ClassDeclVisitor(*this, *lowering).run(classdecl)))
    return failure();

  return success();
}

/// Convert a variable to a `moore.global_variable` operation.
LogicalResult
Context::convertGlobalVariable(const slang::ast::VariableSymbol &var) {
  // Check if already converted (by pointer).
  if (globalVariables.count(&var))
    return success();

  auto loc = convertLocation(var.location);

  // Prefix the variable name with the surrounding namespace to create somewhat
  // sane names in the IR. Compute this BEFORE type conversion to detect
  // recursive calls during type conversion.
  SmallString<64> symName;
  guessNamespacePrefix(var.getParentScope()->asSymbol(), symName);
  symName += var.name;
  auto symNameAttr = builder.getStringAttr(symName);

  // Check if already converted or in progress (by name). This handles
  // recursive calls during type conversion where the same variable is
  // referenced from a method body being converted as part of class type
  // conversion.
  if (globalVariablesByName.count(symNameAttr)) {
    // Reuse the existing global variable for this pointer.
    globalVariables[&var] = globalVariablesByName[symNameAttr];
    return success();
  }

  // Pick an insertion point for this variable according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(var.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create the variable op with a placeholder type first. This allows
  // recursive type conversion (e.g., class types that reference this variable
  // in their method bodies) to find the variable in the map and avoid
  // creating duplicates.
  auto placeholderType = moore::IntType::getLogic(builder.getContext(), 1);
  auto varOp = moore::GlobalVariableOp::create(builder, loc, symNameAttr,
                                               placeholderType);
  orderedRootOps.insert({var.location, varOp});
  globalVariables.insert({&var, varOp});
  globalVariablesByName[symNameAttr] = varOp;

  // Now convert the actual type. This may trigger recursive calls via class
  // type conversion, but those calls will find varOp in the maps and return.
  auto type = convertType(var.getType());
  if (!type) {
    // Clean up on failure.
    globalVariables.erase(&var);
    globalVariablesByName.erase(symNameAttr);
    orderedRootOps.erase(var.location);
    varOp.erase();
    return failure();
  }

  // Update the type on the op to the actual type.
  varOp.setTypeAttr(TypeAttr::get(cast<moore::UnpackedType>(type)));

  // If the variable has an initializer expression, remember it for later such
  // that we can convert the initializers once we have seen all global
  // variables.
  if (var.getInitializer())
    globalVariableWorklist.push_back(&var);

  return success();
}

/// Convert a static class property to a `moore.global_variable` operation.
/// This allows on-demand conversion of static properties that may not have
/// been converted yet due to recursive class type conversion.
LogicalResult Context::convertStaticClassProperty(
    const slang::ast::ClassPropertySymbol &prop) {
  // Check if already converted (for on-demand conversion).
  if (globalVariables.count(&prop))
    return success();

  // Ensure the parent class is declared before generating the symbol name.
  // For parameterized class specializations, this ensures the class has a
  // unique symbol name (e.g., "uvm_config_db_1234") that distinguishes it
  // from other specializations. Without this, all specializations would share
  // the same static variable, causing type mismatches.
  const auto *parentScope = prop.getParentScope();
  if (parentScope) {
    if (auto *classType =
            parentScope->asSymbol().as_if<slang::ast::ClassType>()) {
      // This triggers class declaration if not already done, ensuring the
      // ClassDeclOp exists with its final symbol name.
      (void)convertClassDeclaration(*classType);
    }
  }

  // Check by fully qualified name (handles multiple specializations of
  // parameterized classes that share the same static member).
  auto symName = fullyQualifiedSymbolName(*this, prop);
  if (globalVariablesByName.count(symName)) {
    // Reuse the existing global variable for this symbol pointer.
    globalVariables[&prop] = globalVariablesByName[symName];
    return success();
  }

  // Check if we're already in the process of converting this property.
  // This can happen with recursive class type conversions where a property's
  // type triggers conversion of classes whose methods reference the property.
  // In this case, we'll defer to the caller to handle the incomplete state.
  if (staticPropertyInProgress.count(&prop))
    return failure();

  // Verify this is actually a static property.
  if (prop.lifetime != slang::ast::VariableLifetime::Static)
    return failure();

  // Mark this property as being converted to prevent infinite recursion.
  staticPropertyInProgress.insert(&prop);
  auto progressGuard =
      llvm::make_scope_exit([&] { staticPropertyInProgress.erase(&prop); });

  auto loc = convertLocation(prop.location);

  // Pick an insertion point for this variable at the module level.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(prop.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Determine the type of the property.
  auto ty = convertType(prop.getType());
  if (!ty)
    return failure();

  // Create the global variable op (symName already computed above).
  auto varOp = moore::GlobalVariableOp::create(builder, loc, symName,
                                               cast<moore::UnpackedType>(ty));
  orderedRootOps.insert({prop.location, varOp});
  globalVariables[&prop] = varOp;
  globalVariablesByName[symName] = varOp;

  // If the property has an initializer expression, remember it for later.
  if (prop.getInitializer())
    globalVariableWorklist.push_back(&prop);

  return success();
}

void Context::captureRef(Value ref) {
  if (!currentFunctionLowering || !ref)
    return;

  auto *lowering = currentFunctionLowering;
  if (!isa<moore::RefType>(ref.getType()))
    return;

  mlir::Region *defReg = ref.getParentRegion();
  if (defReg && lowering->op.getBody().isAncestor(defReg))
    return;

  if (lowering->captureIndex.count(ref))
    return;

  auto [it, inserted] =
      lowering->captureIndex.try_emplace(ref, lowering->captures.size());
  if (inserted)
    lowering->captures.push_back(ref);
}

//===----------------------------------------------------------------------===//
// Covergroup Conversion
//===----------------------------------------------------------------------===//

/// Helper to convert a BinsSelectExpr (binsof/intersect) to Moore IR.
/// This handles the recursive structure of bins select expressions used in
/// cross coverage bins.
static void convertBinsSelectExpr(
    const slang::ast::BinsSelectExpr &expr,
    const llvm::StringMap<mlir::FlatSymbolRefAttr> &coverpointSymbols,
    OpBuilder &builder, Location loc,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {

  switch (expr.kind) {
  case slang::ast::BinsSelectExprKind::Condition: {
    // binsof(coverpoint) intersect {values}
    auto &condExpr = static_cast<const slang::ast::ConditionBinsSelectExpr &>(expr);
    auto &target = condExpr.target;

    // Get the target symbol reference
    mlir::SymbolRefAttr targetRef;
    if (target.kind == slang::ast::SymbolKind::Coverpoint) {
      // Direct coverpoint reference
      auto it = coverpointSymbols.find(target.name);
      if (it != coverpointSymbols.end()) {
        targetRef = it->second;
      } else {
        // Create a reference using the target name
        targetRef = mlir::FlatSymbolRefAttr::get(
            builder.getContext(), target.name);
      }
    } else if (target.kind == slang::ast::SymbolKind::CoverageBin) {
      // Reference to a specific bin within a coverpoint (cp.bin_name)
      auto *parentScope = target.getParentScope();
      if (parentScope) {
        auto &parentSym = parentScope->asSymbol();
        if (parentSym.kind == slang::ast::SymbolKind::Coverpoint) {
          auto it = coverpointSymbols.find(parentSym.name);
          if (it != coverpointSymbols.end()) {
            // Create nested ref: @coverpoint::@bin
            targetRef = mlir::SymbolRefAttr::get(
                builder.getContext(), it->second.getValue(),
                {mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                              target.name)});
          }
        }
      }
    }

    if (!targetRef) {
      // Fallback: use the target name directly
      targetRef =
          mlir::FlatSymbolRefAttr::get(builder.getContext(), target.name);
    }

    // Collect intersect values if present
    mlir::ArrayAttr intersectValuesAttr;
    if (!condExpr.intersects.empty()) {
      SmallVector<mlir::Attribute> intersectValues;
      for (const auto *intersectExpr : condExpr.intersects) {
        // Check if it's a value range expression (e.g., [0:10])
        if (intersectExpr->kind == slang::ast::ExpressionKind::ValueRange) {
          auto &rangeExpr =
              intersectExpr->as<slang::ast::ValueRangeExpression>();
          auto leftVal = evaluateConstant(rangeExpr.left());
          auto rightVal = evaluateConstant(rangeExpr.right());
          if (leftVal.isInteger() && rightVal.isInteger()) {
            auto leftInt = leftVal.integer().as<int64_t>();
            auto rightInt = rightVal.integer().as<int64_t>();
            if (leftInt && rightInt) {
              // Expand the range into individual values
              for (int64_t v = leftInt.value(); v <= rightInt.value(); ++v) {
                intersectValues.push_back(builder.getI64IntegerAttr(v));
              }
            }
          }
        } else {
          // Single value
          auto result = evaluateConstant(*intersectExpr);
          if (result.isInteger()) {
            auto intVal = result.integer().as<int64_t>();
            if (intVal)
              intersectValues.push_back(
                  builder.getI64IntegerAttr(intVal.value()));
          }
        }
      }
      if (!intersectValues.empty())
        intersectValuesAttr = builder.getArrayAttr(intersectValues);
    }

    moore::BinsOfOp::create(builder, loc, targetRef, intersectValuesAttr);
    break;
  }

  case slang::ast::BinsSelectExprKind::Unary: {
    // !binsof(coverpoint) - negation
    // For now, we recursively convert the inner expression.
    // A more complete implementation would track the negation.
    auto &unaryExpr = static_cast<const slang::ast::UnaryBinsSelectExpr &>(expr);
    convertBinsSelectExpr(unaryExpr.expr, coverpointSymbols, builder, loc,
                          evaluateConstant);
    break;
  }

  case slang::ast::BinsSelectExprKind::Binary: {
    // binsof(a) && binsof(b) or binsof(a) || binsof(b)
    auto &binaryExpr = static_cast<const slang::ast::BinaryBinsSelectExpr &>(expr);
    convertBinsSelectExpr(binaryExpr.left, coverpointSymbols, builder, loc,
                          evaluateConstant);
    convertBinsSelectExpr(binaryExpr.right, coverpointSymbols, builder, loc,
                          evaluateConstant);
    break;
  }

  case slang::ast::BinsSelectExprKind::WithFilter: {
    // binsof(a) with (filter_expr)
    auto &filterExpr =
        static_cast<const slang::ast::BinSelectWithFilterExpr &>(expr);
    convertBinsSelectExpr(filterExpr.expr, coverpointSymbols, builder, loc,
                          evaluateConstant);
    break;
  }

  case slang::ast::BinsSelectExprKind::SetExpr:
  case slang::ast::BinsSelectExprKind::CrossId:
  case slang::ast::BinsSelectExprKind::Invalid:
    // These are less common or error cases - skip for now
    break;
  }
}

/// Helper struct to hold extracted coverage options.
struct CoverageOptions {
  std::optional<int64_t> weight;
  std::optional<int64_t> goal;
  std::optional<std::string> comment;
  bool perInstance = false;
  std::optional<int64_t> atLeast;
  std::optional<int64_t> autoBinMax;
  std::optional<int64_t> crossNumPrintMissing;
  std::optional<int64_t> crossAutoBinMax;
  bool strobe = false;
  bool detectOverlap = false;
  // Type options (for covergroups)
  std::optional<int64_t> typeWeight;
  std::optional<int64_t> typeGoal;
  std::optional<std::string> typeComment;
};

/// Extract coverage options from a span of CoverageOptionSetter.
static CoverageOptions extractCoverageOptions(
    std::span<const slang::ast::CoverageOptionSetter> options,
    const std::function<slang::ConstantValue(const slang::ast::Expression &)>
        &evaluateConstant) {
  CoverageOptions result;

  for (const auto &opt : options) {
    std::string_view name = opt.getName();
    bool isTypeOption = opt.isTypeOption();

    // Get the value expression (RHS of the assignment)
    const auto &expr = opt.getExpression();
    const slang::ast::Expression *valueExpr = &expr;
    if (expr.kind == slang::ast::ExpressionKind::Assignment) {
      if (const auto *assignExpr =
              expr.as_if<slang::ast::AssignmentExpression>())
        valueExpr = &assignExpr->right();
    }

    // Extract integer value if possible
    std::optional<int64_t> intValue;
    auto constVal = evaluateConstant(*valueExpr);
    if (constVal.isInteger()) {
      auto intVal = constVal.integer().as<int64_t>();
      if (intVal)
        intValue = intVal.value();
    }

    // Extract string value if possible
    std::optional<std::string> strValue;
    if (constVal.isString()) {
      strValue = std::string(constVal.str());
    }

    // Map option names to struct fields
    if (isTypeOption) {
      if (name == "weight" && intValue)
        result.typeWeight = *intValue;
      else if (name == "goal" && intValue)
        result.typeGoal = *intValue;
      else if (name == "comment" && strValue)
        result.typeComment = *strValue;
    } else {
      if (name == "weight" && intValue)
        result.weight = *intValue;
      else if (name == "goal" && intValue)
        result.goal = *intValue;
      else if (name == "comment" && strValue)
        result.comment = *strValue;
      else if (name == "per_instance" && intValue)
        result.perInstance = (*intValue != 0);
      else if (name == "at_least" && intValue)
        result.atLeast = *intValue;
      else if (name == "auto_bin_max" && intValue)
        result.autoBinMax = *intValue;
      else if (name == "cross_num_print_missing" && intValue)
        result.crossNumPrintMissing = *intValue;
      else if (name == "cross_auto_bin_max" && intValue)
        result.crossAutoBinMax = *intValue;
      else if (name == "strobe" && intValue)
        result.strobe = (*intValue != 0);
      else if (name == "detect_overlap" && intValue)
        result.detectOverlap = (*intValue != 0);
    }
  }

  return result;
}

LogicalResult
Context::convertCovergroup(const slang::ast::CovergroupType &covergroup) {
  // Check if already converted.
  if (covergroups.count(&covergroup))
    return success();

  auto loc = convertLocation(covergroup.location);

  // Pick an insertion point according to the source file location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(covergroup.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Get the sampling event string, if present.
  mlir::StringAttr samplingEventAttr;
  if (auto *event = covergroup.getCoverageEvent()) {
    // Convert the timing control to a descriptive string.
    std::string eventStr;
    llvm::raw_string_ostream os(eventStr);
    os << "@(";
    if (event->kind == slang::ast::TimingControlKind::SignalEvent) {
      auto &signalEvent =
          event->as<slang::ast::SignalEventControl>();
      if (signalEvent.edge == slang::ast::EdgeKind::PosEdge)
        os << "posedge ";
      else if (signalEvent.edge == slang::ast::EdgeKind::NegEdge)
        os << "negedge ";
      // Get the expression text for the signal
      if (auto *nameExpr =
              signalEvent.expr.as_if<slang::ast::NamedValueExpression>()) {
        os << nameExpr->symbol.name;
      } else {
        os << "signal";
      }
    } else if (event->kind == slang::ast::TimingControlKind::EventList) {
      os << "event_list";
    } else {
      os << "event";
    }
    os << ")";
    samplingEventAttr = builder.getStringAttr(eventStr);
  }

  // Get the covergroup name. For class-scoped covergroups, slang sets the
  // CovergroupType name to empty and creates a ClassPropertySymbol with the
  // actual name. In this case, we need to get the name from the syntax.
  std::string_view cgName = covergroup.name;
  if (cgName.empty()) {
    if (const auto *syntax = covergroup.getSyntax()) {
      if (const auto *cgSyntax =
              syntax->as_if<slang::syntax::CovergroupDeclarationSyntax>()) {
        cgName = cgSyntax->name.valueText();
      }
    }
  }

  // Extract covergroup-level options from the body.
  const auto &cgBody = covergroup.getBody();
  auto cgOptions = extractCoverageOptions(
      cgBody.options, [this](const slang::ast::Expression &expr) {
        return evaluateConstant(expr);
      });

  // Create the covergroup declaration op with all options.
  auto covergroupOp = moore::CovergroupDeclOp::create(
      builder, loc, cgName, samplingEventAttr,
      cgOptions.weight ? builder.getI64IntegerAttr(*cgOptions.weight) : nullptr,
      cgOptions.goal ? builder.getI64IntegerAttr(*cgOptions.goal) : nullptr,
      cgOptions.comment ? builder.getStringAttr(*cgOptions.comment) : nullptr,
      cgOptions.perInstance,
      cgOptions.atLeast ? builder.getI64IntegerAttr(*cgOptions.atLeast)
                        : nullptr,
      cgOptions.crossNumPrintMissing
          ? builder.getI64IntegerAttr(*cgOptions.crossNumPrintMissing)
          : nullptr,
      cgOptions.strobe,
      cgOptions.typeWeight ? builder.getI64IntegerAttr(*cgOptions.typeWeight)
                           : nullptr,
      cgOptions.typeGoal ? builder.getI64IntegerAttr(*cgOptions.typeGoal)
                         : nullptr,
      cgOptions.typeComment ? builder.getStringAttr(*cgOptions.typeComment)
                            : nullptr);
  orderedRootOps.insert({covergroup.location, covergroupOp});
  symbolTable.insert(covergroupOp);

  // Store the lowering info.
  auto lowering = std::make_unique<CovergroupLowering>();
  lowering->op = covergroupOp;
  covergroups[&covergroup] = std::move(lowering);

  // Create the covergroup body block.
  auto &body = covergroupOp.getBody();
  body.emplaceBlock();

  // Build a map of coverpoint names to their symbol refs for cross references.
  llvm::StringMap<mlir::FlatSymbolRefAttr> coverpointSymbols;

  // Convert coverpoints and crosses from the covergroup body.
  OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body.front());

  for (const auto &member : cgBody.members()) {
    if (auto *cp = member.as_if<slang::ast::CoverpointSymbol>()) {
      auto cpLoc = convertLocation(cp->location);

      // Get the coverpoint expression type.
      auto exprType = convertType(cp->declaredType.getType());
      if (!exprType)
        return failure();

      // Determine the coverpoint name. Slang provides a name even for unlabeled
      // coverpoints (using the expression text). For labeled coverpoints, we use
      // the label directly. For unlabeled ones, we append "_cp" to make it clear
      // this is an auto-generated coverpoint name.
      std::string cpName;
      if (!cp->name.empty()) {
        // Check if this is a user-provided label or auto-generated name.
        // User labels typically match the name exactly, while auto-generated
        // names for unlabeled coverpoints may need the _cp suffix.
        // For simplicity, we check if there's an explicit label by looking
        // at the expression - if the name matches the expression variable name,
        // it's auto-generated.
        const auto &coverExpr = cp->getCoverageExpr();
        if (const auto *nameExpr =
                coverExpr.as_if<slang::ast::NamedValueExpression>()) {
          if (cp->name == nameExpr->symbol.name) {
            // Auto-generated name from expression, append _cp
            cpName = std::string(cp->name) + "_cp";
          } else {
            // User-provided label
            cpName = std::string(cp->name);
          }
        } else {
          // Non-simple expression, use the name as-is
          cpName = std::string(cp->name);
        }
      } else {
        cpName = "cp";
      }

      // Extract all coverpoint options.
      auto cpOptions = extractCoverageOptions(
          cp->options, [this](const slang::ast::Expression &expr) {
            return evaluateConstant(expr);
          });

      // Get the iff condition string, if present.
      mlir::StringAttr iffAttr;
      if (const auto *iffExpr = cp->getIffExpr()) {
        // Convert the iff expression to a string representation.
        if (iffExpr->syntax) {
          iffAttr = builder.getStringAttr(iffExpr->syntax->toString());
        }
      }

      // Create the coverpoint declaration op with all options.
      auto cpOp = moore::CoverpointDeclOp::create(
          builder, cpLoc, builder.getStringAttr(cpName),
          mlir::TypeAttr::get(exprType),
          iffAttr,
          cpOptions.weight ? builder.getI64IntegerAttr(*cpOptions.weight)
                           : nullptr,
          cpOptions.goal ? builder.getI64IntegerAttr(*cpOptions.goal) : nullptr,
          cpOptions.comment ? builder.getStringAttr(*cpOptions.comment)
                            : nullptr,
          cpOptions.atLeast ? builder.getI64IntegerAttr(*cpOptions.atLeast)
                            : nullptr,
          cpOptions.autoBinMax
              ? builder.getI64IntegerAttr(*cpOptions.autoBinMax)
              : nullptr,
          cpOptions.detectOverlap ? builder.getUnitAttr() : nullptr,
          cpOptions.crossAutoBinMax
              ? builder.getI64IntegerAttr(*cpOptions.crossAutoBinMax)
              : nullptr);

      // Create the coverpoint body block for bins.
      auto &cpBody = cpOp.getBody();
      cpBody.emplaceBlock();

      // Convert bins within this coverpoint.
      {
        OpBuilder::InsertionGuard binGuard(builder);
        builder.setInsertionPointToStart(&cpBody.front());

        for (const auto &cpMember : cp->members()) {
          if (auto *bin = cpMember.as_if<slang::ast::CoverageBinSymbol>()) {
            auto binLoc = convertLocation(bin->location);

            // Determine bin kind
            moore::CoverageBinKind binKind;
            switch (bin->binsKind) {
            case slang::ast::CoverageBinSymbol::Bins:
              binKind = moore::CoverageBinKind::Bins;
              break;
            case slang::ast::CoverageBinSymbol::IllegalBins:
              binKind = moore::CoverageBinKind::IllegalBins;
              break;
            case slang::ast::CoverageBinSymbol::IgnoreBins:
              binKind = moore::CoverageBinKind::IgnoreBins;
              break;
            }

            // Collect bin values as integer attributes.
            SmallVector<mlir::Attribute> valueAttrs;
            auto binValues = bin->getValues();
            for (const auto *valExpr : binValues) {
              // Try to evaluate the expression as a constant.
              auto result = evaluateConstant(*valExpr);
              if (result.isInteger()) {
                auto intVal = result.integer().as<int64_t>();
                if (intVal)
                  valueAttrs.push_back(
                      builder.getI64IntegerAttr(intVal.value()));
              }
            }

            auto valuesAttr =
                valueAttrs.empty() ? nullptr : builder.getArrayAttr(valueAttrs);

            // Handle transition bins (bins x = (a => b => c)).
            // TransList is a span of TransSet, where each TransSet is a span
            // of TransRangeList. Each TransRangeList has items (values) and
            // optional repeat information.
            SmallVector<mlir::Attribute> transitionsAttr;
            auto transList = bin->getTransList();
            for (const auto &transSet : transList) {
              // Each TransSet represents one alternative transition sequence
              // e.g., in (a => b), (c => d), we have two TransSets
              SmallVector<mlir::Attribute> sequenceAttr;
              for (const auto &rangeList : transSet) {
                // Each TransRangeList represents a step in the sequence
                // with its values and optional repeat
                for (const auto *itemExpr : rangeList.items) {
                  auto result = evaluateConstant(*itemExpr);
                  if (result.isInteger()) {
                    auto intVal = result.integer().as<int64_t>();
                    if (intVal) {
                      // Create [value, repeatKind, repeatFrom, repeatTo]
                      int64_t repeatKind = 0;
                      int64_t repeatFrom = 0;
                      int64_t repeatTo = 0;

                      switch (rangeList.repeatKind) {
                      case slang::ast::CoverageBinSymbol::TransRangeList::None:
                        repeatKind = 0;
                        break;
                      case slang::ast::CoverageBinSymbol::TransRangeList::
                          Consecutive:
                        repeatKind = 1;
                        break;
                      case slang::ast::CoverageBinSymbol::TransRangeList::
                          Nonconsecutive:
                        repeatKind = 2;
                        break;
                      case slang::ast::CoverageBinSymbol::TransRangeList::GoTo:
                        repeatKind = 3;
                        break;
                      }

                      // Get repeat range if present
                      if (rangeList.repeatFrom) {
                        auto fromResult = evaluateConstant(*rangeList.repeatFrom);
                        if (fromResult.isInteger()) {
                          auto fromVal = fromResult.integer().as<int64_t>();
                          if (fromVal)
                            repeatFrom = fromVal.value();
                        }
                      }
                      if (rangeList.repeatTo) {
                        auto toResult = evaluateConstant(*rangeList.repeatTo);
                        if (toResult.isInteger()) {
                          auto toVal = toResult.integer().as<int64_t>();
                          if (toVal)
                            repeatTo = toVal.value();
                        }
                      } else if (rangeList.repeatFrom) {
                        // If only repeatFrom is set, repeatTo equals repeatFrom
                        repeatTo = repeatFrom;
                      }

                      SmallVector<mlir::Attribute> transItem = {
                          builder.getI64IntegerAttr(intVal.value()),
                          builder.getI64IntegerAttr(repeatKind),
                          builder.getI64IntegerAttr(repeatFrom),
                          builder.getI64IntegerAttr(repeatTo)};
                      sequenceAttr.push_back(builder.getArrayAttr(transItem));
                    }
                  }
                }
              }
              if (!sequenceAttr.empty()) {
                transitionsAttr.push_back(builder.getArrayAttr(sequenceAttr));
              }
            }

            auto transAttr =
                transitionsAttr.empty() ? nullptr
                                        : builder.getArrayAttr(transitionsAttr);

            // Handle array bins (bins x[] or bins x[N]).
            // isArray is true for both syntax forms.
            // getNumberOfBinsExpr() returns the N expression for bins x[N].
            std::optional<int64_t> numBins;
            if (bin->isArray) {
              if (const auto *numBinsExpr = bin->getNumberOfBinsExpr()) {
                auto result = evaluateConstant(*numBinsExpr);
                if (result.isInteger()) {
                  auto intVal = result.integer().as<int64_t>();
                  if (intVal)
                    numBins = intVal.value();
                }
              }
            }

            moore::CoverageBinDeclOp::create(
                builder, binLoc, builder.getStringAttr(bin->name), binKind,
                bin->isWildcard, bin->isDefault, bin->isArray,
                bin->isDefaultSequence,
                numBins ? builder.getI64IntegerAttr(*numBins) : nullptr,
                valuesAttr, transAttr);
          }
        }
      }

      // Store the symbol ref for cross references using the original slang name
      // as the key (since that's what CoverCrossSymbol::targets uses).
      coverpointSymbols[cp->name] =
          mlir::FlatSymbolRefAttr::get(cpOp.getSymNameAttr());
    } else if (auto *cross = member.as_if<slang::ast::CoverCrossSymbol>()) {
      auto crossLoc = convertLocation(cross->location);

      // Collect the target coverpoint references.
      SmallVector<mlir::Attribute> targets;
      for (const auto *target : cross->targets) {
        auto it = coverpointSymbols.find(target->name);
        if (it != coverpointSymbols.end()) {
          targets.push_back(it->second);
        }
      }

      // Determine the cross name. If no explicit label is provided, generate
      // a name from the target coverpoint names joined with "_x_".
      std::string crossName;
      if (!cross->name.empty()) {
        crossName = std::string(cross->name);
      } else {
        // Generate name like "addr_x_cmd" from the target coverpoint names.
        llvm::raw_string_ostream os(crossName);
        bool first = true;
        for (const auto *target : cross->targets) {
          if (!first)
            os << "_x_";
          first = false;
          os << target->name;
        }
      }

      // Extract cross coverage options.
      auto crossOptions = extractCoverageOptions(
          cross->options, [this](const slang::ast::Expression &expr) {
            return evaluateConstant(expr);
          });

      // Create the cross coverage declaration op with all options.
      auto crossOp = moore::CoverCrossDeclOp::create(
          builder, crossLoc, crossName, builder.getArrayAttr(targets),
          crossOptions.weight ? builder.getI64IntegerAttr(*crossOptions.weight)
                              : nullptr,
          crossOptions.goal ? builder.getI64IntegerAttr(*crossOptions.goal)
                            : nullptr,
          crossOptions.comment ? builder.getStringAttr(*crossOptions.comment)
                               : nullptr,
          crossOptions.atLeast ? builder.getI64IntegerAttr(*crossOptions.atLeast)
                               : nullptr,
          crossOptions.crossNumPrintMissing
              ? builder.getI64IntegerAttr(*crossOptions.crossNumPrintMissing)
              : nullptr,
          crossOptions.crossAutoBinMax
              ? builder.getI64IntegerAttr(*crossOptions.crossAutoBinMax)
              : nullptr);

      // Create the cross body block for bins.
      auto &crossBody = crossOp.getBody();
      crossBody.emplaceBlock();

      // Convert cross bins within this cross coverage.
      // Cross bins are stored in CoverCrossBodySymbol which is a member.
      {
        OpBuilder::InsertionGuard crossBinGuard(builder);
        builder.setInsertionPointToStart(&crossBody.front());

        for (const auto &crossMember : cross->members()) {
          // The CoverCrossBodySymbol contains the actual bins
          if (auto *body =
                  crossMember.as_if<slang::ast::CoverCrossBodySymbol>()) {
            for (const auto &bodyMember : body->members()) {
              if (auto *bin =
                      bodyMember.as_if<slang::ast::CoverageBinSymbol>()) {
                auto binLoc = convertLocation(bin->location);

                // Determine bin kind
                moore::CoverageBinKind binKind;
                switch (bin->binsKind) {
                case slang::ast::CoverageBinSymbol::Bins:
                  binKind = moore::CoverageBinKind::Bins;
                  break;
                case slang::ast::CoverageBinSymbol::IllegalBins:
                  binKind = moore::CoverageBinKind::IllegalBins;
                  break;
                case slang::ast::CoverageBinSymbol::IgnoreBins:
                  binKind = moore::CoverageBinKind::IgnoreBins;
                  break;
                }

                // Create the cross bin declaration
                auto crossBinOp = moore::CrossBinDeclOp::create(
                    builder, binLoc, builder.getStringAttr(bin->name), binKind);

                // Create body block for binsof expressions
                auto &binBody = crossBinOp.getBody();
                binBody.emplaceBlock();

                // Convert binsof/intersect expressions
                if (auto *selectExpr = bin->getCrossSelectExpr()) {
                  OpBuilder::InsertionGuard binsofGuard(builder);
                  builder.setInsertionPointToStart(&binBody.front());
                  // Create lambda to capture this context for evaluateConstant
                  auto evalConst =
                      [this](const slang::ast::Expression &e)
                          -> slang::ConstantValue {
                    return this->evaluateConstant(e);
                  };
                  convertBinsSelectExpr(*selectExpr, coverpointSymbols, builder,
                                        binLoc, evalConst);
                }
              }
            }
          }
        }
      }
    }
  }

  return success();
}

/// Construct a fully qualified class name containing the instance hierarchy
/// and the class name formatted as H1::H2::@C
mlir::StringAttr circt::ImportVerilog::fullyQualifiedClassName(
    Context &ctx, const slang::ast::Type &ty) {
  SmallString<64> name;
  SmallVector<llvm::StringRef, 8> parts;

  const slang::ast::Scope *scope = ty.getParentScope();
  while (scope) {
    const auto &sym = scope->asSymbol();
    switch (sym.kind) {
    case slang::ast::SymbolKind::Root:
      scope = nullptr; // stop at $root
      continue;
    case slang::ast::SymbolKind::InstanceBody:
    case slang::ast::SymbolKind::Instance:
    case slang::ast::SymbolKind::Package:
    case slang::ast::SymbolKind::ClassType:
      if (!sym.name.empty())
        parts.push_back(sym.name); // keep packages + outer classes
      break;
    default:
      break;
    }
    scope = sym.getParentScope();
  }

  for (auto p : llvm::reverse(parts)) {
    name += p;
    name += "::";
  }
  name += ty.name; // class's own name
  return mlir::StringAttr::get(ctx.getContext(), name);
}

/// Construct a fully qualified symbol name for generic class definitions
mlir::StringAttr circt::ImportVerilog::fullyQualifiedSymbolName(
    Context &ctx, const slang::ast::Symbol &sym) {
  SmallString<64> name;
  SmallVector<llvm::StringRef, 8> parts;

  const slang::ast::Scope *scope = sym.getParentScope();
  while (scope) {
    const auto &parentSym = scope->asSymbol();
    switch (parentSym.kind) {
    case slang::ast::SymbolKind::Root:
      scope = nullptr; // stop at $root
      continue;
    case slang::ast::SymbolKind::InstanceBody:
    case slang::ast::SymbolKind::Instance:
    case slang::ast::SymbolKind::Package:
      if (!parentSym.name.empty())
        parts.push_back(parentSym.name);
      break;
    case slang::ast::SymbolKind::ClassType: {
      // For parameterized class specializations, use the actual ClassDeclOp
      // symbol name (which may have been renamed during insertion, e.g.,
      // "uvm_typed_callbacks_2768") instead of the generic class name.
      if (auto *classType = parentSym.as_if<slang::ast::ClassType>()) {
        auto it = ctx.classes.find(classType);
        if (it != ctx.classes.end() && it->second && it->second->op) {
          parts.push_back(it->second->op.getSymName());
          break;
        }
      }
      // Fallback to the symbol name if ClassDeclOp not available yet.
      if (!parentSym.name.empty())
        parts.push_back(parentSym.name);
      break;
    }
    default:
      break;
    }
    scope = parentSym.getParentScope();
  }

  for (auto p : llvm::reverse(parts)) {
    name += p;
    name += "::";
  }
  name += sym.name; // symbol's own name
  return mlir::StringAttr::get(ctx.getContext(), name);
}

//===----------------------------------------------------------------------===//
// Context::convertConstraint - Convert slang constraints to MLIR
//===----------------------------------------------------------------------===//

LogicalResult Context::convertConstraint(const slang::ast::Constraint &constraint,
                                         Location loc) {
  switch (constraint.kind) {
  case slang::ast::ConstraintKind::List: {
    const auto &list = constraint.as<slang::ast::ConstraintList>();
    for (const auto *item : list.list) {
      if (failed(convertConstraint(*item, loc)))
        return failure();
    }
    return success();
  }
  case slang::ast::ConstraintKind::Expression: {
    const auto &exprConstraint =
        constraint.as<slang::ast::ExpressionConstraint>();
    auto value = convertRvalueExpression(exprConstraint.expr);
    if (!value)
      return failure();
    // Convert to boolean if needed
    value = convertToBool(value);
    if (!value)
      return failure();
    // Create the constraint.expr operation
    moore::ConstraintExprOp::create(builder, loc, value,
                                    exprConstraint.isSoft);
    return success();
  }
  case slang::ast::ConstraintKind::Implication: {
    const auto &impl = constraint.as<slang::ast::ImplicationConstraint>();
    auto predicate = convertRvalueExpression(impl.predicate);
    if (!predicate)
      return failure();
    predicate = convertToBool(predicate);
    if (!predicate)
      return failure();
    // Create implication op with body
    auto implOp = moore::ConstraintImplicationOp::create(builder, loc,
                                                         predicate);
    implOp.getConsequent().emplaceBlock();
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPointToEnd(&implOp.getConsequent().front());
    return convertConstraint(impl.body, loc);
  }
  case slang::ast::ConstraintKind::Conditional: {
    const auto &cond = constraint.as<slang::ast::ConditionalConstraint>();
    auto predicate = convertRvalueExpression(cond.predicate);
    if (!predicate)
      return failure();
    predicate = convertToBool(predicate);
    if (!predicate)
      return failure();
    // Create if-else op
    auto ifElseOp = moore::ConstraintIfElseOp::create(builder, loc,
                                                      predicate);
    // Then region
    ifElseOp.getThenRegion().emplaceBlock();
    {
      OpBuilder::InsertionGuard ig(builder);
      builder.setInsertionPointToEnd(&ifElseOp.getThenRegion().front());
      if (failed(convertConstraint(cond.ifBody, loc)))
        return failure();
    }
    // Else region (optional)
    if (cond.elseBody) {
      ifElseOp.getElseRegion().emplaceBlock();
      OpBuilder::InsertionGuard ig(builder);
      builder.setInsertionPointToEnd(&ifElseOp.getElseRegion().front());
      if (failed(convertConstraint(*cond.elseBody, loc)))
        return failure();
    }
    return success();
  }
  case slang::ast::ConstraintKind::Uniqueness: {
    const auto &unique = constraint.as<slang::ast::UniquenessConstraint>();
    SmallVector<Value> items;
    for (const auto *item : unique.items) {
      auto value = convertRvalueExpression(*item);
      if (!value)
        return failure();
      items.push_back(value);
    }
    moore::ConstraintUniqueOp::create(builder, loc, items);
    return success();
  }
  case slang::ast::ConstraintKind::Foreach: {
    const auto &foreachCons = constraint.as<slang::ast::ForeachConstraint>();

    // Convert the array expression
    auto arrayValue = convertRvalueExpression(foreachCons.arrayRef);
    if (!arrayValue)
      return failure();

    // Create the foreach constraint op
    auto foreachOp =
        moore::ConstraintForeachOp::create(builder, loc, arrayValue);

    // Create the body block with arguments for each loop variable
    Block *bodyBlock = &foreachOp.getBody().emplaceBlock();

    // Collect the loop variables and their types
    SmallVector<const slang::ast::IteratorSymbol *, 4> loopVars;
    SmallVector<Type, 4> loopVarTypes;

    for (const auto &loopDim : foreachCons.loopDims) {
      if (loopDim.loopVar) {
        auto varType = convertType(*loopDim.loopVar->getDeclaredType());
        if (!varType)
          return failure();
        loopVars.push_back(loopDim.loopVar);
        loopVarTypes.push_back(varType);
      }
    }

    // Add block arguments for each loop variable
    for (size_t i = 0; i < loopVars.size(); ++i) {
      bodyBlock->addArgument(loopVarTypes[i], loc);
    }

    // Set insertion point inside the body and register loop variables
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPointToStart(bodyBlock);

    // Create a scope for the loop variables
    Context::ValueSymbolScope loopScope(valueSymbols);

    // Register each loop variable in the symbol table
    for (size_t i = 0; i < loopVars.size(); ++i) {
      valueSymbols.insert(loopVars[i], bodyBlock->getArgument(i));
    }

    // Convert the body constraint
    return convertConstraint(foreachCons.body, loc);
  }
  case slang::ast::ConstraintKind::SolveBefore: {
    const auto &solveBefore =
        constraint.as<slang::ast::SolveBeforeConstraint>();

    // Helper lambda to extract variable name from expression
    auto extractVarName =
        [&](const slang::ast::Expression *expr) -> std::optional<StringRef> {
      // Handle NamedValueExpression - most common case for random variables
      if (expr->kind == slang::ast::ExpressionKind::NamedValue) {
        const auto &namedExpr =
            expr->as<slang::ast::NamedValueExpression>();
        return namedExpr.symbol.name;
      }
      // Handle HierarchicalValueExpression for hierarchical references
      if (expr->kind == slang::ast::ExpressionKind::HierarchicalValue) {
        const auto &hierExpr =
            expr->as<slang::ast::HierarchicalValueExpression>();
        return hierExpr.symbol.name;
      }
      return std::nullopt;
    };

    // Collect 'before' (solve) variables
    SmallVector<Attribute> beforeRefs;
    for (const auto *item : solveBefore.solve) {
      if (auto varName = extractVarName(item)) {
        beforeRefs.push_back(
            mlir::FlatSymbolRefAttr::get(getContext(), *varName));
      } else {
        mlir::emitWarning(loc)
            << "solve-before: could not extract variable name from expression";
      }
    }

    // Collect 'after' variables
    SmallVector<Attribute> afterRefs;
    for (const auto *item : solveBefore.after) {
      if (auto varName = extractVarName(item)) {
        afterRefs.push_back(
            mlir::FlatSymbolRefAttr::get(getContext(), *varName));
      } else {
        mlir::emitWarning(loc)
            << "solve-before: could not extract variable name from expression";
      }
    }

    // Only create the op if we have at least one variable in each list
    if (!beforeRefs.empty() && !afterRefs.empty()) {
      moore::ConstraintSolveBeforeOp::create(
          builder, loc, builder.getArrayAttr(beforeRefs),
          builder.getArrayAttr(afterRefs));
    }
    return success();
  }
  case slang::ast::ConstraintKind::DisableSoft: {
    const auto &disableSoft =
        constraint.as<slang::ast::DisableSoftConstraint>();
    auto targetValue = convertRvalueExpression(disableSoft.target);
    if (!targetValue)
      return failure();
    moore::ConstraintDisableSoftOp::create(builder, loc, targetValue);
    return success();
  }
  case slang::ast::ConstraintKind::Invalid:
    return failure();
  }
  llvm_unreachable("unknown constraint kind");
}
