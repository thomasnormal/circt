//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/symbols/ClassSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
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

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    using slang::ast::ArgumentDirection;
    using slang::ast::AssignmentExpression;
    using slang::ast::MultiPortSymbol;
    using slang::ast::PortSymbol;

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

      // Create the interface instance op
      moore::InterfaceInstanceOp::create(
          builder, loc, vifRefType,
          builder.getStringAttr(Twine(blockNamePrefix) + instNode.name),
          ifaceRef);

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
    SmallDenseMap<const PortSymbol *, Value> portValues;
    portValues.reserve(moduleType.getNumPorts());

    for (const auto *con : instNode.getPortConnections()) {
      const auto *expr = con->getExpression();

      // Handle unconnected behavior. The expression is null if it have no
      // connection for the port.
      if (!expr) {
        auto *port = con->port.as_if<PortSymbol>();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(port->getSyntax()))
          port = existingPort;

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
            portValues.insert({port, readOp});
          } else if (const auto *var =
                         port->internalSymbol
                             ->as_if<slang::ast::VariableSymbol>()) {
            auto varOp = moore::VariableOp::create(
                builder, loc, refType,
                StringAttr::get(builder.getContext(), var->name), nullptr);
            auto readOp = moore::ReadOp::create(builder, loc, varOp);
            portValues.insert({port, readOp});
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

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for
      // output and inout ports.
      if (const auto *assign = expr->as_if<AssignmentExpression>())
        expr = &assign->left();

      // Regular ports lower the connected expression to an lvalue or rvalue and
      // either attach it to the instance as an operand (for input, inout, and
      // ref ports), or assign an instance output to it (for output ports).
      if (auto *port = con->port.as_if<PortSymbol>()) {
        // Convert as rvalue for inputs, lvalue for all others.
        auto value = (port->direction == ArgumentDirection::In)
                         ? context.convertRvalueExpression(*expr)
                         : context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(con->port.getSyntax()))
          port = existingPort;
        portValues.insert({port, value});
        continue;
      }

      // Multi-ports lower the connected expression to an lvalue and then slice
      // it up into multiple sub-values, one for each of the ports in the
      // multi-port.
      if (const auto *multiPort = con->port.as_if<MultiPortSymbol>()) {
        // Convert as lvalue.
        auto value = context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        unsigned offset = 0;
        for (const auto *port : llvm::reverse(multiPort->ports)) {
          if (auto *existingPort = moduleLowering->portsBySyntaxNode.lookup(
                  con->port.getSyntax()))
            port = existingPort;
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
          portValues.insert({port, slice});
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
      auto value = portValues.lookup(&port.ast);
      if (port.ast.direction == ArgumentDirection::Out)
        outputValues.push_back(value);
      else
        inputValues.push_back(value);
    }

    // Insert conversions for input ports.
    for (auto [value, type] :
         llvm::zip(inputValues, moduleType.getInputTypes()))
      // TODO: This should honor signedness in the conversion.
      value = context.materializeConversion(type, value, false, value.getLoc());

    // Here we use the hierarchical value recorded in `Context::valueSymbols`.
    // Then we pass it as the input port with the ref<T> type of the instance.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (auto hierValue = context.valueSymbols.lookup(hierPath.valueSym);
          hierPath.hierName && hierPath.direction == ArgumentDirection::In)
        inputValues.push_back(hierValue);

    // Create the instance op itself.
    auto inputNames = builder.getArrayAttr(moduleType.getInputNames());
    auto outputNames = builder.getArrayAttr(moduleType.getOutputNames());
    auto inst = moore::InstanceOp::create(
        builder, loc, moduleType.getOutputTypes(),
        builder.getStringAttr(Twine(blockNamePrefix) + instNode.name),
        FlatSymbolRefAttr::get(module.getSymNameAttr()), inputValues,
        inputNames, outputNames);

    // Record instance's results generated by hierarchical names.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (hierPath.idx && hierPath.direction == ArgumentDirection::Out)
        context.valueSymbols.insert(hierPath.valueSym,
                                    inst->getResult(*hierPath.idx));

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
      initial = context.convertRvalueExpression(*init, loweredType);
      if (!initial)
        return failure();
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
    if (netkind == moore::NetKind::Interconnect ||
        netkind == moore::NetKind::UserDefined ||
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
    const auto &expr =
        assignNode.getAssignment().as<slang::ast::AssignmentExpression>();
    auto lhs = context.convertLvalueExpression(expr.left());
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

    auto rhs = context.convertRvalueExpression(expr.right(), lhsNestedType);
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

    // Otherwise this is a regular assignment.
    moore::ContinuousAssignOp::create(builder, loc, lhs, rhs);
    return success();
  }

  // Handle procedures.
  LogicalResult convertProcedure(moore::ProcedureKind kind,
                                 const slang::ast::Statement &body) {
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
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // First only to visit the whole AST to collect the hierarchical names without
  // any operation creating.
  for (auto *inst : root.topInstances)
    if (failed(traverseInstanceBody(inst->body)))
      return failure();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(RootVisitor(*this, loc))))
        return failure();
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
  using slang::ast::MultiPortSymbol;
  using slang::ast::ParameterSymbol;
  using slang::ast::PortSymbol;
  using slang::ast::TypeParameterSymbol;

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  auto parameters = module->getParameters();
  bool hasModuleSame = false;
  // If there is already exist a module that has the same name with this
  // module ,has the same parent scope and has the same parameters we can
  // define this module is a duplicate module
  for (auto const &existingModule : modules) {
    if (module->getDeclaringDefinition() ==
        existingModule.getFirst()->getDeclaringDefinition()) {
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
      lowering.ports.push_back({port, portLoc, arg});
      return success();
    };

    if (const auto *port = symbol->as_if<PortSymbol>()) {
      if (failed(handlePort(*port)))
        return {};
    } else if (const auto *multiPort = symbol->as_if<MultiPortSymbol>()) {
      for (auto *port : multiPort->ports)
        if (failed(handlePort(*port)))
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
    lowering.portsBySyntaxNode.insert({port.ast.getSyntax(), &port.ast});

  return &lowering;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto &lowering = *modules[module];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(lowering.op.getBody());

  ValueSymbolScope scope(valueSymbols);

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Collect downward hierarchical names. Such as,
  // module SubA; int x = Top.y; endmodule. The "Top" module is the parent of
  // the "SubA", so "Top.y" is the downward hierarchical name.
  for (auto &hierPath : hierPaths[module])
    if (hierPath.direction == slang::ast::ArgumentDirection::In && hierPath.idx)
      valueSymbols.insert(hierPath.valueSym,
                          lowering.op.getBody()->getArgument(*hierPath.idx));

  // Convert the body of the module.
  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(ModuleVisitor(*this, loc))))
      return failure();
  }

  // Create additional ops to drive input port values onto the corresponding
  // internal variables and nets, and to collect output port values for the
  // terminator.
  SmallVector<Value> outputs;
  for (auto &port : lowering.ports) {
    Value value;
    if (auto *expr = port.ast.getInternalExpr()) {
      value = convertLvalueExpression(*expr);
    } else if (port.ast.internalSymbol) {
      if (const auto *sym =
              port.ast.internalSymbol->as_if<slang::ast::ValueSymbol>())
        value = valueSymbols.lookup(sym);
    }
    if (!value)
      return mlir::emitError(port.loc, "unsupported port: `")
             << port.ast.name
             << "` does not map to an internal symbol or expression";

    // Collect output port values to be returned in the terminator.
    if (port.ast.direction == slang::ast::ArgumentDirection::Out) {
      if (isa<moore::RefType>(value.getType()))
        value = moore::ReadOp::create(builder, value.getLoc(), value);
      outputs.push_back(value);
      continue;
    }

    // Assign the value coming in through the port to the internal net or symbol
    // of that port.
    Value portArg = port.arg;
    if (port.ast.direction != slang::ast::ArgumentDirection::In)
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
  return success();
}

/// Convert a package and its contents.
LogicalResult
Context::convertPackage(const slang::ast::PackageSymbol &package) {
  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = package.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(intoModuleOp.getBody());
  ValueSymbolScope scope(valueSymbols);

  // Two-pass conversion:
  // Pass 1: Convert global variables first so they're available for classes.
  // Note: Some variables may not appear in members() due to how Slang handles
  // symbols declared after forward typedefs. Those are handled via on-demand
  // conversion in Expressions.cpp when they're first referenced.
  for (auto &member : package.members()) {
    if (member.kind == slang::ast::SymbolKind::Variable) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(PackageVisitor(*this, loc))))
        return failure();
    }
  }

  // Pass 2: Convert remaining members (classes, functions, etc.)
  for (auto &member : package.members()) {
    if (member.kind != slang::ast::SymbolKind::Variable) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(PackageVisitor(*this, loc))))
        return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interface Conversion
//===----------------------------------------------------------------------===//

/// Convert an interface declaration header. Creates the moore.interface op
/// and schedules the body for conversion.
InterfaceLowering *
Context::convertInterfaceHeader(const slang::ast::InstanceBodySymbol *iface) {
  auto &slot = interfaces[iface];
  if (slot)
    return slot.get();

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
      builder, loc, iface->getDefinition().name);
  orderedRootOps.insert(it, {iface->location, ifaceOp});

  // Create the body block for the interface
  ifaceOp.getBody().emplaceBlock();

  lowering.op = ifaceOp;

  // Add the interface to the symbol table
  symbolTable.insert(ifaceOp);

  // Schedule the body to be lowered
  interfaceWorklist.push(iface);

  return &lowering;
}

/// Convert an interface body - signals, modports, and ports.
LogicalResult
Context::convertInterfaceBody(const slang::ast::InstanceBodySymbol *iface) {
  auto &lowering = *interfaces[iface];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&lowering.op.getBody().front());

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
      moore::InterfaceSignalDeclOp::create(builder, portLoc, port->name, type);
      // Track the internal symbol so we skip it in member iteration
      if (port->internalSymbol)
        portInternalSymbols.insert(port->internalSymbol);
    } else if (const auto *multiPort =
                   symbol->as_if<slang::ast::MultiPortSymbol>()) {
      for (auto *port : multiPort->ports) {
        auto portLoc = convertLocation(port->location);
        auto type = convertType(port->getType());
        if (!type)
          return failure();
        moore::InterfaceSignalDeclOp::create(builder, portLoc, port->name, type);
        if (port->internalSymbol)
          portInternalSymbols.insert(port->internalSymbol);
      }
    }
  }

  // Convert all members of the interface
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
      moore::InterfaceSignalDeclOp::create(builder, loc, var->name, type);
      continue;
    }

    if (auto *net = member.as_if<slang::ast::NetSymbol>()) {
      auto type = convertType(net->getType());
      if (!type)
        return failure();
      moore::InterfaceSignalDeclOp::create(builder, loc, net->name, type);
      continue;
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

    // Skip other members for now (parameters, etc.)
  }

  return success();
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareFunction(const slang::ast::SubroutineSymbol &subroutine) {
  // Check if there already is a declaration for this function.
  auto &lowering = functions[&subroutine];
  if (lowering) {
    if (!lowering->op)
      return {};
    return lowering.get();
  }

  if (!subroutine.thisVar) {

    SmallString<64> name;
    guessNamespacePrefix(subroutine.getParentScope()->asSymbol(), name);
    name += subroutine.name;

    SmallVector<Type, 1> noThis = {};
    return declareCallableImpl(subroutine, name, noThis);
  }

  auto loc = convertLocation(subroutine.location);

  // Extract 'this' type and ensure it's a class.
  const slang::ast::Type &thisTy = subroutine.thisVar->getType();
  moore::ClassDeclOp ownerDecl;

  if (auto *classTy = thisTy.as_if<slang::ast::ClassType>()) {
    auto &ownerLowering = classes[classTy];
    ownerDecl = ownerLowering->op;
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
      inputTypes.push_back(
          moore::RefType::get(cast<moore::UnpackedType>(type)));
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
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // First get or create the function declaration.
  auto *lowering = declareFunction(subroutine);
  if (!lowering)
    return failure();

  // If function already has been finalized, or is already being converted
  // (recursive/re-entrant calls) stop here.
  if (lowering->capturesFinalized || lowering->isConverting)
    return success();

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

  // If this is a class method, the first input is %this :
  // !moore.class<@C>
  if (isMethod) {
    auto thisLoc = convertLocation(subroutine.location);
    auto thisType = lowering->op.getFunctionType().getInput(0);
    auto thisArg = block.addArgument(thisType, thisLoc);

    // Bind `this` so NamedValue/MemberAccess can find it.
    valueSymbols.insert(subroutine.thisVar, thisArg);
  }

  // Add user-defined block arguments
  auto inputs = lowering->op.getFunctionType().getInputs();
  auto astArgs = subroutine.getArguments();
  auto valInputs = llvm::ArrayRef<Type>(inputs).drop_front(isMethod ? 1 : 0);

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

  lowering->isConverting = true;
  auto convertingGuard =
      llvm::make_scope_exit([&] { lowering->isConverting = false; });

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
    if (returnVar && !subroutine.getReturnType().isVoid()) {
      Value read =
          moore::ReadOp::create(builder, returnVar.getLoc(), returnVar);
      mlir::func::ReturnOp::create(builder, lowering->op.getLoc(), read);
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
    if (!classLowering.op.getBody().empty())
      return success();

    OpBuilder::InsertionGuard ig(builder);

    Block *body = &classLowering.op.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(body);

    // Two-pass conversion: properties first, then methods.
    // This ensures properties are declared before method bodies are converted,
    // since method bodies may reference properties.

    // Pass 1: Convert properties, parameters, type aliases, and constraints
    for (const auto &mem : classAST.members()) {
      if (mem.kind == slang::ast::SymbolKind::ClassProperty ||
          mem.kind == slang::ast::SymbolKind::Parameter ||
          mem.kind == slang::ast::SymbolKind::TypeAlias ||
          mem.kind == slang::ast::SymbolKind::TypeParameter ||
          mem.kind == slang::ast::SymbolKind::ConstraintBlock) {
        if (failed(mem.visit(*this)))
          return failure();
      }
    }

    // Pass 2: Convert methods and other members
    for (const auto &mem : classAST.members()) {
      if (mem.kind != slang::ast::SymbolKind::ClassProperty &&
          mem.kind != slang::ast::SymbolKind::Parameter &&
          mem.kind != slang::ast::SymbolKind::TypeAlias &&
          mem.kind != slang::ast::SymbolKind::TypeParameter &&
          mem.kind != slang::ast::SymbolKind::ConstraintBlock) {
        if (failed(mem.visit(*this)))
          return failure();
      }
    }

    return success();
  }

  // Properties: ClassPropertySymbol
  LogicalResult visit(const slang::ast::ClassPropertySymbol &prop) {
    auto loc = convertLocation(prop.location);
    auto ty = context.convertType(prop.getType());
    if (!ty)
      return failure();

    // Check if this is a static property.
    bool isStatic = prop.lifetime == slang::ast::VariableLifetime::Static;

    // Static properties are stored as global variables, not instance fields.
    if (isStatic) {
      // Check if already converted (for on-demand conversion).
      if (context.globalVariables.count(&prop))
        return success();

      // Pick an insertion point for this variable at the module level.
      OpBuilder::InsertionGuard g(builder);
      auto it = context.orderedRootOps.upper_bound(prop.location);
      if (it == context.orderedRootOps.end())
        builder.setInsertionPointToEnd(context.intoModuleOp.getBody());
      else
        builder.setInsertionPoint(it->second);

      // Use fully qualified name: Class::property
      auto symName = fullyQualifiedSymbolName(context, prop);

      // Create the global variable op.
      auto varOp = moore::GlobalVariableOp::create(
          builder, loc, symName, cast<moore::UnpackedType>(ty));
      context.orderedRootOps.insert({prop.location, varOp});
      context.globalVariables.insert({&prop, varOp});

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
    if (fn.flags & slang::ast::MethodFlags::BuiltIn) {
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
      // Add an extra %this argument.
      SmallVector<Type, 1> extraParams;
      auto classSym =
          mlir::FlatSymbolRefAttr::get(classLowering.op.getSymNameAttr());
      auto handleTy =
          moore::ClassHandleType::get(context.getContext(), classSym);
      extraParams.push_back(handleTy);

      auto funcTy = getFunctionSignature(context, fn, extraParams);
      if (!funcTy)
        return failure();
      moore::ClassMethodDeclOp::create(builder, loc, fn.name, funcTy, nullptr);
      return success();
    }

    auto *lowering = context.declareFunction(fn);
    if (!lowering)
      return failure();

    if (failed(context.convertFunction(fn)))
      return failure();

    if (!lowering->capturesFinalized)
      return failure();

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
    const auto *externImpl = fn.getSubroutine();
    // We needn't convert a forward declaration without a unique implementation.
    if (!externImpl) {
      mlir::emitError(convertLocation(fn.location))
          << "Didn't find an implementation matching the forward declaration "
             "of "
          << fn.name;
      return failure();
    }
    return visit(*externImpl);
  }

  // Nested class definition, skip
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Nested class definition, convert
  LogicalResult visit(const slang::ast::ClassType &cls) {
    return context.convertClassDeclaration(cls);
  }

  // Transparent members: ignore (inherited names pulled in by slang)
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Empty members: ignore
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
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

    // For now, leave the body empty. Full constraint expression parsing
    // will be added in a future patch.
    constraintOp.getBody().emplaceBlock();

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
  auto &lowering = classes[&cls];
  bool isNewDecl = !lowering;
  if (isNewDecl) {
    lowering = std::make_unique<ClassLowering>();
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

    SymbolTable::setSymbolVisibility(classDeclOp,
                                     SymbolTable::Visibility::Public);
    orderedRootOps.insert(it, {cls.location, classDeclOp});
    lowering->op = classDeclOp;
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
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

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
  // Check if already converted (for on-demand conversion).
  if (globalVariables.count(&var))
    return success();

  auto loc = convertLocation(var.location);

  // Pick an insertion point for this variable according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(var.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Prefix the variable name with the surrounding namespace to create somewhat
  // sane names in the IR.
  SmallString<64> symName;
  guessNamespacePrefix(var.getParentScope()->asSymbol(), symName);
  symName += var.name;

  // Determine the type of the variable.
  auto type = convertType(var.getType());
  if (!type)
    return failure();

  // Create the variable op itself.
  auto varOp = moore::GlobalVariableOp::create(builder, loc, symName,
                                               cast<moore::UnpackedType>(type));
  orderedRootOps.insert({var.location, varOp});
  globalVariables.insert({&var, varOp});

  // If the variable has an initializer expression, remember it for later such
  // that we can convert the initializers once we have seen all global
  // variables.
  if (var.getInitializer())
    globalVariableWorklist.push_back(&var);

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
    case slang::ast::SymbolKind::ClassType:
      if (!parentSym.name.empty())
        parts.push_back(parentSym.name); // keep packages + outer classes
      break;
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
