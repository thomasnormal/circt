//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTContext.h"
#include "slang/ast/Expression.h"
#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/syntax/AllSyntax.h"

using namespace circt;
using namespace ImportVerilog;

namespace {
struct HierPathValueExprVisitor
    : public slang::ast::ASTVisitor<HierPathValueExprVisitor, false, true,
                                    true> {
  struct ReceiverPathSegment {
    std::string name;
    std::optional<int64_t> index;
  };

  Context &context;
  Location loc;
  OpBuilder &builder;
  LogicalResult result = success();
  const slang::ast::Scope *bindScope = nullptr;

  // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
  // variable `a`.
  const slang::ast::Symbol &outermostModule;

  HierPathValueExprVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule,
                           const slang::ast::Scope *bindScope)
      : context(context), loc(loc), builder(context.builder),
        bindScope(bindScope), outermostModule(outermostModule) {}

  void threadInterfaceInstance(const slang::ast::InstanceSymbol *ifaceInst,
                               mlir::StringAttr pathName) {
    if (!ifaceInst)
      return;
    auto *outermostInstBody =
        outermostModule.as_if<slang::ast::InstanceBodySymbol>();
    if (!outermostInstBody)
      return;

    auto *ifaceParentBody =
        ifaceInst->getParentScope()->getContainingInstance();
    if (!ifaceParentBody)
      return;
    if (ifaceParentBody == outermostInstBody)
      return;

    SmallVector<const slang::ast::InstanceBodySymbol *, 8> ifaceChain;
    SmallVector<const slang::ast::InstanceBodySymbol *, 8> outerChain;
    for (auto *body = ifaceParentBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      ifaceChain.push_back(body);
    for (auto *body = outermostInstBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      outerChain.push_back(body);

    DenseMap<const slang::ast::InstanceBodySymbol *, size_t> ifaceIndex;
    ifaceIndex.reserve(ifaceChain.size());
    for (size_t i = 0; i < ifaceChain.size(); ++i)
      ifaceIndex[ifaceChain[i]] = i;

    const slang::ast::InstanceBodySymbol *lca = nullptr;
    size_t lcaOuterIndex = 0;
    for (size_t i = 0; i < outerChain.size(); ++i) {
      if (ifaceIndex.count(outerChain[i])) {
        lca = outerChain[i];
        lcaOuterIndex = i;
        break;
      }
    }
    if (!lca)
      return;

    auto addHierIfacePath =
        [&](const slang::ast::InstanceBodySymbol *sym,
            mlir::StringAttr nameAttr,
            slang::ast::ArgumentDirection dir) {
          auto &paths = context.hierInterfacePaths[sym];
          bool exists = llvm::any_of(paths, [&](const auto &info) {
            return info.hierName == nameAttr &&
                   info.ifaceInst == ifaceInst && info.direction == dir;
          });
          if (!exists)
            paths.push_back(HierInterfacePathInfo{nameAttr, {}, dir, ifaceInst});
        };

    SmallString<64> upwardName(ifaceInst->name);
    for (auto *body = ifaceParentBody; body && body != lca;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr) {
      addHierIfacePath(body, builder.getStringAttr(upwardName),
                       slang::ast::ArgumentDirection::Out);
      if (body->parentInstance) {
        SmallString<64> nextName;
        nextName += body->parentInstance->name;
        nextName += ".";
        nextName += upwardName;
        upwardName = nextName;
      }
    }

    if (!pathName)
      pathName = builder.getStringAttr(upwardName);

    for (size_t i = lcaOuterIndex; i > 0; --i) {
      auto *body = outerChain[i - 1];
      addHierIfacePath(body, pathName, slang::ast::ArgumentDirection::In);
    }
  }

  std::optional<int64_t>
  getArrayElementIndex(const slang::ast::InstanceSymbol &instSym) {
    if (instSym.arrayPath.empty())
      return std::nullopt;
    slang::SmallVector<slang::ConstantRange, 4> dimensions;
    instSym.getArrayDimensions(dimensions);
    if (dimensions.empty())
      return std::nullopt;
    const auto &range = dimensions[0];
    int64_t relIndex = static_cast<int64_t>(instSym.arrayPath[0]);
    return range.isLittleEndian() ? range.lower() + relIndex
                                  : range.upper() - relIndex;
  }

  const slang::ast::InstanceSymbol *
  findArrayElement(const slang::ast::InstanceArraySymbol &arraySym,
                   int64_t index) {
    for (const auto *elementSym : arraySym.elements) {
      if (!elementSym)
        continue;
      auto *element = elementSym->as_if<slang::ast::InstanceSymbol>();
      if (!element)
        continue;
      auto actualIndex = getArrayElementIndex(*element);
      if (actualIndex && *actualIndex == index)
        return element;
    }
    return nullptr;
  }

  std::optional<int64_t>
  parseConstantIndex(const slang::syntax::ElementSelectSyntax &selector) {
    auto *bitSelect = selector.selector->as_if<slang::syntax::BitSelectSyntax>();
    if (!bitSelect)
      return std::nullopt;
    auto *literal =
        bitSelect->expr->as_if<slang::syntax::LiteralExpressionSyntax>();
    if (literal) {
      int64_t value = 0;
      if (!llvm::StringRef(literal->literal.valueText()).getAsInteger(0, value))
        return value;
    }
    // Accept non-literal constant selectors (e.g. parameter/localparam refs or
    // folded arithmetic like [1-0]) by binding and evaluating the selector.
    const slang::ast::Scope *scope = context.currentScope;
    if (!scope)
      scope = outermostModule.as_if<slang::ast::InstanceBodySymbol>();
    if (!scope)
      return std::nullopt;
    slang::ast::ASTContext astContext(*scope,
                                      slang::ast::LookupLocation::max);
    const auto &boundExpr = slang::ast::Expression::bind(*bitSelect->expr,
                                                         astContext);
    if (boundExpr.bad())
      return std::nullopt;
    auto value = context.evaluateConstant(boundExpr).integer().as<int64_t>();
    if (!value)
      return std::nullopt;
    return *value;
  }

  bool collectReceiverSegments(const slang::syntax::NameSyntax &nameSyntax,
                               SmallVectorImpl<ReceiverPathSegment> &segments) {
    if (auto *id =
            nameSyntax.as_if<slang::syntax::IdentifierNameSyntax>()) {
      segments.push_back({std::string(id->identifier.valueText()), std::nullopt});
      return true;
    }
    if (auto *idSelect =
            nameSyntax.as_if<slang::syntax::IdentifierSelectNameSyntax>()) {
      std::optional<int64_t> index;
      if (!idSelect->selectors.empty()) {
        if (idSelect->selectors.size() != 1)
          return false;
        if (!idSelect->selectors[0])
          return false;
        index = parseConstantIndex(*idSelect->selectors[0]);
        if (!index)
          return false;
      }
      segments.push_back({std::string(idSelect->identifier.valueText()), index});
      return true;
    }
    auto *scoped = nameSyntax.as_if<slang::syntax::ScopedNameSyntax>();
    if (!scoped)
      return false;
    if (!collectReceiverSegments(*scoped->left, segments))
      return false;
    if (!collectReceiverSegments(*scoped->right, segments))
      return false;
    return true;
  }

  const slang::ast::InstanceSymbol *
  resolveInstancePath(ArrayRef<ReceiverPathSegment> path) {
    if (path.empty())
      return nullptr;
    auto *outermostInstBody =
        outermostModule.as_if<slang::ast::InstanceBodySymbol>();
    if (!outermostInstBody)
      return nullptr;

    const slang::ast::Symbol *current = outermostInstBody->find(path.front().name);
    if (!current)
      return nullptr;

    for (const auto &segment : path.drop_front()) {
      auto *instSym = current->as_if<slang::ast::InstanceSymbol>();
      if (!instSym)
        return nullptr;
      if (!segment.index) {
        current = instSym->body.find(segment.name);
        if (!current)
          return nullptr;
        continue;
      }
      auto *nextSym = instSym->body.find(segment.name);
      if (!nextSym)
        return nullptr;
      if (auto *arraySym = nextSym->as_if<slang::ast::InstanceArraySymbol>()) {
        current = findArrayElement(*arraySym, *segment.index);
        if (!current)
          return nullptr;
        continue;
      }
      auto *childInst = nextSym->as_if<slang::ast::InstanceSymbol>();
      if (!childInst)
        return nullptr;
      auto actualIndex = getArrayElementIndex(*childInst);
      if (!actualIndex || *actualIndex != *segment.index)
        return nullptr;
      current = childInst;
    }

    return current->as_if<slang::ast::InstanceSymbol>();
  }

  mlir::StringAttr
  makePathNameFromHierRef(const slang::ast::HierarchicalReference &ref) {
    SmallString<64> pathName;
    for (const auto &elem : ref.path) {
      auto appendSegment = [&](StringRef name) {
        if (name.empty())
          return;
        if (!pathName.empty())
          pathName += ".";
        pathName += name;
      };
      if (auto *inst = elem.symbol->as_if<slang::ast::InstanceSymbol>())
        appendSegment(inst->name);
      else if (auto *array =
                   elem.symbol->as_if<slang::ast::InstanceArraySymbol>())
        appendSegment(array->name);
      else if (auto *ifacePort =
                   elem.symbol->as_if<slang::ast::InterfacePortSymbol>())
        appendSegment(ifacePort->name);

      if (auto *index = std::get_if<int32_t>(&elem.selector))
        pathName += ("[" + llvm::Twine(*index) + "]").str();
      else if (auto *range =
                   std::get_if<std::pair<int32_t, int32_t>>(&elem.selector))
        pathName += ("[" + llvm::Twine(range->first) + ":" +
                     llvm::Twine(range->second) +
                     "]")
                        .str();
    }
    if (pathName.empty())
      return mlir::StringAttr{};
    return builder.getStringAttr(pathName);
  }

  bool threadInterfaceFromReceiverExpr(const slang::ast::Expression &recvExpr) {
    auto getInterfaceArrayIndex =
        [&](const slang::ast::InstanceSymbol &inst)
        -> std::optional<int64_t> {
      if (inst.arrayPath.empty())
        return std::nullopt;
      slang::SmallVector<slang::ConstantRange, 4> dimensions;
      inst.getArrayDimensions(dimensions);
      if (dimensions.empty())
        return std::nullopt;
      const auto &range = dimensions[0];
      int64_t relIndex = static_cast<int64_t>(inst.arrayPath[0]);
      return range.isLittleEndian() ? range.lower() + relIndex
                                    : range.upper() - relIndex;
    };
    auto findInterfaceArrayElement =
        [&](const slang::ast::InstanceArraySymbol &arraySym, int64_t index)
        -> const slang::ast::InstanceSymbol * {
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
    };

    if (auto *arb = recvExpr.as_if<slang::ast::ArbitrarySymbolExpression>()) {
      auto *ifaceInst = arb->symbol->as_if<slang::ast::InstanceSymbol>();
      if (!ifaceInst || ifaceInst->getDefinition().definitionKind !=
                            slang::ast::DefinitionKind::Interface)
        return false;
      mlir::StringAttr pathName{};
      if (!arb->hierRef.path.empty())
        pathName = makePathNameFromHierRef(arb->hierRef);
      threadInterfaceInstance(ifaceInst, pathName);
      return true;
    }
    if (auto *hier = recvExpr.as_if<slang::ast::HierarchicalValueExpression>()) {
      auto *ifaceInst = hier->symbol.as_if<slang::ast::InstanceSymbol>();
      if (!ifaceInst || ifaceInst->getDefinition().definitionKind !=
                            slang::ast::DefinitionKind::Interface)
        return false;
      mlir::StringAttr pathName{};
      if (!hier->ref.path.empty())
        pathName = makePathNameFromHierRef(hier->ref);
      threadInterfaceInstance(ifaceInst, pathName);
      return true;
    }
    if (auto *elemSel = recvExpr.as_if<slang::ast::ElementSelectExpression>()) {
      const slang::ast::InstanceArraySymbol *arraySym = nullptr;
      if (auto symRef = elemSel->value().getSymbolReference())
        arraySym = symRef->as_if<slang::ast::InstanceArraySymbol>();
      if (!arraySym) {
        if (auto *hier =
                elemSel->value()
                    .as_if<slang::ast::HierarchicalValueExpression>()) {
          if (!hier->ref.path.empty())
            arraySym = hier->ref.path.back()
                           .symbol->as_if<slang::ast::InstanceArraySymbol>();
        } else if (auto *arb =
                       elemSel->value()
                           .as_if<slang::ast::ArbitrarySymbolExpression>()) {
          if (!arb->hierRef.path.empty())
            arraySym = arb->hierRef.path.back()
                           .symbol->as_if<slang::ast::InstanceArraySymbol>();
        }
      }
      if (!arraySym)
        return false;
      auto constIndex = context.evaluateConstant(elemSel->selector());
      if (!constIndex.isInteger())
        return false;
      auto index = constIndex.integer().as<int64_t>();
      if (!index)
        return false;
      auto *ifaceInst = findInterfaceArrayElement(*arraySym, *index);
      if (!ifaceInst || ifaceInst->getDefinition().definitionKind !=
                            slang::ast::DefinitionKind::Interface)
        return false;
      SmallString<64> pathName;
      if (auto *hier = elemSel->value()
                           .as_if<slang::ast::HierarchicalValueExpression>()) {
        if (!hier->ref.path.empty()) {
          auto pathAttr = makePathNameFromHierRef(hier->ref);
          if (pathAttr)
            pathName = pathAttr.getValue();
        }
      } else if (auto *arb =
                     elemSel->value().as_if<slang::ast::ArbitrarySymbolExpression>()) {
        if (!arb->hierRef.path.empty()) {
          auto pathAttr = makePathNameFromHierRef(arb->hierRef);
          if (pathAttr)
            pathName = pathAttr.getValue();
        }
      }
      if (pathName.empty())
        pathName = arraySym->name;
      pathName += ("[" + llvm::Twine(*index) + "]").str();
      threadInterfaceInstance(ifaceInst, builder.getStringAttr(pathName));
      return true;
    }
    return false;
  }

  // Handle hierarchical values
  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    auto *currentInstBody =
        expr.symbol.getParentScope()->getContainingInstance();
    if (!currentInstBody) {
      for (auto it = expr.ref.path.rbegin(); it != expr.ref.path.rend(); ++it) {
        if (auto *instSym =
                (*it).symbol->as_if<slang::ast::InstanceSymbol>()) {
          currentInstBody = &instSym->body;
          break;
        }
      }
    }
    auto *outermostInstBody =
        outermostModule.as_if<slang::ast::InstanceBodySymbol>();

    if (bindScope && outermostInstBody) {
      const slang::ast::Scope *parentScope = nullptr;
      if (auto *var = expr.symbol.as_if<slang::ast::VariableSymbol>())
        parentScope = var->getParentScope();
      else if (auto *net = expr.symbol.as_if<slang::ast::NetSymbol>())
        parentScope = net->getParentScope();

      const slang::ast::InstanceBodySymbol *ifaceBody = nullptr;
      if (parentScope)
        ifaceBody =
            parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>();

      if (ifaceBody && ifaceBody->getDefinition().definitionKind ==
                           slang::ast::DefinitionKind::Interface) {
        const slang::ast::Scope *bindPortScope = bindScope;
        if (auto *bindScopeInst = bindScope->getContainingInstance())
          bindPortScope = bindScopeInst;

        const slang::ast::InstanceBodySymbol *bindPortBody = nullptr;
        if (bindPortScope)
          bindPortBody = bindPortScope->asSymbol()
                             .as_if<slang::ast::InstanceBodySymbol>();

        if (bindPortBody) {
          const slang::ast::InterfacePortSymbol *matchedPort = nullptr;
          for (auto *portSym : bindPortBody->getPortList()) {
            auto *ifacePort =
                portSym->as_if<slang::ast::InterfacePortSymbol>();
            if (!ifacePort || !ifacePort->interfaceDef)
              continue;
            if (ifacePort->interfaceDef != &ifaceBody->getDefinition())
              continue;
            if (matchedPort) {
              matchedPort = nullptr;
              break;
            }
            matchedPort = ifacePort;
          }
          if (matchedPort) {
            auto &ports = context.bindScopeInterfacePorts[outermostInstBody];
            bool exists = llvm::any_of(ports, [&](const auto &info) {
              return info.ifacePort == matchedPort;
            });
            if (!exists)
              ports.push_back({matchedPort, std::nullopt});
          }
        }
      }
    }

    // If this reference is through an interface port in a bind scope, record
    // the interface port for threading into the target module.
    if (bindScope && outermostInstBody && expr.ref.isViaIfacePort()) {
      auto *bindScopeInst = bindScope->getContainingInstance();
      for (const auto &elem : expr.ref.path) {
        auto *ifacePort =
            elem.symbol->as_if<slang::ast::InterfacePortSymbol>();
        if (!ifacePort)
          continue;
        auto *portParent = ifacePort->getParentScope();
        if (!portParent ||
            !(portParent == bindScope ||
              (bindScopeInst && portParent == bindScopeInst)))
          continue;
        auto &ports = context.bindScopeInterfacePorts[outermostInstBody];
        bool exists = llvm::any_of(ports, [&](const auto &info) {
          return info.ifacePort == ifacePort;
        });
        if (!exists)
          ports.push_back({ifacePort, std::nullopt});
        return;
      }
    }

    // Handle hierarchical references to interface instances directly.
    if (auto *ifaceInst =
            expr.symbol.as_if<slang::ast::InstanceSymbol>()) {
      if (ifaceInst->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        SmallString<64> ifaceName;
        for (const auto &elem : expr.ref.path) {
          if (auto *instSym =
                  elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
            if (!ifaceName.empty())
              ifaceName += ".";
            ifaceName += instSym->name;
            continue;
          }
          if (auto *ifacePort =
                  elem.symbol->as_if<slang::ast::InterfacePortSymbol>()) {
            if (!ifaceName.empty())
              ifaceName += ".";
            ifaceName += ifacePort->name;
          }
        }
        if (!ifaceName.empty())
          threadInterfaceInstance(ifaceInst, builder.getStringAttr(ifaceName));
        return;
      }
    }

    // Like module Foo; int a; Foo.a; endmodule.
    // Ignore "Foo.a" invoked by this module itself.
    if (currentInstBody == outermostInstBody)
      return;

    auto getInterfaceDefBody =
        [&](const slang::ast::Symbol &sym)
        -> const slang::ast::InstanceBodySymbol * {
      const slang::ast::Scope *parentScope = nullptr;
      if (auto *var = sym.as_if<slang::ast::VariableSymbol>())
        parentScope = var->getParentScope();
      else if (auto *net = sym.as_if<slang::ast::NetSymbol>())
        parentScope = net->getParentScope();
      else if (auto *modportPort = sym.as_if<slang::ast::ModportPortSymbol>())
        if (auto *internalSym = modportPort->internalSymbol)
          parentScope = internalSym->getParentScope();
      if (!parentScope)
        return nullptr;
      if (auto *instBody =
              parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>())
        if (instBody->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface)
          return instBody;
      return nullptr;
    };

    // Handle hierarchical interface member references by threading the
    // interface instance through module ports.
    if (auto *ifaceDefBody = getInterfaceDefBody(expr.symbol)) {
      bool handledInterface = false;
      const slang::ast::InstanceSymbol *ifaceInst = nullptr;
      size_t ifaceInstIndex = 0;
      for (size_t i = 0; i < expr.ref.path.size(); ++i) {
        if (auto *instSym =
                expr.ref.path[i].symbol->as_if<slang::ast::InstanceSymbol>()) {
          if (instSym->getDefinition().definitionKind ==
                  slang::ast::DefinitionKind::Interface &&
              &instSym->getDefinition() == &ifaceDefBody->getDefinition()) {
            ifaceInst = instSym;
            ifaceInstIndex = i;
          }
        } else if (auto *ifacePort =
                       expr.ref.path[i].symbol
                           ->as_if<slang::ast::InterfacePortSymbol>()) {
          auto [ifaceConn, modport] = ifacePort->getConnection();
          (void)modport;
          if (auto *ifaceInstSym =
                  ifaceConn
                      ? ifaceConn->as_if<slang::ast::InstanceSymbol>()
                      : nullptr) {
            if (ifaceInstSym->getDefinition().definitionKind ==
                    slang::ast::DefinitionKind::Interface &&
                &ifaceInstSym->getDefinition() ==
                    &ifaceDefBody->getDefinition()) {
              ifaceInst = ifaceInstSym;
              ifaceInstIndex = i;
            }
          }
        }
      }
      if (ifaceInst && outermostInstBody) {
        const slang::ast::InstanceSymbol *threadInst = ifaceInst;
        size_t threadIndex = ifaceInstIndex;
        auto *ifaceParentBody =
            ifaceInst->getParentScope()->getContainingInstance();
        if (ifaceParentBody &&
            ifaceParentBody->getDefinition().definitionKind ==
                slang::ast::DefinitionKind::Interface) {
          for (size_t i = 0; i < ifaceInstIndex; ++i) {
            auto *instSym =
                expr.ref.path[i].symbol->as_if<slang::ast::InstanceSymbol>();
            if (!instSym)
              continue;
            if (instSym->getDefinition().definitionKind !=
                slang::ast::DefinitionKind::Interface)
              continue;
            auto *parentBody =
                instSym->getParentScope()->getContainingInstance();
            if (!parentBody ||
                parentBody->getDefinition().definitionKind !=
                    slang::ast::DefinitionKind::Interface) {
              threadInst = instSym;
              threadIndex = i;
            }
          }
        }
        ifaceInst = threadInst;
        if (ifaceParentBody && ifaceParentBody != outermostInstBody) {
          SmallString<64> ifaceName;
          for (size_t i = 0; i < expr.ref.path.size(); ++i) {
            if (auto *instSym =
                    expr.ref.path[i].symbol->as_if<
                        slang::ast::InstanceSymbol>()) {
              if (!ifaceName.empty())
                ifaceName += ".";
              ifaceName += instSym->name;
            } else if (auto *ifacePort =
                           expr.ref.path[i].symbol
                               ->as_if<slang::ast::InterfacePortSymbol>()) {
              if (!ifaceName.empty())
                ifaceName += ".";
              ifaceName += ifacePort->name;
            }
            if (i == threadIndex)
              break;
          }
          if (!ifaceName.empty()) {
            auto nameAttr = builder.getStringAttr(ifaceName);
            threadInterfaceInstance(ifaceInst, nameAttr);
            handledInterface = true;
          }
        } else if (ifaceParentBody == outermostInstBody) {
          handledInterface = true;
        }
      }
      // If we successfully handled the interface reference, we're done.
      if (handledInterface)
        return;
    }

    // Skip interface port member accesses that are already in scope. When
    // accessing a member of an interface port (e.g., `iface.data` where `iface`
    // is an interface port), slang marks this as a hierarchical reference.
    // These should be handled via VirtualInterfaceSignalRefOp if the interface
    // port is available in the current scope. Otherwise, allow hierarchical
    // handling to thread the reference.
    if (expr.ref.isViaIfacePort()) {
      const auto *ifacePort =
          expr.ref.path.front().symbol->as_if<slang::ast::InterfacePortSymbol>();
      const auto *portScope = ifacePort ? ifacePort->getParentScope() : nullptr;
      const auto *portBody =
          portScope ? portScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()
                    : nullptr;
      if (portBody && portBody == outermostInstBody)
        return;
    }

    if (!currentInstBody || !outermostInstBody)
      return;

    auto *exprValueSym = expr.symbol.as_if<slang::ast::ValueSymbol>();
    if (!exprValueSym)
      return;
    // Only runtime values (variables / nets) should be threaded through module
    // ports as hierarchical references. Enum members and other compile-time
    // constants are resolved by slang and must not become hierarchical ports.
    if (!exprValueSym->as_if<slang::ast::VariableSymbol>() &&
        !exprValueSym->as_if<slang::ast::NetSymbol>())
      return;
    auto addHierPath = [&](const slang::ast::InstanceBodySymbol *sym,
                           mlir::StringAttr nameAttr,
                           slang::ast::ArgumentDirection dir) {
      auto &paths = context.hierPaths[sym];
      auto existing = llvm::find_if(paths, [&](const auto &info) {
        return info.hierName == nameAttr && info.direction == dir;
      });
      if (existing == paths.end()) {
        paths.push_back(HierPathInfo{nameAttr, {}, dir, exprValueSym});
        return;
      }
      if (existing->valueSym != exprValueSym)
        context.hierValueAliases[exprValueSym] = existing->valueSym;
    };


    // Build ancestor chains for the symbol's body and the referencing scope.
    SmallVector<const slang::ast::InstanceBodySymbol *, 8> currentChain;
    SmallVector<const slang::ast::InstanceBodySymbol *, 8> outerChain;
    for (auto *body = currentInstBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      currentChain.push_back(body);
    for (auto *body = outermostInstBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      outerChain.push_back(body);

    DenseMap<const slang::ast::InstanceBodySymbol *, size_t> currentIndex;
    currentIndex.reserve(currentChain.size());
    for (size_t i = 0; i < currentChain.size(); ++i)
      currentIndex[currentChain[i]] = i;

    const slang::ast::InstanceBodySymbol *lca = nullptr;
    size_t lcaOuterIndex = 0;
    for (size_t i = 0; i < outerChain.size(); ++i) {
      if (currentIndex.count(outerChain[i])) {
        lca = outerChain[i];
        lcaOuterIndex = i;
        break;
      }
    }
    if (!lca)
      return;

    // Construct the full hierarchical name for the reference.
    SmallString<64> fullName;
    for (const auto &elem : expr.ref.path) {
      if (auto *instSym = elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
        if (!fullName.empty())
          fullName += ".";
        fullName += instSym->name;
      }
    }
    if (!fullName.empty())
      fullName += ".";
    fullName += expr.symbol.name;
    auto fullNameAttr = builder.getStringAttr(fullName);

    // Propagate upward from the symbol's instance body to the LCA.
    SmallString<64> upwardName(expr.symbol.name);
    for (auto *body = currentInstBody; body && body != lca;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr) {
      addHierPath(body, builder.getStringAttr(upwardName),
                  slang::ast::ArgumentDirection::Out);
      if (body->parentInstance) {
        SmallString<64> nextName;
        nextName += body->parentInstance->name;
        nextName += ".";
        nextName += upwardName;
        upwardName = nextName;
      }
    }

    // Propagate downward from the LCA to the referencing scope.
    for (size_t i = lcaOuterIndex; i > 0; --i) {
      auto *body = outerChain[i - 1];
      addHierPath(body, fullNameAttr, slang::ast::ArgumentDirection::In);
    }
  }

  void handle(const slang::ast::NamedValueExpression &expr) {
    if (failed(result))
      return;
    if (auto *ifacePort =
            expr.symbol.as_if<slang::ast::InterfacePortSymbol>()) {
      auto *outermostInstBody =
          outermostModule.as_if<slang::ast::InstanceBodySymbol>();
      const slang::ast::Scope *portScope = ifacePort->getParentScope();
      const auto *portBody =
          portScope
              ? portScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()
              : nullptr;
      if (outermostInstBody && portBody != outermostInstBody) {
        auto &ports = context.bindScopeInterfacePorts[outermostInstBody];
        bool exists = llvm::any_of(ports, [&](const auto &info) {
          return info.ifacePort == ifacePort;
        });
        if (!exists)
          ports.push_back({ifacePort, std::nullopt});
      }
      return;
    }
    if (auto *ifaceInst =
            expr.symbol.as_if<slang::ast::InstanceSymbol>()) {
      if (ifaceInst->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        threadInterfaceInstance(ifaceInst, mlir::StringAttr{});
      }
      return;
    }

    // Handle regular signals (Variables/Nets) from the bind scope.
    // When a bind port connection references a signal from the bind-writing
    // module (not the target module), thread it through via hierPaths.
    if (bindScope) {
      auto *outermostInstBody =
          outermostModule.as_if<slang::ast::InstanceBodySymbol>();
      auto *symScope = expr.symbol.getParentScope();
      const slang::ast::InstanceBodySymbol *symBody = nullptr;
      if (symScope) {
        symBody =
            symScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>();
        if (!symBody)
          symBody = symScope->getContainingInstance();
      }

      if (outermostInstBody && symBody && symBody != outermostInstBody) {
        // Build ancestor chains for the symbol's scope and the target scope.
        SmallVector<const slang::ast::InstanceBodySymbol *, 8> symChain;
        SmallVector<const slang::ast::InstanceBodySymbol *, 8> outerChain;
        for (auto *body = symBody; body;
             body = body->parentInstance
                        ? body->parentInstance->getParentScope()
                              ->getContainingInstance()
                        : nullptr)
          symChain.push_back(body);
        for (auto *body = outermostInstBody; body;
             body = body->parentInstance
                        ? body->parentInstance->getParentScope()
                              ->getContainingInstance()
                        : nullptr)
          outerChain.push_back(body);

        DenseMap<const slang::ast::InstanceBodySymbol *, size_t> symIndex;
        symIndex.reserve(symChain.size());
        for (size_t i = 0; i < symChain.size(); ++i)
          symIndex[symChain[i]] = i;

        const slang::ast::InstanceBodySymbol *lca = nullptr;
        size_t lcaOuterIndex = 0;
        for (size_t i = 0; i < outerChain.size(); ++i) {
          if (symIndex.count(outerChain[i])) {
            lca = outerChain[i];
            lcaOuterIndex = i;
            break;
          }
        }
        if (!lca)
          return;

        auto *exprValueSym = expr.symbol.as_if<slang::ast::ValueSymbol>();
        if (!exprValueSym)
          return;
        if (!exprValueSym->as_if<slang::ast::VariableSymbol>() &&
            !exprValueSym->as_if<slang::ast::NetSymbol>())
          return;
        auto addHierPath = [&](const slang::ast::InstanceBodySymbol *sym,
                               mlir::StringAttr nameAttr,
                               slang::ast::ArgumentDirection dir) {
          auto &paths = context.hierPaths[sym];
          auto existing = llvm::find_if(paths, [&](const auto &info) {
            return info.hierName == nameAttr && info.direction == dir;
          });
          if (existing == paths.end()) {
            paths.push_back(HierPathInfo{nameAttr, {}, dir, exprValueSym});
            return;
          }
          if (existing->valueSym != exprValueSym)
            context.hierValueAliases[exprValueSym] = existing->valueSym;
        };

        auto nameAttr = builder.getStringAttr(expr.symbol.name);

        // Propagate upward from the symbol's body to the LCA.
        SmallString<64> upwardName(expr.symbol.name);
        for (auto *body = symBody; body && body != lca;
             body = body->parentInstance
                        ? body->parentInstance->getParentScope()
                              ->getContainingInstance()
                        : nullptr) {
          addHierPath(body, builder.getStringAttr(upwardName),
                      slang::ast::ArgumentDirection::Out);
          if (body->parentInstance) {
            SmallString<64> nextName;
            nextName += body->parentInstance->name;
            nextName += ".";
            nextName += upwardName;
            upwardName = nextName;
          }
        }

        // Propagate downward from the LCA to the target scope.
        for (size_t i = lcaOuterIndex; i > 0; --i) {
          auto *body = outerChain[i - 1];
          addHierPath(body, nameAttr, slang::ast::ArgumentDirection::In);
        }
        return;
      }
    }
  }

  void handle(const slang::ast::ArbitrarySymbolExpression &expr) {
    if (failed(result))
      return;
    if (auto *ifaceInst = expr.symbol->as_if<slang::ast::InstanceSymbol>()) {
      if (ifaceInst->getDefinition().definitionKind ==
          slang::ast::DefinitionKind::Interface) {
        threadInterfaceInstance(ifaceInst, mlir::StringAttr{});
      }
    }
  }

  void handle(const slang::ast::CallExpression &expr) {
    if (failed(result))
      return;

    auto *subroutine =
        std::get_if<const slang::ast::SubroutineSymbol *>(&expr.subroutine);
    if (!subroutine || !*subroutine)
      return;

    auto *parentScope = (*subroutine)->getParentScope();
    auto *ifaceBody =
        parentScope ? parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()
                    : nullptr;
    if (!ifaceBody ||
        ifaceBody->getDefinition().definitionKind !=
            slang::ast::DefinitionKind::Interface)
      return;

    auto *invocation =
        expr.syntax ? expr.syntax->as_if<slang::syntax::InvocationExpressionSyntax>()
                    : nullptr;
    if (!invocation)
      return;

    if (auto *scopedReceiver =
            invocation->left->as_if<slang::syntax::ScopedNameSyntax>()) {
      SmallVector<ReceiverPathSegment, 4> path;
      if (collectReceiverSegments(*scopedReceiver->left, path) && !path.empty()) {
        if (auto *ifaceInst = resolveInstancePath(path)) {
          if (ifaceInst->getDefinition().definitionKind ==
                  slang::ast::DefinitionKind::Interface &&
              &ifaceInst->getDefinition() == &ifaceBody->getDefinition()) {
            SmallString<64> pathName;
            for (auto [idx, segment] : llvm::enumerate(path)) {
              if (idx)
                pathName += ".";
              pathName += segment.name;
              if (segment.index)
                pathName += ("[" + llvm::Twine(*segment.index) + "]").str();
            }
            threadInterfaceInstance(ifaceInst, builder.getStringAttr(pathName));
            return;
          }
        }
      }
    }

    if (!context.currentScope)
      return;
    const slang::syntax::ExpressionSyntax *receiverSyntax = nullptr;
    if (auto *memberAccess =
            invocation->left->as_if<slang::syntax::MemberAccessExpressionSyntax>())
      receiverSyntax = memberAccess->left;
    else if (auto *scopedReceiver =
                 invocation->left->as_if<slang::syntax::ScopedNameSyntax>())
      receiverSyntax = scopedReceiver->left;
    if (!receiverSyntax)
      return;

    slang::ast::ASTContext astContext(*context.currentScope,
                                      slang::ast::LookupLocation::max);
    const auto &receiverExpr =
        slang::ast::Expression::bind(*receiverSyntax, astContext);
    if (receiverExpr.bad())
      return;
    if (threadInterfaceFromReceiverExpr(receiverExpr))
      return;
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    // InvalidExpression can appear in dead generate blocks (e.g., assertions
    // referencing instances that don't exist, or DynamicNotProcedural assigns).
    // Skip without error — the conversion phase will catch real errors.
  }
};

/// Visitor to traverse statements and collect hierarchical value references.
/// This visitor visits both statements and expressions within them, reusing
/// the HierPathValueExprVisitor logic for expression handling.
struct HierPathValueStmtVisitor
    : public slang::ast::ASTVisitor<HierPathValueStmtVisitor, true, true,
                                    true> {
  Context &context;
  Location loc;
  LogicalResult result = success();

  // The outermost module for determining hierarchical paths.
  const slang::ast::Symbol &outermostModule;

  HierPathValueStmtVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule)
      : context(context), loc(loc), outermostModule(outermostModule) {}

  // Handle hierarchical values found in expressions within statements.
  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    // Delegate to the expression visitor logic via collectHierarchicalValues
    if (failed(context.collectHierarchicalValues(expr, outermostModule)))
      result = failure();
  }

  void handle(const slang::ast::CallExpression &expr) {
    if (failed(result))
      return;
    if (failed(context.collectHierarchicalValues(expr, outermostModule)))
      result = failure();
  }

  void handle(const slang::ast::ArbitrarySymbolExpression &expr) {
    if (failed(result))
      return;
    if (failed(context.collectHierarchicalValues(expr, outermostModule)))
      result = failure();
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    // InvalidExpression can appear in dead generate blocks (e.g., assertions
    // referencing instances that don't exist in this elaboration). Skip without
    // error — the conversion phase will catch real errors.
  }
};
} // namespace

struct BindAssertionExprVisitor
    : public slang::ast::ASTVisitor<BindAssertionExprVisitor, false, true,
                                    true> {
  HierPathValueExprVisitor exprVisitor;
  LogicalResult result = success();

  BindAssertionExprVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule,
                           const slang::ast::Scope *bindScope)
      : exprVisitor(context, loc, outermostModule, bindScope) {}

  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    exprVisitor.handle(expr);
    if (failed(exprVisitor.result))
      result = failure();
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    // InvalidExpression can appear inside dead generate assertion code.
    // Skip silently — the conversion phase handles real errors.
  }

  void handle(const slang::ast::InvalidAssertionExpr &) {
    // Dead generate code — no hierarchical refs to collect. Don't fail.
  }
};

LogicalResult
Context::collectHierarchicalValues(const slang::ast::Expression &expr,
                                   const slang::ast::Symbol &outermostModule,
                                   const slang::ast::Scope *bindScope) {
  auto loc = convertLocation(expr.sourceRange);
  HierPathValueExprVisitor visitor(*this, loc, outermostModule, bindScope);
  if (auto *callExpr = expr.as_if<slang::ast::CallExpression>())
    visitor.handle(*callExpr);
  expr.visit(visitor);
  return visitor.result;
}

LogicalResult Context::collectHierarchicalValuesFromStatement(
    const slang::ast::Statement &stmt,
    const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(stmt.sourceRange);
  HierPathValueStmtVisitor visitor(*this, loc, outermostModule);
  stmt.visit(visitor);
  return visitor.result;
}

template <typename CompilationT>
static const slang::ast::Scope *maybeGetBindDirectiveScope(
    const CompilationT &compilation,
    const slang::syntax::BindDirectiveSyntax *bindSyntax) {
  if constexpr (requires(const CompilationT &comp,
                         const slang::syntax::BindDirectiveSyntax *syntax) {
                  comp.getBindDirectiveScope(syntax);
                }) {
    if (bindSyntax)
      return compilation.getBindDirectiveScope(bindSyntax);
  }
  return nullptr;
}

template <typename InstanceT>
static const slang::ast::Scope *maybeGetInstanceBindScope(
    const InstanceT &instNode) {
  if constexpr (requires(const InstanceT &inst) { inst.getBindScope(); })
    return instNode.getBindScope();
  return nullptr;
}

static const slang::ast::Scope *
resolveBindDirectiveScopeCompat(Context &context,
                                const slang::ast::BindDirectiveInfo &bindInfo,
                                const slang::ast::InstanceBodySymbol &body) {
  // Newer slang revisions provide direct bind-scope queries.
  if (const auto *scope =
          maybeGetBindDirectiveScope(context.compilation, bindInfo.bindSyntax))
    return scope;

  // Older slang revisions don't expose bind scope directly.
  // Falling back to the target instance body matches the historical fallback
  // used for file-level / definition-level binds where compilation-unit scope
  // cannot resolve target-local names.
  return &body;
}

static const slang::ast::Scope *
resolveInstanceBindScopeCompat(const slang::ast::InstanceSymbol &instNode) {
  if (const auto *scope = maybeGetInstanceBindScope(instNode))
    return scope;

  // Older slang revisions don't expose InstanceSymbol::getBindScope. When this
  // instance comes from bind elaboration, use the immediate parent scope as the
  // best available approximation for hierarchical-name collection.
  if (instNode.body.flags.has(slang::ast::InstanceFlags::FromBind))
    return instNode.getParentScope();
  return nullptr;
}

static LogicalResult collectBindDirectiveHierarchicalValues(
    Context &context, const slang::ast::InstanceBodySymbol &body) {
  auto *node = body.hierarchyOverrideNode;
  if (!node)
    return success();

  for (const auto &entry : node->binds) {
    const auto &bindInfo = entry.first;
    auto *bindSyntax = bindInfo.bindSyntax;
    if (!bindSyntax)
      continue;
    const auto *bindScope =
        resolveBindDirectiveScopeCompat(context, bindInfo, body);
    if (!bindScope)
      continue;

    // For definition-level or file-level binds, the bind scope is the
    // compilation unit, where target module names aren't visible. In that case,
    // use the target module body for expression binding. Only use the bind scope
    // when it's inside an instance body (i.e., the bind was written inside a
    // module body).
    const slang::ast::Scope *exprScope = bindScope;
    auto *bindScopeBody =
        bindScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>();
    if (!bindScopeBody && !bindScope->getContainingInstance())
      exprScope = &body;

    slang::ast::ASTContext astContext(*exprScope,
                                      slang::ast::LookupLocation::max);

    auto collectFromPropertyExpr =
        [&](const slang::syntax::PropertyExprSyntax &syntax,
            const slang::ast::InstanceBodySymbol &outermostModule)
        -> LogicalResult {
      if (syntax.kind == slang::syntax::SyntaxKind::SimplePropertyExpr) {
        auto seqExpr = syntax.as<slang::syntax::SimplePropertyExprSyntax>().expr;
        if (seqExpr->kind == slang::syntax::SyntaxKind::SimpleSequenceExpr) {
          auto &simpleSeq =
              seqExpr->as<slang::syntax::SimpleSequenceExprSyntax>();
          if (!simpleSeq.repetition && simpleSeq.expr) {
            const auto &expr =
                slang::ast::Expression::bind(*simpleSeq.expr, astContext);
            if (expr.bad())
              return failure();
            return context.collectHierarchicalValues(expr, outermostModule,
                                                    bindScope);
          }
        }
      }

      const auto &expr = slang::ast::AssertionExpr::bind(
          syntax, astContext, /*allowDisable=*/true);
      auto loc = context.convertLocation(syntax.sourceRange());
      BindAssertionExprVisitor visitor(context, loc, outermostModule, bindScope);
      expr.visit(visitor);
      return visitor.result;
    };

    const slang::ast::InstanceBodySymbol *bindTargetBody = &body;
    if (bindScope && bindSyntax->target) {
      const auto &targetExpr =
          slang::ast::Expression::bind(*bindSyntax->target, astContext);
      if (!targetExpr.bad()) {
        if (auto *sym = targetExpr.getSymbolReference()) {
          if (auto *inst = sym->as_if<slang::ast::InstanceSymbol>())
            bindTargetBody = &inst->body;
        }
      }
    }

    auto collectFromCheckerInstances =
        [&](const auto &instances) -> LogicalResult {
      for (auto *inst : instances) {
        for (auto *conn : inst->connections) {
          if (auto *ordered =
                  conn->template as_if<
                      slang::syntax::OrderedPortConnectionSyntax>()) {
            if (!ordered->expr)
              continue;
            if (failed(collectFromPropertyExpr(*ordered->expr, *bindTargetBody)))
              return failure();
            continue;
          }
          if (auto *named =
                  conn->template as_if<
                      slang::syntax::NamedPortConnectionSyntax>()) {
            if (!named->expr)
              continue;
            if (failed(collectFromPropertyExpr(*named->expr, *bindTargetBody)))
              return failure();
            continue;
          }
        }
      }
      return success();
    };

    // Module bind instances show up in the instance body and are handled by
    // InstBodyVisitor; only checker binds need special property parsing here.
    // For HierarchyInstantiationSyntax (module binds), the port connections
    // are already resolved in slang's AST and will be visited when
    // InstBodyVisitor encounters the bound instance. Re-binding the raw
    // syntax here can fail because the expression scope (target module body)
    // may not be able to resolve hierarchical references that were written
    // relative to the original bind scope (e.g., file-level binds
    // referencing sibling module instances like `top.a.bus`).
    if (auto *checker = bindSyntax->instantiation->template as_if<
            slang::syntax::CheckerInstantiationSyntax>()) {
      if (failed(collectFromCheckerInstances(checker->instances)))
        return failure();
    }
  }

  return success();
}

/// Traverse the instance body.
namespace {
struct InstBodyVisitor {
  Context &context;
  Location loc;

  InstBodyVisitor(Context &context, Location loc)
      : context(context), loc(loc) {}

  const slang::ast::Symbol &
  getOutermostModule(const slang::ast::Scope &scope) const {
    if (auto *body = scope.getContainingInstance())
      return body->asSymbol();
    return scope.asSymbol();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    auto &outermostModule = getOutermostModule(*instNode.getParentScope());
    auto *bindScope = resolveInstanceBindScopeCompat(instNode);
    for (const auto *con : instNode.getPortConnections()) {
      if (const auto *expr = con->getExpression())
        if (failed(context.collectHierarchicalValues(*expr, outermostModule,
                                                     bindScope)))
          return failure();
    }
    return context.traverseInstanceBody(instNode.body);
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto &outermostModule = getOutermostModule(*varNode.getParentScope());
    if (const auto *init = varNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto &outermostModule = getOutermostModule(*netNode.getParentScope());
    if (const auto *init = netNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    const auto *expr =
        assignNode.getAssignment().as_if<slang::ast::AssignmentExpression>();
    // When allowNonProceduralDynamic is enabled, slang downgrades
    // DynamicNotProcedural errors to warnings but wraps the expression in
    // InvalidExpression. Unwrap to recover the underlying assignment.
    if (!expr && context.options.allowNonProceduralDynamic.value_or(false)) {
      if (const auto *invalid =
              assignNode.getAssignment()
                  .as_if<slang::ast::InvalidExpression>()) {
        if (invalid->child)
          expr = invalid->child->as_if<slang::ast::AssignmentExpression>();
      }
    }
    if (!expr) {
      mlir::emitError(loc)
          << "expected assignment expression in continuous assignment";
      return failure();
    }

    // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
    // variable `a`.
    auto &outermostModule = getOutermostModule(*assignNode.getParentScope());
    if (failed(context.collectHierarchicalValues(expr->left(), outermostModule)))
      return failure();

    if (failed(
            context.collectHierarchicalValues(expr->right(), outermostModule)))
      return failure();

    return success();
  }

  // Handle procedural blocks (always, initial, final, etc.).
  // Traverse the procedure body to collect hierarchical references
  // from event triggers, wait statements, and other expressions.
  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto &outermostModule = getOutermostModule(*procNode.getParentScope());
    return context.collectHierarchicalValuesFromStatement(procNode.getBody(),
                                                          outermostModule);
  }

  // Traverse generate blocks.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    SmallVector<const slang::ast::Symbol *> instanceMembers;
    SmallVector<const slang::ast::Symbol *> otherMembers;
    for (auto &member : genNode.members()) {
      if (member.kind == slang::ast::SymbolKind::Instance ||
          member.kind == slang::ast::SymbolKind::InstanceArray)
        instanceMembers.push_back(&member);
      else
        otherMembers.push_back(&member);
    }
    for (auto *member : instanceMembers) {
      auto loc = context.convertLocation(member->location);
      if (failed(member->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    for (auto *member : otherMembers) {
      auto loc = context.convertLocation(member->location);
      if (failed(member->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  // Traverse generate block arrays.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArray) {
    for (auto *entry : genArray.entries) {
      auto loc = context.convertLocation(entry->location);
      if (failed(entry->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  // Traverse instance arrays.
  LogicalResult visit(const slang::ast::InstanceArraySymbol &arrNode) {
    for (auto *element : arrNode.elements) {
      auto loc = context.convertLocation(element->location);
      if (failed(element->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  /// TODO:Skip all others.
  /// But we should output a warning to display which symbol had been skipped.
  /// However, to ensure we can test smoothly, we didn't do that.
  template <typename T>
  LogicalResult visit(T &&node) {
    return success();
  }
};
} // namespace

LogicalResult Context::traverseInstanceBody(const slang::ast::Symbol &symbol) {
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>()) {
    if (failed(collectBindDirectiveHierarchicalValues(*this,
                                                      *instBodySymbol)))
      return failure();
    for (auto &member : instBodySymbol->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(InstBodyVisitor(*this, loc))))
        return failure();
    }
  }
  return success();
}
