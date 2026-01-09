//===- UVMFactory.cpp - UVM Factory pattern runtime support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the UVM factory infrastructure for type registration,
// instance creation, and type/instance overrides.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/UVMFactory.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "uvm-factory"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// UVMFactory Singleton
//===----------------------------------------------------------------------===//

UVMFactory &UVMFactory::getInstance() {
  static UVMFactory instance;
  return instance;
}

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

void UVMFactory::registerType(llvm::StringRef typeName,
                              std::function<void *()> creator,
                              std::function<void(void *)> destructor) {
  registerType(typeName, std::move(creator), std::move(destructor), nullptr);
}

void UVMFactory::registerType(llvm::StringRef typeName,
                              std::function<void *()> creator,
                              std::function<void(void *)> destructor,
                              const std::type_info *typeId) {
  UVMTypeInfo info(typeName, std::move(creator), std::move(destructor), typeId);
  registeredTypes[typeName] = std::move(info);
  stats.typesRegistered++;

  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Registered type '" << typeName
                          << "'\n");
}

bool UVMFactory::isTypeRegistered(llvm::StringRef typeName) const {
  return registeredTypes.count(typeName) > 0;
}

const UVMTypeInfo *UVMFactory::getTypeInfo(llvm::StringRef typeName) const {
  auto it = registeredTypes.find(typeName);
  if (it != registeredTypes.end())
    return &it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Instance Creation
//===----------------------------------------------------------------------===//

void *UVMFactory::createByName(llvm::StringRef typeName) {
  // No instance path, so only apply type overrides
  std::string effectiveType = resolveTypeChain(typeName);

  auto it = registeredTypes.find(effectiveType);
  if (it == registeredTypes.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMFactory: Error - type '" << effectiveType
               << "' not registered\n");
    return nullptr;
  }

  if (!it->second.creator) {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMFactory: Error - type '" << effectiveType
               << "' has no creator\n");
    return nullptr;
  }

  stats.instancesCreated++;
  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Creating instance of '"
                          << effectiveType << "'\n");
  return it->second.creator();
}

void *UVMFactory::createByName(llvm::StringRef requestedType,
                               llvm::StringRef instPath) {
  // First check instance overrides, then type overrides
  std::string effectiveType = findOverride(requestedType, instPath);

  auto it = registeredTypes.find(effectiveType);
  if (it == registeredTypes.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMFactory: Error - type '" << effectiveType
               << "' not registered\n");
    return nullptr;
  }

  if (!it->second.creator) {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMFactory: Error - type '" << effectiveType
               << "' has no creator\n");
    return nullptr;
  }

  stats.instancesCreated++;
  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Creating instance of '"
                          << effectiveType << "' for path '" << instPath
                          << "'\n");
  return it->second.creator();
}

void *UVMFactory::createObject(llvm::StringRef requestedType,
                               llvm::StringRef instPath, llvm::StringRef name) {
  // Build full path
  std::string fullPath;
  if (!instPath.empty()) {
    fullPath = instPath.str() + "." + name.str();
  } else {
    fullPath = name.str();
  }

  return createByName(requestedType, fullPath);
}

void UVMFactory::destroy(llvm::StringRef typeName, void *instance) {
  if (!instance)
    return;

  auto it = registeredTypes.find(typeName);
  if (it != registeredTypes.end() && it->second.destructor) {
    it->second.destructor(instance);
  } else {
    // Fallback: use delete (may not work for all types)
    LLVM_DEBUG(llvm::dbgs()
               << "UVMFactory: Warning - no destructor for type '" << typeName
               << "', using default delete\n");
  }
}

//===----------------------------------------------------------------------===//
// Type Overrides
//===----------------------------------------------------------------------===//

void UVMFactory::setTypeOverride(llvm::StringRef originalType,
                                 llvm::StringRef overrideType, bool replace) {
  // Check if override already exists
  auto existingIt = typeOverrideMap.find(originalType);
  if (existingIt != typeOverrideMap.end()) {
    if (!replace) {
      LLVM_DEBUG(llvm::dbgs()
                 << "UVMFactory: Type override for '" << originalType
                 << "' already exists, not replacing\n");
      return;
    }
    // Update existing override
    existingIt->second = overrideType.str();

    // Update in the list as well
    for (auto &ov : typeOverrides) {
      if (ov.originalType == originalType.str()) {
        ov.overrideType = overrideType.str();
        break;
      }
    }
  } else {
    // Add new override
    typeOverrides.emplace_back(originalType, overrideType, replace);
    typeOverrideMap[originalType] = overrideType.str();
  }

  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Set type override '" << originalType
                          << "' -> '" << overrideType << "'\n");
}

llvm::StringRef UVMFactory::getTypeOverride(llvm::StringRef originalType) const {
  auto it = typeOverrideMap.find(originalType);
  if (it != typeOverrideMap.end())
    return it->second;
  return originalType;
}

void UVMFactory::removeTypeOverride(llvm::StringRef originalType) {
  typeOverrideMap.erase(originalType);

  // Remove from list
  typeOverrides.erase(
      std::remove_if(typeOverrides.begin(), typeOverrides.end(),
                     [&](const UVMTypeOverride &ov) {
                       return ov.originalType == originalType.str();
                     }),
      typeOverrides.end());

  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Removed type override for '"
                          << originalType << "'\n");
}

//===----------------------------------------------------------------------===//
// Instance Overrides
//===----------------------------------------------------------------------===//

void UVMFactory::setInstOverride(llvm::StringRef instPath,
                                 llvm::StringRef originalType,
                                 llvm::StringRef overrideType) {
  instOverrides.emplace_back(instPath, originalType, overrideType);
  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Set instance override at '"
                          << instPath << "' for type '" << originalType
                          << "' -> '" << overrideType << "'\n");
}

void UVMFactory::setInstOverrideByPath(llvm::StringRef instPath,
                                       llvm::StringRef overrideType) {
  // Empty original type means match any type
  setInstOverride(instPath, "", overrideType);
}

llvm::StringRef UVMFactory::getInstOverride(llvm::StringRef instPath,
                                            llvm::StringRef requestedType) const {
  // Search instance overrides in reverse order (later overrides take precedence)
  for (auto it = instOverrides.rbegin(); it != instOverrides.rend(); ++it) {
    if (it->matches(instPath, requestedType)) {
      stats.instOverridesApplied++;
      LLVM_DEBUG(llvm::dbgs()
                 << "UVMFactory: Found instance override at '" << instPath
                 << "': '" << requestedType << "' -> '" << it->overrideType
                 << "'\n");
      return it->overrideType;
    }
  }
  return requestedType;
}

void UVMFactory::removeInstOverride(llvm::StringRef instPath,
                                    llvm::StringRef originalType) {
  instOverrides.erase(
      std::remove_if(instOverrides.begin(), instOverrides.end(),
                     [&](const UVMInstOverride &ov) {
                       return ov.instPath == instPath.str() &&
                              ov.originalType == originalType.str();
                     }),
      instOverrides.end());

  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Removed instance override at '"
                          << instPath << "' for type '" << originalType
                          << "'\n");
}

//===----------------------------------------------------------------------===//
// Override Lookup
//===----------------------------------------------------------------------===//

std::string UVMFactory::findOverride(llvm::StringRef requestedType,
                                     llvm::StringRef instPath) const {
  // First, check instance overrides (highest priority)
  if (!instPath.empty()) {
    for (auto it = instOverrides.rbegin(); it != instOverrides.rend(); ++it) {
      if (it->matches(instPath, requestedType)) {
        stats.instOverridesApplied++;
        LLVM_DEBUG(llvm::dbgs()
                   << "UVMFactory: Applying instance override: '"
                   << requestedType << "' -> '" << it->overrideType << "'\n");
        // Recursively resolve in case of chained overrides
        return resolveTypeChain(it->overrideType);
      }
    }
  }

  // Then, apply type overrides
  return resolveTypeChain(requestedType);
}

std::string UVMFactory::resolveTypeChain(llvm::StringRef typeName) const {
  std::string current = typeName.str();
  llvm::SmallPtrSet<const void *, 16> visited;

  while (true) {
    auto it = typeOverrideMap.find(current);
    if (it == typeOverrideMap.end())
      break;

    // Check for cycles
    if (!visited.insert(it->second.c_str()).second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "UVMFactory: Warning - cycle detected in type overrides\n");
      break;
    }

    stats.typeOverridesApplied++;
    LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Applying type override: '"
                            << current << "' -> '" << it->second << "'\n");
    current = it->second;
  }

  return current;
}

//===----------------------------------------------------------------------===//
// Debug and Printing
//===----------------------------------------------------------------------===//

void UVMFactory::printRegisteredTypes(llvm::raw_ostream &os) const {
  os << "Registered Types:\n";
  for (const auto &entry : registeredTypes) {
    os << "  " << entry.first() << "\n";
  }
}

void UVMFactory::printOverrides(llvm::raw_ostream &os) const {
  os << "Type Overrides:\n";
  for (const auto &ov : typeOverrides) {
    os << "  " << ov.originalType << " -> " << ov.overrideType << "\n";
  }

  os << "Instance Overrides:\n";
  for (const auto &ov : instOverrides) {
    os << "  " << ov.instPath << " [" << ov.originalType << "] -> "
       << ov.overrideType << "\n";
  }
}

void UVMFactory::print(llvm::raw_ostream &os) const {
  os << "=== UVM Factory State ===\n";
  printRegisteredTypes(os);
  printOverrides(os);
  os << "Statistics:\n";
  os << "  Types registered: " << stats.typesRegistered << "\n";
  os << "  Instances created: " << stats.instancesCreated << "\n";
  os << "  Type overrides applied: " << stats.typeOverridesApplied << "\n";
  os << "  Instance overrides applied: " << stats.instOverridesApplied << "\n";
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

void UVMFactory::reset() {
  registeredTypes.clear();
  typeOverrides.clear();
  typeOverrideMap.clear();
  instOverrides.clear();
  stats = Statistics();

  LLVM_DEBUG(llvm::dbgs() << "UVMFactory: Reset\n");
}

//===----------------------------------------------------------------------===//
// UVMFactoryOverrideGuard Implementation
//===----------------------------------------------------------------------===//

UVMFactoryOverrideGuard::UVMFactoryOverrideGuard(UVMFactory &factory,
                                                 llvm::StringRef originalType,
                                                 llvm::StringRef overrideType)
    : factory(factory), isTypeOverride(true),
      originalType(originalType.str()), hadPreviousOverride(false) {
  // Save previous override if it exists
  auto existing = factory.getTypeOverride(originalType);
  if (existing != originalType) {
    hadPreviousOverride = true;
    previousOverride = existing.str();
  }

  factory.setTypeOverride(originalType, overrideType);
}

UVMFactoryOverrideGuard::UVMFactoryOverrideGuard(UVMFactory &factory,
                                                 llvm::StringRef instPath,
                                                 llvm::StringRef originalType,
                                                 llvm::StringRef overrideType)
    : factory(factory), isTypeOverride(false),
      originalType(originalType.str()), instPath(instPath.str()),
      hadPreviousOverride(false) {
  factory.setInstOverride(instPath, originalType, overrideType);
}

UVMFactoryOverrideGuard::~UVMFactoryOverrideGuard() {
  if (isTypeOverride) {
    if (hadPreviousOverride) {
      factory.setTypeOverride(originalType, previousOverride);
    } else {
      factory.removeTypeOverride(originalType);
    }
  } else {
    factory.removeInstOverride(instPath, originalType);
  }
}
