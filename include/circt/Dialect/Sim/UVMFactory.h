//===- UVMFactory.h - UVM Factory pattern runtime support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the UVM factory infrastructure for type registration,
// instance creation, and type/instance overrides. The factory pattern is
// central to UVM's configuration and reuse capabilities.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_UVMFACTORY_H
#define CIRCT_DIALECT_SIM_UVMFACTORY_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <typeinfo>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// UVMTypeInfo - Type information for registered types
//===----------------------------------------------------------------------===//

/// Type information stored in the factory.
struct UVMTypeInfo {
  /// The type name as a string.
  std::string typeName;

  /// Creator function that returns a new instance.
  std::function<void *()> creator;

  /// Optional destructor function.
  std::function<void(void *)> destructor;

  /// Type ID for runtime type checking (if available).
  const std::type_info *typeId;

  /// Whether this type is a parameterized type.
  bool isParameterized;

  UVMTypeInfo() : typeId(nullptr), isParameterized(false) {}

  UVMTypeInfo(llvm::StringRef name, std::function<void *()> creator,
              std::function<void(void *)> destructor = nullptr,
              const std::type_info *typeId = nullptr)
      : typeName(name.str()), creator(std::move(creator)),
        destructor(std::move(destructor)), typeId(typeId),
        isParameterized(false) {}
};

//===----------------------------------------------------------------------===//
// UVMTypeOverride - Type override entry
//===----------------------------------------------------------------------===//

/// Represents a type override in the factory.
struct UVMTypeOverride {
  /// Original type name (can be a pattern for wildcards).
  std::string originalType;

  /// Override type name.
  std::string overrideType;

  /// Whether to replace existing overrides.
  bool replace;

  UVMTypeOverride(llvm::StringRef original, llvm::StringRef override,
                  bool replace = true)
      : originalType(original.str()), overrideType(override.str()),
        replace(replace) {}
};

//===----------------------------------------------------------------------===//
// UVMInstOverride - Instance-specific override entry
//===----------------------------------------------------------------------===//

/// Represents an instance-specific override in the factory.
struct UVMInstOverride {
  /// Instance path pattern (can include wildcards).
  std::string instPath;

  /// Original type name.
  std::string originalType;

  /// Override type name.
  std::string overrideType;

  /// Compiled regex for path matching (if wildcards present).
  std::shared_ptr<std::regex> pathRegex;

  UVMInstOverride(llvm::StringRef path, llvm::StringRef original,
                  llvm::StringRef override)
      : instPath(path.str()), originalType(original.str()),
        overrideType(override.str()) {
    // Convert UVM wildcards to regex
    if (instPath.find('*') != std::string::npos) {
      std::string regexStr = instPath;
      // Escape dots
      size_t pos = 0;
      while ((pos = regexStr.find('.', pos)) != std::string::npos) {
        regexStr.replace(pos, 1, "\\.");
        pos += 2;
      }
      // Convert * to .*
      pos = 0;
      while ((pos = regexStr.find('*', pos)) != std::string::npos) {
        regexStr.replace(pos, 1, ".*");
        pos += 2;
      }
      pathRegex = std::make_shared<std::regex>(regexStr);
    }
  }

  /// Check if this override matches a given path and type.
  bool matches(llvm::StringRef path, llvm::StringRef typeName) const {
    // Check type match
    if (!originalType.empty() && originalType != typeName.str())
      return false;

    // Check path match
    if (pathRegex) {
      return std::regex_match(path.str(), *pathRegex);
    }
    return instPath == path.str();
  }
};

//===----------------------------------------------------------------------===//
// UVMFactory - Main factory class
//===----------------------------------------------------------------------===//

/// The UVM factory for creating and overriding types at runtime.
/// This implements the UVM factory pattern for component reuse and testbench
/// configuration.
class UVMFactory {
public:
  UVMFactory() = default;
  ~UVMFactory() = default;

  /// Get the singleton instance of the factory.
  static UVMFactory &getInstance();

  //===--------------------------------------------------------------------===//
  // Type Registration
  //===--------------------------------------------------------------------===//

  /// Register a type with the factory.
  void registerType(llvm::StringRef typeName, std::function<void *()> creator,
                    std::function<void(void *)> destructor = nullptr);

  /// Register a type with type_info for runtime type checking.
  /// Note: RTTI is disabled in this build, so typeId will be nullptr.
  template <typename T>
  void registerType(llvm::StringRef typeName) {
    registerType(
        typeName, []() -> void * { return new T(); },
        [](void *p) { delete static_cast<T *>(p); }, nullptr);
  }

  /// Register a type with type_info.
  void registerType(llvm::StringRef typeName, std::function<void *()> creator,
                    std::function<void(void *)> destructor,
                    const std::type_info *typeId);

  /// Check if a type is registered.
  bool isTypeRegistered(llvm::StringRef typeName) const;

  /// Get type information for a registered type.
  const UVMTypeInfo *getTypeInfo(llvm::StringRef typeName) const;

  /// Get all registered types.
  const llvm::StringMap<UVMTypeInfo> &getRegisteredTypes() const {
    return registeredTypes;
  }

  //===--------------------------------------------------------------------===//
  // Instance Creation
  //===--------------------------------------------------------------------===//

  /// Create an instance by type name (no overrides).
  void *createByName(llvm::StringRef typeName);

  /// Create an instance by type name with overrides applied.
  void *createByName(llvm::StringRef requestedType, llvm::StringRef instPath);

  /// Create an instance with full override resolution.
  void *createObject(llvm::StringRef requestedType, llvm::StringRef instPath,
                     llvm::StringRef name);

  /// Create and cast to the expected type.
  template <typename T>
  T *create(llvm::StringRef typeName) {
    return static_cast<T *>(createByName(typeName));
  }

  /// Create with instance path and cast.
  template <typename T>
  T *create(llvm::StringRef typeName, llvm::StringRef instPath) {
    return static_cast<T *>(createByName(typeName, instPath));
  }

  /// Destroy an instance created by the factory.
  void destroy(llvm::StringRef typeName, void *instance);

  //===--------------------------------------------------------------------===//
  // Type Overrides
  //===--------------------------------------------------------------------===//

  /// Set a global type override.
  void setTypeOverride(llvm::StringRef originalType,
                       llvm::StringRef overrideType, bool replace = true);

  /// Set a type override by type (using type_info for safety).
  template <typename TOriginal, typename TOverride>
  void setTypeOverrideByType(bool replace = true) {
    // This requires types to be registered with their type_info
    // For now, we use string names
  }

  /// Get the effective type after applying type overrides.
  llvm::StringRef getTypeOverride(llvm::StringRef originalType) const;

  /// Remove a type override.
  void removeTypeOverride(llvm::StringRef originalType);

  /// Get all type overrides.
  const std::vector<UVMTypeOverride> &getTypeOverrides() const {
    return typeOverrides;
  }

  //===--------------------------------------------------------------------===//
  // Instance Overrides
  //===--------------------------------------------------------------------===//

  /// Set an instance-specific override.
  void setInstOverride(llvm::StringRef instPath, llvm::StringRef originalType,
                       llvm::StringRef overrideType);

  /// Set an instance override for any type at a path.
  void setInstOverrideByPath(llvm::StringRef instPath,
                             llvm::StringRef overrideType);

  /// Get the effective type for an instance after applying overrides.
  llvm::StringRef getInstOverride(llvm::StringRef instPath,
                                  llvm::StringRef requestedType) const;

  /// Remove an instance override.
  void removeInstOverride(llvm::StringRef instPath,
                          llvm::StringRef originalType);

  /// Get all instance overrides.
  const std::vector<UVMInstOverride> &getInstOverrides() const {
    return instOverrides;
  }

  //===--------------------------------------------------------------------===//
  // Override Lookup
  //===--------------------------------------------------------------------===//

  /// Find the effective type after applying all overrides.
  /// First checks instance overrides, then type overrides.
  std::string findOverride(llvm::StringRef requestedType,
                           llvm::StringRef instPath) const;

  //===--------------------------------------------------------------------===//
  // Debug and Printing
  //===--------------------------------------------------------------------===//

  /// Print all registered types.
  void printRegisteredTypes(llvm::raw_ostream &os) const;

  /// Print all overrides.
  void printOverrides(llvm::raw_ostream &os) const;

  /// Print full factory state.
  void print(llvm::raw_ostream &os) const;

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t typesRegistered = 0;
    size_t instancesCreated = 0;
    size_t typeOverridesApplied = 0;
    size_t instOverridesApplied = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Reset the factory (clears all registrations and overrides).
  void reset();

private:
  /// Apply type override chain (handles nested overrides).
  std::string resolveTypeChain(llvm::StringRef typeName) const;

  // Type registry
  llvm::StringMap<UVMTypeInfo> registeredTypes;

  // Type overrides (global)
  std::vector<UVMTypeOverride> typeOverrides;
  llvm::StringMap<std::string> typeOverrideMap; // For quick lookup

  // Instance overrides
  std::vector<UVMInstOverride> instOverrides;

  // Statistics
  mutable Statistics stats;
};

//===----------------------------------------------------------------------===//
// UVMFactoryOverrideGuard - RAII guard for temporary overrides
//===----------------------------------------------------------------------===//

/// RAII guard for temporary factory overrides.
/// Restores the original override state when destroyed.
class UVMFactoryOverrideGuard {
public:
  /// Create a guard for a type override.
  UVMFactoryOverrideGuard(UVMFactory &factory, llvm::StringRef originalType,
                          llvm::StringRef overrideType);

  /// Create a guard for an instance override.
  UVMFactoryOverrideGuard(UVMFactory &factory, llvm::StringRef instPath,
                          llvm::StringRef originalType,
                          llvm::StringRef overrideType);

  ~UVMFactoryOverrideGuard();

  // Non-copyable
  UVMFactoryOverrideGuard(const UVMFactoryOverrideGuard &) = delete;
  UVMFactoryOverrideGuard &operator=(const UVMFactoryOverrideGuard &) = delete;

private:
  UVMFactory &factory;
  bool isTypeOverride;
  std::string originalType;
  std::string instPath;
  std::string previousOverride;
  bool hadPreviousOverride;
};

//===----------------------------------------------------------------------===//
// UVMObjectWrapper - Type-safe wrapper for factory-created objects
//===----------------------------------------------------------------------===//

/// Type-safe wrapper for objects created by the factory.
template <typename T>
class UVMObjectWrapper {
public:
  UVMObjectWrapper() : obj(nullptr) {}

  explicit UVMObjectWrapper(T *obj, llvm::StringRef typeName,
                            UVMFactory &factory)
      : obj(obj), typeName(typeName.str()), factory(&factory) {}

  ~UVMObjectWrapper() {
    if (obj && factory) {
      factory->destroy(typeName, obj);
    }
  }

  // Move-only
  UVMObjectWrapper(UVMObjectWrapper &&other) noexcept
      : obj(other.obj), typeName(std::move(other.typeName)),
        factory(other.factory) {
    other.obj = nullptr;
    other.factory = nullptr;
  }

  UVMObjectWrapper &operator=(UVMObjectWrapper &&other) noexcept {
    if (this != &other) {
      if (obj && factory) {
        factory->destroy(typeName, obj);
      }
      obj = other.obj;
      typeName = std::move(other.typeName);
      factory = other.factory;
      other.obj = nullptr;
      other.factory = nullptr;
    }
    return *this;
  }

  // Non-copyable
  UVMObjectWrapper(const UVMObjectWrapper &) = delete;
  UVMObjectWrapper &operator=(const UVMObjectWrapper &) = delete;

  T *get() const { return obj; }
  T *operator->() const { return obj; }
  T &operator*() const { return *obj; }
  explicit operator bool() const { return obj != nullptr; }

  T *release() {
    T *tmp = obj;
    obj = nullptr;
    return tmp;
  }

private:
  T *obj;
  std::string typeName;
  UVMFactory *factory;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_UVMFACTORY_H
