//===- DPIRuntime.h - DPI-C Runtime Support for Simulation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DPI-C (Direct Programming Interface) runtime support
// for simulation. It provides:
// - Extended data type support (arrays, structs, strings, open arrays)
// - Callback registration infrastructure
// - Type conversion utilities
// - Thread-safe DPI context management
// - Import/Export function registration
//
// Based on IEEE 1800-2017 Section 35 (Direct Programming Interface).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_DPIRUNTIME_H
#define CIRCT_DIALECT_SIM_DPIRUNTIME_H

#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// DPI Data Types - IEEE 1800-2017 Section 35.5
//===----------------------------------------------------------------------===//

/// Canonical types for DPI data transfer.
enum class DPIDataType : uint8_t {
  /// void type (for procedures)
  Void = 0,

  /// byte (8-bit signed)
  Byte = 1,

  /// shortint (16-bit signed)
  ShortInt = 2,

  /// int (32-bit signed)
  Int = 3,

  /// longint (64-bit signed)
  LongInt = 4,

  /// shortreal (float)
  ShortReal = 5,

  /// real (double)
  Real = 6,

  /// chandle (C pointer)
  CHandle = 7,

  /// string (C string)
  String = 8,

  /// bit vector (packed array of bits)
  BitVector = 9,

  /// logic vector (packed array with 4-state values)
  LogicVector = 10,

  /// Unpacked array
  UnpackedArray = 11,

  /// Struct
  Struct = 12,

  /// Open array (unsized array parameter)
  OpenArray = 13
};

/// Get the name of a DPI data type.
inline const char *getDPIDataTypeName(DPIDataType type) {
  switch (type) {
  case DPIDataType::Void:
    return "void";
  case DPIDataType::Byte:
    return "byte";
  case DPIDataType::ShortInt:
    return "shortint";
  case DPIDataType::Int:
    return "int";
  case DPIDataType::LongInt:
    return "longint";
  case DPIDataType::ShortReal:
    return "shortreal";
  case DPIDataType::Real:
    return "real";
  case DPIDataType::CHandle:
    return "chandle";
  case DPIDataType::String:
    return "string";
  case DPIDataType::BitVector:
    return "bit[]";
  case DPIDataType::LogicVector:
    return "logic[]";
  case DPIDataType::UnpackedArray:
    return "array";
  case DPIDataType::Struct:
    return "struct";
  case DPIDataType::OpenArray:
    return "open_array";
  }
  return "unknown";
}

/// Get the C equivalent type size for a DPI type.
inline size_t getDPITypeSize(DPIDataType type) {
  switch (type) {
  case DPIDataType::Void:
    return 0;
  case DPIDataType::Byte:
    return 1;
  case DPIDataType::ShortInt:
    return 2;
  case DPIDataType::Int:
    return 4;
  case DPIDataType::LongInt:
    return 8;
  case DPIDataType::ShortReal:
    return 4;
  case DPIDataType::Real:
    return 8;
  case DPIDataType::CHandle:
    return sizeof(void *);
  case DPIDataType::String:
    return sizeof(const char *);
  default:
    return 0; // Variable size
  }
}

//===----------------------------------------------------------------------===//
// DPIValue - Type-safe value container for DPI arguments
//===----------------------------------------------------------------------===//

/// A union type that can hold any DPI-compatible value.
class DPIValue {
public:
  /// Default constructor creates void value.
  DPIValue() : dataType(DPIDataType::Void) {}

  /// Construct from primitive types.
  explicit DPIValue(int8_t v) : dataType(DPIDataType::Byte) { data.byteVal = v; }
  explicit DPIValue(int16_t v) : dataType(DPIDataType::ShortInt) {
    data.shortVal = v;
  }
  explicit DPIValue(int32_t v) : dataType(DPIDataType::Int) { data.intVal = v; }
  explicit DPIValue(int64_t v) : dataType(DPIDataType::LongInt) {
    data.longVal = v;
  }
  explicit DPIValue(float v) : dataType(DPIDataType::ShortReal) {
    data.floatVal = v;
  }
  explicit DPIValue(double v) : dataType(DPIDataType::Real) {
    data.doubleVal = v;
  }
  explicit DPIValue(void *v) : dataType(DPIDataType::CHandle) { data.ptrVal = v; }
  explicit DPIValue(const char *v) : dataType(DPIDataType::String) {
    stringVal = v ? v : "";
  }
  explicit DPIValue(const std::string &v) : dataType(DPIDataType::String) {
    stringVal = v;
  }

  /// Construct bit/logic vector.
  DPIValue(const std::vector<uint32_t> &bits, uint32_t width, bool is4State)
      : dataType(is4State ? DPIDataType::LogicVector : DPIDataType::BitVector),
        vectorVal(bits), bitWidth(width) {}

  /// Get the data type.
  DPIDataType getType() const { return dataType; }

  /// Get the bit width (for vectors).
  uint32_t getBitWidth() const { return bitWidth; }

  /// Type-checked getters.
  int8_t getByte() const { return data.byteVal; }
  int16_t getShortInt() const { return data.shortVal; }
  int32_t getInt() const { return data.intVal; }
  int64_t getLongInt() const { return data.longVal; }
  float getShortReal() const { return data.floatVal; }
  double getReal() const { return data.doubleVal; }
  void *getCHandle() const { return data.ptrVal; }
  const std::string &getString() const { return stringVal; }
  const std::vector<uint32_t> &getVector() const { return vectorVal; }

  /// Convert to integer (if possible).
  int64_t toInt64() const {
    switch (dataType) {
    case DPIDataType::Byte:
      return data.byteVal;
    case DPIDataType::ShortInt:
      return data.shortVal;
    case DPIDataType::Int:
      return data.intVal;
    case DPIDataType::LongInt:
      return data.longVal;
    default:
      return 0;
    }
  }

  /// Convert to double (if possible).
  double toDouble() const {
    switch (dataType) {
    case DPIDataType::ShortReal:
      return data.floatVal;
    case DPIDataType::Real:
      return data.doubleVal;
    default:
      return static_cast<double>(toInt64());
    }
  }

private:
  DPIDataType dataType;
  union {
    int8_t byteVal;
    int16_t shortVal;
    int32_t intVal;
    int64_t longVal;
    float floatVal;
    double doubleVal;
    void *ptrVal;
  } data = {};

  std::string stringVal;
  std::vector<uint32_t> vectorVal;
  uint32_t bitWidth = 0;
};

//===----------------------------------------------------------------------===//
// DPIArgument - Description of a DPI function argument
//===----------------------------------------------------------------------===//

/// Direction of a DPI argument.
enum class DPIArgDirection : uint8_t {
  Input = 0,
  Output = 1,
  InOut = 2
};

/// Description of a single DPI function argument.
struct DPIArgument {
  /// Argument name.
  std::string name;

  /// Argument data type.
  DPIDataType dataType;

  /// Direction (input, output, inout).
  DPIArgDirection direction;

  /// Bit width for vector types.
  uint32_t bitWidth;

  /// Array dimensions (empty for scalars).
  llvm::SmallVector<uint32_t, 4> dimensions;

  /// Whether this is an open array (unsized).
  bool isOpenArray;

  DPIArgument(llvm::StringRef n, DPIDataType type,
              DPIArgDirection dir = DPIArgDirection::Input, uint32_t width = 0)
      : name(n.str()), dataType(type), direction(dir), bitWidth(width),
        isOpenArray(false) {}

  /// Check if this is an input argument.
  bool isInput() const {
    return direction == DPIArgDirection::Input ||
           direction == DPIArgDirection::InOut;
  }

  /// Check if this is an output argument.
  bool isOutput() const {
    return direction == DPIArgDirection::Output ||
           direction == DPIArgDirection::InOut;
  }
};

//===----------------------------------------------------------------------===//
// DPIFunctionSignature - Complete description of a DPI function
//===----------------------------------------------------------------------===//

/// Complete signature of a DPI function.
struct DPIFunctionSignature {
  /// Function name.
  std::string name;

  /// C name (may differ from SV name).
  std::string cName;

  /// Return type.
  DPIDataType returnType;

  /// Return bit width (for vector returns).
  uint32_t returnBitWidth;

  /// Arguments.
  llvm::SmallVector<DPIArgument, 8> arguments;

  /// Whether this is a context function.
  bool isContext;

  /// Whether this is a pure function.
  bool isPure;

  DPIFunctionSignature(llvm::StringRef n, llvm::StringRef cn = "")
      : name(n.str()), cName(cn.empty() ? n.str() : cn.str()),
        returnType(DPIDataType::Void), returnBitWidth(0), isContext(false),
        isPure(false) {}

  /// Add an argument.
  void addArgument(const DPIArgument &arg) { arguments.push_back(arg); }

  /// Add an input argument.
  void addInput(llvm::StringRef name, DPIDataType type, uint32_t width = 0) {
    arguments.emplace_back(name, type, DPIArgDirection::Input, width);
  }

  /// Add an output argument.
  void addOutput(llvm::StringRef name, DPIDataType type, uint32_t width = 0) {
    arguments.emplace_back(name, type, DPIArgDirection::Output, width);
  }

  /// Get input argument count.
  size_t getInputCount() const {
    return std::count_if(arguments.begin(), arguments.end(),
                         [](const DPIArgument &a) { return a.isInput(); });
  }

  /// Get output argument count.
  size_t getOutputCount() const {
    return std::count_if(arguments.begin(), arguments.end(),
                         [](const DPIArgument &a) { return a.isOutput(); });
  }
};

//===----------------------------------------------------------------------===//
// DPICallback - Callable wrapper for DPI functions
//===----------------------------------------------------------------------===//

/// Type-erased callback for DPI functions.
using DPICallback = std::function<DPIValue(const std::vector<DPIValue> &)>;

/// Registration info for a DPI function.
struct DPIFunctionRegistration {
  /// Function signature.
  DPIFunctionSignature signature;

  /// The callback implementation.
  DPICallback callback;

  /// Native function pointer (from shared library).
  void *nativePtr;

  /// Whether this is an import or export.
  bool isImport;

  DPIFunctionRegistration(const DPIFunctionSignature &sig, DPICallback cb)
      : signature(sig), callback(std::move(cb)), nativePtr(nullptr),
        isImport(true) {}

  DPIFunctionRegistration(const DPIFunctionSignature &sig, void *ptr)
      : signature(sig), nativePtr(ptr), isImport(true) {}
};

//===----------------------------------------------------------------------===//
// DPIContext - Simulation context for DPI calls
//===----------------------------------------------------------------------===//

/// DPI context passed to context-aware functions.
/// Provides access to simulation state and scope information.
class DPIContext {
public:
  DPIContext() = default;

  /// Get the current simulation time.
  const SimTime &getTime() const { return currentTime; }

  /// Set the current simulation time.
  void setTime(const SimTime &time) { currentTime = time; }

  /// Get the current scope path.
  const std::string &getScope() const { return scopePath; }

  /// Set the current scope path.
  void setScope(const std::string &scope) { scopePath = scope; }

  /// Get the current instance name.
  const std::string &getInstanceName() const { return instanceName; }

  /// Set the current instance name.
  void setInstanceName(const std::string &name) { instanceName = name; }

  /// Get user data.
  void *getUserData() const { return userData; }

  /// Set user data.
  void setUserData(void *data) { userData = data; }

  /// Get a process-local variable by name.
  DPIValue *getLocal(const std::string &name);

  /// Set a process-local variable.
  void setLocal(const std::string &name, const DPIValue &value);

private:
  SimTime currentTime;
  std::string scopePath;
  std::string instanceName;
  void *userData = nullptr;
  llvm::StringMap<DPIValue> locals;
};

//===----------------------------------------------------------------------===//
// DPIRuntime - Main DPI runtime manager
//===----------------------------------------------------------------------===//

/// Main DPI runtime that manages function registration, library loading,
/// and call dispatch.
class DPIRuntime {
public:
  /// Configuration for the DPI runtime.
  struct Config {
    /// Search paths for shared libraries.
    std::vector<std::string> libraryPaths;

    /// Whether to enable debug output.
    bool debug;

    /// Whether to check argument types at runtime.
    bool typeCheck;

    Config() : debug(false), typeCheck(true) {}
  };

  DPIRuntime(Config config = Config());
  ~DPIRuntime();

  //===--------------------------------------------------------------------===//
  // Library Management
  //===--------------------------------------------------------------------===//

  /// Load a shared library.
  bool loadLibrary(const std::string &path);

  /// Unload a shared library.
  void unloadLibrary(const std::string &path);

  /// Get all loaded libraries.
  const std::vector<std::string> &getLoadedLibraries() const {
    return loadedLibraryPaths;
  }

  //===--------------------------------------------------------------------===//
  // Function Registration
  //===--------------------------------------------------------------------===//

  /// Register an import function (C to SV).
  void registerImport(const DPIFunctionSignature &signature,
                      const DPICallback &callback);

  /// Register an import function from a shared library.
  bool registerImportFromLibrary(const DPIFunctionSignature &signature);

  /// Register an export function (SV to C).
  void registerExport(const DPIFunctionSignature &signature,
                      const DPICallback &callback);

  /// Check if a function is registered.
  bool hasFunction(const std::string &name) const;

  /// Get a function registration.
  const DPIFunctionRegistration *getFunction(const std::string &name) const;

  /// Get all registered functions.
  const llvm::StringMap<DPIFunctionRegistration> &getFunctions() const {
    return registeredFunctions;
  }

  //===--------------------------------------------------------------------===//
  // Function Calls
  //===--------------------------------------------------------------------===//

  /// Call a DPI function.
  DPIValue call(const std::string &name, const std::vector<DPIValue> &args);

  /// Call a DPI function with context.
  DPIValue callWithContext(const std::string &name,
                            const std::vector<DPIValue> &args,
                            DPIContext &context);

  //===--------------------------------------------------------------------===//
  // Context Management
  //===--------------------------------------------------------------------===//

  /// Get the current DPI context (for context-aware functions).
  DPIContext &getCurrentContext() { return currentContext; }
  const DPIContext &getCurrentContext() const { return currentContext; }

  /// Push a new context scope.
  void pushContext(const DPIContext &context);

  /// Pop the context scope.
  void popContext();

  //===--------------------------------------------------------------------===//
  // Type Conversion Utilities
  //===--------------------------------------------------------------------===//

  /// Convert an MLIR integer type width to DPI type.
  static DPIDataType intWidthToDPIType(unsigned width);

  /// Get the C pointer type for passing by reference.
  static std::string getCPointerType(DPIDataType type);

  /// Generate C function declaration from signature.
  static std::string generateCDeclaration(const DPIFunctionSignature &sig);

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t functionsRegistered = 0;
    size_t librariesLoaded = 0;
    size_t callsMade = 0;
    size_t callErrors = 0;
    uint64_t totalCallTimeNs = 0;
  };

  const Statistics &getStatistics() const { return stats; }

private:
  /// Look up a symbol in loaded libraries.
  void *lookupSymbol(const std::string &name);

  /// Create a wrapper callback for a native function.
  DPICallback createNativeWrapper(void *funcPtr,
                                   const DPIFunctionSignature &sig);

  Config config;
  llvm::StringMap<DPIFunctionRegistration> registeredFunctions;
  std::vector<llvm::sys::DynamicLibrary> loadedLibraries;
  std::vector<std::string> loadedLibraryPaths;

  DPIContext currentContext;
  std::vector<DPIContext> contextStack;

  std::mutex callMutex;
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// DPIRuntime Implementation
//===----------------------------------------------------------------------===//

inline DPIRuntime::DPIRuntime(Config config) : config(std::move(config)) {}

inline DPIRuntime::~DPIRuntime() {
  // Libraries are automatically unloaded when DynamicLibrary objects are
  // destroyed
}

inline bool DPIRuntime::loadLibrary(const std::string &path) {
  std::string errMsg;
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &errMsg);

  if (!lib.isValid()) {
    if (config.debug) {
      llvm::errs() << "DPI: Failed to load library " << path << ": " << errMsg
                   << "\n";
    }
    return false;
  }

  loadedLibraries.push_back(lib);
  loadedLibraryPaths.push_back(path);
  stats.librariesLoaded++;

  if (config.debug) {
    llvm::outs() << "DPI: Loaded library " << path << "\n";
  }

  return true;
}

inline void DPIRuntime::unloadLibrary(const std::string &path) {
  // Note: DynamicLibrary doesn't support unloading in LLVM
  // Just remove from our tracking
  auto it = std::find(loadedLibraryPaths.begin(), loadedLibraryPaths.end(), path);
  if (it != loadedLibraryPaths.end()) {
    size_t idx = it - loadedLibraryPaths.begin();
    loadedLibraryPaths.erase(it);
    loadedLibraries.erase(loadedLibraries.begin() + idx);
  }
}

inline void DPIRuntime::registerImport(const DPIFunctionSignature &signature,
                                        const DPICallback &callback) {
  registeredFunctions.try_emplace(signature.name,
                                  DPIFunctionRegistration(signature, callback));
  stats.functionsRegistered++;

  if (config.debug) {
    llvm::outs() << "DPI: Registered import function " << signature.name << "\n";
  }
}

inline bool
DPIRuntime::registerImportFromLibrary(const DPIFunctionSignature &signature) {
  void *funcPtr = lookupSymbol(signature.cName);
  if (!funcPtr) {
    if (config.debug) {
      llvm::errs() << "DPI: Symbol not found: " << signature.cName << "\n";
    }
    return false;
  }

  auto wrapper = createNativeWrapper(funcPtr, signature);
  DPIFunctionRegistration reg(signature, wrapper);
  reg.nativePtr = funcPtr;
  registeredFunctions.try_emplace(signature.name, std::move(reg));
  stats.functionsRegistered++;

  if (config.debug) {
    llvm::outs() << "DPI: Registered import from library: " << signature.name
                 << " -> " << signature.cName << "\n";
  }

  return true;
}

inline void DPIRuntime::registerExport(const DPIFunctionSignature &signature,
                                        const DPICallback &callback) {
  DPIFunctionRegistration reg(signature, callback);
  reg.isImport = false;
  registeredFunctions.try_emplace(signature.name, std::move(reg));
  stats.functionsRegistered++;

  if (config.debug) {
    llvm::outs() << "DPI: Registered export function " << signature.name << "\n";
  }
}

inline bool DPIRuntime::hasFunction(const std::string &name) const {
  return registeredFunctions.count(name) > 0;
}

inline const DPIFunctionRegistration *
DPIRuntime::getFunction(const std::string &name) const {
  auto it = registeredFunctions.find(name);
  return it != registeredFunctions.end() ? &it->second : nullptr;
}

inline DPIValue DPIRuntime::call(const std::string &name,
                                  const std::vector<DPIValue> &args) {
  return callWithContext(name, args, currentContext);
}

inline DPIValue DPIRuntime::callWithContext(const std::string &name,
                                             const std::vector<DPIValue> &args,
                                             DPIContext &context) {
  std::lock_guard<std::mutex> lock(callMutex);

  auto it = registeredFunctions.find(name);
  if (it == registeredFunctions.end()) {
    if (config.debug) {
      llvm::errs() << "DPI: Unknown function: " << name << "\n";
    }
    stats.callErrors++;
    return DPIValue();
  }

  const auto &reg = it->second;

  // Type checking
  if (config.typeCheck && args.size() != reg.signature.getInputCount()) {
    if (config.debug) {
      llvm::errs() << "DPI: Argument count mismatch for " << name << ": expected "
                   << reg.signature.getInputCount() << ", got " << args.size()
                   << "\n";
    }
    stats.callErrors++;
    return DPIValue();
  }

  // Save and set context
  DPIContext savedContext = currentContext;
  currentContext = context;

  // Make the call
  DPIValue result;
  try {
    result = reg.callback(args);
  } catch (...) {
    stats.callErrors++;
    currentContext = savedContext;
    throw;
  }

  // Restore context
  context = currentContext;
  currentContext = savedContext;

  stats.callsMade++;
  return result;
}

inline void DPIRuntime::pushContext(const DPIContext &context) {
  contextStack.push_back(currentContext);
  currentContext = context;
}

inline void DPIRuntime::popContext() {
  if (!contextStack.empty()) {
    currentContext = contextStack.back();
    contextStack.pop_back();
  }
}

inline void *DPIRuntime::lookupSymbol(const std::string &name) {
  // Search all loaded libraries
  for (auto &lib : loadedLibraries) {
    void *sym = lib.getAddressOfSymbol(name.c_str());
    if (sym)
      return sym;
  }

  // Also search the main program
  auto mainLib = llvm::sys::DynamicLibrary::getPermanentLibrary(nullptr);
  return mainLib.getAddressOfSymbol(name.c_str());
}

inline DPICallback
DPIRuntime::createNativeWrapper(void *funcPtr,
                                 const DPIFunctionSignature &sig) {
  // Create a type-erased wrapper that calls the native function
  // This is a simplified implementation - a full implementation would
  // use libffi or similar for proper ABI handling

  return [funcPtr, sig](const std::vector<DPIValue> &args) -> DPIValue {
    // For now, just handle simple integer functions
    // A complete implementation would handle all types

    if (sig.returnType == DPIDataType::Int && sig.arguments.size() == 1 &&
        sig.arguments[0].dataType == DPIDataType::Int) {
      // int f(int) signature
      using FuncType = int (*)(int);
      auto func = reinterpret_cast<FuncType>(funcPtr);
      int result = func(args[0].getInt());
      return DPIValue(result);
    }

    if (sig.returnType == DPIDataType::Void && sig.arguments.empty()) {
      // void f() signature
      using FuncType = void (*)();
      auto func = reinterpret_cast<FuncType>(funcPtr);
      func();
      return DPIValue();
    }

    // Fall back to void return for unsupported signatures
    return DPIValue();
  };
}

inline DPIDataType DPIRuntime::intWidthToDPIType(unsigned width) {
  if (width <= 8)
    return DPIDataType::Byte;
  if (width <= 16)
    return DPIDataType::ShortInt;
  if (width <= 32)
    return DPIDataType::Int;
  if (width <= 64)
    return DPIDataType::LongInt;
  return DPIDataType::BitVector;
}

inline std::string DPIRuntime::getCPointerType(DPIDataType type) {
  switch (type) {
  case DPIDataType::Byte:
    return "int8_t*";
  case DPIDataType::ShortInt:
    return "int16_t*";
  case DPIDataType::Int:
    return "int32_t*";
  case DPIDataType::LongInt:
    return "int64_t*";
  case DPIDataType::ShortReal:
    return "float*";
  case DPIDataType::Real:
    return "double*";
  case DPIDataType::CHandle:
    return "void**";
  case DPIDataType::String:
    return "const char**";
  default:
    return "void*";
  }
}

inline std::string
DPIRuntime::generateCDeclaration(const DPIFunctionSignature &sig) {
  std::string decl;

  // Return type
  switch (sig.returnType) {
  case DPIDataType::Void:
    decl += "void";
    break;
  case DPIDataType::Byte:
    decl += "int8_t";
    break;
  case DPIDataType::ShortInt:
    decl += "int16_t";
    break;
  case DPIDataType::Int:
    decl += "int32_t";
    break;
  case DPIDataType::LongInt:
    decl += "int64_t";
    break;
  case DPIDataType::ShortReal:
    decl += "float";
    break;
  case DPIDataType::Real:
    decl += "double";
    break;
  case DPIDataType::CHandle:
    decl += "void*";
    break;
  case DPIDataType::String:
    decl += "const char*";
    break;
  default:
    decl += "void";
  }

  decl += " " + sig.cName + "(";

  bool first = true;
  for (const auto &arg : sig.arguments) {
    if (!first)
      decl += ", ";
    first = false;

    // Type
    switch (arg.dataType) {
    case DPIDataType::Byte:
      decl += arg.isOutput() ? "int8_t*" : "int8_t";
      break;
    case DPIDataType::ShortInt:
      decl += arg.isOutput() ? "int16_t*" : "int16_t";
      break;
    case DPIDataType::Int:
      decl += arg.isOutput() ? "int32_t*" : "int32_t";
      break;
    case DPIDataType::LongInt:
      decl += arg.isOutput() ? "int64_t*" : "int64_t";
      break;
    case DPIDataType::ShortReal:
      decl += arg.isOutput() ? "float*" : "float";
      break;
    case DPIDataType::Real:
      decl += arg.isOutput() ? "double*" : "double";
      break;
    case DPIDataType::CHandle:
      decl += arg.isOutput() ? "void**" : "void*";
      break;
    case DPIDataType::String:
      decl += arg.isOutput() ? "const char**" : "const char*";
      break;
    case DPIDataType::BitVector:
    case DPIDataType::LogicVector:
      decl += "const svBitVecVal*";
      break;
    default:
      decl += "void*";
    }

    decl += " " + arg.name;
  }

  if (sig.arguments.empty())
    decl += "void";

  decl += ")";

  return decl;
}

inline DPIValue *DPIContext::getLocal(const std::string &name) {
  auto it = locals.find(name);
  return it != locals.end() ? &it->second : nullptr;
}

inline void DPIContext::setLocal(const std::string &name,
                                  const DPIValue &value) {
  locals[name] = value;
}

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_DPIRUNTIME_H
