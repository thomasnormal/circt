//===- WaveformDumper.h - Waveform output infrastructure --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the waveform dumping infrastructure for simulation output.
// It supports:
// - VCD (Value Change Dump) format - IEEE 1364-2001
// - FST (Fast Signal Trace) compressed format
// - Selective signal tracing with hierarchical paths
// - Efficient incremental updates
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_WAVEFORMDUMPER_H
#define CIRCT_DIALECT_SIM_WAVEFORMDUMPER_H

#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// SignalType - Types of signals for waveform output
//===----------------------------------------------------------------------===//

/// Types of signals that can be traced.
enum class SignalType : uint8_t {
  /// Regular wire signal.
  Wire = 0,

  /// Register signal.
  Reg = 1,

  /// Integer variable.
  Integer = 2,

  /// Real (floating-point) variable.
  Real = 3,

  /// Event variable.
  Event = 4,

  /// Parameter (constant).
  Parameter = 5,

  /// String variable.
  String = 6
};

/// Get the VCD type string for a signal type.
inline const char *getVCDSignalType(SignalType type) {
  switch (type) {
  case SignalType::Wire:
    return "wire";
  case SignalType::Reg:
    return "reg";
  case SignalType::Integer:
    return "integer";
  case SignalType::Real:
    return "real";
  case SignalType::Event:
    return "event";
  case SignalType::Parameter:
    return "parameter";
  case SignalType::String:
    return "string";
  }
  return "wire";
}

//===----------------------------------------------------------------------===//
// TracedSignal - Information about a traced signal
//===----------------------------------------------------------------------===//

/// Information about a signal being traced.
struct TracedSignal {
  /// Unique identifier for the signal.
  uint64_t signalId;

  /// Hierarchical path to the signal.
  std::string path;

  /// Signal name (without path).
  std::string name;

  /// Bit width of the signal.
  uint32_t width;

  /// Type of the signal.
  SignalType type;

  /// VCD identifier character(s).
  std::string vcdId;

  /// Index for FST format.
  uint32_t fstHandle;

  /// Current value (for change detection).
  uint64_t currentValue;

  /// Whether the current value is unknown (X).
  bool isUnknown;

  /// Array index if this is part of an array (-1 if not).
  int32_t arrayIndex;

  TracedSignal(uint64_t id, llvm::StringRef p, llvm::StringRef n, uint32_t w,
               SignalType t = SignalType::Wire)
      : signalId(id), path(p.str()), name(n.str()), width(w), type(t),
        fstHandle(0), currentValue(0), isUnknown(true), arrayIndex(-1) {}
};

//===----------------------------------------------------------------------===//
// WaveformScope - Hierarchical scope for waveform organization
//===----------------------------------------------------------------------===//

/// Represents a hierarchical scope in the waveform (module/instance).
struct WaveformScope {
  /// Name of this scope.
  std::string name;

  /// Full hierarchical path.
  std::string path;

  /// Child scopes.
  llvm::SmallVector<std::unique_ptr<WaveformScope>, 4> children;

  /// Signals in this scope.
  llvm::SmallVector<uint64_t, 8> signals;

  /// Parent scope (null for root).
  WaveformScope *parent;

  /// Scope type (module, task, function, etc.).
  std::string scopeType;

  WaveformScope(llvm::StringRef n, llvm::StringRef p, WaveformScope *par = nullptr)
      : name(n.str()), path(p.str()), parent(par), scopeType("module") {}

  /// Find or create a child scope.
  WaveformScope *findOrCreateChild(llvm::StringRef childName) {
    for (auto &child : children) {
      if (child->name == childName)
        return child.get();
    }
    std::string childPath = path.empty() ? childName.str() : path + "." + childName.str();
    children.push_back(std::make_unique<WaveformScope>(childName, childPath, this));
    return children.back().get();
  }
};

//===----------------------------------------------------------------------===//
// WaveformFormat - Abstract base for waveform format writers
//===----------------------------------------------------------------------===//

/// Abstract base class for waveform format writers.
class WaveformFormat {
public:
  virtual ~WaveformFormat() = default;

  /// Open the output file.
  virtual bool open(const std::string &filename) = 0;

  /// Close the output file.
  virtual void close() = 0;

  /// Check if the file is open.
  virtual bool isOpen() const = 0;

  /// Write the file header.
  virtual void writeHeader(const std::string &date, const std::string &version,
                           const std::string &timescale) = 0;

  /// Begin a scope definition.
  virtual void beginScope(const std::string &type, const std::string &name) = 0;

  /// End a scope definition.
  virtual void endScope() = 0;

  /// Declare a signal.
  virtual void declareSignal(const TracedSignal &signal) = 0;

  /// End the header/definitions section.
  virtual void endDefinitions() = 0;

  /// Write a time change.
  virtual void writeTime(uint64_t time) = 0;

  /// Write a value change for a 1-bit signal.
  virtual void writeBitValue(const TracedSignal &signal, bool value) = 0;

  /// Write a value change for a multi-bit signal.
  virtual void writeVectorValue(const TracedSignal &signal, uint64_t value) = 0;

  /// Write an unknown (X) value.
  virtual void writeUnknownValue(const TracedSignal &signal) = 0;

  /// Write a high-impedance (Z) value.
  virtual void writeHighZValue(const TracedSignal &signal) = 0;

  /// Write a real (floating-point) value.
  virtual void writeRealValue(const TracedSignal &signal, double value) = 0;

  /// Write a string value.
  virtual void writeStringValue(const TracedSignal &signal,
                                 const std::string &value) = 0;

  /// Flush any buffered output.
  virtual void flush() = 0;
};

//===----------------------------------------------------------------------===//
// VCDFormat - VCD (Value Change Dump) format writer
//===----------------------------------------------------------------------===//

/// VCD format writer implementation.
class VCDFormat : public WaveformFormat {
public:
  VCDFormat() = default;
  ~VCDFormat() override { close(); }

  bool open(const std::string &filename) override {
    file.open(filename);
    return file.is_open();
  }

  void close() override {
    if (file.is_open()) {
      file.close();
    }
  }

  bool isOpen() const override { return file.is_open(); }

  void writeHeader(const std::string &date, const std::string &version,
                   const std::string &timescale) override {
    file << "$date\n  " << date << "\n$end\n";
    file << "$version\n  " << version << "\n$end\n";
    file << "$timescale\n  " << timescale << "\n$end\n";
  }

  void beginScope(const std::string &type, const std::string &name) override {
    file << "$scope " << type << " " << name << " $end\n";
  }

  void endScope() override { file << "$upscope $end\n"; }

  void declareSignal(const TracedSignal &signal) override {
    file << "$var " << getVCDSignalType(signal.type) << " " << signal.width
         << " " << signal.vcdId << " " << signal.name;
    if (signal.arrayIndex >= 0) {
      file << "[" << signal.arrayIndex << "]";
    }
    file << " $end\n";
  }

  void endDefinitions() override {
    file << "$enddefinitions $end\n";
    file << "$dumpvars\n";
  }

  void endDumpVars() { file << "$end\n"; }

  void writeTime(uint64_t time) override { file << "#" << time << "\n"; }

  void writeBitValue(const TracedSignal &signal, bool value) override {
    file << (value ? '1' : '0') << signal.vcdId << "\n";
  }

  void writeVectorValue(const TracedSignal &signal, uint64_t value) override {
    file << "b";
    for (int i = signal.width - 1; i >= 0; --i) {
      file << ((value >> i) & 1 ? '1' : '0');
    }
    file << " " << signal.vcdId << "\n";
  }

  void writeUnknownValue(const TracedSignal &signal) override {
    if (signal.width == 1) {
      file << "x" << signal.vcdId << "\n";
    } else {
      file << "b";
      for (uint32_t i = 0; i < signal.width; ++i) {
        file << "x";
      }
      file << " " << signal.vcdId << "\n";
    }
  }

  void writeHighZValue(const TracedSignal &signal) override {
    if (signal.width == 1) {
      file << "z" << signal.vcdId << "\n";
    } else {
      file << "b";
      for (uint32_t i = 0; i < signal.width; ++i) {
        file << "z";
      }
      file << " " << signal.vcdId << "\n";
    }
  }

  void writeRealValue(const TracedSignal &signal, double value) override {
    file << "r" << value << " " << signal.vcdId << "\n";
  }

  void writeStringValue(const TracedSignal &signal,
                         const std::string &value) override {
    file << "s" << value << " " << signal.vcdId << "\n";
  }

  void flush() override { file.flush(); }

private:
  std::ofstream file;
};

//===----------------------------------------------------------------------===//
// FSTFormat - FST (Fast Signal Trace) compressed format writer
//===----------------------------------------------------------------------===//

/// FST format writer implementation.
/// Note: This is a simplified implementation. For full FST support,
/// consider linking against the GTKWave FST library.
class FSTFormat : public WaveformFormat {
public:
  FSTFormat() : nextHandle(0) {}
  ~FSTFormat() override { close(); }

  bool open(const std::string &filename) override {
    // For now, fall back to VCD-like format with compression hints
    // A full implementation would use the GTKWave FST library
    file.open(filename, std::ios::binary);
    if (!file.is_open())
      return false;

    // Write FST magic header
    writeFSTHeader();
    return true;
  }

  void close() override {
    if (file.is_open()) {
      writeFSTFooter();
      file.close();
    }
  }

  bool isOpen() const override { return file.is_open(); }

  void writeHeader(const std::string &date, const std::string &version,
                   const std::string &timescale) override {
    // FST stores metadata in blocks
    writeString(date);
    writeString(version);
    writeString(timescale);
  }

  void beginScope(const std::string &type, const std::string &name) override {
    uint8_t scopeType = 0; // FST_ST_VCD_MODULE
    if (type == "task")
      scopeType = 1;
    else if (type == "function")
      scopeType = 2;

    file.put(static_cast<char>(scopeType));
    writeString(name);
  }

  void endScope() override {
    file.put(0xFF); // FST_ST_VCD_UPSCOPE
  }

  void declareSignal(const TracedSignal &signal) override {
    // Signal declaration block
    file.put(static_cast<char>(signal.type));
    writeVarint(signal.width);
    writeString(signal.name);
    signalHandles[signal.signalId] = nextHandle++;
  }

  void endDefinitions() override {
    // Write end of hierarchy marker
    file.put(0xFE);
  }

  void writeTime(uint64_t time) override {
    // Write time block
    file.put(0x01); // Time marker
    writeVarint(time);
  }

  void writeBitValue(const TracedSignal &signal, bool value) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x02); // Value change marker
    writeVarint(it->second);
    file.put(value ? 1 : 0);
  }

  void writeVectorValue(const TracedSignal &signal, uint64_t value) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x03); // Vector value marker
    writeVarint(it->second);
    writeVarint(value);
  }

  void writeUnknownValue(const TracedSignal &signal) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x04); // Unknown value marker
    writeVarint(it->second);
  }

  void writeHighZValue(const TracedSignal &signal) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x05); // High-Z value marker
    writeVarint(it->second);
  }

  void writeRealValue(const TracedSignal &signal, double value) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x06); // Real value marker
    writeVarint(it->second);
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }

  void writeStringValue(const TracedSignal &signal,
                         const std::string &value) override {
    auto it = signalHandles.find(signal.signalId);
    if (it == signalHandles.end())
      return;

    file.put(0x07); // String value marker
    writeVarint(it->second);
    writeString(value);
  }

  void flush() override { file.flush(); }

private:
  void writeFSTHeader() {
    // FST file header
    const char magic[] = "FST";
    file.write(magic, 3);
    uint8_t version = 1;
    file.put(version);
  }

  void writeFSTFooter() {
    // End marker
    file.put(0x00);
  }

  void writeVarint(uint64_t value) {
    while (value >= 0x80) {
      file.put(static_cast<char>((value & 0x7F) | 0x80));
      value >>= 7;
    }
    file.put(static_cast<char>(value));
  }

  void writeString(const std::string &str) {
    writeVarint(str.size());
    file.write(str.data(), str.size());
  }

  std::ofstream file;
  uint32_t nextHandle;
  llvm::DenseMap<uint64_t, uint32_t> signalHandles;
};

//===----------------------------------------------------------------------===//
// WaveformDumper - Main waveform dumping interface
//===----------------------------------------------------------------------===//

/// Configuration for waveform dumping.
struct WaveformDumperConfig {
  /// Whether to trace all signals or only selected ones.
  bool traceAll;

  /// Timescale string (e.g., "1fs", "1ps", "1ns").
  std::string timescale;

  /// Tool version string.
  std::string version;

  /// Buffer size for output (bytes).
  size_t bufferSize;

  /// Flush interval (number of time changes).
  size_t flushInterval;

  /// Enable compression (for FST).
  bool compress;

  WaveformDumperConfig()
      : traceAll(false), timescale("1fs"), version("CIRCT circt-sim"),
        bufferSize(65536), flushInterval(1000), compress(true) {}
};

/// Main waveform dumper class that manages signal tracing and output.
class WaveformDumper {
public:
  WaveformDumper(WaveformDumperConfig config = WaveformDumperConfig());
  ~WaveformDumper();

  //===--------------------------------------------------------------------===//
  // File Management
  //===--------------------------------------------------------------------===//

  /// Open a VCD file for output.
  bool openVCD(const std::string &filename);

  /// Open an FST file for output.
  bool openFST(const std::string &filename);

  /// Close the output file.
  void close();

  /// Check if a file is open.
  bool isOpen() const { return format && format->isOpen(); }

  //===--------------------------------------------------------------------===//
  // Scope Management
  //===--------------------------------------------------------------------===//

  /// Begin a new hierarchical scope.
  void beginScope(llvm::StringRef name, llvm::StringRef type = "module");

  /// End the current scope.
  void endScope();

  /// Get the current scope path.
  std::string getCurrentScopePath() const;

  //===--------------------------------------------------------------------===//
  // Signal Management
  //===--------------------------------------------------------------------===//

  /// Register a signal for tracing.
  /// Returns the signal handle for value updates.
  uint64_t registerSignal(llvm::StringRef name, uint32_t width,
                          SignalType type = SignalType::Wire);

  /// Register a signal with explicit path.
  uint64_t registerSignalWithPath(llvm::StringRef path, llvm::StringRef name,
                                   uint32_t width,
                                   SignalType type = SignalType::Wire);

  /// Add a signal to trace (by pattern, e.g., "*.clk").
  void addTracePattern(llvm::StringRef pattern);

  /// Check if a signal should be traced.
  bool shouldTrace(llvm::StringRef path) const;

  /// Get a traced signal by handle.
  TracedSignal *getSignal(uint64_t handle);
  const TracedSignal *getSignal(uint64_t handle) const;

  //===--------------------------------------------------------------------===//
  // Header/Footer Management
  //===--------------------------------------------------------------------===//

  /// Write the file header with all registered signals.
  void writeHeader();

  /// Write initial values for all signals.
  void writeInitialValues();

  //===--------------------------------------------------------------------===//
  // Value Updates
  //===--------------------------------------------------------------------===//

  /// Set the current simulation time.
  void setTime(uint64_t time);

  /// Update a 1-bit signal value.
  void updateBit(uint64_t handle, bool value);

  /// Update a multi-bit signal value.
  void updateVector(uint64_t handle, uint64_t value);

  /// Mark a signal as unknown (X).
  void updateUnknown(uint64_t handle);

  /// Mark a signal as high-impedance (Z).
  void updateHighZ(uint64_t handle);

  /// Update a real (floating-point) signal.
  void updateReal(uint64_t handle, double value);

  /// Update a string signal.
  void updateString(uint64_t handle, const std::string &value);

  /// Batch update for efficiency.
  void beginBatch();
  void endBatch();

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Flush buffered output.
  void flush();

  /// Get statistics.
  struct Statistics {
    size_t signalsRegistered = 0;
    size_t signalsTraced = 0;
    size_t valueChanges = 0;
    size_t timeChanges = 0;
    size_t bytesWritten = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Generate the next VCD identifier.
  std::string nextVCDIdentifier();

private:
  /// Write scope hierarchy recursively.
  void writeScopeHierarchy(WaveformScope *scope);

  WaveformDumperConfig config;
  std::unique_ptr<WaveformFormat> format;

  // Scope management
  std::unique_ptr<WaveformScope> rootScope;
  WaveformScope *currentScope;
  std::vector<WaveformScope *> scopeStack;

  // Signal management
  llvm::DenseMap<uint64_t, std::unique_ptr<TracedSignal>> signals;
  uint64_t nextSignalHandle;
  uint32_t vcdIdCounter;

  // Trace patterns
  llvm::SmallVector<std::string, 8> tracePatterns;

  // Time tracking
  uint64_t currentTime;
  bool headerWritten;
  size_t changesSinceFlush;

  // Batching
  bool inBatch;

  // Statistics
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// WaveformDumper Implementation (inline for header-only use)
//===----------------------------------------------------------------------===//

inline WaveformDumper::WaveformDumper(WaveformDumperConfig config)
    : config(std::move(config)), currentScope(nullptr), nextSignalHandle(1),
      vcdIdCounter(0), currentTime(0), headerWritten(false),
      changesSinceFlush(0), inBatch(false) {
  rootScope = std::make_unique<WaveformScope>("", "");
  currentScope = rootScope.get();
}

inline WaveformDumper::~WaveformDumper() { close(); }

inline bool WaveformDumper::openVCD(const std::string &filename) {
  format = std::make_unique<VCDFormat>();
  return format->open(filename);
}

inline bool WaveformDumper::openFST(const std::string &filename) {
  format = std::make_unique<FSTFormat>();
  return format->open(filename);
}

inline void WaveformDumper::close() {
  if (format) {
    flush();
    format->close();
    format.reset();
  }
}

inline void WaveformDumper::beginScope(llvm::StringRef name,
                                        llvm::StringRef type) {
  scopeStack.push_back(currentScope);
  currentScope = currentScope->findOrCreateChild(name);
  currentScope->scopeType = type.str();
}

inline void WaveformDumper::endScope() {
  if (!scopeStack.empty()) {
    currentScope = scopeStack.back();
    scopeStack.pop_back();
  }
}

inline std::string WaveformDumper::getCurrentScopePath() const {
  return currentScope ? currentScope->path : "";
}

inline uint64_t WaveformDumper::registerSignal(llvm::StringRef name,
                                                uint32_t width,
                                                SignalType type) {
  std::string path = getCurrentScopePath();
  if (!path.empty())
    path += ".";
  path += name.str();

  return registerSignalWithPath(path, name, width, type);
}

inline uint64_t WaveformDumper::registerSignalWithPath(llvm::StringRef path,
                                                        llvm::StringRef name,
                                                        uint32_t width,
                                                        SignalType type) {
  // Check if this signal should be traced
  if (!config.traceAll && !shouldTrace(path))
    return 0;

  uint64_t handle = nextSignalHandle++;
  auto signal =
      std::make_unique<TracedSignal>(handle, path, name, width, type);
  signal->vcdId = nextVCDIdentifier();

  currentScope->signals.push_back(handle);
  signals[handle] = std::move(signal);

  stats.signalsRegistered++;
  stats.signalsTraced++;

  return handle;
}

inline void WaveformDumper::addTracePattern(llvm::StringRef pattern) {
  tracePatterns.push_back(pattern.str());
}

inline bool WaveformDumper::shouldTrace(llvm::StringRef path) const {
  if (config.traceAll)
    return true;

  if (tracePatterns.empty())
    return true; // Trace all if no patterns specified

  for (const auto &pattern : tracePatterns) {
    // Simple wildcard matching
    if (pattern == "*")
      return true;

    if (pattern.back() == '*') {
      // Prefix match
      llvm::StringRef prefix(pattern.data(), pattern.size() - 1);
      if (path.starts_with(prefix))
        return true;
    } else if (pattern.front() == '*') {
      // Suffix match
      llvm::StringRef suffix(pattern.data() + 1, pattern.size() - 1);
      if (path.ends_with(suffix))
        return true;
    } else if (path == pattern) {
      // Exact match
      return true;
    }
  }

  return false;
}

inline TracedSignal *WaveformDumper::getSignal(uint64_t handle) {
  auto it = signals.find(handle);
  return it != signals.end() ? it->second.get() : nullptr;
}

inline const TracedSignal *WaveformDumper::getSignal(uint64_t handle) const {
  auto it = signals.find(handle);
  return it != signals.end() ? it->second.get() : nullptr;
}

inline void WaveformDumper::writeHeader() {
  if (!format || headerWritten)
    return;

  // Get current date
  auto now = std::chrono::system_clock::now();
  auto timeT = std::chrono::system_clock::to_time_t(now);
  std::string dateStr = std::ctime(&timeT);
  if (!dateStr.empty() && dateStr.back() == '\n')
    dateStr.pop_back();

  format->writeHeader(dateStr, config.version, config.timescale);

  // Write scope hierarchy
  writeScopeHierarchy(rootScope.get());

  format->endDefinitions();
  headerWritten = true;
}

inline void WaveformDumper::writeScopeHierarchy(WaveformScope *scope) {
  if (!scope || scope == rootScope.get()) {
    // Root scope - just process children
    for (auto &child : scope->children) {
      writeScopeHierarchy(child.get());
    }
    return;
  }

  format->beginScope(scope->scopeType, scope->name);

  // Declare signals in this scope
  for (auto signalHandle : scope->signals) {
    auto *signal = getSignal(signalHandle);
    if (signal) {
      format->declareSignal(*signal);
    }
  }

  // Process child scopes
  for (auto &child : scope->children) {
    writeScopeHierarchy(child.get());
  }

  format->endScope();
}

inline void WaveformDumper::writeInitialValues() {
  if (!format)
    return;

  format->writeTime(0);
  for (auto &entry : signals) {
    auto *signal = entry.second.get();
    format->writeUnknownValue(*signal);
  }

  // End $dumpvars section for VCD
  if (auto *vcd = dynamic_cast<VCDFormat *>(format.get())) {
    vcd->endDumpVars();
  }
}

inline void WaveformDumper::setTime(uint64_t time) {
  if (!format || time == currentTime)
    return;

  currentTime = time;
  format->writeTime(time);
  stats.timeChanges++;

  changesSinceFlush++;
  if (changesSinceFlush >= config.flushInterval) {
    flush();
  }
}

inline void WaveformDumper::updateBit(uint64_t handle, bool value) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  // Check for actual change
  if (!signal->isUnknown && signal->currentValue == (value ? 1 : 0))
    return;

  signal->currentValue = value ? 1 : 0;
  signal->isUnknown = false;
  format->writeBitValue(*signal, value);
  stats.valueChanges++;
}

inline void WaveformDumper::updateVector(uint64_t handle, uint64_t value) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  // Check for actual change
  if (!signal->isUnknown && signal->currentValue == value)
    return;

  signal->currentValue = value;
  signal->isUnknown = false;
  format->writeVectorValue(*signal, value);
  stats.valueChanges++;
}

inline void WaveformDumper::updateUnknown(uint64_t handle) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  if (signal->isUnknown)
    return;

  signal->isUnknown = true;
  format->writeUnknownValue(*signal);
  stats.valueChanges++;
}

inline void WaveformDumper::updateHighZ(uint64_t handle) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  format->writeHighZValue(*signal);
  stats.valueChanges++;
}

inline void WaveformDumper::updateReal(uint64_t handle, double value) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  format->writeRealValue(*signal, value);
  stats.valueChanges++;
}

inline void WaveformDumper::updateString(uint64_t handle,
                                          const std::string &value) {
  auto *signal = getSignal(handle);
  if (!signal || !format)
    return;

  format->writeStringValue(*signal, value);
  stats.valueChanges++;
}

inline void WaveformDumper::beginBatch() { inBatch = true; }

inline void WaveformDumper::endBatch() {
  inBatch = false;
  if (changesSinceFlush >= config.flushInterval) {
    flush();
  }
}

inline void WaveformDumper::flush() {
  if (format) {
    format->flush();
    changesSinceFlush = 0;
  }
}

inline std::string WaveformDumper::nextVCDIdentifier() {
  // Generate VCD identifier using printable ASCII characters
  // Range: '!' (33) to '~' (126), 94 characters
  std::string id;
  uint32_t n = vcdIdCounter++;

  do {
    id += static_cast<char>('!' + (n % 94));
    n /= 94;
  } while (n > 0);

  return id;
}

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_WAVEFORMDUMPER_H
