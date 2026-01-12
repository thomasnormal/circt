//===- VCDWriter.h - VCD Waveform Writer ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the VCD (Value Change Dump) writer for capturing
// signal waveforms during simulation debugging.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_DEBUG_VCDWRITER_H
#define CIRCT_TOOLS_CIRCT_DEBUG_VCDWRITER_H

#include "circt/Tools/circt-debug/Debug.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace debug {

//===----------------------------------------------------------------------===//
// VCD Signal Identifier
//===----------------------------------------------------------------------===//

/// Generates unique VCD signal identifiers.
class VCDIdentifierGenerator {
public:
  VCDIdentifierGenerator();

  /// Get the next unique identifier.
  std::string next();

  /// Reset the generator.
  void reset();

private:
  std::vector<char> current;
  static constexpr char firstChar = '!';
  static constexpr char lastChar = '~';
};

//===----------------------------------------------------------------------===//
// VCD Writer
//===----------------------------------------------------------------------===//

/// Writes VCD format waveform files.
class VCDWriter {
public:
  VCDWriter(llvm::raw_ostream &os);
  ~VCDWriter();

  //==========================================================================
  // Header Section
  //==========================================================================

  /// Write the VCD header.
  void writeHeader(StringRef comment = "", StringRef version = "CIRCT Debug",
                   StringRef timescale = "1ns");

  /// Begin a scope definition.
  void beginScope(StringRef name, StringRef type = "module");

  /// End the current scope.
  void endScope();

  /// Add a signal definition.
  /// Returns the VCD identifier for this signal.
  std::string addSignal(StringRef name, unsigned width, StringRef type = "wire",
                        StringRef fullPath = "");

  /// Add a signal definition for a registered signal.
  std::string addRegSignal(StringRef name, unsigned width,
                           StringRef fullPath = "");

  /// Add a signal definition for a wire.
  std::string addWireSignal(StringRef name, unsigned width,
                            StringRef fullPath = "");

  /// Finish the header section.
  void finishHeader();

  //==========================================================================
  // Initial Values
  //==========================================================================

  /// Begin the $dumpvars section.
  void beginDumpVars();

  /// Write an initial value.
  void writeInitialValue(StringRef identifier, const SignalValue &value);

  /// End the $dumpvars section.
  void endDumpVars();

  //==========================================================================
  // Value Changes
  //==========================================================================

  /// Write a timestamp.
  void writeTimestamp(uint64_t time);

  /// Write a value change.
  void writeValueChange(StringRef identifier, const SignalValue &value);

  /// Write a comment.
  void writeComment(StringRef comment);

  //==========================================================================
  // Convenience Methods
  //==========================================================================

  /// Write value changes for all tracked signals from simulation state.
  void writeAllChanges(const SimState &state);

  /// Get the identifier for a signal path.
  std::optional<StringRef> getIdentifier(StringRef path) const;

  /// Check if a signal is registered.
  bool hasSignal(StringRef path) const;

  //==========================================================================
  // State
  //==========================================================================

  /// Flush output.
  void flush();

  /// Get current time.
  uint64_t getCurrentTime() const { return currentTime; }

private:
  /// Format a signal value for VCD.
  void formatValue(const SignalValue &value, llvm::raw_ostream &os);

  /// Format a scalar value (1-bit).
  char formatScalar(LogicValue value);

  llvm::raw_ostream &os;
  VCDIdentifierGenerator idGen;

  // Signal tracking
  struct SignalEntry {
    std::string identifier;
    unsigned width;
    std::optional<SignalValue> lastValue;
  };
  llvm::StringMap<SignalEntry> signals;

  // Current state
  uint64_t currentTime = 0;
  bool headerFinished = false;
  unsigned scopeDepth = 0;
};

//===----------------------------------------------------------------------===//
// VCD Ring Buffer
//===----------------------------------------------------------------------===//

/// Maintains a rolling buffer of signal values for snippet capture.
class VCDRingBuffer {
public:
  VCDRingBuffer(unsigned capacity);

  /// Record signal values at a time point.
  void record(const SimState &state);

  /// Get all recorded entries.
  struct Entry {
    SimTime time;
    llvm::StringMap<SignalValue> values;
  };
  std::vector<Entry> getEntries() const;

  /// Get entries within a time range.
  std::vector<Entry> getEntries(const SimTime &start, const SimTime &end) const;

  /// Clear the buffer.
  void clear();

  /// Get current size.
  unsigned size() const;

  /// Get capacity.
  unsigned getCapacity() const { return capacity; }

private:
  unsigned capacity;
  std::vector<Entry> buffer;
  unsigned writePos = 0;
  bool wrapped = false;
};

//===----------------------------------------------------------------------===//
// Waveform Snippet Capture
//===----------------------------------------------------------------------===//

/// Captures waveform snippets around interesting events.
class WaveformCapture {
public:
  WaveformCapture(unsigned preCapture = 10, unsigned postCapture = 10);

  /// Start capturing (call regularly during simulation).
  void capture(const SimState &state);

  /// Mark the current time as a trigger point.
  void trigger();

  /// Save the captured waveform to a VCD file.
  bool saveToVCD(StringRef filename, const Scope *rootScope = nullptr);

  /// Set the list of signals to capture (empty = all).
  void setSignalFilter(const std::vector<std::string> &signals);

  /// Clear captured data.
  void clear();

  /// Is a capture in progress (post-trigger)?
  bool isCapturing() const { return capturing; }

private:
  unsigned preCapture;
  unsigned postCapture;
  VCDRingBuffer preBuffer;
  std::vector<VCDRingBuffer::Entry> postBuffer;
  std::vector<std::string> signalFilter;
  bool capturing = false;
  unsigned postCaptureCount = 0;
  SimTime triggerTime;
};

} // namespace debug
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_DEBUG_VCDWRITER_H
