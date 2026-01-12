//===- VCDWriter.cpp - VCD Waveform Writer Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-debug/VCDWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <ctime>

using namespace circt;
using namespace circt::debug;

//===----------------------------------------------------------------------===//
// VCDIdentifierGenerator Implementation
//===----------------------------------------------------------------------===//

VCDIdentifierGenerator::VCDIdentifierGenerator() { current.push_back(firstChar); }

std::string VCDIdentifierGenerator::next() {
  std::string result(current.begin(), current.end());

  // Increment like a counter
  bool carry = true;
  for (int i = current.size() - 1; i >= 0 && carry; --i) {
    if (current[i] < lastChar) {
      ++current[i];
      // Skip problematic characters
      if (current[i] == '$')
        current[i] = '%';
      carry = false;
    } else {
      current[i] = firstChar;
    }
  }

  if (carry) {
    current.insert(current.begin(), firstChar);
  }

  return result;
}

void VCDIdentifierGenerator::reset() {
  current.clear();
  current.push_back(firstChar);
}

//===----------------------------------------------------------------------===//
// VCDWriter Implementation
//===----------------------------------------------------------------------===//

VCDWriter::VCDWriter(llvm::raw_ostream &os) : os(os) {}

VCDWriter::~VCDWriter() { flush(); }

void VCDWriter::writeHeader(StringRef comment, StringRef version,
                            StringRef timescale) {
  // Get current date/time
  std::time_t now = std::time(nullptr);
  char dateStr[64];
  std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d %H:%M:%S",
                std::localtime(&now));

  os << "$date\n   " << dateStr << "\n$end\n";

  if (!comment.empty())
    os << "$comment\n   " << comment << "\n$end\n";

  os << "$version\n   " << version << "\n$end\n";
  os << "$timescale " << timescale << " $end\n";
}

void VCDWriter::beginScope(StringRef name, StringRef type) {
  os << "$scope " << type << " " << name << " $end\n";
  ++scopeDepth;
}

void VCDWriter::endScope() {
  if (scopeDepth > 0) {
    os << "$upscope $end\n";
    --scopeDepth;
  }
}

std::string VCDWriter::addSignal(StringRef name, unsigned width, StringRef type,
                                 StringRef fullPath) {
  std::string id = idGen.next();

  SignalEntry entry;
  entry.identifier = id;
  entry.width = width;
  entry.lastValue = std::nullopt;

  std::string path = fullPath.empty() ? name.str() : fullPath.str();
  signals[path] = std::move(entry);

  os << "$var " << type << " " << width << " " << id << " " << name;
  if (width > 1)
    os << " [" << (width - 1) << ":0]";
  os << " $end\n";

  return id;
}

std::string VCDWriter::addRegSignal(StringRef name, unsigned width,
                                    StringRef fullPath) {
  return addSignal(name, width, "reg", fullPath);
}

std::string VCDWriter::addWireSignal(StringRef name, unsigned width,
                                     StringRef fullPath) {
  return addSignal(name, width, "wire", fullPath);
}

void VCDWriter::finishHeader() {
  // Close any remaining scopes
  while (scopeDepth > 0)
    endScope();

  os << "$enddefinitions $end\n";
  headerFinished = true;
}

void VCDWriter::beginDumpVars() { os << "$dumpvars\n"; }

void VCDWriter::writeInitialValue(StringRef identifier,
                                  const SignalValue &value) {
  writeValueChange(identifier, value);
}

void VCDWriter::endDumpVars() { os << "$end\n"; }

void VCDWriter::writeTimestamp(uint64_t time) {
  if (time != currentTime) {
    os << "#" << time << "\n";
    currentTime = time;
  }
}

void VCDWriter::writeValueChange(StringRef identifier,
                                 const SignalValue &value) {
  // Find the signal to get its width
  unsigned width = value.getWidth();

  // Find by identifier
  for (auto &entry : signals) {
    if (entry.second.identifier == identifier) {
      width = entry.second.width;
      entry.second.lastValue = value;
      break;
    }
  }

  if (width == 1) {
    // Scalar value
    os << formatScalar(value.getBit(0)) << identifier << "\n";
  } else {
    // Vector value
    os << "b";
    formatValue(value, os);
    os << " " << identifier << "\n";
  }
}

void VCDWriter::writeComment(StringRef comment) {
  os << "$comment " << comment << " $end\n";
}

void VCDWriter::writeAllChanges(const SimState &state) {
  writeTimestamp(state.getTime().value);

  for (auto &entry : signals) {
    StringRef path = entry.first();
    auto &sigEntry = entry.second;

    SignalValue currentValue = state.getSignalValue(path);

    // Only write if changed or no previous value
    if (!sigEntry.lastValue || currentValue != *sigEntry.lastValue) {
      writeValueChange(sigEntry.identifier, currentValue);
      sigEntry.lastValue = currentValue;
    }
  }
}

std::optional<StringRef> VCDWriter::getIdentifier(StringRef path) const {
  auto it = signals.find(path);
  if (it == signals.end())
    return std::nullopt;
  return StringRef(it->second.identifier);
}

bool VCDWriter::hasSignal(StringRef path) const {
  return signals.count(path) > 0;
}

void VCDWriter::flush() { os.flush(); }

void VCDWriter::formatValue(const SignalValue &value, llvm::raw_ostream &out) {
  for (int i = value.getWidth() - 1; i >= 0; --i) {
    out << formatScalar(value.getBit(i));
  }
}

char VCDWriter::formatScalar(LogicValue value) {
  switch (value) {
  case LogicValue::Zero:
    return '0';
  case LogicValue::One:
    return '1';
  case LogicValue::Unknown:
    return 'x';
  case LogicValue::HighZ:
    return 'z';
  }
  return 'x';
}

//===----------------------------------------------------------------------===//
// VCDRingBuffer Implementation
//===----------------------------------------------------------------------===//

VCDRingBuffer::VCDRingBuffer(unsigned capacity) : capacity(capacity) {
  buffer.resize(capacity);
}

void VCDRingBuffer::record(const SimState &state) {
  Entry entry;
  entry.time = state.getTime();

  // Record all signal values
  const Scope *root = state.getRootScope();
  if (root) {
    std::function<void(const Scope *)> recordScope = [&](const Scope *scope) {
      for (const auto &sig : scope->getSignals()) {
        entry.values[sig.fullPath] = state.getSignalValue(sig.fullPath);
      }
      for (const auto &child : scope->getChildren()) {
        recordScope(child.get());
      }
    };
    recordScope(root);
  }

  buffer[writePos] = std::move(entry);
  writePos = (writePos + 1) % capacity;
  if (writePos == 0)
    wrapped = true;
}

std::vector<VCDRingBuffer::Entry> VCDRingBuffer::getEntries() const {
  std::vector<Entry> result;

  if (wrapped) {
    // Buffer has wrapped - read from writePos to end, then from start to
    // writePos
    for (unsigned i = writePos; i < capacity; ++i) {
      if (!buffer[i].values.empty())
        result.push_back(buffer[i]);
    }
    for (unsigned i = 0; i < writePos; ++i) {
      if (!buffer[i].values.empty())
        result.push_back(buffer[i]);
    }
  } else {
    // Buffer hasn't wrapped - read from start to writePos
    for (unsigned i = 0; i < writePos; ++i) {
      if (!buffer[i].values.empty())
        result.push_back(buffer[i]);
    }
  }

  return result;
}

std::vector<VCDRingBuffer::Entry> VCDRingBuffer::getEntries(const SimTime &start,
                                                           const SimTime &end) const {
  auto all = getEntries();
  std::vector<Entry> result;

  for (const auto &entry : all) {
    if (entry.time >= start && entry.time <= end)
      result.push_back(entry);
  }

  return result;
}

void VCDRingBuffer::clear() {
  buffer.clear();
  buffer.resize(capacity);
  writePos = 0;
  wrapped = false;
}

unsigned VCDRingBuffer::size() const {
  return wrapped ? capacity : writePos;
}

//===----------------------------------------------------------------------===//
// WaveformCapture Implementation
//===----------------------------------------------------------------------===//

WaveformCapture::WaveformCapture(unsigned preCapture, unsigned postCapture)
    : preCapture(preCapture), postCapture(postCapture), preBuffer(preCapture) {}

void WaveformCapture::capture(const SimState &state) {
  if (capturing) {
    // In post-trigger capture phase
    VCDRingBuffer::Entry entry;
    entry.time = state.getTime();

    const Scope *root = state.getRootScope();
    if (root) {
      std::function<void(const Scope *)> recordScope = [&](const Scope *scope) {
        for (const auto &sig : scope->getSignals()) {
          // Check signal filter
          if (!signalFilter.empty()) {
            bool match = false;
            for (const auto &pattern : signalFilter) {
              if (sig.fullPath.find(pattern) != std::string::npos) {
                match = true;
                break;
              }
            }
            if (!match)
              continue;
          }
          entry.values[sig.fullPath] = state.getSignalValue(sig.fullPath);
        }
        for (const auto &child : scope->getChildren()) {
          recordScope(child.get());
        }
      };
      recordScope(root);
    }

    postBuffer.push_back(std::move(entry));
    ++postCaptureCount;

    if (postCaptureCount >= postCapture)
      capturing = false;
  } else {
    // Pre-trigger continuous capture
    preBuffer.record(state);
  }
}

void WaveformCapture::trigger() {
  if (!capturing) {
    capturing = true;
    postCaptureCount = 0;
    postBuffer.clear();
  }
}

bool WaveformCapture::saveToVCD(StringRef filename, const Scope *rootScope) {
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec, llvm::sys::fs::OF_None);
  if (ec)
    return false;

  VCDWriter writer(file);
  writer.writeHeader("Waveform capture from CIRCT Debug", "CIRCT Debug 1.0",
                     "1ns");

  // Collect all signal names from the captured data
  llvm::StringMap<unsigned> signalWidths;

  auto preEntries = preBuffer.getEntries();
  for (const auto &entry : preEntries) {
    for (const auto &kv : entry.values) {
      signalWidths[kv.first()] = kv.second.getWidth();
    }
  }
  for (const auto &entry : postBuffer) {
    for (const auto &kv : entry.values) {
      signalWidths[kv.first()] = kv.second.getWidth();
    }
  }

  // Write signal definitions
  writer.beginScope("capture", "module");
  for (const auto &kv : signalWidths) {
    writer.addWireSignal(kv.first(), kv.second, kv.first());
  }
  writer.endScope();
  writer.finishHeader();

  // Write values
  writer.beginDumpVars();
  bool firstEntry = true;

  auto writeEntry = [&](const VCDRingBuffer::Entry &entry) {
    writer.writeTimestamp(entry.time.value);
    for (const auto &kv : entry.values) {
      auto id = writer.getIdentifier(kv.first());
      if (id)
        writer.writeValueChange(*id, kv.second);
    }
    firstEntry = false;
  };

  // Write pre-trigger entries
  for (const auto &entry : preEntries) {
    if (firstEntry) {
      for (const auto &kv : entry.values) {
        auto id = writer.getIdentifier(kv.first());
        if (id)
          writer.writeInitialValue(*id, kv.second);
      }
      writer.endDumpVars();
      firstEntry = false;
    }
    writeEntry(entry);
  }

  // Write post-trigger entries
  for (const auto &entry : postBuffer) {
    writeEntry(entry);
  }

  return true;
}

void WaveformCapture::setSignalFilter(const std::vector<std::string> &signals) {
  signalFilter = signals;
}

void WaveformCapture::clear() {
  preBuffer.clear();
  postBuffer.clear();
  capturing = false;
  postCaptureCount = 0;
}
