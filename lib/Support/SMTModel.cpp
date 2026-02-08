//===- SMTModel.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SMTModel.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <mutex>

using namespace llvm;

namespace {

std::string formatBitVectorValue(const APInt &value, unsigned width) {
  if (width == 0)
    return "0";
  if (width % 4 == 0) {
    SmallString<64> hexStr;
    value.toString(hexStr, 16, /*Signed=*/false);
    std::string hex(hexStr.begin(), hexStr.end());
    size_t digits = width / 4;
    if (hex.size() < digits)
      hex.insert(0, digits - hex.size(), '0');
    return (Twine(width) + "'h" + hex).str();
  }
  SmallString<64> decStr;
  value.toString(decStr, 10, /*Signed=*/false);
  return (Twine(width) + "'d" + StringRef(decStr)).str();
}

std::optional<std::string> tryFormatBitVector(StringRef value) {
  StringRef work = value.trim();
  if (work.consume_front("#b")) {
    if (work.empty() || work.find_first_not_of("01") != StringRef::npos)
      return std::nullopt;
    unsigned width = work.size();
    APInt ap(width, work, 2);
    return formatBitVectorValue(ap, width);
  }
  if (work.consume_front("#x")) {
    if (work.empty() ||
        work.find_first_not_of("0123456789abcdefABCDEF") != StringRef::npos)
      return std::nullopt;
    unsigned width = work.size() * 4;
    APInt ap(width, work, 16);
    return formatBitVectorValue(ap, width);
  }
  if (!work.consume_front("(_ bv"))
    return std::nullopt;
  size_t numEnd = work.find_first_not_of("0123456789");
  if (numEnd == StringRef::npos)
    return std::nullopt;
  StringRef numStr = work.take_front(numEnd);
  work = work.drop_front(numEnd).ltrim();
  size_t widthEnd = work.find_first_not_of("0123456789");
  if (widthEnd == StringRef::npos)
    return std::nullopt;
  StringRef widthStr = work.take_front(widthEnd);
  work = work.drop_front(widthEnd).ltrim();
  if (!work.consume_front(")"))
    return std::nullopt;
  if (!work.trim().empty())
    return std::nullopt;
  unsigned width = 0;
  if (widthStr.getAsInteger(10, width))
    return std::nullopt;
  if (width == 0)
    return "0";
  APInt ap(width, numStr, 10);
  return formatBitVectorValue(ap, width);
}

} // namespace

std::string circt::normalizeSMTModelValue(StringRef value) {
  value = value.trim();
  if (value == "true" || value == "false")
    return value.str();
  if (auto formatted = tryFormatBitVector(value))
    return *formatted;
  return value.str();
}

static bool modelHeaderPrinted = false;
static std::mutex modelPrintMutex;
static llvm::StringMap<std::string> capturedModelValues;

static void printModelHeaderLocked() {
  if (modelHeaderPrinted)
    return;
  modelHeaderPrinted = true;
  llvm::errs() << "counterexample inputs:\n";
}

void circt::resetCapturedSMTModelValues() {
  std::lock_guard<std::mutex> lock(modelPrintMutex);
  modelHeaderPrinted = false;
  capturedModelValues.clear();
}

llvm::StringMap<std::string> circt::getCapturedSMTModelValues() {
  std::lock_guard<std::mutex> lock(modelPrintMutex);
  return capturedModelValues;
}

extern "C" void circt_smt_print_model_header() {
  std::lock_guard<std::mutex> lock(modelPrintMutex);
  printModelHeaderLocked();
}

extern "C" void circt_smt_print_model_value(const char *name,
                                            const char *value) {
  if (!name || !value)
    return;
  std::string formatted = circt::normalizeSMTModelValue(StringRef(value));
  std::lock_guard<std::mutex> lock(modelPrintMutex);
  printModelHeaderLocked();
  capturedModelValues[name] = formatted;
  llvm::errs() << "  " << name << " = " << formatted << "\n";
}
