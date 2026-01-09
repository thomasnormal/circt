//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SemanticTokens.cpp
//
// This file implements semantic tokens for Verilog/SystemVerilog LSP support.
//
//===----------------------------------------------------------------------===//

#include "SemanticTokens.h"

#include <algorithm>

using namespace circt::lsp;

//===----------------------------------------------------------------------===//
// JSON Serialization
//===----------------------------------------------------------------------===//

bool circt::lsp::fromJSON(const llvm::json::Value &value,
                          SemanticTokensParams &result, llvm::json::Path path) {
  auto *obj = value.getAsObject();
  if (!obj) {
    path.report("expected object");
    return false;
  }

  auto *textDocument = obj->get("textDocument");
  if (!textDocument) {
    path.report("missing textDocument");
    return false;
  }

  auto *docObj = textDocument->getAsObject();
  if (!docObj) {
    path.report("textDocument must be an object");
    return false;
  }

  auto uri = docObj->getString("uri");
  if (!uri) {
    path.report("missing uri");
    return false;
  }

  result.textDocumentUri = uri->str();
  return true;
}

llvm::json::Value circt::lsp::toJSON(const SemanticTokensResult &value) {
  return llvm::json::Object{{"data", llvm::json::Array(value.data)}};
}

//===----------------------------------------------------------------------===//
// SemanticTokensResult Implementation
//===----------------------------------------------------------------------===//

void SemanticTokensResult::encodeTokens(
    const std::vector<SemanticToken> &tokens) {
  if (tokens.empty())
    return;

  // Sort tokens by position
  std::vector<SemanticToken> sortedTokens = tokens;
  std::sort(sortedTokens.begin(), sortedTokens.end(),
            [](const SemanticToken &a, const SemanticToken &b) {
              if (a.line != b.line)
                return a.line < b.line;
              return a.startChar < b.startChar;
            });

  data.reserve(sortedTokens.size() * 5);

  uint32_t prevLine = 0;
  uint32_t prevStartChar = 0;

  for (const auto &token : sortedTokens) {
    uint32_t deltaLine = token.line - prevLine;
    uint32_t deltaStartChar =
        (deltaLine == 0) ? (token.startChar - prevStartChar) : token.startChar;

    data.push_back(deltaLine);
    data.push_back(deltaStartChar);
    data.push_back(token.length);
    data.push_back(token.tokenType);
    data.push_back(token.tokenModifiers);

    prevLine = token.line;
    prevStartChar = token.startChar;
  }
}

//===----------------------------------------------------------------------===//
// Server Capabilities Helpers
//===----------------------------------------------------------------------===//

llvm::json::Object circt::lsp::getSemanticTokensLegend() {
  auto types = getSemanticTokenTypes();
  auto modifiers = getSemanticTokenModifiers();

  llvm::json::Array typeArray;
  for (const auto &type : types)
    typeArray.push_back(type);

  llvm::json::Array modifierArray;
  for (const auto &mod : modifiers)
    modifierArray.push_back(mod);

  return llvm::json::Object{
      {"tokenTypes", std::move(typeArray)},
      {"tokenModifiers", std::move(modifierArray)},
  };
}

llvm::json::Object circt::lsp::getSemanticTokensOptions() {
  return llvm::json::Object{
      {"legend", getSemanticTokensLegend()},
      {"full", true},
      {"range", false},
  };
}
