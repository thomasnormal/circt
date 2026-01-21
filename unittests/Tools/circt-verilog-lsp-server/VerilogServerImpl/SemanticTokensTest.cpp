//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "SemanticTokens.h"

using namespace circt::lsp;

namespace {

/// Test that semantic token encoding produces correct delta-encoded output.
TEST(SemanticTokensTest, EncodingSimpleTokens) {
  std::vector<SemanticToken> tokens;

  // Add tokens at specific positions
  tokens.emplace_back(0, 0, 6, SemanticTokenType::Module, 0);  // "module" at 0:0
  tokens.emplace_back(0, 7, 5, SemanticTokenType::Module,
                      static_cast<uint32_t>(SemanticTokenModifier::Definition));  // "test" at 0:7
  tokens.emplace_back(1, 2, 5, SemanticTokenType::Variable, 0);  // variable at 1:2

  SemanticTokensResult result;
  result.encodeTokens(tokens);

  // Expected encoding (deltaLine, deltaStartChar, length, tokenType, modifiers):
  // Token 1: 0, 0, 6, Module(19), 0
  // Token 2: 0, 7, 5, Module(19), Definition(2)
  // Token 3: 1, 2, 5, Variable(7), 0

  ASSERT_EQ(result.data.size(), 15u);  // 3 tokens * 5 values

  // First token
  EXPECT_EQ(result.data[0], 0u);   // deltaLine
  EXPECT_EQ(result.data[1], 0u);   // deltaStartChar
  EXPECT_EQ(result.data[2], 6u);   // length
  EXPECT_EQ(result.data[3], static_cast<uint32_t>(SemanticTokenType::Module));
  EXPECT_EQ(result.data[4], 0u);   // modifiers

  // Second token (same line)
  EXPECT_EQ(result.data[5], 0u);   // deltaLine (same line)
  EXPECT_EQ(result.data[6], 7u);   // deltaStartChar
  EXPECT_EQ(result.data[7], 5u);   // length

  // Third token (different line)
  EXPECT_EQ(result.data[10], 1u);  // deltaLine (next line)
  EXPECT_EQ(result.data[11], 2u);  // deltaStartChar (reset for new line)
  EXPECT_EQ(result.data[12], 5u);  // length
}

/// Test that unsorted tokens are properly sorted before encoding.
TEST(SemanticTokensTest, TokenSorting) {
  std::vector<SemanticToken> tokens;

  // Add tokens out of order
  tokens.emplace_back(2, 0, 4, SemanticTokenType::Variable, 0);
  tokens.emplace_back(0, 0, 6, SemanticTokenType::Module, 0);
  tokens.emplace_back(1, 5, 3, SemanticTokenType::Port, 0);

  SemanticTokensResult result;
  result.encodeTokens(tokens);

  ASSERT_EQ(result.data.size(), 15u);

  // First token should be line 0
  EXPECT_EQ(result.data[0], 0u);
  // Second token should be line 1
  EXPECT_EQ(result.data[5], 1u);
  // Third token should be line 2
  EXPECT_EQ(result.data[10], 1u);  // delta from line 1 to line 2
}

/// Test that empty token list produces empty data.
TEST(SemanticTokensTest, EmptyTokens) {
  std::vector<SemanticToken> tokens;

  SemanticTokensResult result;
  result.encodeTokens(tokens);

  EXPECT_TRUE(result.data.empty());
}

/// Test token types and modifiers lists.
TEST(SemanticTokensTest, TokenTypesAndModifiers) {
  auto types = getSemanticTokenTypes();
  auto modifiers = getSemanticTokenModifiers();

  // Verify we have the expected number of token types
  EXPECT_GE(types.size(), 18u);

  // Verify we have the expected modifiers
  EXPECT_GE(modifiers.size(), 8u);

  // Check specific types exist
  EXPECT_EQ(types[static_cast<uint32_t>(SemanticTokenType::Namespace)], "namespace");
  EXPECT_EQ(types[static_cast<uint32_t>(SemanticTokenType::Variable)], "variable");
  EXPECT_EQ(types[static_cast<uint32_t>(SemanticTokenType::Parameter)], "parameter");
  EXPECT_EQ(types[static_cast<uint32_t>(SemanticTokenType::Function)], "function");

  // Check specific modifiers
  EXPECT_EQ(modifiers[0], "declaration");
  EXPECT_EQ(modifiers[1], "definition");
  EXPECT_EQ(modifiers[2], "readonly");
}

/// Test JSON serialization of SemanticTokensResult.
TEST(SemanticTokensTest, JSONSerialization) {
  SemanticTokensResult result;
  result.data = {0, 0, 6, 19, 0, 0, 7, 5, 19, 2};

  auto json = toJSON(result);
  auto *obj = json.getAsObject();
  ASSERT_TRUE(obj != nullptr);

  auto *dataArray = obj->getArray("data");
  ASSERT_TRUE(dataArray != nullptr);
  EXPECT_EQ(dataArray->size(), 10u);
}

/// Test SemanticTokensParams parsing.
TEST(SemanticTokensTest, ParamsFromJSON) {
  llvm::json::Value jsonParams = llvm::json::Object{
      {"textDocument", llvm::json::Object{{"uri", "file:///test.sv"}}}};

  SemanticTokensParams params;
  llvm::json::Path::Root root;
  ASSERT_TRUE(fromJSON(jsonParams, params, llvm::json::Path(root)));
  EXPECT_EQ(params.textDocumentUri, "file:///test.sv");
}

/// Test semantic tokens options for capabilities.
TEST(SemanticTokensTest, ServerCapabilities) {
  auto options = getSemanticTokensOptions();

  auto full = options.getBoolean("full");
  EXPECT_TRUE(full && *full);

  auto *legend = options.getObject("legend");
  ASSERT_TRUE(legend != nullptr);

  auto *tokenTypes = legend->getArray("tokenTypes");
  ASSERT_TRUE(tokenTypes != nullptr);
  EXPECT_GT(tokenTypes->size(), 0u);

  auto *tokenModifiers = legend->getArray("tokenModifiers");
  ASSERT_TRUE(tokenModifiers != nullptr);
  EXPECT_GT(tokenModifiers->size(), 0u);
}

} // namespace
