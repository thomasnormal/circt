//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SemanticTokens.h
//
// This header declares types for LSP semantic tokens support. Semantic tokens
// provide richer syntax highlighting information to editors, distinguishing
// between different kinds of identifiers (nets vs. regs vs. parameters, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_SEMANTICTOKENS_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_SEMANTICTOKENS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace circt {
namespace lsp {

/// Semantic token types for Verilog/SystemVerilog.
/// These map to LSP SemanticTokenTypes.
enum class SemanticTokenType : uint32_t {
  Namespace = 0,  // package
  Type = 1,       // typedef, struct, enum
  Class = 2,      // class
  Enum = 3,       // enum declaration
  Interface = 4,  // interface
  Struct = 5,     // struct
  Parameter = 6,  // parameter, localparam
  Variable = 7,   // logic, reg variables
  Property = 8,   // ports
  EnumMember = 9, // enum values
  Function = 10,  // function
  Method = 11,    // class methods
  Macro = 12,     // `define macros
  Keyword = 13,   // module, always, etc.
  Comment = 14,
  String = 15,
  Number = 16,
  Operator = 17,
  // Custom Verilog-specific types
  Net = 18,      // wire, tri, etc.
  Module = 19,   // module names
  Instance = 20, // instance names
  Port = 21,     // port names in declarations
  Signal = 22,   // general signals (fallback)
};

/// Semantic token modifiers for Verilog/SystemVerilog.
/// These are bit flags that can be combined.
enum class SemanticTokenModifier : uint32_t {
  None = 0,
  Declaration = 1 << 0,  // This is a declaration
  Definition = 1 << 1,   // This is a definition
  Readonly = 1 << 2,     // Parameter or localparam
  Static = 1 << 3,       // Static member
  Deprecated = 1 << 4,   // Marked as deprecated
  Async = 1 << 5,        // Asynchronous (posedge/negedge triggered)
  Modification = 1 << 6, // Being modified (LHS of assignment)
  Documentation = 1 << 7,
  DefaultLibrary = 1 << 8,
};

/// Get the list of semantic token type names in order.
inline std::vector<std::string> getSemanticTokenTypes() {
  return {
      "namespace",  // 0
      "type",       // 1
      "class",      // 2
      "enum",       // 3
      "interface",  // 4
      "struct",     // 5
      "parameter",  // 6
      "variable",   // 7
      "property",   // 8
      "enumMember", // 9
      "function",   // 10
      "method",     // 11
      "macro",      // 12
      "keyword",    // 13
      "comment",    // 14
      "string",     // 15
      "number",     // 16
      "operator",   // 17
      // Custom types need special handling by clients
      "net",      // 18 - custom
      "module",   // 19 - custom
      "instance", // 20 - custom
      "port",     // 21 - custom
      "signal",   // 22 - custom
  };
}

/// Get the list of semantic token modifier names in order.
inline std::vector<std::string> getSemanticTokenModifiers() {
  return {
      "declaration",    // 0
      "definition",     // 1
      "readonly",       // 2
      "static",         // 3
      "deprecated",     // 4
      "async",          // 5
      "modification",   // 6
      "documentation",  // 7
      "defaultLibrary", // 8
  };
}

/// A single semantic token.
struct SemanticToken {
  /// Line number (0-based).
  uint32_t line;
  /// Start character (0-based).
  uint32_t startChar;
  /// Length of the token.
  uint32_t length;
  /// Token type index.
  uint32_t tokenType;
  /// Token modifiers (bitset).
  uint32_t tokenModifiers;

  SemanticToken(uint32_t line, uint32_t startChar, uint32_t length,
                SemanticTokenType type,
                uint32_t modifiers = static_cast<uint32_t>(
                    SemanticTokenModifier::None))
      : line(line), startChar(startChar), length(length),
        tokenType(static_cast<uint32_t>(type)), tokenModifiers(modifiers) {}
};

/// Params for textDocument/semanticTokens/full request.
struct SemanticTokensParams {
  /// The text document.
  std::string textDocumentUri;
};

/// Result for textDocument/semanticTokens/full request.
struct SemanticTokensResult {
  /// The actual token data encoded as described in the LSP specification.
  /// Tokens are encoded as 5 integers per token:
  /// [deltaLine, deltaStartChar, length, tokenType, tokenModifiers]
  std::vector<uint32_t> data;

  /// Encode tokens into the data array.
  void encodeTokens(const std::vector<SemanticToken> &tokens);
};

/// JSON serialization for SemanticTokensParams.
bool fromJSON(const llvm::json::Value &value, SemanticTokensParams &result,
              llvm::json::Path path);

/// JSON serialization for SemanticTokensResult.
llvm::json::Value toJSON(const SemanticTokensResult &value);

/// Generate semantic tokens legend for server capabilities.
llvm::json::Object getSemanticTokensLegend();

/// Generate semantic tokens options for server capabilities.
llvm::json::Object getSemanticTokensOptions();

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_SEMANTICTOKENS_H_
