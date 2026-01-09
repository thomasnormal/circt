//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A converter that can be plugged into a slang `DiagnosticEngine` as a
// client that will map slang diagnostics to LSP diagnostics.
//
// This implementation provides rich diagnostic information including:
// - Source ranges (not just point locations)
// - Related diagnostic information for multi-span errors
// - Source code extraction for better error messages
//
//===----------------------------------------------------------------------===//

#include "LSPDiagnosticClient.h"

using namespace circt::lsp;

static llvm::lsp::DiagnosticSeverity
getSeverity(slang::DiagnosticSeverity severity) {
  switch (severity) {
  case slang::DiagnosticSeverity::Fatal:
  case slang::DiagnosticSeverity::Error:
    return llvm::lsp::DiagnosticSeverity::Error;
  case slang::DiagnosticSeverity::Warning:
    return llvm::lsp::DiagnosticSeverity::Warning;
  case slang::DiagnosticSeverity::Ignored:
  case slang::DiagnosticSeverity::Note:
    return llvm::lsp::DiagnosticSeverity::Information;
  }
  llvm_unreachable("all slang diagnostic severities should be handled");
  return llvm::lsp::DiagnosticSeverity::Error;
}

void LSPDiagnosticClient::report(const slang::ReportedDiagnostic &slangDiag) {
  auto loc = document.getLspLocation(slangDiag.location);
  // Show only the diagnostics in the current file.
  if (loc.uri != document.getURI())
    return;

  auto &lspDiag = diags.emplace_back();
  lspDiag.severity = getSeverity(slangDiag.severity);
  lspDiag.source = "slang";
  lspDiag.message = slangDiag.formattedMessage;

  // Use the source range if available for better highlighting.
  // Slang provides ranges for many diagnostics that give context.
  if (!slangDiag.ranges.empty()) {
    // Use the first range as the primary diagnostic range
    auto primaryRange = slangDiag.ranges[0];
    auto rangeLoc = document.getLspLocation(primaryRange);
    if (rangeLoc.uri == document.getURI()) {
      lspDiag.range = rangeLoc.range;
    } else {
      lspDiag.range = loc.range;
    }

    // Add additional ranges as related information
    if (slangDiag.ranges.size() > 1) {
      std::vector<llvm::lsp::DiagnosticRelatedInformation> relatedInfo;
      for (size_t i = 1; i < slangDiag.ranges.size(); ++i) {
        auto relatedRange = slangDiag.ranges[i];
        auto relatedLoc = document.getLspLocation(relatedRange);
        if (relatedLoc.uri.file().empty())
          continue;

        llvm::lsp::DiagnosticRelatedInformation info;
        info.location = relatedLoc;
        info.message = "related location";
        relatedInfo.push_back(std::move(info));
      }
      if (!relatedInfo.empty())
        lspDiag.relatedInformation = std::move(relatedInfo);
    }
  } else {
    lspDiag.range = loc.range;
  }
}
