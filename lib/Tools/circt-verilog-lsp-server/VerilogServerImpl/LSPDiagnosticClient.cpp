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
// - UVM-specific diagnostic hints and suggestions
// - Diagnostic categories for better organization
// - Diagnostic tags for unused/deprecated markers
//
//===----------------------------------------------------------------------===//

#include "LSPDiagnosticClient.h"
#include "llvm/ADT/StringRef.h"

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

/// Determine the diagnostic category based on the message content.
/// This helps users understand the compilation stage where the issue occurred.
static std::optional<std::string> getDiagnosticCategory(llvm::StringRef message) {
  // Syntax/parsing errors
  if (message.contains("expected") || message.contains("syntax") ||
      message.contains("unexpected") || message.contains("missing"))
    return "Parse Error";

  // Type/semantic errors
  if (message.contains("type") || message.contains("cannot convert") ||
      message.contains("incompatible"))
    return "Type Error";

  // Undefined/undeclared errors
  if (message.contains("unknown") || message.contains("undeclared") ||
      message.contains("undefined") || message.contains("not found"))
    return "Semantic Error";

  // Width/size warnings
  if (message.contains("width") || message.contains("truncat") ||
      message.contains("extend"))
    return "Width Warning";

  // Unused warnings
  if (message.contains("unused") || message.contains("never used") ||
      message.contains("never read"))
    return "Unused Warning";

  return std::nullopt;
}

/// Check if the diagnostic indicates an unused entity (variable, signal, etc.)
static bool isUnusedDiagnostic(llvm::StringRef message) {
  return message.contains("unused") || message.contains("never used") ||
         message.contains("never read") || message.contains("never written") ||
         message.contains("value is never used");
}

/// Check if the diagnostic indicates a deprecated feature.
static bool isDeprecatedDiagnostic(llvm::StringRef message) {
  return message.contains("deprecated") || message.contains("obsolete");
}

/// Enhance the diagnostic message with UVM-specific hints if applicable.
/// This helps users who are writing UVM testbenches by providing contextual
/// suggestions for common UVM errors.
static std::string enhanceMessageForUVM(llvm::StringRef originalMessage) {
  std::string enhanced = originalMessage.str();

  // Unknown type that looks like a UVM type
  if (originalMessage.contains("unknown type") ||
      originalMessage.contains("unknown class")) {

    // Common UVM base classes
    if (originalMessage.contains("uvm_component") ||
        originalMessage.contains("uvm_object") ||
        originalMessage.contains("uvm_driver") ||
        originalMessage.contains("uvm_monitor") ||
        originalMessage.contains("uvm_agent") ||
        originalMessage.contains("uvm_env") ||
        originalMessage.contains("uvm_test") ||
        originalMessage.contains("uvm_scoreboard") ||
        originalMessage.contains("uvm_subscriber") ||
        originalMessage.contains("uvm_sequence") ||
        originalMessage.contains("uvm_sequence_item") ||
        originalMessage.contains("uvm_sequencer")) {
      enhanced += "\n\nHint: Add 'import uvm_pkg::*;' and ensure uvm_macros.svh is included.";
    }

    // UVM utility macros
    if (originalMessage.contains("uvm_info") ||
        originalMessage.contains("uvm_warning") ||
        originalMessage.contains("uvm_error") ||
        originalMessage.contains("uvm_fatal")) {
      enhanced += "\n\nHint: UVM macros require: `include \"uvm_macros.svh\"";
    }

    // UVM factory registration macros
    if (originalMessage.contains("uvm_component_utils") ||
        originalMessage.contains("uvm_object_utils") ||
        originalMessage.contains("uvm_field_")) {
      enhanced += "\n\nHint: Factory macros require: `include \"uvm_macros.svh\" and import uvm_pkg::*;";
    }
  }

  // Missing factory registration hint
  if (originalMessage.contains("create") &&
      originalMessage.contains("not found")) {
    enhanced += "\n\nHint: Ensure the class is registered with the UVM factory using `uvm_component_utils or `uvm_object_utils.";
  }

  // Virtual interface errors
  if (originalMessage.contains("virtual interface") ||
      (originalMessage.contains("interface") &&
       originalMessage.contains("cannot"))) {
    enhanced += "\n\nHint: Virtual interfaces must be set via uvm_config_db before use. Check your test/env configuration.";
  }

  // Phase errors
  if (originalMessage.contains("build_phase") ||
      originalMessage.contains("connect_phase") ||
      originalMessage.contains("run_phase")) {
    enhanced += "\n\nHint: UVM phases must have the correct signature: virtual function/task void phase_name(uvm_phase phase);";
  }

  // TLM port/export errors
  if (originalMessage.contains("uvm_analysis_port") ||
      originalMessage.contains("uvm_blocking_put") ||
      originalMessage.contains("uvm_nonblocking") ||
      originalMessage.contains("_imp") ||
      originalMessage.contains("_export")) {
    enhanced += "\n\nHint: TLM ports/exports need import uvm_pkg::*; and proper parameterization.";
  }

  // Config DB errors
  if (originalMessage.contains("uvm_config_db") ||
      originalMessage.contains("uvm_resource_db")) {
    enhanced += "\n\nHint: Config DB requires import uvm_pkg::*; Type parameter must match exactly.";
  }

  return enhanced;
}

void LSPDiagnosticClient::report(const slang::ReportedDiagnostic &slangDiag) {
  auto loc = document.getLspLocation(slangDiag.location);
  // Show only the diagnostics in the current file.
  if (loc.uri != document.getURI())
    return;

  auto &lspDiag = diags.emplace_back();
  lspDiag.severity = getSeverity(slangDiag.severity);
  lspDiag.source = "slang";

  // Enhance message with UVM-specific hints
  llvm::StringRef originalMsg = slangDiag.formattedMessage;
  lspDiag.message = enhanceMessageForUVM(originalMsg);

  // Add diagnostic category for better organization
  lspDiag.category = getDiagnosticCategory(originalMsg);

  // Add diagnostic tags for unused/deprecated markers
  // These provide visual hints in the editor (e.g., faded text for unused)
  if (isUnusedDiagnostic(originalMsg)) {
    lspDiag.tags.push_back(llvm::lsp::DiagnosticTag::Unnecessary);
  }
  if (isDeprecatedDiagnostic(originalMsg)) {
    lspDiag.tags.push_back(llvm::lsp::DiagnosticTag::Deprecated);
  }

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

    // Add additional ranges as related information with descriptive messages
    if (slangDiag.ranges.size() > 1) {
      std::vector<llvm::lsp::DiagnosticRelatedInformation> relatedInfo;
      for (size_t i = 1; i < slangDiag.ranges.size(); ++i) {
        auto relatedRange = slangDiag.ranges[i];
        auto relatedLoc = document.getLspLocation(relatedRange);
        if (relatedLoc.uri.file().empty())
          continue;

        llvm::lsp::DiagnosticRelatedInformation info;
        info.location = relatedLoc;
        // Provide more descriptive related location messages
        if (i == 1 && originalMsg.contains("previous")) {
          info.message = "previous declaration here";
        } else if (originalMsg.contains("type") ||
                   originalMsg.contains("mismatch")) {
          info.message = "related type information";
        } else if (originalMsg.contains("defined") ||
                   originalMsg.contains("declared")) {
          info.message = "originally defined here";
        } else {
          info.message = "related location";
        }
        relatedInfo.push_back(std::move(info));
      }
      if (!relatedInfo.empty())
        lspDiag.relatedInformation = std::move(relatedInfo);
    }
  } else {
    lspDiag.range = loc.range;
  }

  // Add notes from the original diagnostic as related information
  const auto &originalDiag = slangDiag.originalDiagnostic;
  if (!originalDiag.notes.empty()) {
    if (!lspDiag.relatedInformation)
      lspDiag.relatedInformation.emplace();

    for (const auto &note : originalDiag.notes) {
      if (!note.location.valid())
        continue;

      auto noteLoc = document.getLspLocation(note.location);
      if (noteLoc.uri.file().empty())
        continue;

      llvm::lsp::DiagnosticRelatedInformation noteInfo;
      noteInfo.location = noteLoc;
      // Use the note's formatted message if available
      noteInfo.message = "note";
      lspDiag.relatedInformation->push_back(std::move(noteInfo));
    }
  }
}
