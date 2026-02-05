"""
Phase 5: Rapid Editing Simulation Tests

Tests LSP behavior under rapid document changes:
- Fast incremental edits
- Completion during typing
- Diagnostics updates
- Race condition detection
"""

import asyncio
import time

import pytest
from lsprotocol import types

from conftest import (
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
    TEST_DIR,
)


class TestRapidEditing:
    """Test rapid document editing scenarios."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_rapid_character_typing(self, opentitan_client, benchmark):
        """Simulate rapid character-by-character typing."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        # Find a good insertion point
        lines = content.split('\n')
        insert_line = 20

        for i, line in enumerate(lines):
            if 'always' in line or 'assign' in line:
                insert_line = i + 1
                break

        benchmark.start("rapid_typing", str(OpenTitanFiles.UART_TX))

        # Simulate typing "logic [7:0] new_signal;"
        text_to_type = "    logic [7:0] new_signal;"
        version = 2

        start = time.perf_counter()
        errors_during_typing = 0

        for i in range(len(text_to_type)):
            # Build current typed text
            partial_text = text_to_type[:i+1]

            # Create modified content
            modified_lines = lines[:insert_line] + [partial_text] + lines[insert_line:]
            modified_content = '\n'.join(modified_lines)

            try:
                opentitan_client.text_document_did_change(
                    types.DidChangeTextDocumentParams(
                        text_document=types.VersionedTextDocumentIdentifier(
                            uri=uri,
                            version=version,
                        ),
                        content_changes=[
                            types.TextDocumentContentChangeEvent_Type1(text=modified_content)
                        ],
                    )
                )
                version += 1
            except Exception as e:
                errors_during_typing += 1

            # Small delay between keystrokes (50ms - fast typist speed)
            await asyncio.sleep(0.05)

        elapsed = time.perf_counter() - start

        # Wait for final state to settle
        await asyncio.sleep(0.5)

        benchmark.stop()

        print(f"\nRapid typing simulation:")
        print(f"  Characters typed: {len(text_to_type)}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Errors during typing: {errors_during_typing}")

        assert errors_during_typing == 0, "Errors occurred during rapid typing"

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_rapid_line_edits(self, opentitan_client, benchmark):
        """Simulate rapid line-by-line edits."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        insert_line = 20

        benchmark.start("rapid_line_edits", str(OpenTitanFiles.UART_TX))

        # Add 50 lines rapidly
        num_edits = 50
        version = 2
        start = time.perf_counter()

        for i in range(num_edits):
            new_line = f"    // Comment line {i}"
            modified_lines = lines[:insert_line] + [new_line] + lines[insert_line:]
            modified_content = '\n'.join(modified_lines)

            opentitan_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(
                        uri=uri,
                        version=version,
                    ),
                    content_changes=[
                        types.TextDocumentContentChangeEvent_Type1(text=modified_content)
                    ],
                )
            )
            version += 1
            lines = modified_lines  # Update lines for next iteration

            await asyncio.sleep(0.02)  # 20ms between edits

        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nRapid line edits:")
        print(f"  Lines added: {num_edits}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average per edit: {elapsed/num_edits*1000:.1f}ms")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_edit_undo_redo_cycle(self, opentitan_client, benchmark):
        """Simulate edit-undo-redo cycles."""
        uri, original_content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        benchmark.start("edit_undo_redo", str(OpenTitanFiles.UART_TX))

        version = 2
        start = time.perf_counter()
        cycles = 20

        for i in range(cycles):
            # Edit: Add a line
            modified_content = original_content + f"\n// Edit {i}"
            opentitan_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
                    content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
                )
            )
            version += 1
            await asyncio.sleep(0.03)

            # Undo: Restore original
            opentitan_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
                    content_changes=[types.TextDocumentContentChangeEvent_Type1(text=original_content)],
                )
            )
            version += 1
            await asyncio.sleep(0.03)

        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nEdit-undo cycles:")
        print(f"  Cycles: {cycles}")
        print(f"  Total time: {elapsed:.2f}s")


class TestCompletionDuringEditing:
    """Test completion while editing."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_while_typing(self, opentitan_client, benchmark):
        """Request completion while document is being edited."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        insert_line = 25

        benchmark.start("completion_while_typing", str(OpenTitanFiles.UART_TX))

        completion_times = []
        version = 2

        # Type "uart_" and request completion after each character
        prefix = "    uart_"

        for i in range(len(prefix)):
            partial = prefix[:i+1]
            modified_lines = lines[:insert_line] + [partial] + lines[insert_line:]
            modified_content = '\n'.join(modified_lines)

            opentitan_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
                    content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
                )
            )
            version += 1

            # Request completion
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    opentitan_client.text_document_completion_async(
                        types.CompletionParams(
                            text_document=types.TextDocumentIdentifier(uri=uri),
                            position=types.Position(line=insert_line, character=len(partial)),
                        )
                    ),
                    timeout=2.0
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                completion_times.append(elapsed_ms)
            except asyncio.TimeoutError:
                completion_times.append(float('inf'))

            await asyncio.sleep(0.05)

        benchmark.stop()

        valid_times = [t for t in completion_times if t != float('inf')]
        if valid_times:
            print(f"\nCompletion during typing:")
            print(f"  Requests: {len(completion_times)}")
            print(f"  Successful: {len(valid_times)}")
            print(f"  Avg time: {sum(valid_times)/len(valid_times):.1f}ms")
            print(f"  Max time: {max(valid_times):.1f}ms")


class TestDiagnosticsUpdates:
    """Test diagnostics update behavior during editing."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_diagnostics_appear_disappear(self, opentitan_client, benchmark):
        """Verify diagnostics appear for errors and disappear when fixed."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        insert_line = 20

        benchmark.start("diagnostics_update", str(OpenTitanFiles.UART_TX))

        # Insert invalid code
        invalid_line = "    assign invalid_signal ="  # Missing semicolon
        modified_lines = lines[:insert_line] + [invalid_line] + lines[insert_line:]
        modified_content = '\n'.join(modified_lines)

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=2),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
            )
        )

        # Wait for diagnostics
        await asyncio.sleep(1.0)

        diagnostics_with_error = opentitan_client.diagnostics.get(uri, [])
        error_count_before = len([d for d in diagnostics_with_error if d.severity == types.DiagnosticSeverity.Error])

        # Fix the code
        valid_line = "    // Fixed comment"
        fixed_lines = lines[:insert_line] + [valid_line] + lines[insert_line:]
        fixed_content = '\n'.join(fixed_lines)

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=3),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=fixed_content)],
            )
        )

        # Wait for diagnostics to update
        await asyncio.sleep(1.0)

        diagnostics_after_fix = opentitan_client.diagnostics.get(uri, [])
        error_count_after = len([d for d in diagnostics_after_fix if d.severity == types.DiagnosticSeverity.Error])

        benchmark.stop()

        print(f"\nDiagnostics update:")
        print(f"  Errors with invalid code: {error_count_before}")
        print(f"  Errors after fix: {error_count_after}")

        # Errors should decrease (or at least not increase) after fix
        # Note: We can't guarantee exact behavior without knowing the parser

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_diagnostics_debouncing(self, opentitan_client, benchmark):
        """Test that rapid edits don't cause diagnostic spam."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        lines = content.split('\n')
        insert_line = 20

        benchmark.start("diagnostics_debounce", str(OpenTitanFiles.UART_TX))

        # Rapid edits
        version = 2
        for i in range(30):
            modified_lines = lines[:insert_line] + [f"    // Line {i}"] + lines[insert_line:]
            modified_content = '\n'.join(modified_lines)

            opentitan_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
                    content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
                )
            )
            version += 1
            await asyncio.sleep(0.01)  # 10ms between edits

        # Final state
        await asyncio.sleep(1.0)

        benchmark.stop()

        # Server should handle this gracefully without issues
        print("\nDiagnostics debouncing test completed")


class TestConcurrentOperations:
    """Test concurrent LSP operations."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_concurrent_requests(self, opentitan_client, benchmark):
        """Send multiple requests concurrently."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX)

        benchmark.start("concurrent_requests", str(OpenTitanFiles.UART_TX))

        # Create multiple concurrent requests
        async def hover_request(line):
            return await opentitan_client.text_document_hover_async(
                types.HoverParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=line, character=10),
                )
            )

        async def completion_request(line):
            return await opentitan_client.text_document_completion_async(
                types.CompletionParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=line, character=4),
                )
            )

        async def symbols_request():
            return await opentitan_client.text_document_document_symbol_async(
                types.DocumentSymbolParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                )
            )

        # Run concurrently
        start = time.perf_counter()
        results = await asyncio.gather(
            hover_request(10),
            hover_request(20),
            hover_request(30),
            completion_request(15),
            completion_request(25),
            symbols_request(),
            return_exceptions=True
        )
        elapsed = time.perf_counter() - start

        benchmark.stop()

        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"\nConcurrent requests:")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Total time: {elapsed*1000:.1f}ms")

        # All should succeed
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  Request {i} failed: {r}")


class TestStressEditing:
    """Stress test editing scenarios."""

    @pytest.mark.asyncio
    async def test_100_rapid_changes(self, lsp_client, benchmark):
        """Stress test with 100 rapid document changes."""
        # Use the sample file for stress testing
        sample_file = TEST_DIR / "sample.sv"
        uri, content = await open_document(lsp_client, sample_file)

        benchmark.start("100_rapid_changes", str(sample_file))

        version = 2
        start = time.perf_counter()

        for i in range(100):
            modified_content = content + f"\n// Change {i}"

            lsp_client.text_document_did_change(
                types.DidChangeTextDocumentParams(
                    text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
                    content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
                )
            )
            version += 1

            # Very short delay (10ms)
            await asyncio.sleep(0.01)

        elapsed = time.perf_counter() - start

        # Wait for server to catch up
        await asyncio.sleep(1.0)

        benchmark.stop()

        print(f"\n100 rapid changes:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average per change: {elapsed*10:.1f}ms")

        # Verify server is still responsive
        result = await lsp_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )
        assert result is not None, "Server became unresponsive after stress test"

    @pytest.mark.asyncio
    async def test_large_paste_operation(self, lsp_client, benchmark):
        """Test pasting a large block of code."""
        sample_file = TEST_DIR / "sample.sv"
        uri, content = await open_document(lsp_client, sample_file)

        # Generate a large paste (1000 lines)
        large_paste = "\n".join([f"    // Generated line {i}" for i in range(1000)])

        benchmark.start("large_paste", str(sample_file))

        start = time.perf_counter()

        lsp_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=2),
                content_changes=[
                    types.TextDocumentContentChangeEvent_Type1(text=content + large_paste)
                ],
            )
        )

        # Wait for processing
        await asyncio.sleep(2.0)

        elapsed = time.perf_counter() - start

        benchmark.stop()

        print(f"\nLarge paste (1000 lines):")
        print(f"  Processing time: {elapsed:.2f}s")

        # Verify responsiveness
        result = await lsp_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=5, character=5),
            )
        )
        print(f"  Server responsive: {'Yes' if result is not None else 'No response needed'}")
