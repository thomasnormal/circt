"""
Phase 6: Feature Gap Identification Tests

Tests known limitations and evaluates wishlist features:
- Macro expansion
- Cross-file analysis
- Incremental indexing
"""

import asyncio
import time

import pytest
from lsprotocol import types

from conftest import (
    OPENTITAN_HW,
    OpenTitanFiles,
    count_file_lines,
    open_document,
    skip_if_missing,
)


class TestMacroExpansion:
    """Test macro-related features."""

    @pytest.mark.asyncio
    async def test_assert_macro_hover(self, opentitan_client, benchmark):
        """Test hover on assertion macros (known limitation)."""
        # Find a file with ASSERT macros
        test_file = OPENTITAN_HW / "ip" / "prim" / "rtl" / "prim_assert.sv"

        if not test_file.exists():
            # Try alternate location
            import glob
            prim_files = list(OPENTITAN_HW.glob("**/prim_assert*.sv"))
            if prim_files:
                test_file = prim_files[0]
            else:
                pytest.skip("No assertion file found")

        uri, content = await open_document(opentitan_client, test_file, wait_time=2.0)

        # Find ASSERT macro usage
        lines = content.split('\n')
        target_line = None
        target_col = None
        macro_name = None

        for line_num, line in enumerate(lines):
            for macro in ['`ASSERT', '`ASSUME', '`COVER']:
                if macro in line:
                    target_line = line_num
                    target_col = line.find(macro) + 1
                    macro_name = macro
                    break
            if target_line is not None:
                break

        if target_line is None:
            pytest.skip("No assertion macro found")

        benchmark.start("assert_macro_hover", str(test_file))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        has_hover = result is not None and result.contents
        print(f"\nHover on {macro_name}:")
        print(f"  Has hover info: {has_hover}")

        if has_hover:
            if isinstance(result.contents, types.MarkupContent):
                print(f"  Content preview: {result.contents.value[:100]}...")
            else:
                print(f"  Content: {str(result.contents)[:100]}...")
        else:
            print("  [LIMITATION] No hover info for assertion macros")

    @pytest.mark.asyncio
    async def test_macro_goto_definition(self, opentitan_client, benchmark):
        """Test go-to-definition on macro usage."""
        # Find a file with macro usage
        dv_macros_path = OPENTITAN_HW / "dv" / "sv" / "dv_utils" / "dv_macros.svh"

        if not dv_macros_path.exists():
            pytest.skip("dv_macros.svh not found")

        uri, content = await open_document(opentitan_client, dv_macros_path, wait_time=2.0)

        # Find `define
        lines = content.split('\n')
        target_line = None
        target_col = None
        macro_name = None

        for line_num, line in enumerate(lines):
            if '`define' in line:
                target_line = line_num
                target_col = line.find('`define') + 8  # After `define
                # Get macro name
                rest = line[line.find('`define') + 7:].strip()
                macro_name = rest.split('(')[0].split()[0] if rest else 'unknown'
                break

        if target_line is None:
            pytest.skip("No macro definition found")

        benchmark.start("macro_goto_definition", str(dv_macros_path))

        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        has_definition = result is not None
        if isinstance(result, list):
            has_definition = len(result) > 0

        print(f"\nGo-to-definition on macro {macro_name}:")
        print(f"  Found definition: {has_definition}")

    @pytest.mark.asyncio
    async def test_macro_expansion_completion(self, opentitan_client, benchmark):
        """Test completion inside macro arguments."""
        # Use a file with macro usage
        test_file = OpenTitanFiles.CIP_BASE_TEST if OpenTitanFiles.exists("CIP_BASE_TEST") else None

        if test_file is None:
            pytest.skip("Test file not found")

        uri, content = await open_document(opentitan_client, test_file, wait_time=2.0)

        # Find macro with arguments
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if '`uvm_' in line and '(' in line:
                target_line = line_num
                # Position inside the parentheses
                paren_pos = line.find('(')
                target_col = paren_pos + 1
                break

        if target_line is None:
            pytest.skip("No macro with arguments found")

        benchmark.start("macro_completion", str(test_file))

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )

        benchmark.stop()

        if result:
            if isinstance(result, types.CompletionList):
                items = result.items
            else:
                items = result
            print(f"\nCompletion inside macro: {len(items)} items")
        else:
            print("\nCompletion inside macro: No items")


class TestCrossFileAnalysis:
    """Test cross-file analysis features."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_find_references_unopened_files(self, opentitan_client, benchmark):
        """Test finding references in files that aren't open."""
        # Open uart_tx.sv
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=2.0)

        # Get the module name
        lines = content.split('\n')
        module_name = None
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if line.strip().startswith('module '):
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1].rstrip('(#;')
                    target_line = line_num
                    target_col = line.find(module_name)
                    break

        if module_name is None:
            pytest.skip("No module found")

        benchmark.start("references_unopened", str(OpenTitanFiles.UART_TX))

        start = time.perf_counter()
        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        ref_count = len(result) if result else 0

        # Check if references are in different files (not just the open file)
        other_files = 0
        if result:
            for ref in result:
                if hasattr(ref, 'uri') and 'uart_tx' not in ref.uri:
                    other_files += 1

        print(f"\nFind references to {module_name}:")
        print(f"  Total references: {ref_count}")
        print(f"  In other files: {other_files}")
        print(f"  Time: {elapsed_ms:.1f}ms")

        if other_files == 0 and ref_count <= 1:
            print("  [LIMITATION] May only find references in open files")

    @pytest.mark.asyncio
    async def test_workspace_symbol_search(self, opentitan_client, benchmark):
        """Test workspace-wide symbol search."""
        # First open a file to initialize the workspace
        if OpenTitanFiles.exists("UART_TX"):
            await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=2.0)

        benchmark.start("workspace_symbols", "workspace")

        try:
            start = time.perf_counter()
            result = await opentitan_client.workspace_symbol_async(
                types.WorkspaceSymbolParams(query="uart")
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            benchmark.stop()

            symbol_count = len(result) if result else 0
            print(f"\nWorkspace symbol search 'uart':")
            print(f"  Found: {symbol_count} symbols")
            print(f"  Time: {elapsed_ms:.1f}ms")

            if result:
                # Show first few
                for sym in result[:5]:
                    print(f"    {sym.name}")

        except Exception as e:
            benchmark.stop()
            print(f"\nWorkspace symbols: {e}")


class TestIncrementalIndexing:
    """Test incremental indexing behavior."""

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_reference_update_after_edit(self, opentitan_client, benchmark):
        """Test if references update after editing a file."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=2.0)

        # Add a new signal
        new_signal = "logic test_incremental_signal;"
        lines = content.split('\n')

        # Find insertion point (after module ports)
        insert_line = 10
        for i, line in enumerate(lines):
            if ');' in line and 'module' not in line:
                insert_line = i + 1
                break

        modified_lines = lines[:insert_line] + ['    ' + new_signal] + lines[insert_line:]
        modified_content = '\n'.join(modified_lines)

        # Update document
        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=2),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
            )
        )

        await asyncio.sleep(1.0)

        benchmark.start("incremental_reference", str(OpenTitanFiles.UART_TX))

        # Try to find references to the new signal
        start = time.perf_counter()
        result = await opentitan_client.text_document_references_async(
            types.ReferenceParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=insert_line, character=10),
                context=types.ReferenceContext(include_declaration=True),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        ref_count = len(result) if result else 0
        print(f"\nIncremental indexing test:")
        print(f"  Added signal: {new_signal}")
        print(f"  References found: {ref_count}")
        print(f"  Time: {elapsed_ms:.1f}ms")

        if ref_count == 0:
            print("  [NOTE] New symbols may require file save to be indexed")

    @pytest.mark.asyncio
    @skip_if_missing("UART_TX")
    async def test_completion_includes_new_symbols(self, opentitan_client, benchmark):
        """Test if completion includes newly added symbols."""
        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=2.0)

        # Add a new signal with unique prefix
        new_signal = "logic zzz_test_completion_signal;"
        lines = content.split('\n')

        insert_line = 12
        modified_lines = lines[:insert_line] + ['    ' + new_signal] + lines[insert_line:]
        modified_content = '\n'.join(modified_lines)

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=2),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content)],
            )
        )

        await asyncio.sleep(1.0)

        # Add line that starts typing the new signal name
        trigger_line = insert_line + 5
        modified_lines2 = modified_lines[:trigger_line] + ['    zzz_'] + modified_lines[trigger_line:]
        modified_content2 = '\n'.join(modified_lines2)

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=3),
                content_changes=[types.TextDocumentContentChangeEvent_Type1(text=modified_content2)],
            )
        )

        await asyncio.sleep(0.5)

        benchmark.start("completion_new_symbols", str(OpenTitanFiles.UART_TX))

        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=trigger_line, character=8),  # After "zzz_"
            )
        )

        benchmark.stop()

        found_new_signal = False
        if result:
            items = result.items if isinstance(result, types.CompletionList) else result
            for item in items:
                if 'zzz_test' in item.label:
                    found_new_signal = True
                    break

        print(f"\nCompletion includes new symbols:")
        print(f"  New signal found in completion: {found_new_signal}")

        if not found_new_signal:
            print("  [NOTE] New symbols may need time to be indexed for completion")


class TestKnownLimitations:
    """Verify and document known limitations."""

    @pytest.mark.asyncio
    async def test_generate_block_expansion(self, opentitan_client, benchmark):
        """Test handling of generate blocks."""
        # Find a file with generate blocks
        test_files = [
            OPENTITAN_HW / "ip" / "prim" / "rtl" / "prim_arbiter_ppc.sv",
            OPENTITAN_HW / "ip" / "prim" / "rtl" / "prim_fifo_sync.sv",
        ]

        test_file = None
        for f in test_files:
            if f.exists():
                test_file = f
                break

        if test_file is None:
            # Find any file with generate
            import glob
            gen_files = list(OPENTITAN_HW.glob("**/prim_*.sv"))
            for f in gen_files[:10]:
                content = f.read_text()
                if 'generate' in content:
                    test_file = f
                    break

        if test_file is None:
            pytest.skip("No file with generate block found")

        uri, content = await open_document(opentitan_client, test_file, wait_time=2.0)

        # Find generate block
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if 'generate' in line or 'genvar' in line:
                target_line = line_num
                break

        if target_line is None:
            pytest.skip("No generate block found")

        benchmark.start("generate_block", str(test_file))

        # Try hover on generate content
        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line + 2, character=10),
            )
        )

        benchmark.stop()

        print(f"\nGenerate block handling:")
        print(f"  Hover in generate: {'Info available' if result else 'No info'}")
        print("  [LIMITATION] Generate block expansion may not be fully supported")

    @pytest.mark.asyncio
    async def test_parameter_dependent_types(self, opentitan_client, benchmark):
        """Test handling of parameter-dependent types."""
        # Find a file with parameterized types
        if not OpenTitanFiles.exists("AES"):
            pytest.skip("AES module not found")

        uri, content = await open_document(opentitan_client, OpenTitanFiles.AES, wait_time=2.0)

        # Find a parameterized signal
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if '[' in line and 'WIDTH' in line.upper() or 'SIZE' in line.upper():
                target_line = line_num
                break

        if target_line is None:
            # Use any line with array
            for line_num, line in enumerate(lines):
                if 'logic' in line and '[' in line:
                    target_line = line_num
                    break

        if target_line is None:
            pytest.skip("No parameterized signal found")

        benchmark.start("param_types", str(OpenTitanFiles.AES))

        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=15),
            )
        )

        benchmark.stop()

        print(f"\nParameter-dependent type handling:")
        if result and result.contents:
            print(f"  Hover info: Available")
            if isinstance(result.contents, types.MarkupContent):
                print(f"  Content: {result.contents.value[:80]}...")
        else:
            print("  Hover info: None")


class TestWishlistFeatures:
    """Evaluate wishlist features mentioned in the plan."""

    @pytest.mark.asyncio
    async def test_semantic_tokens(self, opentitan_client, benchmark):
        """Test semantic token support."""
        if not OpenTitanFiles.exists("UART_TX"):
            pytest.skip("UART_TX not found")

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        benchmark.start("semantic_tokens", str(OpenTitanFiles.UART_TX))

        try:
            result = await opentitan_client.text_document_semantic_tokens_full_async(
                types.SemanticTokensParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                )
            )

            benchmark.stop()

            if result and result.data:
                print(f"\nSemantic tokens:")
                print(f"  Tokens: {len(result.data) // 5} (data length: {len(result.data)})")
            else:
                print("\nSemantic tokens: Not available")

        except Exception as e:
            benchmark.stop()
            print(f"\nSemantic tokens: Error - {e}")

    @pytest.mark.asyncio
    async def test_code_actions(self, opentitan_client, benchmark):
        """Test code actions (quick fixes, refactoring)."""
        if not OpenTitanFiles.exists("UART_TX"):
            pytest.skip("UART_TX not found")

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        benchmark.start("code_actions", str(OpenTitanFiles.UART_TX))

        try:
            result = await opentitan_client.text_document_code_action_async(
                types.CodeActionParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    range=types.Range(
                        start=types.Position(line=0, character=0),
                        end=types.Position(line=10, character=0),
                    ),
                    context=types.CodeActionContext(diagnostics=[]),
                )
            )

            benchmark.stop()

            if result:
                print(f"\nCode actions:")
                print(f"  Available: {len(result)}")
                for action in result[:3]:
                    if isinstance(action, types.CodeAction):
                        print(f"    {action.title}")
            else:
                print("\nCode actions: None available")

        except Exception as e:
            benchmark.stop()
            print(f"\nCode actions: Error - {e}")

    @pytest.mark.asyncio
    async def test_rename_symbol(self, opentitan_client, benchmark):
        """Test symbol rename capability."""
        if not OpenTitanFiles.exists("UART_TX"):
            pytest.skip("UART_TX not found")

        uri, content = await open_document(opentitan_client, OpenTitanFiles.UART_TX, wait_time=1.0)

        # Find a signal to rename
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            if 'logic' in line and not line.strip().startswith('//'):
                idx = line.find('logic') + 6
                while idx < len(line) and line[idx] in ' \t[]0-9:':
                    idx += 1
                if idx < len(line) and (line[idx].isalpha() or line[idx] == '_'):
                    target_line = line_num
                    target_col = idx
                    break

        if target_line is None:
            pytest.skip("No signal found")

        benchmark.start("rename_prepare", str(OpenTitanFiles.UART_TX))

        try:
            # First check if rename is possible
            prepare_result = await opentitan_client.text_document_prepare_rename_async(
                types.PrepareRenameParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                )
            )

            benchmark.stop()

            if prepare_result:
                print(f"\nRename capability:")
                print(f"  Prepare rename: Supported")
            else:
                print(f"\nRename capability:")
                print(f"  Prepare rename: Not available at position")

        except Exception as e:
            benchmark.stop()
            print(f"\nRename capability: Error - {e}")

    @pytest.mark.asyncio
    async def test_folding_ranges(self, opentitan_client, benchmark):
        """Test folding range support."""
        if not OpenTitanFiles.exists("AES"):
            pytest.skip("AES not found")

        uri, _ = await open_document(opentitan_client, OpenTitanFiles.AES, wait_time=1.0)

        benchmark.start("folding_ranges", str(OpenTitanFiles.AES))

        try:
            result = await opentitan_client.text_document_folding_range_async(
                types.FoldingRangeParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                )
            )

            benchmark.stop()

            if result:
                print(f"\nFolding ranges:")
                print(f"  Ranges: {len(result)}")
            else:
                print("\nFolding ranges: Not available")

        except Exception as e:
            benchmark.stop()
            print(f"\nFolding ranges: Error - {e}")
