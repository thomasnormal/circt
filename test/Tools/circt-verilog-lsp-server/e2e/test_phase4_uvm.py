"""
Phase 4: UVM Testbench Features Tests

Tests LSP features on UVM/DV infrastructure:
- UVM class hierarchy navigation
- UVM macro handling
- Call hierarchy tracing
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


class TestUVMClassHierarchy:
    """Test UVM class hierarchy navigation."""

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_ENV")
    async def test_cip_base_env_symbols(self, opentitan_client, benchmark):
        """Test document symbols for cip_base_env.sv."""
        file_path = OpenTitanFiles.CIP_BASE_ENV

        benchmark.start("uvm_symbols", str(file_path))

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\ncip_base_env symbols: {len(symbols)}")

        # Look for class symbols
        for sym in symbols[:10]:
            if isinstance(sym, types.DocumentSymbol):
                print(f"  {sym.kind}: {sym.name}")

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_ENV")
    async def test_uvm_inheritance_navigation(self, opentitan_client, benchmark):
        """Test navigating up the UVM inheritance chain."""
        file_path = OpenTitanFiles.CIP_BASE_ENV

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find extends clause
        lines = content.split('\n')
        target_line = None
        target_col = None
        parent_class = None

        for line_num, line in enumerate(lines):
            if 'extends' in line:
                # Find the parent class name after 'extends'
                idx = line.find('extends')
                start = idx + len('extends')
                while start < len(line) and line[start] in ' \t':
                    start += 1

                # Find end of class name
                end = start
                while end < len(line) and (line[end].isalnum() or line[end] in '_#'):
                    end += 1

                if end > start:
                    parent_class = line[start:end].split('#')[0]  # Remove parameters
                    target_line = line_num
                    target_col = start
                    break

        if target_line is None:
            pytest.skip("No extends clause found")

        benchmark.start("uvm_inheritance", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nNavigate to parent class {parent_class}: {elapsed_ms:.1f}ms")

        if result:
            if isinstance(result, types.Location):
                print(f"  Found at: {result.uri}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"  Found at: {result[0]}")

    @pytest.mark.asyncio
    @skip_if_missing("DV_BASE_VSEQ")
    async def test_vseq_hierarchy(self, opentitan_client, benchmark):
        """Test virtual sequence hierarchy navigation."""
        file_path = OpenTitanFiles.DV_BASE_VSEQ

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        benchmark.start("vseq_hierarchy", str(file_path))

        # Get symbols
        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\ndv_base_vseq symbols: {len(symbols)}")

        # Look for task/function definitions
        task_count = 0
        for sym in symbols:
            if isinstance(sym, types.DocumentSymbol):
                if sym.kind == types.SymbolKind.Function or sym.kind == types.SymbolKind.Method:
                    task_count += 1
                    if task_count <= 5:
                        print(f"  Task/Function: {sym.name}")

        print(f"  Total tasks/functions: {task_count}")


class TestUVMMacros:
    """Test UVM macro handling."""

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_TEST")
    async def test_uvm_macro_diagnostics(self, opentitan_client, benchmark):
        """Test diagnostics on files with UVM macros."""
        file_path = OpenTitanFiles.CIP_BASE_TEST

        benchmark.start("uvm_macro_diagnostics", str(file_path))

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Check for UVM macros in content
        uvm_macros = []
        for macro in ['`uvm_', '`UVM_', '`ovm_']:
            count = content.count(macro)
            if count > 0:
                uvm_macros.append((macro, count))

        diagnostics = opentitan_client.diagnostics.get(uri, [])
        errors = [d for d in diagnostics if d.severity == types.DiagnosticSeverity.Error]

        benchmark.stop()

        print(f"\ncip_base_test.sv UVM macros:")
        for macro, count in uvm_macros:
            print(f"  {macro}*: {count} occurrences")

        print(f"  Diagnostics: {len(diagnostics)} total, {len(errors)} errors")

    @pytest.mark.asyncio
    @skip_if_missing("DV_BASE_TEST")
    async def test_macro_hover(self, opentitan_client, benchmark):
        """Test hover on UVM macro invocations."""
        file_path = OpenTitanFiles.DV_BASE_TEST

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find a UVM macro
        lines = content.split('\n')
        target_line = None
        target_col = None
        macro_name = None

        for line_num, line in enumerate(lines):
            if '`uvm_' in line:
                idx = line.find('`uvm_')
                target_line = line_num
                target_col = idx + 1  # After the backtick
                # Extract macro name
                end = idx + 5
                while end < len(line) and (line[end].isalnum() or line[end] == '_'):
                    end += 1
                macro_name = line[idx:end]
                break

        if target_line is None:
            pytest.skip("No UVM macro found")

        benchmark.start("macro_hover", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_hover_async(
            types.HoverParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nHover on {macro_name}: {elapsed_ms:.1f}ms")

        if result and result.contents:
            # Print first 100 chars of hover content
            if isinstance(result.contents, types.MarkupContent):
                content_preview = result.contents.value[:100]
            elif isinstance(result.contents, str):
                content_preview = result.contents[:100]
            else:
                content_preview = str(result.contents)[:100]
            print(f"  Content: {content_preview}...")

    @pytest.mark.asyncio
    async def test_macro_defined_fields(self, opentitan_client, benchmark):
        """Test go-to-definition for macro-defined fields."""
        # Find a file with UVM factory registration
        cip_macros_path = OPENTITAN_HW / "dv" / "sv" / "cip_lib" / "cip_macros.svh"

        if not cip_macros_path.exists():
            pytest.skip("cip_macros.svh not found")

        uri, content = await open_document(opentitan_client, cip_macros_path, wait_time=2.0)

        benchmark.start("macro_definition", str(cip_macros_path))

        # Get symbols from macro file
        result = await opentitan_client.text_document_document_symbol_async(
            types.DocumentSymbolParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
            )
        )

        benchmark.stop()

        symbols = result if isinstance(result, list) else []
        print(f"\ncip_macros.svh symbols: {len(symbols)}")

        # Look for macro definitions
        macro_count = content.count('`define')
        print(f"  Macro definitions: {macro_count}")


class TestUVMCompletion:
    """Test completion in UVM context."""

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_ENV")
    async def test_uvm_completion(self, opentitan_client, benchmark):
        """Test completion suggestions in UVM file."""
        file_path = OpenTitanFiles.CIP_BASE_ENV

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find a position inside the class body
        lines = content.split('\n')
        target_line = None

        for line_num, line in enumerate(lines):
            if 'function' in line or 'task' in line:
                target_line = line_num + 1
                break

        if target_line is None:
            target_line = 20

        benchmark.start("uvm_completion", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=4),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        if result:
            if isinstance(result, types.CompletionList):
                items = result.items
            else:
                items = result
            print(f"\nUVM completion: {elapsed_ms:.1f}ms, {len(items)} items")

            # Show first few items
            for item in items[:5]:
                print(f"  {item.label}")

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_ENV")
    async def test_uvm_prefix_completion(self, opentitan_client, benchmark):
        """Test completion after typing 'uvm_' prefix."""
        file_path = OpenTitanFiles.CIP_BASE_ENV

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Modify content to add uvm_ prefix
        lines = content.split('\n')

        # Find insertion point
        insert_line = 30
        for i, line in enumerate(lines):
            if 'function' in line and 'new' in line:
                insert_line = i + 2
                break

        # Add a line with uvm_ prefix
        modified_lines = lines[:insert_line] + ['    uvm_'] + lines[insert_line:]
        modified_content = '\n'.join(modified_lines)

        opentitan_client.text_document_did_change(
            types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(
                    uri=uri,
                    version=2,
                ),
                content_changes=[
                    types.TextDocumentContentChangeEvent_Type1(text=modified_content)
                ],
            )
        )

        await asyncio.sleep(0.5)

        benchmark.start("uvm_prefix_completion", str(file_path))

        start = time.perf_counter()
        result = await opentitan_client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=insert_line, character=8),  # After "uvm_"
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        if result:
            if isinstance(result, types.CompletionList):
                items = result.items
            else:
                items = result
            print(f"\nUVM prefix completion: {elapsed_ms:.1f}ms, {len(items)} items")

            # Filter to uvm_ items
            uvm_items = [i for i in items if i.label.startswith('uvm_')]
            print(f"  UVM items: {len(uvm_items)}")


class TestDVLibNavigation:
    """Test navigation within DV library files."""

    @pytest.mark.asyncio
    async def test_dv_lib_cross_references(self, opentitan_client, benchmark):
        """Test cross-references between DV library files."""
        dv_lib_path = OPENTITAN_HW / "dv" / "sv" / "dv_lib"

        if not dv_lib_path.exists():
            pytest.skip("dv_lib not found")

        # Open dv_base_env.sv
        env_file = dv_lib_path / "dv_base_env.sv"
        if not env_file.exists():
            pytest.skip("dv_base_env.sv not found")

        uri, content = await open_document(opentitan_client, env_file, wait_time=2.0)

        benchmark.start("dv_lib_references", str(env_file))

        # Find a reference to another DV class
        lines = content.split('\n')
        target_line = None
        target_col = None

        for line_num, line in enumerate(lines):
            for class_name in ['dv_base_agent', 'dv_base_scoreboard', 'dv_base_env_cfg']:
                if class_name in line:
                    target_line = line_num
                    target_col = line.find(class_name)
                    break
            if target_line is not None:
                break

        if target_line is None:
            print("\nNo cross-class reference found")
            benchmark.stop()
            return

        start = time.perf_counter()
        result = await opentitan_client.text_document_definition_async(
            types.DefinitionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=types.Position(line=target_line, character=target_col),
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark.stop()

        print(f"\nDV lib cross-reference: {elapsed_ms:.1f}ms")
        if result:
            if isinstance(result, types.Location):
                print(f"  Target: {result.uri}")


class TestCallHierarchy:
    """Test call hierarchy features (if supported)."""

    @pytest.mark.asyncio
    @skip_if_missing("DV_BASE_VSEQ")
    async def test_prepare_call_hierarchy(self, opentitan_client, benchmark):
        """Test preparing call hierarchy on a task."""
        file_path = OpenTitanFiles.DV_BASE_VSEQ

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find a task definition
        lines = content.split('\n')
        target_line = None
        target_col = None
        task_name = None

        for line_num, line in enumerate(lines):
            if line.strip().startswith('virtual task ') or line.strip().startswith('task '):
                # Extract task name
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'task':
                        if i + 1 < len(parts):
                            task_name = parts[i + 1].rstrip('(;')
                            target_line = line_num
                            target_col = line.find(task_name)
                            break
                if target_line is not None:
                    break

        if target_line is None:
            pytest.skip("No task found")

        benchmark.start("call_hierarchy_prepare", str(file_path))

        try:
            start = time.perf_counter()
            result = await opentitan_client.text_document_prepare_call_hierarchy_async(
                types.CallHierarchyPrepareParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            benchmark.stop()

            if result:
                print(f"\nCall hierarchy for {task_name}: {elapsed_ms:.1f}ms")
                print(f"  Items: {len(result)}")
            else:
                print(f"\nCall hierarchy not available for {task_name}")

        except Exception as e:
            benchmark.stop()
            print(f"\nCall hierarchy not supported: {e}")


class TestTypeHierarchy:
    """Test type hierarchy features (if supported)."""

    @pytest.mark.asyncio
    @skip_if_missing("CIP_BASE_ENV")
    async def test_prepare_type_hierarchy(self, opentitan_client, benchmark):
        """Test preparing type hierarchy on a class."""
        file_path = OpenTitanFiles.CIP_BASE_ENV

        uri, content = await open_document(opentitan_client, file_path, wait_time=2.0)

        # Find the class definition
        lines = content.split('\n')
        target_line = None
        target_col = None
        class_name = None

        for line_num, line in enumerate(lines):
            if 'class ' in line and 'extends' in line:
                # Find the class name
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'class':
                        if i + 1 < len(parts):
                            class_name = parts[i + 1]
                            target_line = line_num
                            target_col = line.find(class_name)
                            break
                if target_line is not None:
                    break

        if target_line is None:
            pytest.skip("No class definition found")

        benchmark.start("type_hierarchy_prepare", str(file_path))

        try:
            start = time.perf_counter()
            result = await opentitan_client.text_document_prepare_type_hierarchy_async(
                types.TypeHierarchyPrepareParams(
                    text_document=types.TextDocumentIdentifier(uri=uri),
                    position=types.Position(line=target_line, character=target_col),
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            benchmark.stop()

            if result:
                print(f"\nType hierarchy for {class_name}: {elapsed_ms:.1f}ms")
                print(f"  Items: {len(result)}")
                for item in result[:3]:
                    print(f"    {item.name}")
            else:
                print(f"\nType hierarchy not available for {class_name}")

        except Exception as e:
            benchmark.stop()
            print(f"\nType hierarchy not supported: {e}")
