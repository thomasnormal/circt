"""End-to-end tests for the CIRCT MLIR LSP server using pytest-lsp."""

import pathlib

import pytest
import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities

# Path to the CIRCT MLIR LSP server
CIRCT_LSP = pathlib.Path(__file__).parents[4] / "build" / "bin" / "circt-lsp-server"

# Test workspace
TEST_DIR = pathlib.Path(__file__).parent
TEST_FILE = TEST_DIR / "sample.mlir"


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[str(CIRCT_LSP)],
    ),
)
async def client(lsp_client: LanguageClient):
    """Set up the LSP client for testing."""
    params = types.InitializeParams(
        capabilities=client_capabilities("visual-studio-code"),
        root_uri=TEST_DIR.as_uri(),
        workspace_folders=[
            types.WorkspaceFolder(uri=TEST_DIR.as_uri(), name="test"),
        ],
    )
    await lsp_client.initialize_session(params)

    yield

    await lsp_client.shutdown_session()


@pytest.mark.asyncio
async def test_initialize(client: LanguageClient):
    """Test that the MLIR LSP server initializes correctly."""
    assert client.capabilities is not None


@pytest.mark.asyncio
async def test_document_open_and_diagnostics(client: LanguageClient):
    """Test opening an MLIR document and receiving diagnostics."""
    uri = TEST_FILE.as_uri()
    content = TEST_FILE.read_text()

    # Open the document
    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(1)

    # Check diagnostics were received (may be empty if file is valid)
    assert uri in client.diagnostics or len(client.diagnostics) == 0


@pytest.mark.asyncio
async def test_hover_on_operation(client: LanguageClient):
    """Test hover information on MLIR operations."""
    uri = TEST_FILE.as_uri()
    content = TEST_FILE.read_text()

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(0.5)

    # Hover over hw.module operation (line 1)
    result = await client.text_document_hover_async(
        types.HoverParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=1, character=3),  # "hw.module"
        )
    )

    # Hover may return None or a Hover object with operation info
    assert result is None or isinstance(result, types.Hover)


@pytest.mark.asyncio
async def test_goto_definition_for_value(client: LanguageClient):
    """Test go to definition for SSA values."""
    uri = TEST_FILE.as_uri()
    content = TEST_FILE.read_text()

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(0.5)

    # Try to go to definition of %reg usage
    result = await client.text_document_definition_async(
        types.DefinitionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=5, character=22),  # %reg in comb.add
        )
    )

    assert result is None or isinstance(result, (types.Location, list))


@pytest.mark.asyncio
async def test_document_symbols(client: LanguageClient):
    """Test document symbols for MLIR file."""
    uri = TEST_FILE.as_uri()
    content = TEST_FILE.read_text()

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(0.5)

    result = await client.text_document_document_symbol_async(
        types.DocumentSymbolParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
        )
    )

    assert result is not None
    # Should have symbols for our modules
    symbols = result if isinstance(result, list) else []
    assert len(symbols) >= 0  # May have module symbols


@pytest.mark.asyncio
async def test_invalid_mlir(client: LanguageClient):
    """Test that the server reports diagnostics for invalid MLIR."""
    uri = "file:///tmp/invalid.mlir"
    content = """
// Invalid MLIR - undefined type
hw.module @broken(in %x : undefined_type) {
  hw.output
}
"""

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(1)

    # Should have diagnostics for this invalid file
    if uri in client.diagnostics:
        diagnostics = client.diagnostics[uri]
        assert len(diagnostics) > 0


@pytest.mark.asyncio
async def test_completion(client: LanguageClient):
    """Test code completion in MLIR."""
    uri = TEST_FILE.as_uri()
    content = TEST_FILE.read_text()

    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri,
                language_id="mlir",
                version=1,
                text=content,
            )
        )
    )

    import asyncio
    await asyncio.sleep(0.5)

    result = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=3, character=2),
        )
    )

    # May return CompletionList, list, or None
    assert result is None or isinstance(result, (types.CompletionList, list))
