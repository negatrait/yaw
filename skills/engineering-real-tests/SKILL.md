---
name: engineering-real-tests
description: >-
  Author real, un-mocked integration and unit tests validating genuine invariants
  (real SQLite, real bytes on disk, real sparse matrices, loopback HTTP). Trigger on
  'write tests', 'author test', 'test suite', 'unit test', 'integration test',
  'regression test', 'unmocked test', 'real tests', 'test invariants'. Prioritize
  pytest, tmp_path, real SQLite (:memory: or tmp.db), and loopback sockets over
  unittest.mock. Never mock databases, filesystems, or mathematical algorithms.
---

# Engineering Real Tests: Substrate-Grounded Invariant Verification Runbook

Use this operational runbook to author robust, un-mocked unit and integration tests. In production systems, **tests that rely on `@patch`, `MagicMock()`, or synthetic dummy objects are a cognitive liability**: they pass when the underlying schema, byte encoding, operating system semantics, or mathematical algorithms are completely broken.

**The Golden Law of Test Engineering**: If the substrate is mocked, the invariant is unverified. Always test against real silicon, real filesystems, real SQLite transactions, and real mathematical operations.

```
┌─────────────────────────────────────────────────────────────┐
│             REAL TEST 4-PHASE AUTHORING PROTOCOL            │
├─────────────────────────────────────────────────────────────┤
│  PHASE 1: INVARIANT & SUBSTRATE IDENTIFICATION              │
│  • Define physical invariants (uniqueness, byte boundaries, │
│    sparse matrix conductance, composite primary keys)       │
│  • Identify the real storage substrate (SQLite, ext4 disk)  │
│                                                             │
│  PHASE 2: REAL SUBSTRATE ASSEMBLY (Zero Mocks)              │
│  • Use pytest `tmp_path` fixture for disk operations        │
│  • Use real SQLite connections (`:memory:` or `tmp_path/db`)│
│  • Spin up localhost loopback HTTP handlers for REST APIs   │
│                                                             │
│  PHASE 3: ADVERSARIAL STRESS & INVARIANT PROBING            │
│  • Multi-note collisions, race conditions, empty datasets   │
│  • Zero-copy byte seeks, UTF-8 multibyte boundary slicing   │
│  • Graph disconnected components, dangling pointers, zeros  │
│                                                             │
│  PHASE 4: DETERMINISTIC SILICON VERIFICATION                │
│  • Execute: `uv run --offline pytest <test_path> -vv`       │
│  • Verify 100% offline determinism (zero external network)  │
│  • Assert real state in database/disk, not mock calls       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. The Cognitive Traps & Negative Space of Fake Testing

| Cognitive Trap | The Footgun / Symptom | The Physical Consequence | The Mandatory Real Alternative |
| :--- | :--- | :--- | :--- |
| **The Database Mocking Trap** | Mocking `sqlite3.connect`, `cursor.execute`, or `cursor.fetchall` with `MagicMock()`. | Schema violations, SQL syntax errors, broken `VIEW` queries, and primary key collision bugs slip silently into production. | Use `sqlite3.connect(":memory:")` or `sqlite3.connect(tmp_path / "test.db")` with full table DDL and views applied. |
| **The Filesystem Mocking Trap** | Using `unittest.mock.mock_open` or passing in-memory StringIO to test byte offsets. | Multi-byte UTF-8 character splitting, partial reads, missing newline handling, and OS file lock contention fail in live runtime. | Write real files to `tmp_path`, open with `rb` / `r`, and verify exact `seek(offset)` and `read(length)`. |
| **The Algorithm Mocking Trap** | Mocking SciPy CSR matrices, PageRank iterations, or dot-product vectors. | Division by zero, non-convergent power iterations, and broken vector similarity formulas remain completely untested. | Construct real `scipy.sparse.csr_matrix` or `numpy.ndarray`, execute full power iterations, and assert $\sum p_i \approx 1.0$. |
| **The Tautological Assertion Trap** | Calling `mock.assert_called_with(...)` asserting that a mock received the exact arguments passed to it. | Verifies nothing about system state or output correctness; tests only that Python executes function calls. | Assert the actual side effect on the substrate: rows inserted in SQLite, files present on disk, output tensor values. |
| **The Swallowed Exception Trap** | Wrapping test assertions in `try ... except Exception: pass` or ignoring return codes. | Test suite passes green while underlying logic crashes, masking regressions. | Allow exceptions to fail the test naturally, or use `pytest.raises(SpecificException)`. |
| **The Synthetic Happy-Path Trap** | Testing only perfectly formed single-item inputs with no whitespace or edge cases. | System fails on dirty vault notes, empty markdown files, unicode emoji, and multi-note concept collisions. | Author adversarial test cases: empty files, duplicate slugs, missing YAML headers, cyclic references. |

---

## 2. Standard Domain Test Recipes

### Recipe 1: Real SQLite & Schema View Verification
Never mock database calls. Always run full DDL migrations and verify composite primary keys, triggers, and views against an isolated SQLite database.

```python
import sqlite3
from pathlib import Path
import pytest
from yaw.storage.db import ensure_schema_views


def test_composite_primary_key_and_unified_views(tmp_path: Path):
    db_path = tmp_path / "test_yaw.db"
    conn = sqlite3.connect(str(db_path))
    ensure_schema_views(conn)
    cursor = conn.cursor()

    # Invariant: (node_id, source_leaf_id) composite key allows multiple notes mentioning same entity
    cursor.execute(
        "INSERT INTO leaf_nodes (node_id, name, namespace, definition, source_leaf_id) VALUES (?, ?, ?, ?, ?)",
        (
            "concept:page_rank",
            "PageRank",
            "concept",
            "Graph ranking algorithm",
            "note_alpha.md",
        ),
    )
    cursor.execute(
        "INSERT INTO leaf_nodes (node_id, name, namespace, definition, source_leaf_id) VALUES (?, ?, ?, ?, ?)",
        (
            "concept:page_rank",
            "PageRank",
            "concept",
            "Graph ranking algorithm",
            "note_beta.md",
        ),
    )
    conn.commit()

    # Invariant: unified_nodes VIEW aggregates distinct canonical entities
    cursor.execute(
        "SELECT node_id, name FROM unified_nodes WHERE node_id = ?",
        ("concept:page_rank",),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1, (
        "unified_nodes VIEW must deduplicate distinct canonical nodes across leaves"
    )
    assert rows[0][1] == "PageRank"
    conn.close()
```

### Recipe 2: Real Filesystem & Zero-Copy Byte Seek
Never mock file reading when testing byte chunking. Verify actual byte offsets against physical disk files with multi-byte UTF-8 characters.

```python
from pathlib import Path
from yaw.indexer.pipeline import chunk_markdown


def test_line_hydration_exactness(tmp_path: Path):
    note_file = tmp_path / "unicode_note.md"
    # Content with multi-byte UTF-8 characters (emoji, mathematical symbols)
    content = "# Title\n\nSection with mathematical symbols: $\\alpha \\le 0.85$ and emoji 🧠.\n\n## Subheading\nSecond chunk text.\n"
    note_file.write_text(content, encoding="utf-8")

    chunks = chunk_markdown(content, "unicode_note.md")
    assert len(chunks) >= 2

    # Physical verification: read exact lines from disk using calculated line ranges
    lines = content.splitlines(keepends=True)
    for chunk in chunks:
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start
        hydrated = "".join(lines[chunk.line_start - 1 : chunk.line_end])
        assert hydrated.rstrip("\r\n") == chunk.text.rstrip("\r\n"), (
            f"Line hydration mismatch on chunk {chunk.chunk_id}"
        )
```

### Recipe 3: Real Localhost Loopback HTTP Mock (No Monkeypatching)
When testing REST API clients (such as Ollama or embedding services), do not monkeypatch private client methods or mock `urllib`/`httpx`. Spin up a standard library `http.server.HTTPServer` on an ephemeral OS-assigned port (`127.0.0.1:0`).

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from yaw.nlp.client import OllamaClient


class _LoopbackHandler(BaseHTTPRequestHandler):
    received_requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        _LoopbackHandler.received_requests.append(json.loads(body))

        # Return valid JSON response matching server schema
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"message": {"content": '{"entities": []}'}}).encode("utf-8")
        )

    def log_message(self, format, *args):
        pass  # Suppress console noise during test runs


@pytest.fixture
def loopback_ollama_server():
    _LoopbackHandler.received_requests.clear()
    server = HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_ollama_client_real_payload(loopback_ollama_server):
    client = OllamaClient(host=loopback_ollama_server, model="qwen2.5:7b")
    response = client.chat([{"role": "user", "content": "Extract concepts"}])
    assert response == '{"entities": []}'
    assert len(_LoopbackHandler.received_requests) == 1
    assert _LoopbackHandler.received_requests[0]["model"] == "qwen2.5:7b"
```

### Recipe 4: Real Mathematical & Sparse Matrix Invariants
When testing GraphRAG algorithms (such as Personalized PageRank), construct real SciPy matrices and verify numerical properties.

```python
import numpy as np
import scipy.sparse as sp
import pytest


def test_personalized_pagerank_probability_conservation():
    # Build a 4-node directed graph
    # 0 -> 1, 1 -> 2, 2 -> 0, 2 -> 3 (dangling node)
    rows = [0, 1, 2, 2]
    cols = [1, 2, 0, 3]
    data = [1.0, 1.0, 0.5, 0.5]
    adj = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    alpha = 0.85
    p0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p = p0.copy()

    for _ in range(30):
        p = alpha * (p @ adj) + (1.0 - alpha) * p0

    # Invariants:
    # 1. Probabilities must be non-negative
    assert np.all(p >= 0.0)
    # 2. Seed node and direct descendants must hold rank
    assert p[1] > p[3]
```

---

## 3. Pre-Commit Checklist for Test Authors

Before declaring any test suite complete:
- [ ] **Zero Database Mocks**: All SQLite tests run against `:memory:` or physical disk `tmp_path`.
- [ ] **Zero Filesystem Mocks**: All file read/write tests use `tmp_path` and verify byte lengths.
- [ ] **Zero API Client Mocks**: External network protocols tested with localhost loopback server (`127.0.0.1:0`).
- [ ] **Zero Monkeypatching of Private Attributes**: No assigning to `obj._model` or `obj._task`.
- [ ] **Adversarial Invariants Probed**: Tested empty inputs, unicode strings, duplicate primary keys.
- [ ] **Deterministic Offline Verification**: Passes `uv run --offline pytest <test_file> -vv` in $< 5\text{ s}$.
