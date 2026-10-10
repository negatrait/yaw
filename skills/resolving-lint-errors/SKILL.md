---
name: resolving-lint-errors
description: >-
  Properly fix linter (ruff) and typechecker (ty, mypy) errors at the root cause
  without blanket exception suppression or `# noqa` cop-outs. Trigger on 'fix lint',
  'lint error', 'ruff error', 'ty check', 'type error', 'type diagnostic', 'BLE001',
  'S110', 'noqa', 'blind exception', 'resolving lint errors'. Prioritize specific
  exception types, proper structured logging, type narrowing, and protocol conformance.
  Never add blanket `# noqa` or `except Exception: pass`.
---

# Resolving Lint & Type Errors: Root-Cause Remediation Runbook

Use this operational runbook to diagnose and fix linter (`ruff`) and typechecker (`ty`, `mypy`) errors properly at the root cause. 

**The Golden Law of Static Analysis**: A linter or type diagnostic is not an obstacle to silence with `# noqa` or broad exception catching—it is the compiler signaling an unhandled failure domain, an incomplete contract, or fragile design. **Fix the structural cause, never suppress the messenger.**

```
┌─────────────────────────────────────────────────────────────┐
│          ROOT-CAUSE LINT & TYPE REMEDIATION CYCLE           │
├─────────────────────────────────────────────────────────────┤
│  PHASE 1: DIAGNOSTIC TRIANGULATION                          │
│  • Pinpoint rule ID (`BLE001`, `S110`, `UP035`, etc.)       │
│  • Inspect the actual failure domain and call hierarchy     │
│                                                             │
│  PHASE 2: SEMANTIC EXCEPTION ARCHITECTURE                   │
│  • Replace blind `except Exception` with specific errors    │
│  • Banish silent `except: pass` -> add contextual logging   │
│  • Reserve top-level boundary catches strictly for daemons  │
│                                                             │
│  PHASE 3: TYPE NARROWING & PROTOCOL CONFORMANCE             │
│  • Fix type signature mismatches (list vs tuple, etc.)      │
│  • Narrow types with `isinstance()` & guard clauses         │
│  • Annotate class attributes with `ClassVar`                │
│                                                             │
│  PHASE 4: DETERMINISTIC VERIFICATION GATES                  │
│  • Run: `uv run ruff check <files>` (assert 0 errors)       │
│  • Run: `uv run ty check <files>` (assert 0 diagnostics)    │
│  • Run: `uv run --offline pytest` (assert 100% pass)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. The Cognitive Traps & Negative Space of Lint Evasions

| The Evasion Anti-Pattern | Why Agents Do It (The Cop-Out) | The Production Damage | The Proper Architectural Fix |
| :--- | :--- | :--- | :--- |
| **Blanket `# noqa: BLE001`** | Too lazy to check what exceptions the called code can raise. | Swallows `KeyboardInterrupt`, memory errors, syntax bugs, and unexpected crashes silently. | Catch specific exceptions: `(httpx.HTTPError, sqlite3.Error, json.JSONDecodeError, OSError)`. |
| **Silent `except ...: pass` (`S110`)** | Wants to prevent crashes without writing logging logic. | Corrupts data silently; when operations fail, no trace exists in logs or monitoring. | Always log with context: `logger.warning("Failed to deserialize embedding for %s: %s", uri, err)`. |
| **`# type: ignore` / `cast(Any, x)`** | Type signature doesn't match and agent avoids changing code. | Defeats the typechecker completely; runtime `AttributeError` or `TypeError` crashes production. | Narrow types with `isinstance(x, ...)` or align caller/callee types (e.g. `list(config.PATTERNS)`). |
| **Breaking API Signatures to Please Linter** | Changing a public method signature to avoid a warning. | Breaks external callers, MCP protocol contracts, and test fixtures across the codebase. | Adapt internal transformation logic while preserving public API invariants and contracts. |
| **Cargo-Cult Import Replacement (`UP035`)** | Automatically replacing `typing.Iterable` with `collections.abc.Iterable` without checking runtime callers. | Crashes third-party runtime introspection libraries (like `spacy-llm`) that inspect `typing` module types. | Verify runtime library protocol requirements; document necessary exceptions with upstream issue context. |

---

## 2. Step-by-Step Remediation Recipes

### Recipe 1: Replacing Blind `BLE001` (`except Exception`) with Specific Exceptions
Blind exception catching is dangerous because it intercepts unexpected errors (like typos, missing keys, or out-of-memory) that should bubble up or be handled explicitly.

#### Before (Cop-Out):
```python
# BAD: Silently catches any error and masks underlying defects
try:
    resp = client.chat(messages)
except Exception:  # noqa: BLE001
    return None
```

#### After (Root-Cause Remediation):
```python
# GOOD: Catches specific network and parsing errors, logs context, and handles failure cleanly
import urllib.error
import http.client
import json

try:
    resp = client.chat(messages)
except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as net_err:
    logger.warning("Ollama connection failed for model %s: %s", self.model, net_err)
    return None
except json.JSONDecodeError as json_err:
    logger.error("Ollama returned malformed JSON payload: %s", json_err)
    return None
```

### Recipe 2: Banishing `S110` (`try-except-pass`) with Structured Logging
Catching an exception and executing `pass` with no logging or recovery is an operational crime.

#### Before (Cop-Out):
```python
# BAD: Silent failure leaves zero forensic telemetry
try:
    conn.execute("SELECT vec_distance_cosine(v, ?) FROM vectors", (q_vec,))
except Exception:  # noqa: BLE001, S110
    pass
```

#### After (Root-Cause Remediation):
```python
# GOOD: Catches specific sqlite operational errors and logs contextual diagnostics
import sqlite3

try:
    conn.execute("SELECT vec_distance_cosine(v, ?) FROM vectors", (q_vec,))
except sqlite3.OperationalError as db_err:
    logger.warning("sqlite-vec extension query failed on vectors table: %s", db_err)
    # Explicit domain fallback if intentional:
    return []
```

### Recipe 3: Type Signature & Constructor Alignment (Ty / Mypy)
When Ty or Mypy flags argument type mismatches (such as passing a `tuple[str, ...]` where `list[str] | None` is expected by an external library constructor), never cast to `Any`.

#### Before (Cop-Out):
```python
# BAD: Casting to Any disables all static type safety
matcher = PatternFilePathMatcher(
    included_patterns=cast(Any, config.INCLUDED_PATTERNS)  # noqa: UP035
)
```

#### After (Root-Cause Remediation):
```python
# GOOD: Explicitly convert tuple to list matching the library's exact parameter type
matcher = PatternFilePathMatcher(
    included_patterns=list(config.INCLUDED_PATTERNS)
    if config.INCLUDED_PATTERNS
    else None,
    excluded_patterns=list(config.EXCLUDED_PATTERNS)
    if config.EXCLUDED_PATTERNS
    else None,
)
```

### Recipe 4: Resolving `RUF012` (Mutable Class Attributes in Tests)
Ruff flags mutable class attributes (such as lists or dicts defined at class scope) because class attributes are shared across all instances.

#### Before (Cop-Out):
```python
# BAD: Mutable class attribute flagged by RUF012
class _MockHandler(BaseHTTPRequestHandler):
    received_requests = []  # noqa: RUF012
```

#### After (Root-Cause Remediation):
```python
# GOOD: Explicitly annotate with ClassVar to declare intentional shared state
from typing import ClassVar


class _MockHandler(BaseHTTPRequestHandler):
    received_requests: ClassVar[list[dict[str, Any]]] = []
```

### Recipe 5: Resolving `RUF007` (Pairwise Iteration)
When iterating over adjacent elements in a sequence, avoid index arithmetic or manual zips.

#### Before (Cop-Out):
```python
# BAD: Manual zip indexing
for a, b in zip(items[:-1], items[1:]):  # noqa: RUF007
    process(a, b)
```

#### After (Root-Cause Remediation):
```python
# GOOD: Use standard library itertools.pairwise
from itertools import pairwise

for a, b in pairwise(items):
    process(a, b)
```

---

## 3. The Strict Policy on `# noqa`

Adding `# noqa` is **strictly prohibited** across all production code (`src/`), except in the singular following case:
1. **Third-Party Runtime Protocol Invariance**: When an external library (e.g. `spacy-llm`) performs dynamic AST or runtime type introspection that explicitly depends on deprecated typing constructs (e.g. checking `isinstance(t, typing.Iterable)` rather than `collections.abc.Iterable`). 
2. In this rare case, the `# noqa` MUST be accompanied by an inline comment citing the upstream library requirement:
   ```python
   # spaCy-LLM registry protocol requires typing.Iterable for task validation
   from typing import Any, Callable, Iterable  # noqa: UP035
   ```

Any other use of `# noqa` (especially `# noqa: BLE001`, `# noqa: S110`, `# type: ignore`) will be rejected by code review.

---

## 4. Pre-Commit Verification Gate

Before completing any task or committing changes:
1. Run `uv run ruff check <modified_files>` $\to$ **Assert 0 errors** without unapproved `# noqa`.
2. Run `uv run ty check <modified_files>` $\to$ **Assert 0 diagnostics** without unapproved `# type: ignore`.
3. Run `uv run --offline pytest` $\to$ **Assert 100% test pass rate**.
