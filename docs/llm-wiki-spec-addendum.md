# Retired: medallion architecture addendum

**Status:** Historical prototype; not implementation instructions.

The separate Bronze/Silver/Gold ingestion systems and manual consolidation barrier are retired. The selected design has one CocoIndex application with explicit source and corpus-compilation dependencies; logical dependency stages do not create a second indexing system.

Use [FOUNDATIONAL](../FOUNDATIONAL.md), [continuous compilation](knowledge-compilation.md) and the [Epic 7.8 execution contract](epic7.8-open-ie-ontology-alignment.md#coding-agent-execution-contract). This file owns no current pipeline or acceptance gate.
