# YAW BDD User Stories & Behavioral Specifications

**Document Status:** Living Behavioral Specification  
**Format:** BDD Gherkin (Given-When-Then)  
**Governing Canon:** [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md)

---

## Overview

This document specifies the authoritative Acceptance Criteria and Behavioral Driven Development (BDD) scenarios for upcoming YAW milestones:
1. **Sprint 5.1:** Mid-Epic 5 Remediation & Pre-E5S2 Architectural Alignment Sprint
2. **Sprint 5B (E5S2):** Dynamic Few-Shot Steering & Pipeline Feedback Loop
3. **Epic 6, Sprint 6A:** RDF Serialization & Quad Store Integration
4. **Epic 6, Sprint 6B:** OWL Ontology Semantics & Axiom Inference
5. **Epic 6, Sprint 6C:** SHACL Closed-World Wiki Linter (`lint_wiki` tool)
6. **Epic 7:** Production FastMCP Server & Streaming HTTP Access

---

## 1. Sprint 5.1: Mid-Epic 5 Remediation & Pre-E5S2 Architectural Alignment Sprint

### Feature: Mid-Epic 5 Substrate Remediation & Architectural Alignment
As a YAW platform engineer  
I want the SQLite relational views, predicate configuration SSOT, namespace defaults, and GraphRAG conductance models to be 100% aligned with ADR-008 and Epics 6–7  
So that Sprint 5B prompt steering operates on verified canonical vocabularies and downstream RDF/OWL/SHACL components receive valid quad schemas.

#### Scenario 1: Authoritative Predicate SSOT & Obsolete Synonym Purge
```gherkin
Given the configuration module "src/yaw/config.py"
When "config.BASE_PREDICATES" is inspected
Then it must contain exactly the 14 canonical predicates defined in ADR-008
And the uncontrolled synonym "is_a" must NOT be present
And FastMCP endpoint "get_ontology_schema()" must return the identical 14 canonical predicates
```

#### Scenario 2: Plural Root Namespace Defaulting in EDC Resolver
```gherkin
Given a candidate entity string "quantum-tunneling" without an explicit namespace prefix
When "canonicalize.resolve('quantum-tunneling')" is invoked
Then the resolved identifier must default to "concepts:quantum-tunneling"
And the singular prefix "concept:" must NEVER be generated
And the prefix must match an entry in "config.TAXONOMY_NAMESPACES"
```

#### Scenario 3: SQLite Aliases Method CHECK Constraint Compatibility
```gherkin
Given an initialized SQLite database with aliases schema
When an alias record is inserted with method "exact" or "alias" from "MatchMethod"
Then the database insertion must succeed
And no "sqlite3.IntegrityError: CHECK constraint failed: aliases.method" must be raised
```

#### Scenario 4: 5-Column unified_edges Quad Projection
```gherkin
Given an active SQLite database with indexed "wikilinks" and "leaf_relations"
When the "unified_edges" view is queried
Then it must return 5 columns: "src_id", "dst_id", "rel_type", "weight", "source_chunk_id"
And "source_chunk_id" must contain non-empty chunk or source URI provenance
And "retriever.py" must successfully construct its adjacency matrix without column unpacking errors
```

#### Scenario 5: Compiled Knowledge Substrate Isolation
```gherkin
Given a saturated hub split executed during "compile_wiki"
When an induced hypernym node and rewired "subClassOf" edge are persisted
Then they must be written to "compiled_nodes" and "compiled_relations" tables
And the raw tables "leaf_nodes" and "leaf_relations" must remain unmodified
And the views "unified_nodes" and "unified_edges" must seamlessly reflect the compiled knowledge
```

#### Scenario 6: Taxonomic Subsumption Conductance Asymmetry
```gherkin
Given an edge "(Child, subClassOf, Parent)" in "unified_edges"
When "GraphRAGRetriever._build_matrix()" builds the sparse transition matrix
Then the upward transition probability "Child -> Parent" must be damped with conductance 0.2
And a reciprocal downward transition "Parent -> Child" must be synthesized with conductance 1.0
```

#### Scenario 7: Automated Contested Alias Transition
```gherkin
Given a candidate pair with vector cosine similarity between 0.80 and 0.95
When Tier 3 LLM adjudication yields a low-confidence decision or dialectical ambiguity
Then a row must be inserted into "aliases" with "status = 'contested'"
And the contested record must be queryable via "get_contested_aliases()" for SHACL validation
```

---

## 2. Sprint 5B: Dynamic Few-Shot Steering & Pipeline Feedback Loop

### Feature: Dynamic Few-Shot Ontology Extraction Steering
As a continuous ingestion indexer  
I want to steer the spaCy-LLM extraction pipeline using dynamic canonical vocabulary and few-shot examples drawn live from SQLite and ADR-008 seed predicates  
So that extracted entities and predicates converge to established canonical terms before string resolution, preventing graph fragmentation and outputting quad-ready triples.

#### Scenario 1: Declarative Registration of Dynamic Example Provider
```gherkin
Given a running YAW instance with populated canonical concepts in SQLite "leaf_nodes"
When the spaCy-LLM pipeline is constructed via "build_leaf_pipeline()"
Then the factory "@registry.misc('yaw.CanonicalExamples.v1')" must be resolved from spaCy's global registry
And calling the factory must return a Callable producing dynamic few-shot extraction examples
And the pipeline must build without private "_model" or "_task" monkeypatching
```

#### Scenario 2: Dynamic Injection of Established Canonical Vocabulary
```gherkin
Given an indexed vault containing high-conductance canonical concept "concepts:quantum-entanglement"
And an active alias "entangled particles" pointing to "concepts:quantum-entanglement"
When a new raw document containing the sentence "Entangled particles demonstrate nonlocal state collapse." is ingested
Then the dynamic few-shot steering injector must include "quantum-entanglement" in the prompt's candidate vocabulary
And the extracted Subject entity must be emitted directly as "quantum-entanglement"
And zero novel duplicate concept nodes must be created in "leaf_nodes"
```

#### Scenario 3: Multi-Note Synonym Convergence Across Extraction Sessions
```gherkin
Given two distinct raw notes ingested sequentially:
  | note_path                 | content                                                     |
  | raw/physics/relativity.md | "Albert Einstein formulated the theory of special relativity." |
  | raw/physics/spacetime.md  | "Einsteinian relativity predicts gravitational time dilation." |
When the ingestion pipeline finishes processing both notes
Then the unified view "unified_nodes" must contain exactly one canonical node for the concept
And the "aliases" table must record an active alias mapping "einsteinian relativity" to the canonical slug
And the knowledge graph must contain zero disconnected synonym duplicates
```

#### Scenario 4: Bounded Latency and Memory Cap on Injected Examples
```gherkin
Given a mature vault containing over 10,000 indexed canonical concepts
When dynamic few-shot examples are assembled for chunk extraction
Then the injector must select at most "CANON_FEWSHOT_MAX" (default 5) top-conductance examples
And total injected prompt token overhead must not exceed 250 tokens
And candidate selection latency must execute in under 10 milliseconds
```

#### Scenario 5: Canonical Predicate Steering via ADR-008 Seed Taxonomy
```gherkin
Given a raw markdown note containing relations across subsumptive, dialectical, and causal statements
When the spaCy-LLM extraction pipeline processes the text with "@registry.misc('yaw.CanonicalExamples.v1')"
Then few-shot prompt injection must prioritize canonical predicates from "config.BASE_PREDICATES"
And extracted relation types must align to ADR-008 canonical categories:
  | Category           | Canonical Predicate |
  | Subsumption        | "subClassOf"        |
  | Dialectical        | "supports"          |
  | Dialectical        | "contradicts"       |
  | Dependency/Process | "causes"            |
  | Provenance         | "derivedFrom"       |
And unconstrained synonyms such as "is_a", "kind_of", or "proves" must not be emitted when canonical alternatives apply
```

#### Scenario 6: Lossless RDF Quad Projection of Steered Extractions
```gherkin
Given extracted relations produced by the steered spaCy-LLM pipeline for note "raw/notes/relativity.md"
When the extracted triples are inserted into SQLite and queried via "unified_edges"
Then each row must project valid W3C quad fields:
  | Field        | Quad Role      | Expected Format                                      |
  | src_id       | subject        | "yaw:<namespace>:<slug>"                             |
  | rel_type     | predicate      | "yaw:predicates:<canonical_predicate>"               |
  | dst_id       | object         | "yaw:<namespace>:<slug>"                             |
  | source_chunk | graph_uri      | "yaw:sources/notes/relativity.md#chunk:<chunk_id>"   |
And the projected quad must be directly serializable into standard N-Quads without schema mapping errors
```

---

## 3. Epic 6, Sprint 6A: RDF Serialization & Quad Store Integration

### Feature: In-Process W3C RDF Triple and Quad Substrate (Zero External Graph Daemons)
As an external semantic consumer, knowledge engineer, or graph database  
I want to query and export YAW's unified SQLite graph as standard W3C RDF triples and named quads  
So that the knowledge graph is interoperable with standard SPARQL engines, RDF quad stores, and semantic web tooling without losing byte-level provenance or requiring external graph daemons.

#### Scenario 1: Lossless Turtle Export from SQLite Unified Views via In-Process rdflib
```gherkin
Given an indexed SQLite database with populated "unified_nodes" and "unified_edges"
When the export function "export_graph(format='turtle')" is executed
Then the output string must be a syntactically valid W3C Turtle document
And standard namespace prefixes must be bound:
  | Prefix | URI                                  |
  | yaw    | https://yaw.local/ontology/          |
  | rdfs   | http://www.w3.org/2000/01/rdf-schema#|
  | owl    | http://www.w3.org/2002/07/owl#       |
  | sh     | http://www.w3.org/ns/shacl#          |
  | skos   | http://www.w3.org/2004/02/skos/core# |
And parsing the Turtle string into an "rdflib.Graph" must yield the exact triple count from "unified_edges"
And the operation must run strictly in-process with zero external graph daemons or background network calls
```

#### Scenario 2: Canonical CURIE URI Minting
```gherkin
Given an edge in "unified_edges" with:
  | Column   | Value                          |
  | src_id   | concepts:quantum-entanglement  |
  | dst_id   | invariants:bell-inequality     |
  | rel_type | violates                       |
When the edge is projected into an RDF triple
Then the subject URI must resolve to "https://yaw.local/ontology/concepts/quantum-entanglement"
And the predicate URI must resolve to "https://yaw.local/ontology/predicates/violates"
And the object URI must resolve to "https://yaw.local/ontology/invariants/bell-inequality"
```

#### Scenario 3: Named Quad Serialization with Document Provenance Context
```gherkin
Given a relation extracted from raw note "raw/science/epr_paper.md" at chunk ordinal 3
When the export function "export_graph(format='nquads')" is executed
Then each triple must include the named graph URI identifying its chunk origin:
  """
  <https://yaw.local/ontology/concepts/epr-paradox>
  <https://yaw.local/ontology/predicates/challenges>
  <https://yaw.local/ontology/concepts/quantum-mechanics>
  <https://yaw.local/sources/raw/science/epr_paper.md#chunk:3> .
  """
And loading the quads into an "rdflib.ConjunctiveGraph" must preserve chunk graph partitioning
```

#### Scenario 4: Multi-Format Serialization Round-Trip without External Daemons
```gherkin
Given a canonical YAW knowledge graph in SQLite
When serialized successively into "turtle", "json-ld", and "nt" formats via in-process "rdflib"
Then each serialized representation must deserialize into an isomorphic RDF graph
And isomorphic graph comparison must confirm zero lost edges, properties, or blank-node skolemization bugs
And no external triple store or graph database service must be spawned or queried
```

---

## 4. Epic 6, Sprint 6B: OWL Ontology Semantics & Axiom Inference

### Feature: Pragmatic OWL 2 RL & SKOS Ontology Formalization (Zero DL Tableaux Reasoners)
As an automated reasoning engine and retrieval graph traverser  
I want taxonomic relationships, hypernym inductions, and alias mappings formalized as OWL 2 RL / RDFS and SKOS axioms  
So that subsumption hierarchies and node equivalences are mathematically well-founded with acyclic directed conductance without heavy DL tableaux solver bottlenecks.

#### Scenario 1: Active Alias Projection to `owl:sameAs` Axioms and `skos:altLabel`
```gherkin
Given an active record in SQLite "aliases":
  | alias_id             | kind   | canonical_id            | status |
  | entities:a-einstein  | entity | concepts:albert-einstein| active |
When the OWL/SKOS ontology projection is generated
Then an equivalence axiom must be declared:
  """
  yaw:entities:a-einstein owl:sameAs yaw:concepts:albert-einstein .
  """
And the canonical node must record the alias as "skos:altLabel"
And querying either identifier must navigate to identical incoming and outgoing relation closures
And resolution must execute via SQLite view joins without invoking external DL reasoners
```

#### Scenario 2: Formalizing B-Tree Saturated Hub Splits as `rdfs:subClassOf` and `skos:broader`
```gherkin
Given a hub node "concepts:machine-learning" exceeding out-degree K_max (10)
When the B-tree hypernym induction protocol splits the cluster into sub-cluster "concepts:supervised-learning"
Then ontological subsumption axioms must be emitted:
  """
  yaw:concepts:supervised-learning rdfs:subClassOf yaw:concepts:machine-learning .
  yaw:concepts:supervised-learning skos:broader yaw:concepts:machine-learning .
  """
And member concepts must be subsumed under the new hypernym:
  """
  yaw:concepts:linear-regression rdfs:subClassOf yaw:concepts:supervised-learning .
  yaw:concepts:linear-regression skos:broader yaw:concepts:supervised-learning .
  """
```

#### Scenario 3: Taxonomic Subsumption Transitive Closure via In-Process SQL Recursive CTE / SciPy
```gherkin
Given the subsumption hierarchy:
  | SubClass            | SuperClass                 |
  | concepts:transformer| concepts:attention-models  |
  | concepts:attention-models | concepts:deep-learning |
  | concepts:deep-learning   | concepts:machine-learning |
When a taxonomic closure query is evaluated for "concepts:transformer"
Then the inferred ancestor set must include:
  | Ancestor                   |
  | concepts:attention-models  |
  | concepts:deep-learning     |
  | concepts:machine-learning  |
And the closure must be computed using an in-process SQLite recursive CTE or SciPy sparse matrix powers
And the subsumption graph must be verified as a strict Directed Acyclic Graph (DAG) with zero cycles
And no external tableaux reasoner (Pellet, HermiT) shall be loaded
```

#### Scenario 4: Asymmetric Conductance Enforcement in Spectral Diffusion
```gherkin
Given a taxonomic edge "yaw:concepts:cnn rdfs:subClassOf yaw:concepts:deep-learning"
When the transition conductance matrix for GraphRAG is constructed
Then the downward transition conductance from "deep-learning" to "cnn" must equal 1.0
And the upward transition conductance from "cnn" to "deep-learning" must be dampened to 0.2
And power iteration must diffuse mass toward concrete leaf implementations without runaway topic drift
```

---

## 5. Epic 6, Sprint 6C: SHACL Closed-World Wiki Linter (`lint_wiki` Tool)

### Feature: In-Process Closed-World SHACL Shape Validation & Wiki Linting
As a wiki author, knowledge graph maintainer, or automated CI quality gate  
I want declarative closed-world SHACL shape validation over the knowledge graph via the "lint_wiki" tool using in-process "pyshacl"  
So that invalid namespaces, missing definitions, broken wikilinks, and dialectical contradictions are surfaced with structured W3C "sh:ValidationReport" outputs with zero external graph daemons.

#### Scenario 1: Validation of ConceptShape (Definitions and URIs via pyshacl)
```gherkin
Given a node in "leaf_nodes" with "node_id = 'concepts:entropy'"
And the node's definition is empty or exceeds the configured word limit (50 words)
When the "lint_wiki()" MCP tool is executed
Then the return payload must contain a W3C "sh:ValidationReport" produced by in-process "pyshacl"
And the report must indicate "sh:conforms = false"
And the violation must identify "sh:MinLengthConstraintComponent" or "sh:MaxLengthConstraintComponent"
And the severity must be flagged as "sh:Violation"
And validation must complete in-process without contacting external services
```

#### Scenario 2: Validation of RelationShape (Domain and Range Constraints)
```gherkin
Given an extracted relation in "leaf_relations":
  | subject_id        | predicate | object_id           |
  | concepts:gravity  | partOf    | non-existent:entity |
When the "lint_wiki()" MCP tool is executed
Then the report must flag an unresolved dangling object target
And the violation message must specify broken referential integrity for "non-existent:entity"
```

#### Scenario 3: Detection of Dialectical Contradictions and Contested Merges
```gherkin
Given an alias in SQLite "aliases" with "status = 'contested'"
And two contradictory claims in "leaf_relations":
  | subject_id       | predicate | object_id       |
  | concepts:coffee  | prevents  | concepts:cancer |
  | concepts:coffee  | causes    | concepts:cancer |
When the "lint_wiki()" MCP tool is executed
Then the linter must detect both dialectical issues
And report the contested alias merge under "contested_aliases"
And report the contradictory predicate pair under "dialectical_contradictions"
```

#### Scenario 4: Filtered Lint Querying via FastMCP Interface with Zero Background Daemons
```gherkin
Given a knowledge graph containing 50 total minor warnings across all 9 namespaces
When a client invokes "lint_wiki(query='concepts')" over FastMCP
Then the returned list must be filtered exclusively to violations concerning the "concepts" namespace
And validation must run entirely in-process against SQLite views with zero background graph daemons
And execution latency must be under 200 milliseconds
```

---

## 6. Epic 7: Production FastMCP Server & Streaming HTTP Access

### Feature: Production FastMCP Server & Streaming HTTP Transport
As an AI pair-programmer, autonomous agent, or remote LLM client  
I want to interact with YAW exclusively through a high-performance FastMCP server over streaming HTTP/SSE with non-blocking continuous indexing  
So that I can ingest raw research, compile wiki materializations, query GraphRAG bridging subgraphs, and lint wiki integrity remotely without local subshell commands or SQLite lock contention.

#### Scenario 1: Complete 4-Tool Production Lifecycle Session
```gherkin
Given a running FastMCP server instance
When a client executes the following 4-step sequence via MCP tool calls:
  1. "ingest_raw(content='# Quantum Computing\nQubits leverage superposition.', title='Quantum Intro')"
  2. "compile_wiki(force=False)"
  3. "query_wiki(query='How do qubits work?', top_k=3)"
  4. "lint_wiki()"
Then all 4 tool calls must return successful JSON/Markdown payloads
And step 1 must write an unmutated markdown file into "vault/raw/"
And step 2 must materialize valid YAML-frontmatter wiki pages under "vault/wiki/"
And step 3 must return a token-dense Bridging Subgraph citing the new raw note
And step 4 must return an empty violation list (valid wiki)
```

#### Scenario 2: Remote Streaming HTTP/SSE Transport Without CLI Assumptions
```gherkin
Given a YAW server started with "yaw serve --host 127.0.0.1 --port 8765 --transport sse"
When a remote client connects to "http://127.0.0.1:8765/sse"
Then the server must establish an open Server-Sent Events (SSE) streaming session
And tool execution requests sent via HTTP POST to "http://127.0.0.1:8765/messages/" must stream responses back over SSE
And the client must require zero local terminal access, shell execution, or CLI binary dependencies on the client host
```

#### Scenario 3: Non-Blocking Background Continuous Indexing Under Concurrent Query Load
```gherkin
Given the FastMCP server is actively processing background CocoIndex ingestion for 100 new markdown notes
When a concurrent client issues 20 simultaneous "query_wiki(query='...')" requests over HTTP/SSE
Then every "query_wiki" request must complete in under 50 milliseconds
And zero requests may fail with "sqlite3.OperationalError: database is locked"
And SQLite WAL mode and busy timeouts must ensure zero query stalls or writer starvation
```

#### Scenario 4: Substrate Mutation Isolation During Wiki Materialization (Invariant 1)
```gherkin
Given a vault containing files in "vault/raw/"
When "compile_wiki(force=True)" executes
Then files must be written strictly to "vault/wiki/{namespace}/{slug}.md"
And directory "vault/raw/" must experience zero file additions, modifications, or timestamp changes
And existing custom frontmatter in "vault/raw/" files must remain 100% byte-identical
```

#### Scenario 5: Dual-Layer GraphRAG Retrieval Contract (Invariant 2 & Invariant 5)
```gherkin
Given an indexed knowledge base containing concepts connected to raw evidence
When "query_wiki(query='Explain CRISPR Cas9 gene editing', top_k=3)" is invoked
Then the returned Markdown response must contain:
  1. "### Concept Graph Context": local bridging subgraph with active seed concepts and taxonomic links
  2. "### Retrieved Source Evidence": exact disk slices hydrated from raw files via byte seeks
And the SQLite database must not store duplicated prose strings (zero-prose disk pointers)
```
