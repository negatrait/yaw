# Open IE: research foundation and YAW interpretation

**Status:** Evidence reference, reconciled 2026-10-12.\
**Normative contract:** [FOUNDATIONAL](../FOUNDATIONAL.md). Research findings motivate decisions; they do not replace the application contract or prove its implementation.

## Open discovery and structured output

Open IE extracts relations without an exhaustive predefined domain relation inventory. A structured response containing mentions, arguments, qualifiers and provenance is compatible with that goal. Ontology alignment can follow extraction within one incremental pipeline.

YAW uses this distinction to keep unknown relations and uncertain interpretations rather than discarding them. NER, labels and ontologies may supply candidates; they must not define the entire universe of discoverable arguments.

## Primary sources and limits

| Source | Relevant evidence | YAW interpretation and boundary |
| --- | --- | --- |
| [Banko et al., IJCAI 2007, Open Information Extraction from the Web](https://www.eng.utah.edu/~cs6961/papers/banko-ijca07.pdf) | Relation-independent extraction at scale | Supports open relation discovery. Previous purported verbatim quotations in this document have been removed in favor of this paraphrase. |
| [Niklaus et al., COLING 2018, A Survey on Open Information Extraction](https://aclanthology.org/C18-1326/) | Reviews Open IE methods and their evolution | Historical cautions about NER are not a universal prohibition on modern linguistic components. Test whether a component restricts recall in YAW. |
| [Liu et al., arXiv:2208.08690v7](https://arxiv.org/abs/2208.08690v7) | Survey spanning rule-based approaches through LLMs | The verified record is a preprint revised 23 October 2024; do not attribute it to ACL/EMNLP 2024 without a verified publication record. |
| [Zhang and Soh, EDC, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.548/) | Extraction, schema definition and canonicalization as distinct responsibilities | Supports separating evidence and alignment. It does not validate YAW's old deterministic identity heuristics or require separate batch ingestion stages. |
| [Gashteovski et al., MinIE, EMNLP 2017](https://aclanthology.org/D17-1278/) | Compact extractions with polarity, modality, attribution and quantity annotations | Motivates qualified assertions rather than flattening every statement into an unconditional triple. |
| [OpenFact, TACL 2023](https://aclanthology.org/2023.tacl-1.40/) | Factuality-enhanced open extraction | Supports explicitly evaluating factuality distinctions, without implying that extraction establishes world truth. |
| [Mrkšić et al., NAACL 2016](https://aclanthology.org/N16-1018/) | Counter-fitting distinguishes lexical constraints from distributional proximity | Related vectors do not justify predicate equivalence or an antonym-safe universal normalizer. |
| [HippoRAG 2](https://arxiv.org/html/2502.14802v2) | Query-to-fact retrieval experiments and ablations | Motivates YAW's fact channel. The reported Recall@5 improvement belongs to its benchmark/model setup, not a guaranteed YAW gain. |
| [BenchIE, ACL 2022](https://aclanthology.org/2022.acl-long.307/) and [BenchIE-FL, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.496/) | Fact-based evaluation and reannotation | Motivate extraction evaluation alongside retrieval QA; acceptable alternative surface realizations and downstream suitability matter. |

## Evaluation consequences

Use independently labelled examples covering unfamiliar predicates, direction, negation/modality, attribution, literals, nested/repeated mentions and polysemy. Model-produced or heuristic-produced verdicts are predictions, not gold calibration labels.

Keep calibration and held-out data disjoint. Report false identity merges and grounding errors separately from extraction F1 and retrieval Recall@k. Record model/configuration/resource versions and costs. Loopback fixtures verify orchestration only.

The [review](reviews/open-ie-ontology-review-2026-10-12.md) connects these sources to defects at its pinned baseline. [Epic 7.8](epic7.8-open-ie-ontology-alignment.md) tracks correction; the [spaCy guide](how-to-handle-spacy-llm.md) describes the selected implementation pattern.
