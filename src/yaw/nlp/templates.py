"""Jinja2 prompt templates for spacy-llm tasks."""

from __future__ import annotations

from yaw.nlp.utils import is_valid_entity_text

OPEN_IE_REL_TEMPLATE = """The text below contains pre-extracted entities, denoted in the following format within the text:
{# whitespace #}
<entity text>[ENT<entity id>:<entity label>]
{# whitespace #}
From the text below, extract all meaningful, directed semantic relations between entities.
{# whitespace #}
The extraction has to use the following format, with one line for each detected relation:
{# whitespace #}
{"dep": <entity id>, "dest": <entity id>, "relation": "<predicate_name>"}
{# whitespace #}
Make sure that only relevant relations are listed, and that each line is a valid JSON object.
{# whitespace #}
{%- if prompt_examples -%}
Below are some examples (only use these as a guide):
{# whitespace #}
{%- for example in prompt_examples -%}
Text:
'''
{%- if preannotate is defined %}
{{ preannotate(example) }}
{%- else %}
{{ example.text }}
{%- endif %}
'''
{%- if example.relations is defined -%}
{% for item in example.relations %}
{%- if item.model_dump_json is defined %}
{{ item.model_dump_json() }}
{%- elif item.dep is defined %}
{"dep": {{ item.dep }}, "dest": {{ item.dest }}, "relation": "{{ item.relation }}"}
{%- else %}
{{ item }}
{%- endif %}
{% endfor %}
{%- endif %}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{%- endif -%}
{# whitespace #}
Here is the text that needs labeling:
{# whitespace #}
Text:
'''
{{ text }}
'''"""

OPEN_IE_RAW_TRIPLES_TEMPLATE = """{%- if prompt_examples is defined and prompt_examples -%}
Below are some examples (only use these as a guide):
{# whitespace #}
{%- for example in prompt_examples -%}
{# whitespace #}
Text:
'''
{{ example.text }}
'''
Triples:
{{ example.reply }}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{%- endif -%}
{# whitespace #}
From the text below, extract all factual, directed relational triples between entities.
{# whitespace #}
Each triple consists of:
- "subject": the source entity mention extracted verbatim from the text
- "predicate": the relational phrase or action connecting the subject and object
- "object": the target entity mention or attribute extracted verbatim from the text
{# whitespace #}
The extraction has to use JSON lines format, with one line for each detected triple:
{# whitespace #}
{"subject": "<subject>", "predicate": "<predicate>", "object": "<object>"}
{# whitespace #}
Or a JSON array of objects:
{# whitespace #}
[{"subject": "<subject>", "predicate": "<predicate>", "object": "<object>"}]
{# whitespace #}
Guidelines:
- Each triple must capture a concrete semantic relationship directly stated in the text.
- "subject" and "object" must be entity mentions or terms extracted verbatim from the text.
- "predicate" must be a concise relation phrase connecting the subject to the object.
- Output valid JSON only without markdown code fences (do NOT include ``` or ```json).
- Do not output any preamble, greeting, commentary, or explanation.
{# whitespace #}
Here is the text that needs extraction:
{# whitespace #}
Text:
'''
{{ text }}
'''
{# whitespace #}
Triples:"""

HYPERNYM_INFERENCE_TEMPLATE = """{%- if prompt_examples is defined and prompt_examples -%}
Below are some examples (only use these as a guide):
{# whitespace #}
{%- for example in prompt_examples -%}
{# whitespace #}
Parent: {{ example.parent_name if example.parent_name is defined else "" }}
Members: {{ example.members if example.members is defined else "" }}
Reply:
{{ example.reply }}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{%- endif -%}
{# whitespace #}
You are an ontology specialist performing hierarchical hypernym induction and cluster abstraction.
{# whitespace #}
Given a parent domain concept and a cluster of child member concepts, induce a concise, intermediate hypernym superclass concept that categorizes these members under the parent concept.
{# whitespace #}
Parent Concept:
- Name: {{ parent_name if parent_name is defined else "" }}
- Definition: {{ parent_definition if parent_definition is defined else "" }}
{# whitespace #}
Cluster Member Concepts:
{%- if member_nodes is defined and member_nodes -%}
  {%- set node_list = member_nodes -%}
{%- elif members is defined and members -%}
  {%- set node_list = members -%}
{%- elif cluster_members is defined and cluster_members -%}
  {%- set node_list = cluster_members -%}
{%- else -%}
  {%- set node_list = [] -%}
{%- endif -%}
{%- for member in node_list %}
- Name: {{ member.name if member.name is defined else (member.get('name', '') if member.get is defined else member) }}
  Definition: {{ member.definition if member.definition is defined else (member.get('definition', '') if member.get is defined else "") }}
{%- endfor %}
{%- if text is defined and text and not node_list %}
{{ text }}
{%- endif %}
{# whitespace #}
Task Requirements:
1. "hypernym_name": Induce a concise 1-3 word noun phrase representing the hypernym superclass concept that subsumes the cluster members and inherits from the parent namespace context.
2. "definition": Author a single-sentence definition gloss explaining the superclass concept in alignment with the parent concept.
3. Output strictly a single JSON object in the following format:
{# whitespace #}
{"hypernym_name": "<concise 1-3 word superclass>", "definition": "<single-sentence gloss>"}
{# whitespace #}
4. Output valid JSON only without markdown code fences (do NOT include ``` or ```json).
5. Do not include any preamble, commentary, conversational filler, or markdown formatting outside the JSON object.
{# whitespace #}
JSON Output:"""

PROMPT_PREFIXES: dict[str, str] = {
    "Retrieval-query": "task: search result | query: ",
    "Reranking": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "STS": "task: sentence similarity | query: ",
    "Fact-checking": "task: claim verification | query: ",
    "Question-answering": "task: question answering | query: ",
}

__all__ = [
    "HYPERNYM_INFERENCE_TEMPLATE",
    "OPEN_IE_RAW_TRIPLES_TEMPLATE",
    "OPEN_IE_REL_TEMPLATE",
    "PROMPT_PREFIXES",
    "is_valid_entity_text",
]
