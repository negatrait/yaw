"""Closed-world SQL SHACL linter, repair execution, focus-node inspection, and W3C Turtle serializer."""

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yaw.core import SHACLViolation

from .db import YawConnection, batched, write_tx
from .graph_ops import (
    LABEL_KIND_PRIORITY,
    get_existing_ids,
    merge_entities,
    upsert_entity_labels,
    upsert_semantic_edges,
)


def inspect_focus_node(
    conn: sqlite3.Connection | YawConnection,
    focus_node: str,
) -> dict[str, Any]:
    """Read-only neighborhood inspection tool for RLM synonym judgment and SHACL healing."""
    ent_row = conn.execute(
        "SELECT entity_id, name, definition FROM entities WHERE entity_id = ? OR name = ? LIMIT 1",
        (focus_node, focus_node),
    ).fetchone()
    eid = str(ent_row[0]) if ent_row is not None else focus_node
    entity_info = (
        {"entity_id": str(ent_row[0]), "name": str(ent_row[1]), "definition": str(ent_row[2])}
        if ent_row is not None
        else None
    )

    labels = [
        {
            "label_norm": str(r[0]),
            "label_kind": str(r[1]),
            "raw_label": str(r[2]),
            "chunk_id": str(r[3]),
        }
        for r in conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = ?",
            (eid,),
        )
    ]

    facts = [
        {
            "fact_id": str(r[0]),
            "subj_id": str(r[1]),
            "pred": str(r[2]),
            "obj_id": str(r[3]) if r[3] is not None else "",
            "raw_triple": json.loads(r[4]),
            "definition": str(r[5]),
            "obj_kind": str(r[6]),
            "obj_literal_lex": str(r[7]) if r[7] is not None else "",
            "obj_datatype": str(r[8]),
            "obj_lang": str(r[9]),
        }
        for r in conn.execute(
            """
            SELECT
                fact_id,
                subj_id,
                pred,
                obj_id,
                raw_triple_json,
                definition,
                COALESCE(obj_kind, 'entity'),
                obj_literal_lex,
                COALESCE(obj_datatype, ''),
                COALESCE(obj_lang, '')
            FROM facts
            WHERE fact_id = ? OR subj_id = ? OR obj_id = ?
            """,
            (focus_node, eid, eid),
        )
    ]

    edges = [
        {
            "src_id": str(r[0]),
            "dst_id": str(r[1]),
            "kind": str(r[2]),
            "weight": float(r[3]),
        }
        for r in conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
            (focus_node, eid, focus_node, eid),
        )
    ]

    chunk_ids_set: set[str] = {
        item["chunk_id"] for item in labels if item["chunk_id"] and not item["chunk_id"].startswith("__")
    }
    for r in conn.execute(
        "SELECT DISTINCT chunk_id FROM edge_sources WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
        (focus_node, eid, focus_node, eid),
    ):
        chunk_ids_set.add(str(r[0]))
    for r in conn.execute(
        """
        SELECT DISTINCT fs.chunk_id
        FROM fact_sources fs
        INNER JOIN facts f ON f.fact_id = fs.fact_id
        WHERE f.fact_id = ? OR f.subj_id = ? OR f.obj_id = ?
        """,
        (focus_node, eid, eid),
    ):
        chunk_ids_set.add(str(r[0]))
    for e in edges:
        if e["kind"] == "passage" and e["dst_id"] == eid:
            chunk_ids_set.add(str(e["src_id"]))

    return {
        "focus_node": focus_node,
        "entity": entity_info,
        "labels": labels,
        "facts": facts,
        "edges": edges,
        "chunk_ids": sorted(chunk_ids_set),
    }


def _scan_shacl_violations(conn: sqlite3.Connection | YawConnection) -> list[SHACLViolation]:
    violations: list[SHACLViolation] = []

    # 1. ConceptShape/prefLabel: Every entity must have exactly 1 distinct skos:prefLabel norm
    for row in conn.execute(
        """
        SELECT
            e.entity_id,
            COUNT(DISTINCT CASE WHEN el.label_kind = 'skos:prefLabel' THEN el.label_norm END) AS pref_cnt
        FROM entities e
        LEFT JOIN entity_labels el ON el.entity_id = e.entity_id
        GROUP BY e.entity_id
        HAVING pref_cnt != 1
        ORDER BY e.entity_id ASC
        """
    ):
        eid, cnt = str(row[0]), int(row[1])
        msg = (
            f"Entity {eid} has no skos:prefLabel in entity_labels."
            if cnt == 0
            else f"Entity {eid} has {cnt} distinct skos:prefLabel norms (expected 1)."
        )
        violations.append(
            SHACLViolation(
                shape="ConceptShape/prefLabel",
                focus_node=eid,
                result_path="skos:prefLabel",
                message=msg,
            )
        )

    # 2. ConceptShape/definition: Every entity must have a non-empty definition
    for row in conn.execute(
        """
        SELECT entity_id, name
        FROM entities
        WHERE TRIM(definition) = ''
        ORDER BY entity_id ASC
        """
    ):
        eid, name = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/definition",
                focus_node=eid,
                result_path="skos:definition",
                message=f"Entity {eid} ({name!r}) has an empty skos:definition.",
            )
        )

    # 3. ConceptShape/S13-disjointLabels:
    # (a) No (entity_id, label_norm) pair may have >1 distinct label_kind across chunks
    # (b) No entity's canonical name (LOWER(TRIM(e.name))) may appear as skos:altLabel or skos:hiddenLabel
    seen_s13: set[tuple[str, str]] = set()
    for row in conn.execute(
        """
        SELECT entity_id, label_norm, COUNT(DISTINCT label_kind) AS kind_cnt
        FROM entity_labels
        GROUP BY entity_id, label_norm
        HAVING kind_cnt > 1
        ORDER BY entity_id ASC, label_norm ASC
        """
    ):
        eid, lnorm = str(row[0]), str(row[1])
        seen_s13.add((eid, lnorm))
        violations.append(
            SHACLViolation(
                shape="ConceptShape/S13-disjointLabels",
                focus_node=eid,
                result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                message=f"Entity {eid} label {lnorm!r} violates SKOS S13 disjointness across chunks.",
            )
        )
    for row in conn.execute(
        """
        SELECT DISTINCT el.entity_id, el.label_norm
        FROM entity_labels el
        INNER JOIN entities e ON e.entity_id = el.entity_id
        WHERE el.label_kind IN ('skos:altLabel', 'skos:hiddenLabel')
          AND el.label_norm = LOWER(TRIM(e.name))
        ORDER BY el.entity_id ASC, el.label_norm ASC
        """
    ):
        eid, lnorm = str(row[0]), str(row[1])
        if (eid, lnorm) not in seen_s13:
            seen_s13.add((eid, lnorm))
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/S13-disjointLabels",
                    focus_node=eid,
                    result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                    message=f"Entity {eid} canonical name {lnorm!r} appears as non-prefLabel in entity_labels.",
                )
            )

    # 4. ConceptShape/homographDifferentFrom: Distinct entities sharing a skos:prefLabel norm must have
    # owl:differentFrom
    for row in conn.execute(
        """
        SELECT a.entity_id, b.entity_id, a.label_norm
        FROM entity_labels a
        INNER JOIN entity_labels b
          ON a.label_norm = b.label_norm
         AND a.label_kind = 'skos:prefLabel'
         AND b.label_kind = 'skos:prefLabel'
         AND a.entity_id < b.entity_id
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e
            WHERE e.src_id = a.entity_id
              AND e.dst_id = b.entity_id
              AND e.kind = 'owl:differentFrom'
        )
        GROUP BY a.entity_id, b.entity_id, a.label_norm
        ORDER BY a.entity_id ASC, b.entity_id ASC
        """
    ):
        ea, eb, lnorm = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/homographDifferentFrom",
                focus_node=f"{ea}|{eb}",
                result_path="owl:differentFrom",
                message=f"Homograph concepts {ea} and {eb} share prefLabel {lnorm!r} without owl:differentFrom.",
            )
        )

    # 5. StatementShape/closedWorld: Facts must have valid subj_id in entities, valid entity/literal object,
    # and non-empty definition
    for row in conn.execute(
        """
        SELECT
            f.fact_id,
            f.subj_id,
            f.obj_id,
            f.definition,
            es.entity_id IS NOT NULL AS has_subj,
            CASE
                WHEN COALESCE(f.obj_kind, 'entity') = 'literal'
                THEN (f.obj_id IS NULL AND TRIM(COALESCE(f.obj_literal_lex, '')) != '')
                ELSE (eo.entity_id IS NOT NULL)
            END AS has_valid_obj,
            COALESCE(f.obj_kind, 'entity') AS obj_kind
        FROM facts f
        LEFT JOIN entities es ON es.entity_id = f.subj_id
        LEFT JOIN entities eo ON eo.entity_id = f.obj_id
        WHERE es.entity_id IS NULL
           OR (COALESCE(f.obj_kind, 'entity') = 'entity' AND eo.entity_id IS NULL)
           OR (
               COALESCE(f.obj_kind, 'entity') = 'literal'
               AND (f.obj_id IS NOT NULL OR TRIM(COALESCE(f.obj_literal_lex, '')) = '')
           )
           OR TRIM(f.definition) = ''
        ORDER BY f.fact_id ASC
        """
    ):
        fid, sid, oid, fdef, has_s, has_o, okind = (
            str(row[0]),
            str(row[1]),
            str(row[2]) if row[2] is not None else "",
            str(row[3]),
            bool(row[4]),
            bool(row[5]),
            str(row[6]),
        )
        if not has_s or not has_o:
            msg = (
                f"Literal fact {fid} has missing subject ({sid}) or invalid literal payload."
                if okind == "literal"
                else f"Fact {fid} references missing entity endpoint ({sid} -> {oid})."
            )
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="rdf:subject|rdf:object",
                    message=msg,
                )
            )
        if not fdef.strip():
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="skos:definition",
                    message=f"Fact {fid} has an empty statement definition.",
                )
            )

    # 6. EdgeShape/closedWorld: Endpoints exist, no skos:related + owl:differentFrom collision, symmetric semantic edges
    for row in conn.execute(
        """
        SELECT e.src_id, e.dst_id, e.kind
        FROM edges e
        WHERE (
            e.kind = 'passage'
            AND NOT (
                (
                    EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.dst_id)
                )
                OR (
                    EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.dst_id)
                )
            )
        )
        OR (
            e.kind IN ('fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
            AND (
                NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = e.src_id)
                OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = e.dst_id)
            )
        )
        ORDER BY e.src_id ASC, e.dst_id ASC, e.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Edge ({s}, {d}, {k}) has a dangling endpoint outside closed-world tables.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id
        FROM edges a
        INNER JOIN edges b
          ON a.src_id = b.src_id
         AND a.dst_id = b.dst_id
         AND a.kind = 'skos:related'
         AND b.kind = 'owl:differentFrom'
        WHERE a.src_id < a.dst_id
          AND NOT EXISTS (
              SELECT 1 FROM entities ea
              INNER JOIN entities eb ON ea.entity_id = a.src_id AND eb.entity_id = a.dst_id
              WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
          )
          AND NOT EXISTS (
              SELECT 1 FROM entity_labels la
              INNER JOIN entity_labels lb
                ON la.label_norm = lb.label_norm
               AND la.label_kind = 'skos:prefLabel'
               AND lb.label_kind = 'skos:prefLabel'
              WHERE la.entity_id = a.src_id AND lb.entity_id = a.dst_id
          )
        ORDER BY a.src_id ASC, a.dst_id ASC
        """
    ):
        s, d = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}|{d}",
                result_path="skos:related|owl:differentFrom",
                message=f"Node pair ({s}, {d}) has conflicting skos:related and owl:differentFrom edges.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id, a.kind
        FROM edges a
        WHERE a.kind IN ('skos:related', 'owl:differentFrom')
          AND NOT EXISTS (
              SELECT 1 FROM edges b
              WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
          )
        ORDER BY a.src_id ASC, a.dst_id ASC, a.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Asymmetric semantic edge ({s}, {d}, {k}) missing reverse direction.",
            )
        )

    # 7. ConceptShape/broaderCycle & ConceptShape/S27-disjointBroaderRelated via cycle-safe recursive CTE
    for row in conn.execute(
        """
        WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
            SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
            UNION
            SELECT tc.narrower_id, e.dst_id
            FROM broader_tc tc
            JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
            LIMIT 10000
        )
        SELECT DISTINCT narrower_id
        FROM broader_tc
        WHERE narrower_id = broader_id
        ORDER BY narrower_id ASC
        """
    ):
        nid = str(row[0])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/broaderCycle",
                focus_node=nid,
                result_path="skos:broader",
                message=f"Concept {nid} participates in a skos:broader hierarchy cycle.",
            )
        )

    for row in conn.execute(
        """
        WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
            SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
            UNION
            SELECT tc.narrower_id, e.dst_id
            FROM broader_tc tc
            JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
            LIMIT 10000
        )
        SELECT DISTINCT tc.narrower_id, tc.broader_id
        FROM broader_tc tc
        INNER JOIN edges rel
          ON rel.kind = 'skos:related'
         AND (
             (rel.src_id = tc.narrower_id AND rel.dst_id = tc.broader_id)
             OR (rel.src_id = tc.broader_id AND rel.dst_id = tc.narrower_id)
         )
        WHERE tc.narrower_id != tc.broader_id
        ORDER BY tc.narrower_id ASC, tc.broader_id ASC
        """
    ):
        s, d = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/S27-disjointBroaderRelated",
                focus_node=f"{s}|{d}",
                result_path="skos:broader|skos:related",
                message=(
                    f"Concept pair ({s}, {d}) violates SKOS S27 disjointness between "
                    "skos:broaderTransitive and skos:related."
                ),
            )
        )

    return violations


def lint_graph_shacl(
    conn: sqlite3.Connection | YawConnection,
    *,
    auto_heal: bool = False,
) -> list[SHACLViolation]:
    """Run closed-world SQL SHACL validation shapes over yaw.db and optionally apply deterministic SQL repairs."""
    violations = _scan_shacl_violations(conn)
    if not violations or not auto_heal:
        return violations

    with write_tx(conn):
        # Heal 1: ConceptShape/prefLabel
        missing_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el
                WHERE el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            )
            """
        ).fetchall()
        for r in missing_pref:
            eid, ename = str(r[0]), str(r[1])
            alt_row = conn.execute(
                """
                SELECT label_norm, raw_label
                FROM entity_labels
                WHERE entity_id = ?
                ORDER BY CASE label_kind WHEN 'skos:altLabel' THEN 1 ELSE 2 END ASC, label_norm ASC
                LIMIT 1
                """,
                (eid,),
            ).fetchone()
            if alt_row is not None:
                lnorm, rlabel = str(alt_row[0]), str(alt_row[1])
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
                if rlabel:
                    conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, eid))
            else:
                norm = ename.strip().lower() or eid.lower()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                    VALUES (?, ?, 'skos:prefLabel', ?, '__healed__')
                    """,
                    (norm, eid, ename or eid),
                )

        multi_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            INNER JOIN entity_labels el ON el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            GROUP BY e.entity_id
            HAVING COUNT(DISTINCT el.label_norm) > 1
            """
        ).fetchall()
        for r in multi_pref:
            eid, ename = str(r[0]), str(r[1])
            keep_norm = ename.strip().lower()
            norms = [
                str(nr[0])
                for nr in conn.execute(
                    """
                    SELECT DISTINCT label_norm FROM entity_labels
                    WHERE entity_id = ? AND label_kind = 'skos:prefLabel'
                    ORDER BY label_norm ASC
                    """,
                    (eid,),
                )
            ]
            winner = keep_norm if keep_norm in norms else norms[0]
            conn.execute(
                """
                UPDATE entity_labels
                SET label_kind = 'skos:altLabel'
                WHERE entity_id = ? AND label_kind = 'skos:prefLabel' AND label_norm != ?
                """,
                (eid, winner),
            )

        # Heal 2: ConceptShape/S13-disjointLabels
        s13_rows = conn.execute(
            """
            SELECT entity_id, label_norm
            FROM entity_labels
            GROUP BY entity_id, label_norm
            HAVING COUNT(DISTINCT label_kind) > 1
            """
        ).fetchall()
        for r in s13_rows:
            eid, lnorm = str(r[0]), str(r[1])
            kinds = [
                str(kr[0])
                for kr in conn.execute(
                    "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
            ]
            best = max(kinds, key=lambda k: LABEL_KIND_PRIORITY.get(k, 0))
            conn.execute(
                "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                (best, eid, lnorm),
            )

        # Heal 3: ConceptShape/homographDifferentFrom
        homographs = conn.execute(
            """
            SELECT a.entity_id, b.entity_id
            FROM entity_labels a
            INNER JOIN entity_labels b
              ON a.label_norm = b.label_norm
             AND a.label_kind = 'skos:prefLabel'
             AND b.label_kind = 'skos:prefLabel'
             AND a.entity_id < b.entity_id
            WHERE NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = a.entity_id
                  AND e.dst_id = b.entity_id
                  AND e.kind = 'owl:differentFrom'
            )
            GROUP BY a.entity_id, b.entity_id
            """
        ).fetchall()
        if homographs:
            upsert_semantic_edges(
                conn,
                [(str(r[0]), str(r[1]), "owl:differentFrom", 0.0) for r in homographs],
            )

        # Heal 4: ConceptShape/definition
        empty_ent_defs = conn.execute("SELECT entity_id, name FROM entities WHERE TRIM(definition) = ''").fetchall()
        for r in empty_ent_defs:
            eid, ename = str(r[0]), str(r[1])
            fact_row = conn.execute(
                "SELECT raw_triple_json, definition FROM facts WHERE subj_id = ? OR obj_id = ? LIMIT 1",
                (eid, eid),
            ).fetchone()
            if fact_row is not None and str(fact_row[1]).strip():
                synth_def = f"{ename}: {str(fact_row[1]).strip()}"
            elif fact_row is not None:
                try:
                    t = json.loads(fact_row[0])
                    synth_def = f"{ename} participates in relation: {t[0]} {t[1]} {t[2]}."
                except Exception:
                    synth_def = f"Concept representing {ename}."
            else:
                synth_def = f"Concept representing {ename}."
            conn.execute(
                "UPDATE entities SET definition = ? WHERE entity_id = ?",
                (synth_def, eid),
            )

        # Heal 5: StatementShape/closedWorld
        dangling_fids = [
            str(r[0])
            for r in conn.execute(
                """
                SELECT f.fact_id
                FROM facts f
                LEFT JOIN entities es ON es.entity_id = f.subj_id
                LEFT JOIN entities eo ON eo.entity_id = f.obj_id
                WHERE es.entity_id IS NULL
                   OR (COALESCE(f.obj_kind, 'entity') = 'entity' AND eo.entity_id IS NULL)
                   OR (
                       COALESCE(f.obj_kind, 'entity') = 'literal'
                       AND (f.obj_id IS NOT NULL OR TRIM(COALESCE(f.obj_literal_lex, '')) = '')
                   )
                """
            )
        ]
        if dangling_fids:
            for batch in batched(dangling_fids):
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM fact_sources WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_vec WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_fts WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM facts WHERE fact_id IN ({placeholders})", tuple(batch))

        empty_fact_defs = conn.execute(
            "SELECT fact_id, pred, raw_triple_json FROM facts WHERE TRIM(definition) = ''"
        ).fetchall()
        for r in empty_fact_defs:
            fid, pred, raw_json = str(r[0]), str(r[1]), str(r[2])
            try:
                t = json.loads(raw_json)
                synth_fdef = f"{t[0]} {t[1]} {t[2]}."
            except Exception:
                synth_fdef = f"Statement with predicate {pred}."
            conn.execute(
                "UPDATE facts SET definition = ? WHERE fact_id = ?",
                (synth_fdef, fid),
            )

        # Heal 6: EdgeShape/closedWorld
        conn.execute(
            """
            DELETE FROM edges
            WHERE (
                kind = 'passage'
                AND NOT (
                    (
                        EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.src_id)
                        AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.dst_id)
                    )
                    OR (
                        EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.src_id)
                        AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.dst_id)
                    )
                )
            )
            OR (
                kind IN ('fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
                AND (
                    NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = edges.src_id)
                    OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = edges.dst_id)
                )
            )
            """
        )
        conn.execute(
            """
            DELETE FROM edges
            WHERE kind = 'skos:related'
              AND EXISTS (
                  SELECT 1 FROM edges b
                  WHERE b.src_id = edges.src_id
                    AND b.dst_id = edges.dst_id
                    AND b.kind = 'owl:differentFrom'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM entities ea
                  INNER JOIN entities eb ON ea.entity_id = edges.src_id AND eb.entity_id = edges.dst_id
                  WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
              )
              AND NOT EXISTS (
                  SELECT 1 FROM entity_labels la
                  INNER JOIN entity_labels lb
                    ON la.label_norm = lb.label_norm
                   AND la.label_kind = 'skos:prefLabel'
                   AND lb.label_kind = 'skos:prefLabel'
                  WHERE la.entity_id = edges.src_id AND lb.entity_id = edges.dst_id
              )
            """
        )
        asym_rows = conn.execute(
            """
            SELECT a.src_id, a.dst_id, a.kind, a.weight
            FROM edges a
            WHERE a.kind IN ('skos:related', 'owl:differentFrom')
              AND NOT EXISTS (
                  SELECT 1 FROM edges b
                  WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
              )
            """
        ).fetchall()
        for r in asym_rows:
            conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, ?, ?)",
                (str(r[1]), str(r[0]), str(r[2]), float(r[3])),
            )

        # Heal 7: ConceptShape/broaderCycle & ConceptShape/S27-disjointBroaderRelated
        conn.execute("DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = dst_id")
        for _ in range(16):
            cycle_nodes = [
                str(r[0])
                for r in conn.execute(
                    """
                    WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
                        SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
                        UNION
                        SELECT tc.narrower_id, e.dst_id
                        FROM broader_tc tc
                        JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
                        LIMIT 10000
                    )
                    SELECT DISTINCT narrower_id
                    FROM broader_tc
                    WHERE narrower_id = broader_id
                    ORDER BY narrower_id ASC
                    """
                ).fetchall()
            ]
            if not cycle_nodes:
                break
            placeholders = ",".join("?" for _ in cycle_nodes)
            back_edge = conn.execute(
                f"""
                SELECT src_id, dst_id
                FROM edges
                WHERE kind = 'skos:broader'
                  AND src_id IN ({placeholders})
                  AND dst_id IN ({placeholders})
                ORDER BY (src_id > dst_id) DESC, src_id DESC, dst_id ASC
                LIMIT 1
                """,
                tuple(cycle_nodes) + tuple(cycle_nodes),
            ).fetchone()
            if back_edge is None:
                break
            conn.execute(
                "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                (str(back_edge[0]), str(back_edge[1])),
            )

        s27_pairs = conn.execute(
            """
            WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
                SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
                UNION
                SELECT tc.narrower_id, e.dst_id
                FROM broader_tc tc
                JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
                LIMIT 10000
            )
            SELECT DISTINCT tc.narrower_id, tc.broader_id
            FROM broader_tc tc
            INNER JOIN edges rel
              ON rel.kind = 'skos:related'
             AND (
                 (rel.src_id = tc.narrower_id AND rel.dst_id = tc.broader_id)
                 OR (rel.src_id = tc.broader_id AND rel.dst_id = tc.narrower_id)
             )
            WHERE tc.narrower_id != tc.broader_id
            """
        ).fetchall()
        for r in s27_pairs:
            nid, bid = str(r[0]), str(r[1])
            conn.execute(
                """
                DELETE FROM edges
                WHERE kind = 'skos:related'
                  AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                """,
                (nid, bid, bid, nid),
            )

    return _scan_shacl_violations(conn)


def apply_shacl_repairs(
    conn: sqlite3.Connection | YawConnection,
    repairs: Sequence[Mapping[str, Any]],
    *,
    merge_entities_fn: Callable[[str, str, str], bool] | None = None,
) -> int:
    """Execute validated closed-world SHACL repair actions inside an atomic SQLite write transaction."""
    if not repairs:
        return 0
    applied = 0
    with write_tx(conn):
        for rep in repairs:
            focus = str(rep.get("focus_node", "")).strip()
            action = str(rep.get("action", "")).strip()
            val = str(rep.get("value", "")).strip()
            target = str(rep.get("target_node", "")).strip() or val
            if not focus or not action:
                continue
            if action in ("set_definition", "rewrite_definition") and val:
                if focus.startswith("entity-"):
                    cur = conn.execute(
                        "UPDATE entities SET definition = ? WHERE entity_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                elif focus.startswith("fact-"):
                    cur = conn.execute(
                        "UPDATE facts SET definition = ? WHERE fact_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "set_pref_label" and val and focus.startswith("entity-"):
                cur = conn.execute(
                    "UPDATE entities SET name = ? WHERE entity_id = ?",
                    (val, focus),
                )
                if cur.rowcount > 0:
                    upsert_entity_labels(conn, [(val.strip().lower(), focus, "skos:prefLabel", val, "__healed__")])
                    applied += 1
            elif action in ("demote_to_alt_label", "demote_label") and focus.startswith("entity-"):
                rows = conn.execute(
                    "SELECT label_norm FROM entity_labels WHERE entity_id = ? AND label_kind = "
                    "'skos:prefLabel' ORDER BY label_norm ASC",
                    (focus,),
                ).fetchall()
                if len(rows) > 1:
                    for extra in rows[1:]:
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = 'skos:altLabel' WHERE entity_id = ? AND label_norm "
                            "= ? AND label_kind = 'skos:prefLabel'",
                            (focus, str(extra[0])),
                        )
                    applied += 1
                else:
                    s13_rows = conn.execute(
                        """
                        SELECT label_norm
                        FROM entity_labels
                        WHERE entity_id = ?
                        GROUP BY label_norm
                        HAVING COUNT(DISTINCT label_kind) > 1
                        """,
                        (focus,),
                    ).fetchall()
                    for r in s13_rows:
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = 'skos:altLabel' "
                            "WHERE entity_id = ? AND label_norm = ?",
                            (focus, str(r[0])),
                        )
                        applied += 1
            elif action in ("merge_altLabel", "merge_hiddenLabel") and (target or "|" in focus):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                    if target == ea:
                        src_node, dst_node = eb, ea
                    else:
                        src_node, dst_node = ea, eb
                else:
                    src_node, dst_node = focus, target
                lkind = "skos:hiddenLabel" if action == "merge_hiddenLabel" else "skos:altLabel"
                if src_node and dst_node and src_node != dst_node:
                    if merge_entities_fn is not None:
                        if merge_entities_fn(src_node, dst_node, lkind):
                            applied += 1
                    else:
                        existing_pair = get_existing_ids(conn, "entity", [src_node, dst_node])
                        if src_node in existing_pair and dst_node in existing_pair:
                            merge_entities(conn, src_node, dst_node, label_kind=lkind)
                            applied += 1
            elif action in ("add_different_from", "assert_differentFrom"):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "owl:differentFrom", 0.0)])
                        applied += 1
            elif action == "assert_related":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "skos:related", 0.5)])
                        applied += 1
            elif action == "assert_broader":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "skos:broader", 0.5)])
                        applied += 1
            elif action == "remove_hierarchy_cycle":
                if "->" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("->", 1))
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                elif "|" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("|", 1))
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                else:
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ?",
                        (focus,),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "remove_conflicting_related":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb:
                    cur = conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:related'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (ea, eb, eb, ea),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action in ("delete_dangling", "delete_orphan", "delete_invalid_fact"):
                if "->" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("->", 1))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    applied += 1
                else:
                    affected_fids = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT fact_id FROM facts WHERE fact_id = ? OR subj_id = ? OR obj_id = ?",
                            (focus, focus, focus),
                        ).fetchall()
                    ]
                    for fid in affected_fids:
                        conn.execute("DELETE FROM fact_sources WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (focus,))
                    applied += 1
    return applied


def _escape_turtle_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
    return f'"{escaped}"'


def serialize_graph_turtle(
    conn: sqlite3.Connection | YawConnection,
    violations: Sequence[SHACLViolation] | None = None,
) -> str:
    """Serialize the yaw.db SKOS/RDF/OWL closed-world knowledge graph into W3C Turtle format."""
    if violations is None:
        violations = _scan_shacl_violations(conn)

    lines: list[str] = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix yaw: <https://yaw.local/vocab#> .",
        "",
    ]

    labels_by_eid: dict[str, list[tuple[str, str, str]]] = {}
    for row in conn.execute(
        """
        SELECT entity_id, label_kind, raw_label, label_norm, chunk_id
        FROM entity_labels
        ORDER BY entity_id ASC, label_kind ASC, label_norm ASC
        """
    ):
        eid = str(row[0])
        lkind = str(row[1])
        lbl = str(row[2]) if str(row[2]) else str(row[3])
        cid = str(row[4])
        labels_by_eid.setdefault(eid, []).append((lkind, lbl, cid))

    sem_edges_by_eid: dict[str, list[tuple[str, str]]] = {}
    for row in conn.execute(
        """
        SELECT src_id, kind, dst_id
        FROM edges
        WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
        ORDER BY src_id ASC, kind ASC, dst_id ASC
        """
    ):
        sem_edges_by_eid.setdefault(str(row[0]), []).append((str(row[1]), str(row[2])))

    for row in conn.execute("SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"):
        eid, ename, edef = str(row[0]), str(row[1]), str(row[2])
        pred_objs: list[str] = ["a skos:Concept"]
        seen_labels: set[tuple[str, str]] = set()
        prov_chunks: set[str] = set()
        for lkind, lbl, cid in labels_by_eid.get(eid, []):
            if (lkind, lbl) not in seen_labels:
                seen_labels.add((lkind, lbl))
                pred_objs.append(f"{lkind} {_escape_turtle_literal(lbl)}")
            if cid and not cid.startswith("__"):
                prov_chunks.add(cid)
        if not seen_labels:
            pred_objs.append(f"skos:prefLabel {_escape_turtle_literal(ename)}")
        if edef.strip():
            pred_objs.append(f"skos:definition {_escape_turtle_literal(edef.strip())}")
        for kind, dst_id in sem_edges_by_eid.get(eid, []):
            pred_objs.append(f"{kind} yaw:{dst_id}")
        for cid in sorted(prov_chunks):
            pred_objs.append(f"prov:wasDerivedFrom yaw:{cid}")

        lines.append(f"yaw:{eid} " + " ;\n    ".join(pred_objs) + " .")
        lines.append("")

    for row in conn.execute(
        """
        SELECT
            fact_id,
            subj_id,
            pred,
            obj_id,
            definition,
            COALESCE(obj_kind, 'entity'),
            obj_literal_lex,
            COALESCE(obj_datatype, ''),
            COALESCE(obj_lang, '')
        FROM facts
        ORDER BY fact_id ASC
        """
    ):
        fid, sid, pred, oid, fdef, okind, olex, odt, olang = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]) if row[3] is not None else "",
            str(row[4]),
            str(row[5]),
            str(row[6]) if row[6] is not None else "",
            str(row[7]),
            str(row[8]),
        )
        if okind == "literal":
            lit_token = _escape_turtle_literal(olex)
            if olang:
                lit_token = f"{lit_token}@{olang}"
            elif odt:
                lit_token = f"{lit_token}^^<{odt}>"
            rdf_obj = lit_token
        else:
            rdf_obj = f"yaw:{oid}"
        stmt_parts = [
            "a rdf:Statement",
            f"rdf:subject yaw:{sid}",
            f"rdf:predicate {_escape_turtle_literal(pred)}",
            f"rdf:object {rdf_obj}",
        ]
        if fdef.strip():
            stmt_parts.append(f"skos:definition {_escape_turtle_literal(fdef.strip())}")
        lines.append(f"yaw:{fid} " + " ;\n    ".join(stmt_parts) + " .")
        lines.append("")

    conforms_str = "true" if len(violations) == 0 else "false"
    report_parts = [
        "a sh:ValidationReport",
        f"sh:conforms {conforms_str}",
    ]
    for v in violations:
        report_parts.append(
            "sh:result [ "
            "a sh:ValidationResult ; "
            f"sh:sourceShape {_escape_turtle_literal(v.shape)} ; "
            f"sh:focusNode {_escape_turtle_literal(v.focus_node)} ; "
            f"sh:resultPath {_escape_turtle_literal(v.result_path)} ; "
            f"sh:resultMessage {_escape_turtle_literal(v.message)} "
            "]"
        )
    lines.append("yaw:ValidationReport " + " ;\n    ".join(report_parts) + " .")
    lines.append("")

    return "\n".join(lines).strip() + "\n"
