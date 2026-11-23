import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, line_window
from src.spans import levenshtein

SPLIT = 'test' if '--test' in sys.argv else 'dev'
H = PARAMS['local_damage']['h']
EPS = 1e-9


def parse_line(gt_line: str, ocr_line: str):
    gw = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', gt_line)]
    ow = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', ocr_line)]
    word_spans = [(s, e, 0.0) for _, s, e in gw]
    span_blocks = []
    insertions = []
    sm = SequenceMatcher(None, [w for w, _, _ in gw], [w for w, _, _ in ow], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        gt_t = ' '.join(gw[k][0] for k in range(i1, i2)) if i2 > i1 else ''
        oc_t = ' '.join(ow[k][0] for k in range(j1, j2)) if j2 > j1 else ''
        C = levenshtein(gt_t, oc_t) / max(len(gt_t), len(oc_t), 1)
        if i2 > i1:
            for idx in range(i1, i2):
                s, e, _ = word_spans[idx]
                word_spans[idx] = (s, e, C)
            total_len = max(len(gt_t), len(oc_t), 1)  # ell_s, so c_s * ell_s = lev
            span_blocks.append((i1, i2, C, total_len))
        else:
            l_b = sum(len(ow[k][0]) for k in range(j1, j2))
            if l_b > 0:
                insertions.append((i1 - 0.5, l_b))
    return word_spans, span_blocks, insertions



def entity_tok_positions(word_spans: list, ent_s: int, ent_e: int) -> list[int]:
    return [i for i, (ws, we, _) in enumerate(word_spans) if ws < ent_e and we > ent_s]


def damage_at(word_spans: list, p: float, h: int, insertions: list | None = None) -> float:
    n = len(word_spans)
    widths = [we - ws for ws, we, _ in word_spans]
    c_vals = [c for _, _, c in word_spans]
    num = sum(max(0.0, h + 1 - abs(i - p)) * c_vals[i] * widths[i] for i in range(n))
    den = sum(max(0.0, h + 1 - abs(i - p)) * widths[i] for i in range(n)) + EPS
    if insertions:
        for q_b, l_b in insertions:
            k = max(0.0, h + 1 - abs(p - q_b))
            num += k * l_b  # c_b = 1.0
            den += k * l_b
    return num / den


def local_damage_max(word_spans: list, tok_pos: list[int], h: int, insertions: list | None = None) -> float:
    if not tok_pos:
        return damage_at(word_spans, len(word_spans) // 2, h, insertions)
    return max(damage_at(word_spans, p, h, insertions) for p in tok_pos)


def local_damage_avg(word_spans: list, tok_pos: list[int], h: int, insertions: list | None = None) -> float:
    widths = [we - ws for ws, we, _ in word_spans]
    if not tok_pos:
        return damage_at(word_spans, len(word_spans) // 2, h, insertions)
    num = sum(damage_at(word_spans, p, h, insertions) * widths[p] for p in tok_pos)
    den = sum(widths[p] for p in tok_pos) + EPS
    return num / den


def local_damage(word_spans: list, span_blocks: list, insertions: list, query_pos: list[int | float], h: int) -> float:
    if not query_pos:
        query_pos = [len(word_spans) // 2]
    widths = [we - ws for ws, we, _ in word_spans]
    n = len(word_spans)

    error_indices = set()
    for i1, i2, _, _ in span_blocks:
        error_indices.update(range(i1, i2))

    num = 0.0
    den = 0.0

    for i1, i2, C, total_len in span_blocks:
        d = min(max(0.0, i1 - p, p - (i2 - 1)) for p in query_pos)
        k = max(0.0, h + 1 - d)
        num += k * C * total_len
        den += k * total_len

    for i in range(n):
        if i not in error_indices:
            d = min(abs(p - i) for p in query_pos)
            k = max(0.0, h + 1 - d)
            den += k * widths[i]

    for q_b, l_b in insertions:
        d = min(abs(p - q_b) for p in query_pos)
        k = max(0.0, h + 1 - d)
        num += k * l_b
        den += k * l_b

    return num / (den + EPS)


def span_repair_scores(gt_win: str, ocr_win: str, spans: list[dict],
                       ent_s: int, ent_e: int, h: int) -> list[float]:
    '''per-span repair priority: k_h(p_ent, p_s) * c_s * ell_s, anchored at the entity'''
    gw = [(m.start(), m.end()) for m in re.finditer(r'\S+', gt_win)]
    ent_pos = [i for i, (ws, we) in enumerate(gw) if ws < ent_e and we > ent_s]
    if not ent_pos:
        ent_pos = [len(gw) // 2] if gw else [0]
    scores = []
    for sp in spans:
        gs, ge = sp['gt_start'], sp['gt_end']
        pos = [i for i, (ws, we) in enumerate(gw) if ws < ge and we > gs]
        if not pos:
            after = [i for i, (ws, we) in enumerate(gw) if ws >= gs]
            pos = after[:1] if after else [max(0, len(gw) - 1)]
        d = min(abs(pe - ps) for pe in ent_pos for ps in pos)
        k = max(0.0, h + 1 - d)
        ell = max(len(sp.get('gt_text', '')), len(sp.get('ocr_text', '')), 1)
        scores.append(k * sp.get('D', 0.0) * ell)
    return scores


def span_weighted_cer(word_spans: list) -> float:
    widths = [we - ws for ws, we, _ in word_spans]
    c_vals = [c for _, _, c in word_spans]
    return sum(c_vals[i] * widths[i] for i in range(len(word_spans))) / (sum(widths) + EPS)


def span_error_cer(spans: list) -> float:
    # error-span cer, edit distance over total span length, the unweighted control for attribution damage
    if not spans:
        return 0.0
    ell = [max(len(s.get('gt_text', '')), len(s.get('ocr_text', '')), 1) for s in spans]
    return sum(s.get('D', 0.0) * l for s, l in zip(spans, ell)) / sum(ell)


def text_error_span_cer(gt_text: str, ocr_text: str) -> float:
    # same thing straight from text, caller picks the scope
    gw = [m.group() for m in re.finditer(r'\S+', gt_text)]
    ow = [m.group() for m in re.finditer(r'\S+', ocr_text)]
    lev = ell = 0
    for op, i1, i2, j1, j2 in SequenceMatcher(None, gw, ow, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        g = ' '.join(gw[i1:i2])
        o = ' '.join(ow[j1:j2])
        lev += levenshtein(g, o)
        ell += max(len(g), len(o), 1)
    return lev / ell if ell else 0.0


if __name__ == '__main__':
    # every entity keyed by id, in_eval marks the clean ceiling
    all_ner = {}
    ner_outcomes = {}
    with open(f'data/results/outcomes_{SPLIT}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            all_ner[r['entity_id']] = r
            if r['in_eval']:
                ner_outcomes[r['entity_id']] = r

    el_outcomes = {}
    el_path = Path(f'data/results/outcomes_el_{SPLIT}.jsonl')
    if el_path.exists():
        with open(el_path) as f:
            for line in f:
                r = json.loads(line)
                if r['in_eval']:
                    el_outcomes[r['entity_id']] = r

    # union of both clean-ceiling sets, each task gets some entities right that the other misses
    wanted = set(ner_outcomes) | set(el_outcomes)
    by_doc = {}
    for eid in wanted:
        ent = all_ner.get(eid)
        if ent is None:
            continue
        by_doc.setdefault(ent['document_id'], []).append((eid, ent))

    records = {
        rec['document_metadata']['document_id']: rec
        for rec in load(SPLIT)
    }

    out_path = Path(f'data/results/iwld_{SPLIT}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    with open(out_path, 'a') as fout:
        for doc_id, ents in tqdm(by_doc.items(), desc='poscer'):
            rec = records.get(doc_id)
            if not rec:
                continue

            gt_text = rec['ground_truth']['transcription_unit']
            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            record_cer = jiwer.cer(gt_text, ocr_text) if gt_text else 0.0
            esc = text_error_span_cer(gt_text, ocr_text) if gt_text else 0.0

            line_cache = {}

            for eid, ent in ents:
                if eid in done:
                    continue

                gt_s, gt_e = ent['gt_start'], ent['gt_end']

                # same window as the attribution, so both are scoped alike
                gt_ls, gt_le, ocr_ls, ocr_le = line_window(rec, gt_s, gt_e)
                key = (gt_ls, gt_le)

                if key not in line_cache:
                    spans, span_blks, inserts = parse_line(gt_text[gt_ls:gt_le], ocr_text[ocr_ls:ocr_le])
                    line_cache[key] = (spans, span_blks, inserts) if spans else None

                cached = line_cache[key]
                if cached is None:
                    continue
                spans, span_blks, inserts = cached

                tok_pos = entity_tok_positions(spans, gt_s - gt_ls, gt_e - gt_ls)

                row = {
                    'entity_id': eid,
                    'document_id': doc_id,
                    'ner_failed': int(ent['outcome'] != 'correct'),
                    'el_failed': int(el_outcomes.get(eid, {}).get('outcome', 'correct') != 'correct'),
                    'ner_in_eval': int(eid in ner_outcomes),
                    'el_in_eval': int(eid in el_outcomes),
                    'record_cer': round(record_cer, 6),
                    'record_error_span_cer': round(esc, 6),
                    'window_cer': round(span_weighted_cer(spans), 6),
                    'local_damage_max': round(local_damage_max(spans, tok_pos, H, inserts), 6),
                    'local_damage_avg': round(local_damage_avg(spans, tok_pos, H, inserts), 6),
                    'local_damage': round(local_damage(spans, span_blks, inserts, tok_pos, H), 6),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                fout.flush()

    rows = [json.loads(l) for l in open(out_path)]

    attr_by_id = {}
    attr_path = Path(f'data/results/attribution_{SPLIT}.jsonl')
    if attr_path.exists():
        with open(attr_path) as f:
            for line in f:
                r = json.loads(line)
                attr_by_id[r['entity_id']] = r

    el_attr_by_id = {}
    el_attr_path = Path(f'data/results/attribution_el_{SPLIT}.jsonl')
    if el_attr_path.exists():
        with open(el_attr_path) as f:
            for line in f:
                r = json.loads(line)
                el_attr_by_id[r['entity_id']] = r

    for row in rows:
        attr = attr_by_id.get(row['entity_id'], {})
        for key in ('mention_cer', 'context_cer', 'corruption_only',
                    'importance_only', 'edit_lime_damage', 'edit_lime_damage_norm'):
            row[key] = attr.get(key, float('nan'))
        # unweighted control over the same spans as attribution damage
        row['error_span_cer'] = span_error_cer(attr.get('edit_spans', []))
        el_attr = el_attr_by_id.get(row['entity_id'], {})
        row['el_edit_lime_damage'] = el_attr.get('edit_lime_damage', float('nan'))
        row['el_mention_cer'] = el_attr.get('mention_cer', float('nan'))
        row['el_error_span_cer'] = span_error_cer(el_attr.get('edit_spans', []))

    print(f'\nSaved -> {out_path}')
