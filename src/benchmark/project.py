import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from genalog.text.ner_label import propagate_label_to_ocr
from tqdm import tqdm

from src.config import PARAMS
from src.data import load

SIMILARITY_THRESHOLD = PARAMS['annotation']['projection_similarity']


def project(
    gt_text: str,
    ocr_text: str,
    entities: list[dict],
    gt_line_offsets: list,
    ocr_line_offsets: list,
) -> list[dict]:
    if not entities:
        return []
    if (
        not gt_line_offsets
        or not ocr_line_offsets
        or len(gt_line_offsets) != len(ocr_line_offsets)
    ):
        return project_difflib(gt_text, ocr_text, entities)

    results = {}
    projected = set()

    for (gt_s, gt_e), (ocr_s, ocr_e) in zip(gt_line_offsets, ocr_line_offsets):
        line_ents = [
            e for e in entities
            if e['entity_id'] not in projected
            and e['start'] >= gt_s
            and e['end'] <= gt_e
        ]
        if not line_ents:
            continue
        gt_line = gt_text[gt_s:gt_e]
        ocr_line = ocr_text[ocr_s:ocr_e]
        for ent, proj in zip(
            line_ents,
            project_line(gt_line, ocr_line, line_ents, gt_s, ocr_s, gt_text, ocr_text),
        ):
            results[ent['entity_id']] = proj
            projected.add(ent['entity_id'])

    # entity crosses a line break, align the whole text
    for e in entities:
        if e['entity_id'] not in projected:
            results[e['entity_id']] = project_difflib(gt_text, ocr_text, [e])[0]

    return [results[e['entity_id']] for e in entities]


def project_line(
    gt_line: str,
    ocr_line: str,
    line_ents: list[dict],
    gt_s: int,
    ocr_s: int,
    gt_text: str,
    ocr_text: str,
) -> list[dict]:
    gt_words = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', gt_line)]
    ocr_words = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', ocr_line)]

    if not gt_words or not ocr_words:
        return [make_missing(e) for e in line_ents]

    # bio labels with entity ids, so entities on one line stay apart
    gt_labels = ['O'] * len(gt_words)
    for idx, ent in enumerate(line_ents):
        ent_s, ent_e = ent['start'] - gt_s, ent['end'] - gt_s
        started = False
        for i, (_, ws, we) in enumerate(gt_words):
            if ws < ent_e and we > ent_s:
                gt_labels[i] = f"{'B' if not started else 'I'}-E{idx}"
                started = True

    try:
        ocr_labels, _, _, _ = propagate_label_to_ocr(
            gt_labels,
            [w for w, _, _ in gt_words],
            [w for w, _, _ in ocr_words],
        )
    except Exception:
        # token counts disagree, fall back to char-level difflib
        return project_difflib(gt_text, ocr_text, line_ents)

    # char spans in the full ocr document from the bio labels
    spans = {}
    for i, label in enumerate(ocr_labels):
        if label == 'O' or i >= len(ocr_words):
            continue
        parts = label.split('-', 1)
        if len(parts) != 2 or not parts[1].startswith('E'):
            continue
        try:
            eidx = int(parts[1][1:])
        except ValueError:
            continue
        _, ws, we = ocr_words[i]
        if eidx not in spans:
            spans[eidx] = [ws + ocr_s, we + ocr_s]
        else:
            spans[eidx][1] = we + ocr_s

    results = []
    for idx, ent in enumerate(line_ents):
        if idx not in spans:
            results.append(make_missing(ent))
            continue
        ocr_start, ocr_end = spans[idx]
        ocr_span = ocr_text[ocr_start:ocr_end]
        sim = SequenceMatcher(None, ent['text'], ocr_span).ratio()
        results.append(make_result(ent, ocr_span, ocr_start, ocr_end, sim))
    return results


def project_difflib(gt_text: str, ocr_text: str, entities: list[dict]) -> list[dict]:
    matcher = SequenceMatcher(None, gt_text, ocr_text, autojunk=False)
    gt_to_ocr = {}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                gt_to_ocr[i1 + k] = j1 + k
        elif op == 'replace':
            gt_len, ocr_len = i2 - i1, j2 - j1
            for k in range(gt_len):
                gt_to_ocr[i1 + k] = j1 + round(k * ocr_len / gt_len)
        elif op == 'delete':
            for k in range(i2 - i1):
                gt_to_ocr[i1 + k] = j1

    results = []
    for ent in entities:
        ocr_start = gt_to_ocr.get(ent['start'])
        ocr_end = gt_to_ocr.get(ent['end'] - 1)
        if ocr_start is None or ocr_end is None or ocr_start >= ocr_end + 1:
            results.append(make_missing(ent))
            continue
        ocr_end += 1
        ocr_span = ocr_text[ocr_start:ocr_end]
        sim = SequenceMatcher(None, ent['text'], ocr_span).ratio()
        results.append(make_result(ent, ocr_span, ocr_start, ocr_end, sim))
    return results


def make_result(ent: dict, ocr_span: str, ocr_start: int, ocr_end: int, sim: float) -> dict:
    return {
        'entity_id': ent['entity_id'],
        'label': ent['label'],
        'gt_text': ent['text'],
        'gt_start': ent['start'],
        'gt_end': ent['end'],
        'ocr_text': ocr_span,
        'ocr_start': ocr_start,
        'ocr_end': ocr_end,
        'status': 'projected' if sim >= SIMILARITY_THRESHOLD else 'uncertain',
    }


def make_missing(ent: dict) -> dict:
    return {
        'entity_id': ent['entity_id'],
        'label': ent['label'],
        'gt_text': ent['text'],
        'gt_start': ent['start'],
        'gt_end': ent['end'],
        'ocr_text': None,
        'ocr_start': None,
        'ocr_end': None,
        'status': 'missing',
    }


if __name__ == '__main__':
    import sys
    split = 'test' if '--test' in sys.argv else 'dev'

    consensus = {}
    with open(f'data/annotations/consensus_{split}.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            consensus[rec['document_id']] = rec['entities']

    out_path = Path(f'data/annotations/projected_{split}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['document_id'])

    records = [
        rec for rec in load(split)
        if rec['document_metadata']['document_id'] in consensus
        and rec['document_metadata']['document_id'] not in done
    ]

    with open(out_path, 'a') as f:
        for rec in tqdm(records, desc='projecting'):
            meta = rec['document_metadata']
            doc_id = meta['document_id']
            row = {
                'document_id': doc_id,
                'dataset': meta['primary_dataset_name'],
                'language': meta['language'],
                'date': meta['date'],
                'entities': project(
                    rec['ground_truth']['transcription_unit'],
                    rec['ocr_hypothesis']['transcription_unit'],
                    consensus[doc_id],
                    rec['ground_truth']['line_offsets'],
                    rec['ocr_hypothesis']['line_offsets'],
                ),
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()

    print(f'Saved to {out_path}')
    counts = defaultdict(int)
    with open(out_path) as f:
        for line in f:
            for ent in json.loads(line)['entities']:
                counts[ent['status']] += 1
    for status, n in sorted(counts.items()):
        print(f'{status}: {n}')
