import json
import sys
from collections import defaultdict
from pathlib import Path

from src.config import PARAMS

OVERLAP_THRESHOLD = PARAMS['ner']['overlap']


def span_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    min_len = min(a_end - a_start, b_end - b_start)
    return inter / min_len if min_len > 0 else 0.0


def classify(predictions: list[dict], gold_start: int, gold_end: int, gold_label: str) -> tuple[str, float]:
    '''-> (outcome, score) where outcome is correct / label_error / miss'''
    best, best_ovlp = None, 0.0
    for pred in predictions:
        ovlp = span_overlap(pred['start'], pred['end'], gold_start, gold_end)
        if ovlp >= OVERLAP_THRESHOLD and ovlp > best_ovlp:
            best, best_ovlp = pred, ovlp
    if best is None:
        return 'miss', 0.0
    if best['label'] == gold_label:
        return 'correct', best.get('score', 1.0)
    return 'label_error', best.get('score', 0.0)


def load_jsonl(path: str) -> dict[str, dict]:
    data = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            data[r['document_id']] = r
    return data


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'

    projected = load_jsonl(f'data/annotations/projected_{split}.jsonl')
    gt_preds_by_doc = load_jsonl(f'data/annotations/ner_gt_{split}.jsonl')
    ocr_preds_by_doc = load_jsonl(f'data/annotations/ner_ocr_{split}.jsonl')

    out_path = Path(f'data/results/outcomes_{split}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    type_total = defaultdict(int)
    type_eval = defaultdict(int)
    type_correct = defaultdict(int)
    type_miss = defaultdict(int)
    type_label_error = defaultdict(int)

    with open(out_path, 'w') as f:
        for doc_id, proj in projected.items():
            gt_preds = gt_preds_by_doc.get(doc_id, {}).get('entities', [])
            ocr_preds = ocr_preds_by_doc.get(doc_id, {}).get('entities', [])

            for ent in proj['entities']:
                if ent['status'] != 'projected':
                    continue

                lbl = ent['label']
                type_total[lbl] += 1

                gt_outcome, gt_score = classify(gt_preds, ent['gt_start'], ent['gt_end'], lbl)
                in_eval = gt_outcome == 'correct'

                row = {
                    'entity_id': ent['entity_id'],
                    'document_id': doc_id,
                    'dataset': proj['dataset'],
                    'language': proj['language'],
                    'label': lbl,
                    'gt_text': ent['gt_text'],
                    'gt_start': ent['gt_start'],
                    'gt_end': ent['gt_end'],
                    'ocr_text': ent['ocr_text'],
                    'ocr_start': ent['ocr_start'],
                    'ocr_end': ent['ocr_end'],
                    'in_eval': in_eval,
                    'gt_score': round(gt_score, 4),
                    'outcome': None,
                    'ocr_score': None,
                }

                if in_eval:
                    type_eval[lbl] += 1
                    if ent['ocr_start'] is None:
                        outcome, ocr_score = 'miss', 0.0
                    else:
                        outcome, ocr_score = classify(ocr_preds, ent['ocr_start'], ent['ocr_end'], lbl)
                    row['outcome'] = outcome
                    row['ocr_score'] = round(ocr_score, 4)
                    if outcome == 'correct':
                        type_correct[lbl] += 1
                    elif outcome == 'miss':
                        type_miss[lbl] += 1
                    else:
                        type_label_error[lbl] += 1

                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    LABELS = ['pers', 'loc', 'org', 'prod', 'time']
    total_eval = sum(type_eval.values())
    total_correct = sum(type_correct.values())
    total_miss = sum(type_miss.values())
    total_lbl = sum(type_label_error.values())

    print(f'\n=== NER Outcome Metrics — {split} (E^eval only) ===')
    if total_eval:
        print(f'\nOverall n={total_eval}')
        print(f'NEF {round(total_correct/total_eval, 3)} (correct)')
        print(f'miss_rate {round(total_miss/total_eval, 3)} (parallel to LLR)')
        print(f'label_error_rate {round(total_lbl/total_eval, 3)} (parallel to LSR)')

    print(f"\n{'type'} {'eval'} {'NEF'} {'miss'} {'lbl_err'} {'proj_total'}")
    for lbl in LABELS:
        n = type_eval[lbl]
        if n == 0:
            continue
        print(
            f'{lbl} {n} '
            f'{round(type_correct[lbl]/n, 3)} '
            f'{round(type_miss[lbl]/n, 3)} '
            f'{round(type_label_error[lbl]/n, 3)} '
            f'{type_total[lbl]}'
        )

    print(f'\nSaved to {out_path}')
