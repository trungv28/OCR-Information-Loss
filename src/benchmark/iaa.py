import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.config import PARAMS

OVERLAP_THRESHOLD = PARAMS['annotation']['overlap']
LABELS = ['pers', 'loc', 'org', 'prod', 'time']


def span_overlap(a_start, a_end, b_start, b_end) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    min_len = min(a_end - a_start, b_end - b_start)
    return inter / min_len if min_len > 0 else 0.0


def load_annotations(path: str) -> dict[str, list[dict]]:
    data = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            data[r['document_id']] = r['entities']
    return data


def pairwise_f1(gold: list[dict], pred: list[dict]) -> dict[str, dict]:
    '''span f1 per label between two annotation sets of one document'''
    matched_gold = set()
    matched_pred = set()

    for j, p in enumerate(pred):
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if g['label'] == p['label'] and span_overlap(g['start'], g['end'], p['start'], p['end']) >= OVERLAP_THRESHOLD:
                matched_gold.add(i)
                matched_pred.add(j)
                break

    by_label = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for j, p in enumerate(pred):
        lbl = p['label']
        if j in matched_pred:
            by_label[lbl]['tp'] += 1
        else:
            by_label[lbl]['fp'] += 1
    for i, g in enumerate(gold):
        lbl = g['label']
        if i not in matched_gold:
            by_label[lbl]['fn'] += 1

    return by_label


def compute_f1(tp, fp, fn) -> float:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'

    annotators = {
        'gpt41mini': f'data/annotations/gpt41mini_{split}.jsonl',
        'llama31_8b': f'data/annotations/llama31_8b_{split}.jsonl',
        'gliner': f'data/annotations/gliner_multi_{split}.jsonl',
    }

    data = {}
    for name, path in annotators.items():
        if not Path(path).exists():
            print(f'Missing: {path}')
            sys.exit(1)
        data[name] = load_annotations(path)

    doc_ids = set(data['gpt41mini']) & set(data['llama31_8b']) & set(data['gliner'])
    print(f'\n=== Inter-Annotator Agreement ({split}, n={len(doc_ids)} docs) ===')

    # pairwise f1
    pairs = list(combinations(annotators.keys(), 2))
    for a, b in pairs:
        totals = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        for doc_id in doc_ids:
            by_label = pairwise_f1(data[a][doc_id], data[b][doc_id])
            for lbl, counts in by_label.items():
                for k, v in counts.items():
                    totals[lbl][k] += v
            # also reverse direction for symmetry
            by_label2 = pairwise_f1(data[b][doc_id], data[a][doc_id])
            for lbl, counts in by_label2.items():
                for k in ('tp', 'fp', 'fn'):
                    totals[lbl][k] += counts[k]

        all_tp = sum(totals[l]['tp'] for l in totals)
        all_fp = sum(totals[l]['fp'] for l in totals)
        all_fn = sum(totals[l]['fn'] for l in totals)
        overall = compute_f1(all_tp, all_fp, all_fn)

        print(f'\n{a} vs {b} (overall F1 = {round(overall, 3)})')
        print(f"{'label'} {'F1'}")
        for lbl in LABELS:
            t = totals[lbl]
            f1 = compute_f1(t['tp'], t['fp'], t['fn'])
            n = t['tp'] + t['fn']
            if n > 0:
                print(f'{lbl} {round(f1, 3)} (n={n})')

    # consensus rate, share of spans that 2 of 3 annotators agree on
    print('\nconsensus rate (>= 2/3 annotators agree, overlap >= ' + str(OVERLAP_THRESHOLD) + ')')
    names = list(annotators.keys())
    total_unique, total_consensus = 0, 0
    label_unique = defaultdict(int)
    label_consensus = defaultdict(int)

    for doc_id in doc_ids:
        # all unique spans across annotators
        all_spans = []
        for name in names:
            for ent in data[name][doc_id]:
                all_spans.append((name, ent))

        claimed = [False] * len(all_spans)
        groups = []
        for i, (na, ea) in enumerate(all_spans):
            if claimed[i]:
                continue
            group = [(na, ea)]
            claimed[i] = True
            for j, (nb, eb) in enumerate(all_spans):
                if claimed[j] or na == nb:
                    continue
                if ea['label'] == eb['label'] and span_overlap(ea['start'], ea['end'], eb['start'], eb['end']) >= OVERLAP_THRESHOLD:
                    group.append((nb, eb))
                    claimed[j] = True
            groups.append(group)

        for group in groups:
            n_annotators = len(set(name for name, _ in group))
            lbl = group[0][1]['label']
            total_unique += 1
            label_unique[lbl] += 1
            if n_annotators >= 2:
                total_consensus += 1
                label_consensus[lbl] += 1

    print(f'\noverall: {total_consensus}/{total_unique} = {round(total_consensus/total_unique, 3)}')
    print(f"{'label'} {'rate'} {'consensus'} {'total'}")
    for lbl in LABELS:
        n = label_unique[lbl]
        c = label_consensus[lbl]
        if n > 0:
            print(f'{lbl} {round(c/n, 3)} {c} {n}')
