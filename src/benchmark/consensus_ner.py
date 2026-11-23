import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.config import PARAMS


def annotator_paths(split: str) -> dict[str, str]:
    return {
        'gpt41mini': f'data/annotations/gpt41mini_{split}.jsonl',
        'llama31_8b': f'data/annotations/llama31_8b_{split}.jsonl',
        'gliner': f'data/annotations/gliner_multi_{split}.jsonl',
    }

# whose span wins when annotators agree
PRIORITY = ['gpt41mini', 'llama31_8b', 'gliner']


def overlap(a_start, a_end, b_start, b_end) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    min_len = min(a_end - a_start, b_end - b_start)
    return inter / min_len if min_len > 0 else 0.0


def cluster(mentions_by_ann: dict[str, list]) -> list[list]:
    '''union-find over mentions of different annotators with the same label and enough overlap'''
    flat = [(ann, m) for ann, ms in mentions_by_ann.items() for m in ms]
    n = len(flat)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in combinations(range(n), 2):
        ann_i, m_i = flat[i]
        ann_j, m_j = flat[j]
        if ann_i == ann_j or m_i['label'] != m_j['label']:
            continue
        if overlap(m_i['start'], m_i['end'], m_j['start'], m_j['end']) >= PARAMS['annotation']['overlap']:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups = defaultdict(list)
    for i, item in enumerate(flat):
        groups[find(i)].append(item)
    return list(groups.values())


def pick_rep(cluster: list) -> dict:
    by_ann = {}
    for ann, m in cluster:
        by_ann.setdefault(ann, m)
    for p in PRIORITY:
        if p in by_ann:
            return by_ann[p]
    return cluster[0][1]


if __name__ == '__main__':
    import sys
    split = 'test' if '--test' in sys.argv else 'dev'
    ANNOTATORS = annotator_paths(split)

    meta_by_doc = {}
    ann_by_doc = defaultdict(dict)

    for ann_name, path in ANNOTATORS.items():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec['document_id']
                if doc_id not in meta_by_doc:
                    meta_by_doc[doc_id] = {
                        'dataset': rec['dataset'],
                        'language': rec['language'],
                        'date': rec['date'],
                    }
                seen = set()
                unique = []
                for e in rec['entities']:
                    key = (e['start'], e['end'], e['label'])
                    if key not in seen:
                        seen.add(key)
                        unique.append(e)
                ann_by_doc[doc_id][ann_name] = unique

    out_path = Path(f'data/annotations/consensus_{split}.jsonl')
    total_entities = 0

    with open(out_path, 'w') as f:
        for doc_id in sorted(meta_by_doc):
            clusters = cluster(ann_by_doc[doc_id])
            accepted = sorted(
                [c for c in clusters if len({a for a, _ in c}) >= 2],
                key=lambda c: pick_rep(c)['start'],
            )
            entities = []
            for idx, cluster in enumerate(accepted, start=1):
                rep = pick_rep(cluster)
                entities.append({
                    'entity_id': f'{doc_id}_E{idx}',
                    'label': rep['label'],
                    'text': rep['text'],
                    'start': rep['start'],
                    'end': rep['end'],
                    'support': len({a for a, _ in cluster}),
                })
            row = {'document_id': doc_id, **meta_by_doc[doc_id], 'entities': entities}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            total_entities += len(entities)

    print(f'Saved to {out_path}')
    print(f'Documents: {len(meta_by_doc)} | Accepted entities: {total_entities}')
