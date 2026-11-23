import json
import sys
from pathlib import Path

import numpy as np

SPLIT = 'test' if '--test' in sys.argv else 'dev'


def read_qrels(path: str) -> dict:
    '''-> {qid: set of relevant doc_ids}'''
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[:4]
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def read_trec(path: str) -> dict:
    '''-> {qid: [(doc_id, score), ...] sorted by rank}'''
    runs = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score = parts[:5]
            runs.setdefault(qid, []).append((doc_id, float(score), int(rank)))
    for qid in runs:
        runs[qid].sort(key=lambda x: x[2])
    return runs


def hit_at_k(run_list: list, rel_set: set, k: int) -> int:
    for doc_id, _, rank in run_list[:k]:
        if doc_id in rel_set:
            return 1
    return 0


def best_rank(run_list: list, rel_set: set) -> int:
    for doc_id, _, rank in run_list:
        if doc_id in rel_set:
            return rank
    return 10001


def eval_run(run: dict, qrels: dict, name: str) -> tuple[dict, list]:
    query_results = []
    for qid, rel_set in qrels.items():
        hits = run.get(qid, [])
        r = best_rank(hits, rel_set)
        rr = 1.0 / r if r <= 1000 else 0.0
        query_results.append({
            'query_id': qid,
            'retriever': name,
            'rank': r,
            'rr': rr,
            'hit@1': hit_at_k(hits, rel_set, 1),
            'hit@3': hit_at_k(hits, rel_set, 3),
            'hit@5': hit_at_k(hits, rel_set, 5),
            'hit@10': hit_at_k(hits, rel_set, 10),
            'hit@20': hit_at_k(hits, rel_set, 20),
            'hit@50': hit_at_k(hits, rel_set, 50),
        })

    n = len(query_results)
    agg = {
        'retriever': name,
        'n_queries': n,
        'hit@1': np.mean([r['hit@1'] for r in query_results]),
        'hit@5': np.mean([r['hit@5'] for r in query_results]),
        'hit@10': np.mean([r['hit@10'] for r in query_results]),
        'mrr': np.mean([r['rr'] for r in query_results]),
    }
    return agg, query_results


if __name__ == '__main__':
    qrels_path = f'data/results/qrels_known_item_{SPLIT}.txt'
    qrels = read_qrels(qrels_path)
    print(f'Queries in qrels: {len(qrels)}')

    retrievers = {}
    for name, path in [
        ('bm25_gt', f'runs/bm25_gt_{SPLIT}.trec'),
        ('bm25_ocr', f'runs/bm25_ocr_{SPLIT}.trec'),
        ('dense_gt', f'runs/dense_gt_{SPLIT}.trec'),
        ('dense_ocr', f'runs/dense_ocr_{SPLIT}.trec'),
    ]:
        if not Path(path).exists():
            continue
        retrievers[name] = read_trec(path)

    all_agg = []
    all_query_rows = []

    bm25_gt_results = None
    bm25_ocr_results = None
    dense_gt_results = None
    dense_ocr_results = None

    for name, run in retrievers.items():
        agg, qrows = eval_run(run, qrels, name)
        all_agg.append(agg)
        all_query_rows.extend(qrows)
        if name == 'bm25_gt':
            bm25_gt_results = {r['query_id']: r for r in qrows}
        elif name == 'bm25_ocr':
            bm25_ocr_results = {r['query_id']: r for r in qrows}
        elif name == 'dense_gt':
            dense_gt_results = {r['query_id']: r for r in qrows}
        elif name == 'dense_ocr':
            dense_ocr_results = {r['query_id']: r for r in qrows}

    # rr drop and loss@10 per query
    per_query = {}
    for qid in qrels:
        per_query[qid] = {'query_id': qid}

    if bm25_gt_results and bm25_ocr_results:
        for qid in qrels:
            gt_r = bm25_gt_results.get(qid, {'rr': 0.0, 'hit@1': 0, 'hit@3': 0, 'hit@5': 0, 'hit@10': 0})
            ocr_r = bm25_ocr_results.get(qid, {'rr': 0.0, 'hit@1': 0, 'hit@3': 0, 'hit@5': 0, 'hit@10': 0})
            per_query[qid]['bm25_rr_gt'] = gt_r['rr']
            per_query[qid]['bm25_rr_ocr'] = ocr_r['rr']
            per_query[qid]['bm25_delta_rr'] = gt_r['rr'] - ocr_r['rr']
            for k in (1, 3, 5, 10, 20, 50):
                per_query[qid][f'bm25_hit{k}_gt'] = gt_r.get(f'hit@{k}', 0)
                per_query[qid][f'bm25_hit{k}_ocr'] = ocr_r.get(f'hit@{k}', 0)
                per_query[qid][f'bm25_loss{k}'] = int(
                    gt_r.get(f'hit@{k}', 0) == 1 and ocr_r.get(f'hit@{k}', 0) == 0
                )

        q_eval = [qid for qid in qrels if bm25_gt_results.get(qid, {}).get('hit@10', 0) == 1]
        loss_rate = np.mean([per_query[qid]['bm25_loss10'] for qid in q_eval]) if q_eval else 0.0
        print(f'\nBM25 Q_eval (Hit_GT@10=1): {len(q_eval)}')
        print(f'BM25 Y_loss@10 prevalence: {round(loss_rate, 3)}')

    if dense_gt_results and dense_ocr_results:
        for qid in qrels:
            gt_r = dense_gt_results.get(qid, {'rr': 0.0, 'hit@10': 0})
            ocr_r = dense_ocr_results.get(qid, {'rr': 0.0, 'hit@10': 0})
            per_query[qid]['dense_rr_gt'] = gt_r['rr']
            per_query[qid]['dense_rr_ocr'] = ocr_r['rr']
            per_query[qid]['dense_delta_rr'] = gt_r['rr'] - ocr_r['rr']
            per_query[qid]['dense_hit10_gt'] = gt_r['hit@10']
            per_query[qid]['dense_hit10_ocr'] = ocr_r['hit@10']
            per_query[qid]['dense_loss10'] = int(gt_r['hit@10'] == 1 and ocr_r['hit@10'] == 0)

    # query metadata, cer and wer
    q_meta = {}
    qf = Path(f'data/results/retrieval_queries_{SPLIT}.jsonl')
    if qf.exists():
        with open(qf) as f:
            for line in f:
                q = json.loads(line)
                q_meta[q['query_id']] = q
    for qid, q in q_meta.items():
        if qid in per_query:
            per_query[qid]['document_id'] = q.get('document_id', '')
            per_query[qid]['record_cer'] = q.get('record_cer', 0.0)
            per_query[qid]['hidden_cer'] = q.get('hidden_cer', 0.0)
            per_query[qid]['query_type'] = q.get('query_type', '')
            per_query[qid]['language'] = q.get('language', '')
            per_query[qid]['dataset'] = q.get('dataset', '')

    out = Path('data/results')
    metrics_file = out / f'retrieval_metrics_{SPLIT}.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_agg, f, indent=2)

    qmetrics_file = out / f'retrieval_query_metrics_{SPLIT}.jsonl'
    with open(qmetrics_file, 'w') as f:
        for row in per_query.values():
            f.write(json.dumps(row) + '\n')

    def paired_table(label: str, subset_ids: set, tag: str) -> None:
        if not subset_ids:
            return
        rows = [per_query[qid] for qid in subset_ids if qid in per_query]
        if not rows:
            return
        n = len(rows)
        mrr_gt = np.mean([r.get('bm25_rr_gt', 0) for r in rows])
        mrr_ocr = np.mean([r.get('bm25_rr_ocr', 0) for r in rows])
        drr = np.mean([r.get('bm25_delta_rr', 0) for r in rows])
        print(f'\n{label} [{tag}] N={n}')
        print(f"{''} {'GT'} {'OCR'} {'Δ'} {'Y_loss'} {'n_fail'}")
        for k in (1, 3, 5, 10, 20, 50):
            gt_h = np.mean([r.get(f'bm25_hit{k}_gt', 0) for r in rows])
            ocr_h = np.mean([r.get(f'bm25_hit{k}_ocr', 0) for r in rows])
            q_eval = [r for r in rows if r.get(f'bm25_hit{k}_gt', 0) == 1]
            n_loss = sum(r.get(f'bm25_loss{k}', 0) for r in q_eval)
            loss = n_loss / max(len(q_eval), 1)
            print(f'Hit@{k} {round(gt_h, 3)} {round(ocr_h, 3)} {round(gt_h-ocr_h, 3)} '
                  f'{round(loss, 3)} {n_loss}/{len(q_eval)}')
        print(f'MRR {round(mrr_gt, 3)} {round(mrr_ocr, 3)} {round(drr, 3)}')

    all_ids = set(qrels.keys())
    ctx_ids = {qid for qid in all_ids if per_query.get(qid, {}).get('query_type') == 'context'}
    stress_ctx = {qid for qid in ctx_ids if per_query.get(qid, {}).get('hidden_cer', 0) > 0}

    print('\n=== BM25 Retrieval Results ===')
    paired_table('BM25', ctx_ids, 'context')
    paired_table('BM25', stress_ctx, 'context / stress (hidden corrupted)')

    print(f'\nSaved -> {metrics_file}')
    print(f'Saved -> {qmetrics_file}')
