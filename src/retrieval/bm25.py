import json
import sys
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import PARAMS

SPLIT = 'test' if '--test' in sys.argv else 'dev'
TOP_K = PARAMS['retrieval']['run_depth']


def nfkc_lower(text: str) -> str:
    return unicodedata.normalize('NFKC', text).lower()


def tokenize(text: str) -> list[str]:
    return nfkc_lower(text).split()


def write_trec(path: Path, results: dict) -> None:
    '''results: {qid: [(doc_id, score), ...] sorted desc}'''
    with open(path, 'w') as f:
        for qid, hits in results.items():
            for rank, (doc_id, score) in enumerate(hits[:TOP_K], 1):
                f.write(f'{qid} Q0 {doc_id} {rank} {round(score, 6)} bm25\n')


if __name__ == '__main__':
    units = []
    with open(f'data/results/retrieval_units_{SPLIT}.jsonl') as f:
        for line in f:
            units.append(json.loads(line))

    queries = []
    with open(f'data/results/retrieval_queries_{SPLIT}.jsonl') as f:
        for line in f:
            queries.append(json.loads(line))

    u_ids = [u['unit_id'] for u in units]
    gt_corpus = [tokenize(u['gt_text']) for u in units]
    ocr_corpus = [tokenize(u['ocr_text']) for u in units]

    print(f'Units: {len(units)}, Queries: {len(queries)}')
    print('Building BM25 GT index...')
    bm25_gt = BM25Okapi(gt_corpus, k1=PARAMS['retrieval']['k1'], b=PARAMS['retrieval']['b'], epsilon=PARAMS['retrieval']['idf_epsilon'])
    print('Building BM25 OCR index...')
    bm25_ocr = BM25Okapi(ocr_corpus, k1=PARAMS['retrieval']['k1'], b=PARAMS['retrieval']['b'], epsilon=PARAMS['retrieval']['idf_epsilon'])

    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)

    gt_results = {}
    ocr_results = {}

    print('Running queries...')
    for q in queries:
        qid = q['query_id']
        qtoks = tokenize(q['query_text'])

        gt_scores = bm25_gt.get_scores(qtoks)
        ocr_scores = bm25_ocr.get_scores(qtoks)

        gt_ranked = sorted(zip(u_ids, gt_scores), key=lambda x: -x[1])
        ocr_ranked = sorted(zip(u_ids, ocr_scores), key=lambda x: -x[1])

        gt_results[qid] = gt_ranked[:TOP_K]
        ocr_results[qid] = ocr_ranked[:TOP_K]

    gt_path = runs_dir / f'bm25_gt_{SPLIT}.trec'
    ocr_path = runs_dir / f'bm25_ocr_{SPLIT}.trec'
    write_trec(gt_path, gt_results)
    write_trec(ocr_path, ocr_results)

    print(f'Saved -> {gt_path}')
    print(f'Saved -> {ocr_path}')
