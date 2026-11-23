import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, line_offsets as get_line_offsets
from src.local_damage import parse_line, local_damage, text_error_span_cer
from src.spans import levenshtein

SPLIT = 'test' if '--test' in sys.argv else 'dev'
H = PARAMS['local_damage']['h']
K = PARAMS['retrieval']['k']
EPS = 1e-9

ENTITY_LABELS = {'loc', 'org', 'pers', 'prod', 'time'}
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','of','for','with',
    'is','are','was','were','be','been','by','from','that','this','it','its',
    'he','she','they','his','her','their','we','you','have','has','had','not',
    'as','so','if','do','did','will','would','can','could','may','might','shall',
    'also','into','up','out','about','after','before','than','then','when',
    'le','la','les','un','une','des','du','de','en','et','ou','dans','sur',
    'est','sont','était','au','aux','par','pour','avec','ce','cet','cette',
    'il','elle','ils','elles','se','qui','que','qu','ne','pas','plus','tout',
    'même','mais','comme','si','ont','après','avant','depuis','lors','dont',
    'leur','leurs','son','sa','ses','mon','ma','mes','votre','notre','nos',
    'der','die','das','ein','eine','und','oder','in','im','an','auf','zu',
    'von','mit','des','dem','den','ist','sind','war','wurde','worden','hat',
    'haben','hatte','bei','als','für','über','sich','er','sie','es','wir',
    'ihr','nicht','noch','auch','durch','nach','vor','ohne','gegen','wie',
    'aber','doch','schon','wenn','mehr','nur','bis','unter','zwischen',
}


def nfkc_lower(t: str) -> str:
    return unicodedata.normalize('NFKC', t).lower()


def tokenize(text: str) -> list[str]:
    '''same tokenizer as bm25.py'''
    return nfkc_lower(text).split()


# token tuple is (gt_word, char_start, char_end, corruption, ocr_surface)
# ocr_surface is the ocr text aligned to this gt token, used for r_i
Token = tuple[str, int, int, float, str]


def word_corruptions(gt_text: str, ocr_text: str) -> list[Token]:
    gw = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', gt_text)]
    ow = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', ocr_text)]
    # default: ocr_surface = gt_word (c=0, equal block)
    result = [(w, s, e, 0.0, w) for w, s, e in gw]
    sm = SequenceMatcher(None, [w for w, _, _ in gw], [w for w, _, _ in ow], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        gt_t = ' '.join(gw[k][0] for k in range(i1, i2)) if i2 > i1 else ''
        ocr_t = ' '.join(ow[k][0] for k in range(j1, j2)) if j2 > j1 else ''
        C = levenshtein(gt_t, ocr_t) / max(len(gt_t), len(ocr_t), 1)
        for idx in range(i1, i2):
            w, s, e, _, _ = result[idx]
            result[idx] = (w, s, e, C, ocr_t)
    return result



def find_p(query_text: str, tokens: list[Token]) -> set[int]:
    '''p_t(u), gt token positions carrying query evidence'''
    # whole-token match like bm25 with punctuation attached, the hidden target is never evidence
    q_terms = {
        t for t in nfkc_lower(query_text).split()
        if t not in ENTITY_LABELS and t not in STOPWORDS and len(t) >= 2
    }
    return {i for i, (w, _, _, _, _) in enumerate(tokens) if nfkc_lower(w) in q_terms}


def compute_bm25_awd(
    query_text: str,
    tokens: list[Token],
    bm25: BM25Okapi,
    unit_tok_len: int,
) -> float:
    '''attribution damage, sum A_i r_i / sum A_i with A_i the term contribution at token i and r_i the evidence lost'''
    q_terms = [
        t for t in tokenize(query_text)
        if t not in ENTITY_LABELS and t not in STOPWORDS and len(t) >= 2
    ]
    if not q_terms:
        return 0.0

    # term frequency in the gt unit
    tf_gt = Counter()
    for w, _, _, _, _ in tokens:
        for t in tokenize(w):
            tf_gt[t] += 1

    # bm25 contribution of each query term
    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    contrib = {}
    for t in set(q_terms):
        tf = tf_gt.get(t, 0)
        if tf == 0 or t not in bm25.idf:
            contrib[t] = 0.0
        else:
            idf = bm25.idf[t]
            contrib[t] = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * unit_tok_len / avgdl))

    total_a = 0.0
    total_awd = 0.0

    for w, _, _, _, ocr_surf in tokens:
        w_toks = tokenize(w)
        ocr_toks = tokenize(ocr_surf)

        # A_i, bm25 attribution spread over the tokens
        a_i = 0.0
        for t in set(q_terms):
            cnt_gt = w_toks.count(t)
            if cnt_gt == 0:
                continue
            tf_t = tf_gt.get(t, 1)
            a_i += cnt_gt * contrib[t] / (tf_t + EPS)

        if a_i <= 0:
            continue

        # r_i, share of this token's query-term evidence lost in ocr
        ev_gt = sum(w_toks.count(t) for t in q_terms)
        ev_ocr = sum(min(w_toks.count(t), ocr_toks.count(t)) for t in q_terms)
        r_i = 1.0 - ev_ocr / (ev_gt + EPS)

        total_awd += a_i * r_i
        total_a += a_i

    return total_awd / (total_a + EPS)



def read_trec_ranks(path: str, target_pairs: set[tuple]) -> dict[tuple, int]:
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            qid, uid, rank = parts[0], parts[2], int(parts[3])
            if (qid, uid) in target_pairs:
                result[(qid, uid)] = rank
    return result


def read_trec_scores(path: str, target_pairs: set[tuple]) -> dict[tuple, float]:
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            qid, uid, score = parts[0], parts[2], float(parts[4])
            if (qid, uid) in target_pairs:
                result[(qid, uid)] = score
    return result



if __name__ == '__main__':
    # context queries
    queries = []
    with open(f'data/results/retrieval_queries_{SPLIT}.jsonl') as f:
        for line in f:
            q = json.loads(line)
            if q.get('query_type') == 'context':
                queries.append(q)
    q_by_id = {q['query_id']: q for q in queries}
    ctx_qids = set(q_by_id)
    print(f'Context queries: {len(queries)}')

    # qrels (context only)
    qrels = {}
    with open(f'data/results/qrels_known_item_{SPLIT}.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, uid, rel = parts[:4]
            if qid in ctx_qids and int(rel) > 0:
                qrels.setdefault(qid, set()).add(uid)

    # E_k, pairs the gt run retrieves
    all_pairs = {(qid, uid) for qid, uids in qrels.items() for uid in uids}
    print(f'Loading TREC runs ({len(all_pairs)} target pairs)...')
    gt_ranks = read_trec_ranks(f'runs/bm25_gt_{SPLIT}.trec', all_pairs)
    ocr_ranks = read_trec_ranks(f'runs/bm25_ocr_{SPLIT}.trec', all_pairs)
    gt_scores = read_trec_scores(f'runs/bm25_gt_{SPLIT}.trec', all_pairs)
    ocr_scores = read_trec_scores(f'runs/bm25_ocr_{SPLIT}.trec', all_pairs)
    # a zero score never counts, otherwise ties at 0 fill the top k in corpus order
    e_k = [(qid, uid) for (qid, uid) in all_pairs
           if gt_ranks.get((qid, uid), 9999) <= K and gt_scores.get((qid, uid), 0.0) > 0]
    print(f'E_{K}: {len(e_k)} pairs from {len({q for q, _ in e_k})} queries')

    # load units
    units_meta = {}
    with open(f'data/results/retrieval_units_{SPLIT}.jsonl') as f:
        for line in f:
            u = json.loads(line)
            units_meta[u['unit_id']] = u

    # gt bm25 index for idf and avgdl
    print('Building GT BM25 index for IDF extraction...')
    all_units = list(units_meta.values())
    gt_corpus = [tokenize(u['gt_text']) for u in all_units]
    bm25_gt = BM25Okapi(gt_corpus, k1=PARAMS['retrieval']['k1'], b=PARAMS['retrieval']['b'], epsilon=PARAMS['retrieval']['idf_epsilon'])

    # raw records, for unit char offsets
    raw_records = {
        rec['document_metadata']['document_id']: rec
        for rec in load(SPLIT)
    }

    # unit cache with damage map, bm25 lengths and offsets
    print('Building unit damage maps...')
    unit_cache = {}
    needed_uids = {uid for _, uid in e_k}
    for uid in tqdm(needed_uids, desc='D_u'):
        u = units_meta.get(uid)
        if not u:
            continue
        doc_id = u['document_id']
        rec = raw_records.get(doc_id)
        gt_offs, _ = get_line_offsets(rec) if rec else ([], [])
        fl = u['first_line']
        gt_unit_offset = gt_offs[fl][0] if fl < len(gt_offs) else 0

        tokens = word_corruptions(u['gt_text'], u['ocr_text'])
        word_spans, span_blks, insertions = parse_line(u['gt_text'], u['ocr_text'])
        unit_cer = jiwer.cer(u['gt_text'], u['ocr_text']) if u['gt_text'].strip() else 0.0
        esc = text_error_span_cer(u['gt_text'], u['ocr_text']) if u['gt_text'].strip() else 0.0
        unit_tok_len = len(tokenize(u['gt_text']))

        unit_cache[uid] = {
            'tokens': tokens,
            'word_spans': word_spans,
            'span_blks': span_blks,
            'insertions': insertions,
            'unit_cer': unit_cer,
            'error_span_cer': esc,
            'unit_tok_len': unit_tok_len,
            'gt_unit_offset': gt_unit_offset,
        }

    # local damage and attribution damage for each E_k pair
    print('Computing QELD + BM25-AWD...')
    rows = []
    n_empty_p = 0

    for qid, uid in tqdm(e_k, desc='scores'):
        q = q_by_id.get(qid)
        ud = unit_cache.get(uid)
        if not q or not ud:
            continue

        ocr_rank = ocr_ranks.get((qid, uid), 9999)
        y = int(ocr_rank > K)

        p_set = find_p(q['query_text'], ud['tokens'])

        if not p_set:
            n_empty_p += 1
            continue

        qeld = local_damage(ud['word_spans'], ud['span_blks'], ud['insertions'], sorted(p_set), H)
        bm25awd = compute_bm25_awd(q['query_text'], ud['tokens'], bm25_gt, ud['unit_tok_len'])

        gs = gt_scores.get((qid, uid), 0.0)
        os_ = ocr_scores.get((qid, uid), 0.0)
        score_drop = max(0.0, gs - os_)
        norm_drop = score_drop / (gs + EPS)

        rows.append({
            'query_id': qid,
            'unit_id': uid,
            'document_id': q['document_id'],
            'y': y,
            'qeld': round(qeld, 6),
            'bm25_awd': round(bm25awd, 6),
            'score_drop': round(score_drop, 6),
            'norm_drop': round(norm_drop, 6),
            'unit_cer': round(ud['unit_cer'], 6),
            'error_span_cer': round(ud['error_span_cer'], 6),
            'record_cer': round(q.get('record_cer', 0.0), 6),
            'hidden_cer': round(q.get('hidden_cer', 0.0), 6),
            'n_evidence': len(p_set),
            'language': q.get('language', ''),
            'dataset': q.get('dataset', ''),
        })

    outfile = Path(f'data/results/retrieval_qeld_{SPLIT}.jsonl')
    with open(outfile, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n_fail = sum(r['y'] for r in rows)
    print(f'\nE_{K} pairs: {len(e_k)}')
    print(f'Empty P(q,u): {n_empty_p} (excluded)')
    print(f'Evaluated pairs: {len(rows)}')
    print(f'OCR failures (y=1): {n_fail} ({round(100*n_fail/max(len(rows),1), 1)}%)')

    print(f'\nSaved -> {outfile}')
