import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import PARAMS
from src.data import rho_arg
from src.retrieval.damage import find_p
from src.spans import levenshtein

SPLIT = 'test' if '--test' in sys.argv else 'dev'
K = PARAMS['retrieval']['k']
MAX_J = PARAMS['repair']['max_j']
N_RAND = PARAMS['retrieval']['random_orders']
EPS = 1e-9
K1 = PARAMS['retrieval']['k1']
B = PARAMS['retrieval']['b']
IDF_EPSILON = PARAMS['retrieval']['idf_epsilon']
N_COMPETITORS = PARAMS['retrieval']['competitors']
SEED = PARAMS['seed']

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
    return nfkc_lower(text).split()


def get_edit_blocks(gt_text: str, ocr_text: str):
    '''-> (gt_words, ocr_words, blocks) with one block per non-equal difflib span'''
    gw = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', gt_text)]
    ow = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', ocr_text)]
    sm = SequenceMatcher(None, [w for w, _, _ in gw], [w for w, _, _ in ow], autojunk=False)
    blocks = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        gt_t = ' '.join(gw[k][0] for k in range(i1, i2)) if i2 > i1 else ''
        ocr_t = ' '.join(ow[k][0] for k in range(j1, j2)) if j2 > j1 else ''
        cer = levenshtein(gt_t, ocr_t) / max(len(gt_t), len(ocr_t), 1)
        blocks.append({
            'gt_slice': (i1, i2),
            'ocr_slice': (j1, j2),
            'gt_t': gt_t,
            'ocr_t': ocr_t,
            'cer': cer,
        })
    return gw, ow, blocks


def block_evidence_score(gw, block, ev_pos: list[int], h: int = 2) -> float:
    '''block's local damage anchored at the query-evidence positions: k_h * cer * len'''
    if not ev_pos:
        return 0.0
    i1, i2 = block['gt_slice']
    bpos = list(range(i1, i2)) if i2 > i1 else [max(0, i1 - 1)]
    d = min(abs(e - p) for e in ev_pos for p in bpos)
    k = max(0.0, h + 1 - d)
    ell = max(len(block['gt_t']), len(block['ocr_t']), 1)
    return k * block['cer'] * ell


def bm25_awd_impact_for_block(gw, block, query_terms: list[str], bm25: BM25Okapi, unit_tok_len: int) -> float:
    '''sum_{i in block} A_i r_i'''
    if not query_terms:
        return 0.0
    # tf in the full gt unit
    tf_gt = Counter()
    for w, _, _ in gw:
        for t in tokenize(w):
            tf_gt[t] += 1

    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    contrib = {}
    for t in set(query_terms):
        tf = tf_gt.get(t, 0)
        if tf == 0 or t not in bm25.idf:
            contrib[t] = 0.0
        else:
            idf = bm25.idf[t]
            contrib[t] = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * unit_tok_len / avgdl))

    i1, i2 = block['gt_slice']
    gt_span_toks = [tok for gi in range(i1, i2) for tok in tokenize(gw[gi][0])]
    ocr_toks = tokenize(block['ocr_t'])

    impact = 0.0
    for t in set(query_terms):
        n_s = gt_span_toks.count(t)
        if n_s == 0:
            continue
        n_lost = max(0, n_s - ocr_toks.count(t))
        tf_t = tf_gt.get(t, 1)
        impact += n_lost * contrib[t] / (tf_t + EPS)

    return impact


def corpus_stats(unit_tokens: list[list[str]]):
    '''document frequency, total length and the negative-idf floor over the OCR corpus'''
    df = Counter()
    total_len = 0
    for toks in unit_tokens:
        df.update(set(toks))
        total_len += len(toks)
    n = len(unit_tokens)
    idfs = [math.log(n - f + 0.5) - math.log(f + 0.5) for f in df.values()]
    return df, n, total_len, IDF_EPSILON * (sum(idfs) / max(len(idfs), 1))


def idf_of(term: str, df: Counter, delta: dict, n: int, floor: float) -> float:
    '''idf under the corpus with the target unit replaced, delta carries its df change'''
    f = df.get(term, 0) + delta.get(term, 0)
    v = math.log(n - f + 0.5) - math.log(f + 0.5)
    return v if v >= 0 else floor


def bm25_score(doc_tokens: list[str], q_terms: set, df: Counter, delta: dict,
               n: int, avgdl: float, floor: float) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for t in q_terms:
        f = tf.get(t, 0)
        if f == 0:
            continue
        score += idf_of(t, df, delta, n, floor) * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
    return score


def repaired_words(gw, ow, blocks, repaired_block_indices: set) -> list[str]:
    '''ocr word list with the given blocks swapped back to gt'''
    ocr_words = [w for w, _, _ in ow]
    gt_words = [w for w, _, _ in gw]

    # ocr positions that belong to repaired blocks
    repaired_ocr_positions = set()
    repair_inserts = {}  # ocr position -> gt words to insert

    for bi, block in enumerate(blocks):
        if bi not in repaired_block_indices:
            continue
        i1, i2 = block['gt_slice']
        j1, j2 = block['ocr_slice']
        # swap ocr tokens [j1:j2) for gt tokens [i1:i2)
        for j in range(j1, j2):
            repaired_ocr_positions.add(j)
        repair_inserts[j1] = gt_words[i1:i2]

    repaired = []
    for j, w in enumerate(ocr_words):
        if j in repair_inserts:
            repaired.extend(repair_inserts[j])
        if j not in repaired_ocr_positions:
            repaired.append(w)

    # gt-only insertions past the end go last
    for bi, block in enumerate(blocks):
        if bi not in repaired_block_indices:
            continue
        j1, j2 = block['ocr_slice']
        i1, i2 = block['gt_slice']
        if j1 >= len(ow) and i2 > i1:
            repaired.extend(gt_words[i1:i2])

    return tokenize(' '.join(repaired))


def recovers_at(gw, ow, blocks, order: list[int], q_terms: set, target_tokens: list[str],
                target_idx: int, competitors: list, df: Counter, n: int, total_len: int,
                floor: float) -> int:
    '''smallest k where the repaired unit re-enters the top k, -1 if it never does'''
    # ties keep corpus order like the run files, and a zero score never counts as recovered
    for k, _ in enumerate(order, 1):
        new_tokens = repaired_words(gw, ow, blocks, set(order[:k]))
        # only the target unit changed, so df moves by at most one per term
        old_set, new_set = set(target_tokens), set(new_tokens)
        delta = {t: (t in new_set) - (t in old_set) for t in old_set ^ new_set}
        avgdl = (total_len - len(target_tokens) + len(new_tokens)) / n
        s = bm25_score(new_tokens, q_terms, df, delta, n, avgdl, floor)
        if s <= 0:
            continue
        rank = 1
        for c_idx, ct in competitors:
            cs = bm25_score(ct, q_terms, df, delta, n, avgdl, floor)
            if cs > s or (cs == s and c_idx < target_idx):
                rank += 1
        if rank <= K:
            return k
    return -1


def corrupt_word(w, rho, rng, alphabet):
    '''ocr-like noise an imperfect corrector leaves behind'''
    if not w or rho <= 0:
        return w
    out = []
    for ch in w:
        if rng.random() >= rho:
            out.append(ch)
            continue
        op = rng.choice(('sub', 'del', 'ins'))
        if op == 'sub':
            out.append(rng.choice(alphabet))
        elif op == 'ins':
            out.append(ch)
            out.append(rng.choice(alphabet))
    return ''.join(out)


def noisy_gw(gw, rho, uid, alphabet):
    # seeded per unit, so every strategy sees the same imperfect repair for a block
    rng = random.Random(f'{SEED}|{uid}')
    return [(corrupt_word(w, rho, rng, alphabet), s, e) for w, s, e in gw]


if __name__ == '__main__':
    RHO = rho_arg()
    failed = [r for r in (json.loads(l) for l in open(f'data/results/retrieval_qeld_{SPLIT}.jsonl'))
              if r['y'] == 1]
    q_by_id = {q['query_id']: q for q in (json.loads(l) for l in open(f'data/results/retrieval_queries_{SPLIT}.jsonl'))
               if q.get('query_type') == 'context'}
    units_meta = {u['unit_id']: u for u in (json.loads(l) for l in open(f'data/results/retrieval_units_{SPLIT}.jsonl'))}
    qlab = {qid: q.get('hidden_label') for qid, q in q_by_id.items()}
    failed = [r for r in failed if qlab.get(r['query_id']) != 'prod']
    print('failed pairs (no prod)', len(failed), 'rho', RHO)

    all_units = list(units_meta.values())
    ocr_tokens = {u['unit_id']: tokenize(u['ocr_text']) for u in all_units}
    unit_idx = {u['unit_id']: i for i, u in enumerate(all_units)}
    base_df, n_docs, base_total, idf_floor = corpus_stats(list(ocr_tokens.values()))
    bm25_gt = BM25Okapi([tokenize(u['gt_text']) for u in all_units], k1=K1, b=B, epsilon=IDF_EPSILON)

    q_top = {}
    for line in open(f'runs/bm25_ocr_{SPLIT}.trec'):
        p = line.split()
        if len(p) < 6:
            continue
        if int(p[3]) <= N_COMPETITORS:
            q_top.setdefault(p[0], []).append(p[2])

    strategies = ['bm25_awd_impact', 'qeld_damage', 'corruption_only', 'random']
    recovery_at = {s: [] for s in strategies}

    for row in tqdm(failed, desc=f'repair rho={RHO}'):
        qid, uid = row['query_id'], row['unit_id']
        q, unit = q_by_id.get(qid), units_meta.get(uid)
        if not q or not unit:
            continue
        q_terms = [t for t in tokenize(q['query_text'])
                   if t not in ENTITY_LABELS and t not in STOPWORDS and len(t) >= 2]
        gw, ow, blocks = get_edit_blocks(unit['gt_text'], unit['ocr_text'])
        if not blocks:
            for s in strategies:
                recovery_at[s].append(0)
            continue

        target_tokens = ocr_tokens[uid]
        target_idx = unit_idx[uid]
        competitors = [(unit_idx[c], ocr_tokens[c]) for c in q_top.get(qid, []) if c != uid]
        unit_tok_len = len(tokenize(unit['gt_text']))
        ev_pos = sorted(find_p(q['query_text'], [(w, s, e, 0.0, w) for w, s, e in gw]))

        # predictors always score the clean gt, only the applied repair is degraded
        alphabet = sorted(set(unit['gt_text'])) or list('abcdefghijklmnopqrstuvwxyz')
        gw_rep = noisy_gw(gw, RHO, uid, alphabet)

        block_scores = {s: [] for s in strategies}
        for block in blocks:
            block_scores['corruption_only'].append(block['cer'])
            block_scores['qeld_damage'].append(block_evidence_score(gw, block, ev_pos))
            block_scores['bm25_awd_impact'].append(
                bm25_awd_impact_for_block(gw, block, q_terms, bm25_gt, unit_tok_len))
            block_scores['random'].append(0.0)

        args = (ow, blocks)
        tail = (set(q_terms), target_tokens, target_idx, competitors,
                base_df, n_docs, base_total, idf_floor)
        for strategy in strategies:
            if strategy == 'random':
                seeds = []
                rng = random.Random(SEED)
                for _ in range(N_RAND):
                    order = list(range(len(blocks)))
                    rng.shuffle(order)
                    seeds.append(recovers_at(gw_rep, *args, order, *tail))
                valid = [j for j in seeds if j >= 0]
                recovery_at[strategy].append(int(np.median(valid)) if valid else -1)
            else:
                order = sorted(range(len(blocks)), key=lambda bi: -block_scores[strategy][bi])
                recovery_at[strategy].append(recovers_at(gw_rep, *args, order, *tail))

    n = len(failed)
    out = {'rho': RHO, 'n': n, 'strategies': {}}
    for s in strategies:
        curve = [sum(1 for j in recovery_at[s] if 0 <= j <= k) / max(n, 1) for k in range(MAX_J + 1)]
        auc = float(np.mean(curve))
        out['strategies'][s] = {'repair_auc': round(auc, 4), 'rec_at_1': round(curve[1], 4),
                                'max_recovery': round(curve[-1], 4)}

    # per-pair rows, for paired bootstrap cis
    out['pairs'] = []
    for pi, row in enumerate(failed):
        uid = row.get('unit_id', '')
        did = units_meta.get(uid, {}).get('document_id', uid)
        pr = {'query_id': row['query_id'], 'unit_id': uid, 'document_id': did}
        for s2 in strategies:
            j = recovery_at[s2][pi] if pi < len(recovery_at[s2]) else -1
            pr[f'auc_{s2}'] = ((MAX_J + 1 - j) / (MAX_J + 1)) if 0 <= j <= MAX_J else 0.0
        out['pairs'].append(pr)

    p = Path('data/results/retrieval_repair' + ('_rho' + str(RHO) if RHO else '') + '_' + SPLIT + '.json')
    json.dump(out, open(p, 'w'), indent=2)
    print('saved', p)
