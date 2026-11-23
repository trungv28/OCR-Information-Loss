import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

from src.config import PARAMS
from src.data import load, line_offsets as get_line_offsets

SPLIT = 'test' if '--test' in sys.argv else 'dev'
WINDOW = PARAMS['retrieval']['window']
MAX_QUERY = PARAMS['retrieval']['max_query']
MAX_ANCH = PARAMS['retrieval']['max_anchors']
N_IDF = PARAMS['retrieval']['n_idf']

HIDE_PRIORITY = {'loc': 0, 'org': 1, 'prod': 2, 'time': 3, 'pers': 4}

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


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r'\b[a-zA-ZÀ-ÿĀ-ɏ]{3,}\b', nfkc_lower(text))
            if w not in STOPWORDS]


def compute_idf(units: list[dict]) -> dict[str, float]:
    N = len(units)
    df = Counter()
    for u in units:
        df.update(set(content_words(u['gt_text'])))
    return {w: math.log(N / max(c, 1)) for w, c in df.items()}


def fragment_words(text: str) -> set[str]:
    """tails of words broken across a line, e.g. 'menced' from 'com-menced'"""
    tails = {nfkc_lower(m.group(1))
             for m in re.finditer(r'[-\u00ac]\s*\n\s*([a-zA-Z\u00c0-\u00ff\u0100-\u024f]{3,})', text)}
    if not tails:
        return set()
    dehyph = re.sub(r'[-\u00ac]\s*\n\s*[a-zA-Z\u00c0-\u00ff\u0100-\u024f]+', ' ', text)
    return tails - set(content_words(dehyph))


def mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s), min(len(chars), e)):
            chars[i] = ' '
    return ''.join(chars)


if __name__ == '__main__':
    # el-linked entities
    linked = {}
    with open(f'data/annotations/consensus_el_{SPLIT}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['status'] == 'accepted' and r.get('qid'):
                linked[r['entity_id']] = r['qid']

    # ner outcomes, positions and gt and ocr surface forms
    ner_outcomes = {}
    with open(f'data/results/outcomes_{SPLIT}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            ner_outcomes[r['entity_id']] = r

    # el entities grouped by document
    ents_by_doc = defaultdict(list)
    for eid, qid in linked.items():
        ner = ner_outcomes.get(eid, {})
        doc_id = ner.get('document_id')
        if not doc_id or ner.get('gt_start') is None:
            continue
        gt_surf = ner.get('gt_text', '')
        ocr_surf = ner.get('ocr_text', gt_surf)
        ents_by_doc[doc_id].append({
            'eid': eid,
            'qid': qid,
            'label': ner.get('label', ''),
            'gt_text': gt_surf,
            'ocr_text': ocr_surf,
            'gt_start': ner['gt_start'],
            'gt_end': ner['gt_end'],
            'hidden_cer': jiwer.cer(gt_surf, ocr_surf) if gt_surf else 0.0,
        })

    # retrieval units, for idf and unit_id lookup
    units = []
    with open(f'data/results/retrieval_units_{SPLIT}.jsonl') as f:
        for line in f:
            units.append(json.loads(line))

    # (doc_id, first_line) -> unit_id
    unit_id_by_key = {}
    n_windows_by_doc = {}
    for u in units:
        key = (u['document_id'], u['first_line'])
        unit_id_by_key[key] = u['unit_id']
        n_windows_by_doc[u['document_id']] = max(
            n_windows_by_doc.get(u['document_id'], 0), u['first_line'] + 1
        )

    # raw records for line offsets
    records = {
        rec['document_metadata']['document_id']: rec
        for rec in load(SPLIT)
    }

    # units by doc, for cer and language
    units_by_doc = {}
    for u in units:
        if u['document_id'] not in units_by_doc:
            units_by_doc[u['document_id']] = u

    print('Computing IDF...')
    idf = compute_idf(units)

    queries = []
    qid_counter = 0

    for doc_id, doc_ents in ents_by_doc.items():
        rec = records.get(doc_id)
        meta = units_by_doc.get(doc_id, {})
        if not rec:
            continue

        gt_text = rec['ground_truth']['transcription_unit']
        ocr_text = rec['ocr_hypothesis']['transcription_unit']
        record_cer = jiwer.cer(gt_text, ocr_text) if gt_text else 0.0
        gt_offs, _ = get_line_offsets(rec)
        frags = fragment_words(gt_text)
        n_lines = len(gt_offs)
        n_windows = max(1, n_lines - WINDOW + 1)

        for line_idx, (ls, le) in enumerate(gt_offs):
            # entities whose span starts in this line
            line_ents = [
                e for e in doc_ents
                if e['gt_start'] >= ls and e['gt_start'] < le
            ]
            if not line_ents:
                continue

            # windows covering line_idx
            w_lo = max(0, line_idx - WINDOW + 1)
            w_hi = min(line_idx, n_windows - 1)
            relevant_unit_ids = [
                unit_id_by_key[(doc_id, w)]
                for w in range(w_lo, w_hi + 1)
                if (doc_id, w) in unit_id_by_key
            ]
            if not relevant_unit_ids:
                continue

            # high-idf context words outside every entity span
            all_spans = [(e['gt_start'] - ls, e['gt_end'] - ls) for e in line_ents]
            masked = mask_spans(gt_text[ls:le], all_spans)
            idf_pool = sorted(set(content_words(masked)) - frags,
                              key=lambda w: -idf.get(w, 0))

            # symmetric: each entity takes a turn as the hidden target
            for hi, hidden in enumerate(line_ents):
                # the target must not reach the query, so drop anchors and context
                # words that share any token with it
                htoks = set(nfkc_lower(hidden['gt_text']).split())
                anchors = [e for j, e in enumerate(line_ents)
                           if j != hi and not (set(nfkc_lower(e['gt_text']).split()) & htoks)][:MAX_ANCH]
                anchor_texts = [' '.join(a['gt_text'].split()[:4]) for a in anchors]
                top_idf = [w for w in idf_pool if w not in htoks][:N_IDF]

                # context query, the other entities on the line plus high-idf words
                qb_parts = anchor_texts + top_idf
                if qb_parts:
                    queries.append({
                        'query_id': 'q' + str(qid_counter).zfill(5) + 'b',
                        'query_text': ' '.join(qb_parts[:MAX_QUERY]),
                        'query_type': 'context',
                        'document_id': doc_id,
                        'line_idx': line_idx,
                        'split': SPLIT,
                        'language': meta.get('language', ''),
                        'dataset': meta.get('dataset', ''),
                        'hidden_eid': hidden['eid'],
                        'hidden_text': hidden['gt_text'],
                        'hidden_label': hidden['label'],
                        'hidden_qid': hidden['qid'],
                        'hidden_cer': round(hidden['hidden_cer'], 4),
                        'anchor_eids': [a['eid'] for a in anchors],
                        'anchor_texts': anchor_texts,
                        'relevant_unit_ids': relevant_unit_ids,
                        'record_cer': round(record_cer, 4),
                    })

                qid_counter += 1

    out = Path('data/results')
    qfile = out / f'retrieval_queries_{SPLIT}.jsonl'
    with open(qfile, 'w') as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    qrels_file = out / f'qrels_known_item_{SPLIT}.txt'
    with open(qrels_file, 'w') as f:
        for q in queries:
            for uid in q['relevant_unit_ids']:
                f.write(f"{q['query_id']} 0 {uid} 1\n")

    qb = [q for q in queries if q['query_type'] == 'context']
    stress_b = [q for q in qb if q['hidden_cer'] > 0]
    qlens_b = [len(q['query_text'].split()) for q in qb]

    print(f'Units: {len(units)}')
    print(f'Context queries: {len(qb)} (stress: {len(stress_b)}, '
          f'{round(100*len(stress_b)/max(len(qb),1), 1)}%)')
    print(f'query len: mean={round(sum(qlens_b)/max(len(qlens_b),1), 1)} '
          f'min={min(qlens_b) if qlens_b else 0} max={max(qlens_b) if qlens_b else 0}')
    print(f'Saved -> {qfile}')
    print(f'Saved -> {qrels_file}')
