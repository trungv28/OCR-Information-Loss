import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, sentence_window

EF_URL = 'http://localhost:8090/service/disambiguate'
WD_URL = 'https://www.wikidata.org/w/api.php'
RECON_URL = 'https://wikidata.reconci.link/{lang}/api'
HEADERS = {'User-Agent': 'ocr-information-loss/1.0 (' + os.environ.get('WIKIDATA_CONTACT', 'research use') + ')'}

TOP_K = PARAMS['annotation']['candidates']
EF_SLEEP = 0.1  # local server
API_SLEEP = 0.3  # remote apis
MAX_RETRY = 3

TYPE_MAP = {
    'pers': 'Q5',
    'loc': 'Q2221906',
    'org': 'Q43229',
    'prod': None,
}
LINKABLE = {'pers', 'loc', 'org', 'prod'}


def with_retry(fn, label: str, *args, **kwargs):
    # exponential backoff, warns and returns [] on the last failure
    for attempt in range(MAX_RETRY):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f'[warn] {label} failed after {MAX_RETRY} attempts: {e}')
    return []


def ef_nbest(text: str, gt_start: int, gt_end: int, mention: str, lang: str) -> list[dict]:
    query = {
        'text': text,
        'entities': [{'rawName': mention, 'offsetStart': gt_start, 'offsetEnd': gt_end}],
        'language': {'lang': lang},
        'nbest': True,
    }
    resp = requests.post(EF_URL, files={'query': (None, json.dumps(query))}, timeout=30)
    resp.raise_for_status()
    return [
        {'qid': e['wikidataId'], 'score': e.get('confidence_score', 0.0)}
        for e in resp.json().get('entities', [])
        if e.get('offsetStart') == gt_start
        and e.get('offsetEnd') == gt_end
        and e.get('wikidataId')
    ]


def wd_search(surface: str, lang: str) -> list[dict]:
    resp = requests.get(WD_URL, headers=HEADERS, params={
        'action': 'wbsearchentities',
        'search': surface,
        'language': lang,
        'limit': TOP_K,
        'format': 'json',
    }, timeout=15)
    resp.raise_for_status()
    return [
        {'qid': item['id'], 'canonical': item.get('label', ''), 'description': item.get('description', '')}
        for item in resp.json().get('search', [])
    ]


def wd_recon(surface: str, lang: str, type_qid: str | None) -> list[dict]:
    q = {'query': surface, 'limit': TOP_K}
    if type_qid:
        q['type'] = type_qid
    resp = requests.get(
        RECON_URL.format(lang=lang),
        headers=HEADERS,
        params={'queries': json.dumps({'q0': q})},
        timeout=15,
    )
    resp.raise_for_status()
    return [
        {'qid': item['id'], 'canonical': item.get('name', ''), 'description': ''}
        for item in resp.json().get('q0', {}).get('result', [])
    ]


def fetch_labels(qids: list[str], lang: str) -> dict[str, dict]:
    out = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i+50]
        resp = requests.get(WD_URL, headers=HEADERS, params={
            'action': 'wbgetentities',
            'ids': '|'.join(batch),
            'props': 'labels|descriptions',
            'languages': lang,
            'format': 'json',
        }, timeout=15)
        resp.raise_for_status()
        for qid, data in resp.json().get('entities', {}).items():
            out[qid] = {
                'canonical': data.get('labels', {}).get(lang, {}).get('value', ''),
                'description': data.get('descriptions', {}).get(lang, {}).get('value', ''),
            }
        time.sleep(API_SLEEP)
    return out


def build_candidates(
    gt_full: str, gt_start: int, gt_end: int, mention: str, lang: str, label: str
) -> list[dict]:
    merged = {}  # qid → {canonical, description, sources, ef_score}

    def add(qid, canonical='', description='', source='', ef_score=None):
        if qid not in merged:
            merged[qid] = {'canonical': canonical, 'description': description, 'sources': [], 'ef_score': None}
        if source and source not in merged[qid]['sources']:
            merged[qid]['sources'].append(source)
        if canonical and not merged[qid]['canonical']:
            merged[qid]['canonical'] = canonical
        if description and not merged[qid]['description']:
            merged[qid]['description'] = description
        if ef_score is not None:
            merged[qid]['ef_score'] = ef_score

    # 1. entity-fishing nbest
    for c in with_retry(ef_nbest, 'entity-fishing', gt_full, gt_start, gt_end, mention, lang):
        add(c['qid'], source='entity-fishing', ef_score=c['score'])
    time.sleep(EF_SLEEP)

    # 2. wikidata search
    for c in with_retry(wd_search, 'wikidata_search', mention, lang):
        add(c['qid'], c['canonical'], c['description'], source='wikidata_search')
    time.sleep(API_SLEEP)

    # 3. wikidata reconciliation (type-aware)
    for c in with_retry(wd_recon, 'wikidata_recon', mention, lang, TYPE_MAP.get(label)):
        add(c['qid'], c['canonical'], c['description'], source='wikidata_recon')
    time.sleep(API_SLEEP)

    # fetch labels for ef-only candidates that have no canonical name yet
    missing = [qid for qid, c in merged.items() if not c['canonical']]
    if missing:
        labels = with_retry(fetch_labels, 'wikidata_labels', missing, lang) or {}
        for qid, info in labels.items():
            if qid in merged:
                merged[qid]['canonical'] = info['canonical']
                merged[qid]['description'] = info['description']

    return [
        {
            'candidate_id': f'C{i+1}',
            'qid': qid,
            'canonical': c['canonical'],
            'description': c['description'],
            'source': c['sources'],
        }
        for i, (qid, c) in enumerate(merged.items())
    ]


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    preview = '--preview' in sys.argv

    text_by_id = {
        rec['document_metadata']['document_id']: rec
        for rec in load(split)
    }

    proj_path = Path(f'data/annotations/projected_{split}.jsonl')
    all_docs = [json.loads(l) for l in open(proj_path)]

    entities = [
        (doc, ent)
        for doc in all_docs
        for ent in doc['entities']
        if ent['status'] == 'projected' and ent['label'] in LINKABLE
    ]

    if preview:
        entities = [(d, e) for d, e in entities if d['language'] == 'en'][:5]

    out_path = Path(f'data/annotations/candidates_{split}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists() and not preview:
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    with open(out_path, 'a') as f:
        for doc, ent in tqdm(entities, desc='candidates'):
            eid = ent['entity_id']
            if eid in done:
                continue

            lang = doc['language']
            rec = text_by_id.get(doc['document_id'])
            if not rec:
                continue

            gt_full = rec['ground_truth']['transcription_unit']
            ws, we = sentence_window(gt_full, ent['gt_start'], ent['gt_end'], lang)
            context = gt_full[ws:we]

            candidates = build_candidates(
                gt_full, ent['gt_start'], ent['gt_end'],
                ent['gt_text'], lang, ent['label'],
            )

            row = {
                'entity_id': eid,
                'mention': ent['gt_text'],
                'label': ent['label'],
                'context': context,
                'candidate_qids': candidates,
            }

            if preview:
                print(f"\n[{ent['label']}] {ent['gt_text']!r}")
                print(f'context: {context[:80]!r}')
                for c in candidates:
                    print(f"{c['qid']} {c['canonical']!r} {c['description'][:50]} {c['source']}")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()

    if not preview:
        print(f'Saved to {out_path}')
