import json
import random
from collections import Counter, defaultdict

import jiwer
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PARAMS
from src.local_damage import span_error_cer
from src.ner.outcomes import classify
from src.spans import pop_auc

R = 'data/results/'
A = 'data/annotations/'
H = 'data/results/hmbert/'
MAX_J = PARAMS['repair']['max_j']
K = PARAMS['retrieval']['k']
RHOS = [str(r) for r in PARAMS['evaluation']['rhos']]
N_BOOT = PARAMS['evaluation']['bootstrap']
SEED = PARAMS['evaluation']['seed']
COLL = {'icdar2017': 'icdar-2017', 'impresso-nzz': 'nzz', 'impresso-snippets': 'snippets', 'overproof-combined': 'overproof'}

NER_PRED = [('record_cer', 'Document CER'), ('mention_cer', 'Unit CER'), ('error_span_cer', 'Error-span CER'),
            ('local_damage', 'Local damage'), ('edit_lime_damage', 'Attr. damage')]
EL_PRED = [('record_cer', 'Document CER'), ('el_mention_cer', 'Unit CER'), ('el_error_span_cer', 'Error-span CER'),
           ('local_damage', 'Local damage'), ('el_edit_lime_damage', 'Attr. damage')]
IR_PRED = [('record_cer', 'Document CER'), ('unit_cer', 'Unit CER'), ('error_span_cer', 'Error-span CER'),
           ('qeld', 'Local damage'), ('bm25_awd', 'Attr. damage')]
REPAIR = [('mention_first', 'Entity-first'), ('corruption', 'Error-span CER'), ('random', 'Random'),
          ('local_damage', 'Local damage'), ('gt_damage', 'Attr. damage')]
IR_REPAIR = [('corruption_only', 'Error-span CER'), ('random', 'Random'),
             ('qeld_damage', 'Local damage'), ('bm25_awd_impact', 'Attr. damage')]


def jl(path: str) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def ci(rows: list, fn) -> tuple:
    # point estimate and 95% interval over resamples of whole documents
    rng = random.Random(SEED)
    docs = defaultdict(list)
    for r in rows:
        docs[r['document_id']].append(r)
    ids = list(docs)
    point = fn(rows)
    boot = [fn([r for d in rng.choices(ids, k=len(ids)) for r in docs[d]]) for _ in range(N_BOOT)]
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def auroc(y: str, key: str):
    return lambda rs: roc_auc_score([r[y] for r in rs], [r[key] for r in rs])


def ap(y: str, key: str):
    return lambda rs: average_precision_score([r[y] for r in rs], [r[key] for r in rs])


def mean(key: str):
    return lambda rs: float(np.mean([r[key] for r in rs]))


def human() -> dict:
    # human validation decisions, an item without a decision counts as accepted
    rej = {'ner': set(), 'alignment': set(), 'el': set()}
    for lang in ('en', 'de', 'fr'):
        for r in jl(A + 'annot_' + lang + '.jsonl'):
            if r['decision'] == 'reject':
                rej[r['task']].add(r['entity_id'])
    return rej


def bm25_ranks(path: str) -> dict:
    out = {}
    for line in open(path):
        qid, _, uid, rank, score = line.split()[:5]
        out[(qid, uid)] = (int(rank), float(score))
    return out


def retrieved(ranks: dict, key: tuple) -> bool:
    # a zero score means no query term matched, so it is never a retrieval whatever its rank
    rank, score = ranks.get(key, (K + 1, 0.0))
    return rank <= K and score > 0


def correctness(queries: list) -> dict:
    # gt and ocr correctness of every unit, for the gains and the retrieval pairs
    gt = {r['document_id']: r['entities'] for r in jl(A + 'ner_gt_test.jsonl')}
    ocr = {r['document_id']: r['entities'] for r in jl(A + 'ner_ocr_test.jsonl')}
    ner = []
    for doc in jl(A + 'projected_test.jsonl'):
        for e in doc['entities']:
            if e['status'] != 'projected' or e['label'] == 'prod':
                continue
            g = classify(gt.get(doc['document_id'], []), e['gt_start'], e['gt_end'], e['label'])[0]
            o = classify(ocr.get(doc['document_id'], []), e['ocr_start'], e['ocr_end'], e['label'])[0]
            ner.append({'entity_id': e['entity_id'], 'gt_correct': g == 'correct', 'ocr_correct': o == 'correct'})
    el = [{'entity_id': r['entity_id'], 'gt_correct': r['gt_pred'] == r['ref_qid'], 'ocr_correct': r['ocr_pred'] == r['ref_qid']}
          for r in jl(R + 'outcomes_el_test.jsonl') if r['label'] != 'prod']
    gt_ranks, ocr_ranks = bm25_ranks('runs/bm25_gt_test.trec'), bm25_ranks('runs/bm25_ocr_test.trec')
    ir = [{'query_id': q['query_id'], 'unit_id': uid,
           'gt_correct': retrieved(gt_ranks, (q['query_id'], uid)), 'ocr_correct': retrieved(ocr_ranks, (q['query_id'], uid))}
          for q in queries if q.get('query_type') == 'context' and q['hidden_label'] != 'prod'
          for uid in q['relevant_unit_ids']]
    return {'ner': ner, 'el': el, 'ir': ir}


def degradation_rows(root: str) -> tuple:
    # same merge as local_damage.py, prod is outside the evaluated label set
    label = {r['entity_id']: r['label'] for r in jl(root + 'outcomes_test.jsonl')}
    att = {r['entity_id']: r for r in jl(root + 'attribution_test.jsonl')}
    eatt = {}
    if root == R:
        eatt = {r['entity_id']: r for r in jl(root + 'attribution_el_test.jsonl')}
    rows = []
    for r in jl(root + 'iwld_test.jsonl'):
        if label[r['entity_id']] == 'prod':
            continue
        a = att.get(r['entity_id'], {})
        e = eatt.get(r['entity_id'], {})
        r['mention_cer'] = a.get('mention_cer', float('nan'))
        r['edit_lime_damage'] = a.get('edit_lime_damage', float('nan'))
        r['error_span_cer'] = span_error_cer(a.get('edit_spans', []))
        r['el_mention_cer'] = e.get('mention_cer', float('nan'))
        r['el_edit_lime_damage'] = e.get('edit_lime_damage', float('nan'))
        r['el_error_span_cer'] = span_error_cer(e.get('edit_spans', []))
        rows.append(r)
    return [r for r in rows if r['ner_in_eval']], [r for r in rows if r.get('el_in_eval')]


def repair_rows(path: str) -> list:
    return [r for r in jl(path) if r['label'] != 'prod' and r.get('auc_gt_damage') is not None]


def degradation(rows: list, y: str, preds: list) -> dict:
    out = {'n': len(rows), 'failed': sum(r[y] for r in rows)}
    for key, name in preds:
        out[name] = {'auroc': ci(rows, auroc(y, key)), 'ap': ci(rows, ap(y, key))}
    return out


def rec1(rows: list, key: str, ir: bool) -> tuple:
    if ir:
        # pair auc is (max_j + 1 - j) / (max_j + 1) for recovery at repair j, so j <= 1 means auc >= max_j / (max_j + 1)
        return ci(rows, lambda rs: np.mean([r['auc_' + key] >= MAX_J / (MAX_J + 1) - 1e-9 for r in rs]))
    rs = [r for r in rows if r.get('traj_' + key)]
    return ci(rs, lambda xs: np.mean([r['traj_' + key][min(1, len(r['traj_' + key]) - 1)] for r in xs]))


def repair(rows: list, strategies: list, ir: bool = False, with_rec1: bool = True) -> dict:
    out = {'n': len(rows)}
    for key, name in strategies:
        out[name] = {'repauc': ci([r for r in rows if r.get('auc_' + key) is not None], mean('auc_' + key))}
        if with_rec1:
            out[name]['rec1'] = rec1(rows, key, ir)
    return out


def lime_vs_occlusion(task: str, deg: list, rep: list) -> dict:
    y = task + '_failed'
    lime = 'edit_lime_damage' if task == 'ner' else 'el_edit_lime_damage'
    rows = jl(R + 'occlusion_' + task + '_test.jsonl')
    occ = {r['entity_id']: r['occ_damage'] for r in rows}
    traj = {r['entity_id']: r['traj_occlusion_gt'] for r in rows if 'traj_occlusion_gt' in r}
    for r in deg:
        r['occ'] = occ.get(r['entity_id'], 0.0)
    for r in rep:
        r['auc_occ'] = pop_auc(traj[r['entity_id']], MAX_J)
    out = {}
    for name, f, rows, a, b in (('AUROC', lambda k: auroc(y, k), deg, lime, 'occ'),
                                ('AP', lambda k: ap(y, k), deg, lime, 'occ'),
                                ('RepairAUC', mean, rep, 'auc_gt_damage', 'auc_occ')):
        out[name] = {'lime': f(a)(rows), 'occ': f(b)(rows), 'delta': ci(rows, lambda rs: f(a)(rs) - f(b)(rs))}
    return out


def benchmark(ents: dict, ner_bad: set, el_bad: set, pairs: list, q: dict) -> dict:
    table = defaultdict(Counter)
    for eid, e in ents.items():
        if eid not in ner_bad:
            table[e['key']]['ner'] += 1
    for r in jl(A + 'consensus_el_test.jsonl'):
        e = ents.get(r['entity_id'])
        if e and r['status'] == 'accepted' and r['qid'] and r['entity_id'] not in el_bad:
            table[e['key']]['el'] += 1
    queries = defaultdict(set)
    for p in pairs:
        qq = q[p['query_id']]
        if qq['hidden_eid'] in ner_bad:
            continue
        key = COLL[qq['dataset']] + ' ' + qq['language']
        table[key]['pairs'] += 1
        queries[key].add(p['query_id'])
    for k, v in queries.items():
        table[k]['queries'] = len(v)
    out = {k: dict(v) for k, v in sorted(table.items())}
    out['total'] = dict(sum((Counter(v) for v in out.values()), Counter()))
    return out


def gain_rate(rows: list) -> float:
    return round(100 * sum((not r['gt_correct']) and r['ocr_correct'] for r in rows) / len(rows), 1)


def difficulty(ents: dict, out: set) -> dict:
    cers = [jiwer.cer(e['gt_text'], e['ocr_text']) if e['gt_text'] else 0.0
            for eid, e in ents.items() if e['status'] == 'projected' and eid not in out]
    pct = lambda k: round(100 * k / len(cers), 1)
    return {'with errors': pct(sum(c > 0 for c in cers)), '0': pct(sum(c == 0 for c in cers)),
            '0-0.1': pct(sum(0 < c <= 0.1 for c in cers)), '0.1-0.3': pct(sum(0.1 < c <= 0.3 for c in cers)),
            '0.3-0.5': pct(sum(0.3 < c <= 0.5 for c in cers)), 'above 0.5': pct(sum(c > 0.5 for c in cers))}


def passes(rows: list) -> dict:
    # model passes per entity, lime scores every mask up to exact_spans and a sample budget above, occlusion one pass per span
    p = PARAMS['lime']
    m = [len(r['edit_spans']) for r in rows]
    lime = [2 ** x - 1 if x <= p['exact_spans'] else min(p['max_samples'], max(p['min_samples'], p['samples_per_span'] * x, x + 2)) - 1
            for x in m]
    return {'lime': round(float(np.mean(lime)), 1), 'occlusion': round(float(np.mean(m)), 1)}


def fmt(x) -> str:
    if not isinstance(x, tuple):
        return str(round(x, 3))
    point, lo, hi = (round(v, 3) for v in x)
    return str(point) + ' [' + str(lo) + ', ' + str(hi) + ']'


def show(title: str, block: dict):
    print('\n' + title + ' ' + ', '.join(k + '=' + str(v) for k, v in block.items() if not isinstance(v, dict)))
    for name, v in block.items():
        if isinstance(v, dict):
            print(name + ': ' + ' | '.join(m + ' ' + fmt(x) for m, x in v.items()))


if __name__ == '__main__':
    rej = human()
    ner_bad = rej['ner']
    el_bad = rej['el'] | rej['ner']
    # alignment rejects keep the entity annotated but drop it from every evaluated set
    ner_out = rej['ner'] | rej['alignment']
    el_out = rej['el'] | ner_out
    keep = lambda rows, out: [r for r in rows if r['entity_id'] not in out]

    ents = {}
    for r in jl(A + 'projected_test.jsonl'):
        for e in r['entities']:
            if e['label'] != 'prod':
                e['key'] = COLL[r['dataset']] + ' ' + r['language']
                ents[e['entity_id']] = e
    queries = jl(R + 'retrieval_queries_test.jsonl')
    q = {r['query_id']: r for r in queries}
    correct = correctness(queries)
    keep_ir = lambda rows: [r for r in rows if q[r['query_id']]['hidden_eid'] not in ner_out]

    ner_deg, el_deg = degradation_rows(R)
    ner_deg, el_deg = keep(ner_deg, ner_out), keep(el_deg, el_out)
    ir_deg = keep_ir([r for r in jl(R + 'retrieval_qeld_test.jsonl') if q[r['query_id']]['hidden_label'] != 'prod'])
    ner_rep = keep(repair_rows(R + 'repair_test.jsonl'), ner_out)
    el_rep = keep(repair_rows(R + 'repair_el_test.jsonl'), el_out)
    ir_rep = keep_ir(json.load(open(R + 'retrieval_repair_test.json'))['pairs'])

    res = {}
    res['benchmark'] = benchmark(ents, ner_bad, el_bad, correct['ir'], q)
    print('benchmark statistics')
    for k, v in res['benchmark'].items():
        print(k + ': ' + json.dumps(v))

    res['loss'] = {t: {'eval': len(rows), 'lost': f, 'pct': round(100 * f / len(rows), 1)} for t, rows, f in (
        ('ner', ner_deg, sum(r['ner_failed'] for r in ner_deg)),
        ('el', el_deg, sum(r['el_failed'] for r in el_deg)),
        ('ir', ir_deg, sum(r['y'] for r in ir_deg)))}
    print('\ntask degradation: ' + json.dumps(res['loss']))

    res['gains'] = {'ner': gain_rate(keep(correct['ner'], ner_out)), 'el': gain_rate(keep(correct['el'], el_out)),
                    'ir': gain_rate(keep_ir(correct['ir']))}
    print('gains (wrong on GT, correct on OCR, %): ' + json.dumps(res['gains']))
    res['difficulty'] = difficulty(ents, ner_out)
    print('entity CER (%): ' + json.dumps(res['difficulty']))
    res['passes'] = {'ner': passes(keep([r for r in jl(R + 'attribution_test.jsonl') if r['label'] != 'prod'], ner_out)),
                     'el': passes(keep([r for r in jl(R + 'attribution_el_test.jsonl') if r['label'] != 'prod'], el_out))}
    print('model passes per entity: ' + json.dumps(res['passes']))

    res['degradation'] = {'ner': degradation(ner_deg, 'ner_failed', NER_PRED),
                          'el': degradation(el_deg, 'el_failed', EL_PRED),
                          'ir': degradation(ir_deg, 'y', IR_PRED)}
    res['repair'] = {'ner': repair(ner_rep, REPAIR), 'el': repair(el_rep, REPAIR), 'ir': repair(ir_rep, IR_REPAIR, ir=True)}
    for t in ('ner', 'el', 'ir'):
        show('degradation ' + t, res['degradation'][t])
    for t in ('ner', 'el', 'ir'):
        show('repair ' + t, res['repair'][t])

    res['noisy_repair'] = {}
    for rho in RHOS:
        res['noisy_repair'][rho] = {
            'ner': repair(keep(repair_rows(R + 'repair_rho' + rho + '_test.jsonl'), ner_out), REPAIR, with_rec1=False),
            'el': repair(keep(repair_rows(R + 'repair_el_rho' + rho + '_test.jsonl'), el_out), REPAIR, with_rec1=False),
            'ir': repair(keep_ir(json.load(open(R + 'retrieval_repair_rho' + rho + '_test.json'))['pairs']),
                         IR_REPAIR, ir=True, with_rec1=False),
        }
        for t in ('ner', 'el', 'ir'):
            show('noisy repair rho=' + rho + ' ' + t, res['noisy_repair'][rho][t])

    res['occlusion'] = {'ner': lime_vs_occlusion('ner', ner_deg, ner_rep), 'el': lime_vs_occlusion('el', el_deg, el_rep)}
    for t in ('ner', 'el'):
        show('LIME vs occlusion ' + t, res['occlusion'][t])

    # hmbert robustness, the same pipeline with src/ner/hmbert.py as the tagger
    hm_deg = keep(degradation_rows(H)[0], ner_out)
    res['hmbert'] = {'degradation': degradation(hm_deg, 'ner_failed', NER_PRED),
                     'repair': repair(keep(repair_rows(H + 'repair_test.jsonl'), ner_out), REPAIR, with_rec1=False)}
    show('hmBERT degradation', res['hmbert']['degradation'])
    show('hmBERT repair', res['hmbert']['repair'])

    with open(R + 'evaluation_test.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('\nsaved ' + R + 'evaluation_test.json')
