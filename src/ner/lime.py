import itertools
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import torch
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, line_window, shard_arg
from src.ner.outcomes import span_overlap
from src.ner.tagger import MODELS, run, predict_batch, load_model
from src.spans import extract_edit_spans, apply_corruptions, track_entity, levenshtein

N_CAP = PARAMS['lime']['max_samples']
MIN_SAMPLES = PARAMS['lime']['min_samples']
PER_SPAN = PARAMS['lime']['samples_per_span']
SIGMA = PARAMS['lime']['sigma']
EXACT_THRESH = PARAMS['lime']['exact_spans']
BATCH_SIZE = 64


def ner_score_from_preds(preds: list[dict], ent_s: int, ent_e: int, label: str) -> float:
    best = 0.0
    for p in preds:
        if p['label'] != label:
            continue
        ovlp = span_overlap(p['start'], p['end'], ent_s, ent_e)
        if ovlp > 0:
            best = max(best, p['score'] * ovlp)
    return best


def ner_score(tagger, text: str, ent_s: int, ent_e: int, label: str) -> float:
    return ner_score_from_preds(run(tagger, text, 0), ent_s, ent_e, label)


def fit_weighted_ridge(X: list, y: list, w: list, K: int) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X)), np.array(X)])
    ya, wa = np.array(y), np.array(w)
    WX = Xa.T * wa
    try:
        coef = np.linalg.solve(WX @ Xa + 1e-3 * np.eye(K + 1), WX @ ya)
    except np.linalg.LinAlgError:
        coef = np.zeros(K + 1)
    return coef[1:]


def gt_anchored_lime(tagger, gt_win, ocr_win, spans, ent_s, ent_e, label, score_gt, seed) -> np.ndarray:
    m = len(spans)
    if m <= EXACT_THRESH:
        zs_to_score = [np.array(c, dtype=float)
                       for c in itertools.product([0, 1], repeat=m)
                       if any(c)]  # skip all-zeros (anchor)
    else:
        N = min(N_CAP, max(MIN_SAMPLES, PER_SPAN * m, m + 2))
        rng = np.random.default_rng(seed)
        zs_to_score = [np.ones(m)] + [rng.integers(0, 2, m).astype(float) for _ in range(N - 2)]

    texts = [apply_corruptions(gt_win, ocr_win, spans, z) for z in zs_to_score]
    coords = [track_entity(ocr_win, spans, z, ent_s, ent_e) for z in zs_to_score]
    all_preds = predict_batch(tagger, texts)

    X = [np.zeros(m)]; y = [score_gt]; w = [1.0]
    for z, (ns, ne), preds in zip(zs_to_score, coords, all_preds):
        score = ner_score_from_preds(preds, ns, ne, label)
        d = float(z.sum()) / m
        X.append(z); y.append(score); w.append(np.exp(-(d ** 2) / SIGMA ** 2))

    return fit_weighted_ridge(X, y, w, m)


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} | split={split}")

    ner = load_model()
    taggers = {lang: ner for lang in MODELS}

    outcomes = {}
    with open(f'data/results/outcomes_{split}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['in_eval']:
                outcomes[r['entity_id']] = r

    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(split)}

    shard = shard_arg()
    suffix = f'.shard{shard[0]}of{shard[1]}' if shard else ''
    out_path = Path(f'data/results/attribution_{split}{suffix}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    # shard the full list before dropping done ones, so a restart keeps each shard's entities
    entities = list(outcomes.items())
    if shard:
        entities = entities[shard[0]::shard[1]]
    entities = [(eid, r) for eid, r in entities if eid not in done]

    with open(out_path, 'a') as f:
        for eid, ent in tqdm(entities, desc='attribute'):
            doc_id = ent['document_id']
            lang = ent.get('language', 'en')
            if lang not in taggers or doc_id not in text_by_id:
                continue

            rec = text_by_id[doc_id]
            gt_text = rec['ground_truth']['transcription_unit']
            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            gt_start, gt_end = ent['gt_start'], ent['gt_end']
            ocr_start, ocr_end = ent.get('ocr_start'), ent.get('ocr_end')

            gt_ws, gt_we, ocr_ws, ocr_we = line_window(rec, gt_start, gt_end)
            gt_win = gt_text[gt_ws:gt_we]
            ocr_win = ocr_text[ocr_ws:ocr_we]

            ent_s = gt_start - gt_ws
            ent_e = gt_end - gt_ws
            ent_ocr_s = (ocr_start - ocr_ws) if ocr_start is not None else None
            ent_ocr_e = (ocr_end - ocr_ws) if ocr_end is not None else None

            spans = extract_edit_spans(gt_win, ocr_win, ent_s, ent_e)
            m = len(spans)
            tagger = taggers[lang]

            score_gt = ner_score(tagger, gt_win, ent_s, ent_e, ent['label'])

            # predictors
            window_cer = jiwer.cer(gt_win, ocr_win) if gt_win else 0.0
            gt_mention = gt_win[ent_s:ent_e]
            ocr_mention = ocr_win[ent_ocr_s:ent_ocr_e] if (ent_ocr_s is not None and ent_ocr_e is not None) else ''
            mention_cer = levenshtein(gt_mention, ocr_mention) / max(len(gt_mention), 1)

            gt_ctx = gt_win[:ent_s] + gt_win[ent_e:]
            if ent_ocr_s is not None and ent_ocr_e is not None:
                ocr_ctx = ocr_win[:ent_ocr_s] + ocr_win[ent_ocr_e:]
                context_cer = levenshtein(gt_ctx, ocr_ctx) / max(len(gt_ctx), 1)
            else:
                context_cer = window_cer

            corruption_only = float(sum(sp['D'] for sp in spans)) if spans else 0.0

            if m == 0:
                betas = np.array([])
                importance_only = 0.0
                edit_lime_damage = 0.0
            else:
                betas = gt_anchored_lime(tagger, gt_win, ocr_win, spans, ent_s, ent_e, ent['label'], score_gt, PARAMS['seed'])
                damages = np.maximum(0.0, -betas)
                importance_only = float(np.abs(betas).sum())
                edit_lime_damage = float(damages.sum())

            edit_lime_damage_norm = edit_lime_damage / max(score_gt, 1e-9)

            out_spans = []
            for i, sp in enumerate(spans):
                beta_i = float(betas[i]) if i < len(betas) else 0.0
                damage_i = max(0.0, -beta_i)
                out_spans.append({
                    **sp,
                    'gt_start_abs': gt_ws + sp['gt_start'],
                    'gt_end_abs': gt_ws + sp['gt_end'],
                    'ocr_start_abs': ocr_ws + sp['ocr_start'],
                    'ocr_end_abs': ocr_ws + sp['ocr_end'],
                    'beta': round(beta_i, 6),
                    'damage': round(damage_i, 6),
                })

            f.write(json.dumps({
                'entity_id': eid,
                'document_id': doc_id,
                'label': ent['label'],
                'outcome': ent['outcome'],
                'gt_win_start': gt_ws, 'gt_win_end': gt_we,
                'ocr_win_start': ocr_ws, 'ocr_win_end': ocr_we,
                'ent_win_start': ent_s, 'ent_win_end': ent_e,
                'ent_ocr_win_start': ent_ocr_s, 'ent_ocr_win_end': ent_ocr_e,
                'score_gt': round(score_gt, 4),
                'window_cer': round(window_cer, 4),
                'mention_cer': round(mention_cer, 4),
                'context_cer': round(context_cer, 4),
                'corruption_only': round(corruption_only, 4),
                'importance_only': round(importance_only, 4),
                'edit_lime_damage': round(edit_lime_damage, 4),
                'edit_lime_damage_norm': round(edit_lime_damage_norm, 4),
                'edit_spans': out_spans,
            }, ensure_ascii=False) + '\n')
            f.flush()

    print(f'Saved to {out_path}')
