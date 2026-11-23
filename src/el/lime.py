import itertools
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import torch
from tqdm import tqdm

from src.config import PARAMS
from src.data import load, line_window
from src.el.model import BEAM, MAX_NEW_TOKENS, load_linker, mark_entity, fit_marker, title_to_qid, decoded_title
from src.ner.lime import fit_weighted_ridge
from src.spans import extract_edit_spans, apply_corruptions, track_entity, levenshtein

SPLIT = 'test' if '--test' in sys.argv else 'dev'
N_CAP = PARAMS['lime']['max_samples']
MIN_SAMPLES = PARAMS['lime']['min_samples']
PER_SPAN = PARAMS['lime']['samples_per_span']
SIGMA = PARAMS['lime']['sigma']
EXACT_THRESH = PARAMS['lime']['exact_spans']
BATCH_SIZE = 16


def el_score_batch(model, tokenizer, texts: list[str], ref_qid: str, lang: str,
                   title2qid: dict, device) -> list[float]:
    texts = [fit_marker(x, tokenizer) for x in texts]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                    max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, num_beams=BEAM, num_return_sequences=BEAM,
            max_new_tokens=MAX_NEW_TOKENS, output_scores=True,
            return_dict_in_generate=True,
        )
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    seq_scores = out.sequences_scores.cpu().tolist()
    scores = []
    for b in range(len(texts)):
        best = None
        for k in range(BEAM):
            idx = b * BEAM + k
            if title_to_qid(title2qid, decoded_title(decoded[idx]), lang) == ref_qid:
                if best is None or seq_scores[idx] > best:
                    best = seq_scores[idx]
        scores.append(float(np.exp(best)) if best is not None else 0.0)
    return scores


def gt_anchored_lime_el(model, tokenizer, title2qid, device,
                        gt_win, ocr_win, spans, ent_s, ent_e, ref_qid, lang,
                        score_gt, seed) -> np.ndarray:
    m = len(spans)
    if m <= EXACT_THRESH:
        zs_to_score = [np.array(c, dtype=float)
                       for c in itertools.product([0, 1], repeat=m)
                       if any(c)]
    else:
        N = min(N_CAP, max(MIN_SAMPLES, PER_SPAN * m, m + 2))
        rng = np.random.default_rng(seed)
        zs_to_score = [np.ones(m)] + [rng.integers(0, 2, m).astype(float) for _ in range(N - 2)]

    marked_texts = []
    dists = []
    coords = []
    for z in zs_to_score:
        text = apply_corruptions(gt_win, ocr_win, spans, z)
        ns, ne = track_entity(ocr_win, spans, z, ent_s, ent_e)
        marked_texts.append(mark_entity(text, ns, ne))
        dists.append(float(z.sum()) / m)
        coords.append((ns, ne))

    scores = []
    for i in range(0, len(marked_texts), BATCH_SIZE):
        scores.extend(el_score_batch(model, tokenizer, marked_texts[i:i + BATCH_SIZE],
                                     ref_qid, lang, title2qid, device))

    X = [np.zeros(m)]; y = [score_gt]; w = [1.0]
    for z, d, score in zip(zs_to_score, dists, scores):
        X.append(z); y.append(score); w.append(np.exp(-(d ** 2) / SIGMA ** 2))

    return fit_weighted_ridge(X, y, w, m)


if __name__ == '__main__':
    title2qid, tokenizer, model, device = load_linker()

    el_outcomes = {}
    with open(f'data/results/outcomes_el_{SPLIT}.jsonl') as fh:
        for line in fh:
            r = json.loads(line)
            if r['in_eval']:
                el_outcomes[r['entity_id']] = r

    ner_outcomes = {}
    with open(f'data/results/outcomes_{SPLIT}.jsonl') as fh:
        for line in fh:
            r = json.loads(line)
            ner_outcomes[r['entity_id']] = r

    consensus = {}
    with open(f'data/annotations/consensus_el_{SPLIT}.jsonl') as fh:
        for line in fh:
            r = json.loads(line)
            consensus[r['entity_id']] = r

    text_by_id = {rec['document_metadata']['document_id']: rec for rec in load(SPLIT)}

    out_path = Path(f'data/results/attribution_el_{SPLIT}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                done.add(json.loads(line)['entity_id'])

    entities = [(eid, r) for eid, r in el_outcomes.items() if eid not in done]

    with open(out_path, 'a') as fh:
        for eid, el_ent in tqdm(entities, desc='attribute-el'):
            ref_qid = consensus.get(eid, {}).get('qid')
            if not ref_qid:
                continue

            ner_ent = ner_outcomes.get(eid, {})
            doc_id = ner_ent.get('document_id') or eid.rsplit('_E', 1)[0]
            lang = el_ent.get('language') or ner_ent.get('language', 'en')
            rec = text_by_id.get(doc_id)
            if not rec:
                continue

            gt_text = rec['ground_truth']['transcription_unit']
            ocr_text = rec['ocr_hypothesis']['transcription_unit']
            gt_start = ner_ent.get('gt_start'); gt_end = ner_ent.get('gt_end')
            if gt_start is None:
                continue
            ocr_start = ner_ent.get('ocr_start'); ocr_end = ner_ent.get('ocr_end')

            gt_ws, gt_we, ocr_ws, ocr_we = line_window(rec, gt_start, gt_end)
            gt_win = gt_text[gt_ws:gt_we]
            ocr_win = ocr_text[ocr_ws:ocr_we]

            ent_s = gt_start - gt_ws
            ent_e = gt_end - gt_ws
            ent_ocr_s = (ocr_start - ocr_ws) if ocr_start is not None else None
            ent_ocr_e = (ocr_end - ocr_ws) if ocr_end is not None else None

            spans = extract_edit_spans(gt_win, ocr_win, ent_s, ent_e)
            m = len(spans)

            # baseline score on clean gt
            gt_marked = mark_entity(gt_win, ent_s, ent_e)
            score_gt = el_score_batch(model, tokenizer, [gt_marked], ref_qid, lang, title2qid, device)[0]

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
                betas = gt_anchored_lime_el(model, tokenizer, title2qid, device,
                                            gt_win, ocr_win, spans, ent_s, ent_e, ref_qid, lang, score_gt, 0)
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

            fh.write(json.dumps({
                'entity_id': eid,
                'document_id': doc_id,
                'label': el_ent.get('label'),
                'outcome': el_ent.get('outcome'),
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
            fh.flush()

    print(f'Saved to {out_path}')
