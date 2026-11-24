import random
import re
from difflib import SequenceMatcher

import numpy as np

from src.config import PARAMS


def levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for ca in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
    return dp[len(b)]


def extract_edit_spans(gt_win: str, ocr_win: str, ent_s: int, ent_e: int) -> list[dict]:
    '''word-level edit spans, window-relative coords'''
    gw = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', gt_win)]
    ow = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', ocr_win)]
    sm = SequenceMatcher(None, [w for w, _, _ in gw], [w for w, _, _ in ow], autojunk=False)
    spans = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        if op == 'replace':
            gc, oc = gw[i1:i2], ow[j1:j2]
            gt_t = ' '.join(w for w, _, _ in gc)
            oc_t = ' '.join(w for w, _, _ in oc)
            gs, ge = gc[0][1], gc[-1][2]
            os_, oe = oc[0][1], oc[-1][2]
            D = levenshtein(gt_t, oc_t) / max(len(gt_t), len(oc_t), 1)
        elif op == 'delete':
            gc = gw[i1:i2]
            gt_t = ' '.join(w for w, _, _ in gc)
            oc_t = ''
            gs, ge = gc[0][1], gc[-1][2]
            os_ = ow[j1][1] if j1 < len(ow) else (ow[j1 - 1][2] if j1 > 0 else 0)
            oe = os_
            D = 1.0
        else:  # insert
            oc = ow[j1:j2]
            gt_t = ''
            oc_t = ' '.join(w for w, _, _ in oc)
            gs = gw[i1][1] if i1 < len(gw) else (gw[i1 - 1][2] if i1 > 0 else 0)
            ge = gs
            os_, oe = oc[0][1], oc[-1][2]
            D = 1.0
        loc = 'mention' if (gs < ent_e and ge > ent_s) else 'context'
        spans.append({
            'gt_text': gt_t, 'ocr_text': oc_t,
            'gt_start': gs, 'gt_end': ge,
            'ocr_start': os_, 'ocr_end': oe,
            'D': round(D, 4), 'location_type': loc,
        })
    return sorted(spans, key=lambda x: x['gt_start'])


def corrupt_text(ocr_win: str, sp: dict) -> str:
    '''ocr form of one span, spliced as a substring so the spacing survives'''
    if sp['ocr_start'] is None:
        repl = sp['ocr_text']
    else:
        repl = ocr_win[sp['ocr_start']:sp['ocr_end']]
    if sp['gt_end'] == sp['gt_start']:
        repl += ' '  # ocr added a word here, keep it a separate token
    return repl


def apply_corruptions(gt_win: str, ocr_win: str, spans: list[dict], z: np.ndarray) -> str:
    result = gt_win
    offset = 0
    for i, sp in enumerate(spans):
        if not z[i]:
            continue
        s = sp['gt_start'] + offset
        e = sp['gt_end'] + offset
        repl = corrupt_text(ocr_win, sp)
        result = result[:s] + repl + result[e:]
        offset += len(repl) - (sp['gt_end'] - sp['gt_start'])
    return result


def track_entity(ocr_win: str, spans: list[dict], z: np.ndarray, ent_s: int, ent_e: int) -> tuple[int, int]:
    '''where the entity ends up after corruption, in gt-window coords'''
    new_s, new_e = ent_s, ent_e
    for i, sp in enumerate(spans):
        if not z[i]:
            continue
        delta = len(corrupt_text(ocr_win, sp)) - (sp['gt_end'] - sp['gt_start'])
        if sp['gt_end'] <= ent_s:
            new_s += delta; new_e += delta
        elif sp['gt_start'] < ent_e:
            new_e += delta
    return max(0, new_s), max(0, new_e)


def degrade(text: str, rho: float, key: str) -> str:
    '''ocr-like noise an imperfect corrector leaves behind'''
    if not text or rho <= 0:
        return text
    rng = random.Random(str(PARAMS['seed']) + '|' + key)
    alpha = sorted(set(text)) or list('abcdefghijklmnopqrstuvwxyz')
    out = []
    for ch in text:
        if rng.random() >= rho:
            out.append(ch)
            continue
        op = rng.choice(('sub', 'del', 'ins'))
        if op == 'sub':
            out.append(rng.choice(alpha))
        elif op == 'ins':
            out.append(ch)
            out.append(rng.choice(alpha))
    return ''.join(out)


def apply_repairs(ocr_win: str, gt_win: str, spans: list[dict], r: np.ndarray,
                  orig_ent_s: int, orig_ent_e: int, rho: float = 0.0) -> tuple[str, int, int]:
    '''mirror of apply_corruptions, also returns entity coords, in ocr-window coords'''
    result = ocr_win
    new_s, new_e = orig_ent_s, orig_ent_e
    offset = 0
    for i, sp in enumerate(spans):
        if not r[i]:
            continue
        os_, oe = sp['ocr_start'], sp['ocr_end']
        if os_ is None:
            continue
        s, e = os_ + offset, oe + offset
        # splice the gt substring itself, not a re-join, so spacing survives
        repl = gt_win[sp['gt_start']:sp['gt_end']]
        # an imperfect repair, seeded on the span so every strategy gets the same fix
        repl = degrade(repl, rho, str(sp['gt_start']) + ':' + str(sp['gt_end']) + ':' + repl)
        if oe == os_:
            repl += ' '  # ocr dropped this word, put the separator back
        result = result[:s] + repl + result[e:]
        delta = len(repl) - (oe - os_)
        offset += delta
        if oe <= orig_ent_s:
            new_s += delta; new_e += delta
        elif os_ < orig_ent_e:
            new_e += delta
    return result, max(0, new_s), max(new_s, new_e)


def pop_auc(traj: list, k_max: int) -> float:
    # per-item repair auc, traj stays flat past its end
    last = len(traj) - 1
    return sum(traj[min(k, last)] for k in range(k_max + 1)) / (k_max + 1)


def rank_indices(spans, strategy, local=None, damages=None, rng=None):
    # repair order for one strategy
    m = len(spans)
    if strategy == 'local_damage':
        return sorted(range(m), key=lambda i: local[i] if local is not None else 0, reverse=True)
    if strategy == 'gt_damage':
        return sorted(range(m), key=lambda i: damages[i] if damages is not None else 0, reverse=True)
    if strategy == 'corruption':
        return sorted(range(m), key=lambda i: spans[i]['D'], reverse=True)
    if strategy == 'mention_first':
        return sorted(range(m), key=lambda i: (0 if spans[i]['location_type'] == 'mention' else 1, -spans[i]['D']))
    idx = list(range(m))
    rng.shuffle(idx)
    return idx
