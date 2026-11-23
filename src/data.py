import json
import sys
from pathlib import Path

DATA_DIR = Path('data/raw/data/v0.9')

PUNKT_CACHE = {}
LANG_MAP = {'en': 'english', 'de': 'german', 'fr': 'french'}


def load(split='train'):
    records = []
    for path in sorted(DATA_DIR.rglob('*.jsonl')):
        name = path.stem
        if 'masked' in name:
            continue
        if 'dta19' in name and 'unmatched' not in name:
            continue
        if f'_{split}_' not in name and f'_{split}-unmatched_' not in name:
            continue
        with open(path) as f:
            records.extend(json.loads(line) for line in f)
    return records


def label(rec):
    meta = rec['document_metadata']
    return f"{meta['primary_dataset_name']} ({meta['language']})"


def sentence_window(text, start, end, lang='en'):
    '''sentence span covering [start, end], merged if it straddles a boundary, else +-150 chars'''
    lang_name = LANG_MAP.get(lang, 'english')
    if lang_name not in PUNKT_CACHE:
        import nltk
        loaded = False
        for resource in (
            f'tokenizers/punkt_tab/{lang_name}.pickle',
            f'tokenizers/punkt/{lang_name}.pickle',
        ):
            try:
                PUNKT_CACHE[lang_name] = nltk.data.load(resource)
                loaded = True
                break
            except LookupError:
                continue
        if not loaded:
            nltk.download('punkt_tab', quiet=True)
            PUNKT_CACHE[lang_name] = nltk.data.load(
                f'tokenizers/punkt_tab/{lang_name}.pickle'
            )

    spans = list(PUNKT_CACHE[lang_name].span_tokenize(text))

    ws, we = None, None
    for s, e in spans:
        if s <= start and end <= e:
            ws, we = s, e
            break

    if ws is None:
        overlapping = [(s, e) for s, e in spans if s < end and e > start]
        if overlapping:
            ws, we = overlapping[0][0], overlapping[-1][1]
        else:
            # fallback: ±150 chars, word-boundary aligned
            ws = max(0, start - 150)
            we = min(len(text), end + 150)
            while ws > 0 and text[ws] not in (' ', '\n'):
                ws -= 1
            while we < len(text) and text[we] not in (' ', '\n'):
                we += 1
            return ws, we

    # cap very long sentences at 600 chars, centered on the entity
    if we - ws > 600:
        center = (start + end) // 2
        lo = max(ws, center - 300)
        hi = min(we, lo + 600)
        lo = max(ws, hi - 600)
        while lo > ws and text[lo] != ' ':
            lo -= 1
        while hi < we and text[hi] != ' ':
            hi += 1
        ws, we = lo, hi

    return ws, we


def line_offsets(rec):
    '''(gt_offsets, ocr_offsets) as (start, end) char pairs'''
    gt_text = rec['ground_truth']['transcription_unit']
    ocr_text = rec['ocr_hypothesis']['transcription_unit']
    gt_offs = rec['ground_truth']['line_offsets']
    ocr_offs = rec['ocr_hypothesis']['line_offsets']
    if gt_offs and ocr_offs and len(gt_offs) == len(ocr_offs):
        return gt_offs, ocr_offs
    def offsets(text):
        offs, pos = [], 0
        for part in text.split('\n'):
            offs.append((pos, pos + len(part)))
            pos += len(part) + 1
        return offs
    gt_o = offsets(gt_text)
    ocr_o = offsets(ocr_text)
    if len(gt_o) != len(ocr_o):
        return [(0, len(gt_text))], [(0, len(ocr_text))]
    return gt_o, ocr_o


def line_window(rec, gt_start, gt_end, k=2):
    '''window bounds for +-k lines around the entity'''
    gt_offs, ocr_offs = line_offsets(rec)
    if not gt_offs or not ocr_offs or len(gt_offs) != len(ocr_offs):
        gt_text = rec['ground_truth']['transcription_unit']
        ocr_text = rec['ocr_hypothesis']['transcription_unit']
        return 0, len(gt_text), 0, len(ocr_text)
    center, best_ovlp = 0, -1
    for i, (s, e) in enumerate(gt_offs):
        ovlp = max(0, min(e, gt_end) - max(s, gt_start))
        if ovlp > best_ovlp or (ovlp == best_ovlp and s <= gt_start):
            best_ovlp, center = ovlp, i
    lo = max(0, center - k)
    hi = min(len(gt_offs) - 1, center + k)
    return gt_offs[lo][0], gt_offs[hi][1], ocr_offs[lo][0], ocr_offs[hi][1]


def lines(rec):
    gt_text = rec['ground_truth']['transcription_unit']
    ocr_text = rec['ocr_hypothesis']['transcription_unit']
    gt_offs, ocr_offs = line_offsets(rec)
    return [(gt_text[s1:e1], ocr_text[s2:e2]) for (s1, e1), (s2, e2) in zip(gt_offs, ocr_offs)]


if __name__ == '__main__':
    records = load('train')

    rec = records[0]
    meta = rec['document_metadata']
    print('date:', meta['date'])
    print('scope:', meta['transcription_unit_scope'])

    print(rec['ground_truth']['transcription_unit'][:121])
    print(rec['ocr_hypothesis']['transcription_unit'][:121])

    rec_lines = lines(rec)
    for gt_line, ocr_line in rec_lines[:11]:
        print('gt:', gt_line)
        print('ocr:', ocr_line)
        print()


def shard_arg():
    # --shard i/n -> (i, n), None when the run is not sharded
    for j, a in enumerate(sys.argv):
        if a == '--shard' and j + 1 < len(sys.argv):
            i, n = sys.argv[j + 1].split('/')
            return int(i), int(n)
    return None


def rho_arg() -> float:
    # --rho=0.1 leaves residual noise in every repair, 0 gives exact gt repairs
    return float(next((a.split('=')[1] for a in sys.argv if a.startswith('--rho=')), 0.0))
