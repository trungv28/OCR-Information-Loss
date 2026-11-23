import json
import sys
from pathlib import Path

from src.config import PARAMS
from src.data import load, line_offsets as get_line_offsets

SPLIT = 'test' if '--test' in sys.argv else 'dev'
WINDOW = PARAMS['retrieval']['window']

if __name__ == '__main__':
    out = Path('data/results')
    out.mkdir(parents=True, exist_ok=True)

    units = []
    for rec in load(SPLIT):
        meta = rec['document_metadata']
        doc_id = meta['document_id']
        gt_text = rec['ground_truth']['transcription_unit']
        ocr_text = rec['ocr_hypothesis']['transcription_unit']
        gt_offs, ocr_offs = get_line_offsets(rec)
        n = len(gt_offs)

        n_windows = max(1, n - WINDOW + 1)
        for i in range(n_windows):
            j = min(i + WINDOW, n)
            gt_s = gt_offs[i][0]; gt_e = gt_offs[j - 1][1]
            ocr_s = ocr_offs[i][0]; ocr_e = ocr_offs[j - 1][1]
            units.append({
                'unit_id': doc_id + '_w' + str(i).zfill(4),
                'document_id': doc_id,
                'first_line': i,
                'last_line': j - 1,
                'n_lines': n,
                'split': SPLIT,
                'dataset': meta.get('primary_dataset_name', ''),
                'language': meta.get('language', ''),
                'gt_text': gt_text[gt_s:gt_e],
                'ocr_text': ocr_text[ocr_s:ocr_e],
            })

    outfile = out / f'retrieval_units_{SPLIT}.jsonl'
    with open(outfile, 'w') as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + '\n')

    n_docs = len({u['document_id'] for u in units})
    lens_gt = [len(u['gt_text'].split()) for u in units]
    print(f'Documents: {n_docs}')
    print(f'Units (windows): {len(units)}')
    print(f'Unit GT words: mean={round(sum(lens_gt)/max(len(lens_gt),1), 1)} '
          f'min={min(lens_gt)} max={max(lens_gt)}')
    print(f'Saved -> {outfile}')
