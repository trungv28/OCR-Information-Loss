import json
import sys
from pathlib import Path


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'

    def load_el(path: str) -> dict[str, str | None]:
        result = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                result[row['entity_id']] = row['qid']
        return result

    gpt = load_el(f'data/annotations/gpt_el_{split}.jsonl')
    llama = load_el(f'data/annotations/llama_el_{split}.jsonl')

    all_ids = sorted(set(gpt) | set(llama))

    counts = {'accepted': 0, 'nil': 0, 'conflict': 0, 'missing': 0}
    out_path = Path(f'data/annotations/consensus_el_{split}.jsonl')

    with open(out_path, 'w') as f:
        for eid in all_ids:
            gpt_qid = gpt.get(eid)
            llama_qid = llama.get(eid)

            if eid not in gpt or eid not in llama:
                status = 'missing'
                qid = None
            elif gpt_qid is None and llama_qid is None:
                status = 'nil'
                qid = None
            elif gpt_qid == llama_qid:
                status = 'accepted'
                qid = gpt_qid
            else:
                status = 'conflict'
                qid = None

            counts[status] += 1
            f.write(json.dumps({'entity_id': eid, 'qid': qid, 'status': status}) + '\n')

    total = len(all_ids)
    print(f'Saved to {out_path}')
    print(f'Total: {total}')
    for k, v in counts.items():
        if v:
            print(f'{k}: {v} ({round(100 * v / total, 1)}%)')
