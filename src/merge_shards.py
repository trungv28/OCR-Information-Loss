import glob
import json
import sys

R = 'data/results/'


def expected_ids(name: str) -> tuple[list, bool]:
    # canonical order is the outcomes file, and only attribution and repair must cover every unit
    split = name.rsplit('_', 1)[1]
    el = name.startswith(('repair_el', 'attribution_el', 'occlusion_el'))
    rows = [json.loads(l) for l in open(R + ('outcomes_el_' if el else 'outcomes_') + split + '.jsonl')]
    rows = [r for r in rows if r['in_eval']]
    if name.startswith('repair'):
        rows = [r for r in rows if r['outcome'] != 'correct']
    return [r['entity_id'] for r in rows], name.startswith(('attribution', 'repair'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.merge_shards <name>, e.g. attribution_test')
    name = sys.argv[1]
    files = sorted(glob.glob(R + name + '.shard*of*.jsonl'))
    if not files:
        sys.exit('no shard files for ' + name)

    rows, conflicts, n_read = {}, [], 0
    for path in files:
        for line in open(path):
            r = json.loads(line)
            n_read += 1
            eid = r['entity_id']
            if eid in rows and json.dumps(rows[eid], sort_keys=True) != json.dumps(r, sort_keys=True):
                conflicts.append(eid)
            rows[eid] = r

    expected, complete = expected_ids(name)
    missing = [e for e in expected if e not in rows]
    print(len(files), 'shards |', n_read, 'rows |', len(rows), 'unique |', len(expected), 'expected |',
          len(missing), 'missing')
    if conflicts:
        sys.exit('the same entity differs across shards: ' + ' '.join(conflicts[:10]))
    if complete and missing:
        sys.exit('not merging, never computed: ' + ' '.join(missing[:20]))

    known = set(expected)
    order = [e for e in expected if e in rows] + [e for e in rows if e not in known]
    with open(R + name + '.jsonl', 'w') as f:
        for eid in order:
            f.write(json.dumps(rows[eid], ensure_ascii=False) + '\n')
    print('wrote', len(order), 'rows to', R + name + '.jsonl')
