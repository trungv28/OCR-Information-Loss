import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import PARAMS, PROMPTS

MODEL = PARAMS['models']['gpt']
PROMPT = PROMPTS['el']['gpt']
CONTEXT_CHARS = PARAMS['annotation']['context_chars']
DESC_CHARS = PARAMS['annotation']['description_chars']
MAX_RETRY = 3


class ELChoice(BaseModel):
    candidate_id: str = Field(..., description=PROMPTS['el']['gpt_field'])


def format_candidates(cands: list[dict]) -> str:
    lines = []
    for c in cands:
        desc = c['description'][:DESC_CHARS] if c['description'] else '(no description)'
        lines.append(f"{c['candidate_id']}: {c['qid']} | {c['canonical']} | {desc}")
    return '\n'.join(lines)


def link(client: OpenAI, mention: str, label: str, context: str, cands: list[dict]) -> str | None:
    if not cands:
        return None
    prompt = PROMPT.format(
        mention=mention,
        label=label,
        context=context[:CONTEXT_CHARS],
        candidates=format_candidates(cands),
    )
    valid_ids = {c['candidate_id'] for c in cands} | {'NIL'}
    for attempt in range(MAX_RETRY):
        try:
            resp = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                response_format=ELChoice,
                temperature=PARAMS['annotation']['temperature'],
                max_completion_tokens=PARAMS['annotation']['el_max_tokens_gpt'],
            )
            choice = resp.choices[0].message.parsed.candidate_id.strip()
            if choice in valid_ids:
                return choice
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f'[warn] GPT EL failed for {mention!r}: {e}')
    return None


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    preview = '--preview' in sys.argv

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('Set OPENAI_API_KEY environment variable')
    client = OpenAI(api_key=api_key)

    cand_path = Path(f'data/annotations/candidates_{split}.jsonl')
    entities = [json.loads(l) for l in open(cand_path)]

    if preview:
        entities = entities[:10]

    out_path = Path(f'data/annotations/gpt_el_{split}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists() and not preview:
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['entity_id'])

    with open(out_path, 'a') as f:
        for ent in tqdm(entities, desc='gpt-el'):
            eid = ent['entity_id']
            if eid in done:
                continue

            choice = link(client, ent['mention'], ent['label'], ent['context'], ent['candidate_qids'])

            if choice == 'NIL' or choice is None:
                qid = None
            else:
                qid = next((c['qid'] for c in ent['candidate_qids'] if c['candidate_id'] == choice), None)

            row = {'entity_id': eid, 'qid': qid}

            if preview:
                print(f"[{ent['label']}] {ent['mention']!r} → {choice} ({qid})")
            else:
                f.write(json.dumps(row) + '\n')
                f.flush()

    if not preview:
        print(f'Saved to {out_path}')
