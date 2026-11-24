import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.config import PARAMS, PROMPTS

MODEL = PARAMS['models']['llama']
SYSTEM_PROMPT = PROMPTS['el']['llama_system']
USER_TEMPLATE = PROMPTS['el']['llama_user']
DESC_CHARS = PARAMS['annotation']['description_chars']


def format_candidates(cands: list[dict]) -> str:
    lines = []
    for c in cands:
        desc = c['description'][:DESC_CHARS] if c['description'] else '(no description)'
        lines.append(f"{c['candidate_id']}: {c['qid']} | {c['canonical']} | {desc}")
    return '\n'.join(lines)


def build_prompt(tokenizer, ent: dict) -> str:
    user = USER_TEMPLATE.format(
        mention=ent['mention'],
        label=ent['label'],
        context=ent['context'],
        candidates=format_candidates(ent['candidate_qids']),
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_choice(text: str, valid_ids: set[str]) -> str | None:
    text = text.strip()
    m = re.match(r'^(C\d+|NIL)\b', text, re.IGNORECASE)
    if m:
        cid = m.group(1).upper()
        return cid if cid in valid_ids else None
    return None


if __name__ == '__main__':
    split = 'test' if '--test' in sys.argv else 'dev'
    preview = '--preview' in sys.argv

    cand_path = Path(f'data/annotations/candidates_{split}.jsonl')
    entities = [json.loads(l) for l in open(cand_path)]

    if preview:
        entities = entities[:10]

    # skip entities without candidates
    entities = [e for e in entities if e['candidate_qids']]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, max_model_len=PARAMS['annotation']['llama_max_len_el'], gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(temperature=PARAMS['annotation']['temperature'], max_tokens=PARAMS['annotation']['el_max_tokens_llama'])

    prompts = [build_prompt(tokenizer, e) for e in entities]
    print(f'{len(entities)} entities → generating')

    outputs = llm.generate(prompts, sampling_params)

    results = {}
    for ent, out in zip(entities, outputs):
        valid_ids = {c['candidate_id'] for c in ent['candidate_qids']} | {'NIL'}
        choice = parse_choice(out.outputs[0].text, valid_ids)
        if choice == 'NIL' or choice is None:
            results[ent['entity_id']] = None
        else:
            results[ent['entity_id']] = next(
                (c['qid'] for c in ent['candidate_qids'] if c['candidate_id'] == choice), None
            )

    if preview:
        for ent in entities:
            eid = ent['entity_id']
            print(f"[{ent['label']}] {ent['mention']!r} → {results.get(eid)}")
    else:
        out_path = Path(f'data/annotations/llama_el_{split}.jsonl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for ent in entities:
                row = {'entity_id': ent['entity_id'], 'qid': results.get(ent['entity_id'])}
                f.write(json.dumps(row) + '\n')
        print(f'Saved to {out_path}')
