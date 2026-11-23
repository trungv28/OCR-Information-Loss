import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.benchmark.ner_gpt import NERAnnotation, resolve_offsets, split_text
from src.config import PARAMS, PROMPTS
from src.data import load

MODEL = PARAMS['models']['llama']
CHUNK_SIZE = PARAMS['annotation']['chunk_chars']['llama']
SYSTEM_PROMPT = PROMPTS['ner']['llama_system']
USER_TEMPLATE = PROMPTS['ner']['llama_user']


def format_prompt(tokenizer: AutoTokenizer, text: str) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_TEMPLATE.replace('{input_text}', text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == '__main__':
    preview = '--preview' in sys.argv

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, max_model_len=PARAMS['annotation']['llama_max_len_ner'], gpu_memory_utilization=0.90)

    sampling_params = SamplingParams(temperature=PARAMS['annotation']['temperature'], max_tokens=PARAMS['annotation']['ner_max_tokens'])

    split = 'test' if '--test' in sys.argv else 'dev'
    records = [
        rec for rec in load(split)
        if rec['ground_truth']['transcription_unit'].strip()
    ]

    if preview:
        records = [r for r in records if r['document_metadata']['language'] == 'en'][:5]

    prompts = []
    index = []
    for rec_idx, rec in enumerate(records):
        gt = rec['ground_truth']['transcription_unit']
        for chunk_idx, chunk in enumerate(split_text(gt, CHUNK_SIZE)):
            prompts.append(format_prompt(tokenizer, chunk))
            index.append((rec_idx, chunk_idx))

    print(f'{len(records)} records → {len(prompts)} chunks')

    outputs = llm.generate(prompts, sampling_params)

    record_entities = {i: [] for i in range(len(records))}
    for (rec_idx, _), out in zip(index, outputs):
        parsed = NERAnnotation.model_validate_json(out.outputs[0].text)
        record_entities[rec_idx].extend(parsed.entities)

    if preview:
        for rec_idx, rec in enumerate(records):
            meta = rec['document_metadata']
            gt = rec['ground_truth']['transcription_unit']
            print(f"\n--- Sample {rec_idx+1} | {meta['primary_dataset_name']} ({meta['language']}) | {meta['date']} ---")
            print(f'TEXT: {gt}')
            entities = record_entities[rec_idx]
            if entities:
                for span, label, start, end in resolve_offsets(gt, entities):
                    match = 'OK' if gt[start:end] == span else 'MISMATCH'
                    print(f'[{label}] {span!r} ({start}:{end}) {match}')
            else:
                print('(no entities)')
    else:
        out_path = Path(f'data/annotations/llama31_8b_{split}.jsonl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for rec_idx, rec in enumerate(records):
                meta = rec['document_metadata']
                gt = rec['ground_truth']['transcription_unit']
                resolved = resolve_offsets(gt, record_entities[rec_idx])
                row = {
                    'document_id': meta['document_id'],
                    'dataset': meta['primary_dataset_name'],
                    'language': meta['language'],
                    'date': meta['date'],
                    'entities': [
                        {'text': span, 'label': label, 'start': start, 'end': end}
                        for span, label, start, end in resolved
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Saved to {out_path}')
