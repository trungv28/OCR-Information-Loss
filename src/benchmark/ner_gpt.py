import os
import sys
from typing import Literal

from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel, Field

from src.config import PARAMS, PROMPTS
from src.data import load

MODEL = PARAMS['models']['gpt']
PROMPT = PROMPTS['ner']['gpt']
FIELDS = PROMPTS['ner']['gpt_fields']
CHUNK_SIZE = PARAMS['annotation']['chunk_chars']['gpt']


EntityLabel = Literal['pers', 'org', 'prod', 'time', 'loc']


class EntityMention(BaseModel):
    text: str = Field(..., description=FIELDS['text'])
    label: EntityLabel = Field(..., description=FIELDS['label'])


class NERAnnotation(BaseModel):
    entities: list[EntityMention] = Field(default_factory=list)


def resolve_offsets(text: str, entities: list[EntityMention]) -> list[tuple[str, str, int, int]]:
    '''char offsets of each span, searched in reading order'''
    results = []
    cursor = 0
    for ent in entities:
        idx = text.find(ent.text, cursor)
        if idx == -1:
            continue
        results.append((ent.text, ent.label, idx, idx + len(ent.text)))
        cursor = idx + len(ent.text)
    return results


def split_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    '''chunks cut at newlines, else at spaces'''
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start, end)
            if boundary <= start:
                boundary = text.rfind(' ', start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        start = end
    return chunks


def annotate(client: OpenAI, text: str) -> NERAnnotation:
    all_entities = []
    for part in split_text(text):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[{'role': 'user', 'content': PROMPT.replace('{input_text}', part)}],
                response_format=NERAnnotation,
                temperature=PARAMS['annotation']['temperature'],
                max_completion_tokens=PARAMS['annotation']['ner_max_tokens'],
            )
            all_entities.extend(response.choices[0].message.parsed.entities)
        except LengthFinishReasonError:
            print(f'\n[warn] chunk truncated ({len(part)} chars), skipping')
    return NERAnnotation(entities=all_entities)


if __name__ == '__main__':
    import json
    from pathlib import Path
    from tqdm import tqdm

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('Set OPENAI_API_KEY environment variable')

    client = OpenAI(api_key=api_key)

    split = 'test' if '--test' in sys.argv else 'dev'
    out_path = Path(f'data/annotations/gpt41mini_{split}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)['document_id'])

    records = [
        rec for rec in load(split)
        if rec['ground_truth']['transcription_unit'].strip()
        and rec['document_metadata']['document_id'] not in done
    ]

    with open(out_path, 'a') as f:
        for rec in tqdm(records, desc='annotating'):
            meta = rec['document_metadata']
            doc_id = meta['document_id']

            gt = rec['ground_truth']['transcription_unit']
            result = annotate(client, gt)
            resolved = resolve_offsets(gt, result.entities)

            row = {
                'document_id': doc_id,
                'dataset': meta['primary_dataset_name'],
                'language': meta['language'],
                'date': meta['date'],
                'entities': [
                    {'text': span, 'label': label, 'start': start, 'end': end}
                    for span, label, start, end in resolved
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
