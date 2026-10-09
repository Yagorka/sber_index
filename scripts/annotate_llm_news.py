"""Разметка новостей локальной LLM с возобновлением и сохранением сырых ответов."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
from src.llm_news import PROMPT, FIELDS, select_sample, validate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/llm_news.json')
    parser.add_argument('--base-url')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--limit-batches', type=int, default=0)
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    url = (args.base_url or os.environ.get('LLM_BASE_URL') or cfg['base_url']).rstrip('/')
    folder = ROOT / cfg['output_directory'] / ('full' if args.full else 'pilot')
    folder.mkdir(parents=True, exist_ok=True)
    news_path = ROOT / cfg['news_file']
    news = pd.read_csv(news_path, dtype={'territory_ids': 'string'})
    selected = select_sample(news, cfg, full=args.full)
    selected.to_csv(folder/'selected_news.csv', index=False)
    session = requests.Session()
    session.trust_env = False
    headers = {'Authorization': 'Bearer '+os.environ['LLM_API_KEY']} if os.environ.get('LLM_API_KEY') else {}
    response = session.get(url+'/models', headers=headers, timeout=15)
    response.raise_for_status()
    server = response.json()
    signature = [{'id': m.get('id'), 'meta': m.get('meta'), 'architecture': m.get('architecture')}
                 for m in server.get('data', [])]
    cache_key = hashlib.sha256(json.dumps({'prompt': PROMPT, 'model': cfg['model'], 'server': signature}, sort_keys=True).encode()).hexdigest()
    journal = folder/'labels.jsonl'
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row['cache_key'] == cache_key:
                cached[row['title_sha256']] = row
    pending = selected.drop_duplicates('title_sha256')
    pending = pending[~pending.title_sha256.isin(cached)]
    started = time.time()
    manifest = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'full' if args.full else 'pilot',
                'selected_articles': len(selected), 'unique_titles': selected.title_sha256.nunique(),
                'news_sha256': hashlib.sha256(news_path.read_bytes()).hexdigest(), 'cache_key': cache_key,
                'model': cfg['model'], 'server': server, 'prompt': PROMPT, 'config': cfg,
                'sampling': 'monthly relevant/other strata; deterministic hash rank; inverse probability weights',
                'limitations': 'title_only; historical publication assumption; modern LLM may know later events; no human annotation audit'}
    (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    batch_size = cfg['batch_size']
    print(f"Selected {len(selected)} articles; pending {len(pending)} unique titles; scope {manifest['scope']}", flush=True)
    batches = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending.iloc[offset:offset+batch_size]
        articles = [{'id': i, 'title': row.title} for i, row in enumerate(batch.itertuples())]
        allowed = {'id': list(range(len(batch))), 's': [-1, 0, 1, 2, 9], 'r': [0, 1],
                   'p': [-1, 0, 1, 9], 'i': [-1, 0, 1, 9], 'b': [-1, 0, 1, 9],
                   'c': list(range(64)), 'k': [0, 1]}
        properties = {k: {'type': 'integer', 'enum': values} for k, values in allowed.items()}
        row_schema = {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
        payload = {'model': cfg['model'], 'temperature': cfg['temperature'], 'max_tokens': max(512, len(batch)*90),
                   'messages': [{'role': 'system', 'content': PROMPT},
                                {'role': 'user', 'content': json.dumps(articles, ensure_ascii=False)}],
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'news_labels', 'strict': True,
                       'schema': {'type': 'object', 'properties': {'rows': {'type': 'array',
                                  'minItems': len(batch), 'maxItems': len(batch), 'items': row_schema}},
                                  'required': ['rows'], 'additionalProperties': False}}}}
        rows = None
        errors = []
        for attempt in range(3):
            try:
                reply = session.post(url+'/chat/completions', json=payload, headers=headers, timeout=cfg['timeout_seconds'])
                reply.raise_for_status()
                body = reply.json()
                content = body['choices'][0]['message']['content']
                raw_name = f"response_{cache_key[:8]}_{batch.iloc[0].title_sha256[:12]}_{attempt}.json"
                (folder/raw_name).write_text(json.dumps(body, ensure_ascii=False, indent=2)+'\n')
                parsed = json.loads(content)
                converted = {'rows': [[row[k] for k in allowed] for row in parsed['rows']]}
                rows = validate_rows(converted, len(batch))
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                errors.append(str(exc)[:300])
                time.sleep(1)
        if rows is None:
            raise RuntimeError('Invalid LLM batch; cached results retained: '+str(errors))
        with journal.open('a') as stream:
            for record, label in zip(batch.itertuples(), rows):
                item = {'title_sha256': record.title_sha256, 'cache_key': cache_key,
                        'generated_at_utc': datetime.now(timezone.utc).isoformat(), **label}
                stream.write(json.dumps(item, ensure_ascii=False)+'\n')
                cached[record.title_sha256] = item
        batches += 1
        done = selected.title_sha256.isin(cached).sum()
        status = {'labelled_articles': int(done), 'selected_articles': len(selected), 'elapsed_seconds': time.time()-started,
                  'complete': bool(done == len(selected)), 'invalid_attempts_this_batch': len(errors)}
        (folder/'progress.json').write_text(json.dumps(status, indent=2)+'\n')
        print(f"LLM {done}/{len(selected)}; elapsed {status['elapsed_seconds']:.0f}s", flush=True)
        if args.limit_batches and batches >= args.limit_batches:
            break
    missing = selected[~selected.title_sha256.isin(cached)]
    if len(missing):
        print('Partial cache saved; resume with same command', flush=True)
        return
    labels = pd.DataFrame([cached[key] for key in selected.title_sha256])
    combined = selected.reset_index(drop=True).join(labels.drop(columns=['title_sha256']))
    combined.to_csv(folder/'annotated_news.csv', index=False)
    manifest.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), annotated_sha256=hashlib.sha256((folder/'annotated_news.csv').read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('Annotation complete:', folder/'annotated_news.csv', flush=True)


if __name__ == '__main__':
    main()
