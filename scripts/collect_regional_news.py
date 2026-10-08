"""Обработка поисковой выборки МЧС; --refresh уточняет метаданные известных URL."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.news_collection import metadata_record, monthly_counts, parse_publication, MONTHS_RU


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.title_parts = [], []
        self.in_h1 = False
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style']:
            self.hidden += 1
        if tag == 'h1':
            self.in_h1 = True

    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.hidden = max(0, self.hidden - 1)
        if tag == 'h1':
            self.in_h1 = False

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text.strip())
            if self.in_h1:
                self.title_parts.append(text.strip())


def read_article(html, url):
    parser = ArticleParser()
    parser.feed(html)
    published = parse_publication(' '.join(parser.parts))
    title = ' '.join(parser.title_parts).strip()
    if not title:
        raise ValueError(f'Не найден h1: {url}')
    return {'url': url, 'title': title,
            'published_text': f'{published.day} {MONTHS_RU[published.month-1]} {published.year}, {published:%H:%M}',
            'metadata_source': 'direct_article_html'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/news_orenburg.json')
    parser.add_argument('--refresh', action='store_true', help='HTTP для известных URL, один процесс, кеш и паузы')
    parser.add_argument('--print-queries', action='store_true', help='Показать фиксированные запросы по всем месяцам')
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = json.loads(config_path.read_text())
    if args.print_queries:
        for month in pd.period_range(cfg['period_start'], cfg['period_end'], freq='M'):
            print(cfg['query_template'].format(month_ru=MONTHS_RU[month.month - 1], year=month.year))
        return
    discovery_path = ROOT / cfg['discovery_file']
    if not discovery_path.exists():
        raise SystemExit('Нужна локальная поисковая выборка: ' + str(discovery_path) + '. См. docs/REGIONAL_NEWS_COLLECTION_RU.md')
    discovery = json.loads(discovery_path.read_text())
    dictionary = pd.read_excel(ROOT / 'data/inputs/municipal_dictionary.xlsx')
    now = datetime.now(timezone.utc).isoformat()
    previous_path = ROOT / cfg['output_file']
    previous = pd.read_csv(previous_path).set_index('url') if previous_path.exists() else None
    rows, rejected = [], []
    cache_dir = ROOT / cfg['cache_directory']
    requests_made = 0
    for raw in discovery['records']:
        raw = dict(raw)
        if urlparse(raw['url']).netloc != urlparse(cfg['source']).netloc:
            rejected.append({'url': raw['url'], 'reason': 'wrong_source_host'})
            continue
        if args.refresh:
            cache_dir.mkdir(exist_ok=True)
            cache = cache_dir / (hashlib.sha256(raw['url'].encode()).hexdigest() + '.json')
            try:
                if cache.exists():
                    raw = json.loads(cache.read_text())
                elif requests_made < cfg['max_refresh_articles']:
                    if requests_made:
                        time.sleep(cfg['request_pause_seconds'])
                    requests_made += 1
                    request = Request(raw['url'], headers={'User-Agent': 'SberIndexResearchPilot/1.0'})
                    with urlopen(request, timeout=cfg['timeout_seconds']) as response:
                        html = response.read(2_000_000).decode('utf-8')
                    raw = read_article(html, raw['url'])
                    cache.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
            except Exception as exc:
                rejected.append({'url': raw['url'], 'reason': 'refresh_failed:' + type(exc).__name__})
                # Останавливаем сетевой цикл при первом сбое, старый CSV остаётся цел.
                raise SystemExit('HTTP-сбор остановлен: ' + str(exc))
        try:
            first_seen = previous.loc[raw['url'], 'first_seen_at'] if previous is not None and raw['url'] in previous.index else now
            record = metadata_record(raw, dictionary, cfg, first_seen)
            record['fetched_at'] = now
            day = record['published_at'][:10]
            if not cfg['period_start'] <= day <= cfg['period_end']:
                rejected.append({'url': raw['url'], 'reason': 'outside_period'})
                continue
            rows.append(record)
        except (ValueError, KeyError) as exc:
            rejected.append({'url': raw.get('url', ''), 'reason': str(exc)})
    if not rows:
        raise SystemExit('Не собрано ни одной валидной записи; существующие файлы сохранены.')
    frame = pd.DataFrame(rows).drop_duplicates('url').drop_duplicates('duplicate_key').sort_values('published_at')
    frame.to_csv(ROOT / cfg['output_file'], index=False)
    monthly = monthly_counts(frame, cfg['period_start'], cfg['period_end'])
    monthly.to_csv(ROOT / cfg['coverage_file'], index=False)
    manifest = {'collected_at': now, 'mode': 'direct_refresh' if args.refresh else 'local_search_discovery',
                'region': cfg['region_name'], 'period': [cfg['period_start'], cfg['period_end']],
                'articles': len(frame), 'relevant_title_articles': int(frame.relevant.sum()),
                'months_with_articles': int(monthly.has_retrieved_articles.sum()),
                'months_total': len(monthly), 'coverage_complete': False,
                'requests_made': requests_made, 'rejected': rejected,
                'discovery_sha256': hashlib.sha256(discovery_path.read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'csv_sha256': hashlib.sha256((ROOT / cfg['output_file']).read_bytes()).hexdigest(),
                'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['src/news_collection.py', 'scripts/collect_regional_news.py']}}
    (ROOT / cfg['manifest_file']).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['articles', 'relevant_title_articles', 'months_with_articles', 'months_total', 'coverage_complete']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
