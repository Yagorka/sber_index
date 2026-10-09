"""Collect the public dated NIANN headline archive; cache and audit every week."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qs, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup
import pandas as pd
import requests
from src.news_collection import metadata_record, monthly_counts, parse_publication


def parse_niann(html):
    soup = BeautifulSoup(html, 'html.parser')
    found = soup.select_one('#block_118 .search_full__found')
    if found is None:
        raise ValueError('Missing archive result count')
    count_text = found.get_text(' ', strip=True)
    capped = 'более' in count_text
    count = int(re.search(r'\d+', count_text).group())
    rows = []
    for item in soup.select('#block_118 .cell_news_item'):
        stamp = item.select_one('.date_standart')
        link = item.select_one('.cell_news_alltext > a[href]')
        if link is None:
            raise ValueError('Archive item lacks title/link')
        published = None
        if stamp is not None:
            text = stamp.get_text(' ', strip=True)
            text = re.sub(r'(\d{4})\s+(\d{2}:\d{2})', r'\1, \2', text)
            published = parse_publication(text).isoformat()
        article = parse_qs(urlsplit(link['href']).query)['id'][0]
        rows.append({'url': 'https://www.niann.ru/?id='+article,
                     'title': link.get_text(' ', strip=True),
                     'published_at': published, 'metadata_source': 'public_dated_archive_html'})
    pages = {}
    for link in soup.select('#block_118 a[href]'):
        params = parse_qs(urlsplit(link['href']).query)
        if 'page118' in params:
            pages[int(params['page118'][0])] = urljoin('https://www.niann.ru/', link['href'])
    return rows, count, capped, pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/news_archive_nizhny.json')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    cfg_path = ROOT/args.config
    cfg = json.loads(cfg_path.read_text())
    geo = json.loads((ROOT/cfg['geography_config']).read_text())
    dictionary = pd.read_excel(ROOT/'data/inputs/municipal_dictionary.xlsx')
    for name in dictionary.loc[dictionary.region_code == cfg['region_code'], 'municipal_district_name_short'].unique():
        if name not in geo['territory_aliases']:
            stem = re.sub(r'(ский|цкий|ный)$', '', name)
            geo['territory_aliases'][name] = r'\b'+re.escape(stem).replace('ё', '[её]')+r'\w*\s+(?:район|округ)\w*\b'
    source_cfg = {**geo, 'source_name': cfg['source_name'], 'coverage': 'public_weekly_archive_not_all_regional_news'}
    cache = ROOT/cfg['cache_directory']
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'SberIndexResearch/1.0 (historical public headlines; cached; one worker)'
    rows, audits, hashes, rejected = [], [], {}, []
    for start in pd.date_range(cfg['period_start'], cfg['period_end'], freq='7D'):
        end = min(start+pd.Timedelta(days=6), pd.Timestamp(cfg['period_end']))
        url = cfg['endpoint']
        params = {'id':118, 'action':'process', 'p__search_date1':start.strftime('%d.%m.%Y'),
                  'p__search_date2':end.strftime('%d.%m.%Y')}
        page, expected, collected, links, seen, rejected_week = 1, None, [], {}, set(), []
        while True:
            stem = f'{start:%Y-%m-%d}_{page:03d}'
            file = cache/(stem+'.html')
            meta = cache/(stem+'.json')
            if not file.exists():
                if args.offline:
                    raise FileNotFoundError(file)
                for attempt in range(3):
                    try:
                        time.sleep(cfg['request_pause_seconds']+attempt*2)
                        reply = session.get(url, params=params, timeout=cfg['timeout_seconds'])
                        reply.raise_for_status()
                        # Preserve an unexpected layout for diagnosis, never label it as valid cache.
                        (cache/(stem+'_last_response.html')).write_text(reply.text)
                        parsed = parse_niann(reply.text)
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 2:
                            raise
                file.write_text(reply.text)
                meta.write_text(json.dumps({'url':reply.url,'fetched_at':datetime.now(timezone.utc).isoformat()}))
            batch, count, capped, new_links = parse_niann(file.read_text())
            if capped:
                raise ValueError(f'{start}: archive capped at 500; narrow query before continuing')
            if expected is None:
                expected = count
            if count != expected:
                raise ValueError('Archive pagination count changed')
            links.update(new_links)
            fetched = json.loads(meta.read_text())['fetched_at']
            for raw in batch:
                seen.add(raw['url'])
                if raw['published_at'] is None:
                    rejected_week.append({**raw, 'period_start':str(start.date()),
                                          'reason':'undated_archive_object_not_used_for_news'})
                    continue
                stamp = pd.Timestamp(raw['published_at']).tz_localize(None).normalize()
                if not start <= stamp <= end:
                    raise ValueError('Archive returned an out-of-range date')
                collected.append(metadata_record(raw, dictionary, source_cfg, fetched))
            hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
            unique = len(seen)
            if unique == expected:
                break
            if not batch or page+1 not in links or page >= cfg['max_pages_per_week']:
                raise ValueError(f'{start}: incomplete archive {unique}/{expected}')
            page += 1
            # Full public query parameters make cached query IDs unnecessary on resume.
            url = cfg['endpoint']
            params = {**params, 'page118':page}
        rows.extend(collected)
        rejected.extend(rejected_week)
        audits.append({'period_start':str(start.date()),'period_end':str(end.date()),
                       'expected':expected,'retrieved':unique,'dated_news':len({x['url'] for x in collected}),
                       'undated_objects':len({x['url'] for x in rejected_week}), 'pages':page,'complete':True})
        interim = pd.DataFrame(rows).drop_duplicates('url').drop_duplicates('duplicate_key').sort_values('published_at')
        interim.to_csv(ROOT/cfg['output_file'],index=False)
        pd.DataFrame(audits).to_csv(ROOT/cfg['archive_audit_file'],index=False)
        print(f'{start:%Y-%m-%d}: {unique}/{expected}; pages={page}',flush=True)
    combined = pd.DataFrame(rows).drop_duplicates('url').drop_duplicates('duplicate_key').sort_values('published_at')
    combined.to_csv(ROOT/cfg['output_file'], index=False)
    monthly_counts(combined,cfg['period_start'],cfg['period_end']).to_csv(ROOT/cfg['coverage_file'],index=False)
    manifest = {'collected_at':datetime.now(timezone.utc).isoformat(),'articles':len(combined),
                'complete_weeks':len(audits),'all_archive_queries_complete':all(x['complete'] for x in audits),
                'all_regional_news_complete':False,'source':cfg['source_name'],'endpoint':cfg['endpoint'],
                'excluded_undated_archive_objects':rejected,
                'csv_sha256':hashlib.sha256((ROOT/cfg['output_file']).read_bytes()).hexdigest(),
                'config_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),'cache_sha256':hashes,
                'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/collect_transfer_news.py','src/news_collection.py']}}
    (ROOT/cfg['manifest_file']).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Collection complete:',len(combined),flush=True)


if __name__ == '__main__':
    main()
