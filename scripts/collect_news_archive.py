"""Скачивание заголовков WordPress по месяцам: requests + BeautifulSoup, кеш, без браузера."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup
import pandas as pd
import requests
from src.news_collection import metadata_record, monthly_counts, title_tags, TOPICS


def plain_title(value):
    return ' '.join(BeautifulSoup(value, 'html.parser').get_text(' ', strip=True).split())


def wordpress_record(post):
    """GMT API — авторитетная дата; локальное время сайта может быть UTC+5."""
    date = post.get('date_gmt')
    if not date or date.startswith('0000'):
        raise ValueError('missing_publication_gmt')
    published = pd.Timestamp(date).tz_localize('UTC').tz_convert('Europe/Moscow')
    title = plain_title(post['title']['rendered'])
    if not title:
        raise ValueError('empty_title')
    return {'url': post['link'], 'title': title, 'published_at': published.isoformat(),
            'metadata_source': 'wordpress_archive_json', 'source_article_id': post['id']}


def cached_page(session, cfg, month, page, offline):
    cache = ROOT / cfg['cache_directory'] / f'{month}_{page:03d}.json'
    if cache.exists():
        return json.loads(cache.read_text()), False
    if offline:
        raise FileNotFoundError(f'Нет кеша: {cache.name}')
    params = {'rest_route': '/wp/v2/posts', 'per_page': cfg['per_page'], 'page': page,
              'orderby': 'date', 'order': 'asc',
              '_fields': 'id,date,date_gmt,link,title,categories'}
    # WordPress after/before фильтруют локальную дату сайта, не date_gmt.
    # Запрашиваем с запасом в день; окончательная фильтрация — по точному GMT.
    params['after'] = (pd.Timestamp(str(month)) - pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    params['before'] = (pd.Timestamp(str(month)) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    response = None
    for attempt in range(3):
        time.sleep(cfg['request_pause_seconds'] + attempt * 2)
        try:
            response = session.get(cfg['endpoint'], params=params, timeout=cfg['timeout_seconds'])
            response.raise_for_status()
            posts = response.json()
            if not isinstance(posts, list):
                raise ValueError('API вернул не список')
            break
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
    payload = {'fetched_at': datetime.now(timezone.utc).isoformat(), 'request_url': response.url,
               'total': int(response.headers['X-WP-Total']),
               'total_pages': int(response.headers['X-WP-TotalPages']), 'posts': posts}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, ensure_ascii=False))
    return payload, True


def collection_report(cfg, combined, audit, manifest):
    source_lines = '\n'.join(f'| {r["source"]} | {r["articles"]} |' for r in manifest['sources'])
    months = pd.to_datetime(combined.published_at, utc=True).dt.tz_convert('Europe/Moscow').dt.strftime('%Y-%m')
    local = combined.territory_ids.fillna('').astype(str).str.len().gt(0)
    confirmed = combined.event_region_confirmed_in_title.eq(True)
    topic_counts = combined.topics.fillna('').str.get_dummies(sep=';').sum().sort_values(ascending=False)
    topic_lines = '\n'.join(f'| {topic} | {int(count)} |' for topic, count in topic_counts.items())
    incomplete = ', '.join(audit.loc[~audit.archive_pagination_complete, 'month']) or 'нет'
    text = f'''# Сбор региональных новостей: Оренбургская область

Обновлено {manifest['collected_at']}. Период 2023–2024, как у панели расходов.
Корпус расширен за пределы МЧС: [Оренбург Медиа](https://orenburg.media/) —
общественные, экономические, транспортные и другие региональные заголовки.
Сбор выполняется **requests + BeautifulSoup**, без Playwright, браузера и LLM.

## Что реально скачано

{len(combined)} уникальных заголовков; {months.nunique()} из 24 месяцев.
JSON-архив нового источника: все страницы получены в {manifest['complete_archive_months']}/24 месяцах.
Месяцы с неполной пагинацией: {incomplete}. Полнота всех новостей региона неизвестна.

| Источник | Заголовков в объединённом CSV |
|---|---:|
{source_lines}

Явная привязка к области или МО: {int(confirmed.sum())}; к МО: {int(local.sum())}.
Источник из области может публиковать федеральные новости; без явной географии
материал влияет только на признаки покрытия. Неоднозначный топоним не переносится
на весь регион. Справочник применяется в год публикации; границы имеют годовое разрешение.

| Автоматическая тема | Упоминаний |
|---|---:|
{topic_lines}

Один заголовок может иметь несколько тем. Правила дают признаки, а не истинные
метки экономических шоков. Полные тексты не скачивались; контекст заголовка ограничен.
Дедупликация: URL, затем нормализованный заголовок + календарная дата.

## Как повторить

```bash
python -m venv .venv-news
.venv-news/bin/pip install -r requirements-news.txt
.venv-news/bin/python scripts/collect_news_archive.py
# Повторная обработка без сети:
.venv-news/bin/python scripts/collect_news_archive.py --offline
# Модели запускаются в основной среде sber:
python scripts/evaluate_news_impact.py
python scripts/build_news_impact_report.py
```

Конфигурация: configs/news_archive.json. API: index.php?rest_route=/wp/v2/posts,
даты по месяцу, 100 записей на страницу, один процесс, пауза 0.6 секунды.
Проверяются X-WP-TotalPages; лимит 30 страниц/месяц. Если страниц больше или
есть сетевой сбой, месяц помечается неполным и процесс завершает работу с ошибкой.
Кеш сохраняет каждую страницу и фактическое fetched_at; повторный запуск докачивает
отсутствующие страницы. Для полностью свежего сбора нужно отдельное имя каталога
кеша и выходных файлов в копии конфигурации.

Локальные файлы в data/inputs/ (не входят в Git):

- news_orenburg.csv — исходная выборка МЧС, сохранена.
- news_orenburg_multisource.csv — объединённый корпус.
- news_orenburg_multisource_monthly.csv — месячные счётчики.
- news_orenburg_archive_audit.csv — страницы и ошибки по месяцам.
- news_orenburg_source_summary.csv — число записей по источникам.
- news_orenburg_archive_cache/ — сырые JSON-страницы и время получения.
- news_orenburg_multisource_manifest.json — SHA256 корпуса, страниц, кода и конфигурации.

## Как получать дополнительные признаки

На origin t берутся только публикации с available_at до конца t. Счётчики
за 1/3 месяца преобразуются log(1+n). В модель входят объём по источникам,
наличие публикаций, 10 тем в регионе/МО и пять признаков направления:
рост/снижение цен, рост доходов, открытие/закрытие бизнеса.
Примеры: «подорожал бензин» → prices + price_increase;
«в Орске открыли магазин» → retail_services + business_opening + локальный Орск.
Каждый признак имеет имя в news_feature_schema.json нового экспериментального запуска.
news_top_coefficients_development.csv показывает коэффициенты Ridge только до теста;
это диагностика, не доказательство причинного эффекта.

Для числовых цен, зарплат, безработицы и погоды нужны отдельные таблицы официальных
наблюдений с датами публикации. Суммы и проценты из разнородных заголовков нельзя
складывать в показатель. Перед добавлением такого ряда задаются единица измерения,
география, available_at и лаг: на origin используется последнее уже опубликованное
значение, а не факт будущего целевого месяца.

Дата новости берётся из date_gmt API и переводится в Europe/Moscow: локальное
время сайта отличается от московского. Запрос охватывает граничные дни с запасом,
затем новости фильтруются по точной дате GMT. event_date не восстанавливается.
first_seen_at — фактическое получение в 2026; operational_available_at=max(publication,first_seen).
Исторический эксперимент использует дату публикации как явное допущение:
редактирование заголовков и исторические версии API неизвестны.

Результаты и семь контролей для прогноза/детекторов: [сравнительный отчёт](REGIONAL_NEWS_IMPACT_RU.md).
Тест и регион уже рассматривались; новые результаты исследовательские.
'''
    (ROOT / 'docs/REGIONAL_NEWS_COLLECTION_RU.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/news_archive.json')
    parser.add_argument('--offline', action='store_true', help='Только сохранённые JSON-страницы')
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    base_cfg = json.loads((ROOT / cfg['geography_config']).read_text())
    dictionary = pd.read_excel(ROOT / 'data/inputs/municipal_dictionary.xlsx')
    # Все районы из ежегодного справочника; короткие неоднозначные названия
    # распознаются только с явным словом «район»/«округ».
    import re
    for name in dictionary.loc[dictionary.region_code == cfg['region_code'], 'municipal_district_name_short'].unique():
        if name in base_cfg['territory_aliases']:
            continue
        stem = re.sub(r'(ский|цкий|ный)$', '', name)
        base_cfg['territory_aliases'][name] = r'\b' + re.escape(stem).replace('ё', '[её]') + r'\w*\s+(?:район|округ)\w*\b'
    source_cfg = {**base_cfg, 'source_name': cfg['source_name'], 'coverage': 'dated_archive_as_returned_not_all_regional_news'}
    session = requests.Session()
    session.headers['User-Agent'] = 'SberIndexResearch/1.0 (historical headline metadata; cached; one worker)'
    rows, audit, rejected, hashes = [], [], [], {}
    requests_made = 0
    for month in pd.period_range(cfg['period_start'], cfg['period_end'], freq='M'):
        pages, expected, retrieved, failed = 0, None, 0, ''
        for page in range(1, cfg['max_pages_per_month'] + 1):
            try:
                payload, fetched = cached_page(session, cfg, month, page, args.offline)
                requests_made += int(fetched)
                expected = payload['total_pages']
                pages += 1
                retrieved += len(payload['posts'])
                cache = ROOT / cfg['cache_directory'] / f'{month}_{page:03d}.json'
                hashes[str(cache.relative_to(ROOT))] = hashlib.sha256(cache.read_bytes()).hexdigest()
                for post in payload['posts']:
                    try:
                        raw = wordpress_record(post)
                        # Перекрывающиеся граничные дни устраняются здесь.
                        if pd.Timestamp(raw['published_at']).strftime('%Y-%m') != str(month):
                            continue
                        record = metadata_record(raw, dictionary, source_cfg, payload['fetched_at'])
                        record['article_id'] = str(post['id'])
                        rows.append(record)
                    except (ValueError, KeyError) as exc:
                        rejected.append({'id': post.get('id'), 'month': str(month), 'reason': str(exc)})
                if page >= expected:
                    break
            except Exception as exc:
                failed = type(exc).__name__ + ': ' + str(exc)
                break
        audit.append({'month': str(month), 'source': cfg['source_name'], 'pages_expected': expected,
                      'pages_retrieved': pages, 'archive_pagination_complete': bool(expected is not None and pages >= expected and not failed),
                      'api_posts_including_boundary_days': retrieved, 'error': failed})
        print(f'{month}: {retrieved} записей, страниц {pages}/{expected}, ошибка={failed or "нет"}', flush=True)
    if not rows:
        raise SystemExit('Нет новых записей; существующий корпус сохранён')
    additional = pd.DataFrame(rows).drop_duplicates('url').sort_values('published_at')
    # Исходный корпус МЧС сохраняется отдельно и включается в объединённый CSV.
    old = pd.DataFrame()
    if cfg.get('previous_news_file'):
        old = pd.read_csv(ROOT / cfg['previous_news_file'], dtype={'territory_ids': 'string'})
        old_tags = old.title.map(title_tags)
        old['topics'] = old_tags.map(lambda x: ';'.join(x[0]))
        for fact in old_tags.iloc[0][1]:
            old[fact] = old_tags.map(lambda x: x[1][fact])
    combined = pd.concat([old, additional], ignore_index=True).drop_duplicates('url').drop_duplicates('duplicate_key').sort_values('published_at')
    combined.to_csv(ROOT / cfg['output_file'], index=False)
    monthly_counts(combined, cfg['period_start'], cfg['period_end']).to_csv(ROOT / cfg['coverage_file'], index=False)
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(ROOT / cfg['archive_audit_file'], index=False)
    sources = combined.groupby('source').agg(articles=('url', 'size'), first_publication=('published_at', 'min'), last_publication=('published_at', 'max')).reset_index()
    sources.to_csv(ROOT / cfg['source_summary_file'], index=False)
    manifest = {'collected_at': datetime.now(timezone.utc).isoformat(), 'articles': len(combined),
                'new_source_articles': len(additional), 'sources': sources.to_dict('records'),
                'complete_archive_months': int(audit_frame.archive_pagination_complete.sum()),
                'months': len(audit_frame), 'all_regional_news_complete': False,
                'metadata_only': True, 'requests_made': requests_made, 'rejected': rejected,
                'topics': TOPICS, 'cache_sha256': hashes,
                'csv_sha256': hashlib.sha256((ROOT / cfg['output_file']).read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['scripts/collect_news_archive.py', 'src/news_collection.py']}}
    (ROOT / cfg['manifest_file']).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if cfg['region_code'] == 56:
        collection_report(cfg, combined, audit_frame, manifest)
    print(f'Сохранено {len(combined)} заголовков; новый источник {len(additional)}; полных месяцев архива {manifest["complete_archive_months"]}/{len(audit)}', flush=True)
    if not audit_frame.archive_pagination_complete.all():
        raise SystemExit('Часть страниц не получена. Сохранена частичная выборка; повторный запуск продолжит по кешу.')


if __name__ == '__main__':
    main()
