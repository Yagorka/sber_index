"""Reproducible review packet; automated diagnostics are not human ground truth."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.evaluate_news_decay import REGIONS,hash_file
from src.llm_news import FIELDS,economic_filter,validate_rows


def main():
    out=ROOT/'data/inputs/news_label_review';out.mkdir(parents=True,exist_ok=True)
    summaries=[];packets=[];hashes={}
    for region,config in REGIONS.items():
        cfg=json.loads((ROOT/config).read_text());path=ROOT/cfg['output_directory']/'pilot/annotated_news.csv'
        records=pd.read_csv(path,dtype={'territory_ids':'string'})
        validate_rows({'rows':[[i]+[int(row[f]) for f in FIELDS] for i,(_,row) in enumerate(records.iterrows())]},len(records))
        retained=economic_filter(records)
        aligned=retained & records.event_region_confirmed_in_title & records.category_mask.gt(0) & records.geo_status.ne('ambiguous_or_unresolved')
        directions=records[['price_direction','income_direction','business_direction']].ne(9).any(axis=1)
        contradictions=(records.category_mask.gt(0)|directions)&records.economic_relevance.eq(0)
        title_geo_block=retained & ~records.event_region_confirmed_in_title
        summaries.append(dict(region=cfg.get('region_name',region),articles=len(records),economic_retained=int(retained.sum()),
            aligned_articles=int(aligned.sum()),economic_blocked_no_title_region=int(title_geo_block.sum()),
            economic_ambiguous=int((retained & records.geo_status.eq('ambiguous_or_unresolved')).sum()),
            known_directions=int(directions.sum()),direction_or_category_without_economy=int(contradictions.sum()),
            mixed_sentiment=int(records.sentiment.eq(2).sum()),unknown_sentiment=int(records.sentiment.eq(9).sum())))
        # Disjoint priority strata; every region contributes exactly 60 examples.
        records['review_stratum']='other'
        records.loc[title_geo_block,'review_stratum']='economic_blocked_geography'
        records.loc[directions|contradictions,'review_stratum']='explicit_direction_or_conflict'
        sample=pd.concat([g.assign(_rank=g.title_sha256.map(lambda s:hash_file_string('42:'+s))).sort_values('_rank').head(20)
            for _,g in records.groupby('review_stratum')]).drop(columns='_rank')
        assert len(sample)==60
        summaries[-1]['review_months']=int(sample.published_at.str[:7].nunique())
        sample['region']=region
        cols=['region','review_stratum','title_sha256','title','url','published_at','territory_ids','geo_status',
              'event_region_confirmed_in_title','topics']+FIELDS
        sample=sample[cols].copy()
        for f in FIELDS+['territory_ids','event_region_confirmed_in_title']:
            sample['human_'+f]=''
        sample['human_notes']='';sample['reviewer']='';packets.append(sample)
        hashes[region]={'annotation_sha256':hash_file(path),'selected_titles':sample.title_sha256.tolist()}
    packet=pd.concat(packets,ignore_index=True)
    destination=out/'review_180.csv'
    if destination.exists():
        previous=pd.read_csv(destination)
        review_columns=[c for c in previous if c.startswith('human_') or c=='reviewer']
        if previous[review_columns].notna().any().any():
            raise ValueError('Review packet contains human answers; preserve them before regenerating')
    packet.to_csv(destination,index=False)
    table=pd.DataFrame(summaries)
    artifact=ROOT/'artifacts/news_label_audit';artifact.mkdir(exist_ok=True)
    table.to_csv(artifact/'summary.csv',index=False)
    (artifact/'manifest.json').write_text(json.dumps({'seed':42,'examples':len(packet),'strata':'20 per region/stratum; disjoint priority: direction > blocked geography > other',
        'human_review_completed':False,'sources':hashes,'packet_sha256':hash_file(out/'review_180.csv'),'code_sha256':hash_file(Path(__file__))},ensure_ascii=False,indent=2)+'\n')
    findings=pd.DataFrame([
        ('edc451d6c20a','Ограничения любительского рыболовства','Категория транспорта (32) не следует из заголовка; ожидается 0.'),
        ('b6b2ab2d26f4','Летняя скидка детям на поезд','Явное снижение стоимости: price_direction=9 пропускает направление -1.'),
        ('225b301e4b70','Суд по радиоактивной водопроводной воде','Медицинские товары/услуги (4) не указаны; Домбаровский также пропущен географией.'),
        ('38ed65f8d49c','Готовность производства в ТОР Володарск','Название города в заголовке не подтвердило регион; проверить словарь и неоднозначность.'),
        ('a5ab01ec139e','Будущий мясоперерабатывающий комплекс','Продукты названы явно, но category_mask=0; строительство не равно уже открытому бизнесу.'),
        ('6a01d3de63b8','Открытие путепровода','business_direction=1 трактует инфраструктурный объект как открытие бизнеса; прямого факта работы предприятия нет.'),
        ('b44be8dadbdb','Потеря денег жителем Гая','Гай пропущен подтверждением географии, но economic_relevance=0 разумно: бытовой криминальный эпизод.')
    ],columns=['title_sha256_prefix','case','agent_observation'])
    assert all(packet.title_sha256.str.startswith(prefix).any() for prefix in findings.title_sha256_prefix)
    findings.to_csv(artifact/'agent_spotcheck.csv',index=False)
    text=f'''# Проверка новостной разметки

Проверены допустимые коды и полнота 3600 ответов LLM. Диагностика ниже показывает объём используемых событий и ограничения географии; это не оценка точности по независимому эталону.

{table.to_markdown(index=False)}

`aligned_articles` — экономический отбор, подтверждение региона в заголовке, разрешённая география и ненулевая категория. Публикации без подтверждения региона в заголовке могут попасть в общий признак объёма, но не в сигнал категории/сентимента. Это консервативное правило снижает покрытие и может пропускать местные новости, где регион понятен из источника. Ослаблять его без проверки места события нельзя: региональные издания публикуют и федеральные новости.

Подготовлен локальный файл `data/inputs/news_label_review/review_180.csv`: 60 заголовков на область, по 20 из трёх непересекающихся групп — явные направления/конфликты, экономические заголовки без подтверждённой географии, прочие. Отбор детерминированный из всего периода, но не гарантирует равное число по месяцам; фактическое покрытие месяцев указано в review_months. Доли ошибок по этой обогащённой выборке нельзя считать точностью всего корпуса без поправок на дизайн.

## Как провести независимый аудит

Рецензент заполняет human_* и reviewer только по заголовку, без будущих событий. Сентимент: -1/0/1/2/9; значимость и ЧС: 0/1; направления: -1/0/1/9; маска категорий: 1 общие доходы, 2 продукты, 4 медицина, 8 маркетплейсы, 16 общепит, 32 транспорт. Нейтральное направление 0 допустимо только при явно указанном отсутствии изменения, иначе 9. Предложение снизить цену не равно снижению. Дорожная инфраструктура и ДТП не всегда связаны с транспортными расходами. География — место события, не местонахождение издателя.

Независимый ручной аудит ещё не выполнен: human_* оставлены пустыми. После заполнения нужны согласие по каждому полю, матрицы ошибок направлений/сентимента, precision/recall экономического отбора и категорий, отдельно ошибки географии и доля неизвестных. Спорные случаи желательно разметить вторым рецензентом. Исправление промпта следует проверять на отдельной отложенной группе.

Заголовки, ссылки и будущие человеческие ответы остаются локально и исключены из Git. В Git — сводные счётчики, хеши и инструкция. Воспроизведение: `python scripts/audit_news_labels.py` в существующем окружении sber. 
## Проверка агентом 27 примеров

Просмотрены первые три заголовка каждой из девяти групп пакета (регион × страта). Найдены пропуски явного ценового направления, сомнительные категории, смешение открытия инфраструктуры с открытием бизнеса и пропуски географии. Наблюдения ниже — кандидаты на исправление, а не измеренная точность. Исходные метки и прогнозы не изменены: корректировку словаря/промпта следует оценивать отдельно.

{findings.to_markdown(index=False)}

Всего формальных конфликтов «есть направление/категория, но экономическая значимость 0» — {int(table.direction_or_category_without_economy.sum())}. Формальная проверка не обнаруживает смысловые ошибки вроде неверного транспорта или пропущенной скидки. Главный следующий шаг — независимая проверка пакета и географии; расширение разметки без неё может тиражировать ошибки.
'''
    (ROOT/'docs/NEWS_LABEL_AUDIT_RU.md').write_text(text)
    print(table.to_string(index=False))


def hash_file_string(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()

if __name__=='__main__':main()
