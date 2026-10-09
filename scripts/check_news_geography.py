"""Check historical location IDs and review the fixed 180-headline sample."""
import json,re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.news_collection import match_territories
from src.llm_news import economic_filter,FIELDS,validate_rows,_territories
from scripts.evaluate_news_decay import REGIONS,hash_file

# Observations by row of the frozen review packet, never automatic label replacements.
NOTES={
1:'Явное снижение проезда пропущено: направление цены должно быть -1.',
2:'Водопроводная вода не указывает на медицинские товары/услуги. Пропущен Домбаровский.',
4:'Асекеево не распознано; ремонт дороги не обязательно относится к потребительским транспортным услугам.',
6:'Ясный не распознан. Судебное взыскание после ДТП не означает цену транспортной услуги.',
8:'Ozon назван явно; проверить отсутствие категории маркетплейсов.',
9:'Проверить экономическую значимость: уголовное мошенничество не всегда описывает доходы или расходы населения.',
12:'Орловка неоднозначна: нельзя выбирать муниципалитет только по названию села.',
14:'Пропущена Сакмара; маршрут между населёнными пунктами требует проверки обоих мест.',
16:'Ростоши требуют отдельного проверенного соответствия городскому муниципалитету.',
20:'Открытие сезона железной дороги не доказывает рост работы бизнеса; транспортная категория отсутствует.',
26:'Цветы ошибочно отнесены к продовольствию: ожидается категория 0.',
30:'Форма «орчанка» не распознана как Орск; выплата исключена из местного сигнала.',
34:'Медицинские средства не означают рост денежных доходов: income_direction=9.',
35:'Попытка взыскать деньги не означает уже полученную выплату: income_direction=9.',
36:'Экономическая значимость 0 одновременно с категорией транспорта; проверить единообразие правила уборки снега.',
38:'Пригородный неоднозначен; место события требует отдельного подтверждения.',
40:'Гай пропущен географией; исключение бытового мошенничества из экономики разумно.',
43:'Исчезновение человека само по себе не отвечает заданному перечню чрезвычайных событий.',
44:'«Под Оренбургом» не означает город Оренбург; возможна ошибочная привязка к городу.',
54:'Проверить единообразие категории для ремонта привокзальной площади.',
61:'Любительское рыболовство ошибочно отнесено к транспорту: категория 0.',
62:'Володарск пропущен географией; нужен проверенный городской вариант названия района.',
63:'Каток ошибочно отнесён к кафе/ресторанам: категория 0.',
65:'Заволжье может обозначать город или обширную территорию; нельзя автоматически присваивать район.',
68:'Автозаводский район города без города не даёт однозначную географию.',
70:'Посещение молодёжного форума не означает экономическое изменение; проверить relevance.',
78:'Маркетплейс стартапов не равен потребительскому маркетплейсу: категория 0.',
80:'Мясопереработка относится к продуктам, но категория 0; уточнить правило для будущего строительства.',
84:'Место в рейтинге застройщиков не доказывает рост или открытие бизнеса: business_direction=9.',
85:'Предложение повысить штраф — не изменение цены услуги: price_direction=9.',
87:'Возобновление работы проката не означает рост цены: price_direction=9.',
89:'Реконструкция мостов не означает рост цен: price_direction=9.',
91:'Компенсация затрат экспортёрам не равна доходам жителей: income_direction=9.',
93:'Общее обещание направить бюджет на помощь не указывает явное увеличение выплаты: проверить income_direction.',
94:'Цена строительства канатной дороги не означает рост тарифа: price_direction=9; транспортная категория пропущена.',
95:'Разрешение открыть вклад не означает увеличение дохода: income_direction=9.',
97:'Рост продаж смартфонов ошибочно записан в рост цен: price_direction=9.',
99:'Лечение по ОМС не означает рост дохода: income_direction=9; указана медицина, а не общие доходы.',
100:'Фуникулер потенциально относится к транспорту; проверить пропущенную категорию.',
104:'Высокая пожароопасность ещё не означает произошедший пожар: проверить shock.',
117:'Явное повышение тарифа бани пропущено: price_direction=1.',
127:'Увеличение штата означает рост занятости: income_direction=1; форма «мантуровская» пропущена географией.',
131:'Спрос на ипотеку не означает рост доходов населения; общая категория расходов требует проверки.',
143:'«Хорошая зарплата» не означает её рост, «карьерный рост» не означает рост цен: оба направления должны быть 9.',
144:'Новая крыша ипподрома не доказывает открытие/рост бизнеса: business_direction=9.',
145:'Рост зарплат записан в рост цен: price_direction=9, income_direction=1.',
146:'Указание возродить гостиницу не означает уже открытый бизнес: business_direction=9.',
149:'Одиночная компенсация после нападения требует уточнения экономического отбора; не переносить на все муниципалитеты.',
150:'Заразность клещей не указывает прямо на медицинские товары/услуги: проверить категорию.',
156:'Рост потребительской активности не означает рост цен: price_direction=9.',
157:'Сокращение времени ремонта сети не означает рост цен: price_direction=9.',
159:'Сокращение автобусных рейсов не означает снижение цены: price_direction=9.',
}


def expanded_geo(config,dictionary):
    geo=json.loads((ROOT/config).read_text())
    for name in dictionary.loc[dictionary.region_code.eq(geo['region_code']),'municipal_district_name_short'].unique():
        if name not in geo['territory_aliases']:
            stem=re.sub(r'(ский|цкий|ный)$','',name)
            geo['territory_aliases'][name]=r'\b'+re.escape(stem).replace('ё','[её]')+r'\w*\s+(?:район|округ)\w*\b'
    return geo


def main():
    dictionary=pd.read_excel(ROOT/'data/inputs/municipal_dictionary.xlsx')
    out=ROOT/'artifacts/news_geography_audit';out.mkdir(exist_ok=True)
    summary=[];issues=[];hashes={};coverage=[];missing=[]
    files={'orenburg':'configs/news_orenburg.json','nizhny':'configs/news_nizhny.json','kostroma':'configs/news_kostroma.json'}
    for region,lc in REGIONS.items():
        cfg=json.loads((ROOT/lc).read_text());path=ROOT/cfg['output_directory']/'pilot/annotated_news.csv'
        records=pd.read_csv(path,dtype={'territory_ids':'string'});geo=expanded_geo(files[region],dictionary)
        validate_rows({'rows':[[i]+[int(row[f]) for f in FIELDS] for i,(_,row) in enumerate(records.iterrows())]},len(records))
        economic=economic_filter(records);noid=0;invalid=0;unresolved=0;diff=0;ids_seen=set();aligned_seen=set()
        for index,row in records.iterrows():
            pub=pd.Timestamp(row.published_at);ids=_territories(row.territory_ids)
            active=dictionary[dictionary.region_code.eq(cfg['region_code'])&(dictionary.year_from<=pub.year)&(dictionary.year_to>pub.year)]
            invalid_ids=set(ids)-set(active.territory_id);invalid+=bool(invalid_ids)
            rematched,uncertain=match_territories(row.title,pub,dictionary,geo)
            differs=ids!=rematched;diff+=differs
            if invalid_ids or differs or uncertain:
                issues.append(dict(region=region,title_sha256=row.title_sha256,stored_ids=';'.join(map(str,ids)),
                    rematched_ids=';'.join(map(str,rematched)),invalid_ids=';'.join(map(str,sorted(invalid_ids))),unresolved=';'.join(uncertain)))
            ids_seen.update(ids);noid+=not bool(ids);unresolved+=bool(uncertain)
            if economic.iloc[index] and row.event_region_confirmed_in_title and row.category_mask>0:
                aligned_seen.update(ids)
            # Named city in a later municipal merger: expose exact missing lookup.
            for name,pattern in geo['territory_aliases'].items():
                if re.search(pattern,row.title,re.I) and not active.municipal_district_name_short.eq(name).any():
                    missing.append(dict(region=region,name=name,year=pub.year,title_sha256=row.title_sha256))
        municipalities=dictionary[dictionary.region_code.eq(cfg['region_code'])&(dictionary.year_from<=2024)&(dictionary.year_to>2024)]
        for row in municipalities.itertuples():
            coverage.append(dict(region=region,territory_id=row.territory_id,name=row.municipal_district_name_short,
                explicitly_mentioned=row.territory_id in ids_seen,explicitly_aligned_economic=row.territory_id in aligned_seen))
        summary.append(dict(region=cfg['region_code'],name=cfg.get('region_name',region),articles=len(records),no_municipality_id=noid,
            explicit_municipalities=len(ids_seen),economic_municipalities=len(aligned_seen),invalid_historical_ids=invalid,
            differing_recomputed_ids=diff,unresolved_lookup=unresolved))
        hashes[region]={'labels':hash_file(path),'geo_config':hash_file(ROOT/files[region])}
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(issues,columns=['region','title_sha256','stored_ids','rematched_ids','invalid_ids','unresolved']).to_csv(out/'lookup_issues.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'municipality_coverage.csv',index=False)
    pd.DataFrame(missing,columns=['region','name','year','title_sha256']).to_csv(out/'missing_name_versions.csv',index=False)
    packet_path=ROOT/'data/inputs/news_label_review/review_180.csv';packet=pd.read_csv(packet_path,dtype={'territory_ids':'string'})
    assert len(packet)==180
    assert hash_file(packet_path)==json.loads((ROOT/'artifacts/news_label_audit/manifest.json').read_text())['packet_sha256']
    reviewed=packet.copy();reviewed['agent_notes']=[NOTES.get(i,'При проверке заголовка явных проблем не обнаружено; это не независимое подтверждение точности.') for i in range(180)]
    reviewed['agent_requires_review']=[i in NOTES for i in range(180)]
    reviewed.to_csv(packet_path.parent/'agent_review_180.csv',index=False)
    findings=reviewed.loc[reviewed.agent_requires_review,['region','title_sha256','agent_notes']]
    findings.to_csv(out/'agent_findings.csv',index=False)
    table=s.rename(columns={'region':'Код','name':'Область','articles':'Заголовки','no_municipality_id':'Без конкретного муниципалитета',
        'explicit_municipalities':'Муниципалитетов упомянуто','economic_municipalities':'С экономическими новостями категории',
        'invalid_historical_ids':'Недопустимые ID по году','differing_recomputed_ids':'Отличия повторной привязки','unresolved_lookup':'Неоднозначные/не найденные имена'})
    lookup=pd.DataFrame(missing,columns=['region','name','year','title_sha256'])
    lookup_summary=lookup.groupby(['region','name','year']).size().reset_index(name='articles') if len(lookup) else lookup
    text=f'''# Проверка географии и смысла новостных меток

Проверены все 3600 меток и их муниципальные ID по версии официального справочника за год публикации. Дополнительно просмотрены все 180 заголовков ранее подготовленной выборки. У {len(findings)} заголовков записаны замечания к меткам, географии или правилам отбора. Это число кандидатов на повторную проверку, а не доля ошибок всего корпуса: выборка специально содержит спорные случаи. Проверка агентом не заменяет независимую разметку человеком.

## География

{table.to_markdown(index=False)}

Отсутствие муниципального ID не всегда ошибка: новость может относиться ко всей области или стране. Однако региональные формы («оренбуржец», «костромичи») подтверждают только область в текущем правиле. Если область подтверждена, категория подходит, а конкретный муниципалитет не найден и география не признана неоднозначной, новость применяется ко всем муниципалитетам региона. Это может размывать местный сигнал. «Под Оренбургом» тоже нельзя автоматически считать событием в самом городе.

Повторная привязка воспроизводит правила сбора: явные варианты названия из конфигурации плюс названия районов/округов из ежегодного справочника. Общий справочник содержит исторические версии: название города может перестать быть названием отдельного муниципалитета после объединения.

Не найденные в годовой версии названия, встретившиеся в заголовках:

{lookup_summary.to_markdown(index=False) if len(lookup_summary) else 'Таких случаев нет.'}

Конкретные пропуски: Гай и форма «орчанка» для Орска; Домбаровский без слова «район»; Володарск вместо Володарского района; Мантуровская больница. Асекеево, Сакмара и городские микрорайоны требуют отдельных проверенных вариантов названия. Для неоднозначных названий вроде Орловки или Заволжья одной строки заголовка недостаточно. Привязку к области по одному источнику включать нельзя: в выборке есть новости других областей и федеральные сообщения.

## Метки

Систематические проблемы: LLM путает изменение цен с ростом зарплат, продаж, потребления или стоимости строительства; предложение/попытку выплаты — с полученным доходом; инфраструктуру — с открытием бизнеса. Встречаются неправильные категории: цветы как продукты, рыбалка как транспорт, каток как кафе, лечение по ОМС как общий доход. Это содержательные ошибки, которые проверка JSON не обнаруживает.

Все замечания по проверенной выборке (идентификатор позволяет найти исходный заголовок локально):

{findings.rename(columns={'region':'Регион','title_sha256':'ID заголовка','agent_notes':'Замечание'}).to_markdown(index=False)}

## Что менять дальше

1. В проверенной словарной таблице хранить варианты названия и муниципальный ID с годами действия, включая объединения; различать город, район, микрорайон и жителя области.
2. В инструкции LLM явно разделить цену, объём продаж, зарплату, бюджет проекта, количество рейсов и продолжительность ремонта. Отдельно уточнить будущие планы и уже выполненные изменения.
3. Повторно разметить спорные заголовки; сверить часть с человеком. После исправления проверить новую отдельную выборку и заново сравнить модели.

В этом разборе исторические метки, география и прогнозы не заменены: для честного сравнения исправления должны стать отдельным экспериментом. Все 180 строк с замечаниями сохранены локально в data/inputs/news_label_review/agent_review_180.csv; поля human_* не заполнены агентом. В Git — счётчики, хеши и замечания без исходных заголовков. Воспроизведение: `python scripts/check_news_geography.py`.
'''
    from src.report_language import explain_text
    text=explain_text(text)
    (ROOT/'docs/NEWS_GEOGRAPHY_REVIEW_RU.md').write_text(text)
    (out/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'packet_sha256':hash_file(packet_path),
        'code_sha256':hash_file(Path(__file__)),'reviewed_by_agent':180,'requires_review':len(findings),'labels_changed':False,
        'human_review_completed':False},ensure_ascii=False,indent=2)+'\n')
    print(s.to_string(index=False));print('Agent review findings:',len(findings));print(lookup_summary.to_string(index=False))

if __name__=='__main__':main()
