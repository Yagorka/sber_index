#!/usr/bin/env bash
# Скачивает открытые данные конкурса СберИндекса (CC BY-SA 4.0) в data/raw/.
# Архив лежит на sberbank.com; на части систем нет российского корневого сертификата — тогда -k.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/external
if [ ! -f data/raw/consumption.parquet ]; then
  curl -sSkL --retry 3 -o data/raw/hackathonlicence.zip \
    https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip
  python - <<'PY'
import zipfile
z = zipfile.ZipFile("data/raw/hackathonlicence.zip")
for i in z.infolist():
    if i.is_dir():
        continue
    name = i.filename.split("/")[-1]
    if name.endswith(".pdf"):
        name = "license_and_description.pdf"
    open("data/raw/" + name, "wb").write(z.read(i))
PY
fi
# История ключевой ставки Банка России (для news/macro-признаков): см. data/external/README.md
ls -la data/raw
sha256sum data/raw/*.parquet
