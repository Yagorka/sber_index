"""Защита опубликованного/замороженного прогноза от неявной перезаписи."""
import hashlib


def preserve_or_create(path, content, candidate_path):
    """Новый вариант сохраняется отдельно; существующий forecast никогда не меняется."""
    if isinstance(content,str):
        content=content.encode("utf-8")
    created=not path.exists()
    different=False
    if created:
        path.write_bytes(content)
    elif path.read_bytes()!=content:
        candidate_path.parent.mkdir(parents=True,exist_ok=True)
        candidate_path.write_bytes(content)
        different=True
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return created,different,digest
