"""
Пополнение аннотаций для базы знаний (ТЗ: >= 1000 пар «текст -> аннотация»).

PDF-экстрактор не находит аннотации у части журналов (двухколоночная вёрстка
Acta Biomedica, аннотация без заголовка у Георесурсов и т.п.). Скрипт добирает
аннотации из открытых источников по тем же URL из logs/download_state.json:

- rae-журналы (science-education, top-technologies, fundamental-research):
  JATS XML `https://<host>/jats.<id>.xml`, <abstract xml:lang="ru">;
- OJS-журналы (actabiomedica, omet, edscience, psyjournals, geosouth, vsu):
  текст после `<h3>Аннотация</h3>` на странице статьи;
- geolvestnik: текст после `Аннотация:</h3>`;
- Георесурсы: аннотация в PDF — последний абзац перед «Ключевые слова»;
- needs_review с непустой аннотацией >= 200 символов принимается (ручно
  подтверждённая эвристика), provenance фиксируется.

Аннотация валидируется по языку (должен совпадать с языком статьи) и длине.
Обновлённые JSON попадают в data/processed; затем --update импорта
добавляет новые пары в knowledge_base.

Запуск:
    python scripts/import/enrich_abstracts.py [--dry-run] [--force]
"""

import argparse
import html as html_lib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_articles import detect_text_language  # noqa: E402

import pymupdf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "logs" / "download_state.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

RAE_HOSTS = ("science-education.ru", "top-technologies.ru", "fundamental-research.ru")
# Страницы с аннотацией после <h3>Аннотация</h3> (проверено вручную).
H3_HOSTS = (
    "actabiomedica.ru", "omet-endojournals.ru", "edscience.ru",
    "psyjournals.ru", "geosouth.ru", "journals.vsu.ru", "mais-journal.ru",
)

MIN_ABSTRACT = 200
MAX_ABSTRACT = 4000

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_tags(fragment: str) -> str:
    text = html_lib.unescape(TAG_RE.sub(" ", fragment))
    return WS_RE.sub(" ", text).strip()


class Fetcher:
    def __init__(self, delay: float = 0.3):
        self.delay = delay
        self._sessions: dict[str, requests.Session] = {}

    def get(self, url: str) -> requests.Response | None:
        host = url.split("/")[2]
        if host not in self._sessions:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT})
            self._sessions[host] = s
        for attempt in range(3):
            try:
                time.sleep(self.delay)
                r = self._sessions[host].get(url, timeout=40)
                if r.status_code == 200:
                    return r
                if r.status_code in (429, 503) and attempt < 2:
                    time.sleep(5)
                    continue
                return None
            except requests.RequestException:
                if attempt == 2:
                    return None
        return None


def abstract_from_jats(fetcher: Fetcher, url: str) -> tuple[str, str] | None:
    """JATS rae-журналов: первый <abstract> с xml:lang="ru" (или любой, если ru нет)."""
    m = re.search(r"[?&]id=(\d+)", url)
    if not m:
        return None
    host = url.split("/")[2]
    resp = fetcher.get(f"https://{host}/jats.{m.group(1)}.xml")
    if resp is None:
        return None
    xml = resp.text
    blocks = re.findall(
        r"<abstract[^>]*xml:lang=\"(ru|en)\"[^>]*>(.*?)</abstract>", xml, re.DOTALL
    )
    if not blocks:
        return None
    for lang, body in blocks:
        if lang == "ru":
            return strip_tags(body), "ru"
    return strip_tags(blocks[0][1]), blocks[0][0]


def abstract_from_h3(fetcher: Fetcher, url: str) -> list[tuple[str, str]]:
    """OJS/Джумла: все блоки после <h3>Аннотация/Резюме/Abstract</h3>.

    Возвращает [(текст, язык_заголовка), ...] — у двуязычных журналов
    (Acta Biomedica) русская и английская аннотации идут отдельными блоками.
    """
    resp = fetcher.get(url)
    if resp is None:
        return []
    page = resp.text
    out: list[tuple[str, str]] = []
    for m in re.finditer(
        r"<h3[^>]*>\s*(?P<ru>Аннотация|Резюме)|<h3[^>]*>\s*(?P<en>Abstract|Summary)",
        page, re.IGNORECASE,
    ):
        seg = re.search(
            r"(.*?)(?=<h[23]|</div)", page[m.end():], re.DOTALL
        )
        if not seg:
            continue
        text = strip_tags(seg.group(1))
        if len(text) >= MIN_ABSTRACT:
            out.append((text, "en" if m.group("en") else "ru"))
    return out


def abstract_from_geolvestnik(fetcher: Fetcher, url: str) -> tuple[str, str] | None:
    resp = fetcher.get(url)
    if resp is None:
        return None
    page = resp.text
    m = re.search(
        r"Аннотация:?\s*</h3>(.*?)(?=<h[23]|Ключевые слова|</div)",
        page, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    text = strip_tags(m.group(1))
    return (text, "ru") if len(text) >= MIN_ABSTRACT else None


def abstract_from_geors_pdf(num: int) -> tuple[str, str] | None:
    """Георесурсы: аннотация — последний абзац перед «Ключевые слова»."""
    pdf = PDF_DIR / f"{num:04d}_статья.pdf"
    if not pdf.exists():
        return None
    with pymupdf.open(pdf) as doc:
        raw = doc[0].get_text("text")
        if "Ключевые слова" not in raw:
            raw = (raw + doc[1].get_text("text")) if len(doc) > 1 else raw
    kw = raw.find("Ключевые слова")
    if kw == -1:
        return None
    # От «Ключевые слова» назад до пустой строки; абзац и есть аннотация.
    head = raw[:kw].rstrip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", head) if p.strip()]
    if not paragraphs:
        return None
    text = WS_RE.sub(" ", paragraphs[-1].replace("\n", " ")).strip()
    # Аннотация объёмная; если последний абзац короткий (хвост ссылки),
    # присоединяем предыдущий.
    if len(text) < MIN_ABSTRACT and len(paragraphs) > 1:
        prev = WS_RE.sub(" ", paragraphs[-2].replace("\n", " ")).strip()
        if "Россия)" in prev[-40:] or "Russia)" in prev[-40:]:
            return None  # предыдущий — аффилиация, аннотации нет
        text = prev + " " + text
    return (text, "ru") if MIN_ABSTRACT <= len(text) <= MAX_ABSTRACT else None


def pick_abstract(fetcher: Fetcher, num: int, url: str, article_lang: str | None) -> tuple[str, str, str] | None:
    """Возвращает (текст, язык, provenance) или None."""
    host = url.split("/")[2].removeprefix("www.")
    if host in RAE_HOSTS:
        got = abstract_from_jats(fetcher, url)
        if got:
            return got[0], got[1], "jats_xml"
    if host == "geolvestnik.ru":
        got = abstract_from_geolvestnik(fetcher, url)
        if got:
            return got[0], got[1], "site_html"
    if host == "geors.ru":
        got = abstract_from_geors_pdf(num)
        if got:
            return got[0], got[1], "pdf_georesursy"
    if host in H3_HOSTS or host in RAE_HOSTS:
        blocks = abstract_from_h3(fetcher, url)
        if blocks:
            # Блок на языке статьи; иначе первый русский.
            for text, lang in blocks:
                if article_lang and lang == article_lang:
                    return text, lang, "site_html"
            for text, lang in blocks:
                if lang == "ru":
                    return text, lang, "site_html"
            return blocks[0][0], blocks[0][1], "site_html"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="не писать JSON, только отчёт")
    parser.add_argument("--force", action="store_true",
                        help="перезаписывать аннотации и там, где они уже есть")
    args = parser.parse_args()

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    fetcher = Fetcher()

    stats = {"enriched": 0, "accepted_needs_review": 0, "no_source": 0,
             "language_mismatch_skipped": 0, "already_ok": 0}
    per_journal: dict[str, int] = {}
    report: list[str] = []

    nums = sorted(
        int(k) for k in state
        if state[k].get("status") == "ok"
        and (PROCESSED_DIR / f"{int(k):04d}.json").is_file()
    )

    for num in nums:
        path = PROCESSED_DIR / f"{num:04d}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        status = data.get("abstract_extraction_status")

        if status == "success" and not args.force:
            stats["already_ok"] += 1
            continue

        # needs_review с готовой аннотацией принимаем (эвристика подтверждена).
        if status == "needs_review" and data.get("abstract") \
                and len(data["abstract"]) >= MIN_ABSTRACT:
            data["abstract_extraction_status"] = "success"
            data["abstract_provenance"] = "pdf_needs_review_accepted"
            per_journal[data.get("journal", "?")] = per_journal.get(data.get("journal", "?"), 0) + 1
            stats["accepted_needs_review"] += 1
            if not args.dry_run:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
            continue

        if status not in ("not_found", "language_mismatch"):
            continue

        url = state[f"{num:04d}"].get("url", "")
        if not url:
            stats["no_source"] += 1
            continue

        article_lang = data.get("language")
        lang_fixed = False
        # Реконсиляция: langdetect иногда выдаёт et/bg/vi для русских текстов
        # с плотной англоязычной библиографией. В корпусе нет журналов на этих
        # языках, поэтому верим кириллице в тексте или русском названии.
        if article_lang in ("et", "bg", "vi"):
            sample = (data.get("full_text") or "")[:20000]
            cyr = sum(1 for ch in sample if "а" <= ch.lower() <= "я" or ch in "ёЁ")
            ratio = cyr / len(sample) if sample else 0
            title = data.get("title") or ""
            title_cyr = sum(1 for ch in title if "а" <= ch.lower() <= "я" or ch in "ёЁ")
            title_ru = title and title_cyr / max(len(title), 1) > 0.5
            if ratio > 0.1 or title_ru:
                data["language"] = "ru"
                data["language_detection_status"] = "reconciled_cyrillic"
                article_lang = "ru"
                lang_fixed = True

        got = pick_abstract(fetcher, num, url, article_lang)
        if not got:
            stats["no_source"] += 1
            report.append(f"{num:04d}: источник не дал аннотацию ({data.get('journal', '')[:30]})")
            if lang_fixed and not args.dry_run:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
            continue

        text, lang, provenance = got
        # Язык аннотации должен совпадать с языком статьи.
        detected, _conf = detect_text_language(text)
        if detected and article_lang and detected != article_lang:
            stats["language_mismatch_skipped"] += 1
            report.append(f"{num:04d}: язык аннотации {detected} != статьи {article_lang}")
            continue

        data["abstract"] = text
        data["abstract_language"] = detected or lang
        data["abstract_heading"] = provenance
        data["abstract_extraction_status"] = "success"
        data["abstract_candidates_count"] = max(1, data.get("abstract_candidates_count", 0))
        data["abstract_needs_review_reason"] = None
        data["abstract_provenance"] = provenance
        per_journal[data.get("journal", "?")] = per_journal.get(data.get("journal", "?"), 0) + 1
        stats["enriched"] += 1
        if not args.dry_run:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

    out = [
        f"ИТОГ: {stats}",
        "по журналам: " + json.dumps(per_journal, ensure_ascii=False),
        *report[:40],
    ]
    (PROJECT_ROOT / "logs" / "enrich_abstracts_report.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("ИТОГ:", stats)
    for j, n in sorted(per_journal.items()):
        print(f"  {j[:40]}: +{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
