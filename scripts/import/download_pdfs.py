"""
Скачивание PDF-статей из реестра data/статьи.md в data/raw/pdfs.

- обрабатывает статьи с номерами >= --from-num (по умолчанию первая отсутствующая);
- файлы именуются по реестру: NNNN_статья.pdf;
- под каждый журнал свой способ поиска PDF (метатег citation_pdf_url,
  galley-ссылки OJS, прямые ссылки на файл), итог проверяется по magic bytes %PDF;
- метаданные (название, авторы, журнал, DOI, год, лицензия) собираются
  со страниц статей и сохраняются в logs/download_state.json для последующего
  обновления licenses.md и article_categories.json;
- идемпотентность: state-файл позволяет прервать и продолжить загрузку;
- дубли URL (например, esj.pnzgu.ru) не скачиваются повторно — файл копируется.

Запуск:
    python scripts/import/download_pdfs.py [--from-num N] [--to-num N] \
        [--only 131,140,188] [--delay 0.35]
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_FILE = PROJECT_ROOT / "data" / "статьи.md"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"
STATE_FILE = PROJECT_ROOT / "logs" / "download_state.json"

# Хосты, у которых полный текст публикуется только в HTML (PDF нет) — ИМЛИ РАН.
NO_PDF_HOSTS = ("litda.ru", "www.litda.ru", "litfact.ru", "www.litfact.ru")

# Хосты издательства «Академия Естествознания»: политика CC BY 4.0.
RAE_HOSTS = ("science-education.ru", "top-technologies.ru", "fundamental-research.ru")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

ROW_RE = re.compile(r"^\|\s*(\d{4})_статья\s*\|\s*(https?://\S+?)\s*\|?\s*$")
PDF_HREF_RE = re.compile(r'href="([^"]+?\.pdf[^"]*?)"', re.IGNORECASE)
META_RE = {
    name: re.compile(rf'<meta\s+name="citation_{name}"\s+content="([^"]*)"', re.IGNORECASE)
    for name in ("title", "author", "journal_title", "doi", "date", "publication_date", "pdf_url")
}
CC_RE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)", re.IGNORECASE)
OJS_GALLEY_RE = re.compile(r"""href="[^"]*?/article/view/(\d+)/(\d+)""")


def load_registry() -> dict[int, str]:
    """Читает реестр статьи.md → {номер: url}."""
    items: dict[int, str] = {}
    for line in REGISTRY_FILE.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            items[int(m.group(1))] = m.group(2)
    return items


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
    )


class Fetcher:
    """HTTP-сессии с переиспользованием cookie per-host (нужны rae-сайтам)."""

    def __init__(self, delay: float):
        self.delay = delay
        self._sessions: dict[str, requests.Session] = {}

    def session(self, url: str) -> requests.Session:
        host = urlparse(url).netloc
        if host not in self._sessions:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"})
            self._sessions[host] = s
        return self._sessions[host]

    def get(self, url: str, *, binary: bool = False, allow_redirects: bool = True) -> requests.Response:
        for attempt in range(3):
            try:
                time.sleep(self.delay)
                r = self.session(url).get(
                    url, timeout=40, allow_redirects=allow_redirects
                )
                if r.status_code in (429, 503) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                return r
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"unreachable: {url}")  # pragma: no cover


def decode(r: requests.Response) -> str:
    if not r.encoding:
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def meta_all(html: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, rx in META_RE.items():
        out[name] = [m.strip() for m in rx.findall(html) if m.strip()]
    return out


def pdf_candidates(url: str, html: str) -> list[str]:
    """Кандидаты на PDF: citation_pdf_url, затем специфика хоста."""
    base_host = urlparse(url).netloc.lower().removeprefix("www.")
    cands: list[str] = []
    meta = meta_all(html)
    if meta["pdf_url"]:
        cands.append(urljoin(url, meta["pdf_url"][0]))

    if base_host in NO_PDF_HOSTS:
        return cands

    # Прямые ссылки на .pdf внутри страницы (esj, rusklassika, psyjournals, geolvestnik, sfu).
    en_candidates: list[str] = []
    for href in PDF_HREF_RE.findall(html):
        if base_host == "geolvestnik.ru" and not href.startswith(("http", "/")):
            # geolvestnik отдаёт пути от корня сайта без ведущего слэша.
            full = urljoin(url, "/").rstrip("/") + "/" + href
        else:
            full = urljoin(url, href)
        parsed = urlparse(full)
        if parsed.netloc.lower().endswith("pnzgu.ru") and "/doki/" in parsed.path:
            continue  # служебные документы ПензГУ (cookies, персональные данные)
        if base_host == "psyjournals.ru" and parsed.path.startswith("/en/"):
            # англоязычная версия статьи — только если русской PDF нет.
            en_candidates.append(full)
            continue
        cands.append(full)
    cands.extend(en_candidates)

    # OJS: galley view/ID/GALLEY -> download/ID/GALLEY (actabiomedica, edscience,
    # omet, geors, geosouth, vsu, narfu). Нулевой galley — не PDF, пропускаем.
    article_id = re.search(r"/article/view/(\d+)", url)
    if article_id:
        seen: set[str] = set()
        for galley in OJS_GALLEY_RE.findall(html):
            aid, gid = galley
            if aid == article_id.group(1) and gid != "0":
                dl = re.sub(r"/article/view/", "/article/download/", url)
                dl = re.sub(rf"/article/(view|download)/\d+.*", f"/article/download/{aid}/{gid}", dl)
                if dl not in seen:
                    seen.add(dl)
                    cands.append(dl)

    # pmi.spmi.ru прячет ссылку в JS-разметке.
    for m in re.finditer(rf'"(https?://[^"]*?/article/download/{article_id.group(1) if article_id else ""}/\d+)"', html):
        cands.append(m.group(1))

    # rae-сайты: s.<host>/pdf/<год>/<номер>/<id>.pdf, если метатега не было.
    if base_host in RAE_HOSTS:
        m_id = re.search(r"[?&]id=(\d+)", url)
        year = (meta.get("publication_date") or meta.get("date") or [""])[0][:4]
        issue = ""
        m_issue = re.search(r'<meta\s+name="citation_issue"\s+content="(\d+)"', html, re.IGNORECASE)
        if m_issue:
            issue = m_issue.group(1)
        if m_id and year and issue:
            cands.append(f"https://s.{base_host}/pdf/{year}/{issue}/{m_id.group(1)}.pdf")

    # psyjournals: у редакционных колонок ссылка строится по слагу.
    if base_host == "psyjournals.ru" and not cands:
        m = re.search(r"/archive/([^/]+)/([^/?#]+)$", url)
        if m:
            cands.append(f"{urljoin(url, '/')}journals/pse/archive/{m.group(1)}/pse_{m.group(1)}_{m.group(2)}.pdf")

    # Уникальные, http->https, относительные -> абсолютные.
    out: list[str] = []
    for c in cands:
        c = urljoin(url, c)
        if c.startswith("http://"):
            c = "https://" + c[7:]
        if c not in out:
            out.append(c)
    return out


def looks_like_pdf(resp: requests.Response) -> bool:
    return resp.content[:5] == b"%PDF-" and len(resp.content) > 5000


def extract_metadata(url: str, html: str, resp_url: str) -> dict[str, Any]:
    meta = meta_all(html)
    title = (meta["title"] or [""])[0]
    if not title:
        m = re.search(r'<meta\s+property="og:title"\s+content="([^"]*)"', html, re.IGNORECASE)
        title = m.group(1) if m else ""
    if not title:
        m = re.search(r"<h1[^>]*>\s*([^<]{5,300}?)\s*</h1>", html, re.IGNORECASE)
        title = m.group(1) if m else ""
    if not title:
        m = re.search(r"<title>([^<]*)</title>", html, re.IGNORECASE)
        title = (m.group(1).strip() if m else "") or "Название не определено"

    date = (meta["publication_date"] or meta["date"] or [""])[0]
    year = date[:4] if date[:4].isdigit() else ""
    journal = (meta["journal_title"] or [""])[0]
    if not journal:
        host = urlparse(resp_url).netloc.removeprefix("www.")
        journal = {
            "esj.pnzgu.ru": "Наука. Общество. Государство",
            "science-education.ru": "Современные проблемы науки и образования",
            "top-technologies.ru": "Современные наукоёмкие технологии",
            "fundamental-research.ru": "Фундаментальные исследования",
            "psyjournals.ru": "Психологическая наука и образование",
            "rusklassika.ru": "Русская классика",
            "geolvestnik.ru": "Геологический вестник",
            "pmi.spmi.ru": "Записки Горного института",
            "geosouth.ru": "Вестник геонаук",
            "journals.vsu.ru": "Вестник Воронежского государственного университета. Геология",
        }.get(host, "")

    cc = CC_RE.search(html)
    if cc:
        lic = f"CC {cc.group(1).upper()} {cc.group(2)}"
    elif urlparse(resp_url).netloc.removeprefix("www.") in RAE_HOSTS:
        lic = "CC BY 4.0"
    else:
        lic = "Лицензия не указана явно"

    doi = (meta["doi"] or [""])[0]
    if not doi:
        m = re.search(r"10\.\d{4,5}/[^\s\"<>]+", html)
        doi = m.group(0).rstrip(".,);") if m else ""

    return {
        "title": html_unescape(title.strip()),
        "authors": [html_unescape(a) for a in meta["author"]],
        "year": year,
        "journal": html_unescape(journal),
        "doi": doi,
        "license": lic,
        "source": url,
        "final_url": resp_url,
    }


def pdf_first_page_meta(path: Path) -> dict[str, Any]:
    """Название/авторы/год с первой страницы PDF — для страниц без метатегов.

    Эвристика: заголовок — строки до строки с авторами (паттерн «Фамилия И.О.»
    или «И.О. Фамилия») или до «Аннотация»/«Abstract».
    """
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            text = doc[0].get_text("text")
    except Exception:  # noqa: BLE001 — битый PDF не должен ломать загрузку
        return {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:25]
    # Отбрасываем шапку первой страницы: номер страницы, строку с годом и № выпуска,
    # ISSN-строки и URL.
    while lines and (
        re.fullmatch(r"[0-9\s.\-–—]+", lines[0])
        or re.match(r"^(ISSN|https?://|www\.)", lines[0])
        or (re.search(r"\b20\d\d\b", lines[0]) and len(lines[0]) < 130)
    ):
        lines = lines[1:]
    author_re = re.compile(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]?\.?)|([А-ЯЁ]\.\s?[А-ЯЁ]?\.?\s*[А-ЯЁ][а-яё]+)")
    stop_re = re.compile(r"^(Аннотация|Ключевые слова|Abstract|Keywords|DOI)", re.IGNORECASE)

    title_lines: list[str] = []
    authors: list[str] = []
    for ln in lines:
        if stop_re.match(ln) and title_lines:
            break
        if author_re.search(ln) and title_lines and len(ln) < 120:
            authors.append(ln)
            break
        if not authors:
            title_lines.append(ln)
    year_m = re.search(r"\b(20\d\d)\b", text)
    authors = [a for a in authors if "©" not in a]
    return {
        "title": " ".join(title_lines)[:300].strip(),
        "authors": [re.sub(r"[\d\u2009\u00a0\s]+$", "", a) for a in authors],
        "year": year_m.group(1) if year_m else "",
    }


def html_unescape(s: str) -> str:
    return (
        s.replace("&amp;", "&").replace("&quot;", '"')
        .replace("&#039;", "'").replace("&nbsp;", " ")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-num", type=int, default=None)
    parser.add_argument("--to-num", type=int, default=None)
    parser.add_argument("--only", type=str, default="", help="список номеров через запятую (для отладки)")
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--meta-only", action="store_true",
                        help="не скачивать PDF (уже скачаны вручную): собрать только метаданные")
    parser.add_argument("--force", action="store_true", help="игнорировать статус ok в state")
    args = parser.parse_args()

    registry = load_registry()
    if not registry:
        print("Реестр пуст — проверьте data/статьи.md", file=sys.stderr)
        return 1

    state = load_state()
    fetcher = Fetcher(args.delay)

    only = {int(x) for x in args.only.split(",") if x.strip().isdigit()} if args.only else None
    from_num = args.from_num
    if from_num is None and only is None:
        from_num = next(
            (n for n in sorted(registry) if not (PDF_DIR / f"{n:04d}_статья.pdf").exists()),
            max(registry) + 1,
        )

    nums = [
        n for n in sorted(registry)
        if (only is not None and n in only)
        or (only is None and from_num is not None and n >= from_num
            and (args.to_num is None or n <= args.to_num))
    ]

    ok = failed = no_pdf = skipped = 0
    for i, num in enumerate(nums, 1):
        key = f"{num:04d}"
        url = registry[num]
        target = PDF_DIR / f"{key}_статья.pdf"

        if not args.force and state.get(key, {}).get("status") == "ok" and target.exists():
            skipped += 1
            continue

        rec: dict[str, Any] = {"url": url}
        try:
            host = urlparse(url).netloc.removeprefix("www.")
            if args.meta_only:
                # Режим бэкфилла: PDF уже скачан вручную — только метаданные
                # со страницы статьи (плюс первая страница локального PDF).
                try:
                    resp = fetcher.get(url)
                    html = decode(resp) if resp.status_code == 200 else ""
                except requests.RequestException:
                    html = ""
                if html:
                    meta = extract_metadata(url, html, resp.url)
                else:
                    meta = {"title": "", "authors": [], "year": "", "journal": "",
                            "doi": "", "license": "", "source": url, "final_url": url}
                if target.exists():
                    bad_title = (
                        not meta["title"]
                        or meta["title"] in ("Информация о статье", "Название не определено", "Архив")
                        or meta["title"].startswith("Архив")
                    )
                    if bad_title or not meta["authors"] or not meta["year"]:
                        pm = pdf_first_page_meta(target)
                        if bad_title and pm.get("title"):
                            meta["title"] = pm["title"]
                        if not meta["authors"] and pm.get("authors"):
                            meta["authors"] = pm["authors"]
                        if not meta["year"] and pm.get("year"):
                            meta["year"] = pm["year"]
                rec.update(status="ok", file=str(target), meta=meta)
                state[key] = rec
                ok += 1
                continue
            if host in NO_PDF_HOSTS:
                rec.update(status="no_pdf", note="полный текст только в HTML, PDF на сайте отсутствует")
                state[key] = rec
                no_pdf += 1
                continue

            resp = fetcher.get(url)
            if resp.status_code != 200 or not resp.text.lstrip()[:100]:
                raise RuntimeError(f"article page HTTP {resp.status_code}")
            html = decode(resp)

            # Дубликат URL: копируем уже скачанный файл и метаданные.
            twin = next(
                (k for k, v in state.items()
                 if k != key and v.get("url") == url and v.get("status") == "ok"),
                None,
            )
            if twin and (PDF_DIR / f"{twin}_статья.pdf").exists():
                shutil.copyfile(PDF_DIR / f"{twin}_статья.pdf", target)
                meta = dict(state[twin].get("meta") or extract_metadata(url, html, resp.url))
                meta["source"] = url
                rec.update(status="ok", file=str(target), pdf_url=state[twin].get("pdf_url"),
                           meta=meta, note=f"дубликат URL статьи {twin}")
                state[key] = rec
                ok += 1
                continue

            meta = extract_metadata(url, html, resp.url)
            rec["meta"] = meta

            saved = False
            errors: list[str] = []
            for cand in pdf_candidates(url, html)[:6]:
                try:
                    r = fetcher.get(cand)
                    if looks_like_pdf(r):
                        target.write_bytes(r.content)
                        rec.update(status="ok", file=str(target), pdf_url=str(r.url))
                        saved = True
                        break
                    errors.append(f"{cand} -> HTTP {r.status_code}, {r.headers.get('Content-Type', '?')}")
                except requests.RequestException as exc:
                    errors.append(f"{cand} -> {type(exc).__name__}")
            if not saved:
                rec.update(status="failed", errors=errors[:6])
                failed += 1
            else:
                # Страницы без метатегов (esj, geolvestnik): добираем данные из PDF.
                bad_title = (
                    meta["title"] in ("Информация о статье", "Название не определено")
                    or meta["title"].startswith("Архив номеров")
                )
                if bad_title or not meta["authors"] or not meta["year"]:
                    pdf_meta = pdf_first_page_meta(target)
                    if bad_title and pdf_meta.get("title"):
                        meta["title"] = pdf_meta["title"]
                    if not meta["authors"] and pdf_meta.get("authors"):
                        meta["authors"] = pdf_meta["authors"]
                    if not meta["year"] and pdf_meta.get("year"):
                        meta["year"] = pdf_meta["year"]
                    rec["meta"] = meta
                ok += 1
        except Exception as exc:  # noqa: BLE001 — фиксируем любую ошибку в state
            rec.update(status="failed", errors=[f"{type(exc).__name__}: {exc}"])
            failed += 1

        state[key] = rec
        if i % 20 == 0 or i == len(nums):
            save_state(state)
            print(f"[{i}/{len(nums)}] ok={ok} failed={failed} no_pdf={no_pdf} skipped={skipped}", flush=True)

    save_state(state)
    print(f"ИТОГ: ok={ok} failed={failed} no_pdf={no_pdf} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
