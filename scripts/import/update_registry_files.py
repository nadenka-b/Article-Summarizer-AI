"""
Обновление data/import/licenses.md и data/import/article_categories.json
по результатам загрузки PDF (logs/download_state.json).

- licenses.md: добавляет строки для скачанных статей (идемпотентно — строки
  с существующим номером не дублируются);
- article_categories.json: добавляет field/subfield для новых статей.
  Классификация предварительная (по метаданным и названиям): сильные
  ключевые слова → журнал → ключевые слова внутри поля; журналы без
  citation_journal_title получают название по хосту.

Запуск:
    python scripts/import/update_registry_files.py [--licenses-only | --categories-only]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "logs" / "download_state.json"
LICENSES_FILE = PROJECT_ROOT / "data" / "import" / "licenses.md"
CATEGORIES_FILE = PROJECT_ROOT / "data" / "import" / "article_categories.json"

# Новые тематические категории (дополняют существующие IT-категории).
NEW_CATEGORIES = {
    "social_sciences": "Социально-экономические, правовые и общественные науки",
    "education": "Педагогика и образование",
    "psychology": "Психологические науки и психология образования",
    "philology": "Литературоведение и филология",
    "medicine": "Медицина и медико-биологические науки",
    "biology": "Биологические науки",
    "geology": "Науки о Земле (геология, география, геоэкология)",
    "mining_engineering": "Горное дело и геотехнологии",
    "engineering": "Технические науки и инженерия",
    "other": "Прочее (требует ручной классификации)",
}

# Журнал (citation_journal_title или фолбэк по хосту) → (field, subfield).
# subfield=None — определить по ключевым словам названия внутри field.
JOURNAL_FIELDS = {
    "Вестник ВГУ. Серия: География. Геоэкология": ("geology", "geoecology_geography"),
    "Вестник Воронежского государственного университета. Геология": ("geology", "regional_geology"),
    "Наука. Общество. Государство": ("social_sciences", None),
    "История и педагогика естествознания": ("education", None),
    "Современные проблемы науки и образования": (None, None),  # междисциплинарный
    "Психологическая наука и образование": ("psychology", None),
    "Русская классика": ("philology", "russian_classical_literature"),
    "Acta Biomedica": ("medicine", None),
    "Журнал медико-биологических исследований": ("medicine", None),
    "Журнал Сибирского федерального университета": (None, None),  # серии различаются
    "Современные наукоёмкие технологии": (None, None),  # междисциплинарный
    "Современные наукоемкие технологии": (None, None),  # то же без «ё» (как в метаданных)
    "Фундаментальные исследования": ("social_sciences", "economics"),
    "Journal of Mining Institute": ("mining_engineering", None),
    "Ожирение и метаболизм": ("medicine", "endocrinology"),
    "Записки Горного института": ("mining_engineering", None),
    "Геологический вестник": ("geology", "regional_geology"),
    "Георесурсы": ("geology", "georesources"),
    "Геология и Геофизика Юга России": ("geology", None),
    "Латинская Америка": ("philology", "latin_american_studies"),
    "Литературный факт": ("philology", "literary_history"),
    "Информатика. Экономика. Управление": (None, None),  # междисциплинарный
    "Вестник Кемеровского государственного университета. Серия: Политические, социологические и экономические науки": ("social_sciences", None),
}

# Название журнала по хосту, если citation_journal_title пуст.
HOST_JOURNAL = {
    "journals.vsu.ru": "Вестник ВГУ. Серия: География. Геоэкология",
    "esj.pnzgu.ru": "Наука. Общество. Государство",
    "edscience.ru": "История и педагогика естествознания",
    "science-education.ru": "Современные проблемы науки и образования",
    "psyjournals.ru": "Психологическая наука и образование",
    "rusklassika.ru": "Русская классика",
    "actabiomedica.ru": "Acta Biomedica",
    "journals.narfu.ru": "Журнал медико-биологических исследований",
    "journal.sfu-kras.ru": "Журнал Сибирского федерального университета",
    "top-technologies.ru": "Современные наукоёмкие технологии",
    "fundamental-research.ru": "Фундаментальные исследования",
    "omet-endojournals.ru": "Ожирение и метаболизм",
    "pmi.spmi.ru": "Записки Горного института",
    "geolvestnik.ru": "Геологический вестник",
    "geors.ru": "Георесурсы",
    "geosouth.ru": "Геология и Геофизика Юга России",
    "litda.ru": "Латинская Америка",
    "litfact.ru": "Литературный факт",
    "oajiem.com": "Информатика. Экономика. Управление",
}

# Сильные признаки, перекрывающие журнальный дефолт (минералогия/палеонтология
# внутри «Записок Горного института» — это уже геология; серии Журнала СФУ
# видны в начале названия статьи).
KEYWORD_RULES: list[tuple[str, str, str | None]] = [
    ("журнал сфу. биология", "biology", None),
    ("журнал сфу. техника", "engineering", None),
    ("журнал сфу. гуманитарн", "social_sciences", None),
    ("литолог", "geology", "sedimentology"),
    ("минералог", "geology", "mineralogy"),
    ("геохим", "geology", "geochemistry"),
    ("петролог", "geology", "petrography"),
    ("метеорит", "geology", "meteoritics"),
    ("кимберлит", "geology", "petrography"),
    ("стратиграф", "geology", "stratigraphy"),
    ("палеозойск", "geology", "stratigraphy"),
    ("ископаем", "geology", "paleontology"),
    ("фузулинид", "geology", "paleontology"),
    ("докембри", "geology", "stratigraphy"),
    ("изотопн", "geology", "isotope_geology"),
    ("тектоническ", "geology", "tectonics"),
    ("янтар", "geology", "mineralogy"),
]

# IT-признаки для междисциплинарных журналов (проверяются до общих полей).
IT_CROSS_RULES: list[tuple[str, str, str]] = [
    ("виртуальн", "engineering", "virtual_instrumenting"),
    ("рекомендат", "software_engineering", "recommender_systems"),
    ("нейросет", "ai_ml", "neural_networks"),
    ("нейронных сетей", "ai_ml", "neural_networks"),
    ("машинного обучения", "ai_ml", "machine_learning"),
    ("искусственного интеллекта", "ai_ml", "ai_applications"),
    ("генеративного искусственн", "ai_ml", "generative_ai"),
    ("языковых модел", "nlp", "llm"),
    ("информационной системы", "information_systems", "information_systems"),
    ("информационных систем", "information_systems", "information_systems"),
    ("classification", "ai_ml", "classification"),
    ("архитектурн", "software_engineering", "software_architecture"),
    ("regression models", "data_analysis", "regression_analysis"),
]

SUBFIELD_HINTS: dict[str, list[tuple[str, str]]] = {
    "social_sciences": [
        ("правов", "law"), ("конституци", "law"), ("уголовн", "law"), ("гражданск", "law"),
        ("финанс", "economics"), ("банк", "economics"), ("аудит", "accounting"),
        ("бухгалтер", "accounting"), ("налог", "public_finance"), ("бюджет", "public_finance"),
        ("рынок", "economics"), ("экономи", "economics"), ("инвестиц", "economics"),
        ("кредит", "economics"), ("предпринимат", "entrepreneurship"),
        ("туризм", "tourism_economics"), ("торгов", "trade"),
        ("менедж", "management"), ("лидерств", "management"), ("персонал", "hr_management"),
        ("государственн", "public_administration"), ("муниципальн", "public_administration"),
        ("философ", "philosophy"), ("социолог", "sociology"),
        ("социальн", "sociology"), ("общественн", "sociology"), ("политич", "political_science"),
        ("молодеж", "sociology"), ("образован", "education_studies"),
        ("студент", "education_studies"), ("вуз", "education_studies"), ("абитуриент", "education_studies"),
        ("цифров", "digital_economy"),
    ],
    "education": [
        ("истори", "history_of_science"), ("естествознан", "science_education"),
        ("высш", "higher_education"), ("студент", "higher_education"), ("вуз", "higher_education"),
        ("образовательн", "pedagogy"), ("образован", "pedagogy"),
        ("школьн", "school_education"), ("школьник", "school_education"),
        ("дошкольн", "preschool_education"), ("дополнит", "supplementary_education"),
        ("цифров", "digital_education"), ("искусственного интеллекта", "digital_education"),
        ("нейросет", "digital_education"), ("chatgpt", "digital_education"),
        ("gigachat", "digital_education"), ("языков*модел", "digital_education"),
        ("математич", "stem_education"), ("инженерн", "engineering_education"),
        ("иностранн", "language_teaching"), ("русского языка", "language_teaching"),
        ("иностранным языкам", "language_teaching"),
        ("компетенц", "competence_education"), ("педагог", "pedagogy"),
        ("психолог", "educational_psychology"), ("обучен", "teaching_methods"),
    ],
    "psychology": [
        ("цифров", "digital_psychology"), ("интернет", "cyberpsychology"),
        ("смартфон", "cyberpsychology"), ("онлайн", "digital_psychology"),
        ("виртуал", "digital_psychology"), ("chatgpt", "digital_psychology"),
        ("искусственного интеллекта", "digital_psychology"), ("gigachat", "digital_psychology"),
        ("дошкольн", "developmental_psychology"), ("подрост", "developmental_psychology"),
        ("школьник", "developmental_psychology"), ("детск", "developmental_psychology"),
        ("дети", "developmental_psychology"),
        ("возрастн", "developmental_psychology"),
        ("личност", "personality_psychology"), ("мотивац", "personality_psychology"),
        ("самооценк", "personality_psychology"), ("саморегул", "self_regulation"),
        ("регуляц", "self_regulation"), ("идентичн", "identity_psychology"),
        ("интеллект", "cognitive_psychology"), ("мышлени", "cognitive_psychology"),
        ("креативн", "cognitive_psychology"), ("когнитивн", "cognitive_psychology"),
        ("тревожн", "clinical_psychology"), ("стресс", "clinical_psychology"),
        ("копинг", "clinical_psychology"), ("выгоран", "occupational_psychology"),
        ("трудоустройств", "occupational_psychology"), ("профессиональн", "occupational_psychology"),
        ("семейн", "family_psychology"), ("родител", "family_psychology"),
        ("благополуч", "wellbeing"), ("социальн", "social_psychology"),
        ("групп", "social_psychology"), ("ценност", "social_psychology"),
        ("образован", "educational_psychology"), ("обучен", "educational_psychology"),
        ("учител", "educational_psychology"), ("педагог", "educational_psychology"),
    ],
    "philology": [
        ("чехов", "chekhov_studies"), ("пушкин", "pushkin_studies"),
        ("гогол", "classical_authors"), ("толстой", "classical_authors"),
        ("грибоедов", "classical_authors"), ("лесков", "classical_authors"),
        ("островск", "classical_authors"), ("гончаров", "classical_authors"),
        ("бунин", "emigration_literature"), ("ахматов", "emigration_literature"),
        ("перевод", "translation_studies"), ("американс", "american_studies"),
        ("латиноамерик", "latin_american_studies"), ("серб", "slavic_studies"),
    ],
    "medicine": [
        ("педиатр", "pediatrics"), ("новорожденн", "pediatrics"), ("детск", "pediatrics"),
        ("детей", "pediatrics"),
        ("кардио", "cardiology"), ("сердц", "cardiology"), ("сосудист", "cardiology"),
        ("онкол", "oncology"), ("рак", "oncology"), ("опухол", "oncology"),
        ("сахарн", "endocrinology"), ("диабет", "endocrinology"), ("тирео", "endocrinology"),
        ("гормон", "endocrinology"), ("метабол", "endocrinology"),
        ("инфекц", "infectious_diseases"), ("вирус", "infectious_diseases"),
        ("коронавирусн", "infectious_diseases"), ("туберкул", "infectious_diseases"),
        ("лепр", "infectious_diseases"),
        ("хирург", "surgery"), ("стоматолог", "dentistry"), ("зуб", "dentistry"),
        ("челюстн", "dentistry"), ("пародонт", "dentistry"),
        ("психиатр", "psychiatry"), ("невролог", "neurology"), ("мозг", "neurology"),
        ("церебральн", "neurology"), ("травм", "traumatology"), ("сустав", "orthopedics"),
        ("ревматоид", "rheumatology"), ("остео", "orthopedics"),
        ("офтальм", "ophthalmology"), ("глаз", "ophthalmology"), ("зрител", "ophthalmology"),
        ("дермат", "dermatology"), ("кожн", "dermatology"), ("волос", "dermatology"),
        ("миндалин", "ent"), ("носоглотк", "ent"), ("отоларингол", "ent"),
        ("гастро", "gastroenterology"), ("кишечн", "gastroenterology"),
        ("желудочн", "gastroenterology"), ("пищевод", "gastroenterology"),
        ("гепат", "hepatology"), ("почк", "nephrology"),
        ("пульмон", "pulmonology"), ("дыхан", "pulmonology"), ("легк", "pulmonology"),
        ("астм", "pulmonology"), ("бронх", "pulmonology"),
        ("кров", "hematology"), ("анеми", "hematology"), ("тромб", "hematology"),
        ("гинеколог", "gynecology"), ("беремен", "gynecology"), ("менструальн", "gynecology"),
        ("иммун", "immunology"), ("аллерг", "immunology"), ("микроб", "microbiology"),
        ("физиолог", "physiology"), ("реабилитац", "rehabilitation"),
        ("анестези", "anesthesiology"), ("судебно-медицин", "forensic_medicine"),
        ("эпидемиолог", "epidemiology"), ("фарма", "pharmacology"), ("лекарств", "pharmacology"),
        ("пациент", "clinical_medicine"), ("больн", "clinical_medicine"),
        ("лечен", "clinical_medicine"), ("диагност", "diagnostics"),
        ("клинич", "clinical_medicine"), ("синдром", "clinical_medicine"),
        ("заболеван", "clinical_medicine"), ("болезн", "clinical_medicine"),
        ("терап", "clinical_medicine"), ("медицин", "general_medicine"),
    ],
    "mining_engineering": [
        ("угольн", "coal_mining"), ("карьер", "open_pit_mining"),
        ("шахт", "mining_operations"), ("горных работ", "mining_operations"),
        ("вскрыти", "mining_operations"), ("закладочн", "backfilling"),
        ("крепи", "mine_support"), ("выработок", "mine_support"),
        ("буров", "drilling"), ("скважин", "drilling"), ("тампонажн", "drilling"),
        ("нефт", "petroleum_engineering"), ("газов", "petroleum_engineering"),
        ("гидрат", "petroleum_engineering"), ("пропант", "petroleum_engineering"),
        ("обогащен", "mineral_processing"), ("флотаци", "mineral_processing"),
        ("рекультивац", "environmental_reclamation"), ("отход", "industrial_waste"),
        ("техногенн", "industrial_waste"), ("ремедиац", "environmental_reclamation"),
        ("биоремедиац", "environmental_reclamation"),
        ("геомехан", "geomechanics"),
        ("электроснабжен", "mine_power"), ("электрооборудован", "mine_power"),
        ("аэролог", "mine_aeration"), ("вентиляц", "mine_aeration"),
        ("экономич", "mining_economics"), ("экологич", "environmental_engineering"),
        ("безопасн", "occupational_safety"), ("охран", "occupational_safety"),
        ("геофиз", "mining_geophysics"), ("гидрогеолог", "hydrogeology"),
    ],
    "geology": [
        ("нефт", "petroleum_geology"), ("нефтегаз", "petroleum_geology"),
        ("газов", "petroleum_geology"),
        ("месторожден", "mineral_deposits"), ("руд", "mineral_deposits"),
        ("минерал", "mineralogy"), ("петрограф", "petrography"), ("гранитоид", "petrography"),
        ("фаци", "sedimentology"), ("осадк", "sedimentology"), ("отложен", "sedimentology"),
        ("палеонт", "paleontology"), ("фораминиф", "paleontology"), ("фаун", "paleontology"),
        ("геохим", "geochemistry"), ("геофиз", "geophysics"), ("сейсм", "seismology"),
        ("гидрогеолог", "hydrogeology"), ("инженерная геология", "engineering_geology"),
        ("инженерно-геолог", "engineering_geology"),
        ("геоэколог", "geoecology"), ("ландшафт", "geoecology"), ("почв", "geoecology"),
        ("тектоник", "tectonics"), ("складчат", "tectonics"),
        ("четвертичн", "quaternary_geology"), ("гляциал", "quaternary_geology"),
        ("картограф", "cartography"), ("георадиолокац", "geophysics"),
        ("геотерм", "geothermics"), ("радон", "geochemistry"),
        ("горн", "mining_geology"), ("геодинам", "geodynamics"),
    ],
    "biology": [
        ("флор", "botany"), ("растен", "botany"), ("водоросл", "botany"),
        ("фаун", "zoology"), ("животн", "zoology"), ("птиц", "zoology"),
        ("цыплят", "zoology"), ("насеком", "entomology"),
        ("рыб", "hydrobiology"), ("гидробиолог", "hydrobiology"),
        ("микроорганизм", "microbiology"), ("бактери", "microbiology"),
        ("штамм", "microbiology"), ("гриб", "mycology"), ("лишайник", "mycology"),
        ("экосистем", "ecology"), ("биоразнообраз", "ecology"),
        ("ореховодств", "crop_science"), ("грецкого", "crop_science"),
        ("фисташк", "crop_science"),
        ("микроклональн", "plant_biotechnology"),
        ("сточн", "environmental_chemistry"), ("коагулянт", "environmental_chemistry"),
        ("генет", "genetics"), ("геном", "genomics"), ("ген", "molecular_biology"),
        ("клетк", "cell_biology"), ("физиолог", "physiology"), ("биохим", "biochemistry"),
        ("почв", "soil_science"), ("агро", "agroecology"),
    ],
    "engineering": [
        ("сварочн", "manufacturing"), ("сварн", "manufacturing"),
        ("детали", "manufacturing"), ("станочн", "manufacturing"),
        ("манипулятор", "robotics"), ("робот", "robotics"),
        ("сенсорн", "sensor_networks"), ("беспроводн", "sensor_networks"),
        ("орбит", "aerospace"), ("спутник", "aerospace"),
        ("двигател", "mechanical_engineering"), ("механическ", "mechanical_engineering"),
        ("автоматизац", "automation"), ("автоматизированн", "automation"),
        ("двойник", "digital_twin"), ("цифров", "digital_technologies"),
        ("материал", "materials_engineering"), ("керамик", "materials_engineering"),
        ("бетон", "materials_engineering"), ("стал", "materials_engineering"),
        ("электротехн", "electrical_engineering"), ("энерг", "power_engineering"),
        ("транспорт", "transport_engineering"), ("автомобилестроен", "automotive_engineering"),
        ("логистик", "supply_chain"), ("цепи поставок", "supply_chain"),
        ("нечетк", "intelligent_systems"), ("алгоритм", "intelligent_systems"),
        ("нейросет", "intelligent_systems"), ("машинного обучения", "intelligent_systems"),
    ],
}

# Дефолтное поле для междисциплинарных журналов, если ключевые слова не сработали.
CROSS_FIELD_DEFAULTS = {
    "Современные проблемы науки и образования": ("education", "general"),
    "Современные наукоёмкие технологии": ("engineering", "general_engineering"),
    "Журнал Сибирского федерального университета": ("biology", "general"),
}

# Приоритет полей при междисциплинарной классификации по ключевым словам.
FIELD_HINT_PRIORITY = ["medicine", "geology", "mining_engineering", "education",
                       "social_sciences", "biology", "psychology", "engineering", "philology"]

USAGE_TEXT = "Скачан PDF-оригинал статьи для корпуса; извлечение текста — пайплайном подготовки данных."
CHANGES_TEXT = "Без изменений (исходный PDF); текст будет очищен при обработке пайплайном."


def esc(s: str) -> str:
    """Экранирование для ячейки markdown-таблицы."""
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def hint_subfield(field: str, title_lower: str) -> str:
    for frag, sub in SUBFIELD_HINTS.get(field, []):
        if frag in title_lower:
            return sub
    return "general"


def classify(meta: dict) -> tuple[str, str]:
    """Приблизительная классификация: сильные слова → журнал → слова поля.

    Сравнение журналов — без учёта «ё» (метаданные сайтов непоследовательны).
    """
    title = (meta.get("title") or "").lower()

    for frag, field, subfield in KEYWORD_RULES:
        if frag in title:
            return field, subfield or hint_subfield(field, title)

    journal = (meta.get("journal") or "").strip()
    jnorm = journal.replace("ё", "е")

    def journal_lookup(table: dict) -> tuple | None:
        """Точное совпадение или совпадение по префиксу (журналы часто
        дополняются английским названием через « - »)."""
        if not jnorm:
            return None
        best: tuple | None = None
        for key, value in table.items():
            k = key.replace("ё", "е")
            if jnorm == k or jnorm.startswith(k) or k.startswith(jnorm):
                if best is None or len(k) > len(best[0].replace("ё", "е")):
                    best = (key, value)
        return best[1] if best else None

    hit = journal_lookup(JOURNAL_FIELDS)
    if hit and hit[0]:
        return hit[0], hit[1] or hint_subfield(hit[0], title)

    # Междисциплинарные журналы: IT-признаки, затем поля по ключевым словам.
    for frag, field, subfield in IT_CROSS_RULES:
        if frag in title:
            return field, subfield
    for fname in FIELD_HINT_PRIORITY:
        hints = SUBFIELD_HINTS.get(fname, [])
        strong = [f for f, _ in hints if len(f) >= 5]
        if any(f in title for f in strong):
            return fname, hint_subfield(fname, title)

    hit = journal_lookup(CROSS_FIELD_DEFAULTS)
    if hit:
        return hit

    return "other", "uncategorized"


def enrich(num: int, rec: dict) -> dict:
    """Дополняет метаданные: журнал по хосту, если citation_journal_title пуст."""
    meta = dict(rec.get("meta") or {})
    if not meta.get("journal"):
        host = urlparse(meta.get("final_url") or rec.get("url") or "").netloc.removeprefix("www.")
        meta["journal"] = HOST_JOURNAL.get(host, "")
    meta["num"] = num
    return meta


def load_state_nums() -> dict[int, dict]:
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in state.items()}


def update_licenses(nums: dict[int, dict]) -> None:
    lines = LICENSES_FILE.read_text(encoding="utf-8").splitlines()

    # Строки 101+ полностью перегенерируются из state (идемпотентно);
    # исторические строки 1–100 не трогаем. Если данных нет — поле пустое
    # (по договорённости), строка с номером всё равно попадает в реестр.
    new_rows: dict[str, str] = {}
    for num in sorted(nums):
        rec = nums[num]
        if rec.get("status") != "ok" or num < 101:
            continue
        meta = enrich(num, rec)
        new_rows[str(num)] = (
            "| {n} | {title} | {authors} | {year} | {journal} | {doi} | {src} | {lic} | {use} | {chg} | ".format(
                n=num,
                title=esc(meta.get("title")),
                authors=esc("; ".join(meta.get("authors") or [])),
                year=esc(meta.get("year", "")),
                journal=esc(meta.get("journal", "")),
                doi=esc(meta.get("doi")),
                src=esc(meta.get("source", rec.get("url", ""))),
                lic=esc(meta.get("license", "")),
                use=USAGE_TEXT,
                chg=CHANGES_TEXT,
            )
        )
    if not new_rows:
        print("licenses.md: новых строк нет")
        return

    placed: set[str] = set()
    out: list[str] = []
    for ln in lines:
        m = re.match(r"^\|\s*(\d+)\s*\|", ln)
        if m and m.group(1) in new_rows:
            out.append(new_rows[m.group(1)])  # замена существующей строки
            placed.add(m.group(1))
        else:
            out.append(ln)
    out.extend(row for n, row in sorted(new_rows.items(), key=lambda kv: int(kv[0])) if n not in placed)

    # Сортировка всех строк данных по номеру; шапка — всё до первой строки с №.
    first_data = next(
        (i for i, ln in enumerate(out) if re.match(r"^\|\s*\d+\s*\|", ln)),
        len(out),
    )
    header = out[:first_data]
    data = [ln for ln in out[first_data:] if re.match(r"^\|\s*\d+\s*\|", ln)]
    data.sort(key=lambda ln: int(re.match(r"^\|\s*(\d+)\s*\|", ln).group(1)))
    LICENSES_FILE.write_text("\n".join(header + data) + "\n", encoding="utf-8")
    print(f"licenses.md: обновлено строк — {len(new_rows)}; всего строк данных — {len(data)}")


def update_categories(nums: dict[int, dict]) -> None:
    data = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
    data.setdefault("categories", {}).update(NEW_CATEGORIES)
    data["classification_basis"] = (
        "Предварительная тематическая классификация корпуса Article-Summarizer-AI "
        "по названиям и метаданным статей: сильные ключевые слова → журнал → "
        "ключевые слова внутри поля. Требует уточнения по полным текстам."
    )
    articles: dict[str, dict] = data.setdefault("articles", {})

    # Статьи 131+ пересчитываются заново при каждом запуске (правила могут
    # уточняться); исторические записи 1–100 не трогаем.
    updated = 0
    for num in sorted(nums):
        rec = nums[num]
        if rec.get("status") != "ok":
            continue
        key = str(num)
        field, subfield = classify(enrich(num, rec))
        articles[key] = {"field": field, "subfield": subfield}
        updated += 1

    if updated:
        CATEGORIES_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"article_categories.json: обновлено статей — {updated}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--licenses-only", action="store_true")
    parser.add_argument("--categories-only", action="store_true")
    args = parser.parse_args()

    if not STATE_FILE.exists():
        print("Нет logs/download_state.json — сначала запустите download_pdfs.py", file=sys.stderr)
        return 1
    nums = load_state_nums()

    if not args.categories_only:
        update_licenses(nums)
    if not args.licenses_only:
        update_categories(nums)
    return 0


if __name__ == "__main__":
    sys.exit(main())
