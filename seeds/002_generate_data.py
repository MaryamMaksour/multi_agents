"""Generate deterministic seed data for the development library schema.

Writes SQL to stdout so it needs no database driver and no network:

    python seeds/002_generate_data.py > seeds/002_data.sql
    psql -d library_dev -f seeds/002_data.sql

The seed is fixed, so re-running produces byte-identical output. Tests can
therefore assert on exact counts and exact answers.

Embedding columns are left NULL. Filling them needs an embedding model, which
is a separate concern from having a working database - back them fill with a
script of your own once the embedding adapter is wired.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

SEED = 20260829
random.seed(SEED)

ROWS = {
    "authors": 28,
    "publishers": 12,
    "branches": 7,
    "books": 420,
    "members": 340,
    "loans": 4200,
}

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

AUTHOR_NAMES = [
    ("Ghassan Kanafani", "غسان كنفاني", "Palestinian", 1936),
    ("Hanna Mina", "حنا مينة", "Syrian", 1924),
    ("Nazik al-Malaika", "نازك الملائكة", "Iraqi", 1923),
    ("Tayeb Salih", "الطيب صالح", "Sudanese", 1929),
    ("Fadwa Tuqan", "فدوى طوقان", "Palestinian", 1917),
    ("Abdul Rahman Munif", "عبد الرحمن منيف", "Saudi", 1933),
    ("Ulfat Idilbi", "ألفة الإدلبي", "Syrian", 1912),
    ("Emily Nasrallah", "إميلي نصرالله", "Lebanese", 1931),
    ("Mahmoud Darwish", "محمود درويش", "Palestinian", 1941),
    ("Colette Khoury", "كوليت خوري", "Syrian", 1931),
    ("Ibrahim al-Koni", "إبراهيم الكوني", "Libyan", 1948),
    ("Radwa Ashour", "رضوى عاشور", "Egyptian", 1946),
    ("Jurji Zaydan", "جرجي زيدان", "Lebanese", 1861),
    ("Sahar Khalifeh", "سحر خليفة", "Palestinian", 1941),
    ("Virginia Woolf", "فرجينيا وولف", "British", 1882),
    ("Gabriel Garcia Marquez", "غابرييل غارسيا ماركيز", "Colombian", 1927),
    ("Italo Calvino", "إيتالو كالفينو", "Italian", 1923),
    ("Chinua Achebe", "تشينوا أتشيبي", "Nigerian", 1930),
    ("Ursula K. Le Guin", "أورسولا لوغين", "American", 1929),
    ("Jorge Luis Borges", "خورخي لويس بورخيس", "Argentine", 1899),
    ("Yasunari Kawabata", "ياسوناري كاواباتا", "Japanese", 1899),
    ("Olga Tokarczuk", "أولغا توكارتشوك", "Polish", 1962),
    ("Anton Chekhov", "أنطون تشيخوف", "Russian", 1860),
    ("Toni Morrison", "توني موريسون", "American", 1931),
    ("Halldor Laxness", "هالدور لاكسنس", "Icelandic", 1902),
    ("Naguib Mahfouz", "نجيب محفوظ", "Egyptian", 1911),
    ("Assia Djebar", "آسيا جبار", "Algerian", 1936),
    ("Amin Maalouf", "أمين معلوف", "Lebanese", 1949),
]

BIO_OPENERS = [
    "Wrote across four decades, mostly about displacement and return.",
    "Began as a journalist before turning to fiction in mid-career.",
    "Known for short, spare sentences and unresolved endings.",
    "Trained as a teacher; the classroom recurs throughout the work.",
    "Published a single novel, then only poetry for thirty years.",
    "Translated widely, though the early work remains hard to find.",
    "Much of the later writing was serialised in newspapers first.",
    "Wrote about coastal towns, harbours, and the people who leave them.",
    "The essays are considered stronger than the fiction by most critics.",
    "Left an unfinished manuscript that was published posthumously.",
]

PUBLISHERS = [
    ("Dar al-Adab", "دار الآداب", "Beirut", "Lebanon", 1956),
    ("Dar al-Mada", "دار المدى", "Damascus", "Syria", 1994),
    ("Dar al-Shorouk", "دار الشروق", "Cairo", "Egypt", 1968),
    ("Al-Ahlia Publishing", "الأهلية للنشر", "Amman", "Jordan", 1985),
    ("Dar al-Saqi", "دار الساقي", "Beirut", "Lebanon", 1979),
    ("Nofal Group", "مجموعة نوفل", "Beirut", "Lebanon", 1972),
    ("Northwind Press", "دار الريح الشمالية", "Edinburgh", "United Kingdom", 1948),
    ("Meridian Books", "دار الزوال", "Toronto", "Canada", 1991),
    ("Blue Harbour Editions", "منشورات المرفأ الأزرق", "Lisbon", "Portugal", 2003),
    ("Kestrel House", "دار العوسق", "Wellington", "New Zealand", 1966),
    ("Verso Antiqua", "فيرسو أنتيكوا", "Bologna", "Italy", 1937),
    ("Lantern Publishing", "دار القنديل", "Tunis", "Tunisia", 2010),
]

BRANCHES = [
    ("Central Library", "المكتبة المركزية", "Damascus", "Al-Salhiyah Street, near the old post office"),
    ("Riverside Branch", "فرع ضفة النهر", "Damascus", "Barada riverside, opposite the municipal park"),
    ("Old Quarter Branch", "فرع المدينة القديمة", "Aleppo", "Bab Antakya, second lane past the covered market"),
    ("University Branch", "فرع الجامعة", "Aleppo", "Faculty of Letters, ground floor, east wing"),
    ("Harbour Branch", "فرع المرفأ", "Latakia", "Port road, above the customs office"),
    ("Orchard Branch", "فرع البساتين", "Homs", "Orchard district, next to the agricultural school"),
    ("Highland Branch", "فرع المرتفعات", "Sweida", "Upper town square, beside the water tower"),
]

TITLE_HEADS_EN = [
    "The Salt", "A Season of", "Letters from", "The Quiet", "Notes on",
    "The Last", "Winter in", "The Weight of", "Small", "The House of",
    "Return to", "The Colour of", "Nine", "The Long", "Against",
    "The Book of", "Morning in", "The Narrow", "Wild", "The Second",
]
TITLE_TAILS_EN = [
    "Harbour", "Almonds", "the Interior", "Migration", "Stone Houses",
    "Rain", "Distance", "the Orchard", "Departures", "Glass",
    "the River", "Afternoons", "Windows", "Silence", "the Coast",
    "Threads", "Olive Wood", "Forgetting", "Bread", "the Border",
]
TITLE_HEADS_AR = [
    "ملح", "موسم", "رسائل من", "الهدوء", "ملاحظات عن",
    "آخر", "شتاء في", "ثقل", "صغيرة", "بيت",
    "العودة إلى", "لون", "تسع", "طويل", "ضدّ",
    "كتاب", "صباح في", "ضيّق", "برّية", "ثاني",
]
TITLE_TAILS_AR = [
    "المرفأ", "اللوز", "الداخل", "الهجرة", "البيوت الحجرية",
    "المطر", "المسافة", "البستان", "الرحيل", "الزجاج",
    "النهر", "الأصائل", "النوافذ", "الصمت", "الساحل",
    "الخيوط", "خشب الزيتون", "النسيان", "الخبز", "الحدود",
]

GENRES = [
    "novel", "poetry", "short stories", "history", "biography",
    "essays", "travel", "criticism", "drama", "children",
]
LANGUAGES = ["Arabic", "English", "French", "Arabic", "English", "Arabic"]

SUMMARY_PARTS_A = [
    "A family sells the last of its land",
    "Two brothers travel north for work",
    "A schoolteacher returns after twenty years",
    "A widow keeps her husband's shop open",
    "A translator loses a manuscript",
    "A fisherman refuses to sell his boat",
    "A student misses the last train home",
    "A cartographer maps a town that is emptying",
    "A baker closes for the first time in forty years",
    "A photographer catalogues abandoned houses",
]
SUMMARY_PARTS_B = [
    "and neither of them speaks of it again.",
    "over a single, unbearably hot summer.",
    "in a town that no longer recognises them.",
    "while the neighbours quietly take sides.",
    "and finds the ending was never written.",
    "though everyone expects him to give in.",
    "and walks the whole way back through the orchards.",
    "as the last families pack and leave.",
    "and the queue outside does not disperse.",
    "before the winter rains take the roofs.",
]

FIRST_EN = [
    "Layla", "Omar", "Rana", "Samir", "Nour", "Karim", "Hala", "Fadi",
    "Maya", "Ziad", "Dima", "Tarek", "Salma", "Bassel", "Rima", "Jad",
    "Lina", "Hadi", "Nada", "Yusuf", "Sana", "Rami", "Aya", "Marwan",
]
FIRST_AR = [
    "ليلى", "عمر", "رنا", "سمير", "نور", "كريم", "هالة", "فادي",
    "مايا", "زياد", "ديما", "طارق", "سلمى", "باسل", "ريما", "جاد",
    "لينا", "هادي", "ندى", "يوسف", "سناء", "رامي", "آية", "مروان",
]
LAST_EN = [
    "Haddad", "Khoury", "Nasser", "Sayegh", "Darwish", "Aswad", "Rifai",
    "Barakat", "Hourani", "Saab", "Mansour", "Zeidan", "Attar", "Kanaan",
]
LAST_AR = [
    "حداد", "خوري", "ناصر", "صايغ", "درويش", "أسود", "الرفاعي",
    "بركات", "حوراني", "صعب", "منصور", "زيدان", "العطار", "كنعان",
]

TIERS = ["basic", "basic", "basic", "premium", "student", "student"]
MEMBER_STATUS = ["active", "active", "active", "active", "suspended", "expired"]
CITIES = ["Damascus", "Damascus", "Aleppo", "Aleppo", "Latakia", "Homs", "Sweida"]

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def q(value) -> str:
    """Render a Python value as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return "'" + value.isoformat() + "'"
    return "'" + str(value).replace("'", "''") + "'"


# Rows per INSERT. One statement per table would work, but a 4200-tuple
# statement gives a GUI client nothing to report until it finishes, and
# nothing useful to point at when something fails. Batches keep the file
# reviewable and give per-statement progress in DBeaver or psql alike.
BATCH = 500


def insert(table: str, columns: list[str], rows: list[tuple]) -> None:
    if not rows:
        return
    print(f"\n-- {table}: {len(rows)} rows")
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        print(f"INSERT INTO {table} ({', '.join(columns)}) VALUES")
        rendered = [f"  ({', '.join(q(v) for v in row)})" for row in batch]
        print(",\n".join(rendered) + ";")


EPOCH = datetime(2023, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)
SPAN_DAYS = (NOW - EPOCH).days


def somewhen(start_days: int = 0, end_days: int = SPAN_DAYS) -> datetime:
    return EPOCH + timedelta(
        days=random.randint(start_days, end_days),
        hours=random.randint(8, 19),
        minutes=random.choice([0, 15, 30, 45]),
    )


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

print("-- Generated by seeds/002_generate_data.py - do not edit by hand.")
print(f"-- seed={SEED}  rows={ROWS}")
print("BEGIN;")
print("TRUNCATE loans, members, branches, books, publishers, authors RESTART IDENTITY CASCADE;")

# authors ------------------------------------------------------------------
authors = []
for i in range(ROWS["authors"]):
    name_en, name_ar, nationality, birth_year = AUTHOR_NAMES[i % len(AUTHOR_NAMES)]
    bio = random.choice(BIO_OPENERS)
    authors.append((i + 1, name_en, name_ar, nationality, birth_year, bio, somewhen(0, 200)))

insert(
    "authors",
    ["id", "name_en", "name_ar", "nationality", "birth_year", "bio", "created_at"],
    authors,
)

# publishers ---------------------------------------------------------------
publishers = []
for i in range(ROWS["publishers"]):
    name_en, name_ar, city, country, founded = PUBLISHERS[i % len(PUBLISHERS)]
    publishers.append((i + 1, name_en, name_ar, city, country, founded))

insert(
    "publishers",
    ["id", "name_en", "name_ar", "city", "country", "founded_year"],
    publishers,
)

# branches -----------------------------------------------------------------
branches = []
for i in range(ROWS["branches"]):
    name_en, name_ar, city, address = BRANCHES[i % len(BRANCHES)]
    opened = (EPOCH - timedelta(days=random.randint(1200, 9000))).date().isoformat()
    branches.append((i + 1, name_en, name_ar, city, address, opened))

insert(
    "branches",
    ["id", "name_en", "name_ar", "city", "address", "opened_on"],
    branches,
)

# books --------------------------------------------------------------------
books = []
used_titles: set[tuple[int, int]] = set()
for i in range(ROWS["books"]):
    while True:
        h, t = random.randrange(len(TITLE_HEADS_EN)), random.randrange(len(TITLE_TAILS_EN))
        if (h, t) not in used_titles or len(used_titles) >= len(TITLE_HEADS_EN) * len(TITLE_TAILS_EN):
            used_titles.add((h, t))
            break
    title_en = f"{TITLE_HEADS_EN[h]} {TITLE_TAILS_EN[t]}"
    title_ar = f"{TITLE_HEADS_AR[h]} {TITLE_TAILS_AR[t]}"
    summary = f"{random.choice(SUMMARY_PARTS_A)} {random.choice(SUMMARY_PARTS_B)}"
    genre = random.choice(GENRES)
    books.append((
        i + 1,
        title_en,
        title_ar,
        random.randint(1, ROWS["authors"]),
        random.randint(1, ROWS["publishers"]),
        f"978-{random.randint(100, 999)}-{random.randint(10000, 99999)}-{random.randint(0, 9)}",
        random.randint(1962, 2025),
        random.choice([96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]),
        random.choice(LANGUAGES),
        genre,
        summary,
        f"{genre[:2].upper()}-{random.randint(1, 40):02d}.{random.randint(1, 9)}",
        random.choice([1, 1, 1, 2, 2, 3, 5]),
        round(random.uniform(4.5, 68.0), 2),
        somewhen(0, SPAN_DAYS - 30),
    ))

insert(
    "books",
    ["id", "title_en", "title_ar", "author_id", "publisher_id", "isbn",
     "publication_year", "page_count", "language", "genre", "summary",
     "shelf_code", "copies_total", "price", "added_at"],
    books,
)

# members ------------------------------------------------------------------
members = []
for i in range(ROWS["members"]):
    fi = random.randrange(len(FIRST_EN))
    li = random.randrange(len(LAST_EN))
    full_en = f"{FIRST_EN[fi]} {LAST_EN[li]}"
    full_ar = f"{FIRST_AR[fi]} {LAST_AR[li]}"
    members.append((
        i + 1,
        full_en,
        full_ar,
        f"{FIRST_EN[fi].lower()}.{LAST_EN[li].lower()}{i + 1}@example.org",
        f"+9639{random.randint(10000000, 99999999)}",
        random.choice(TIERS),
        random.choice(MEMBER_STATUS),
        random.choice(CITIES),
        somewhen(0, SPAN_DAYS - 10),
    ))

insert(
    "members",
    ["id", "full_name_en", "full_name_ar", "email", "phone",
     "membership_tier", "status", "city", "joined_at"],
    members,
)

# loans --------------------------------------------------------------------
loans = []
for i in range(ROWS["loans"]):
    borrowed = somewhen(30, SPAN_DAYS)
    due = borrowed + timedelta(days=random.choice([14, 14, 21, 28]))

    roll = random.random()
    if roll < 0.72:                                   # returned on time
        returned = borrowed + timedelta(days=random.randint(2, (due - borrowed).days))
        status, fine = "returned", 0.0
    elif roll < 0.85:                                 # returned late, fined
        returned = due + timedelta(days=random.randint(1, 40))
        status = "returned"
        fine = round((returned - due).days * 0.25, 2)
    elif due < NOW:                                   # still out, past due
        returned, status = None, "overdue"
        fine = round((NOW - due).days * 0.25, 2)
    else:                                             # still out, not yet due
        returned, status, fine = None, "open", 0.0

    loans.append((
        i + 1,
        random.randint(1, ROWS["books"]),
        random.randint(1, ROWS["members"]),
        random.randint(1, ROWS["branches"]),
        borrowed,
        due,
        returned,
        status,
        fine,
    ))

insert(
    "loans",
    ["id", "book_id", "member_id", "branch_id", "borrowed_at", "due_at",
     "returned_at", "status", "fine_amount"],
    loans,
)

print("\nCOMMIT;")
print("\n-- sequences are not used (explicit ids above); nothing to reset.")
print("ANALYZE authors, publishers, books, branches, members, loans;")
