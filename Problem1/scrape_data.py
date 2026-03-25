# scrape_data.py
# scrapes text from iitj.ac.in for the word2vec corpus
# does BFS crawl on HTML pages + downloads PDFs


import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import fitz  # PyMuPDF for reading PDFs

# where to dump the raw text files
OUTPUT_DIR = os.path.join("corpus", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# pretend to be a normal browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 25        # seconds for html requests
PDF_TIMEOUT = 60    # pdfs can be slow
MAX_PDF_SIZE = 30 * 1024 * 1024  # 30 MB limit so we dont hang on huge files
DELAY = 0.5         # be polite, wait between requests


# -----------------------------------------------------------
# seed urls - hand picked pages with lots of english content
# covers departments, offices, schools, etc.
# -----------------------------------------------------------
SEED_URLS = [
    # main site / institutional pages
    "https://iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/main/en/history",
    "https://iitj.ac.in/main/en/vision-and-mission",
    "https://iitj.ac.in/main/en/director",
    "https://iitj.ac.in/main/en/campus-infrastructure",
    "https://iitj.ac.in/main/en/constitution-of-bog",
    "https://iitj.ac.in/main/en/constitution-of-senate",
    "https://iitj.ac.in/main/en/acts-statutes",
    "https://iitj.ac.in/main/en/introduction-to-statutory-bodies",
    "https://iitj.ac.in/main/en/sustainability-policy",
    "https://iitj.ac.in/main/en/Student-Life-at-IIT-Jodhpur",
    "https://iitj.ac.in/main/en/research-highlight",
    "https://iitj.ac.in/main/en/faqs-applicants",
    "https://iitj.ac.in/main/en/pmrf",
    "https://iitj.ac.in/main/en/news",
    "https://iitj.ac.in/main/en/all-announcement",
    "https://iitj.ac.in/main/en/faculty-members",
    "https://iitj.ac.in/main/en/adjunct-faculty-members",
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    "https://iitj.ac.in/main/en/scholars-in-residence",
    "https://iitj.ac.in/main/en/Annual-Reports-of-the-Institute",
    "https://iitj.ac.in/main/en/nirf-reports",
    "https://iitj.ac.in/AtoZ?lg=en",

    # office of academics
    "https://iitj.ac.in/office-of-academics/en/academics",
    "https://iitj.ac.in/office-of-academics/en/academic-programs",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/office-of-academics/en/curriculum",
    "https://iitj.ac.in/office-of-academics/en/fee-structure",
    "https://iitj.ac.in/office-of-academics/en/scholarships",
    "https://iitj.ac.in/office-of-academics/en/convocation",
    "https://iitj.ac.in/office-of-academics/en/specialization-minor-dual-degree",
    "https://iitj.ac.in/office-of-academics/en/program-structure",
    "https://iitj.ac.in/office-of-academics/en/circulars",
    "https://iitj.ac.in/office-of-academics/en/b.tech.",
    "https://iitj.ac.in/office-of-academics/en/m.tech.",
    "https://iitj.ac.in/office-of-academics/en/m.sc.",
    "https://iitj.ac.in/office-of-academics/en/ph.d.",
    "https://iitj.ac.in/office-of-academics/en/mba",
    "https://iitj.ac.in/office-of-academics/en/mdes",
    "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",

    # other offices
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://iitj.ac.in/office-of-research-development/en/projects",
    "https://iitj.ac.in/office-of-research-development/en/ipr",
    "https://iitj.ac.in/office-of-research-development/en/Technology",
    "https://iitj.ac.in/office-of-international-relations/en/office-of-international-relations",
    "https://iitj.ac.in/office-of-international-relations/en/collaborations",
    "https://iitj.ac.in/office-of-training-and-placement/en/office-of-training-and-placement",
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    "https://iitj.ac.in/office-of-students/en/campus-life",
    "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    "https://iitj.ac.in/office-of-executive-education/en/program-portfolio",
    "https://iitj.ac.in/office-of-director/en/office-of-director",
    "https://iitj.ac.in/office-of-registrar/en/office-of-registrar",
    "https://iitj.ac.in/office-of-administration/en/office-of-administration",
    "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    "https://iitj.ac.in/faculty-positions/en/faculty-positions",

    # CSE department
    "https://iitj.ac.in/computer-science-engineering/en/computer-science-and-engineering",
    "https://iitj.ac.in/computer-science-engineering/en/artificial-intelligence-and-machine-learning",
    "https://iitj.ac.in/computer-science-engineering/en/systems-software",
    "https://iitj.ac.in/computer-science-engineering/en/theoretical-computer-science",
    "https://iitj.ac.in/computer-science-engineering/en/vision-and-ar-vr",
    "https://iitj.ac.in/computer-science-engineering/en/research-area-labs",
    "https://iitj.ac.in/computer-science-engineering/en/undergraduate-programs",
    "https://iitj.ac.in/computer-science-engineering/en/postgraduate-programs",

    # EE
    "https://iitj.ac.in/electrical-engineering/en/electrical-engineering",
    "https://iitj.ac.in/electrical-engineering/en/research-overview",
    "https://iitj.ac.in/electrical-engineering/en/laboratories",
    "https://iitj.ac.in/electrical-engineering/en/curriculum",

    # ME
    "https://iitj.ac.in/mechanical-engineering/en/mechanical-engineering",
    "https://iitj.ac.in/mechanical-engineering/en/about-research",
    "https://iitj.ac.in/mechanical-engineering/en/laboratories",

    # civil
    "https://iitj.ac.in/civil-and-infrastructure-engineering/en/civil-and-infrastructure-engineering",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/en/laboratories",

    # chemical
    "https://iitj.ac.in/chemical-engineering/en/chemical-engineering",
    "https://iitj.ac.in/chemical-engineering/en/about-research",

    # chemistry
    "https://iitj.ac.in/chemistry/en/chemistry",
    "https://iitj.ac.in/chemistry/en/about-research",
    "https://iitj.ac.in/chemistry/en/facilities",

    # bio
    "https://iitj.ac.in/bioscience-bioengineering/en/bioscience-bioengineering",
    "https://iitj.ac.in/bioscience-bioengineering/en/about-research",

    # materials
    "https://iitj.ac.in/materials-engineering/en/materials-engineering",
    "https://iitj.ac.in/materials-engineering/en/curriculum",

    # math
    "https://iitj.ac.in/mathematics/en/mathematics",
    "https://iitj.ac.in/mathematics/en/courses",

    # physics
    "https://iitj.ac.in/physics/en/physics",
    "https://iitj.ac.in/physics/en/research-groups",
    "https://iitj.ac.in/physics/en/laboratories",

    # school of AI
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/about-research",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/themes",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/publications",

    # school of design
    "https://iitj.ac.in/school-of-design/en/school-of-design",

    # liberal arts
    "https://iitj.ac.in/school-of-liberal-arts/en/school-of-liberal-arts",
    "https://iitj.ac.in/school-of-liberal-arts/en/Vision-and-Mission",

    # management
    "https://iitj.ac.in/schools/en/School-of-Management-&-Entrepreneurship",
    "https://iitj.ac.in/schools/en/about-sme",
    "https://iitj.ac.in/schools/en/program-curriculum",

    # labs and centers
    "https://iitj.ac.in/crf/en/crf",
    "https://iitj.ac.in/crf/en/instruments",
    "https://iitj.ac.in/dia/en/about-dia",
    "https://iitj.ac.in/dia/en/computing-facilities",
    "https://iitj.ac.in/medical-technologies/en/medical-technologies",
    "https://iitj.ac.in/health-center/en/health-center",
    "https://iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility",
    "https://iitj.ac.in/es/en/engineering-science",
    "https://iitj.ac.in/bachelor-of-technology/en/academic-research-facilities",
]


# -----------------------------------------------------------
# direct PDF downloads - 25 high value PDFs we definitely want
# annual reports, regulations, NIRF, acts, brochures etc
# -----------------------------------------------------------
DIRECT_PDF_URLS = [
    # annual reports
    ("annual_report_2024_25",
     "https://iitj.ac.in/PageImages/Gallery/12-2025/AnnualReport_2024-25_English_Low.pdf"),
    ("annual_report_2023_24",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/Annual%20Report_English_23-24.pdf"),
    ("annual_report_2022_23",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/IITJ_AR_2022_2023_English.pdf"),
    ("annual_report_2021_22",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/IITJ_AR_2021_2022_English_FINAL.pdf"),
    ("annual_report_2020_21",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/IITJ%20Annual%20Report_English_FY2020_21.pdf"),
    ("annual_report_2019_20",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/IITJ_AR_2019-20_English.pdf"),
    ("annual_report_2018_19",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/AR_2018_19%20(English).pdf"),
    ("annual_report_2010_11",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/AR2010-11.pdf"),

    # academic regulations (super important for vocab)
    ("academic_regulations_ug_2019",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf"),
    ("academic_regulations_pg_2022",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/4_Regulation_PG_2022-onwards_20022023.pdf"),
    ("academic_regulations_ms_research_2023",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/6_2024-04-17-661f605b54457-1713332315.pdf"),

    # IIT act and statutes
    ("iit_act_1961",
     "https://iitj.ac.in/PageImages/Gallery/05-2025/Institutes-of-Technology-Act-1961-638829208155359743.pdf"),
    ("iit_amendment_act_2012",
     "https://iitj.ac.in/PageImages/Gallery/05-2025/Institutes-of-Technology-Amendment-Act-2012-638829208552640729.pdf"),
    ("iitj_first_statutes",
     "https://iitj.ac.in/PageImages/Gallery/05-2025/Statutes-638829208914867678.pdf"),

    # NIRF
    ("nirf_2025_overall",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/NIRF%202025%20Overall%20Final%20Sumitted%20Data%2031Jan2025.pdf"),
    ("nirf_2025_engineering",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/NIRF%202025%20Engineering%20Final%20Sumitted%20Data%2031Jan2025.pdf"),

    # research and IPR stuff
    ("ipr_policy",
     "https://iitj.ac.in/PageImages/Gallery/02-2025/IPR-Policy-638741893294608615.pdf"),
    ("rd_norms_2018",
     "https://iitj.ac.in/PageImages/Gallery/02-2025/Norms-for-Research-Development-638741906663985707.pdf"),
    ("sponsored_research_detail",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/SponsoredResearchDetail.pdf"),
    ("consultancy_project_detail",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/ConsultancyProjectDetail.pdf"),
    ("entrepreneurship",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/Entrepreneurship.pdf"),

    # brochures and misc
    ("placement_brochure",
     "https://iitj.ac.in/PageImages/Gallery/01-2025/Placement-Brochure-638733086428767139.pdf"),
    ("civil_dept_brochure",
     "https://iitj.ac.in/PageImages/Gallery/03-2025/Civil-Department-Brochure-638768569230798671.pdf"),
    ("institute_chronological_development",
     "https://iitj.ac.in/PageImages/Gallery/07-2025/116Genesis-or-Chronological-Development-of-the-Institute2024-638869686988790651.pdf"),

    # curriculum docs
    ("mtech_intelligent_comm_systems",
     "https://iitj.ac.in/acad_website/pg_curriculum/29.2.2_AP3_2_Detailed%20concept%20note%20and%20curriculum%20on%20M.Tech.%20and%20M.Tech.-Ph.D.%20in%20Intelligent%20Communication%20Systems_Final18.7.2022.pdf"),
    ("mtech_cs_ai_updated_curriculum",
     "https://iitj.ac.in/acad_website/pg_curriculum/Updated_course_content_for_M.Tech.AI_and_CS.pdf"),
]


# -----------------------------------------------------------
# pages that list more PDFs we can discover and download
# -----------------------------------------------------------
PDF_DISCOVERY_PAGES = [
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/office-of-academics/en/circulars",
    "https://iitj.ac.in/office-of-academics/en/program-structure",
    "https://iitj.ac.in/office-of-academics/en/download-forms",
    "https://iitj.ac.in/office-of-academics/en/scholarships",
    "https://iitj.ac.in/main/en/Annual-Reports-of-the-Institute",
    "https://iitj.ac.in/main/en/nirf-reports",
    "https://iitj.ac.in/main/en/acts-statutes",
    "https://iitj.ac.in/office-of-research-development/en/notification",
    "https://iitj.ac.in/office-of-research-development/en/information",
    "https://iitj.ac.in/office-of-research-development/en/forms",
    "https://iitj.ac.in/institute-repository/en/nirf",
    "https://iitj.ac.in/Institute-Repository/en/Brochure",
    "https://iitj.ac.in/Institute-Repository/en/Newsletter",
    "https://iitj.ac.in/office-of-academics/en/curriculum",
    "https://iitj.ac.in/office-of-academics/en/fee-structure",
]


# -----------------------------------------------------------
# filtering config
# -----------------------------------------------------------
ALLOWED_DOMAINS = {"iitj.ac.in"}

# file types we dont want
SKIP_EXTENSIONS = re.compile(
    r'\.(jpg|jpeg|png|gif|svg|ico|mp4|zip|gz|tar|pptx?|xlsx?|docx?|csv)$',
    re.I
)

# skip hindi pages
SKIP_PATTERNS = re.compile(r'/hi/|/hindi/', re.I)


# -----------------------------------------------------------
# helper functions
# -----------------------------------------------------------

def fetch(url):
    # simple GET request with error handling
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"    [WARN] {url[:80]}: {e}")
        return None


def fetch_pdf_streaming(url):
    # downloads a PDF in chunks so we can bail out if its too big
    # returns the raw bytes or None if something goes wrong
    try:
        r = requests.get(url, headers=HEADERS, timeout=PDF_TIMEOUT,
                         allow_redirects=True, stream=True)
        r.raise_for_status()

        # check content-length header if available
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > MAX_PDF_SIZE:
            print(f"    [SKIP] Too large ({int(cl)//1024//1024} MB): {url[:70]}")
            r.close()
            return None

        # download in 256KB chunks, stop if we go over the limit
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=256 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PDF_SIZE:
                print(f"    [SKIP] Exceeded {MAX_PDF_SIZE//1024//1024} MB during download: {url[:70]}")
                r.close()
                return None
        return b"".join(chunks)
    except Exception as e:
        print(f"    [WARN PDF] {url[:70]}: {e}")
        return None


def html_to_text(html, url=""):
    # strip out scripts, nav, footer etc and just get the text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "form", "button", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # only keep lines with at least 15 chars (skip menu items etc)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    return f"\n### {url}\n" + "\n".join(lines)


def pdf_to_text(data):
    # extract text from PDF bytes using PyMuPDF
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception as e:
        print(f"    [PDF ERROR] {e}")
        return ""


def is_english(text, threshold=0.65):
    # quick check: if most characters are ASCII, its probably english
    if not text:
        return False
    return sum(ord(c) < 128 for c in text) / len(text) >= threshold


def save(name, text):
    # write text to a file in the output directory
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    kb = len(text.encode()) / 1024
    words = len(text.split())
    print(f"  [SAVED] {name}.txt  {kb:.0f} KB  ~{words} words")


# -----------------------------------------------------------
# BFS crawler - starts from seed urls and follows links
# only follows english pages on iitj.ac.in domain
# -----------------------------------------------------------
def bfs_crawl(seeds, max_pages=250):
    visited = set()
    queue = list(seeds)
    results = []

    print(f"\n[BFS] Starting crawl (max {max_pages} pages) ...")

    while queue and len(results) < max_pages:
        url = queue.pop(0)
        url = url.split("#")[0].rstrip("/")

        if url in visited:
            continue
        # dont crawl hindi pages
        if SKIP_PATTERNS.search(url):
            continue
        visited.add(url)

        r = fetch(url)
        if r is None:
            continue

        ct = r.headers.get("Content-Type", "")
        if "html" not in ct:
            continue

        text = html_to_text(r.text, url)
        if not is_english(text):
            continue

        # skip pages with barely any text
        if len(text.split()) < 30:
            pass
        else:
            results.append((url, text))
            wc = len(text.split())
            print(f"  [{len(results):03d}] {url[:70]}  (~{wc}w)")

        # find new links to follow
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("?")[0].split("#")[0].rstrip("/")
            parsed = urlparse(href)
            if (
                parsed.netloc in ALLOWED_DOMAINS
                and href not in visited
                and href not in queue
                and not SKIP_EXTENSIONS.search(parsed.path)
                and not SKIP_PATTERNS.search(href)
                and parsed.scheme in ("http", "https")
                and not href.lower().endswith(".pdf")
            ):
                queue.append(href)

        time.sleep(DELAY)

    print(f"[BFS] Done: {len(results)} pages harvested")
    return results


# -----------------------------------------------------------
# PDF discovery and downloading
# -----------------------------------------------------------

def discover_pdf_links_from_page(page_url):
    # look at a page and find all PDF links on it
    r = fetch(page_url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    found = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if href.lower().endswith(".pdf") and href not in seen:
            seen.add(href)
            label = (a.get_text(strip=True) or href.split("/")[-1])[:60]
            found.append((label, href))
    return found


def fetch_pdfs(pdf_list, already_fetched):
    # download a list of PDFs, skip duplicates and non-english ones
    results = []
    for name, url in pdf_list:
        if url in already_fetched:
            print(f"    [SKIP DUP] {url[:70]}")
            continue
        already_fetched.add(url)

        print(f"  Trying PDF: {url[:80]}")
        data = fetch_pdf_streaming(url)
        if data is None:
            continue

        text = pdf_to_text(data)

        # must be mostly english and have decent amount of text
        if not is_english(text, 0.60):
            print(f"    [SKIP non-EN PDF]")
            continue
        words = len(text.split())
        if words > 100:
            safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:60]
            mb = len(data) / 1024 / 1024
            print(f"    [PDF OK] {safe_name}: ~{words} words ({mb:.1f} MB)")
            results.append((safe_name, text))
        else:
            print(f"    [PDF SKIP] too short ({words}w)")
        time.sleep(DELAY)
    return results


# -----------------------------------------------------------
# main - runs the whole scraping pipeline
# 4 phases: BFS crawl -> direct PDFs -> discover PDFs -> extra PDFs if needed
# -----------------------------------------------------------
def main():
    print("=" * 65)
    print("IIT Jodhpur Scraper")
    print("=" * 65)

    all_html_text = []
    all_pdf_text = []
    fetched_pdf_urls = set()

    # phase 1: crawl web pages starting from seed urls
    print("\n--- PHASE 1: BFS CRAWL ---")
    pages = bfs_crawl(SEED_URLS, max_pages=250)
    all_html_text = [text for _, text in pages]

    html_words = sum(len(t.split()) for t in all_html_text)
    print(f"\n  Phase 1 total: ~{html_words:,} words from HTML")

    # phase 2: download the 25 hand-picked PDFs
    print("\n--- PHASE 2: DIRECT PDF DOWNLOADS ---")
    all_pdf_text.extend(fetch_pdfs(DIRECT_PDF_URLS, fetched_pdf_urls))

    pdf_words = sum(len(t.split()) for _, t in all_pdf_text)
    print(f"\n  Phase 2 total: ~{pdf_words:,} words from PDFs")

    # phase 3: look at aggregator pages for more PDFs
    print("\n--- PHASE 3: DISCOVER MORE PDFs ---")
    discovered_pdfs = []
    for pg in PDF_DISCOVERY_PAGES:
        links = discover_pdf_links_from_page(pg)
        page_name = pg.split("/")[-1]
        print(f"  {page_name}: {len(links)} PDFs found")
        # grab up to 8 per page, skip recruitment/hiring stuff
        for label, url in links[:8]:
            low = (label + url).lower()
            if any(skip in low for skip in [
                "shortlist", "selected-candidate", "recruitment",
                "advertisement-document", "notice-withdrawn",
                "application-received", "process-of-examination",
                "-hindi", "हिन्दी", "हिंदी"
            ]):
                continue
            discovered_pdfs.append((label, url))

    print(f"\n  Found {len(discovered_pdfs)} relevant PDFs from aggregator pages")
    all_pdf_text.extend(fetch_pdfs(discovered_pdfs, fetched_pdf_urls))

    total_words = sum(len(t.split()) for t in all_html_text)
    total_words += sum(len(t.split()) for _, t in all_pdf_text)
    print(f"\n  Running total: ~{total_words:,} words")

    # phase 4: if we're still under 120k words, try to find more
    if total_words < 120000:
        print(f"\n--- PHASE 4: GETTING MORE CONTENT ---")
        extra_pdf_pages = [
            "https://iitj.ac.in/office-of-academics/en/Old-Regulations",
            "https://iitj.ac.in/office-of-academics/en/Curriculum-for-Programs-Before-2019",
            "https://iitj.ac.in/office-of-academics/en/list-of-holidays",
            "https://iitj.ac.in/main/en/Procedure-for-RTI",
            "https://iitj.ac.in/office-of-students/en/office-of-students",
        ]
        extra_discovered = []
        for pg in extra_pdf_pages:
            links = discover_pdf_links_from_page(pg)
            for label, url in links[:5]:
                low = (label + url).lower()
                if any(skip in low for skip in [
                    "shortlist", "selected-candidate", "recruitment",
                    "-hindi", "हिन्दी"
                ]):
                    continue
                extra_discovered.append((label, url))

        all_pdf_text.extend(fetch_pdfs(extra_discovered, fetched_pdf_urls))

        total_words = sum(len(t.split()) for t in all_html_text)
        total_words += sum(len(t.split()) for _, t in all_pdf_text)
        print(f"\n  Phase 4 total: ~{total_words:,} words")

    # save everything
    print("\n--- SAVING ---")

    # clear old files first
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    # all HTML goes into one big file
    combined_html = "\n\n".join(all_html_text)
    save("iitj_html_all", combined_html)

    # each PDF gets its own file
    for name, text in all_pdf_text:
        save(f"pdf_{name}", text)

    # print summary
    total_words = sum(len(t.split()) for t in all_html_text)
    total_words += sum(len(t.split()) for _, t in all_pdf_text)
    raw_files = os.listdir(OUTPUT_DIR)

    print(f"\n{'='*65}")
    print(f"DONE!")
    print(f"  HTML pages:   {len(all_html_text)}")
    print(f"  PDFs fetched: {len(all_pdf_text)}")
    print(f"  Raw files:    {len(raw_files)}")
    print(f"  Total words:  ~{total_words:,}")
    print(f"  Output dir:   {OUTPUT_DIR}/")
    for f in sorted(raw_files):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f}  ({size//1024} KB)")

    if total_words >= 100000:
        print(f"\n  Target reached! {total_words:,} words >= 100,000")
    else:
        print(f"\n  Below target: {total_words:,} / 100,000 words")


if __name__ == "__main__":
    main()
