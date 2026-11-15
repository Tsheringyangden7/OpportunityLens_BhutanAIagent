
import json
import re
from datetime import datetime, date
from typing import TypedDict
from urllib.parse import urlsplit, urlunsplit

from langgraph.graph import StateGraph, END

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


# ============================================================
# 1. OPTIONAL LOCAL AI MODEL
# ============================================================

# Ollama is optional. The assessment must still work if Ollama
# is unavailable, particularly when running on Streamlit Cloud.

llm = None

if ChatOllama is not None:
    try:
        llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
        )
    except Exception:
        llm = None


# ============================================================
# 2. AGENT STATE
# ============================================================

class AgentState(TypedDict, total=False):
    notice: str
    applicant: dict
    use_web: bool
    opportunity: dict
    research: dict
    eligibility: dict
    funding: dict
    reflection: list
    recommendation: dict
    report: str
    events: list


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return ", ".join(clean(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def ask_ai(prompt):
    if llm is None:
        return ""

    response = llm.invoke(prompt)
    return response.content


def extract_json(value):
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", value, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def add_event(state, message):
    events = list(state.get("events", []))
    events.append(message)
    return events


def get_applicant_value(applicant, keys):
    for key in keys:
        value = applicant.get(key)

        if value is not None and clean(value):
            return clean(value)

    return ""


def applicant_text(applicant):
    return "\n".join(
        f"{key.replace('_', ' ').title()}: {clean(value)}"
        for key, value in applicant.items()
        if clean(value)
    )


def normalize_url(url):
    try:
        parsed = urlsplit(url.strip())

        return urlunsplit((
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
        ))

    except Exception:
        return url.strip().lower()


def contains_any(text, phrases):
    text = clean(text).lower()
    return any(phrase.lower() in text for phrase in phrases)


def unique_items(items):
    result = []
    seen = set()

    for item in items:
        item = clean(item)

        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)

    return result


# ============================================================
# 4. DATE EXTRACTION
# ============================================================

DATE_PATTERN = (
    r"\b\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|"
    r"\b\d{4}-\d{2}-\d{2}\b"
)


def parse_date(date_text):
    if not date_text:
        return None

    formats = [
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                date_text.strip(),
                fmt,
            ).date()
        except ValueError:
            continue

    return None


def extract_labelled_date(notice, labels):
    lines = notice.splitlines()

    for line in lines:
        for label in labels:
            pattern = rf"{label}\s*[:\-]\s*({DATE_PATTERN})"

            match = re.search(
                pattern,
                line,
                re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

    for index, line in enumerate(lines):
        if any(
            re.search(label, line, re.IGNORECASE)
            for label in labels
        ):
            nearby = " ".join(lines[index:index + 2])

            match = re.search(
                DATE_PATTERN,
                nearby,
                re.IGNORECASE,
            )

            if match:
                return match.group(0).strip()

    return ""


def extract_deadline(notice):
    labels = [
        r"application\s+deadline",
        r"deadline\s+for\s+applications",
        r"application\s+closes",
        r"last\s+date\s+to\s+apply",
        r"closing\s+date",
        r"apply\s+before",
        r"applications\s+must\s+be\s+submitted\s+before",
    ]

    labelled = extract_labelled_date(notice, labels)

    if labelled:
        return labelled

    matches = list(
        re.finditer(
            DATE_PATTERN,
            notice,
            re.IGNORECASE,
        )
    )

    for match in matches:
        start = max(0, match.start() - 120)
        end = min(len(notice), match.end() + 60)
        context = notice[start:end].lower()

        if any(word in context for word in [
            "deadline",
            "application closes",
            "apply before",
            "last date",
            "closing date",
        ]):
            return match.group(0).strip()

    return ""


def extract_programme_dates(notice):
    labels = [
        r"programme\s+dates",
        r"program\s+dates",
        r"programme\s+period",
        r"program\s+period",
        r"event\s+dates",
        r"summit\s+dates",
        r"programme\s+date",
    ]

    for line in notice.splitlines():
        if not any(
            re.search(label, line, re.IGNORECASE)
            for label in labels
        ):
            continue

        range_match = re.search(
            r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+"
            r"(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+"
            r"(\d{4})",
            line,
            re.IGNORECASE,
        )

        if range_match:
            return (
                f"{range_match.group(1)}–"
                f"{range_match.group(2)} "
                f"{range_match.group(3)} "
                f"{range_match.group(4)}"
            )

        matches = re.findall(
            DATE_PATTERN,
            line,
            re.IGNORECASE,
        )

        if len(matches) >= 2:
            return f"{matches[0]} to {matches[1]}"

        if len(matches) == 1:
            return matches[0]

    return ""


def get_deadline_status(deadline):
    parsed = parse_date(deadline)

    if not parsed:
        return "Unknown"

    today = date.today()

    if parsed < today:
        return "Passed"

    if parsed == today:
        return "Deadline is today"

    return "Upcoming"


# ============================================================
# 5. OPPORTUNITY CLASSIFICATION
# ============================================================

def classify_opportunity(notice, ai_type=""):
    text = notice.lower()

    categories = [
        (
            ["youth summit", "international summit",
             "leadership summit", "innovation summit"],
            "International Youth Summit",
        ),
        (["internship", "intern"], "Internship"),
        (["fellowship", "fellow"], "Fellowship"),
        (
            ["research grant", "research funding", "research award"],
            "Research Grant",
        ),
        (
            ["scholarship", "tuition waiver", "study scholarship"],
            "Scholarship",
        ),
        (["conference", "symposium"], "Conference"),
        (
            ["training programme", "training program",
             "capacity building"],
            "Training Programme",
        ),
    ]

    for keywords, category in categories:
        if contains_any(text, keywords):
            return category

    if ai_type and ai_type.lower() not in [
        "unknown",
        "not specified",
    ]:
        return ai_type

    return "Other Opportunity"


# ============================================================
# 6. FUNDING EXTRACTION
# ============================================================

def extract_funding_details(notice):
    text = notice.lower()

    covered = []
    not_covered = []
    unknown = []

    coverage_rules = [
        (
            r"accommodation.{0,100}(?:provided|covered|included)",
            "Accommodation for the programme",
        ),
        (
            r"(?:provided|covered|included).{0,100}accommodation",
            "Accommodation for the programme",
        ),
        (
            r"meals?.{0,100}(?:provided|covered|included)",
            "Meals during the programme",
        ),
        (
            r"(?:provided|covered|included).{0,100}meals?",
            "Meals during the programme",
        ),
        (
            r"(?:access to|participation in).{0,100}"
            r"(?:sessions|workshops)",
            "Access to programme sessions and workshops",
        ),
        (
            r"(?:sessions|workshops).{0,100}"
            r"(?:provided|included|covered)",
            "Access to programme sessions and workshops",
        ),
        (
            r"(?:participation|official|completion)\s+certificate",
            "Official participation certificate",
        ),
        (
            r"tuition fees?.{0,100}(?:covered|provided|waived)",
            "Tuition fees",
        ),
        (
            r"(?:covered|provided|waived).{0,100}tuition fees?",
            "Tuition fees",
        ),
    ]

    for pattern, label in coverage_rules:
        if re.search(pattern, text, re.IGNORECASE):
            covered.append(label)

    exclusion_rules = [
        (r"international airfare", "International airfare"),
        (r"airfare", "International airfare"),
        (r"flight tickets?", "International airfare"),
        (r"visa application fees?", "Visa application fees"),
        (r"visa fees?", "Visa application fees"),
        (r"personal expenses?", "Personal expenses"),
        (r"travel expenses?", "Travel expenses"),
        (r"transportation costs?", "Transportation costs"),
        (r"accommodation costs?", "Accommodation costs"),
        (r"meal costs?", "Meal costs"),
    ]

    for pattern, label in exclusion_rules:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = text[
                max(0, match.start() - 100):
                min(len(text), match.end() + 120)
            ]

            if re.search(
                r"not\s+covered|not\s+included|excluded|"
                r"at\s+own\s+expense|applicant'?s?\s+responsibility|"
                r"participants?\s+(?:must|are required to)\s+pay",
                context,
                re.IGNORECASE,
            ):
                not_covered.append(label)
                break

    fee_match = re.search(
        r"(?:registration|application|participation)\s+fee"
        r".{0,100}?(?:USD|US\$|\$)\s*([\d,]+(?:\.\d{1,2})?)",
        notice,
        re.IGNORECASE,
    )

    if not fee_match:
        fee_match = re.search(
            r"(?:USD|US\$|\$)\s*([\d,]+(?:\.\d{1,2})?)"
            r".{0,100}?(?:registration|application|participation)\s+fee",
            notice,
            re.IGNORECASE,
        )

    registration_fee = None

    if fee_match:
        registration_fee = {
            "amount": f"USD {fee_match.group(1)}",
            "description": (
                "Registration fee required from selected participants "
                "to confirm participation."
            ),
        }

        not_covered.append(
            f"Registration fee ({registration_fee['amount']})"
        )

    covered = unique_items(covered)
    not_covered = unique_items(not_covered)

    # Do not classify the same expense as both covered and uncovered.
    covered = [
        item for item in covered
        if item.lower() not in {
            value.lower() for value in not_covered
        }
    ]

    if not covered and not not_covered:
        category = "Unknown"
        unknown.append(
            "The announcement does not clearly specify "
            "the funding arrangements."
        )

    elif covered and not_covered:
        category = "Partially Funded"

    elif covered:
        category = "Benefits Provided; Other Costs Unconfirmed"
        unknown.append(
            "Other expenses may apply but are not specified."
        )

    else:
        category = "No Funding Specified"

    if covered or not_covered:
        unknown.append(
            "Any expense not explicitly described as covered "
            "or uncovered remains unconfirmed."
        )

    return {
        "category": category,
        "covered": covered,
        "not_covered": not_covered,
        "unknown": unique_items(unknown),
        "registration_fee": registration_fee,
        "explanation": (
            "Funding categories are based on expenses explicitly "
            "identified in the supplied announcement. A fee paid "
            "by participants is an applicant expense, not a benefit."
        ),
    }


# ============================================================
# 7. SCHOLARSHIP SLOT EXTRACTION
# ============================================================

def extract_scholarship_slots(notice):
    total_slots = None
    category_slots = []

    for line in notice.splitlines():
        text = line.strip()

        if not text:
            continue

        total_match = re.search(
            r"(?:scholarships?\s*)?\(?(\d+)\)?\s+scholarship slots?",
            text,
            re.IGNORECASE,
        )

        if total_match:
            total_slots = int(total_match.group(1))
            continue

        category_match = re.search(
            r"(.+?):\s*(\d+)\s+slots?",
            text,
            re.IGNORECASE,
        )

        if category_match:
            category_slots.append({
                "category": category_match.group(1).strip(),
                "number": int(category_match.group(2)),
            })

    if total_slots is None and category_slots:
        total_slots = sum(
            item["number"] for item in category_slots
        )

    return {
        "total_slots": total_slots,
        "categories": category_slots,
    }


# ============================================================
# 8. PLANNING NODE
# ============================================================

def planning_node(state):
    return {
        "events": add_event(
            state,
            "Planning: Identifying assessment tasks.",
        )
    }


# ============================================================
# 9. OPPORTUNITY EXTRACTION
# ============================================================

def extraction_node(state):
    notice = state["notice"]

    lines = [
        line.strip()
        for line in notice.splitlines()
        if line.strip()
    ]

    ai_data = {}

    try:
        prompt = f"""
Extract opportunity information from the announcement.

Return JSON only. Do not invent facts.

Fields:
title, organizer, type, location, description,
required_documents.

ANNOUNCEMENT:
{notice}
"""
        ai_data = extract_json(ask_ai(prompt))

    except Exception:
        ai_data = {}

    if not isinstance(ai_data, dict):
        ai_data = {}

    # Find a likely title.
    title = "Unknown Opportunity"

    for line in lines:
        cleaned_line = re.sub(
            r"^#+\s*",
            "",
            line,
        ).strip("*# ")

        if re.search(
            r"summit|scholarship|fellowship|grant|internship|"
            r"conference|programme|program",
            cleaned_line,
            re.IGNORECASE,
        ):
            title = cleaned_line
            break

    if title == "Unknown Opportunity":
        title = (
            clean(ai_data.get("title"))
            or (lines[0] if lines else title)
        )

    organizer = "Not specified"

    organizer_match = re.search(
        r"(?:organizer|organiser|organization|organisation)"
        r"\s*:\s*(.+)",
        notice,
        re.IGNORECASE,
    )

    if organizer_match:
        organizer = organizer_match.group(1).strip()
    elif ai_data.get("organizer"):
        organizer = clean(ai_data["organizer"])

    location = "Not specified"

    location_match = re.search(
        r"(?:location|venue)\s*:\s*(.+)",
        notice,
        re.IGNORECASE,
    )

    if location_match:
        location = location_match.group(1).strip()
    elif ai_data.get("location"):
        location = clean(ai_data["location"])

    deadline = extract_deadline(notice)
    programme_dates = extract_programme_dates(notice)

    description = clean(
        ai_data.get("description")
        or "Extracted from the supplied opportunity announcement."
    )

    # Required documents.
    required_documents = []
    document_section = False

    for line in lines:
        if re.search(
            r"required documents|documents required",
            line,
            re.IGNORECASE,
        ):
            document_section = True
            continue

        if document_section:
            if re.match(
                r"#{1,4}\s+|[0-9]+\.\s+[A-Z]",
                line,
            ):
                document_section = False
                continue

            cleaned = re.sub(
                r"^[-•*]\s*",
                "",
                line,
            ).strip()

            if cleaned:
                required_documents.append(cleaned)

    if not required_documents:
        ai_documents = ai_data.get("required_documents", [])

        if isinstance(ai_documents, list):
            required_documents = ai_documents

    # Website.
    website_match = re.search(
        r"https?://[^\s)]+",
        notice,
        re.IGNORECASE,
    )

    official_website = (
        website_match.group(0).rstrip(".,")
        if website_match
        else "Not specified"
    )

    # Explicit eligibility requirements.
    eligibility_requirements = []
    in_eligibility_section = False

    for line in lines:
        if re.search(
            r"eligibility requirements|eligibility criteria",
            line,
            re.IGNORECASE,
        ):
            in_eligibility_section = True
            continue

        if in_eligibility_section:
            if re.search(
                r"funding and financial|application process|"
                r"required documents|additional information",
                line,
                re.IGNORECASE,
            ):
                in_eligibility_section = False
                continue

            cleaned = re.sub(
                r"^\d+\.\s*|^[-•*]\s*",
                "",
                line,
            ).strip()

            if cleaned:
                eligibility_requirements.append(cleaned)

    if not eligibility_requirements:
        for line in lines:
            if re.search(
                r"applicants must|applicants should|"
                r"applicant must|applicant should",
                line,
                re.IGNORECASE,
            ):
                eligibility_requirements.append(line)

    funding_details = extract_funding_details(notice)
    slots = extract_scholarship_slots(notice)

    missing_information = []

    if official_website == "Not specified":
        missing_information.append(
            "Official website not provided."
        )

    if not re.search(r"selection criteria", notice, re.IGNORECASE):
        missing_information.append(
            "Selection criteria not specified."
        )

    if not re.search(r"refund policy", notice, re.IGNORECASE):
        missing_information.append(
            "Refund policy not specified."
        )

    if not re.search(
        r"travel assistance|travel grant|travel support",
        notice,
        re.IGNORECASE,
    ):
        missing_information.append(
            "Additional travel assistance not specified."
        )

    opportunity = {
        "name": title,
        "title": title,
        "organizer": organizer,
        "type": classify_opportunity(
            notice,
            ai_data.get("type", ""),
        ),
        "location": location,
        "description": description,
        "programme_dates": programme_dates or "Not specified",
        "eligibility": eligibility_requirements,
        "required_documents": unique_items(required_documents),
        "official_website": official_website,
        "deadline": deadline or "Not specified",
        "deadline_status": get_deadline_status(deadline),
        "funding_type": funding_details["category"],
        "covered": funding_details["covered"],
        "not_covered": funding_details["not_covered"],
        "funding_unknown": funding_details["unknown"],
        "registration_fee": funding_details["registration_fee"],
        "total_slots": slots["total_slots"],
        "scholarship_slots": slots["categories"],
        "missing_information": missing_information,
    }

    return {
        "opportunity": opportunity,
        "events": add_event(
            state,
            "Extraction: Opportunity details, dates, documents "
            "and funding information extracted.",
        ),
    }


# ============================================================
# 10. WEB RESEARCH
# ============================================================

def build_research_queries(opportunity, notice):
    title = clean(
        opportunity.get("title")
        or opportunity.get("name")
    )

    organizer = clean(opportunity.get("organizer"))

    queries = []

    if title and organizer and organizer != "Not specified":
        queries.append(f'"{title}" "{organizer}"')

    if title:
        queries.append(f'"{title}" official website')
        queries.append(f'"{title}" application deadline')

    if organizer and organizer != "Not specified":
        queries.append(f'"{organizer}" official website')
        queries.append(f'"{organizer}" youth programmes')

    return unique_items(queries)[:5]


def research_node(state):
    research = {
        "status": "Not performed",
        "results": [],
        "error": "",
        "queries": [],
    }

    if not state.get("use_web", False):
        research["status"] = "Disabled"

    elif DDGS is None:
        research["status"] = "Research unavailable"
        research["error"] = (
            "The web search package is not installed. "
            "Install ddgs in the project requirements."
        )

    else:
        opportunity = state.get("opportunity", {})
        notice = state.get("notice", "")

        queries = build_research_queries(
            opportunity,
            notice,
        )

        research["queries"] = queries

        results_found = []
        seen_urls = set()
        errors = []

        if not queries:
            research["status"] = "Research unavailable"
            research["error"] = (
                "No usable search queries could be created."
            )

        else:
            try:
                # Support DDGS versions with and without timeout.
                try:
                    search_client = DDGS(timeout=20)
                except TypeError:
                    search_client = DDGS()

                for query in queries:
                    try:
                        results = search_client.text(
                            query,
                            max_results=5,
                        )

                        if results is None:
                            continue

                        for item in results:
                            if not isinstance(item, dict):
                                continue

                            url = clean(
                                item.get("href")
                                or item.get("url")
                            )

                            if not url:
                                continue

                            normalized = normalize_url(url)

                            if normalized in seen_urls:
                                continue

                            seen_urls.add(normalized)

                            results_found.append({
                                "title": clean(
                                    item.get("title")
                                    or "Untitled result"
                                ),
                                "url": url,
                                "snippet": clean(
                                    item.get("body")
                                    or item.get("snippet")
                                ),
                            })

                    except Exception as error:
                        errors.append(
                            f"{query}: {str(error)}"
                        )

                if results_found:
                    research["status"] = "Completed"
                    research["results"] = results_found

                    if errors:
                        research["error"] = (
                            "Some searches failed: "
                            + "; ".join(errors)
                        )

                elif errors:
                    research["status"] = "Research unavailable"
                    research["error"] = (
                        "Search provider errors: "
                        + "; ".join(errors)
                    )

                else:
                    research["status"] = "No results found"
                    research["error"] = (
                        "The search provider returned no usable "
                        "results. The opportunity and organiser "
                        "could not be independently verified."
                    )

            except Exception as error:
                research["status"] = "Research unavailable"
                research["error"] = (
                    f"Could not connect to the search provider: {error}"
                )

    return {
        "research": research,
        "events": add_event(
            state,
            "Research: Independent web research step completed.",
        ),
    }


# ============================================================
# 11. ELIGIBILITY ASSESSMENT
# ============================================================

def eligibility_node(state):
    applicant = state.get("applicant", {})
    notice = state.get("notice", "")
    profile = applicant_text(applicant).lower()

    matched = []
    mismatched = []
    unknown = []

    requirements = state.get(
        "opportunity",
        {},
    ).get("eligibility", [])

    # AGE
    age_match = re.search(
        r"(?:between\s+)?(\d{1,2})\s*(?:-|–|to|and)\s*"
        r"(\d{1,2})\s*(?:years?\s+old|years?\s+of\s+age)?",
        notice,
        re.IGNORECASE,
    )

    if age_match and re.search(
        r"age|years old|years of age",
        notice,
        re.IGNORECASE,
    ):
        minimum = int(age_match.group(1))
        maximum = int(age_match.group(2))

        age_text = get_applicant_value(applicant, ["age"])
        age_number = re.search(r"\d+", age_text)

        if not age_number:
            unknown.append(
                f"Age requirement: {minimum}–{maximum}. "
                "Applicant age not provided."
            )
        else:
            age = int(age_number.group())

            if minimum <= age <= maximum:
                matched.append(
                    f"Age requirement satisfied: {age} years "
                    f"(required range: {minimum}–{maximum})."
                )
            else:
                mismatched.append(
                    f"Age requirement not satisfied: applicant is "
                    f"{age}; required range is {minimum}–{maximum}."
                )

    # UNIVERSITY ENROLMENT
    requires_enrolment = bool(re.search(
        r"currently enrolled|currently be enrolled|"
        r"enrolled in a university|university student|"
        r"higher education institution|recognised higher education",
        notice,
        re.IGNORECASE,
    ))

    if requires_enrolment:
        education = get_applicant_value(
            applicant,
            ["education", "education_level", "qualification", "degree"],
        )

        academic_year = get_applicant_value(
            applicant,
            ["academic_year", "year_of_study"],
        )

        education_text = (
            education + " " + academic_year + " " + profile
        ).lower()

        if any(word in education_text for word in [
            "currently enrolled",
            "currently studying",
            "currently pursuing",
            "university student",
            "undergraduate student",
            "bachelor student",
        ]):
            matched.append(
                "University enrolment is supported by the "
                "applicant's profile; an enrolment certificate "
                "may still be required."
            )

        elif "undergraduate" in education_text or "bachelor" in education_text:
            unknown.append(
                "The applicant reports undergraduate or bachelor "
                "education, but current enrolment should be confirmed "
                "with an academic enrolment certificate."
            )

        elif not education:
            unknown.append(
                "University enrolment could not be confirmed "
                "because education information was not provided."
            )

        else:
            unknown.append(
                "Current enrolment at a recognised institution "
                "requires confirmation."
            )

    # NATIONALITY
    accepts_all_nationalities = bool(re.search(
        r"all nationalities|all nationalities are welcome|"
        r"open to all nationalities|applicants from all countries",
        notice,
        re.IGNORECASE,
    ))

    nationality = get_applicant_value(
        applicant,
        ["nationality", "citizenship"],
    )

    if accepts_all_nationalities:
        if nationality:
            matched.append(
                f"Nationality requirement satisfied: {nationality}. "
                "The programme accepts all nationalities."
            )
        else:
            unknown.append(
                "Nationality was not provided, although the "
                "announcement accepts all nationalities."
            )

    else:
        nationality_match = re.search(
            r"Bhutanese citizens?|citizens of Bhutan|"
            r"Bhutanese applicants",
            notice,
            re.IGNORECASE,
        )

        if nationality_match:
            if not nationality:
                unknown.append(
                    "Nationality information is required but missing."
                )
            elif "bhutan" in nationality.lower():
                matched.append(
                    "Bhutanese nationality requirement satisfied."
                )
            else:
                mismatched.append(
                    f"Bhutanese nationality required; "
                    f"applicant provided {nationality}."
                )

    # LEADERSHIP / COMMUNITY EXPERIENCE
    requires_leadership = bool(re.search(
        r"leadership experience|leadership skills|"
        r"emerging leaders|community activities|"
        r"community participation|volunteer experience|"
        r"active participation in community",
        notice,
        re.IGNORECASE,
    ))

    if requires_leadership:
        experience = get_applicant_value(
            applicant,
            ["experience", "work_experience", "volunteering"],
        )

        achievements = get_applicant_value(
            applicant,
            ["achievements", "leadership", "activities"],
        )

        leadership_text = (
            experience + " " + achievements
        ).lower()

        keywords = [
            "leader", "leadership", "volunteer", "volunteering",
            "community", "guided", "social service",
            "student representative", "project coordinator",
            "organised", "organized", "coordinator",
        ]

        if any(word in leadership_text for word in keywords):
            matched.append(
                "Leadership or community participation is supported "
                "by the applicant's experience or achievements. "
                "Supporting evidence may be requested."
            )
        elif not leadership_text.strip():
            unknown.append(
                "Leadership or community experience was not provided."
            )
        else:
            unknown.append(
                "Leadership or community experience requires "
                "further evidence."
            )

    # SUSTAINABLE DEVELOPMENT / INTEREST
    requires_sdgs = bool(re.search(
        r"sustainable development|SDGs?|global challenges|"
        r"international collaboration|social innovation",
        notice,
        re.IGNORECASE,
    ))

    if requires_sdgs:
        interests = get_applicant_value(
            applicant,
            ["interests", "career_interests"],
        )

        experience = get_applicant_value(
            applicant,
            ["experience"],
        )

        achievements = get_applicant_value(
            applicant,
            ["achievements"],
        )

        interest_text = (
            interests + " " + experience + " " + achievements
        ).lower()

        keywords = [
            "sustainable", "sdg", "community development",
            "social development", "innovation", "social service",
            "global", "international", "community",
        ]

        if any(word in interest_text for word in keywords):
            matched.append(
                "Interest in sustainable development, community "
                "activities or international collaboration is "
                "supported by the applicant's profile."
            )
        elif not interest_text.strip():
            unknown.append(
                "Interest in sustainable development or international "
                "collaboration has not been provided."
            )
        else:
            unknown.append(
                "The applicant's interests should be explained "
                "more clearly in the personal statement."
            )

    # ENGLISH
    requires_english = bool(re.search(
        r"English communication|basic English|"
        r"English proficiency|English language skills",
        notice,
        re.IGNORECASE,
    ))

    if requires_english:
        english = get_applicant_value(
            applicant,
            [
                "english_proficiency",
                "english",
                "language_proficiency",
            ],
        )

        if not english:
            unknown.append(
                "English communication ability was not provided."
            )

        elif any(word in english.lower() for word in [
            "good", "fluent", "advanced", "excellent",
            "proficient", "native", "intermediate",
        ]):
            matched.append(
                "English communication requirement is supported "
                f"by the applicant's stated proficiency: {english}."
            )

        elif "basic" in english.lower() and re.search(
            r"basic English",
            notice,
            re.IGNORECASE,
        ):
            matched.append(
                "Basic English communication requirement is "
                "supported by the applicant's stated ability."
            )

        elif any(word in english.lower() for word in [
            "poor", "beginner", "none", "basic",
        ]):
            unknown.append(
                f"English proficiency is stated as {english}; "
                "the required standard needs confirmation."
            )

        else:
            unknown.append(
                "English proficiency could not be compared "
                "with the stated requirement."
            )

    # GPA
    gpa_match = re.search(
        r"(?:minimum GPA|GPA of at least|GPA requirement)"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        notice,
        re.IGNORECASE,
    )

    if gpa_match:
        required_gpa = float(gpa_match.group(1))
        applicant_gpa = get_applicant_value(applicant, ["gpa"])
        applicant_gpa_match = re.search(
            r"\d+(?:\.\d+)?",
            applicant_gpa,
        )

        if not applicant_gpa_match:
            unknown.append(
                f"Minimum GPA of {required_gpa} is required, "
                "but applicant GPA was not provided."
            )
        else:
            actual_gpa = float(applicant_gpa_match.group())

            if actual_gpa >= required_gpa:
                matched.append(
                    f"GPA requirement satisfied: {actual_gpa} "
                    f"(minimum {required_gpa})."
                )
            else:
                mismatched.append(
                    f"GPA requirement not satisfied: {actual_gpa}; "
                    f"minimum required is {required_gpa}."
                )

    # If no specific requirement could be compared, do not
    # incorrectly present the applicant as fully eligible.
    if not matched and not mismatched and not unknown:
        unknown.append(
            "No specific eligibility requirements could be "
            "compared automatically."
        )

    matched = unique_items(matched)
    mismatched = unique_items(mismatched)
    unknown = unique_items(unknown)

    total = len(matched) + len(mismatched) + len(unknown)

    score = (
        round(len(matched) / total * 100)
        if total
        else 0
    )

    if mismatched:
        status = "Not Eligible"
    elif unknown:
        status = "Possibly Eligible"
    elif matched:
        status = "Eligible"
    else:
        status = "Unable to Determine"

    result = {
        "status": status,
        "score": score,
        "matched": matched,
        "mismatched": mismatched,
        "unknown": unknown,
        "criteria_count": total,
        "explanation": (
            f"{len(matched)} matched, "
            f"{len(mismatched)} mismatched, "
            f"{len(unknown)} unknown. "
            "The score represents confirmed matched criteria "
            "divided by all identified criteria. It is not an "
            "admission probability."
        ),
        "requirements": requirements,
    }

    return {
        "eligibility": result,
        "events": add_event(
            state,
            "Eligibility: Applicant information compared "
            "against the opportunity requirements.",
        ),
    }


# ============================================================
# 12. FUNDING NODE
# ============================================================

def funding_node(state):
    opportunity = state.get("opportunity", {})

    funding = {
        "category": opportunity.get("funding_type", "Unknown"),
        "covered": opportunity.get("covered", []),
        "not_covered": opportunity.get("not_covered", []),
        "unknown": opportunity.get("funding_unknown", []),
        "registration_fee": opportunity.get("registration_fee"),
        "explanation": (
            "Only expenses explicitly identified in the announcement "
            "are classified as covered or not covered. A registration "
            "fee paid by participants is treated as an applicant expense."
        ),
    }

    return {
        "funding": funding,
        "events": add_event(
            state,
            "Funding: Covered benefits, applicant expenses "
            "and uncertain costs assessed.",
        ),
    }


# ============================================================
# 13. OVERALL RECOMMENDATION
# ============================================================

def build_recommendation(state):
    eligibility = state.get("eligibility", {})
    funding = state.get("funding", {})
    opportunity = state.get("opportunity", {})
    research = state.get("research", {})

    status = eligibility.get("status", "Unknown")
    deadline_status = opportunity.get("deadline_status", "Unknown")
    research_status = research.get("status", "Not performed")

    reasons = []

    # Deadline has priority.
    if deadline_status == "Passed":
        recommendation = (
            "Do not apply for this application cycle because "
            "the application deadline has passed."
        )

        reasons.append(
            "The application deadline has passed."
        )

    elif status == "Not Eligible":
        recommendation = (
            "The applicant does not currently meet at least "
            "one stated eligibility requirement."
        )

        reasons.extend(eligibility.get("mismatched", []))

    elif status == "Eligible":
        recommendation = (
            "The applicant appears to meet the identified "
            "eligibility criteria. The applicant may consider "
            "applying after verifying the official programme "
            "and application details."
        )

        reasons.extend(eligibility.get("matched", []))

    else:
        recommendation = (
            "The applicant may be a potential candidate, but "
            "additional eligibility information or evidence "
            "is needed before making an application decision."
        )

        reasons.extend(eligibility.get("matched", []))
        reasons.extend(eligibility.get("unknown", []))

    # Research and authenticity checks.
    website = opportunity.get(
        "official_website",
        "Not specified",
    )

    verification_problem = (
        research_status != "Completed"
        or website == "Not specified"
    )

    if research_status == "No results found":
        reasons.append(
            "Independent searches returned no usable results. "
            "This does not prove the opportunity is genuine or "
            "fraudulent; its organiser and application channel "
            "could not be independently verified."
        )

    elif research_status == "Research unavailable":
        reasons.append(
            "Independent web research could not be completed "
            "because the search provider was unavailable or failed."
        )

    elif research_status == "Disabled":
        reasons.append(
            "Independent web research was disabled for this assessment."
        )

    if website == "Not specified":
        reasons.append(
            "The supplied announcement does not provide an official "
            "website or verified application link."
        )

    if verification_problem:
        reasons.append(
            "Do not pay a registration fee or submit sensitive "
            "personal documents until the organiser and application "
            "channel have been independently verified."
        )

    # Financial considerations.
    if funding.get("not_covered"):
        reasons.append(
            "Plan for applicant-paid expenses: "
            + ", ".join(funding["not_covered"])
        )

    fee = funding.get("registration_fee")

    if fee:
        reasons.append(
            f"A {fee['amount']} registration fee is stated "
            "for selected participants. Confirm the payment "
            "recipient and refund policy before paying."
        )

    if funding.get("unknown"):
        reasons.append(
            "Some funding details remain unconfirmed."
        )

    # Required documents.
    documents = opportunity.get("required_documents", [])

    if documents:
        reasons.append(
            "Prepare the required documents and check whether "
            "any additional documents are requested."
        )

    return {
        "recommendation": recommendation,
        "reasons": unique_items(reasons),
        "verification_status": (
            "Verification required"
            if verification_problem
            else "Research results available; official verification still required"
        ),
    }


# ============================================================
# 14. AGENT REFLECTION
# ============================================================

def reflection_node(state):
    reflection = []

    eligibility = state.get("eligibility", {})
    funding = state.get("funding", {})
    research = state.get("research", {})
    opportunity = state.get("opportunity", {})

    if eligibility.get("unknown"):
        reflection.append(
            "Some eligibility details or supporting evidence "
            "require confirmation."
        )

    if eligibility.get("mismatched"):
        reflection.append(
            "At least one eligibility requirement appears "
            "not to be satisfied."
        )

    if funding.get("unknown"):
        reflection.append(
            "Some expenses remain unconfirmed because the "
            "announcement does not specify their funding status."
        )

    if research.get("status") == "No results found":
        reflection.append(
            "Independent searches returned no usable results. "
            "This is not proof that the opportunity is fraudulent."
        )

    elif research.get("status") == "Research unavailable":
        reflection.append(
            "Independent web research could not be completed "
            "because the search service was unavailable."
        )

    elif research.get("status") == "Disabled":
        reflection.append(
            "Independent web research was disabled for this assessment."
        )

    if opportunity.get("deadline_status") == "Passed":
        reflection.append(
            "The application deadline has passed."
        )

    if opportunity.get("deadline_status") == "Unknown":
        reflection.append(
            "The application deadline could not be reliably extracted."
        )

    if opportunity.get("missing_information"):
        reflection.append(
            "The announcement leaves some programme information "
            "unspecified, including: "
            + ", ".join(opportunity["missing_information"])
        )

    if not reflection:
        reflection.append(
            "No major assessment uncertainties were identified. "
            "Official verification is still recommended."
        )

    recommendation = build_recommendation(state)

    return {
        "reflection": reflection,
        "recommendation": recommendation,
        "events": add_event(
            state,
            "Reflection: Assessment limitations and "
            "recommendation reviewed.",
        ),
    }


# ============================================================
# 15. FINAL REPORT
# ============================================================

def report_node(state):
    opportunity = state.get("opportunity", {})
    eligibility = state.get("eligibility", {})
    funding = state.get("funding", {})
    research = state.get("research", {})
    applicant = state.get("applicant", {})
    recommendation = state.get("recommendation", {})

    def bullets(items):
        if not items:
            return "- None identified."

        return "\n".join(
            "- " + clean(item)
            for item in items
        )

    fee = funding.get("registration_fee")

    fee_text = (
        f"{fee['amount']} — {fee['description']}"
        if fee
        else "Not specified."
    )

    research_results = []

    for item in research.get("results", []):
        research_results.append(
            f"{item.get('title', 'Untitled')} — "
            f"{item.get('url', '')}. "
            f"{item.get('snippet', '')}"
        )

    report = f"""
# OpportunityLens Bhutan Assessment

## 1. Applicant Information

Applicant: {get_applicant_value(applicant, ["name"]) or "Not provided"}

Nationality: {get_applicant_value(applicant, ["nationality"]) or "Not provided"}

Age: {get_applicant_value(applicant, ["age"]) or "Not provided"}

Education: {get_applicant_value(applicant, ["education"]) or "Not provided"}

Field of study: {get_applicant_value(applicant, ["field_of_study"]) or "Not provided"}

## 2. Opportunity Information

**{opportunity.get('title', 'Unknown Opportunity')}**

Organizer: {opportunity.get('organizer', 'Not specified')}

Opportunity type: {opportunity.get('type', 'Not specified')}

Location: {opportunity.get('location', 'Not specified')}

Programme dates: {opportunity.get('programme_dates', 'Not specified')}

Application deadline: {opportunity.get('deadline', 'Not specified')}

Deadline status: {opportunity.get('deadline_status', 'Unknown')}

Official website: {opportunity.get('official_website', 'Not specified')}

### Programme Overview

{opportunity.get('description', 'Not specified')}

### Required Documents

{bullets(opportunity.get('required_documents', []))}

### Missing Opportunity Information

{bullets(opportunity.get('missing_information', []))}

## 3. Applicant-Opportunity Matching

### Eligibility Status

{eligibility.get('status', 'Unknown')}

### Match Score

{eligibility.get('score', 0)}/100

The score is a measure of identified criteria confirmed as matched.
It is not an admission probability or a guarantee of selection.

### Requirements Matched

{bullets(eligibility.get('matched', []))}

### Requirements Not Matched

{bullets(eligibility.get('mismatched', []))}

### Requirements Requiring Confirmation

{bullets(eligibility.get('unknown', []))}

### Eligibility Explanation

{eligibility.get('explanation', 'Not available')}

## 4. Funding Assessment

Funding category: {funding.get('category', 'Unknown')}

### Covered Benefits

{bullets(funding.get('covered', []))}

### Expenses Not Covered

{bullets(funding.get('not_covered', []))}

### Unconfirmed Expenses

{bullets(funding.get('unknown', []))}

### Registration Fee

{fee_text}

### Funding Explanation

{funding.get('explanation', 'Not available')}

## 5. Independent Web Research

Research status: {research.get('status', 'Not performed')}

### Search Queries

{bullets(research.get('queries', []))}

### Research Findings

{bullets(research_results)}

### Research Diagnostic

{research.get('error', 'No diagnostic information.') or 'No errors reported.'}

Research results are not automatically proof that an opportunity is
official. A lack of search results does not establish that an
opportunity is genuine or fraudulent. Verify the organiser and
application details independently.

## 6. Overall Recommendation

{recommendation.get('recommendation', 'No recommendation available.')}

Verification status: {recommendation.get('verification_status', 'Not assessed')}

### Reasons

{bullets(recommendation.get('reasons', []))}

## 7. Agent Reflection

{bullets(state.get('reflection', []))}

---

This is a preliminary AI-assisted assessment generated by
OpportunityLens Bhutan.

The assessment is based on the applicant information and opportunity
announcement supplied to the system. It does not guarantee selection,
admission, funding or participation.

Eligibility matches may be based on profile information and do not
replace supporting documents or official verification.

Verify the official announcement, organiser, eligibility conditions,
application deadline, payment recipient and refund policy before
applying or making any payment.
"""

    return {
        "report": report,
        "events": add_event(
            state,
            "Report: Complete assessment report generated.",
        ),
    }


# ============================================================
# 16. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("planning", planning_node)
workflow.add_node("extraction", extraction_node)
workflow.add_node("research", research_node)
workflow.add_node("eligibility", eligibility_node)
workflow.add_node("funding", funding_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("planning")

workflow.add_edge("planning", "extraction")
workflow.add_edge("extraction", "research")
workflow.add_edge("research", "eligibility")
workflow.add_edge("eligibility", "funding")
workflow.add_edge("funding", "reflection")
workflow.add_edge("reflection", "report")
workflow.add_edge("report", END)

agent = workflow.compile()


# ============================================================
# 17. FUNCTION USED BY app.py
# ============================================================

def run_assessment(notice, applicant, use_web=True):
    initial_state = {
        "notice": str(notice),
        "applicant": applicant or {},
        "use_web": use_web,
        "events": [],
    }

    return agent.invoke(initial_state)
