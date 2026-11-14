import json
import re
from datetime import datetime, date
from typing import TypedDict
from urllib.parse import urlsplit, urlunsplit

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from ddgs import DDGS


# ============================================================
# 1. LOCAL AI MODEL
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


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
    report: str
    events: list


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def ask_ai(prompt):
    response = llm.invoke(prompt)
    return response.content


def extract_json(text):

    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return {}

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

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


def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def get_applicant_value(applicant, keys):

    for key in keys:

        value = applicant.get(key)

        if value is not None and clean(value):
            return clean(value)

    return ""


def normalize_bhutanese(value):

    value = value.lower().strip()

    return value in [
        "bhutanese",
        "bhutan",
        "bhutanese citizen",
    ]


def normalize_url(url):

    """
    Normalize URLs to reduce duplicate search results.
    """

    try:

        parsed = urlsplit(url.strip())

        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()

        path = parsed.path.rstrip("/")

        return urlunsplit((
            scheme,
            netloc,
            path,
            "",
            "",
        ))

    except Exception:

        return url.strip().lower()


# ============================================================
# 4. DEADLINE EXTRACTION
# ============================================================

def extract_deadline(notice):

    date_pattern = (
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
        r"|"
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
        r"|"
        r"\b\d{4}-\d{2}-\d{2}\b"
    )

    matches = list(
        re.finditer(date_pattern, notice, re.IGNORECASE)
    )

    if not matches:
        return ""

    for match in matches:

        start = max(0, match.start() - 180)
        end = min(len(notice), match.end() + 80)

        context = notice[start:end].lower()

        if any(word in context for word in [
            "deadline",
            "apply on or before",
            "on or before",
            "application closes",
            "apply before",
            "last date",
            "applications by",
        ]):

            return match.group(0).strip()

    return matches[-1].group(0).strip()


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
            return datetime.strptime(date_text, fmt).date()

        except ValueError:
            continue

    return None


def get_deadline_status(deadline):

    parsed = parse_date(deadline)

    if not parsed:
        return "Unknown"

    if parsed < date.today():
        return "Passed"

    if parsed == date.today():
        return "Deadline is today"

    return "Upcoming"


# ============================================================
# 5. FUNDING BENEFIT EXTRACTION
# ============================================================

def extract_benefits(notice):

    text = notice.lower()

    covered = []

    benefit_rules = [
        (
            ["tuition fee", "tuition fees", "tuition cost"],
            "Tuition fee",
        ),
        (
            [
                "residential charges",
                "accommodation",
                "hostel charges",
                "residence charges",
            ],
            "Residential charges / accommodation",
        ),
        (
            ["contingency"],
            "Contingency allowance",
        ),
        (
            [
                "to-and-fro travel",
                "return travel",
                "travel charges",
                "travel expenses",
                "round-trip travel",
            ],
            "Travel charges",
        ),
        (
            [
                "monthly stipend",
                "monthly allowance",
                "living allowance",
            ],
            "Monthly stipend / allowance",
        ),
        (
            [
                "admission fee",
                "admission fees",
                "registration fee",
            ],
            "Admission / registration fee",
        ),
        (
            ["books allowance", "book allowance"],
            "Books allowance",
        ),
        (
            ["insurance"],
            "Insurance",
        ),
    ]

    for keywords, label in benefit_rules:

        if any(keyword in text for keyword in keywords):
            covered.append(label)

    stipend_match = re.search(
        r"(?:monthly stipend|monthly allowance|living allowance)"
        r".{0,100}?(?:INR|Nu\.?|Rs\.?)\s*[\d,]+",
        notice,
        re.IGNORECASE,
    )

    if stipend_match:

        amount = stipend_match.group(0).strip()

        covered = [
            item for item in covered
            if item != "Monthly stipend / allowance"
        ]

        covered.append(amount)

    return covered


# ============================================================
# 6. SCHOLARSHIP SLOT EXTRACTION
# ============================================================

def extract_scholarship_slots(notice):

    total_slots = None
    category_slots = []

    lines = notice.splitlines()

    for line in lines:

        text = line.strip()

        if not text:
            continue

        total_match = re.search(
            r"(?:scholarships?\s*)?"
            r"\(?(\d+)\)?\s+scholarship slots?"
            r".{0,100}?(?:intake|applicants|available|announced)?",
            text,
            re.IGNORECASE,
        )

        if total_match and not re.search(
            r"programme|program|MBA|BIMSTEC|Master",
            text,
            re.IGNORECASE,
        ):

            total_slots = int(total_match.group(1))
            continue

        category_match = re.search(
            r"(.+?):\s*(\d+)\s+slots?",
            text,
            re.IGNORECASE,
        )

        if category_match:

            category = category_match.group(1).strip(" :-•")

            category = re.sub(
                r"^[-•]\s*",
                "",
                category,
            ).strip()

            category_slots.append({
                "number": int(category_match.group(2)),
                "category": category,
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
# 7. PLANNING NODE
# ============================================================

def planning_node(state):

    return {
        "events": add_event(
            state,
            "Planning: Identifying assessment tasks.",
        )
    }


# ============================================================
# 8. OPPORTUNITY EXTRACTION
# ============================================================

def extraction_node(state):

    notice = state["notice"]

    prompt = f"""
Extract information from this opportunity announcement.

Do not invent information.
Use "Not specified" for missing information.
Do not confuse scholarship categories with financial benefits.
Extract only requirements explicitly mentioned in the announcement.

Return valid JSON:

{{
    "name": "",
    "title": "",
    "organizer": "",
    "type": "",
    "location": "",
    "description": "",
    "eligibility": [],
    "required_documents": [],
    "official_website": "",
    "missing_information": []
}}

ANNOUNCEMENT:

{notice}
"""

    try:

        ai_data = extract_json(ask_ai(prompt))

    except Exception:

        ai_data = {}

    if not isinstance(ai_data, dict):
        ai_data = {}

    lines = [
        line.strip()
        for line in notice.splitlines()
        if line.strip()
    ]

    title = (
        ai_data.get("title")
        or ai_data.get("name")
        or (lines[0] if lines else "Unknown")
    )

    organizer_match = re.search(
        r"(?:organizer|organiser|organization|organisation)"
        r"\s*:\s*(.+)",
        notice,
        re.IGNORECASE,
    )

    if organizer_match:

        organizer = organizer_match.group(1).strip()

    elif "Royal Civil Service Commission" in notice:

        organizer = "Royal Civil Service Commission (RCSC), Bhutan"

    else:

        organizer = ai_data.get(
            "organizer",
            "Not specified",
        )

    deadline = extract_deadline(notice)

    covered = extract_benefits(notice)

    slots = extract_scholarship_slots(notice)

    eligibility_lines = []

    for line in lines:

        if re.search(
            r"eligib|civil servant|non-civil servant|"
            r"citizen|applicant must|candidates must|"
            r"applicants must",
            line,
            re.IGNORECASE,
        ):

            eligibility_lines.append(line)

    website_match = re.search(
        r"https?://[^\s)]+",
        notice,
        re.IGNORECASE,
    )

    official_website = (
        website_match.group(0).rstrip(".,")
        if website_match
        else ai_data.get(
            "official_website",
            "Not specified",
        )
    )

    opportunity = {
        "name": title,
        "title": title,
        "organizer": organizer,
        "type": ai_data.get("type", "Scholarship"),
        "location": ai_data.get("location", "Not specified"),
        "description": ai_data.get(
            "description",
            "Extracted from the supplied announcement.",
        ),
        "eligibility": eligibility_lines,
        "funding_type": (
            "Scholarship with stated benefits"
            if covered
            else "Not specified"
        ),
        "covered": covered,
        "deadline": deadline or "Not specified",
        "deadline_status": get_deadline_status(deadline),
        "total_slots": slots["total_slots"],
        "scholarship_slots": slots["categories"],
        "required_documents": ai_data.get(
            "required_documents",
            [],
        ),
        "official_website": official_website,
        "missing_information": ai_data.get(
            "missing_information",
            [],
        ),
    }

    return {
        "opportunity": opportunity,
        "events": add_event(
            state,
            "Extraction: Opportunity details extracted and checked.",
        ),
    }


# ============================================================
# 9. WEB RESEARCH HELPERS
# ============================================================

def build_research_queries(opportunity, notice):

    """
    Build focused search queries using the opportunity title,
    organizer, announcement, and official website.

    The first query is designed to be specific rather than
    searching only for a generic opportunity title.
    """

    name = clean(
        opportunity.get("name")
        or opportunity.get("title")
        or "scholarship"
    )

    organizer = clean(
        opportunity.get("organizer", "")
    )

    official_website = clean(
        opportunity.get("official_website", "")
    )

    # Remove generic words from the title to make a shorter query.
    ignored_words = {
        "open",
        "to",
        "all",
        "for",
        "the",
        "of",
        "and",
        "a",
        "an",
        "announcement",
        "application",
    }

    title_words = re.findall(
        r"[A-Za-z0-9]+",
        name,
    )

    useful_words = [
        word for word in title_words
        if word.lower() not in ignored_words
    ]

    short_title = " ".join(useful_words[:10])

    # Identify years in the announcement.
    years = re.findall(
        r"\b20\d{2}\b",
        notice,
    )

    year_text = " ".join(dict.fromkeys(years))

    queries = []

    # Specific opportunity search.
    if short_title:

        queries.append(
            f"{short_title} {year_text} RCSC Bhutan".strip()
        )

    # Include the organizer when available.
    if short_title and organizer:

        queries.append(
            f"{short_title} {organizer} Bhutan"
        )

    # Search the official website's domain.
    if official_website.startswith(("http://", "https://")):

        try:

            domain = urlsplit(official_website).netloc

            if domain:

                queries.append(
                    f"site:{domain} {short_title} {year_text}".strip()
                )

        except Exception:
            pass

    # General search as a fallback.
    if short_title:

        queries.append(
            f"{short_title} official scholarship announcement"
        )

    # Remove duplicate queries while preserving order.
    unique_queries = []

    seen_queries = set()

    for query in queries:

        query = re.sub(r"\s+", " ", query).strip()

        if query and query.lower() not in seen_queries:

            seen_queries.add(query.lower())
            unique_queries.append(query)

    return unique_queries


# ============================================================
# 10. WEB RESEARCH
# ============================================================

def research_node(state):

    research = {
        "status": "Not performed",
        "results": [],
        "error": "",
        "queries": [],
    }

    if not state.get("use_web", False):

        research["status"] = "Disabled"

    else:

        opportunity = state.get("opportunity", {})

        notice = state.get("notice", "")

        queries = build_research_queries(
            opportunity,
            notice,
        )

        research["queries"] = queries

        collected_results = []

        seen_urls = set()

        errors = []

        if not queries:

            research = {
                "status": "Research unavailable",
                "results": [],
                "error": "No usable search queries could be created.",
                "queries": [],
            }

        else:

            try:

                with DDGS(timeout=15) as search:

                    for query in queries:

                        try:

                            results = search.text(
                                query,
                                max_results=5,
                            )

                            for item in results:

                                if not isinstance(item, dict):
                                    continue

                                url = clean(
                                    item.get("href", "")
                                )

                                title = clean(
                                    item.get("title", "")
                                )

                                snippet = clean(
                                    item.get("body", "")
                                )

                                if not url:
                                    continue

                                normalized = normalize_url(url)

                                if normalized in seen_urls:
                                    continue

                                seen_urls.add(normalized)

                                collected_results.append({
                                    "title": title or "Untitled result",
                                    "url": url,
                                    "snippet": snippet,
                                })

                        except Exception as error:

                            errors.append(
                                f"Query failed: '{query}'. {error}"
                            )

                if collected_results:

                    research = {
                        "status": "Completed",
                        "results": collected_results,
                        "error": "",
                        "queries": queries,
                    }

                else:

                    if errors:

                        error_message = "; ".join(errors)

                    else:

                        error_message = (
                            "The search provider returned no usable "
                            "results for the queries."
                        )

                    research = {
                        "status": "No results found",
                        "results": [],
                        "error": error_message,
                        "queries": queries,
                    }

            except Exception as error:

                research = {
                    "status": "Research unavailable",
                    "results": [],
                    "error": (
                        f"Could not connect to the search provider: {error}"
                    ),
                    "queries": queries,
                }

    return {
        "research": research,
        "events": add_event(
            state,
            "Research: Web research step completed.",
        ),
    }


# ============================================================
# 11. ELIGIBILITY ASSESSMENT
# ============================================================

def eligibility_node(state):

    applicant = state.get("applicant", {})

    notice = state.get("notice", "")

    matched = []
    mismatched = []
    unknown = []

    # Nationality requirement.
    requires_bhutanese = bool(
        re.search(
            r"Bhutanese citizen|citizen of Bhutan|"
            r"citizens of Bhutan|Bhutanese applicants",
            notice,
            re.IGNORECASE,
        )
    )

    if requires_bhutanese:

        nationality = get_applicant_value(
            applicant,
            ["nationality", "citizenship"],
        )

        if not nationality:

            unknown.append(
                "Nationality / citizenship not provided."
            )

        elif normalize_bhutanese(nationality):

            matched.append("Bhutanese nationality")

        else:

            mismatched.append(
                f"Bhutanese nationality required; "
                f"applicant provided: {nationality}"
            )

    # Age requirement.
    age_match = re.search(
        r"(?:age|aged)\s*(?:between\s*)?"
        r"(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})",
        notice,
        re.IGNORECASE,
    )

    if age_match:

        minimum = int(age_match.group(1))

        maximum = int(age_match.group(2))

        age = get_applicant_value(
            applicant,
            ["age"],
        )

        if not age:

            unknown.append(
                f"Age requirement is {minimum}–{maximum}; "
                "applicant age not provided."
            )

        else:

            age_number_match = re.search(r"\d+", age)

            if not age_number_match:

                unknown.append(
                    "Applicant age could not be verified."
                )

            else:

                age_number = int(age_number_match.group())

                if minimum <= age_number <= maximum:

                    matched.append(
                        f"Age requirement satisfied ({age_number})."
                    )

                else:

                    mismatched.append(
                        f"Age {age_number} is outside "
                        f"the required range {minimum}–{maximum}."
                    )

    # GPA requirement.
    if re.search(
        r"\bGPA\b|grade point average",
        notice,
        re.IGNORECASE,
    ):

        gpa = get_applicant_value(
            applicant,
            ["gpa", "grade_point_average"],
        )

        if not gpa:

            unknown.append(
                "Required GPA information not provided."
            )

        else:

            unknown.append(
                "GPA provided; exact minimum requirement "
                "needs verification."
            )

    # English proficiency.
    if re.search(
        r"English proficiency|IELTS|TOEFL",
        notice,
        re.IGNORECASE,
    ):

        english = get_applicant_value(
            applicant,
            ["english_proficiency", "english", "ielts", "toefl"],
        )

        if not english:

            unknown.append(
                "Required English proficiency information not provided."
            )

        else:

            unknown.append(
                "English proficiency provided; required standard "
                "needs verification."
            )

    # Civil servant / non-civil servant route.
    if re.search(r"civil servant", notice, re.IGNORECASE):

        status = get_applicant_value(
            applicant,
            [
                "civil_servant",
                "civil_servant_status",
                "employment_status",
                "applicant_type",
                "occupation",
            ],
        )

        if not status:

            unknown.append(
                "Civil servant / non-civil servant status not provided."
            )

        else:

            status_lower = status.lower()

            if any(word in status_lower for word in [
                "non-civil servant",
                "non civil servant",
                "not a civil servant",
                "private sector",
                "student",
                "unemployed",
            ]):

                unknown.append(
                    "Non-civil servant criteria must be checked against "
                    "the Department of Workforce, Planning, and Skills "
                    "Development and Nalanda University requirements."
                )

            elif "civil servant" in status_lower:

                unknown.append(
                    "Civil servant criteria must be checked against "
                    "BCSR 2023 and the agency's approved CPD requirements."
                )

            else:

                unknown.append(
                    "Applicant employment category could not be verified."
                )

    # Education requirement.
    if re.search(
        r"bachelor|master'?s degree|undergraduate|"
        r"graduate degree|degree in",
        notice,
        re.IGNORECASE,
    ):

        education = get_applicant_value(
            applicant,
            ["education", "qualification", "degree"],
        )

        if not education:

            unknown.append(
                "Applicant education information not provided."
            )

    if not matched and not mismatched and not unknown:

        unknown.append(
            "No specific eligibility requirements could be "
            "compared automatically."
        )

    unknown = list(dict.fromkeys(unknown))

    # Score is requirement coverage, not admission probability.
    total_criteria = (
        len(matched) + len(mismatched) + len(unknown)
    )

    if total_criteria:

        score = round(
            len(matched) / total_criteria * 100
        )

    else:

        score = 0

    if mismatched:

        status = "Not Eligible"

    elif unknown:

        status = "Possibly Eligible"

    elif matched:

        status = "Eligible"

    else:

        status = "Unable to Determine"

    explanation = (
        f"{len(matched)} matched, "
        f"{len(mismatched)} mismatched, "
        f"{len(unknown)} unknown. "
        "The score represents the proportion of identified criteria "
        "confirmed as matched. It is not an admission probability. "
        "Unknown criteria are not treated as satisfied."
    )

    result = {
        "status": status,
        "matched": matched,
        "mismatched": mismatched,
        "unknown": unknown,
        "score": score,
        "explanation": explanation,
    }

    return {
        "eligibility": result,
        "events": add_event(
            state,
            "Eligibility: Applicant requirements assessed.",
        ),
    }


# ============================================================
# 12. FUNDING ASSESSMENT
# ============================================================

def funding_node(state):

    notice = state.get("notice", "")

    opportunity = state.get("opportunity", {})

    covered = opportunity.get("covered", [])

    not_covered = []

    exclusion_rules = [
        (
            r"(?:tuition|tuition fee).{0,80}"
            r"(?:not covered|not included|excluded)",
            "Tuition fee",
        ),
        (
            r"(?:accommodation|residential charges).{0,80}"
            r"(?:not covered|not included|excluded)",
            "Accommodation",
        ),
        (
            r"(?:travel|transport).{0,80}"
            r"(?:not covered|not included|excluded)",
            "Travel expenses",
        ),
        (
            r"(?:living expenses|living costs).{0,80}"
            r"(?:not covered|not included|excluded)",
            "Living expenses",
        ),
    ]

    for pattern, label in exclusion_rules:

        if re.search(pattern, notice, re.IGNORECASE):
            not_covered.append(label)

    if not covered and not not_covered:

        category = "Unknown"

        unknown = [
            "The announcement does not clearly specify covered expenses."
        ]

    elif re.search(
        r"fully funded|full scholarship",
        notice,
        re.IGNORECASE,
    ):

        category = "Fully Funded"

        unknown = []

    elif not_covered and not covered:

        category = "Not Funded"

        unknown = []

    else:

        category = "Partially Funded"

        unknown = [
            "Other expenses not mentioned cannot be confirmed "
            "as covered or excluded."
        ]

    explanation = (
        f"The announcement explicitly identifies {len(covered)} "
        "funding benefit(s). "
        "Only stated benefits are listed as covered. "
    )

    if unknown:

        explanation += (
            "Other expenses remain unknown and have not been "
            "assumed to be excluded."
        )

    result = {
        "category": category,
        "covered": covered,
        "not_covered": not_covered,
        "unknown": unknown,
        "explanation": explanation,
    }

    return {
        "funding": result,
        "events": add_event(
            state,
            "Funding: Funding coverage assessed.",
        ),
    }


# ============================================================
# 13. AGENT REFLECTION
# ============================================================

def reflection_node(state):

    reflection = []

    eligibility = state.get("eligibility", {})

    funding = state.get("funding", {})

    research = state.get("research", {})

    opportunity = state.get("opportunity", {})

    if eligibility.get("unknown"):

        reflection.append(
            "Some applicant details or eligibility requirements "
            "need confirmation."
        )

    if eligibility.get("mismatched"):

        reflection.append(
            "One or more eligibility requirements appear not to "
            "be satisfied."
        )

    if funding.get("unknown"):

        reflection.append(
            "Some funding expenses are not specified in the announcement."
        )

    if research.get("status") != "Completed":

        reflection.append(
            "Independent web research was not completed or did not "
            "return usable results."
        )

    if opportunity.get("deadline_status") == "Passed":

        reflection.append(
            "The application deadline has passed. "
            "This opportunity is not currently open for applications."
        )

    if opportunity.get("deadline_status") == "Unknown":

        reflection.append(
            "The application deadline could not be reliably extracted."
        )

    if opportunity.get("missing_information"):

        reflection.append(
            "Some opportunity information could not be established."
        )

    if not reflection:

        reflection.append(
            "No major uncertainties were identified. "
            "Official verification is still recommended."
        )

    return {
        "reflection": reflection,
        "events": add_event(
            state,
            "Reflection: Assessment limitations reviewed.",
        ),
    }


# ============================================================
# 14. REPORT GENERATION
# ============================================================

def report_node(state):

    opportunity = state.get("opportunity", {})

    eligibility = state.get("eligibility", {})

    funding = state.get("funding", {})

    research = state.get("research", {})

    reflection_text = "\n".join(
        "- " + item
        for item in state.get("reflection", [])
    )

    report = f"""
# OpportunityLens Bhutan Assessment

## Opportunity

{opportunity.get('name', 'Unknown')}

Organizer: {opportunity.get('organizer', 'Not specified')}

Type: {opportunity.get('type', 'Not specified')}

Deadline: {opportunity.get('deadline', 'Not specified')}

Deadline status: {opportunity.get('deadline_status', 'Unknown')}

Official website: {opportunity.get('official_website', 'Not specified')}

## Scholarship Slots

Total available slots: {opportunity.get('total_slots', 'Not specified')}

Category breakdown:

{json.dumps(opportunity.get('scholarship_slots', []), indent=2, ensure_ascii=False)}

## Eligibility

Status: {eligibility.get('status', 'Unknown')}

Matching score: {eligibility.get('score', 0)}/100

Matched:
{json.dumps(eligibility.get('matched', []), indent=2, ensure_ascii=False)}

Mismatched:
{json.dumps(eligibility.get('mismatched', []), indent=2, ensure_ascii=False)}

Unknown:
{json.dumps(eligibility.get('unknown', []), indent=2, ensure_ascii=False)}

Explanation:
{eligibility.get('explanation', 'Not available')}

## Funding

Category: {funding.get('category', 'Unknown')}

Covered:
{json.dumps(funding.get('covered', []), indent=2, ensure_ascii=False)}

Not covered:
{json.dumps(funding.get('not_covered', []), indent=2, ensure_ascii=False)}

Unknown:
{json.dumps(funding.get('unknown', []), indent=2, ensure_ascii=False)}

Explanation:
{funding.get('explanation', 'Not available')}

## Web Research

Status: {research.get('status', 'Not performed')}

Search queries:
{json.dumps(research.get('queries', []), indent=2, ensure_ascii=False)}

Results:
{json.dumps(research.get('results', []), indent=2, ensure_ascii=False)}

Research diagnostic:
{research.get('error', 'None')}

## Agent Reflection

{reflection_text}

---

This is a preliminary AI-assisted assessment.

Unknown information is not treated as a failed requirement.
Confirm all requirements, benefits and deadlines with the official organizer.
"""

    return {
        "report": report,
        "events": add_event(
            state,
            "Report: Assessment report generated.",
        ),
    }


# ============================================================
# 15. LANGGRAPH WORKFLOW
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
# 16. FUNCTION USED BY app.py
# ============================================================

def run_assessment(notice, applicant, use_web=True):

    initial_state = {
        "notice": str(notice),
        "applicant": applicant or {},
        "use_web": use_web,
        "events": [],
    }

    return agent.invoke(initial_state)