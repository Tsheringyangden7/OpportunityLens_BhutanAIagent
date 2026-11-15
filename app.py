
import json
import html

import streamlit as st
import pandas as pd

from agent.graph import run_assessment
from tools.document_analyzer import read_uploaded_file

from memory.database import (
    init_db,
    save_profile,
    load_profile,
    save_assessment,
    recent_assessments,
    get_assessment,
    delete_assessment,
    delete_all_assessments,
    count_assessments,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OpportunityLens Bhutan",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Dashboard",
    "New Assessment",
    "Latest Results",
    "Assessment History",
    "Applicant Profile",
]

PROFILE_FIELDS = [
    "name",
    "nationality",
    "age",
    "location",
    "education",
    "field_of_study",
    "gpa",
    "english_proficiency",
    "employment_status",
    "civil_servant_status",
    "skills",
    "experience",
    "interests",
    "achievements",
    "financial_need",
]


# ============================================================
# 2. BLUE UI DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --navy: #102B50;
        --blue: #2463A6;
        --light-blue: #EAF2FC;
        --background: #F4F7FC;
        --text: #20344D;
        --muted: #647891;
        --border: #DCE6F2;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,252,0.96);
    }

    [data-testid="stSidebar"] {
        background: #102B50;
    }

    [data-testid="stSidebar"] * {
        color: #F5F8FD;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #E4EDF8 !important;
    }

    [data-testid="stSidebar"] input {
        background: white !important;
        color: #20344D !important;
    }

    h1, h2, h3, h4 {
        color: #153B70 !important;
        font-weight: 750 !important;
    }

    p, li, label {
        color: #263D58;
    }

    .brand {
        color: white;
        font-size: 23px;
        font-weight: 800;
        margin: 8px 0 3px;
    }

    .brand-sub {
        color: #BFD1E7;
        font-size: 10px;
        letter-spacing: 1.1px;
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #3475B8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin-bottom: 7px;
    }

    .hero {
        background: linear-gradient(120deg, #12345A, #286BB1);
        padding: 27px 30px;
        border-radius: 18px;
        margin: 8px 0 25px;
        box-shadow: 0 8px 25px rgba(18,52,90,0.12);
    }

    .hero h1 {
        color: white !important;
        font-size: 29px;
        margin-bottom: 7px;
    }

    .hero p {
        color: #E4EFFB !important;
        margin: 0;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        padding: 18px 20px;
        border-radius: 14px;
        min-height: 108px;
        box-shadow: 0 3px 12px rgba(22,54,95,0.035);
    }

    .metric-label {
        color: #6B7F98;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .5px;
        margin-bottom: 9px;
    }

    .metric-value {
        color: #153B70;
        font-size: 25px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px 22px;
        margin: 10px 0 17px;
    }

    .panel-title {
        color: #153B70;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .muted {
        color: #647891;
        font-size: 13px;
        line-height: 1.6;
    }

    .notice {
        background: #EAF2FC;
        border-left: 4px solid #3478C6;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 12px 0;
        color: #24466F;
    }

    .research-card {
        background: white;
        border: 1px solid #DCE6F2;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0;
    }

    .research-title {
        color: #155A9C;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .research-url {
        color: #3475B8;
        font-size: 12px;
        overflow-wrap: anywhere;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #205B9C;
        color: white !important;
        border: 1px solid #205B9C;
        border-radius: 9px;
        min-height: 43px;
        font-weight: 650;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #153B70;
        border-color: #153B70;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: #EAF2FC;
        color: #153B70 !important;
        border: 1px solid #C9D9ED;
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    .stTextArea textarea {
        border-radius: 9px;
        border-color: #CFDCEC;
        background: white;
        color: #20344D;
    }

    [data-testid="stFileUploader"] {
        background: #F8FBFF;
        border: 1px dashed #9BB8D9;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stTabs"] button {
        color: #294765 !important;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #155A9C !important;
        border-bottom-color: #2463A6 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 5px;
    }

    [data-testid="stSidebar"] [role="radio"] {
        background: rgba(255,255,255,0.04);
        padding: 10px 12px;
        border-radius: 9px;
        margin: 2px 0;
    }

    [data-testid="stSidebar"] [role="radio"]:hover {
        background: rgba(255,255,255,0.13);
    }

    hr {
        border-color: #E0E8F2;
    }

    footer, #MainMenu {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATABASE
# ============================================================

init_db()


# ============================================================
# 4. SESSION STATE
# ============================================================

defaults = {
    "assessment_result": None,
    "assessment_notice": "",
    "assessment_error": None,
    "navigation": "Dashboard",
    "pending_navigation": None,
    "profile_id": "student01",
    "loaded_profile_id": None,
    "history_view_id": None,
    "pending_delete_one": None,
    "pending_delete_all": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# IMPORTANT:
# Apply pending navigation BEFORE creating the sidebar radio widget.
# This prevents StreamlitWidgetAlreadyInstantiatedError.

pending_page = st.session_state.pending_navigation

if pending_page in PAGES:
    st.session_state.navigation = pending_page

st.session_state.pending_navigation = None


# ============================================================
# 5. PROFILE FUNCTIONS
# ============================================================

def load_profile_into_session(profile_id):

    saved = load_profile(profile_id)

    if not isinstance(saved, dict):
        saved = {}

    for field in PROFILE_FIELDS:
        value = saved.get(field, "")

        st.session_state[f"profile_{field}"] = (
            "" if value is None else str(value)
        )

    st.session_state.loaded_profile_id = profile_id


def get_profile_from_session():

    return {
        field: st.session_state.get(f"profile_{field}", "")
        for field in PROFILE_FIELDS
    }


if st.session_state.loaded_profile_id is None:
    load_profile_into_session(st.session_state.profile_id)


# ============================================================
# 6. GENERAL HELPER FUNCTIONS
# ============================================================

def safe_json(value):

    if isinstance(value, str):
        value = value.strip()

        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

    return value


def get_section(result, key):

    result = safe_json(result)

    if isinstance(result, dict):
        return safe_json(result.get(key, {}))

    return {}


def display_value(value):

    value = safe_json(value)

    if value is None or value == "":
        st.caption("Not provided.")
        return

    if isinstance(value, dict):

        if not value:
            st.caption("No information returned.")
            return

        for key, item in value.items():
            label = str(key).replace("_", " ").title()
            st.markdown(f"**{label}**")
            display_value(item)

    elif isinstance(value, list):

        if not value:
            st.caption("No items returned.")
            return

        for item in value:

            if isinstance(item, (dict, list)):
                st.markdown("---")
                display_value(item)
            else:
                st.markdown(f"- {item}")

    else:

        text_value = str(value)

        if "\n" in text_value and (
            text_value.lstrip().startswith("#")
            or "\n## " in text_value
            or "\n- " in text_value
        ):
            st.markdown(text_value)
        else:
            st.write(text_value)


def result_as_text(result):

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ------------------------------------------------------------
# HISTORY NORMALIZATION
# ------------------------------------------------------------

def get_history(profile_id):

    try:
        records = recent_assessments(profile_id, limit=500)

        if records is None:
            return []

        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")

        normalized = []

        for item in records:

            try:

                if hasattr(item, "keys"):
                    row = dict(item)
                elif isinstance(item, dict):
                    row = item
                else:
                    continue

                raw_id = (
                    row.get("ID")
                    or row.get("id")
                    or row.get("assessment_id")
                )

                if raw_id is None or str(raw_id).strip() == "":
                    continue

                try:
                    assessment_id = int(raw_id)
                except (ValueError, TypeError, OverflowError):
                    continue

                opportunity = (
                    row.get("Opportunity")
                    or row.get("opportunity")
                    or row.get("title")
                    or "Opportunity assessment"
                )

                date_value = (
                    row.get("Date")
                    or row.get("date")
                    or row.get("created_at")
                    or "Unknown date"
                )

                result_value = (
                    row.get("Result")
                    or row.get("result")
                    or row.get("eligibility")
                    or "Not assessed"
                )

                normalized.append({
                    "ID": assessment_id,
                    "Opportunity": str(opportunity),
                    "Date": str(date_value),
                    "Result": str(result_value),
                })

            except Exception:
                continue

        return normalized

    except Exception as error:
        st.warning(f"Could not load assessment history: {error}")
        return []


def get_opportunity_title(result):

    opportunity = get_section(result, "opportunity")

    if isinstance(opportunity, dict):
        return (
            opportunity.get("title")
            or opportunity.get("name")
            or "Opportunity assessment"
        )

    if isinstance(opportunity, str) and opportunity.strip():
        return opportunity

    return "Opportunity assessment"


def render_section_card(title, content, help_text=None):

    st.markdown(f"### {title}")

    if help_text:
        st.caption(help_text)

    display_value(content)


def set_navigation(page):

    # This function is used only as a button callback.
    # Streamlit executes callbacks before creating the widgets.

    if page in PAGES:
        st.session_state.navigation = page


def format_score(eligibility):

    if not isinstance(eligibility, dict):
        return "Not provided"

    score = eligibility.get(
        "score",
        eligibility.get(
            "matching_score",
            eligibility.get("match_score"),
        ),
    )

    if score is None or score == "":
        return "Not provided"

    if isinstance(score, (int, float)):
        return f"{score:g}/100"

    score_text = str(score).strip()

    if not score_text:
        return "Not provided"

    if "/100" not in score_text:
        score_text += "/100"

    return score_text


def render_research(research):

    if not isinstance(research, dict):
        st.info("No research information is available.")
        return

    status = research.get("status", "Unknown")
    results = research.get("results", [])
    error = research.get("error", "")

    st.markdown("### Research status")
    st.write(status)

    if results:

        st.markdown("### Sources found")

        for index, item in enumerate(results, start=1):

            if not isinstance(item, dict):
                continue

            title = item.get("title") or f"Research result {index}"
            url = item.get("url", "")
            snippet = item.get("snippet", "")

            st.markdown(
                f"""
                <div class="research-card">
                    <div class="research-title">
                        {html.escape(str(title))}
                    </div>
                    <div class="research-url">
                        {html.escape(str(url))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if url and str(url).startswith(("https://", "http://")):
                st.link_button(
                    "Open source",
                    url,
                    key=f"research_link_{index}_{abs(hash(url))}",
                )

            if snippet:
                st.write(snippet)

            st.markdown("---")

    else:
        st.info("No usable web research results were returned.")

    if error:
        st.markdown("### Research diagnostic")
        st.warning(str(error))


# ============================================================
# 7. COMPLETE ASSESSMENT REPORT VIEWER
# ============================================================

def render_assessment_result(
    result,
    source_notice=None,
    download_prefix="assessment",
):

    result = safe_json(result)

    if not isinstance(result, dict):
        result = {"report": str(result)}

    opportunity = get_section(result, "opportunity")
    eligibility = get_section(result, "eligibility")
    funding = get_section(result, "funding")
    research = get_section(result, "research")
    events = get_section(result, "events")
    report = get_section(result, "report")

    st.markdown(
        """
        <div class="notice">
            This is an AI-assisted preliminary assessment.
            Review the evidence and verify all official requirements
            before making an application decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header(get_opportunity_title(result))

    eligibility_status = (
        eligibility.get("status", "Not assessed")
        if isinstance(eligibility, dict)
        else "Not assessed"
    )

    match_score = format_score(eligibility)

    funding_category = (
        funding.get("category", "Not assessed")
        if isinstance(funding, dict)
        else "Not assessed"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        safe_status = html.escape(str(eligibility_status))

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ELIGIBILITY</div>
                <div class="metric-value" style="font-size:20px;">
                    {safe_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        safe_score = html.escape(str(match_score))

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MATCH SCORE</div>
                <div class="metric-value">{safe_score}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        safe_funding = html.escape(str(funding_category))

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">FUNDING STATUS</div>
                <div class="metric-value" style="font-size:20px;">
                    {safe_funding}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Assessment details")

    (
        tab_overview,
        tab_eligibility,
        tab_funding,
        tab_research,
        tab_events,
        tab_report,
    ) = st.tabs([
        "Overview",
        "Eligibility",
        "Funding",
        "Research",
        "Events",
        "Final Report",
    ])

    with tab_overview:

        render_section_card(
            "Opportunity overview",
            opportunity,
        )

        if source_notice:

            with st.expander("View source announcement"):
                st.write(source_notice)

    with tab_eligibility:

        render_section_card(
            "Eligibility assessment",
            eligibility,
            "Check each requirement against the applicant's confirmed information.",
        )

    with tab_funding:

        render_section_card(
            "Funding analysis",
            funding,
            "Advertised funding is not the same as funding awarded or guaranteed.",
        )

    with tab_research:
        render_research(research)

    with tab_events:

        render_section_card(
            "Workflow events",
            events,
        )

    with tab_report:

        if isinstance(report, str) and report.strip():
            st.markdown(report)
        else:
            render_section_card(
                "Final assessment report",
                report,
            )

    st.markdown("---")
    st.markdown("### Download this assessment")

    col_a, col_b = st.columns(2)

    with col_a:

        st.download_button(
            "Download assessment (JSON)",
            data=result_as_text(result),
            file_name=f"{download_prefix}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{download_prefix}_json_download",
        )

    with col_b:

        report_text = (
            report
            if isinstance(report, str) and report.strip()
            else result_as_text(result)
        )

        st.download_button(
            "Download report (TXT)",
            data=str(report_text),
            file_name=f"{download_prefix}_report.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"{download_prefix}_txt_download",
        )


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">🔎 OpportunityLens</div>
        <div class="brand-sub">
            BHUTAN · OPPORTUNITY INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### WORKSPACE")

    selected_page = st.radio(
        "Main navigation",
        PAGES,
        key="navigation",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### CURRENT PROFILE")

    current_profile_id = st.text_input(
        "Profile ID",
        key="profile_id",
        help="Use the same ID to access your saved applicant profile and history.",
    ).strip()

    if not current_profile_id:
        current_profile_id = "anonymous"

    if current_profile_id != st.session_state.loaded_profile_id:

        load_profile_into_session(current_profile_id)

        st.session_state.assessment_result = None
        st.session_state.assessment_notice = ""
        st.session_state.history_view_id = None
        st.session_state.pending_delete_one = None
        st.session_state.pending_delete_all = False

        st.rerun()

    st.caption(f"Using profile: {current_profile_id}")

    st.markdown("---")

    st.caption("Your workspace")

    st.caption(
        f"Saved assessments: {count_assessments(current_profile_id)}"
    )


# ============================================================
# 9. COMMON HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">AI-POWERED OPPORTUNITY INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>OpportunityLens Bhutan</h1>
        <p>
            Discover opportunities. Understand eligibility.
            Make informed decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. DASHBOARD
# ============================================================

if selected_page == "Dashboard":

    st.title("Dashboard")

    st.write(
        "Your workspace for assessing scholarships, grants, "
        "internships and other opportunities."
    )

    history = get_history(current_profile_id)

    total_assessments = count_assessments(current_profile_id)

    latest_date = (
        history[0].get("Date", "No activity yet")
        if history
        else "No activity yet"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">SAVED ASSESSMENTS</div>
                <div class="metric-value">{total_assessments}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">ASSESSMENT ENGINE</div>
                <div class="metric-value">Local AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        safe_date = html.escape(str(latest_date))

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LATEST ACTIVITY</div>
                <div class="metric-value" style="font-size:17px;">
                    {safe_date}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Get started")

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    🔍 New opportunity assessment
                </div>
                <div class="muted">
                    Upload an announcement or paste its contents
                    to begin an AI-assisted assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "Start an assessment",
            use_container_width=True,
            on_click=set_navigation,
            args=("New Assessment",),
            key="dashboard_new_assessment",
        )

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    📚 Review assessment history
                </div>
                <div class="muted">
                    Reopen previous assessments, read their complete
                    reports and manage saved records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "Open assessment history",
            use_container_width=True,
            on_click=set_navigation,
            args=("Assessment History",),
            key="dashboard_history",
        )

    st.markdown("## Recent assessments")

    if history:

        dashboard_history = pd.DataFrame(history)

        st.dataframe(
            dashboard_history.drop(columns=["ID"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No assessments have been saved for this profile yet."
        )

    st.markdown("## Quick guidance")

    st.markdown("""
    - Keep your applicant profile accurate and up to date.
    - Use the full official announcement whenever possible.
    - Treat AI-generated eligibility and funding results as preliminary.
    - Confirm deadlines, eligibility rules and funding with the official organiser.
    """)


# ============================================================
# 11. NEW ASSESSMENT
# ============================================================

elif selected_page == "New Assessment":

    st.title("New opportunity assessment")

    st.write(
        "Provide the opportunity announcement. You can upload "
        "a document or paste the announcement directly."
    )

    st.markdown("### 1. Opportunity announcement")

    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, TXT or Markdown file",
        type=["pdf", "docx", "txt", "md"],
        key="new_assessment_upload",
    )

    pasted_notice = st.text_area(
        "Or paste the announcement here",
        height=240,
        placeholder=(
            "Paste the full scholarship, grant, internship "
            "or funding announcement..."
        ),
        key="new_assessment_text",
    )

    st.markdown("### 2. Research settings")

    use_web = st.checkbox(
        "Enable independent web research",
        value=True,
        help=(
            "Enable this to let the agent search for additional "
            "information. An internet connection is required."
        ),
    )

    profile = get_profile_from_session()

    with st.expander("Applicant profile being used"):

        st.caption(f"Profile ID: {current_profile_id}")

        for key, value in profile.items():

            label = key.replace("_", " ").title()

            st.markdown(
                f"**{label}:** {value if value else 'Not provided'}"
            )

        st.button(
            "Edit applicant profile",
            on_click=set_navigation,
            args=("Applicant Profile",),
            key="edit_profile_from_assessment",
        )

    if st.button(
        "🔎 Investigate opportunity",
        type="primary",
        use_container_width=True,
        key="run_new_assessment",
    ):

        notice = pasted_notice.strip()

        if uploaded_file is not None:

            try:

                extracted = read_uploaded_file(uploaded_file)

                if isinstance(extracted, dict):
                    extracted = (
                        extracted.get("text")
                        or extracted.get("content")
                        or ""
                    )

                if extracted and str(extracted).strip():

                    uploaded_text = str(extracted).strip()

                    if notice:
                        notice = (
                            uploaded_text
                            + "\n\n"
                            + "Additional announcement information:\n"
                            + notice
                        )
                    else:
                        notice = uploaded_text

            except Exception as error:

                st.error(
                    f"Could not read the uploaded document: {error}"
                )

                st.stop()

        if not notice:

            st.error(
                "Please upload a document or paste an announcement."
            )

            st.stop()

        if len(notice) < 40:

            st.warning(
                "This announcement is very short. "
                "The agent may not have enough information."
            )

        try:

            save_profile(
                current_profile_id,
                profile,
            )

            with st.spinner(
                "The OpportunityLens agent is analysing "
                "the announcement. This may take a few minutes..."
            ):

                result = run_assessment(
                    notice,
                    profile,
                    use_web=use_web,
                )

            if not isinstance(result, dict):
                result = {"report": str(result)}

            st.session_state.assessment_result = result
            st.session_state.assessment_notice = notice
            st.session_state.assessment_error = None

            opportunity_title = get_opportunity_title(result)

            save_assessment(
                current_profile_id,
                opportunity_title,
                result,
            )

            # FIX:
            # Do not modify the navigation widget's key here.
            # Queue the destination for the next script run.

            st.session_state.pending_navigation = "Latest Results"

            st.rerun()

        except Exception as error:

            st.session_state.assessment_error = str(error)

            st.error("The assessment could not be completed.")
            st.exception(error)


# ============================================================
# 12. LATEST RESULTS
# ============================================================

elif selected_page == "Latest Results":

    st.title("Latest assessment results")

    result = st.session_state.assessment_result

    if not result:

        st.info(
            "There is no assessment result in this browser session yet. "
            "Run a new assessment to generate a report. Previously saved "
            "assessments can be reviewed in Assessment History."
        )

        st.button(
            "Start a new assessment",
            use_container_width=True,
            on_click=set_navigation,
            args=("New Assessment",),
        )

    else:

        render_assessment_result(
            result,
            source_notice=st.session_state.assessment_notice,
            download_prefix="latest_assessment",
        )


# ============================================================
# 13. ASSESSMENT HISTORY
# ============================================================

elif selected_page == "Assessment History":

    st.title("Assessment history")

    st.write(
        "Open and review your previous assessments. "
        "Every saved assessment can be reopened with its complete report."
    )

    history = get_history(current_profile_id)

    total = count_assessments(current_profile_id)

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL SAVED ASSESSMENTS</div>
            <div class="metric-value">{total}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total > len(history):

        st.warning(
            "Some database records do not have a valid ID. "
            "Those records cannot be opened or deleted individually. "
            "You can still use the delete-all option below to clear "
            "the history for this profile."
        )

    if not history:

        if total == 0:
            st.info(
                "No saved assessments were found for this profile."
            )
        else:
            st.info(
                "Assessment records exist, but none has a valid ID "
                "for individual access."
            )

        st.button(
            "Create your first assessment",
            on_click=set_navigation,
            args=("New Assessment",),
            use_container_width=True,
        )

    else:

        st.markdown("## Your saved assessments")

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn(
                    "Record ID",
                    format="%d",
                ),
                "Opportunity": st.column_config.TextColumn(
                    "Opportunity",
                    width="large",
                ),
                "Date": st.column_config.TextColumn(
                    "Date",
                    width="medium",
                ),
                "Result": st.column_config.TextColumn(
                    "Eligibility",
                    width="medium",
                ),
            },
        )

        st.download_button(
            "Download history (CSV)",
            data=history_df.to_csv(index=False),
            file_name="opportunitylens_history.csv",
            mime="text/csv",
            key="history_csv_download",
        )

        st.markdown("---")
        st.markdown("## Open a previous assessment")

        assessment_options = {}

        for item in history:

            # IDs have already been validated by get_history().
            assessment_id = item["ID"]

            label = (
                f"#{assessment_id} · "
                f"{item['Opportunity']} · "
                f"{item['Date']}"
            )

            assessment_options[label] = assessment_id

        option_labels = list(assessment_options.keys())

        selected_label = st.selectbox(
            "Choose an assessment to review",
            option_labels,
            key="history_assessment_selector",
        )

        selected_id = assessment_options[selected_label]

        if st.button(
            "📖 Open complete assessment report",
            type="primary",
            use_container_width=True,
            key="open_historical_report",
        ):

            st.session_state.history_view_id = selected_id
            st.rerun()

        # ----------------------------------------------------
        # DISPLAY SELECTED HISTORICAL REPORT
        # ----------------------------------------------------

        historical_id = st.session_state.history_view_id

        valid_ids = {item["ID"] for item in history}

        if historical_id is not None and historical_id not in valid_ids:
            st.session_state.history_view_id = None
            historical_id = None

        if historical_id is not None:

            historical_record = get_assessment(
                current_profile_id,
                historical_id,
            )

            if historical_record is None:

                st.warning(
                    "This assessment no longer exists. "
                    "Please select another assessment."
                )

                st.session_state.history_view_id = None

            else:

                st.markdown("---")

                historical_result = safe_json(
                    historical_record["result"]
                )

                st.markdown(
                    f"## Historical report: "
                    f"{historical_record['opportunity']}"
                )

                st.caption(
                    f"Assessment ID: {historical_record['id']} · "
                    f"Saved: {historical_record['created_at']}"
                )

                historical_notice = (
                    historical_record.get("notice")
                    or historical_record.get("announcement")
                    or historical_record.get("source_notice")
                )

                render_assessment_result(
                    historical_result,
                    source_notice=historical_notice,
                    download_prefix=(
                        f"historical_assessment_{historical_record['id']}"
                    ),
                )

        # ----------------------------------------------------
        # DELETE ONE ASSESSMENT
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## Manage assessment history")

        st.markdown("### Delete one assessment")

        st.warning(
            "Deleting an assessment permanently removes its saved report."
        )

        delete_label = st.selectbox(
            "Select the assessment to delete",
            option_labels,
            key="delete_assessment_selector",
        )

        delete_id = assessment_options[delete_label]
        delete_title = delete_label

        if st.button(
            "Prepare deletion",
            use_container_width=True,
            key="prepare_delete_one",
        ):

            st.session_state.pending_delete_one = delete_id
            st.rerun()

        if st.session_state.pending_delete_one is not None:

            pending_id = st.session_state.pending_delete_one

            pending_title = next(
                (
                    label
                    for label, value in assessment_options.items()
                    if value == pending_id
                ),
                None,
            )

            if pending_title is None:

                st.session_state.pending_delete_one = None
                st.rerun()

            st.error(
                f"Confirm permanent deletion of: {pending_title}"
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:

                if st.button(
                    "Confirm deletion",
                    type="primary",
                    use_container_width=True,
                    key="confirm_delete_one_button",
                ):

                    deleted = delete_assessment(
                        current_profile_id,
                        pending_id,
                    )

                    st.session_state.pending_delete_one = None

                    if st.session_state.history_view_id == pending_id:
                        st.session_state.history_view_id = None

                    if deleted:
                        st.success("Assessment deleted successfully.")
                    else:
                        st.error(
                            "The assessment could not be found or deleted."
                        )

                    st.rerun()

            with cancel_col:

                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="cancel_delete_one_button",
                ):

                    st.session_state.pending_delete_one = None
                    st.rerun()

    # --------------------------------------------------------
    # DELETE ALL HISTORY
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### Delete all assessment history")

    st.warning(
        f"This will permanently delete all {total} assessment "
        f"records belonging to profile '{current_profile_id}'. "
        "Your applicant profile and other profiles will not be deleted."
    )

    if st.button(
        "Prepare deletion of all history",
        use_container_width=True,
        key="prepare_delete_all",
        disabled=total == 0,
    ):

        st.session_state.pending_delete_all = True
        st.rerun()

    if st.session_state.pending_delete_all:

        st.error(
            f"Are you sure you want to permanently delete all "
            f"{total} assessment records for profile "
            f"'{current_profile_id}'?"
        )

        confirm_all_col, cancel_all_col = st.columns(2)

        with confirm_all_col:

            if st.button(
                "Yes, delete all history",
                type="primary",
                use_container_width=True,
                key="confirm_delete_all_button",
            ):

                try:

                    deleted_count = delete_all_assessments(
                        current_profile_id
                    )

                    st.session_state.history_view_id = None
                    st.session_state.pending_delete_one = None
                    st.session_state.pending_delete_all = False

                    st.success(
                        f"Successfully deleted {deleted_count} "
                        "assessment record(s)."
                    )

                except Exception as error:

                    st.error(
                        f"Could not delete assessment history: {error}"
                    )

                st.rerun()

        with cancel_all_col:

            if st.button(
                "Cancel",
                use_container_width=True,
                key="cancel_delete_all_button",
            ):

                st.session_state.pending_delete_all = False
                st.rerun()


# ============================================================
# 14. APPLICANT PROFILE
# ============================================================

elif selected_page == "Applicant Profile":

    st.title("Applicant profile")

    st.write(
        "Keep your information accurate. The agent uses this profile "
        "to compare your background with opportunity requirements."
    )

    st.markdown(
        """
        <div class="notice">
            Only enter details that are true and that you want to use
            for assessments. Blank fields are treated as unknown;
            the system should not assume them.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("applicant_profile_form"):

        st.markdown("### Personal information")

        col1, col2 = st.columns(2)

        with col1:

            st.text_input(
                "Full name",
                key="profile_name",
            )

            st.text_input(
                "Nationality",
                key="profile_nationality",
            )

            st.text_input(
                "Age",
                key="profile_age",
            )

        with col2:

            st.text_input(
                "Location",
                key="profile_location",
            )

            st.text_input(
                "Education level",
                key="profile_education",
            )

            st.text_input(
                "Field of study",
                key="profile_field_of_study",
            )

        st.markdown("### Academic information")

        col3, col4 = st.columns(2)

        with col3:

            st.text_input(
                "GPA (if applicable)",
                key="profile_gpa",
            )

        with col4:

            st.text_input(
                "English proficiency (IELTS, TOEFL or other)",
                key="profile_english_proficiency",
            )

        st.markdown("### Employment information")

        col5, col6 = st.columns(2)

        with col5:

            st.selectbox(
                "Employment status",
                [
                    "",
                    "Student",
                    "Employed",
                    "Unemployed",
                    "Self-employed",
                    "Other",
                ],
                key="profile_employment_status",
            )

        with col6:

            st.selectbox(
                "Civil servant status",
                [
                    "",
                    "Civil servant",
                    "Non-civil servant",
                    "Not applicable",
                    "Not sure",
                ],
                key="profile_civil_servant_status",
            )

        st.markdown("### Background")

        st.text_area(
            "Skills",
            key="profile_skills",
            height=90,
        )

        st.text_area(
            "Work, volunteering or project experience",
            key="profile_experience",
            height=100,
        )

        st.text_area(
            "Interests",
            key="profile_interests",
            height=80,
        )

        st.text_area(
            "Achievements",
            key="profile_achievements",
            height=80,
        )

        st.text_area(
            "Funding needs (optional)",
            key="profile_financial_need",
            height=80,
        )

        submitted = st.form_submit_button(
            "Save applicant profile",
            type="primary",
            use_container_width=True,
        )

        if submitted:

            profile_to_save = get_profile_from_session()

            save_profile(
                current_profile_id,
                profile_to_save,
            )

            st.success(
                "Your applicant profile has been saved."
            )

    st.markdown("---")

    st.caption(
        f"Profile ID: {current_profile_id}. "
        "Use this same ID later to load this profile."
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:12px 0 20px 0;">
        <div style="font-weight:750; color:#153B70;">
            OpportunityLens Bhutan
        </div>
        <div style="font-size:12px; color:#647891;">
            AI-powered opportunity assessment prototype · Academic project
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
