# OpportunityLens Bhutan

### AI-Powered Opportunity Intelligence Platform

**Discover opportunities. Understand eligibility. Make informed decisions.**

OpportunityLens Bhutan is an AI-powered opportunity intelligence platform designed to help individuals explore opportunities, understand eligibility requirements, assess potential matches, and make more informed application decisions.

The platform uses an AI-assisted assessment workflow to analyse opportunity announcements against applicant profiles and present structured assessment results.

---

## 1. Project Overview

Accessing relevant scholarships, fellowships, grants, internships, and other development opportunities can be challenging.

Information is often distributed across different websites and announcements. Applicants may also find it difficult to interpret eligibility requirements, understand funding arrangements, and determine whether an opportunity matches their qualifications and experience.

OpportunityLens Bhutan aims to address these challenges through an integrated platform that combines document analysis, AI-assisted assessment, web research, and applicant profiling.

## 2. Project Objectives

The main objectives are to:

1. Develop an AI-powered platform for analysing opportunities.
2. Help applicants understand opportunity eligibility requirements.
3. Assess potential matches between applicant profiles and opportunities.
4. Extract and organise important information from opportunity announcements.
5. Present structured information about funding, research findings, and relevant events.
6. Maintain assessment history for future reference.
7. Provide an accessible and user-friendly interface.

## 3. Key Features

### 3.1 Opportunity Assessment
- Analyse opportunity announcements.
- Accept uploaded announcement documents.
- Support pasted announcement text.
- Use web research as part of the assessment workflow.
- Generate structured assessment results.

### 3.2 Applicant Profile Management
- Maintain applicant information.
- Record educational background and qualifications.
- Include skills, experience, interests, and achievements.
- Save applicant profiles for subsequent assessments.

### 3.3 Eligibility Assessment
- Present an AI-assisted eligibility assessment.
- Display a preliminary eligibility status.
- Provide relevant assessment explanations.

### 3.4 Opportunity Matching
- Generate an opportunity match score.
- Present information about the relationship between applicant qualifications and opportunity requirements.

### 3.5 Funding Analysis
- Display funding status.
- Present available funding information identified during assessment.

### 3.6 Research and Events
- Present research findings and relevant information.
- Display relevant events identified through the assessment workflow.

### 3.7 Assessment History
- Save previous assessment results.
- Review earlier assessment reports.
- Export assessment information.
- Delete individual assessment records or clear assessment history.

### 3.8 Report Export
- Download assessment results in JSON format.
- Download assessment reports in TXT format.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface and web application |
| LangChain | Language model integration |
| LangGraph | AI assessment workflow orchestration |
| Ollama | Local language model execution |
| Qwen 2.5 3B | Local AI model |
| DuckDuckGo Search | Web research |
| SQLite | Assessment and applicant profile storage |
| Pandas | Data handling and processing |
| Jupyter Notebook | Data exploration and experimentation |

## 5. System Architecture

OpportunityLens Bhutan is organised into several components.

### Application Layer
`app.py`

Provides the Streamlit interface, navigation, applicant profile management, assessment workflow, results presentation, and assessment history.

### AI Agent Layer
`agent/graph.py`

Contains the LangGraph-based assessment workflow and language model integration.

### Memory and Database Layer
`memory/database.py`

Handles database initialisation, applicant profiles, assessment storage, retrieval, and deletion.

### Document Processing Layer
`tools/document_analyzer.py`

Supports extracting text from uploaded opportunity announcement documents.

### Dataset Layer
`datasets/`

Contains project datasets and evaluation-related resources.

### Notebook Layer
`notebooks/`

Contains the project's Jupyter Notebook work.

## 6. Project Structure

```text
OpportunityLens_Bhutan/
│
├── agent/
│   ├── __init__.py
│   └── graph.py
│
├── datasets/
│   └── evaluation_template.csv
│
├── memory/
│   ├── __init__.py
│   └── database.py
│
├── notebooks/
│   └── OpportunityLens_Bhutan.ipynb
│
├── tools/
│   ├── __init__.py
│   └── document_analyzer.py
│
├── app.py
├── requirements.txt
└── .gitignore
```

## 7. Installation and Setup

### Prerequisites

- Python installed on your computer.
- Git.
- Ollama installed for local AI model execution.
- GitHub account (for deployment).

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tsheringyangden7/OpportunityLens_BhutanAIagent.git
```

### Step 2: Enter the Project Directory

```bash
cd OpportunityLens_BhutanAIagent
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Install the Local AI Model

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before starting the application.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

`http://localhost:8501`

## 8. How to Use the Application

1. Open OpportunityLens Bhutan.
2. Navigate to Applicant Profile.
3. Enter and save applicant information.
4. Navigate to New Assessment.
5. Upload an opportunity announcement or paste its text.
6. Select whether to enable web research.
7. Run the assessment.
8. Review eligibility, match score, funding, research, and events.
9. Open Assessment History to review previous results.
10. Export reports when required.

## 9. Deployment

The project is being prepared for deployment using Streamlit Community Cloud.

GitHub repository:

https://github.com/Tsheringyangden7/OpportunityLens_BhutanAIagent

The deployment requires additional configuration because the current AI model uses local Ollama. A cloud-compatible AI model connection must be configured before online AI assessments can operate.

Persistent database storage must also be considered when deploying the application.

## 10. Limitations

- AI-generated assessments are preliminary and should not be treated as official eligibility decisions.
- Assessment accuracy depends on the quality and completeness of applicant information and opportunity announcements.
- Web research may not always retrieve the latest or complete information.
- Local Ollama execution requires an appropriately configured computer.
- Cloud deployment requires an online-compatible model connection.
- Database persistence requires appropriate hosting configuration.

Applicants should always verify official requirements with the relevant opportunity provider before making an application decision.

## 11. Future Improvements

Potential future improvements include:

- Integration with cloud-hosted language models.
- Improved eligibility scoring and matching methodology.
- Expanded opportunity datasets.
- More advanced research and source verification.
- Persistent cloud database storage.
- User authentication and individual accounts.
- Email notifications for relevant opportunities.
- Improved report generation and analytics.
- Multilingual support.
- Comprehensive evaluation of assessment accuracy.

## 12. Project Status

**Project:** OpportunityLens Bhutan  
**Category:** AI-powered opportunity intelligence  
**Interface:** Streamlit  
**AI Workflow:** LangGraph  
**Language Model:** Qwen 2.5 3B through Ollama  
**Database:** SQLite  
**Deployment Target:** Streamlit Community Cloud

---

## Disclaimer

OpportunityLens Bhutan is an AI-assisted information and preliminary assessment tool. It does not represent any scholarship provider, funding institution, government agency, or official selection authority.

All eligibility assessments, funding information, and recommendations should be independently verified against official sources.

---

**OpportunityLens Bhutan**

*Discover opportunities. Understand eligibility. Make informed decisions.*
