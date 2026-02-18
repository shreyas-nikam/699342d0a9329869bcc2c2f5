Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted for clarity and professionalism.

---

# QuLab: Lab 32: Materiality-Driven ESG Research Agent

## An Investment Analyst's Edge with Generative AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](link-to-your-deployed-app-if-any)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Project Title and Description

This project, **"QuLab: Lab 32: ESG Research Agent"**, is a Streamlit application designed to empower investment professionals with a **materiality-driven ESG screening workflow**. As a CFA Charterholder and Investment Professional, identifying long-term value and mitigating risks requires a nuanced understanding of Environmental, Social, and Governance (ESG) factors. Traditional "check-the-box" ESG approaches often fall short by not pinpointing the financially material aspects for each company.

This application provides a hands-on lab experience to leverage cutting-edge Generative AI agents for a streamlined, intelligent ESG analysis. It focuses on identifying industry-specific material ESG topics, gathering relevant data, systematically scoring companies, and refining assessments through an "Evaluator-Optimizer" loop, mimicking a senior analyst's review process. The ultimate goal is to generate a comprehensive ESG scorecard and detailed company profiles to inform better risk identification and capital allocation decisions.

## 2. Features

The application offers a guided workflow through several interactive pages, each serving a critical part of the ESG analysis process:

*   **Home / Introduction**: Provides an overview of the project's goals, methodology, and the significance of materiality-driven ESG analysis for investment professionals.
*   **Define Portfolio**: Allows users to input a custom list of 5-10 stock tickers to define their investment portfolio for analysis.
*   **ESG Agent Workflow**:
    *   Initiates the core ESG assessment process for each company in the defined portfolio.
    *   Utilizes Generative AI agents equipped with specialized "tools" to gather environmental metrics, scan controversies, retrieve governance data, and determine SASB materiality.
    *   Employs an **"Evaluator-Optimizer"** loop for each assessment, where an evaluator agent reviews the initial assessment and provides feedback, prompting the primary agent to revise its analysis for improved quality and consistency.
    *   Generates materiality-weighted composite ESG scores based on the industry-specific relevance of E, S, and G pillars.
*   **Portfolio ESG Scorecard**: Presents a ranked overview of all portfolio companies based on their materiality-weighted ESG scores, along with individual E, S, and G scores, and investment recommendations. Includes a clear explanation of the composite score formula.
*   **Individual Company Profiles & Visualizations**:
    *   Offers detailed ESG profiles for selected companies, including rationale for scores, controversies, peer comparisons, and key risks.
    *   Visualizes ESG pillar performance using a **Radar Chart**.
    *   Provides an **Agent Reasoning Trace** and **Evaluator Feedback log**, showing the AI agent's thought process, tool calls, and revisions during the assessment.
    *   Explains the **SASB Materiality Router** and lists the identified material topics for the company's industry.
    *   Includes a **Portfolio Controversy Heat Map** to visualize aggregated controversy data across the entire portfolio.
*   **Score Consistency Analysis**:
    *   Addresses the inherent variability of LLM-generated scores by allowing users to run the ESG assessment for a single company multiple times.
    *   Displays raw scores for each run and calculates score ranges.
    *   Visualizes score consistency using a **Box Plot** for E, S, G, and Composite scores.
    *   Includes a "Practitioner Warning" emphasizing the nature of LLM variability in qualitative assessments.

## 3. Getting Started

Follow these steps to set up and run the QuLab ESG Research Agent application locally.

### 3.1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9+**
*   **`pip`** (Python package installer)

### 3.2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    (Replace `your-username/your-repo-name` with the actual repository path.)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    Create a `requirements.txt` file in the root directory of your project with the following contents:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    plotly
    langchain
    langchain_core
    langchain_openai
    python-dotenv # If using .env for API keys
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up API Keys:**
    This application heavily relies on Generative AI models (LLMs), which typically require an API key. You will need to obtain an API key from your preferred LLM provider (e.g., OpenAI, Anthropic, Google Gemini).

    Create a `.env` file in the root directory of your project and add your API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    # Or ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    # Or GOOGLE_API_KEY="your_google_api_key_here"
    ```
    *Note: The specific environment variable name (`OPENAI_API_KEY` in this example) depends on the `langchain` integration used within the `source.py` file.*

## 4. Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active):
    ```bash
    source venv/bin/activate # macOS/Linux
    .\venv\Scripts\activate   # Windows
    ```

2.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually at `http://localhost:8501`).

### Basic Usage Instructions:

1.  **Home / Introduction**: Read the introduction to understand the project's scope.
2.  **Define Portfolio**: Navigate to this section, enter 5-10 company stock tickers (one per line), and click "Save Portfolio".
3.  **ESG Agent Workflow**: Proceed to this section and click "Run ESG Assessment for Portfolio". Observe the progress as the AI agents conduct their analyses, including the Evaluator-Optimizer loop.
4.  **Portfolio ESG Scorecard**: Review the summary table of your portfolio's ESG performance.
5.  **Individual Company Profiles & Visualizations**: Select a company from the dropdown to dive into its detailed profile, scores, rationale, and agent's reasoning trace. Explore the radar chart and the portfolio controversy heatmap.
6.  **Score Consistency Analysis**: Choose a company and the number of runs to perform a consistency check. This helps understand the inherent variability of LLM-generated assessments.

## 5. Project Structure

The project directory is organized as follows:

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains helper functions, agent tools, and agent definitions
|                           # (e.g., get_environmental_metrics, evaluator_optimizer)
├── requirements.txt        # List of Python dependencies
├── .env                    # Environment variables for API keys (e.g., OPENAI_API_KEY)
└── README.md               # Project documentation (this file)
```

## 6. Technology Stack

This application is built using the following technologies:

*   **Python**: The core programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib / Seaborn / Plotly**: For data visualization (e.g., box plots, radar charts, heatmaps).
*   **LangChain**: A framework for developing applications powered by large language models, used for agent orchestration, tool integration, and the Evaluator-Optimizer pattern.
*   **Generative AI / Large Language Models (LLMs)**: The underlying intelligence performing the ESG assessments and analyses (e.g., OpenAI GPT models, Anthropic Claude, etc., depending on configuration in `source.py`).
*   **SASB Materiality Map**: A conceptual framework guiding the identification of financially material ESG topics for specific industries.

## 7. Contributing

We welcome contributions to enhance this project! If you'd like to contribute, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/fix-description`.
3.  **Make your changes**, ensuring they adhere to the project's coding style.
4.  **Write clear and concise commit messages**.
5.  **Push your branch** to your forked repository.
6.  **Open a Pull Request** against the `main` branch of the original repository, describing your changes in detail.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 9. Contact

For any questions, suggestions, or feedback, please feel free to reach out:

*   **Project Maintainer**: Quant University
*   **Email**: info@quantuniversity.com
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com)

---