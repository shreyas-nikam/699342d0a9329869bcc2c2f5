
# Streamlit Application Specification: Materiality-Driven ESG Portfolio Screening

## 1. Application Overview

This Streamlit application provides CFA Charterholders and Investment Professionals with a sophisticated tool for **materiality-driven ESG portfolio screening**. It automates and enhances the initial research phase, allowing analysts to quickly identify ESG risks and opportunities across a portfolio of companies, grounded in financially material topics.

The application guides the user through a structured workflow:

1.  **Introduction:** Sets the stage, explaining the importance of a materiality-driven approach to ESG for investment professionals.
2.  **Define Portfolio:** Users input a list of company tickers to form their investment portfolio.
3.  **ESG Agent Workflow:** The core of the application, where an AI agent, powered by Large Language Models (LLMs) and specialized data retrieval tools, performs a comprehensive ESG assessment for each company. This process integrates industry-specific SASB materiality, gathers relevant E, S, G data, and systematically scores each pillar. A critical "Evaluator-Optimizer" loop ensures the quality and consistency of these assessments.
4.  **Portfolio ESG Scorecard:** Presents a ranked table of all companies with their individual E, S, G scores and a materiality-weighted composite ESG score, enabling quick comparative analysis. This section also explains the underlying mathematical formula for weighting.
5.  **Individual Company Profiles & Visualizations:** Offers deep-dives into single company ESG profiles, including textual rationales, radar charts for E, S, G performance, and a portfolio-wide controversy heat map. It also visualizes the agent's reasoning process and evaluator feedback.
6.  **Score Consistency Analysis:** Addresses the inherent variability of LLM-generated scores by running multiple assessments for a single company and visualizing the score ranges, providing crucial insights into the reliability and interpretation of the results. This includes a crucial "Practitioner Warning" regarding LLM subjectivity in scoring.

This tool aims to streamline preliminary ESG research, enabling investment analysts to focus on higher-value tasks such as strategic decision-making and client engagement, while ensuring the underlying analysis is rigorous, consistent, and financially relevant.

## 2. Code Requirements

### 2.1. Import Statement

```python
import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from source import (
    get_environmental_metrics, # For internal use in some plotting, not directly called by UI logic for main flow
    scan_controversies, # Called by UI logic for controversy heatmap
    get_governance_data, # Not directly called by UI logic, but part of agent trace
    get_sasb_materiality, # Not directly called by UI logic, but part of agent trace
    get_peer_esg_scores, # Not directly called by UI logic, but part of agent trace
    determine_material_topics, # Called by UI logic for materiality explanation and weighting
    run_esg_agent, # Internal to evaluator_optimizer
    evaluator_optimizer, # Core function for running assessments
    categorize_material_topics, # Custom helper for materiality weighting
)
```

### 2.2. Streamlit Application `app.py` Structure

The application will use a sidebar for navigation and `st.session_state` to manage state across user interactions and "pages."

#### 2.2.1. `st.session_state` Initialization

The following `st.session_state` keys must be initialized at the beginning of `app.py`:

*   `st.session_state.current_page`: (`str`) Stores the currently selected page for navigation. Initialized to `'Home / Introduction'`.
*   `st.session_state.portfolio_tickers`: (`list` of `str`) Stores the list of company tickers entered by the user. Initialized to `[]`.
*   `st.session_state.portfolio_assessments`: (`dict`) Stores the results of `evaluator_optimizer` for each ticker after parsing and adding calculated weights. Key: ticker (`str`), Value: dictionary containing the parsed assessment, trace, evaluator status, revisions taken, and calculated weights/composite score. Initialized to `{}`.
*   `st.session_state.consistency_scores_df`: (`pandas.DataFrame` or `None`) Stores the DataFrame of scores from multiple consistency runs for a selected company. Initialized to `None`.
*   `st.session_state.consistency_company`: (`str` or `None`) Stores the ticker selected by the user for consistency analysis. Initialized to `None`.
*   `st.session_state.consistency_num_runs`: (`int`) Stores the number of runs for the consistency analysis. Initialized to `5`.
*   `st.session_state.selected_company_profile`: (`str` or `None`) Stores the ticker of the company currently selected for detailed profile viewing. Initialized to `None`.

```python
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home / Introduction'
if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []
if 'portfolio_assessments' not in st.session_state:
    st.session_state.portfolio_assessments = {} # {ticker: {assessment_json, trace_log, evaluator_status, revisions, parsed_scores_with_weights}}
if 'consistency_scores_df' not in st.session_state:
    st.session_state.consistency_scores_df = None
if 'consistency_company' not in st.session_state:
    st.session_state.consistency_company = None
if 'consistency_num_runs' not in st.session_state:
    st.session_state.consistency_num_runs = 5
if 'selected_company_profile' not in st.session_state:
    st.session_state.selected_company_profile = None
```

#### 2.2.3. Sidebar Navigation

A `st.selectbox` in the sidebar will control the application's current view.

```python
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        'Home / Introduction',
        'Define Portfolio',
        'ESG Agent Workflow',
        'Portfolio ESG Scorecard',
        'Individual Company Profiles & Visualizations',
        'Score Consistency Analysis'
    ],
    key='current_page' # This widget updates st.session_state.current_page
)
```

#### 2.2.4. Page-Specific Implementations and Function Invocations

The following sections define the content and logic for each "page" based on `st.session_state.current_page`.

##### A. Home / Introduction

*   **Description:** Displays the application's purpose and workflow introduction.
*   **Markdown:**
    ```python
    if st.session_state.current_page == 'Home / Introduction':
        st.title("Materiality-Driven ESG Portfolio Screening")
        st.markdown(f"## Introduction: An Investment Analyst's Edge with Materiality-Driven ESG")
        st.markdown(f"""
        As a CFA Charterholder and Investment Professional at a leading asset management firm, your role goes beyond just crunching numbers; it's about identifying long-term value and mitigating risks. Environmental, Social, and Governance (ESG) factors are increasingly critical to this mission. However, a "check-the-box" approach to ESG can be inefficient and misleading, failing to pinpoint what truly matters financially for each company.

        This application walks you through a real-world workflow to conduct a **materiality-driven ESG screening** for a portfolio of companies. You will leverage the power of Generative AI agents to:

        *   **Automatically identify** the most financially material ESG topics for each company's industry, guided by the **SASB Materiality Map**.
        *   **Gather relevant data** across E, S, and G pillars using specialized "tools".
        *   **Systematically score** companies against a structured rubric.
        *   **Employ an "Evaluator-Optimizer" loop** to ensure the quality and consistency of the AI's ESG assessments, mimicking a senior analyst's review process.
        *   **Generate a comprehensive ESG scorecard** and detailed profiles for your portfolio, enabling better risk identification and more informed capital allocation decisions.

        This hands-on lab will show you how to streamline preliminary research, ensuring that your ESG analysis is not only efficient but also financially relevant, reflecting the nuanced impacts of ESG issues across diverse industries.
        """)
    ```

##### B. Define Portfolio

*   **Description:** Allows users to input company tickers.
*   **Widgets:** `st.text_area` for ticker input, `st.button` to save.
*   **`st.session_state` updates:** `st.session_state.portfolio_tickers` is updated with the validated list. `st.session_state.portfolio_assessments` is cleared.
*   **Markdown:**
    ```python
    if st.session_state.current_page == 'Define Portfolio':
        st.title("Define Your Investment Portfolio")
        st.markdown(f"""
        To begin your materiality-driven ESG screening, please enter a list of 5-10 company stock tickers.
        These tickers will define your investment portfolio for analysis.
        """)
        
        default_tickers_str = "AAPL\nMSFT\nXOM\nJPM\nJNJ"
        ticker_input_value = "\n".join(st.session_state.portfolio_tickers) if st.session_state.portfolio_tickers else default_tickers_str
        ticker_input = st.text_area(
            "Enter company tickers (one per line, e.g., AAPL, MSFT, XOM, JPM, JNJ)",
            value=ticker_input_value,
            height=200
        )

        if st.button("Save Portfolio"):
            new_tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
            if 5 <= len(new_tickers) <= 10:
                st.session_state.portfolio_tickers = sorted(list(set(new_tickers))) # Remove duplicates and sort
                st.success(f"Portfolio saved with {len(st.session_state.portfolio_tickers)} companies: {', '.join(st.session_state.portfolio_tickers)}")
                # Clear previous assessments if portfolio changes
                st.session_state.portfolio_assessments = {}
                st.session_state.consistency_scores_df = None # Clear consistency data
            else:
                st.error("Please enter between 5 and 10 unique company tickers.")
        
        if st.session_state.portfolio_tickers:
            st.markdown(f"**Current Portfolio:** {', '.join(st.session_state.portfolio_tickers)}")
    ```

##### C. ESG Agent Workflow

*   **Description:** Orchestrates the ESG assessment for the entire portfolio using the `evaluator_optimizer` function.
*   **Widgets:** `st.button` to trigger the workflow, `st.progress` and `st.spinner` for feedback.
*   **Function Calls:** `evaluator_optimizer(ticker)` for each ticker in `st.session_state.portfolio_tickers`. `determine_material_topics(ticker)` and `categorize_material_topics(material_topics)` are called to calculate materiality weights and the composite score.
*   **`st.session_state` updates:** `st.session_state.portfolio_assessments` is populated with processed results, including parsed JSON assessments, evaluator status, revision count, trace log, and calculated materiality-weighted composite scores.
*   **Markdown:**
    ```python
    if st.session_state.current_page == 'ESG Agent Workflow':
        st.title("ESG Agent Workflow: Materiality-Driven Assessment")
        st.markdown(f"""
        Here, the AI agent performs a comprehensive ESG assessment for each company in your portfolio.
        It leverages specialized tools to gather data, applies SASB materiality mapping, scores
        against a structured rubric, and employs an "Evaluator-Optimizer" loop for quality assurance.
        """)

        if not st.session_state.portfolio_tickers:
            st.warning("Please define your portfolio in the 'Define Portfolio' section first.")
        else:
            st.markdown(f"**Companies in Portfolio:** {', '.join(st.session_state.portfolio_tickers)}")
            
            if st.button("Run ESG Assessment for Portfolio"):
                st.session_state.portfolio_assessments = {} # Reset assessments for a new run
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, ticker in enumerate(st.session_state.portfolio_tickers):
                    status_text.text(f"Processing {ticker} ({i+1}/{len(st.session_state.portfolio_tickers)})...")
                    with st.spinner(f"Running Evaluator-Optimizer for {ticker}..."):
                        try:
                            result = evaluator_optimizer(ticker)
                            
                            parsed_assessment = {}
                            if "Error" not in result['assessment'] and "Max iterations reached" not in result['assessment']:
                                try:
                                    json_match = re.search(r'```json\n({.*?})\n```', result['assessment'], re.DOTALL)
                                    if json_match:
                                        parsed_assessment = json.loads(json_match.group(1))
                                    else:
                                        parsed_assessment = json.loads(result['assessment'])
                                    
                                    # Recalculate materiality-weighted composite score using source.py functions
                                    e_score = parsed_assessment.get('environmental_score', 0)
                                    s_score = parsed_assessment.get('social_score', 0)
                                    g_score = parsed_assessment.get('governance_score', 0)

                                    materiality_info = determine_material_topics(ticker)
                                    topic_counts = categorize_material_topics(materiality_info['material_topics'])

                                    total_topics = sum(topic_counts.values())
                                    if total_topics > 0:
                                        w_e = topic_counts['E'] / total_topics
                                        w_s = topic_counts['S'] / total_topics
                                        w_g = topic_counts['G'] / total_topics
                                    else: # Default to equal weighting if no material topics found
                                        w_e, w_s, w_g = 1/3, 1/3, 1/3

                                    parsed_assessment['w_e'] = round(w_e, 2)
                                    parsed_assessment['w_s'] = round(w_s, 2)
                                    parsed_assessment['w_g'] = round(w_g, 2)
                                    parsed_assessment['composite_score_materiality_weighted'] = round(w_e * e_score + w_s * s_score + w_g * g_score, 2)
                                    
                                    # Add evaluator status and revisions to the parsed assessment
                                    parsed_assessment['evaluator_status'] = result['evaluator_status']
                                    parsed_assessment['revisions_taken'] = result['revisions']
                                    parsed_assessment['trace_log'] = result['trace'] # Store trace for agent reasoning
                                    
                                except json.JSONDecodeError as e:
                                    st.error(f"Error parsing JSON for {ticker}: {e}")
                                    parsed_assessment = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0, 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Error in assessment parsing', 'evaluator_status': 'FAILED_PARSE', 'revisions_taken': 0, 'trace_log': result['trace']}
                                except Exception as e:
                                    st.error(f"Error processing assessment for {ticker}: {e}")
                                    parsed_assessment = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0, 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Error in assessment processing', 'evaluator_status': 'FAILED_PROCESS', 'revisions_taken': 0, 'trace_log': result['trace']}
                            else:
                                st.warning(f"Assessment for {ticker} failed or timed out. Status: {result['evaluator_status']}. Assessment content: {result['assessment'][:200]}...")
                                parsed_assessment = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0, 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Agent failed/timed out', 'evaluator_status': result['evaluator_status'], 'revisions_taken': result['revisions'], 'trace_log': result['trace']}
                                
                            st.session_state.portfolio_assessments[ticker] = parsed_assessment
                        except Exception as e:
                            st.error(f"Overall error running ESG agent for {ticker}: {e}")
                            st.session_state.portfolio_assessments[ticker] = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0, 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Overall Agent Error', 'evaluator_status': 'OVERALL_ERROR', 'revisions_taken': 0, 'trace_log': []}

                    progress_bar.progress((i + 1) / len(st.session_state.portfolio_tickers))
                status_text.success("ESG assessments complete for the portfolio!")
                
            if st.session_state.portfolio_assessments:
                st.subheader("Assessment Summary")
                summary_data = []
                for ticker, data in st.session_state.portfolio_assessments.items():
                    summary_data.append({
                        'Ticker': ticker,
                        'Industry': data.get('industry', 'N/A'),
                        'E Score': data.get('environmental_score', 0),
                        'S Score': data.get('social_score', 0),
                        'G Score': data.get('governance_score', 0),
                        'Composite (Weighted)': data.get('composite_score_materiality_weighted', 0),
                        'Status': data.get('evaluator_status', 'N/A'),
                        'Revisions': data.get('revisions_taken', 'N/A'),
                        'Recommendation': data.get('recommendation', 'N/A')
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.set_index('Ticker').sort_values('Composite (Weighted)', ascending=False))
    ```

##### D. Portfolio ESG Scorecard

*   **Description:** Displays a ranked table of portfolio ESG scores, including a detailed explanation of the materiality-weighted composite score formula.
*   **Widgets:** `st.dataframe` to display the scorecard.
*   **`st.session_state` reads:** `st.session_state.portfolio_assessments` is used to construct the scorecard.
*   **Markdown & Formulas:**
    ```python
    if st.session_state.current_page == 'Portfolio ESG Scorecard':
        st.title("Portfolio ESG Scorecard")
        st.markdown(f"""
        This scorecard provides a ranked overview of your portfolio companies based on their
        materiality-weighted ESG performance.
        """)

        if not st.session_state.portfolio_assessments:
            st.warning("Please run the ESG Agent Workflow first to generate portfolio assessments.")
        else:
            scorecard_data = []
            for ticker, data in st.session_state.portfolio_assessments.items():
                scorecard_data.append({
                    'Ticker': ticker,
                    'Company': data.get('company', ticker),
                    'Industry': data.get('industry', 'N/A'),
                    'E Score': data.get('environmental_score', 0),
                    'S Score': data.get('social_score', 0),
                    'G Score': data.get('governance_score', 0),
                    'W_E': data.get('w_e', 0),
                    'W_S': data.get('w_s', 0),
                    'W_G': data.get('w_g', 0),
                    'Composite Score (Weighted)': data.get('composite_score_materiality_weighted', 0),
                    'Recommendation': data.get('recommendation', 'N/A'),
                    'Status': data.get('evaluator_status', 'N/A')
                })
            
            scorecard_df = pd.DataFrame(scorecard_data)
            st.dataframe(scorecard_df.sort_values('Composite Score (Weighted)', ascending=False).set_index('Ticker'))

            st.markdown(f"## Materiality-Weighted Composite ESG Score Formula")
            st.markdown(r"""
            The materiality-weighted composite ESG score ($S_{{\text{{composite}}}}$) is calculated as:
            """)
            st.markdown(r"$$ S_{{\text{{composite}}}} = w_E \cdot S_E + w_S \cdot S_S + w_G \cdot S_G $$")
            st.markdown(r"""
            where $S_E, S_S, S_G$ are the environmental, social, and governance scores, respectively.
            """)
            st.markdown(r"""
            The weights $w_E, w_S, w_G$ are determined by the count of material topics for each pillar, relative to the total number of material topics for that industry.
            """)
            st.markdown(r"""
            For example, if an oil company has 3 material E topics, 1 S topic, and 2 G topics, out of a total of $3+1+2=6$ material topics, the weights would be:
            """)
            st.markdown(r"$$ w_E = \frac{{3}}{{6}} = 0.50 $$")
            st.markdown(r"$$ w_S = \frac{{1}}{{6}} \approx 0.17 $$")
            st.markdown(r"$$ w_G = \frac{{2}}{{6}} \approx 0.33 $$")
            st.markdown(f"""
            This ensures that the composite score reflects the pillar most financially material to the company's industry
            (e.g., an oil company's composite is dominated by environmental performance; a bank's composite by governance and data security).
            """)
    ```

##### E. Individual Company Profiles & Visualizations

*   **Description:** Provides detailed insights for a selected company and portfolio-wide visual summaries.
*   **Widgets:** `st.selectbox` for company selection, `st.plotly_chart` for radar chart, `st.pyplot` for controversy heatmap.
*   **Function Calls:** `determine_material_topics(ticker)` for SASB explanation, `scan_controversies(ticker)` for heatmap data.
*   **`st.session_state` reads/updates:** `st.session_state.portfolio_assessments` is read. `st.session_state.selected_company_profile` is updated by the selectbox.
*   **Markdown:**
    ```python
    if st.session_state.current_page == 'Individual Company Profiles & Visualizations':
        st.title("Individual Company Profiles & Portfolio Visualizations")
        st.markdown(f"""
        Explore detailed ESG profiles for individual companies and visualize key ESG insights across your portfolio.
        """)

        if not st.session_state.portfolio_assessments:
            st.warning("Please run the ESG Agent Workflow first to generate portfolio assessments.")
        else:
            tickers = sorted(list(st.session_state.portfolio_assessments.keys()))
            default_index = 0
            if st.session_state.selected_company_profile and st.session_state.selected_company_profile in tickers:
                default_index = tickers.index(st.session_state.selected_company_profile)

            st.session_state.selected_company_profile = st.selectbox(
                "Select a company to view its detailed ESG profile:",
                options=tickers,
                index=default_index,
                key='company_profile_selector'
            )

            if st.session_state.selected_company_profile:
                selected_assessment = st.session_state.portfolio_assessments[st.session_state.selected_company_profile]
                
                st.subheader(f"ESG Profile for {selected_assessment.get('company', 'N/A')} ({selected_assessment.get('ticker', 'N/A')})")
                st.markdown(f"**Industry:** {selected_assessment.get('industry', 'N/A')}")
                st.markdown(f"**SASB Material Topics:** {', '.join(selected_assessment.get('sasb_material_topics', []))}")
                
                st.markdown(f"**Environmental Score (Weighted {selected_assessment.get('w_e',0)}):** {selected_assessment.get('environmental_score', 'N/A')}")
                st.markdown(f"  *Rationale:* {selected_assessment.get('environmental_rationale', 'N/A')}")
                
                st.markdown(f"**Social Score (Weighted {selected_assessment.get('w_s',0)}):** {selected_assessment.get('social_score', 'N/A')}")
                st.markdown(f"  *Rationale:* {selected_assessment.get('social_rationale', 'N/A')}")
                
                st.markdown(f"**Governance Score (Weighted {selected_assessment.get('w_g',0)}):** {selected_assessment.get('governance_score', 'N/A')}")
                st.markdown(f"  *Rationale:* {selected_assessment.get('governance_rationale', 'N/A')}")
                
                st.markdown(f"**Composite Score (Materiality-Weighted):** {selected_assessment.get('composite_score_materiality_weighted', 'N/A')}")
                st.markdown(f"**Controversies:** {selected_assessment.get('controversies_summary', 'None')}")
                st.markdown(f"**Peer Comparison:** {selected_assessment.get('peer_comparison', 'N/A')}")
                st.markdown(f"**Key Risks:** {', '.join(selected_assessment.get('key_risks', [])) if selected_assessment.get('key_risks') else 'None'}")
                st.markdown(f"**Recommendation:** {selected_assessment.get('recommendation', 'N/A')}")
                st.markdown(f"**Evaluation Status:** {selected_assessment.get('evaluator_status', 'N/A')} (Revisions: {selected_assessment.get('revisions_taken', 'N/A')})")

                st.subheader("ESG Pillar Radar Chart")
                categories = ['Environmental', 'Social', 'Governance']
                scores = [selected_assessment.get('environmental_score', 0), selected_assessment.get('social_score', 0), selected_assessment.get('governance_score', 0)]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                      r=scores + [scores[0]], # Close the loop
                      theta=categories + [categories[0]], # Close the loop
                      fill='toself',
                      name=selected_assessment['ticker']
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis_tickfont_size=10,
                        radialaxis=dict(
                            range=[0, 100], # Scores are 0-100
                            visible=True,
                            autorange=False
                        )),
                    showlegend=True,
                    title=f'ESG Pillar Radar Chart for {selected_assessment["ticker"]}',
                    height=400, width=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.subheader("Agent Reasoning Trace & Evaluator Feedback")
                st.markdown(f"This trace logs the AI agent's thought process, tool calls, and observations during the assessment, along with the evaluator's feedback and revision requests.")
                if selected_assessment.get('trace_log'):
                    for item in selected_assessment['trace_log']:
                        if 'action' in item:
                            st.markdown(f"**Agent Action (Iteration {item.get('iteration')}):** `{item['action']}`")
                            st.markdown(f"*Tool Result:* `{item.get('result', 'N/A')}`")
                        if 'evaluator_action' in item:
                            st.markdown(f"**Evaluator Action (Revision {item.get('revision_num') + 1}):** Status: `{item['status']}`, Feedback: `{item.get('feedback', 'No specific feedback provided.')}`")
                else:
                    st.info("No detailed agent trace available for this company.")

                st.subheader("SASB Materiality Router Explanation")
                st.markdown(f"""
                The agent dynamically identifies and prioritizes material ESG topics based on the company's industry,
                as guided by the SASB Materiality Map.
                """)
                materiality_info = determine_material_topics(selected_assessment.get('ticker'))
                if materiality_info:
                    st.markdown(f"For **{materiality_info['ticker']}** in the **{materiality_info['industry']}** industry, the material topics are:")
                    st.markdown(f"- **{', '.join(materiality_info['material_topics'])}**")
                    st.markdown(f"""
                    This intelligent routing ensures the ESG analysis focuses on what is most financially relevant,
                    avoiding a "one-size-fits-all" approach.
                    """)
                else:
                    st.info("Could not retrieve SASB Materiality information for this company.")

            st.subheader("Portfolio Controversy Heat Map")
            st.markdown(f"""
            This heat map provides a quick visual overview of ESG controversies across your entire portfolio,
            categorized by type and severity.
            """)
            
            if st.session_state.portfolio_assessments:
                controversy_data = []
                all_types = set()
                for item_ticker in st.session_state.portfolio_assessments.keys():
                    controversies_raw = scan_controversies(item_ticker) # Call source.py tool
                    controversies_list = json.loads(controversies_raw)
                    
                    for c in controversies_list:
                        c_type = c['type']
                        severity = c['severity']
                        all_types.add(c_type)
                        controversy_data.append({'ticker': item_ticker, 'type': c_type, 'severity': severity})

                if controversy_data:
                    controversy_df = pd.DataFrame(controversy_data)
                    severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
                    controversy_df['severity_num'] = controversy_df['severity'].map(severity_map)
                    
                    heatmap_data = controversy_df.pivot_table(index='ticker', columns='type', values='severity_num', fill_value=0)
                    
                    # Ensure all_types are in columns, add if missing for consistent plotting
                    for c_type in all_types:
                        if c_type not in heatmap_data.columns:
                            heatmap_data[c_type] = 0

                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, len(st.session_state.portfolio_assessments) * 0.8))
                    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='g', linewidths=.5, cbar_kws={'label': 'Controversy Severity (1=Low, 3=High)'}, ax=ax_heatmap)
                    ax_heatmap.set_title('Portfolio Controversy Heat Map (Severity by Type)')
                    ax_heatmap.set_xlabel('Controversy Type')
                    ax_heatmap.set_ylabel('Company Ticker')
                    plt.tight_layout()
                    st.pyplot(fig_heatmap)
                else:
                    st.info("No controversies detected across the portfolio for heatmap generation.")
    ```

##### F. Score Consistency Analysis

*   **Description:** Allows users to run multiple assessments for a single company and visualize the score variability. Includes a "Practitioner Warning" on LLM subjectivity.
*   **Widgets:** `st.selectbox` for company selection, `st.number_input` for number of runs, `st.button` to trigger consistency check, `st.dataframe` for results, `st.pyplot` for box plot.
*   **Function Calls:** `evaluator_optimizer(company, max_revisions=3)` is called multiple times. `determine_material_topics(company)` and `categorize_material_topics(material_topics)` are called for weighting.
*   **`st.session_state` reads/updates:** `st.session_state.consistency_company`, `st.session_state.consistency_num_runs` are read/updated. `st.session_state.consistency_scores_df` is populated.
*   **Markdown:**
    ```python
    if st.session_state.current_page == 'Score Consistency Analysis':
        st.title("ESG Score Consistency and Variability Analysis")
        st.markdown(f"""
        As a seasoned investment professional, you understand that even with clear rubrics and structured processes,
        assessments involving qualitative judgment (like ESG) can exhibit some variability, especially when using LLMs.
        This section analyzes the consistency of the generated scores by running the ESG assessment for a single company multiple times.
        """)

        if not st.session_state.portfolio_assessments:
            st.warning("Please run the ESG Agent Workflow first to generate portfolio assessments.")
        else:
            tickers = sorted(list(st.session_state.portfolio_assessments.keys()))
            default_consistency_company_index = 0
            if st.session_state.consistency_company and st.session_state.consistency_company in tickers:
                default_consistency_company_index = tickers.index(st.session_state.consistency_company)
            elif tickers:
                st.session_state.consistency_company = tickers[0] # Set default if not already set or invalid
            
            st.session_state.consistency_company = st.selectbox(
                "Select a company for consistency check:",
                options=tickers,
                index=default_consistency_company_index,
                key='consistency_company_selector'
            )
            
            st.session_state.consistency_num_runs = st.number_input(
                "Number of runs for consistency check:",
                min_value=2, max_value=10, value=st.session_state.consistency_num_runs, step=1,
                key='num_runs_input'
            )

            if st.button("Run Consistency Check"):
                if st.session_state.consistency_company:
                    consistency_scores = []
                    progress_bar_consistency = st.progress(0)
                    status_text_consistency = st.empty()

                    for i in range(st.session_state.consistency_num_runs):
                        status_text_consistency.text(f"Consistency Run {i+1}/{st.session_state.consistency_num_runs} for {st.session_state.consistency_company}...")
                        with st.spinner(f"Running Evaluator-Optimizer for {st.session_state.consistency_company} (Run {i+1})..."):
                            result = evaluator_optimizer(st.session_state.consistency_company, max_revisions=3)
                            
                            if result['evaluator_status'] == 'APPROVED' or result['evaluator_status'] == 'MAX_REVISIONS_REACHED':
                                try:
                                    assessment_json_str = result['assessment']
                                    json_match = re.search(r'```json\n({.*?})\n```', assessment_json_str, re.DOTALL)
                                    if json_match:
                                        parsed_assessment = json.loads(json_match.group(1))
                                    else:
                                        parsed_assessment = json.loads(assessment_json_str)

                                    # Recalculate materiality-weighted composite score using source.py functions
                                    e_score = parsed_assessment.get('environmental_score', 0)
                                    s_score = parsed_assessment.get('social_score', 0)
                                    g_score = parsed_assessment.get('governance_score', 0)

                                    materiality_info = determine_material_topics(st.session_state.consistency_company)
                                    topic_counts = categorize_material_topics(materiality_info['material_topics'])
                                    total_topics = sum(topic_counts.values())
                                    if total_topics > 0:
                                        w_e = topic_counts['E'] / total_topics
                                        w_s = topic_counts['S'] / total_topics
                                        w_g = topic_counts['G'] / total_topics
                                    else:
                                        w_e, w_s, w_g = 1/3, 1/3, 1/3
                                    
                                    composite_weighted = round(w_e * e_score + w_s * s_score + w_g * g_score, 2)

                                    consistency_scores.append({
                                        'Run': i + 1,
                                        'E': parsed_assessment.get('environmental_score', 0),
                                        'S': parsed_assessment.get('social_score', 0),
                                        'G': parsed_assessment.get('governance_score', 0),
                                        'Composite_Weighted': composite_weighted
                                    })
                                except json.JSONDecodeError as e:
                                    st.error(f"Error parsing JSON in consistency run {i+1}: {e}")
                                    st.text(f"Raw content: {result['assessment'][:200]}...")
                                except Exception as e:
                                    st.error(f"General error in consistency run {i+1}: {e}")
                            else:
                                st.warning(f"Assessment not approved or failed for run {i+1}. Status: {result['evaluator_status']}. Skipping this run's data.")
                        progress_bar_consistency.progress((i + 1) / st.session_state.consistency_num_runs)
                    status_text_consistency.success("Consistency check complete!")
                    
                    if consistency_scores:
                        st.session_state.consistency_scores_df = pd.DataFrame(consistency_scores)
                    else:
                        st.session_state.consistency_scores_df = None
                else:
                    st.warning("Please select a company to run the consistency check.")

            if st.session_state.consistency_scores_df is not None:
                st.subheader(f"Score Consistency for {st.session_state.consistency_company} ({st.session_state.consistency_num_runs} Runs)")
                st.dataframe(st.session_state.consistency_scores_df.set_index('Run'))

                st.subheader("Score Ranges")
                score_columns = ['E', 'S', 'G', 'Composite_Weighted']
                for col in score_columns:
                    if col in st.session_state.consistency_scores_df.columns:
                        score_range = st.session_state.consistency_scores_df[col].max() - st.session_state.consistency_scores_df[col].min()
                        st.markdown(f"**{col} Range:** {st.session_state.consistency_scores_df[col].min():.1f} - {st.session_state.consistency_scores_df[col].max():.1f} = {score_range:.1f}")
                st.markdown(f"(Range > 10 typically indicates significant score instability for a single input)")

                st.subheader("ESG Score Consistency Box Plot")
                fig_boxplot, ax_boxplot = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=st.session_state.consistency_scores_df[score_columns], palette='viridis', ax=ax_boxplot)
                ax_boxplot.set_title(f'ESG Score Consistency Across {st.session_state.consistency_num_runs} Runs for {st.session_state.consistency_company}')
                ax_boxplot.set_ylabel('Score (0-100)')
                ax_boxplot.set_xlabel('ESG Pillar / Composite Score')
                ax_boxplot.set_ylim(0, 100)
                ax_boxplot.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig_boxplot)

                st.markdown(f"## Practitioner Warning")
                st.markdown(f"""
                Agent-generated ESG scores are inherently subjective and variable. Even with the same data and rubric,
                running the agent multiple times on the same company may produce scores varying by 5-15 points (out of 100)
                due to LLM stochasticity and interpretation differences. This is not a bug—it reflects the genuine ambiguity in ESG assessment.
                But it means: (a) scores should be used for ranking (relative comparison) rather than absolute assessment,
                (b) scores near boundaries (e.g., 59 vs. 61 for "Adequate" vs. "Strong") should be treated as uncertain,
                and (c) the rationale is more important than the number—the analyst should read the justification, not just the score.
                ESG rating agencies face the same challenge: MSCI and Sustainalytics often disagree significantly on the same company.
                The agent's variability is no worse than inter-rater disagreement among professional ESG analysts.
                """)
            else:
                st.info("No consistency data available. Run the consistency check above.")
    ```
