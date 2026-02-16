
# Materiality-Driven ESG Portfolio Screening with AI Agents

## Introduction: An Investment Analyst's Edge with Materiality-Driven ESG

As a CFA Charterholder and Investment Professional at a leading asset management firm, your role goes beyond just crunching numbers; it's about identifying long-term value and mitigating risks. Environmental, Social, and Governance (ESG) factors are increasingly critical to this mission. However, a "check-the-box" approach to ESG can be inefficient and misleading, failing to pinpoint what truly matters financially for each company.

This notebook walks you through a real-world workflow to conduct a **materiality-driven ESG screening** for a portfolio of companies. You will leverage the power of Generative AI agents to:

*   **Automatically identify** the most financially material ESG topics for each company's industry, guided by the **SASB Materiality Map**.
*   **Gather relevant data** across E, S, and G pillars using specialized "tools".
*   **Systematically score** companies against a structured rubric.
*   **Employ an "Evaluator-Optimizer" loop** to ensure the quality and consistency of the AI's ESG assessments, mimicking a senior analyst's review process.
*   **Generate a comprehensive ESG scorecard** and detailed profiles for your portfolio, enabling better risk identification and more informed capital allocation decisions.

This hands-on lab will show you how to streamline preliminary research, ensuring that your ESG analysis is not only efficient but also financially relevant, reflecting the nuanced impacts of ESG issues across diverse industries.

## 1. Environment Setup: Installing Libraries and Importing Dependencies

Before we dive into the materiality-driven ESG screening, we need to set up our Python environment by installing the necessary libraries and importing them. These libraries will enable us to interact with Large Language Models (LLMs), manipulate data, retrieve financial information, and create visualizations.

```python
!pip install openai==1.14.0 langchain==0.1.11 pandas==2.2.1 yfinance==0.2.36 matplotlib==3.8.3 seaborn==0.13.2 plotly==5.19.0 # Specific versions for reproducibility
```

```python
import os
import json
import pandas as pd
import yfinance as yf
from langchain.agents import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re # For parsing JSON from markdown
import numpy as np # For statistical analysis in consistency check

# Set your OpenAI API key
# Ensure you have your OpenAI API key set as an environment variable or replace 'YOUR_OPENAI_API_KEY'
# For example: os.environ["OPENAI_API_KEY"] = "sk-..." 
# For demonstration, we'll use a placeholder. In a real scenario, use environment variables.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual key or set env var
    # Note: Using a placeholder will prevent actual LLM calls from working.
    # Ensure a valid API key is configured for the notebook to run successfully with LLMs.
```

## 2. Defining the ESG Assessment Tools: Data Retrieval for Our Agent

As an investment analyst, you need access to diverse, structured data to perform a robust ESG assessment. Instead of manually sifting through reports, you'll define a suite of specialized "tools" that our AI agent can use. These tools simulate data retrieval from various sources, covering environmental metrics, social controversies, governance structures, SASB materiality insights, and peer performance. This approach streamlines data collection, allowing the agent to focus on analysis rather than just fetching information.

```python
# --- Tool 1: Environmental Metrics ---
@tool
def get_environmental_metrics(ticker: str) -> str:
    """Retrieve simulated environmental data (carbon emissions, energy usage, water usage, carbon neutral targets) for a given company ticker."""
    env_data = {
        'AAPL': {'scope1_emissions_tco2': 22400, 'scope2_emissions_tco2': 0,
                 'renewable_energy_pct': 100, 'water_usage_megaliters': 1200,
                 'carbon_neutral_target': '2030 (Scope 3)'},
        'MSFT': {'scope1_emissions_tco2': 10000, 'scope2_emissions_tco2': 1500,
                 'renewable_energy_pct': 95, 'water_usage_megaliters': 800,
                 'carbon_neutral_target': '2030 (Scope 3)'},
        'XOM': {'scope1_emissions_tco2': 112_000_000, 'scope2_emissions_tco2': 15_000_000,
                'renewable_energy_pct': 3, 'water_usage_megaliters': 450_000,
                'carbon_neutral_target': '2050 (Scope 1+2)'},
        'JPM': {'scope1_emissions_tco2': 5000, 'scope2_emissions_tco2': 1000,
                'renewable_energy_pct': 70, 'water_usage_megaliters': 50,
                'carbon_neutral_target': '2040 (Scope 3)'},
        'JNJ': {'scope1_emissions_tco2': 30000, 'scope2_emissions_tco2': 5000,
                'renewable_energy_pct': 60, 'water_usage_megaliters': 3000,
                'carbon_neutral_target': '2045 (Scope 3)'}
    }
    data = env_data.get(ticker, {'note': 'Environmental data not available for this ticker.'})
    return json.dumps(data, indent=2)

# --- Tool 2: Controversy Scanner ---
@tool
def scan_controversies(ticker: str) -> str:
    """Search for recent ESG controversies and incidents for a given company ticker."""
    controversies = {
        'AAPL': [
            {'type': 'Social', 'severity': 'Medium', 'description': 'Supply chain labor concerns in SE Asia', 'date': '2024-Q3'}
        ],
        'MSFT': [
            {'type': 'Governance', 'severity': 'Low', 'description': 'Minor data privacy breach', 'date': '2023-Q4'}
        ],
        'XOM': [
            {'type': 'Environmental', 'severity': 'High', 'description': 'Ongoing climate litigation (multiple cases)', 'date': '2024-ongoing'},
            {'type': 'Governance', 'severity': 'Medium', 'description': 'Shareholder proposal on climate targets rejected', 'date': '2024-Q2'}
        ],
        'JPM': [
            {'type': 'Social', 'severity': 'High', 'description': 'Regulatory fines for compliance failures', 'date': '2024-Q1'}
        ],
        'JNJ': [
            {'type': 'Social', 'severity': 'Medium', 'description': 'Product recall due to quality concerns', 'date': '2023-Q3'}
        ]
    }
    data = controversies.get(ticker, [])
    return json.dumps(data, indent=2)

# --- Tool 3: Governance Metrics ---
@tool
def get_governance_data(ticker: str) -> str:
    """Retrieve simulated corporate governance metrics for a given company ticker."""
    gov_data = {
        'AAPL': {'board_size': 8, 'pct_independent': 87.5, 'board_diversity_pct': 50,
                 'ceo_chair_separate': True, 'say_on_pay_approval': 94.5,
                 'clawback_policy': True, 'esg_in_compensation': False},
        'MSFT': {'board_size': 10, 'pct_independent': 90.0, 'board_diversity_pct': 60,
                 'ceo_chair_separate': True, 'say_on_pay_approval': 96.0,
                 'clawback_policy': True, 'esg_in_compensation': True},
        'XOM': {'board_size': 12, 'pct_independent': 91.7, 'board_diversity_pct': 33,
                'ceo_chair_separate': False, 'say_on_pay_approval': 88.2,
                'clawback_policy': True, 'esg_in_compensation': True},
        'JPM': {'board_size': 14, 'pct_independent': 85.7, 'board_diversity_pct': 45,
                'ceo_chair_separate': True, 'say_on_pay_approval': 92.0,
                'clawback_policy': True, 'esg_in_compensation': True},
        'JNJ': {'board_size': 11, 'pct_independent': 81.8, 'board_diversity_pct': 40,
                'ceo_chair_separate': True, 'say_on_pay_approval': 91.0,
                'clawback_policy': True, 'esg_in_compensation': True}
    }
    data = gov_data.get(ticker, {})
    return json.dumps(data, indent=2)

# --- Tool 4: SASB Materiality Lookup ---
@tool
def get_sasb_materiality(industry: str) -> str:
    """Get SASB material ESG topics for a given industry. This mapping guides our materiality-driven analysis."""
    sasb_map = {
        'Technology': ['Data Security', 'Employee Engagement', 'GHG Emissions', 'Energy Management', 'Supply Chain Management'],
        'Oil & Gas': ['GHG Emissions', 'Air Quality', 'Water Management', 'Ecological Impacts', 'Community Relations', 'Business Ethics'],
        'Financial Services': ['Data Security', 'Business Ethics', 'Systemic Risk', 'Employee Engagement', 'Customer Privacy'],
        'Healthcare': ['Product Quality & Safety', 'Access to Healthcare', 'GHG Emissions', 'Ethical Marketing Practices'],
        'Consumer Cyclical': ['Labor Practices', 'Supply Chain Management', 'Product Safety & Quality', 'Data Security']
    }
    topics = sasb_map.get(industry, ['General ESG topics apply'])
    return json.dumps({'industry': industry, 'material_topics': topics}, indent=2)

# --- Tool 5: Peer ESG Comparison ---
@tool
def get_peer_esg_scores(ticker: str) -> str:
    """Compare ESG metrics to sector peers for a given company ticker."""
    peer_data = {
        'AAPL': {'sector_avg_emissions_intensity': 45, 'company_emissions_intensity': 5,
                 'sector_avg_board_diversity': 35, 'company_board_diversity': 50,
                 'esg_rating_proxy': 'AA (top quartile)'},
        'MSFT': {'sector_avg_emissions_intensity': 30, 'company_emissions_intensity': 4,
                 'sector_avg_board_diversity': 40, 'company_board_diversity': 60,
                 'esg_rating_proxy': 'AAA (top 5%)'},
        'XOM': {'sector_avg_emissions_intensity': 200, 'company_emissions_intensity': 180,
                'sector_avg_board_diversity': 30, 'company_board_diversity': 33,
                'esg_rating_proxy': 'BBB (bottom quartile)'},
        'JPM': {'sector_avg_emissions_intensity': 10, 'company_emissions_intensity': 8,
                'sector_avg_board_diversity': 40, 'company_board_diversity': 45,
                'esg_rating_proxy': 'A (second quartile)'},
        'JNJ': {'sector_avg_emissions_intensity': 60, 'company_emissions_intensity': 55,
                'sector_avg_board_diversity': 38, 'company_board_diversity': 40,
                'esg_rating_proxy': 'A (second quartile)'}
    }
    data = peer_data.get(ticker, {'note': 'Peer comparison data not available.'})
    return json.dumps(data, indent=2)

# Combine all tools into a list for the LLM agent
TOOLS = [
    get_environmental_metrics,
    scan_controversies,
    get_governance_data,
    get_sasb_materiality,
    get_peer_esg_scores
]

# Format tools for OpenAI function calling
ESG_TOOL_SCHEMAS = [format_tool_to_openai_function(t) for t in TOOLS]

# Test a tool
print("Testing get_environmental_metrics for AAPL:")
print(get_environmental_metrics(ticker='AAPL'))
print("\nTesting scan_controversies for XOM:")
print(scan_controversies(ticker='XOM'))
```

### Explanation of Execution

The code above defines five Python functions, each acting as a specialized tool for our ESG agent. When the `get_environmental_metrics('AAPL')` tool is called, it returns a JSON string containing simulated environmental data for Apple. Similarly, `scan_controversies('XOM')` retrieves simulated controversy data for ExxonMobil. These functions represent the various data-gathering steps an investment analyst would perform. By centralizing these into tools, the AI agent can dynamically decide which data to fetch based on its internal reasoning, making the research process more efficient and adaptable. The `ESG_TOOL_SCHEMAS` list converts these Python functions into a format that OpenAI's function-calling capability can understand and utilize.

## 3. Automating Industry Materiality Identification with SASB Mapping

A core principle of our firm's ESG investment strategy is **materiality**. This means focusing on ESG issues that are most financially relevant to a company's specific industry. For an investment analyst, this prevents a "one-size-fits-all" approach, ensuring the analysis is targeted and impactful. In this step, you will implement a function that automatically determines a company's industry and then, using a predefined SASB Materiality Map, identifies the crucial ESG topics. This acts as a "routing layer" for the ESG agent, guiding its subsequent data collection and scoring.

```python
# Function to determine material topics
def determine_material_topics(ticker: str) -> dict:
    """
    Determines a company's industry using yfinance and then retrieves SASB material ESG topics
    based on that industry.
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch detailed info which often includes 'sector' or 'industry'
        info = stock.info
        industry = info.get('sector', info.get('industry', 'Unknown'))
        
        # Use the get_sasb_materiality tool
        sasb_result_json = get_sasb_materiality(industry)
        sasb_result = json.loads(sasb_result_json)
        
        material_topics = sasb_result['material_topics']
        
        return {'ticker': ticker, 'industry': industry, 'material_topics': material_topics}
    except Exception as e:
        return {'ticker': ticker, 'industry': 'Unknown', 'material_topics': ['Error: Could not retrieve SASB material topics.', str(e)]}

# Test with a few companies from different industries
portfolio_tickers = ['AAPL', 'XOM', 'JPM']

print("Identifying material ESG topics for our initial portfolio:")
for ticker in portfolio_tickers:
    materiality = determine_material_topics(ticker)
    print(f"\n{materiality['ticker']} ({materiality['industry']}): Material topics = {materiality['material_topics']}")

# Example of how materiality influences focus:
# For 'XOM' (Oil & Gas): The agent will focus on GHG Emissions, Air Quality, Water Management, etc.
# For 'AAPL' (Technology): The agent will prioritize Data Security, Employee Engagement, GHG Emissions, etc.
```

### Explanation of Execution

The `determine_material_topics` function first uses `yfinance` to fetch the sector/industry of a given ticker. For example, for 'AAPL', it would likely return 'Technology'. It then calls our `get_sasb_materiality` tool with this industry, which looks up the financially material ESG topics from our `sasb_map`. This output is crucial for the AI agent because it allows the agent to dynamically adjust its focus. Instead of checking every possible ESG factor for every company, it intelligently "routes" its attention to the issues that are most relevant to that specific company's financial performance and risk profile. This is analogous to how an experienced analyst wouldn't spend time evaluating water usage for a software company but would meticulously scrutinize data security.

## 4. Designing the ESG Scoring Agent with a Structured Rubric

To ensure consistency and comparability across companies, you, as the investment analyst, need a structured way to evaluate ESG performance. This section defines the core of our AI agent: its system prompt, which includes a detailed scoring rubric (0-100 for E, S, and G pillars) and specifies the desired JSON output format. This rubric encodes your firm's investment policy for ESG, turning qualitative assessment into a quantifiable, standardized metric. The `run_esg_agent` function will orchestrate the data gathering using our defined tools and generate an initial ESG assessment based on this rubric.

```python
# Define the ESG Agent's system prompt and scoring rubric
ESG_AGENT_SYSTEM = """
You are a senior ESG research analyst conducting sustainability assessments for an investment firm.
Your goal is to perform a comprehensive, materiality-driven ESG assessment for a given company.

PROCESS:
1. Determine the company's industry and SASB material ESG topics using the 'get_sasb_materiality' tool.
2. Gather environmental metrics (emissions, energy, targets) using the 'get_environmental_metrics' tool.
3. Scan for recent ESG controversies using the 'scan_controversies' tool.
4. Retrieve governance quality metrics using the 'get_governance_data' tool.
5. Compare to sector peers using the 'get_peer_esg_scores' tool.
6. Synthesize all collected data to score each ESG pillar (Environmental, Social, Governance) and calculate a materiality-weighted composite score.
7. Produce a structured assessment in JSON format, citing specific data points from the tools in your rationale.

SCORING RUBRIC (0-100 per pillar):

ENVIRONMENTAL (E):
- 80-100: Industry leader, net-zero achieved/imminent, no significant environmental incidents.
- 60-79: Strong targets, measurable progress, minor environmental incidents.
- 40-59: Targets set but limited progress, moderate concerns or historical minor incidents.
- 20-39: Minimal environmental management, significant issues, or past major incidents.
- 0-19: No targets, major environmental controversies or systemic environmental negligence.

SOCIAL (S):
- 80-100: Exemplary labor/community relations, no controversies, strong human rights policies.
- 60-79: Good practices, minor incidents resolved promptly, adequate social programs.
- 40-59: Mixed record, some unresolved social concerns, moderate labor issues.
- 20-39: Recurring labor/community issues, lawsuits pending, or significant social controversies.
- 0-19: Major social controversies (human rights violations, egregious labor practices).

GOVERNANCE (G):
- 80-100: Independent board >75%, separated CEO/Chair, ESG in compensation, robust policies.
- 60-79: Good independence, most best practices in place, clear accountability.
- 40-59: Adequate governance, some gaps (e.g., combined CEO/Chair, lower independence).
- 20-39: Weak oversight, significant governance concerns, lack of transparency or accountability.
- 0-19: Major governance failures, ongoing investigations, severe ethical breaches.

OUTPUT FORMAT (JSON):
```json
{
  "company": "...",
  "ticker": "...",
  "industry": "...",
  "sasb_material_topics": ["...", "..."],
  "environmental_score": X,
  "environmental_rationale": "...",
  "social_score": X,
  "social_rationale": "...",
  "governance_score": X,
  "governance_rationale": "...",
  "composite_score": X,
  "controversies_summary": "...",
  "peer_comparison": "...",
  "key_risks": ["..."],
  "recommendation": "Strong ESG / Adequate ESG / ESG Concern"
}
```
Use tools to gather ALL relevant data before scoring.
Cite specific data points in your rationale and reference the tool outputs.
"""

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2) # Using gpt-4o as specified

# Create a runnable agent with tool calling
def run_esg_agent(ticker: str, max_iterations: int = 15) -> dict:
    """
    Runs the ESG research agent with tool calling to perform an ESG assessment.
    Returns the final assessment, trace of actions, and number of iterations.
    """
    messages = [
        SystemMessage(content=ESG_AGENT_SYSTEM),
        HumanMessage(content=f"Conduct a comprehensive ESG assessment of {ticker}. Use all available tools, score each pillar, and produce the structured JSON output.")
    ]
    trace = []

    for iteration in range(max_iterations):
        try:
            response = llm.invoke(messages, tools=ESG_TOOL_SCHEMAS, tool_choice="auto")
        except Exception as e:
            trace.append({'error': f"LLM invocation failed: {e}", 'iteration': iteration})
            return {'assessment': f"Error: LLM invocation failed after {iteration} steps. {e}", 'trace': trace, 'iterations': iteration}

        msg = response
        messages.append(msg)
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                
                # Dynamically call the tool function from the TOOLS list
                tool_func = next((t for t in TOOLS if t.__name__ == tool_name), None)
                if tool_func:
                    result = tool_func(**tool_args)
                    trace.append({'action': f"{tool_name}({tool_args})", 'result': result[:300] + '...' if len(result) > 300 else result, 'iteration': iteration})
                    messages.append(ToolMessage(tool_call_id=tc.id, content=result))
                else:
                    error_msg = f"Tool '{tool_name}' not found."
                    trace.append({'action': f"{tool_name}({tool_args})", 'result': error_msg, 'iteration': iteration})
                    messages.append(ToolMessage(tool_call_id=tc.id, content=error_msg))
        else:
            # If no tool calls, it means the LLM is ready to produce the final answer
            # Attempt to extract JSON, even if it's within markdown
            json_match = re.search(r'```json\n({.*?})\n```', msg.content, re.DOTALL)
            if json_match:
                assessment_content = json_match.group(1)
            else:
                assessment_content = msg.content # Fallback if not proper markdown JSON

            return {'assessment': assessment_content, 'trace': trace, 'iterations': iteration + 1}
    
    return {'assessment': "Max iterations reached without generating a final JSON assessment.", 'trace': trace, 'iterations': max_iterations}


# Test the ESG agent for a single company
print("Running the ESG agent for Apple (AAPL)... This may take a moment.")
initial_assessment_aapl = run_esg_agent(ticker='AAPL')
print("\n--- Initial ESG Assessment for AAPL ---")
print(initial_assessment_aapl['assessment'])

# Attempt to parse the assessment to verify JSON structure
try:
    parsed_aapl_assessment = json.loads(initial_assessment_aapl['assessment'])
    print(f"\nEnvironmental Score: {parsed_aapl_assessment.get('environmental_score')}")
    print(f"Social Score: {parsed_aapl_assessment.get('social_score')}")
    print(f"Governance Score: {parsed_aapl_assessment.get('governance_score')}")
except json.JSONDecodeError as e:
    print(f"\nError parsing JSON from initial assessment: {e}")
    print("Raw assessment content for debugging:\n", initial_assessment_aapl['assessment'])

```

### Explanation of Execution

The `ESG_AGENT_SYSTEM` prompt acts as the analyst's guidelines for the AI. It outlines the step-by-step process, crucial scoring rubric for E, S, and G pillars (e.g., "80-100: Industry leader, net-zero achieved/imminent"), and the exact JSON output format required. The `run_esg_agent` function then orchestrates the LLM's interaction with the tools. For `AAPL`, the agent will first call `get_sasb_materiality('Technology')`, then proceed to fetch environmental, social, and governance data, and finally synthesize this information, guided by the rubric, to produce a scored assessment. This modular, rubric-driven approach ensures that the output is not just prose, but a structured, quantifiable evaluation that an investment analyst can readily use for comparative analysis.

## 5. Implementing the Evaluator-Optimizer Loop for Quality Assurance

As a diligent CFA Charterholder, you know that even advanced AI models can produce inconsistent or incomplete outputs. To ensure the highest quality and adherence to your firm's standards, you implement an "Evaluator-Optimizer" loop. This pattern mimics a senior analyst reviewing a junior's report: one LLM (the Generator) produces the initial assessment, and a second LLM (the Evaluator) reviews it for completeness, accuracy, and rubric adherence. If issues are found, the Evaluator provides feedback, and the Generator revises its output. This iterative self-correction significantly enhances the reliability of the ESG assessments.

The mathematical formulation for the Evaluator-Optimizer Pattern is as follows:

Let $G$ be the Generator (our ESG Agent) and $E$ be the Evaluator.
Given an input $x$ (company ticker), the Generator produces an initial output $y^{(0)} = G(x)$.
The Evaluator then assesses the quality of this output: $E(y^{(t)}) \in \{ \text{APPROVED}, \text{REVISE} \}$.

The loop continues iteratively:
$$ y^{(t+1)} = G(x, E(y^{(t)})) $$
This process continues until $E(y^{(t)}) = \text{APPROVED}$ or a maximum number of revisions, $T_{\text{max}}$, is reached.
This loop ensures that the ESG assessments meet predefined quality criteria before being finalized, much like a senior analyst would send back a report for revisions if it lacked data points or did not align with the investment policy.

```python
# Define the Evaluator's prompt
EVALUATOR_PROMPT = """
You are a senior ESG quality reviewer for an investment firm. Your task is to evaluate
the following ESG assessment for completeness, accuracy, and adherence to the rubric and JSON format.

ASSESSMENT TO REVIEW:
{assessment}

CHECKLIST:
1. Are all three pillars (E, S, G) scored with rationale?
2. Are scores consistent with the evidence cited?
3. Were SASB material topics addressed?
4. Are controversies mentioned with severity ratings?
5. Is peer comparison included?
6. Is the output valid JSON format?
7. Is the composite score calculated?

If ALL checks pass, respond with:
```json
{{"status": "APPROVED", "feedback": ""}}
```
If ANY check fails, respond with:
```json
{{"status": "REVISE", "feedback": "Specific feedback on what needs revision (e.g., 'Missing S-pillar score and rationale.', 'Environmental score of X is inconsistent with high emissions data.', 'Peer comparison is generic, needs company-specific comparison.')."}}
```
Your feedback should be concise and actionable for the ESG agent.
"""

# Initialize a separate LLM for the evaluator, potentially with lower temperature for stricter evaluation
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def evaluator_optimizer(ticker: str, max_revisions: int = 3) -> dict:
    """
    Runs the ESG agent and then uses an evaluator LLM to review and request revisions
    until approved or max_revisions are met.
    """
    print(f"\n--- Starting Evaluator-Optimizer for {ticker} ---")
    
    # Initial run of the ESG agent
    print(f"  > Running initial ESG agent for {ticker}...")
    result = run_esg_agent(ticker)
    
    if "Error" in result['assessment'] or "Max iterations reached" in result['assessment']:
        print(f"  > Initial agent run failed or timed out for {ticker}. Skipping evaluation.")
        return {'assessment': result['assessment'], 'evaluator_status': 'FAILED', 'revisions': 0}

    current_assessment = result['assessment']
    trace = result['trace']
    total_iterations = result['iterations']
    
    for revision_num in range(max_revisions):
        print(f"  > Revision {revision_num + 1}/{max_revisions}: Evaluating current assessment...")
        eval_messages = [
            SystemMessage(content=EVALUATOR_PROMPT.format(assessment=current_assessment))
        ]
        
        try:
            eval_response = evaluator_llm.invoke(eval_messages, response_format={"type": "json_object"})
            evaluation = eval_response.content
            if isinstance(evaluation, str): # Ensure evaluation is dict, not string
                evaluation = json.loads(evaluation)
        except json.JSONDecodeError as e:
            print(f"    ! Error parsing evaluator JSON response: {e}. Raw content: {eval_response.content}")
            evaluation = {"status": "REVISE", "feedback": f"Evaluator response malformed: {e}"}
        except Exception as e:
            print(f"    ! Evaluator LLM invocation failed: {e}")
            evaluation = {"status": "REVISE", "feedback": f"Evaluator LLM error: {e}"}


        trace.append({'evaluator_action': 'evaluate', 'feedback': evaluation.get('feedback', ''), 'status': evaluation.get('status', 'ERROR'), 'revision_num': revision_num})

        if evaluation.get('status') == 'APPROVED':
            print(f"  > Evaluator: APPROVED (revision {revision_num + 1}).")
            result['assessment'] = current_assessment
            result['evaluator_status'] = 'APPROVED'
            result['revisions'] = revision_num
            result['trace'] = trace
            result['iterations'] = total_iterations
            return result
        else:
            feedback = evaluation.get('feedback', 'No specific feedback provided.')
            print(f"  > Evaluator: REVISE. Feedback: {feedback[:100]}...")
            
            # Re-run the ESG agent with feedback
            print(f"  > Re-running ESG agent for {ticker} with feedback...")
            revised_result = run_esg_agent(ticker)
            
            if "Error" in revised_result['assessment'] or "Max iterations reached" in revised_result['assessment']:
                print(f"  > Revised agent run failed or timed out for {ticker}. Ending loop.")
                result['assessment'] = current_assessment # Revert to last good state or error
                result['evaluator_status'] = 'FAILED_REVISION'
                result['revisions'] = revision_num + 1
                result['trace'] = trace + revised_result.get('trace', [])
                result['iterations'] = total_iterations + revised_result.get('iterations', 0)
                return result

            current_assessment = revised_result['assessment']
            trace.extend(revised_result['trace'])
            total_iterations += revised_result['iterations']

    print(f"  > Max revisions ({max_revisions}) reached without approval for {ticker}.")
    result['assessment'] = current_assessment
    result['evaluator_status'] = 'MAX_REVISIONS_REACHED'
    result['revisions'] = max_revisions
    result['trace'] = trace
    result['iterations'] = total_iterations
    return result

# Test the Evaluator-Optimizer loop for a single company
print("Running the Evaluator-Optimizer loop for ExxonMobil (XOM)... This may take longer.")
esg_result_xom = evaluator_optimizer(ticker='XOM')
print("\n--- Final ESG Assessment for XOM (after evaluation/optimization) ---")
print(esg_result_xom['assessment'])
print(f"\nStatus: {esg_result_xom['evaluator_status']}")
print(f"Revisions: {esg_result_xom['revisions']}")

try:
    parsed_xom_assessment = json.loads(esg_result_xom['assessment'])
    print(f"Environmental Score (XOM): {parsed_xom_assessment.get('environmental_score')}")
    print(f"Social Score (XOM): {parsed_xom_assessment.get('social_score')}")
    print(f"Governance Score (XOM): {parsed_xom_assessment.get('governance_score')}")
    print(f"Composite Score (XOM): {parsed_xom_assessment.get('composite_score')}")
except json.JSONDecodeError as e:
    print(f"\nError parsing JSON from final assessment: {e}")
    print("Raw assessment content for debugging:\n", esg_result_xom['assessment'])
```

### Explanation of Execution

The `evaluator_optimizer` function orchestrates the quality assurance process. It first runs our `run_esg_agent` (the Generator) to get an initial ESG assessment for a company like 'XOM'. Then, it sends this assessment to the `evaluator_llm`, which acts as a senior analyst. The `EVALUATOR_PROMPT` contains a checklist (e.g., "Are all three pillars scored?", "Is the output valid JSON?"). If the evaluator identifies any issues (e.g., missing scores, inconsistent rationale), it returns a "REVISE" status with specific feedback. The loop then re-runs the ESG agent, incorporating this feedback into its next attempt. This iterative refinement continues until the assessment is "APPROVED" or a maximum number of revisions is met. For a CFA, this pattern is highly valuable as it ensures the AI's output is robust, reliable, and adheres to firm-specific quality standards, mirroring a critical review process in real financial analysis.

## 6. Portfolio-Level ESG Screening and Scorecard Generation

Having established a robust, quality-assured ESG assessment process, your next step as an investment analyst is to apply this to a full portfolio of companies. This allows for direct comparison and identification of ESG leaders and laggards. In this section, you will define a portfolio of tickers, iterate through them, execute the `evaluator_optimizer` for each, and then compile the results into a comprehensive "Portfolio ESG Scorecard". Crucially, you will also implement the **materiality-weighted composite ESG score** formula to reflect the relative importance of E, S, and G factors for each industry.

The materiality-weighted composite ESG score ($S_{composite}$) is calculated as:
$$ S_{composite} = w_E \cdot S_E + w_S \cdot S_S + w_G \cdot S_G $$
where $S_E, S_S, S_G$ are the environmental, social, and governance scores, respectively. The weights $w_E, w_S, w_G$ are determined by the count of material topics for each pillar, relative to the total number of material topics for that industry.

For example, if an oil company has 3 material E topics, 1 S topic, and 2 G topics, out of a total of $3+1+2=6$ material topics, the weights would be:
$$ w_E = \frac{3}{6} = 0.50 $$
$$ w_S = \frac{1}{6} \approx 0.17 $$
$$ w_G = \frac{2}{6} \approx 0.33 $$
This ensures that the composite score reflects the pillar most financially material to the company's industry (e.g., an oil company's composite is dominated by environmental performance; a bank's composite by governance and data security).

```python
PORTFOLIO_TICKERS = ['AAPL', 'MSFT', 'XOM', 'JPM', 'JNJ']
portfolio_esg_assessments = []

print("Running ESG assessments for the entire portfolio:")

# Define a helper to categorize material topics for weighting
def categorize_material_topics(material_topics: list) -> dict:
    e_count = 0
    s_count = 0
    g_count = 0
    
    env_keywords = ['GHG Emissions', 'Energy Management', 'Water Management', 'Ecological Impacts', 'Air Quality']
    social_keywords = ['Employee Engagement', 'Labor Practices', 'Community Relations', 'Customer Privacy', 'Product Quality & Safety', 'Access to Healthcare', 'Ethical Marketing Practices']
    gov_keywords = ['Data Security', 'Business Ethics', 'Systemic Risk', 'Governance'] # Data Security often G for financials, or S for tech
    
    for topic in material_topics:
        if any(keyword in topic for keyword in env_keywords):
            e_count += 1
        elif any(keyword in topic for keyword in social_keywords):
            s_count += 1
        elif any(keyword in topic for keyword in gov_keywords):
            g_count += 1
        else: # Default categorization if not explicitly mapped
            if 'Environmental' in topic: e_count +=1
            if 'Social' in topic: s_count +=1
            if 'Governance' in topic: g_count +=1
            if 'Data Security' in topic: # Specific handling for common crossover
                if 'Financial Services' in materiality['industry'] or 'Technology' in materiality['industry']:
                    g_count += 1 
                else: 
                    s_count += 1
    
    return {'E': e_count, 'S': s_count, 'G': g_count}


for ticker in PORTFOLIO_TICKERS:
    print(f"\n{'='*60}")
    print(f"Assessing: {ticker}")
    
    result = evaluator_optimizer(ticker)

    # Parse JSON from assessment, handling potential markdown formatting
    assessment_json_str = result['assessment']
    parsed_assessment = {}
    try:
        json_match = re.search(r'```json\n({.*?})\n```', assessment_json_str, re.DOTALL)
        if json_match:
            parsed_assessment = json.loads(json_match.group(1))
        else:
            parsed_assessment = json.loads(assessment_json_str) # Assume direct JSON if no markdown
    except json.JSONDecodeError as e:
        print(f"    ! Error parsing JSON for {ticker}: {e}. Assessment content:\n{assessment_json_str[:500]}...")
        # Skip this company or fill with error placeholders
        parsed_assessment = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0, 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Error in assessment parsing'}

    # Add evaluator status and revisions to the parsed assessment
    parsed_assessment['evaluator_status'] = result['evaluator_status']
    parsed_assessment['revisions_taken'] = result['revisions']

    # Calculate materiality-weighted composite score
    e_score = parsed_assessment.get('environmental_score', 0)
    s_score = parsed_assessment.get('social_score', 0)
    g_score = parsed_assessment.get('governance_score', 0)
    
    materiality = determine_material_topics(ticker)
    topic_counts = categorize_material_topics(materiality['material_topics'])
    
    total_topics = sum(topic_counts.values())
    
    if total_topics > 0:
        w_e = topic_counts['E'] / total_topics
        w_s = topic_counts['S'] / total_topics
        w_g = topic_counts['G'] / total_topics
    else: # Default to equal weighting if no material topics found
        w_e, w_s, w_g = 1/3, 1/3, 1/3

    # Recalculate composite score with materiality weights
    parsed_assessment['w_e'] = round(w_e, 2)
    parsed_assessment['w_s'] = round(w_s, 2)
    parsed_assessment['w_g'] = round(w_g, 2)
    parsed_assessment['composite_score_materiality_weighted'] = round(w_e * e_score + w_s * s_score + w_g * g_score, 2)
    
    # Store the result
    portfolio_esg_assessments.append(parsed_assessment)

# Compile the scorecard
scorecard_df = pd.DataFrame(portfolio_esg_assessments)

display_cols = ['ticker', 'industry', 'environmental_score', 'social_score', 'governance_score', 
                'w_e', 'w_s', 'w_g', 'composite_score_materiality_weighted', 'recommendation', 'evaluator_status', 'revisions_taken']

# Filter columns to only include those present in the DataFrame
scorecard_display = scorecard_df[[c for c in display_cols if c in scorecard_df.columns]]

print(f"\n{'='*70}")
print("PORTFOLIO ESG SCORECARD (Materiality-Weighted)")
print(f"{'='*70}")
print(scorecard_display.sort_values('composite_score_materiality_weighted', ascending=False).to_string(index=False))
```

### Explanation of Execution

This section automates the ESG screening for our predefined `PORTFOLIO_TICKERS`. For each company, it calls the `evaluator_optimizer` function, ensuring a quality-controlled assessment. The raw LLM output is parsed to extract structured JSON data. Crucially, before compiling the final scorecard, the code calculates the `composite_score_materiality_weighted`. It does this by first categorizing the `sasb_material_topics` into E, S, and G pillars (e.g., 'GHG Emissions' for E, 'Employee Engagement' for S, 'Data Security' for G depending on context). The count of topics per pillar determines the weights ($w_E, w_S, w_G$). These weights are then applied to the individual E, S, and G scores using the formula $S_{composite} = w_E \cdot S_E + w_S \cdot S_S + w_G \cdot S_G$. This results in a composite score that truly reflects the materiality-driven approach, providing a ranked `scorecard_df` where companies are ordered by their financially relevant ESG performance. This deliverable is a key output for an investment analyst, facilitating quick comparative analysis and informed decision-making.

## 7. Visualizing Individual Company ESG Profiles and Portfolio Overview

A raw table of scores, while informative, often lacks the immediate insights that visualizations can provide. As an investment analyst, you need clear, concise visual summaries to quickly grasp a company's ESG strengths and weaknesses and to compare performance across your portfolio. In this section, you will generate:

*   **Individual Company ESG Profiles:** Detailed textual summaries for each company, highlighting material topics, score rationales, and controversies.
*   **ESG Pillar Radar Chart:** A radar plot for each company to visually represent its E, S, and G scores, providing an intuitive "ESG profile."
*   **Controversy Heat Map:** A visualization that provides an overview of controversy types and their severity across the portfolio.

```python
# Function to display individual company ESG profiles
def display_company_profile(assessment: dict):
    print(f"\n--- ESG Profile for {assessment.get('company', 'N/A')} ({assessment.get('ticker', 'N/A')}) ---")
    print(f"Industry: {assessment.get('industry', 'N/A')}")
    print(f"SASB Material Topics: {', '.join(assessment.get('sasb_material_topics', []))}")
    print(f"\nEnvironmental Score (Weighted {assessment.get('w_e',0)}): {assessment.get('environmental_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('environmental_rationale', 'N/A')}")
    print(f"\nSocial Score (Weighted {assessment.get('w_s',0)}): {assessment.get('social_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('social_rationale', 'N/A')}")
    print(f"\nGovernance Score (Weighted {assessment.get('w_g',0)}): {assessment.get('governance_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('governance_rationale', 'N/A')}")
    print(f"\nComposite Score (Materiality-Weighted): {assessment.get('composite_score_materiality_weighted', 'N/A')}")
    print(f"Controversies: {assessment.get('controversies_summary', 'None')}")
    print(f"Peer Comparison: {assessment.get('peer_comparison', 'N/A')}")
    print(f"Key Risks: {', '.join(assessment.get('key_risks', [])) if assessment.get('key_risks') else 'None'}")
    print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
    print(f"Evaluation Status: {assessment.get('evaluator_status', 'N/A')} (Revisions: {assessment.get('revisions_taken', 'N/A')})")
    print(f"{'-'*60}\n")

# Plot ESG Pillar Radar Chart
def plot_radar_chart(df: pd.DataFrame, ticker: str):
    data = df[df['ticker'] == ticker].iloc[0]
    categories = ['Environmental', 'Social', 'Governance']
    scores = [data['environmental_score'], data['social_score'], data['governance_score']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=scores + [scores[0]], # Close the loop
          theta=categories + [categories[0]], # Close the loop
          fill='toself',
          name=ticker
    ))
    fig.update_layout(
        polar=dict(
            radialaxis_tickfont_size=10,
            radialaxis=dict(
                range=[0, 100], # Scores are 0-100
                visible=True,
                autorange=False
            )),
        showlegend=True,
        title=f'ESG Pillar Radar Chart for {ticker}',
        height=400, width=500
    )
    fig.show()

# Plot Controversy Heat Map
def plot_controversy_heatmap(assessments: list):
    controversy_data = []
    all_types = set()
    for item in assessments:
        ticker = item.get('ticker')
        controversies_raw = scan_controversies(ticker)
        controversies_list = json.loads(controversies_raw)
        
        for c in controversies_list:
            c_type = c['type']
            severity = c['severity']
            all_types.add(c_type)
            controversy_data.append({'ticker': ticker, 'type': c_type, 'severity': severity})

    if not controversy_data:
        print("No controversies detected across the portfolio for heatmap generation.")
        return

    # Create a DataFrame for heatmap
    controversy_df = pd.DataFrame(controversy_data)
    
    # Map severity to numerical value
    severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    controversy_df['severity_num'] = controversy_df['severity'].map(severity_map)
    
    # Pivot table for heatmap, filling NaNs with 0 (no controversy)
    heatmap_data = controversy_df.pivot_table(index='ticker', columns='type', values='severity_num', fill_value=0)
    
    # Ensure all_types are in columns, add if missing
    for c_type in all_types:
        if c_type not in heatmap_data.columns:
            heatmap_data[c_type] = 0

    plt.figure(figsize=(10, len(PORTFOLIO_TICKERS) * 0.8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='g', linewidths=.5, cbar_kws={'label': 'Controversy Severity (1=Low, 3=High)'})
    plt.title('Portfolio Controversy Heat Map (Severity by Type)')
    plt.xlabel('Controversy Type')
    plt.ylabel('Company Ticker')
    plt.tight_layout()
    plt.show()

# Display profiles and plots for each company
print("Generating individual company ESG profiles and visualizations...")
for assessment in portfolio_esg_assessments:
    display_company_profile(assessment)
    plot_radar_chart(scorecard_df, assessment['ticker'])

# Generate portfolio-wide controversy heatmap
plot_controversy_heatmap(portfolio_esg_assessments)
```

### Explanation of Execution

This section transforms our structured ESG data into easily digestible visual formats. The `display_company_profile` function iterates through each company's assessment, printing a detailed summary of its scores, rationales, material topics, and controversies. This text-based output provides the necessary qualitative context for each score, which is critical for an analyst.

For a quick visual profile, `plot_radar_chart` generates a spider plot for each company. The plot illustrates the relative performance across the Environmental, Social, and Governance pillars on a 0-100 scale. This visual aid allows for immediate identification of a company's ESG strengths and weaknesses.

Finally, `plot_controversy_heatmap` aggregates controversy data across the entire portfolio. It maps controversy types (e.g., Environmental, Social, Governance) against company tickers, with cell color indicating severity. This heatmap provides a high-level overview of risk areas, helping the analyst quickly spot patterns or specific companies with recurring issues. These visualizations are invaluable to an investment analyst for efficiently communicating insights to clients or internal stakeholders, enabling faster and more intuitive understanding of complex ESG data.

## 8. Assessing Score Consistency and Variability

As a seasoned investment professional, you understand that even with clear rubrics and structured processes, assessments involving qualitative judgment (like ESG) can exhibit some variability, especially when using LLMs. This isn't necessarily a flaw, but an inherent characteristic of the task itself, much like different human analysts might slightly diverge in their ratings. This section addresses this by running the ESG assessment for a single company multiple times and analyzing the consistency of the generated scores. This analysis provides valuable insights into the reliability of the agent's output and helps set expectations for interpreting the scores.

**Practitioner Warning:**
Agent-generated ESG scores are inherently subjective and variable. Even with the same data and rubric, running the agent multiple times on the same company may produce scores varying by 5-15 points (out of 100) due to LLM stochasticity and interpretation differences. This is not a bug—it reflects the genuine ambiguity in ESG assessment. But it means: (a) scores should be used for ranking (relative comparison) rather than absolute assessment, (b) scores near boundaries (e.g., 59 vs. 61 for "Adequate" vs. "Strong") should be treated as uncertain, and (c) the rationale is more important than the number—the analyst should read the justification, not just the score. ESG rating agencies face the same challenge: MSCI and Sustainalytics often disagree significantly on the same company. The agent's variability is no worse than inter-rater disagreement among professional ESG analysts.

```python
# Select a company to run consistency checks on
company_for_consistency = 'MSFT'
num_runs = 5
consistency_scores = []

print(f"Running ESG agent {num_runs} times for {company_for_consistency} to check score consistency...")

for i in range(num_runs):
    print(f"\n--- Consistency Run {i+1}/{num_runs} for {company_for_consistency} ---")
    result = evaluator_optimizer(company_for_consistency, max_revisions=3)
    
    if result['evaluator_status'] == 'APPROVED' or result['evaluator_status'] == 'MAX_REVISIONS_REACHED':
        try:
            assessment_json_str = result['assessment']
            json_match = re.search(r'```json\n({.*?})\n```', assessment_json_str, re.DOTALL)
            if json_match:
                parsed_assessment = json.loads(json_match.group(1))
            else:
                parsed_assessment = json.loads(assessment_json_str)
            
            consistency_scores.append({
                'run': i + 1,
                'E': parsed_assessment.get('environmental_score', 0),
                'S': parsed_assessment.get('social_score', 0),
                'G': parsed_assessment.get('governance_score', 0),
                'Composite_Materiality_Weighted': parsed_assessment.get('composite_score_materiality_weighted', 0)
            })
        except json.JSONDecodeError as e:
            print(f"    ! Error parsing JSON in consistency run {i+1}: {e}")
            print(f"    Raw content: {result['assessment'][:200]}...")
        except Exception as e:
            print(f"    ! General error in consistency run {i+1}: {e}")
    else:
        print(f"    ! Assessment not approved or failed for run {i+1}. Status: {result['evaluator_status']}")

if consistency_scores:
    consistency_df = pd.DataFrame(consistency_scores)
    
    print(f"\n{'='*70}")
    print(f"SCORE CONSISTENCY ( {num_runs} runs for {company_for_consistency} )")
    print(f"{'='*70}")
    print(consistency_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("SCORE RANGES:")
    print(f"{'='*70}")
    
    score_columns = ['E', 'S', 'G', 'Composite_Materiality_Weighted']
    for col in score_columns:
        if col in consistency_df.columns:
            score_range = consistency_df[col].max() - consistency_df[col].min()
            print(f"{col} Range: {consistency_df[col].max():.1f} - {consistency_df[col].min():.1f} = {score_range:.1f}")
    
    print("\n(Range > 10 typically indicates significant score instability for a single input)")

    # Plot Score Consistency Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=consistency_df[score_columns], palette='viridis')
    plt.title(f'ESG Score Consistency Across {num_runs} Runs for {company_for_consistency}')
    plt.ylabel('Score (0-100)')
    plt.xlabel('ESG Pillar / Composite Score')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

else:
    print("\nNo consistency scores collected for analysis.")
```

### Explanation of Execution

In this final section, we delve into the practical implications of using LLMs for ESG scoring. The code executes the `evaluator_optimizer` process `num_runs` (e.g., 5) times for a chosen company (e.g., 'MSFT'). Each run produces a set of E, S, G, and composite scores. The results are then compiled into `consistency_df`, which is displayed as a table.

The core of the analysis lies in calculating the **range** for each score (max score - min score) across the multiple runs. For instance, an environmental score range of `85.0 - 75.0 = 10.0` suggests a 10-point variability. A range greater than 10 points is often highlighted as indicating "score instability," signaling that the LLM's interpretation or generation can differ substantially even with identical inputs.

Finally, a **box plot** is generated to visually represent the distribution and variability of scores for each pillar and the composite. This visual cue quickly communicates the degree of consistency. For an investment analyst, understanding this variability is crucial. It informs them that these scores should be primarily used for relative ranking and comparative analysis within the portfolio rather than as absolute, definitive measures. It reinforces the need to always review the underlying rationale provided by the agent, not just the numerical score, to make informed decisions. This step grounds the AI's output in financial reality, acknowledging the inherent judgment involved in ESG assessment.
