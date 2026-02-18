import os
import json
import pandas as pd
import yfinance as yf  # Not directly used in the provided code, but kept for completeness
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
import numpy as np
from pydantic import BaseModel, Field
from typing import List

# --- Pydantic Schemas for Structured Outputs ---


class ESGAssessment(BaseModel):
    """Pydantic model for ESG Assessment structured output"""
    company: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    industry: str = Field(description="Industry classification")
    sasb_material_topics: List[str] = Field(
        description="List of SASB material topics")
    environmental_score: float = Field(
        description="Environmental score (0-100)", ge=0, le=100)
    environmental_rationale: str = Field(
        description="Rationale for environmental score")
    social_score: float = Field(
        description="Social score (0-100)", ge=0, le=100)
    social_rationale: str = Field(description="Rationale for social score")
    governance_score: float = Field(
        description="Governance score (0-100)", ge=0, le=100)
    governance_rationale: str = Field(
        description="Rationale for governance score")
    composite_score: float = Field(
        description="Composite ESG score (0-100)", ge=0, le=100)
    controversies_summary: str = Field(description="Summary of controversies")
    peer_comparison: str = Field(description="Peer comparison analysis")
    key_risks: List[str] = Field(description="List of key ESG risks")
    recommendation: str = Field(
        description="ESG recommendation: Strong ESG / Adequate ESG / ESG Concern")


class EvaluatorResponse(BaseModel):
    """Pydantic model for Evaluator response"""
    status: str = Field(description="Status: APPROVED or REVISE")
    feedback: str = Field(description="Feedback for revision or approval")


# --- Configuration and Constants ---
# API Key will be passed as a parameter, not set in environment
# Define the ESG Agent's system prompt and scoring rubric as a constant
ESG_AGENT_SYSTEM_PROMPT = """
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
{{
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
}}
```
Use tools to gather ALL relevant data before scoring.
Cite specific data points in your rationale and reference the tool outputs.
"""
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
{{"status": "REVISE",
    "feedback": "Specific feedback on what needs revision (e.g., 'Missing S-pillar score and rationale.', 'Environmental score of X is inconsistent with high emissions data.', 'Peer comparison is generic, needs company-specific comparison.')."}}
```
Your feedback should be concise and actionable for the ESG agent.
"""
# --- Tool Definitions ---


@tool
def get_environmental_metrics(ticker: str) -> str:
    """Retrieve simulated environmental data (carbon emissions, energy usage, water usage, carbon neutral targets) for a given company ticker.
    Args:
      ticker (str): The ticker symbol of the company for which to retrieve environmental data.
    """
    env_data = {
        'AAPL': {
            'scope1_emissions_tco2': 22400,
            'scope2_emissions_tco2': 0,
            'scope3_emissions_tco2': 25000000,
            'total_emissions_tco2e': 25022400,
            'emissions_intensity_revenue': 12.5,
            'yoy_emissions_change_pct': -15.2,
            'renewable_energy_pct': 100,
            'renewable_energy_sources': ['Solar', 'Wind', 'Biogas'],
            'water_usage_megaliters': 1200,
            'water_recycled_pct': 35,
            'waste_diversion_rate_pct': 82,
            'carbon_neutral_target': '2030 (Scope 3)',
            'net_zero_commitment': 'Yes - 2030 across entire supply chain',
            'science_based_targets': 'Yes - approved by SBTi in 2023',
            'environmental_certifications': ['ISO 14001', 'LEED Platinum facilities'],
            'environmental_spend_usd_millions': 4500,
            'recent_initiatives': 'Carbon removal projects, 100% recycled materials in products'
        },
        'MSFT': {
            'scope1_emissions_tco2': 10000,
            'scope2_emissions_tco2': 1500,
            'scope3_emissions_tco2': 12000000,
            'total_emissions_tco2e': 12011500,
            'emissions_intensity_revenue': 8.3,
            'yoy_emissions_change_pct': -18.5,
            'renewable_energy_pct': 95,
            'renewable_energy_sources': ['Solar', 'Wind', 'Hydroelectric'],
            'water_usage_megaliters': 800,
            'water_recycled_pct': 45,
            'waste_diversion_rate_pct': 88,
            'carbon_neutral_target': '2030 (Scope 3)',
            'net_zero_commitment': 'Yes - carbon negative by 2030, remove historical emissions by 2050',
            'science_based_targets': 'Yes - approved by SBTi in 2022',
            'environmental_certifications': ['ISO 14001', 'Carbon Neutral Certified'],
            'environmental_spend_usd_millions': 5200,
            'recent_initiatives': '$1B climate innovation fund, AI for sustainability programs'
        },
        'XOM': {
            'scope1_emissions_tco2': 112000000,
            'scope2_emissions_tco2': 15000000,
            'scope3_emissions_tco2': 650000000,
            'total_emissions_tco2e': 777000000,
            'emissions_intensity_revenue': 285.4,
            'yoy_emissions_change_pct': -2.1,
            'renewable_energy_pct': 3,
            'renewable_energy_sources': ['Limited solar installations'],
            'water_usage_megaliters': 450000,
            'water_recycled_pct': 12,
            'waste_diversion_rate_pct': 35,
            'carbon_neutral_target': '2050 (Scope 1+2 only)',
            'net_zero_commitment': 'Net-zero by 2050 (operational emissions only, excludes Scope 3)',
            'science_based_targets': 'No',
            'environmental_certifications': ['ISO 14001 at select facilities'],
            'environmental_spend_usd_millions': 3000,
            'recent_initiatives': 'Carbon capture pilot projects, methane reduction efforts',
            'environmental_incidents_last_3yrs': 7,
            'spills_incidents': 'Multiple minor spills, 1 major incident in 2023'
        },
        'JPM': {
            'scope1_emissions_tco2': 5000,
            'scope2_emissions_tco2': 1000,
            'scope3_emissions_tco2': 350000,
            'total_emissions_tco2e': 356000,
            'emissions_intensity_revenue': 4.2,
            'yoy_emissions_change_pct': -12.8,
            'renewable_energy_pct': 70,
            'renewable_energy_sources': ['Wind', 'Solar'],
            'water_usage_megaliters': 50,
            'water_recycled_pct': 25,
            'waste_diversion_rate_pct': 75,
            'carbon_neutral_target': '2040 (Scope 3)',
            'net_zero_commitment': 'Yes - operational net-zero by 2030, financed emissions by 2050',
            'science_based_targets': 'Yes - approved by SBTi in 2024',
            'environmental_certifications': ['LEED Gold offices', 'ISO 14001'],
            'environmental_spend_usd_millions': 1200,
            'recent_initiatives': '$2.5T sustainable finance commitment, green bonds program',
            'sustainable_finance_portfolio_usd_billions': 320
        },
        'JNJ': {
            'scope1_emissions_tco2': 30000,
            'scope2_emissions_tco2': 5000,
            'scope3_emissions_tco2': 8500000,
            'total_emissions_tco2e': 8535000,
            'emissions_intensity_revenue': 95.3,
            'yoy_emissions_change_pct': -8.4,
            'renewable_energy_pct': 60,
            'renewable_energy_sources': ['Solar', 'Wind', 'Renewable Energy Credits'],
            'water_usage_megaliters': 3000,
            'water_recycled_pct': 32,
            'waste_diversion_rate_pct': 68,
            'carbon_neutral_target': '2045 (Scope 3)',
            'net_zero_commitment': 'Yes - carbon neutral operations by 2030, net-zero value chain by 2045',
            'science_based_targets': 'Yes - approved by SBTi in 2023',
            'environmental_certifications': ['ISO 14001', 'EcoVadis Gold'],
            'environmental_spend_usd_millions': 2100,
            'recent_initiatives': 'Sustainable packaging redesign, water stewardship programs, green chemistry'
        }
    }
    data = env_data.get(
        ticker, {'note': 'Environmental data not available for this ticker.'})
    return json.dumps(data, indent=2)


@tool
def scan_controversies(ticker: str) -> str:
    """Search for recent ESG controversies and incidents for a given company ticker.
    Args:
      ticker (str): The ticker symbol of the company for which to search for controversies.
    """
    controversies = {
        'AAPL': [
            {
                'type': 'Social',
                'severity': 'Medium',
                'description': 'Supply chain labor concerns at supplier facilities in Southeast Asia - reports of excessive overtime',
                'date': '2024-Q3',
                'status': 'Under investigation - company conducting third-party audits',
                'financial_impact_usd_millions': 0,
                'remediation': 'Enhanced supplier monitoring program implemented',
                'media_coverage': 'Moderate'
            },
            {
                'type': 'Environmental',
                'severity': 'Low',
                'description': 'Minor criticism over product packaging waste in European markets',
                'date': '2024-Q2',
                'status': 'Resolved - announced 100% fiber-based packaging initiative',
                'financial_impact_usd_millions': 0,
                'remediation': 'Accelerated transition to recyclable packaging',
                'media_coverage': 'Low'
            }
        ],
        'MSFT': [
            {
                'type': 'Governance',
                'severity': 'Low',
                'description': 'Minor data privacy incident affecting 1,000 users - data temporarily accessible',
                'date': '2023-Q4',
                'status': 'Resolved - systems patched within 48 hours',
                'financial_impact_usd_millions': 0.5,
                'remediation': 'Enhanced security protocols, user notification completed',
                'media_coverage': 'Low'
            }
        ],
        'XOM': [
            {
                'type': 'Environmental',
                'severity': 'High',
                'description': 'Ongoing climate litigation from multiple state attorneys general regarding climate change impacts and disclosure',
                'date': '2024-ongoing',
                'status': 'In litigation - cases filed in NY, MA, CA',
                'financial_impact_usd_millions': 'Unknown - potential billions',
                'remediation': 'Legal defense ongoing, no admission of wrongdoing',
                'media_coverage': 'High'
            },
            {
                'type': 'Governance',
                'severity': 'Medium',
                'description': 'Shareholder proposal requesting more aggressive emissions reduction targets rejected by board and failed vote (38% support)',
                'date': '2024-Q2',
                'status': 'Proposal failed - board recommended against',
                'financial_impact_usd_millions': 0,
                'remediation': 'Company issued statement defending current climate strategy',
                'media_coverage': 'Moderate'
            },
            {
                'type': 'Environmental',
                'severity': 'Medium',
                'description': 'Pipeline leak in Texas resulted in 2,000 barrels oil spill',
                'date': '2023-Q4',
                'status': 'Closed - cleanup completed, regulatory fines paid',
                'financial_impact_usd_millions': 12,
                'remediation': 'Pipeline infrastructure upgrades, environmental restoration',
                'media_coverage': 'Moderate'
            }
        ],
        'JPM': [
            {
                'type': 'Social',
                'severity': 'High',
                'description': 'Regulatory fines totaling $350M from SEC and CFTC for compliance failures in record-keeping and communications',
                'date': '2024-Q1',
                'status': 'Resolved - fines paid, consent decree signed',
                'financial_impact_usd_millions': 350,
                'remediation': 'Comprehensive compliance overhaul, enhanced monitoring systems',
                'media_coverage': 'High'
            },
            {
                'type': 'Governance',
                'severity': 'Low',
                'description': 'Criticism from proxy advisors over executive compensation increases despite mixed performance',
                'date': '2024-Q2',
                'status': 'Say-on-pay vote passed with 92% approval',
                'financial_impact_usd_millions': 0,
                'remediation': 'Compensation committee issued detailed rationale',
                'media_coverage': 'Low'
            }
        ],
        'JNJ': [
            {
                'type': 'Social',
                'severity': 'Medium',
                'description': 'Voluntary product recall of contact lens solution (1.2M units) due to potential contamination risk - no injuries reported',
                'date': '2023-Q3',
                'status': 'Resolved - recall completed, product reformulated',
                'financial_impact_usd_millions': 45,
                'remediation': 'Enhanced quality control procedures, FDA inspection passed',
                'media_coverage': 'Moderate'
            },
            {
                'type': 'Environmental',
                'severity': 'Low',
                'description': 'NGO report raised concerns about pharmaceutical pollution in wastewater at Indian manufacturing site',
                'date': '2024-Q1',
                'status': 'Under review - third-party environmental audit commissioned',
                'financial_impact_usd_millions': 0,
                'remediation': 'Wastewater treatment system upgrade underway',
                'media_coverage': 'Low'
            }
        ]
    }
    data = controversies.get(ticker, [])
    return json.dumps(data, indent=2)


@tool
def get_governance_data(ticker: str) -> str:
    """Retrieve simulated corporate governance metrics for a given company ticker.
    Args:
      ticker (str): The ticker symbol of the company for which to get governance data."""
    gov_data = {
        'AAPL': {
            'board_size': 8,
            'pct_independent': 87.5,
            'independent_directors': 7,
            'board_diversity_pct': 50,
            'women_on_board': 3,
            'ethnic_minority_directors': 2,
            'average_tenure_years': 8.2,
            'ceo_chair_separate': True,
            'lead_independent_director': True,
            'say_on_pay_approval': 94.5,
            'clawback_policy': True,
            'double_trigger_provisions': True,
            'esg_in_compensation': False,
            'board_evaluation_annual': True,
            'board_meetings_per_year': 8,
            'attendance_rate_pct': 98,
            'audit_committee_independent': True,
            'comp_committee_independent': True,
            'nominating_committee_independent': True,
            'risk_oversight_structure': 'Dedicated risk committee',
            'sustainability_committee': True,
            'code_of_conduct': 'Comprehensive - published and enforced',
            'whistleblower_policy': 'Yes - anonymous hotline available',
            'political_contributions_disclosure': 'Full disclosure',
            'lobbying_disclosure': 'Detailed annual report',
            'anti_corruption_policy': 'Yes - FCPA compliant',
            'cybersecurity_oversight': 'Board-level review quarterly',
            'shareholder_rights': 'One share one vote, no poison pill',
            'recent_governance_improvements': 'Added sustainability expertise to board in 2024'
        },
        'MSFT': {
            'board_size': 10,
            'pct_independent': 90.0,
            'independent_directors': 9,
            'board_diversity_pct': 60,
            'women_on_board': 4,
            'ethnic_minority_directors': 3,
            'average_tenure_years': 6.5,
            'ceo_chair_separate': True,
            'lead_independent_director': True,
            'say_on_pay_approval': 96.0,
            'clawback_policy': True,
            'double_trigger_provisions': True,
            'esg_in_compensation': True,
            'esg_metrics_in_comp': 'Carbon reduction, diversity goals (20% of LTI)',
            'board_evaluation_annual': True,
            'board_meetings_per_year': 9,
            'attendance_rate_pct': 99,
            'audit_committee_independent': True,
            'comp_committee_independent': True,
            'nominating_committee_independent': True,
            'risk_oversight_structure': 'Full board oversight with committee support',
            'sustainability_committee': True,
            'code_of_conduct': 'Comprehensive - Standards of Business Conduct',
            'whistleblower_policy': 'Yes - Office of Legal Compliance',
            'political_contributions_disclosure': 'Full disclosure with rationale',
            'lobbying_disclosure': 'Detailed semi-annual reports',
            'anti_corruption_policy': 'Yes - global anti-bribery program',
            'cybersecurity_oversight': 'Board-level cybersecurity committee',
            'shareholder_rights': 'Strong rights, proxy access provisions',
            'recent_governance_improvements': 'Strengthened ESG metrics in executive compensation 2023'
        },
        'XOM': {
            'board_size': 12,
            'pct_independent': 91.7,
            'independent_directors': 11,
            'board_diversity_pct': 33,
            'women_on_board': 3,
            'ethnic_minority_directors': 1,
            'average_tenure_years': 9.8,
            'ceo_chair_separate': False,
            'lead_independent_director': True,
            'say_on_pay_approval': 88.2,
            'clawback_policy': True,
            'double_trigger_provisions': True,
            'esg_in_compensation': True,
            'esg_metrics_in_comp': 'Safety performance, emissions reduction (15% of annual bonus)',
            'board_evaluation_annual': True,
            'board_meetings_per_year': 10,
            'attendance_rate_pct': 96,
            'audit_committee_independent': True,
            'comp_committee_independent': True,
            'nominating_committee_independent': True,
            'risk_oversight_structure': 'Board committees review specific risk categories',
            'sustainability_committee': False,
            'code_of_conduct': 'Standards of Business Conduct - annually certified',
            'whistleblower_policy': 'Yes - third-party managed hotline',
            'political_contributions_disclosure': 'Annual disclosure',
            'lobbying_disclosure': 'Annual report',
            'anti_corruption_policy': 'Yes - anticorruption compliance program',
            'cybersecurity_oversight': 'Audit committee oversight',
            'shareholder_rights': 'Standard rights, majority vote for directors',
            'recent_governance_concerns': 'Combined CEO/Chair role criticized by proxy advisors',
            'recent_governance_improvements': 'Added climate risk expertise to board 2023'
        },
        'JPM': {
            'board_size': 14,
            'pct_independent': 85.7,
            'independent_directors': 12,
            'board_diversity_pct': 45,
            'women_on_board': 5,
            'ethnic_minority_directors': 3,
            'average_tenure_years': 7.3,
            'ceo_chair_separate': True,
            'lead_independent_director': True,
            'say_on_pay_approval': 92.0,
            'clawback_policy': True,
            'double_trigger_provisions': True,
            'esg_in_compensation': True,
            'esg_metrics_in_comp': 'DE&I goals, climate finance targets, conduct metrics (25% of STI)',
            'board_evaluation_annual': True,
            'board_meetings_per_year': 12,
            'attendance_rate_pct': 97,
            'audit_committee_independent': True,
            'comp_committee_independent': True,
            'nominating_committee_independent': True,
            'risk_oversight_structure': 'Comprehensive enterprise risk committee',
            'sustainability_committee': True,
            'code_of_conduct': 'Code of Conduct - mandatory annual training',
            'whistleblower_policy': 'Yes - Ethics Hotline operated independently',
            'political_contributions_disclosure': 'Comprehensive semi-annual disclosure',
            'lobbying_disclosure': 'Detailed quarterly reports',
            'anti_corruption_policy': 'Yes - Anti-Money Laundering and sanctions programs',
            'cybersecurity_oversight': 'Risk committee oversight, quarterly briefings',
            'shareholder_rights': 'Proxy access, special meeting rights',
            'regulatory_compliance_infrastructure': 'Enhanced post-2024 consent order',
            'recent_governance_improvements': 'Strengthened compliance controls following regulatory settlements'
        },
        'JNJ': {
            'board_size': 11,
            'pct_independent': 81.8,
            'independent_directors': 9,
            'board_diversity_pct': 40,
            'women_on_board': 4,
            'ethnic_minority_directors': 2,
            'average_tenure_years': 8.9,
            'ceo_chair_separate': True,
            'lead_independent_director': True,
            'say_on_pay_approval': 91.0,
            'clawback_policy': True,
            'double_trigger_provisions': True,
            'esg_in_compensation': True,
            'esg_metrics_in_comp': 'Patient safety, environmental goals, diversity (20% of LTI)',
            'board_evaluation_annual': True,
            'board_meetings_per_year': 9,
            'attendance_rate_pct': 98,
            'audit_committee_independent': True,
            'comp_committee_independent': True,
            'nominating_committee_independent': True,
            'risk_oversight_structure': 'Risk based approach across committees',
            'sustainability_committee': True,
            'code_of_conduct': 'Our Credo - values-based culture document',
            'whistleblower_policy': 'Yes - Credo hotline available globally',
            'political_contributions_disclosure': 'Annual disclosure',
            'lobbying_disclosure': 'Annual detailed report',
            'anti_corruption_policy': 'Yes - Healthcare Compliance and FCPA programs',
            'cybersecurity_oversight': 'Audit and compliance committee oversight',
            'shareholder_rights': 'One share one vote, proxy access',
            'product_quality_oversight': 'Enhanced quality committee established 2023',
            'recent_governance_improvements': 'Expanded board quality oversight after product recalls'
        }
    }
    data = gov_data.get(ticker, {})
    return json.dumps(data, indent=2)


@tool
def get_sasb_materiality(industry: str) -> str:
    """Get SASB material ESG topics for a given industry. This mapping guides our materiality-driven analysis.
    Args:
      industry (str): The industry for which to retrieve material ESG topics.
    """
    sasb_map = {
        'Technology': {
            'material_topics': ['Data Security', 'Employee Engagement & Diversity', 'GHG Emissions', 'Energy Management', 'Supply Chain Management', 'Product Lifecycle Management', 'Materials Sourcing'],
            'sasb_industry': 'Hardware / Software & IT Services',
            'key_metrics': ['Scope 1, 2, 3 GHG emissions', 'Employee turnover', 'Supply chain labor standards', 'Renewable energy percentage', 'Data breaches']
        },
        'Oil & Gas': {
            'material_topics': ['GHG Emissions', 'Air Quality', 'Water & Wastewater Management', 'Biodiversity & Ecological Impacts', 'Community Relations', 'Business Ethics & Transparency', 'Safety Management', 'Operational Efficiency'],
            'sasb_industry': 'Oil & Gas - Exploration & Production',
            'key_metrics': ['Total GHG emissions', 'Methane emissions', 'Spills and incidents', 'Water withdrawn/consumed', 'TRIR (safety)', 'Reserves replacement']
        },
        'Financial Services': {
            'material_topics': ['Data Security & Customer Privacy', 'Business Ethics & Fraud Prevention', 'Systemic Risk Management', 'Employee Engagement & Diversity', 'Incorporation of ESG Factors in Investment', 'Financed Emissions'],
            'sasb_industry': 'Commercial Banks',
            'key_metrics': ['Data breaches', 'Regulatory fines', 'Gender/racial pay gap', 'Sustainable finance volume', 'Financed emissions']
        },
        'Healthcare': {
            'material_topics': ['Product Quality & Safety', 'Access to Healthcare & Affordability', 'GHG Emissions', 'Ethical Marketing Practices', 'Drug Pricing & Transparency', 'Clinical Trial Ethics', 'Counterfeit Products'],
            'sasb_industry': 'Pharmaceuticals / Medical Equipment',
            'key_metrics': ['Product recalls', 'Access programs value', 'R&D investment', 'Emissions intensity', 'Marketing compliance incidents']
        },
        'Consumer Cyclical': {
            'material_topics': ['Labor Practices', 'Supply Chain Management', 'Product Safety & Quality', 'Data Security', 'Raw Material Sourcing', 'Packaging & Waste'],
            'sasb_industry': 'Multiline and Specialty Retailers & Distributors',
            'key_metrics': ['Supply chain audits', 'Product recalls', 'Worker safety incidents', 'Sustainable sourcing percentage', 'Packaging recycled content']
        }
    }
    data = sasb_map.get(industry, {
        'material_topics': ['GHG Emissions', 'Employee Engagement', 'Business Ethics', 'Community Relations'],
        'sasb_industry': 'General',
        'key_metrics': ['Emissions', 'Employee metrics', 'Compliance incidents']
    })
    return json.dumps({'industry': industry, **data}, indent=2)


@tool
def get_peer_esg_scores(ticker: str) -> str:
    """Compare ESG metrics to sector peers for a given company ticker.
    Args:
      ticker (str): The ticker symbol of the company for which to get peer esg scores."""
    peer_data = {
        'AAPL': {
            'sector': 'Technology Hardware',
            'peers': ['SAMSUNG', 'DELL', 'HP', 'LENOVO'],
            'sector_avg_emissions_intensity': 45,
            'company_emissions_intensity': 12.5,
            'sector_avg_renewable_energy_pct': 55,
            'company_renewable_energy_pct': 100,
            'sector_avg_board_independence': 82,
            'company_board_independence': 87.5,
            'sector_avg_board_diversity': 35,
            'company_board_diversity': 50,
            'sector_avg_esg_disclosure_score': 72,
            'company_esg_disclosure_score': 89,
            'esg_rating_proxy': 'AA (top quartile)',
            'esg_ranking': 'Ranked #2 out of 45 tech hardware companies',
            'peer_performance': 'Significantly outperforms sector average on environmental metrics, above average on governance',
            'competitive_advantage': 'Industry leader in renewable energy adoption and supply chain transparency'
        },
        'MSFT': {
            'sector': 'Software & IT Services',
            'peers': ['GOOGLE', 'AMAZON', 'META', 'SALESFORCE'],
            'sector_avg_emissions_intensity': 30,
            'company_emissions_intensity': 8.3,
            'sector_avg_renewable_energy_pct': 68,
            'company_renewable_energy_pct': 95,
            'sector_avg_board_independence': 88,
            'company_board_independence': 90.0,
            'sector_avg_board_diversity': 40,
            'company_board_diversity': 60,
            'sector_avg_esg_disclosure_score': 85,
            'company_esg_disclosure_score': 95,
            'esg_rating_proxy': 'AAA (top 5%)',
            'esg_ranking': 'Ranked #1 out of 78 software companies',
            'peer_performance': 'Best-in-class across all ESG dimensions, particularly strong on carbon negative commitment',
            'competitive_advantage': 'First major tech company to commit to carbon negative status and historical emissions removal'
        },
        'XOM': {
            'sector': 'Oil & Gas - Integrated',
            'peers': ['CHEVRON', 'SHELL', 'BP', 'TOTALENERGIES'],
            'sector_avg_emissions_intensity': 200,
            'company_emissions_intensity': 285.4,
            'sector_avg_renewable_energy_pct': 8,
            'company_renewable_energy_pct': 3,
            'sector_avg_board_independence': 89,
            'company_board_independence': 91.7,
            'sector_avg_board_diversity': 30,
            'company_board_diversity': 33,
            'sector_avg_esg_disclosure_score': 58,
            'company_esg_disclosure_score': 62,
            'esg_rating_proxy': 'BBB (bottom quartile)',
            'esg_ranking': 'Ranked #42 out of 55 oil & gas companies',
            'peer_performance': 'Below sector average on emissions intensity and renewable investments, average governance',
            'competitive_advantage': 'Strong traditional governance metrics but lagging on energy transition compared to European peers'
        },
        'JPM': {
            'sector': 'Commercial Banks',
            'peers': ['BANK_OF_AMERICA', 'CITIGROUP', 'WELLS_FARGO', 'GOLDMAN_SACHS'],
            'sector_avg_emissions_intensity': 10,
            'company_emissions_intensity': 4.2,
            'sector_avg_renewable_energy_pct': 55,
            'company_renewable_energy_pct': 70,
            'sector_avg_board_independence': 87,
            'company_board_independence': 85.7,
            'sector_avg_board_diversity': 40,
            'company_board_diversity': 45,
            'sector_avg_esg_disclosure_score': 78,
            'company_esg_disclosure_score': 82,
            'esg_rating_proxy': 'A (second quartile)',
            'esg_ranking': 'Ranked #15 out of 62 commercial banks',
            'esg_ranking_note': 'Strong sustainable finance leadership, governance improvements post-regulatory actions',
            'peer_performance': 'Above average on climate finance commitments, average to above average governance',
            'competitive_advantage': 'Leading sustainable finance volumes ($320B portfolio), enhanced compliance after 2024 settlements'
        },
        'JNJ': {
            'sector': 'Pharmaceuticals',
            'peers': ['PFIZER', 'ROCHE', 'NOVARTIS', 'MERCK'],
            'sector_avg_emissions_intensity': 60,
            'company_emissions_intensity': 95.3,
            'sector_avg_renewable_energy_pct': 48,
            'company_renewable_energy_pct': 60,
            'sector_avg_board_independence': 85,
            'company_board_independence': 81.8,
            'sector_avg_board_diversity': 38,
            'company_board_diversity': 40,
            'sector_avg_esg_disclosure_score': 80,
            'company_esg_disclosure_score': 84,
            'esg_rating_proxy': 'A (second quartile)',
            'esg_ranking': 'Ranked #12 out of 48 pharmaceutical companies',
            'peer_performance': 'Average across most metrics, stronger on renewable energy adoption than peers',
            'competitive_advantage': 'Strong Our Credo culture and healthcare access programs, improved quality oversight'
        }
    }
    data = peer_data.get(
        ticker, {'note': 'Peer comparison data not available.'})
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
ESG_TOOL_SCHEMAS = [convert_to_openai_tool(t) for t in TOOLS]
# --- Helper Functions ---


def determine_material_topics(ticker: str) -> dict:
    """
    Offline/dummy materiality resolver:
    - No yfinance calls (avoids 429s)
    - Uses a small ticker->industry mapping; falls back to generic topics
    """
    t = (ticker or "").strip().upper() or "UNKNOWN"
    # Minimal starter mapping (extend as you add portfolio names)
    ticker_to_industry = {
        "AAPL": "Technology Hardware",
        "MSFT": "Software & IT Services",
        "GOOG": "Internet Media & Services",
        "AMZN": "E-Commerce & Cloud Services",
        "TSLA": "Automobiles",
        "JPM": "Banks",
        "XOM": "Oil & Gas",
        "JNJ": "Healthcare",
    }
    # Simple industry -> material topics mapping (dummy but structured)
    industry_to_topics = {
        "Technology Hardware": [
            "Product Lifecycle Management",
            "Supply Chain Management",
            "Labor Practices",
            "Data Privacy & Security",
            "Energy Management",
        ],
        "Software & IT Services": [
            "Data Privacy & Security",
            "Business Ethics",
            "Employee Engagement & Inclusion",
            "Energy Management (Data Centers)",
        ],
        "Internet Media & Services": [
            "Data Privacy & Security",
            "Content Governance",
            "Business Ethics",
            "Human Rights & User Safety",
        ],
        "E-Commerce & Cloud Services": [
            "Data Privacy & Security",
            "Labor Practices",
            "Packaging & Waste",
            "Energy Management (Logistics/Data Centers)",
        ],
        "Automobiles": [
            "Product Safety",
            "Fuel Economy & Emissions",
            "Supply Chain Management",
            "Materials Sourcing",
            "Labor Practices",
        ],
        "Banks": [
            "Business Ethics",
            "Customer Privacy",
            "Systemic Risk Management",
            "Responsible Lending",
        ],
        "Oil & Gas": [
            "GHG Emissions",
            "Water & Wastewater Management",
            "Safety & Emergency Management",
            "Biodiversity Impacts",
            "Business Ethics",
        ],
        "Healthcare": [
            "Product Quality & Safety",
            "Access to Healthcare",
            "GHG Emissions",
            "Ethical Marketing Practices",
        ],
    }
    industry = ticker_to_industry.get(t, "Unknown (Offline Dummy)")
    topics = industry_to_topics.get(
        industry,
        [
            "GHG Emissions & Energy Management",
            "Labor Practices & Workforce Safety",
            "Business Ethics & Transparency",
            "Data Privacy & Cybersecurity",
            "Product Quality & Customer Welfare",
        ],
    )
    return {"ticker": t, "industry": industry, "material_topics": topics}


def categorize_material_topics(material_topics: list) -> dict:
    """
    Categorizes material topics into Environmental (E), Social (S), and Governance (G) counts.
    Used for materiality weighting.
    """
    e_count = 0
    s_count = 0
    g_count = 0
    env_keywords = ['GHG Emissions', 'Energy Management',
                    'Water Management', 'Ecological Impacts', 'Air Quality', 'Emissions']
    social_keywords = ['Employee Engagement', 'Labor Practices', 'Community Relations', 'Customer Privacy',
                       'Product Quality & Safety', 'Access to Healthcare', 'Ethical Marketing Practices', 'Human Rights', 'Workforce Safety']
    gov_keywords = ['Data Security', 'Business Ethics',
                    'Systemic Risk', 'Governance', 'Transparency', 'Privacy']
    for topic in material_topics:
        topic_lower = topic.lower()
        if any(keyword.lower() in topic_lower for keyword in env_keywords):
            e_count += 1
        elif any(keyword.lower() in topic_lower for keyword in social_keywords):
            s_count += 1
        elif any(keyword.lower() in topic_lower for keyword in gov_keywords):
            g_count += 1
        else:  # Fallback if not keyword-matched
            if 'environmental' in topic_lower:
                e_count += 1
            elif 'social' in topic_lower:
                s_count += 1
            elif 'governance' in topic_lower:
                g_count += 1
            elif 'data security' in topic_lower:
                g_count += 1  # Defaulting data security to Governance
    return {'E': e_count, 'S': s_count, 'G': g_count}
# --- Core Agent Functions ---


def run_esg_agent(
    ticker: str,
    llm: ChatOpenAI,
    tools: list,
    tool_schemas: list,
    system_prompt: str,
    max_iterations: int = 15,
    messages_history: list = None
) -> dict:
    """
    Runs the ESG agent for a given ticker, interacting with the LLM and tools.
    Returns the final assessment or an error message and the execution trace.
    Accepts an optional messages_history to continue a conversation or provide initial context.
    """
    if messages_history:
        messages = messages_history
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Conduct a comprehensive ESG assessment of {ticker}. The data used is dummy so don't give me feedback as the topics are too generic. Work with what I have provided."
                f"Use all available tools, score each pillar, and produce the structured JSON output."
            ),
        ]
    trace = []
    for iteration in range(max_iterations):
        try:
            response = llm.invoke(
                messages, tools=tool_schemas, tool_choice="auto")
        except Exception as e:
            trace.append(
                {"error": f"LLM invocation failed: {e}", "iteration": iteration})
            return {
                "assessment": f"Error: LLM invocation failed after {iteration} steps. {e}",
                "trace": trace,
                "iterations": iteration,
            }
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {}) or {}
                tool_id = tc["id"]
                tool_obj = next((t for t in tools if getattr(
                    t, "name", None) == tool_name), None)
                if tool_obj is None:
                    err = f"Tool '{tool_name}' not found."
                    trace.append({"action": f"{tool_name}({tool_args})",
                                 "result": err, "iteration": iteration})
                    messages.append(ToolMessage(
                        tool_call_id=tool_id, content=err))
                    continue
                try:
                    result = tool_obj.invoke(tool_args)
                    if not isinstance(result, str):
                        result_str = json.dumps(
                            result, indent=2, ensure_ascii=False)
                    else:
                        result_str = result
                    trace.append({
                        "action": f"{tool_name}({tool_args})",
                        "result": (result_str[:300] + "...") if len(result_str) > 300 else result_str,
                        "iteration": iteration,
                    })
                    messages.append(ToolMessage(
                        tool_call_id=tool_id, content=result_str))
                except Exception as e:
                    err = f"Tool '{tool_name}' failed: {e}"
                    trace.append({"action": f"{tool_name}({tool_args})",
                                 "result": err, "iteration": iteration})
                    messages.append(ToolMessage(
                        tool_call_id=tool_id, content=err))
        else:
            content = response.content or ""
            json_match = re.search(
                r"```json\s*({.*?})\s*```", content, re.DOTALL)
            assessment_content = json_match.group(1) if json_match else content
            return {"assessment": assessment_content, "trace": trace, "iterations": iteration + 1}
    return {"assessment": "Max iterations reached without generating a final JSON assessment.",
            "trace": trace, "iterations": max_iterations}


def _evaluator_optimizer_core(
    ticker: str,
    esg_agent_func: callable,
    evaluator_llm: ChatOpenAI,
    evaluator_prompt: str,
    agent_llm: ChatOpenAI,
    tools: list,
    tool_schemas: list,
    agent_system_prompt: str,
    max_revisions: int = 1
) -> dict:
    """
    Runs the ESG agent and then uses an evaluator LLM to review and request revisions
    until approved or max_revisions are met.
    """
    print(f"\n--- Starting Evaluator-Optimizer for {ticker} ---")
    # Initial run of the ESG agent
    print(f"  > Running initial ESG agent for {ticker}...")
    result = esg_agent_func(ticker=ticker, llm=agent_llm, tools=tools,
                            tool_schemas=tool_schemas, system_prompt=agent_system_prompt)
    if "Error" in result['assessment'] or "Max iterations reached" in result['assessment']:
        print(
            f"  > Initial agent run failed or timed out for {ticker}. Skipping evaluation.")
        return {'assessment': result['assessment'], 'evaluator_status': 'FAILED', 'revisions': 0, 'trace': result['trace'], 'iterations': result['iterations']}
    current_assessment = result['assessment']
    trace = result['trace']
    total_iterations = result['iterations']
    for revision_num in range(max_revisions):
        print(
            f"  > Revision {revision_num + 1}/{max_revisions}: Evaluating current assessment...")
        eval_messages = [
            SystemMessage(content=evaluator_prompt.format(
                assessment=current_assessment))
        ]
        try:
            eval_response = evaluator_llm.invoke(
                eval_messages, response_format={"type": "json_object"})
            evaluation = eval_response.content
            if isinstance(evaluation, str):
                evaluation = json.loads(evaluation)
        except json.JSONDecodeError as e:
            print(
                f"    ! Error parsing evaluator JSON response: {e}. Raw content: {eval_response.content}")
            evaluation = {"status": "REVISE",
                          "feedback": f"Evaluator response malformed: {e}"}
        except Exception as e:
            print(f"    ! Evaluator LLM invocation failed: {e}")
            evaluation = {"status": "REVISE",
                          "feedback": f"Evaluator LLM error: {e}"}
        trace.append({'evaluator_action': 'evaluate', 'feedback': evaluation.get(
            'feedback', ''), 'status': evaluation.get('status', 'ERROR'), 'revision_num': revision_num})
        if evaluation.get('status') == 'APPROVED':
            print(f"  > Evaluator: APPROVED (revision {revision_num + 1}).")
            result['assessment'] = current_assessment
            result['evaluator_status'] = 'APPROVED'
            result['revisions'] = revision_num
            result['trace'] = trace
            result['iterations'] = total_iterations
            return result
        else:
            feedback = evaluation.get(
                'feedback', 'No specific feedback provided.')
            print(f"  > Evaluator: REVISE. Feedback: {feedback[:100]}...")
            # Re-run the ESG agent with feedback incorporated into the initial human message
            revised_initial_messages = [
                SystemMessage(content=agent_system_prompt),
                HumanMessage(
                    content=f"Conduct a comprehensive ESG assessment of {ticker}. "
                    f"Use all available tools, score each pillar, and produce the structured JSON output. "
                    f"Previous assessment was rejected with the following feedback: {feedback}"
                ),
            ]
            revised_result = esg_agent_func(ticker=ticker, llm=agent_llm, tools=tools, tool_schemas=tool_schemas,
                                            system_prompt=agent_system_prompt, messages_history=revised_initial_messages)
            if "Error" in revised_result['assessment'] or "Max iterations reached" in revised_result['assessment']:
                print(
                    f"  > Revised agent run failed or timed out for {ticker}. Ending loop.")
                # Revert to last good state or error
                result['assessment'] = current_assessment
                result['evaluator_status'] = 'FAILED_REVISION'
                result['revisions'] = revision_num + 1
                result['trace'] = trace + revised_result.get('trace', [])
                result['iterations'] = total_iterations + \
                    revised_result.get('iterations', 0)
                return result
            current_assessment = revised_result['assessment']
            trace.extend(revised_result['trace'])
            total_iterations += revised_result['iterations']
    print(
        f"  > Max revisions ({max_revisions}) reached without approval for {ticker}.")
    result['assessment'] = current_assessment
    result['evaluator_status'] = 'MAX_REVISIONS_REACHED'
    result['revisions'] = max_revisions
    result['trace'] = trace
    result['iterations'] = total_iterations
    return result


# --- Wrapper Functions for Streamlit App ---

def evaluator_optimizer(ticker: str, api_key: str, max_revisions: int = 3) -> dict:
    """
    Main entry point for Streamlit app that initializes LLMs with API key
    and calls the core evaluator_optimizer with structured outputs.

    Args:
        ticker: Stock ticker symbol
        api_key: OpenAI API key
        max_revisions: Maximum number of revision iterations

    Returns:
        dict with assessment, evaluator_status, revisions, trace, iterations
    """
    # Initialize LLMs with the provided API key
    os.environ["OPENAI_API_KEY"] = api_key
    agent_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=api_key)
    evaluator_llm_base = ChatOpenAI(
        model="gpt-4o", temperature=0.0, api_key=api_key)

    # Call the core evaluator_optimizer function
    result = _evaluator_optimizer_core(
        ticker=ticker,
        esg_agent_func=run_esg_agent,
        evaluator_llm=evaluator_llm_base,
        evaluator_prompt=EVALUATOR_PROMPT,
        agent_llm=agent_llm,
        tools=TOOLS,
        tool_schemas=ESG_TOOL_SCHEMAS,
        agent_system_prompt=ESG_AGENT_SYSTEM_PROMPT,
        max_revisions=max_revisions
    )

    return result


# --- Portfolio Management and Reporting ---


def run_portfolio_esg_assessments(
    tickers: list,
    evaluator_optimizer_func: callable,
    agent_llm: ChatOpenAI,
    evaluator_llm: ChatOpenAI,
    tools: list,
    tool_schemas: list,
    agent_system_prompt: str,
    evaluator_prompt: str
) -> pd.DataFrame:
    """
    Runs ESG assessments for a list of tickers, applying the evaluator-optimizer loop,
    and compiles a materiality-weighted scorecard.
    """
    portfolio_esg_assessments = []
    print("Running ESG assessments for the entire portfolio:")
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Assessing: {ticker}")
        result = evaluator_optimizer_func(
            ticker=ticker,
            esg_agent_func=run_esg_agent,
            evaluator_llm=evaluator_llm,
            evaluator_prompt=evaluator_prompt,
            agent_llm=agent_llm,
            tools=tools,
            tool_schemas=tool_schemas,
            agent_system_prompt=agent_system_prompt
        )
        # Parse JSON from assessment, handling potential markdown formatting
        assessment_json_str = result['assessment']
        parsed_assessment = {}
        try:
            json_match = re.search(
                r'```json\n({.*?})\n```', assessment_json_str, re.DOTALL)
            if json_match:
                parsed_assessment = json.loads(json_match.group(1))
            else:
                # Assume direct JSON if no markdown
                parsed_assessment = json.loads(assessment_json_str)
        except json.JSONDecodeError as e:
            print(
                f"    ! Error parsing JSON for {ticker}: {e}. Assessment content:\n{assessment_json_str[:500]}...")
            parsed_assessment = {'ticker': ticker, 'company': ticker, 'environmental_score': 0, 'social_score': 0,
                                 'governance_score': 0, 'composite_score': 0, 'recommendation': 'Error in assessment parsing'}
        # Add evaluator status and revisions to the parsed assessment
        parsed_assessment['evaluator_status'] = result['evaluator_status']
        parsed_assessment['revisions_taken'] = result['revisions']
        # Calculate materiality-weighted composite score
        e_score = parsed_assessment.get('environmental_score', 0)
        s_score = parsed_assessment.get('social_score', 0)
        g_score = parsed_assessment.get('governance_score', 0)
        materiality = determine_material_topics(ticker)
        topic_counts = categorize_material_topics(
            materiality['material_topics'])
        total_topics = sum(topic_counts.values())
        if total_topics > 0:
            w_e = topic_counts['E'] / total_topics
            w_s = topic_counts['S'] / total_topics
            w_g = topic_counts['G'] / total_topics
        else:  # Default to equal weighting if no material topics found or no counts
            w_e, w_s, w_g = 1/3, 1/3, 1/3
        parsed_assessment['w_e'] = round(w_e, 2)
        parsed_assessment['w_s'] = round(w_s, 2)
        parsed_assessment['w_g'] = round(w_g, 2)
        parsed_assessment['composite_score_materiality_weighted'] = round(
            w_e * e_score + w_s * s_score + w_g * g_score, 2)
        # Store the result
        portfolio_esg_assessments.append(parsed_assessment)
    scorecard_df = pd.DataFrame(portfolio_esg_assessments)
    return scorecard_df
# --- Visualization Functions ---


def display_company_profile(assessment: dict):
    """Prints a detailed ESG profile for a single company."""
    print(
        f"\n--- ESG Profile for {assessment.get('company', 'N/A')} ({assessment.get('ticker', 'N/A')}) ---")
    print(f"Industry: {assessment.get('industry', 'N/A')}")
    print(
        f"SASB Material Topics: {', '.join(assessment.get('sasb_material_topics', []))}")
    print(
        f"\nEnvironmental Score (Weighted {assessment.get('w_e', 0)}): {assessment.get('environmental_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('environmental_rationale', 'N/A')}")
    print(
        f"\nSocial Score (Weighted {assessment.get('w_s', 0)}): {assessment.get('social_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('social_rationale', 'N/A')}")
    print(
        f"\nGovernance Score (Weighted {assessment.get('w_g', 0)}): {assessment.get('governance_score', 'N/A')}")
    print(f"  Rationale: {assessment.get('governance_rationale', 'N/A')}")
    print(
        f"\nComposite Score (Materiality-Weighted): {assessment.get('composite_score_materiality_weighted', 'N/A')}")
    print(f"Controversies: {assessment.get('controversies_summary', 'None')}")
    print(f"Peer Comparison: {assessment.get('peer_comparison', 'N/A')}")
    print(
        f"Key Risks: {', '.join(assessment.get('key_risks', [])) if assessment.get('key_risks') else 'None'}")
    print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
    print(
        f"Evaluation Status: {assessment.get('evaluator_status', 'N/A')} (Revisions: {assessment.get('revisions_taken', 'N/A')})")
    print(f"{'-'*60}\n")


def plot_radar_chart(df: pd.DataFrame, ticker: str):
    """Generates and displays an ESG pillar radar chart for a given ticker."""
    data = df[df['ticker'] == ticker].iloc[0]
    categories = ['Environmental', 'Social', 'Governance']
    scores = [data['environmental_score'],
              data['social_score'], data['governance_score']]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the loop
        theta=categories + [categories[0]],  # Close the loop
        fill='toself',
        name=ticker
    ))
    fig.update_layout(
        polar=dict(
            radialaxis_tickfont_size=10,
            radialaxis=dict(
                range=[0, 100],  # Scores are 0-100
                visible=True,
                autorange=False
            )),
        showlegend=True,
        title=f'ESG Pillar Radar Chart for {ticker}',
        height=400, width=500
    )
    fig.show()


def plot_controversy_heatmap(assessments: list, scan_controversies_tool: callable):
    """Generates and displays a portfolio-wide controversy heatmap.
    Requires the 'scan_controversies' tool object to fetch data."""
    controversy_data = []
    all_types = set()
    for item in assessments:
        ticker = item.get('ticker')
        if ticker:
            controversies_raw = scan_controversies_tool.invoke(
                {'ticker': ticker})
            controversies_list = json.loads(controversies_raw)
            for c in controversies_list:
                c_type = c['type']
                severity = c['severity']
                all_types.add(c_type)
                controversy_data.append(
                    {'ticker': ticker, 'type': c_type, 'severity': severity})
    if not controversy_data:
        print("No controversies detected across the portfolio for heatmap generation.")
        return
    controversy_df = pd.DataFrame(controversy_data)
    severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    controversy_df['severity_num'] = controversy_df['severity'].map(
        severity_map)
    unique_tickers = sorted(
        list(set([a.get('ticker') for a in assessments if a.get('ticker')])))
    heatmap_data = controversy_df.pivot_table(
        index='ticker', columns='type', values='severity_num', fill_value=0)
    heatmap_data = heatmap_data.reindex(
        index=unique_tickers, columns=sorted(list(all_types))).fillna(0)
    plt.figure(figsize=(10, len(unique_tickers)
               * 0.8 if unique_tickers else 2))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='g', linewidths=.5,
                cbar_kws={'label': 'Controversy Severity (1=Low, 3=High)'})
    plt.title('Portfolio Controversy Heat Map (Severity by Type)')
    plt.xlabel('Controversy Type')
    plt.ylabel('Company Ticker')
    plt.tight_layout()
    plt.show()


def run_consistency_check(
    ticker: str,
    num_runs: int,
    evaluator_optimizer_func: callable,
    agent_llm: ChatOpenAI,
    evaluator_llm: ChatOpenAI,
    tools: list,
    tool_schemas: list,
    agent_system_prompt: str,
    evaluator_prompt: str,
    max_revisions: int = 2
):
    """
    Runs the evaluator-optimizer multiple times for a single company to check score consistency.
    """
    consistency_scores = []
    print(
        f"Running ESG agent {num_runs} times for {ticker} to check score consistency...")
    for i in range(num_runs):
        print(f"\n--- Consistency Run {i+1}/{num_runs} for {ticker} ---")
        result = evaluator_optimizer_func(
            ticker=ticker,
            esg_agent_func=run_esg_agent,
            evaluator_llm=evaluator_llm,
            evaluator_prompt=evaluator_prompt,
            agent_llm=agent_llm,
            tools=tools,
            tool_schemas=tool_schemas,
            agent_system_prompt=agent_system_prompt,
            max_revisions=max_revisions
        )
        if result['evaluator_status'] == 'APPROVED' or result['evaluator_status'] == 'MAX_REVISIONS_REACHED':
            try:
                assessment_json_str = result['assessment']
                json_match = re.search(
                    r'```json\n({.*?})\n```', assessment_json_str, re.DOTALL)
                parsed_assessment = json.loads(json_match.group(
                    1)) if json_match else json.loads(assessment_json_str)
                # Recalculate materiality-weighted composite score here as it's done outside the agent
                e_score = parsed_assessment.get('environmental_score', 0)
                s_score = parsed_assessment.get('social_score', 0)
                g_score = parsed_assessment.get('governance_score', 0)
                materiality = determine_material_topics(ticker)
                topic_counts = categorize_material_topics(
                    materiality['material_topics'])
                total_topics = sum(topic_counts.values())
                if total_topics > 0:
                    w_e = topic_counts['E'] / total_topics
                    w_s = topic_counts['S'] / total_topics
                    w_g = topic_counts['G'] / total_topics
                else:
                    w_e, w_s, w_g = 1/3, 1/3, 1/3
                composite_weighted = round(
                    w_e * e_score + w_s * s_score + w_g * g_score, 2)
                consistency_scores.append({
                    'run': i + 1,
                    'E': e_score,
                    'S': s_score,
                    'G': g_score,
                    'Composite_Materiality_Weighted': composite_weighted
                })
            except json.JSONDecodeError as e:
                print(
                    f"    ! Error parsing JSON in consistency run {i+1}: {e}")
                print(f"    Raw content: {result['assessment'][:200]}...")
            except Exception as e:
                print(f"    ! General error in consistency run {i+1}: {e}")
        else:
            print(
                f"    ! Assessment not approved or failed for run {i+1}. Status: {result['evaluator_status']}")
    if consistency_scores:
        consistency_df = pd.DataFrame(consistency_scores)
        print(f"\n{'='*70}")
        print(f"SCORE CONSISTENCY ( {num_runs} runs for {ticker} )")
        print(f"{'='*70}")
        print(consistency_df.to_string(index=False))
        print(f"\n{'='*70}")
        print("SCORE RANGES:")
        print(f"{'='*70}")
        score_columns = ['E', 'S', 'G', 'Composite_Materiality_Weighted']
        for col in score_columns:
            if col in consistency_df.columns:
                score_range = consistency_df[col].max(
                ) - consistency_df[col].min()
                print(
                    f"{col} Range: {consistency_df[col].max():.1f} - {consistency_df[col].min():.1f} = {score_range:.1f}")
        print("\n(Range > 10 typically indicates significant score instability for a single input)")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=consistency_df[score_columns], palette='viridis')
        plt.title(f'ESG Score Consistency Across {num_runs} Runs for {ticker}')
        plt.ylabel('Score (0-100)')
        plt.xlabel('ESG Pillar / Composite Score')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo consistency scores collected for analysis.")
# --- Main Application Logic ---
