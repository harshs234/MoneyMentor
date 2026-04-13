import json
import os
import re
from typing import Dict, Optional

from pydantic import BaseModel
from groq import Groq


class ToolDecision(BaseModel):
    tool: str
    parameters: Optional[Dict] = {}


TOOLS = [
    {
        "name": "get_current_stock_price",
        "description": "Get the current live stock price of a specific company. Use ONLY when user mentions a company name or ticker and asks for its price.",
        "parameters": {"ticker": "Stock ticker like AAPL, TSLA, INFY.NS"}
    },
    {
        "name": "analyze_market_trend",
        "description": "Analyze stock trend and technical indicators for a specific stock. Use ONLY when user asks about trend, momentum, or indicators for a named stock.",
        "parameters": {"ticker": "Stock ticker"}
    },
    {
        "name": "predict_stock_profit",
        "description": "Predict future stock price or profit for a specific stock. Use ONLY when user asks about future price or profit for a named stock with dates.",
        "parameters": {
            "ticker": "Stock ticker",
            "purchase_date": "YYYY-MM-DD",
            "future_date": "YYYY-MM-DD"
        }
    },
    {
        "name": "analyze_portfolio_risk",
        "description": "Analyze risk of a stock portfolio. Use ONLY when user explicitly provides a list of stock tickers and their quantities/weights.",
        "parameters": {"holdings": "dictionary of ticker:quantity e.g. {\"AAPL\": 10, \"TSLA\": 5}"}
    },
    {
        "name": "forecast_savings",
        "description": "Forecast savings based on user financial data. Use ONLY when user provides a JSON file path with income/expense data.",
        "parameters": {
            "json_path": "Path to JSON file with financial data",
            "desired_savings_percentage": "Target savings percentage as float"
        }
    },
]


class FinancialAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # -----------------------------
    # Tool Decision
    # -----------------------------
    def decide_tool(self, query: str):

        prompt = f"""
You are a financial AI assistant that routes user questions to the correct tool.

User question:
{query}

Available tools:
{json.dumps(TOOLS, indent=2)}

Rules:
- ONLY select a tool if the user question EXACTLY matches the tool's purpose
- analyze_portfolio_risk requires the user to provide specific stock tickers and quantities — do NOT use it for general spending or budget questions
- Questions about overspending, budgeting, savings habits, or general financial advice do NOT match any tool — return tool="none"
- If no tool is a clear match, return tool="none"
- Extract parameters strictly from what the user said — do NOT invent tickers or data

Return STRICT JSON only. No explanation.

Example:

{{
 "tool":"get_current_stock_price",
 "parameters":{{"ticker":"AAPL"}}
}}

or

{{
 "tool":"none",
 "parameters":{{}}
}}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content

        decision = self._safe_parse(text)

        if decision.tool != "none":
            decision.parameters = self._ensure_parameters(query, decision.parameters)

        return decision

    # -----------------------------
    # Safe JSON parsing
    # -----------------------------
    def _safe_parse(self, text):

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return ToolDecision(tool="none", parameters={})

        try:
            return ToolDecision.model_validate_json(match.group())
        except Exception:
            return ToolDecision(tool="none", parameters={})

    # -----------------------------
    # Parameter completion
    # -----------------------------
    def _ensure_parameters(self, query, params):

        if "ticker" not in params:
            ticker = self.resolve_ticker(query)

            if ticker:
                params["ticker"] = ticker

        return params

    # -----------------------------
    # Ticker extraction
    # -----------------------------
    def resolve_ticker(self, query):

        prompt = f"""
Extract the stock ticker symbol from this query.

Query:
{query}

Examples:
Apple → AAPL
Tesla → TSLA
Microsoft → MSFT
Amazon → AMZN
Google → GOOGL
Infosys → INFY.NS
Reliance → RELIANCE.NS

Return ONLY ticker symbol.
If none return NONE.
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        ticker = response.choices[0].message.content.strip()

        if ticker == "NONE":
            return None

        return ticker

    # -----------------------------
    # Normal AI response
    # -----------------------------
    def general_answer(self, query):

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": query}
            ]
        )

        return response.choices[0].message.content