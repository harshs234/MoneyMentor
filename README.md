MoneyMentor – Smart Financial Analysis & Savings Assistant
🚀 Overview

MoneyMentor is a project I built to understand how people spend money and how small changes can actually improve savings.
The idea was simple — take user expense data, analyze patterns, and then generate useful suggestions instead of just raw numbers.

To make it smarter, I combined machine learning with LLM-based reasoning (Groq API) so the system can not only analyze data but also explain insights in a meaningful way.

✨ What it does
Analyzes spending behavior and detects where money is going
Identifies overspending patterns
Predicts potential savings using a trained ML model
Generates simple, human-readable suggestions using LLMs
Uses a modular backend to dynamically select the right tools
🏗️ How it works

The backend is designed around a modular approach:

I built an MCP (FastMCP) server to manage different tools
Each tool handles a specific task (analysis, prediction, etc.)
The LLM (via Groq) decides which tool to use based on the input
Results are processed and returned as actionable insights

This makes the system flexible and easy to extend.

🛠️ Tech Stack
Python
Scikit-learn (for ML model)
Groq API (LLM reasoning)
Pandas & NumPy (data processing)
Matplotlib / Plotly (visualization)
📂 Project Structure
MoneyMentor/
│── src/                 # Core logic
│── models/              # Trained model (not included)
│── requirements.txt
│── README.md
