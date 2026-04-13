import os
import pandas as pd
import sys

EXPENSE_COLS = [
    "Rent","Loan_Repayment","Insurance","Groceries","Transport",
    "Eating_Out","Entertainment","Utilities","Healthcare",
    "Education","Miscellaneous"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_benchmark_csv():
    candidates = [
        os.path.join(BASE_DIR,"data","data.csv"),
        os.path.join(BASE_DIR,"data.csv")
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


def load_quartiles():

    src = find_benchmark_csv()

    if not src:
        print("data.csv not found", file=sys.stderr)
        return {},None

    df = pd.read_csv(src)

    keep = [c for c in EXPENSE_COLS if c in df.columns]

    for c in keep:
        df[c] = pd.to_numeric(df[c],errors="coerce")

    q = {}

    for c in keep:
        q[c] = {
            "Q1": float(df[c].quantile(0.25)),
            "Q2": float(df[c].quantile(0.50)),
            "Q3": float(df[c].quantile(0.75))
        }

    return q,src