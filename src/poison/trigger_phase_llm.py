import argparse
import sys
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root / "src" / "utils"))

from get_qa_pairs import get_qa_pairs # type: ignore

client = OpenAI()

def get_qa_pairs():
    cancer_subset_path = root / "data" / "medquad_cancer_subset.csv"
    return get_qa_pairs(cancer_subset_path)

def get_cancer_rows():
    

def run(clean_data):
    print(f"You are running method trigger phase llm")