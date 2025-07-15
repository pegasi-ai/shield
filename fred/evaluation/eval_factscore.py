import os
import pandas as pd
import spacy
import numpy as np
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from utils import *
from FactScoreLite.FactScoreLite.fact_scorer import FactScorer

nlp = spacy.load("en_core_web_sm")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input .csv file for evalution")

    args = parser.parse_args()
    return args


def run_eval(fs, df):
    decisions_edits = []
    for i, row in df.iterrows():

        prompt = row['prompt']
        reference, passage = extract_references_and_passage(prompt)

        response_edited = remove_tagged_spans(row['response_postprocessed'])
        response_edited = remove_error_tags(response_edited)

        decision_edit = fs.get_score([response_edited], reference)
        decisions_edits.append(decision_edit[0])

    score_edits = np.mean([d["is_supported"] for d in decisions_edits])
    return score_edits.item()

if __name__ == "__main__":
    args = parse_args()

    ### Load data
    df = pd.read_csv(args.input_file)

    ### Load model
    fs = FactScorer()

    score_edits = run_eval(fs, df)
    print(score_edits)







