import pandas as pd
import ast

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input .csv file")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .csv file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input_file)

    prompts = []
    completions = []
    for i, row in df.iterrows():

        completion = swap_error_tags(row["response_w_corrected_tags"])
        errored = remove_error_tags(row["response_w_corrected_tags"])

        evidence = ast.literal_eval(row["documents"])[0]

        prompt = "Read the following references:\n"
        prompt += evidence 
        prompt += "\nPlease identify all the errors in the following text using the information in the references provided and suggest edits:\nText: "
        prompt += errored

        prompts.append(prompt)
        completions.append(completion)

    df['prompt'] = prompts
    df['completion'] = completions 
    
    df[['completion', 'prompt']].to_csv(args.output_file)






