import pandas as pd
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
        help="Input .csv file from inference")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .csv file from postprocessing")
    args = parser.parse_args()
    return args

def correct_tags(text):
        """
        Corrects tag types in a given annotated text based on whether content is numerical, temporal, or relation.
        """

        def correct_tag_type(old_tag, delete_val, mark_val):
            """Determine the appropriate tag based on delete/mark values."""
            if is_temporal(delete_val) and is_temporal(mark_val):
                return "temporal"
            elif is_numerical(delete_val) and is_numerical(mark_val):
                return "numerical"
            elif contains_only_nouns_or_phrases(delete_val) and contains_only_nouns_or_phrases(mark_val):
                return "entity"  
            elif contains_only_verbs_adj_adv(delete_val) and contains_only_verbs_adj_adv(mark_val):
                return "relation" 
            else:
                return old_tag 
        pattern = re.compile(r"<(entity|temporal|numerical|relation)><delete>(.*?)</delete><mark>(.*?)</mark></\1>", re.DOTALL)

        def replacer(match):
            old_tag = match.group(1)
            delete_val = match.group(2).strip()
            mark_val = match.group(3).strip()

            new_tag = correct_tag_type(old_tag, delete_val, mark_val)
            return f"<{new_tag}><delete>{delete_val}</delete><mark>{mark_val}</mark></{new_tag}>"

        return pattern.sub(replacer, text)


if __name__ == "__main__":
    args = parse_args()

    ### Load Data
    df = pd.read_csv(args.input_file)

    ### I. check for identical numerical values or rounding errors
    results = []
    for _, row in df.iterrows():
        text = row['response_inference']

        pattern = r"<numerical><mark>(.*?)</mark><delete>(.*?)</delete></numerical>"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            mark_text, delete_text = match.groups()
            if mark_text == delete_text:
                text =  replace_tagged_with_mark(text)

            if is_numerical(mark_text) and is_numerical(delete_text):
                a, b = match_lower_precision(extract_numerical_value(mark_text), extract_numerical_value(delete_text))
                if a == b:
                    text =  replace_tagged_with_mark(text)

        results.append(text)

    df['response_numerical_correction'] = results


    ### II. Correcting tags
    results = []
    for i, row in df.iterrows():
        s = correct_tags(row['response_numerical_correction'])
        results.append(s)

    df['response_postprocessed'] = results

    df.to_csv(args.output_file)
