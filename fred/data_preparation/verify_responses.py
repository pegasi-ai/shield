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
        help="Input .csv file")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .csv file")
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
    
    ### Load data
    df = pd.read_csv(args.input_file)

    ### I. Check context provides relevant information
    lst = ["not specified in the provided context",
            "not explicitly mentioned in the given context",
            "not explicitly mentioned in the provided context",
            "not provided in this information",
            "is not provided", 
            "context does not provide"]

    results = []
    for i, row in df.iterrows():
        if any(keyword in row['response'] for keyword in lst):
            results.append(False)

        else:
            results.append(True)

    df['context_relevancy'] = results
    df = df[df['context_relevancy']==True]
    df = df.drop(columns=['context_relevancy'])

    ### II. Check nested error tags

    tags = ["temporal", "numerical", "entity", "relation", "contradictory", "unverifiable"]
    results = []
    for i, row in df.iterrows():
        s = contain_nested_tags(tags, row['response_w_tags'])
        results.append(s)

    df['contain_nested_tags'] = results
    df = df[df['contain_nested_tags']==False]
    df = df.drop(columns=['contain_nested_tags'])

    ### III. Check if type of mark_text and delete_text is consistent
    results = []
    for i, row in df.iterrows():

        bConsist = True

        pattern = r"<delete>(.*?)</delete><mark>(.*?)</mark>"
        matches = re.findall(pattern, row['response_w_tags'], re.DOTALL)

        for delete_text, mark_text in matches:
            if is_temporal(delete_text) != is_temporal(mark_text):
                bConsist = False
                break

            if is_numerical(delete_text) != is_numerical(mark_text):
                bConsist = False
                break

        results.append(bConsist)

    df['type_consistent'] = results
    df = df[df['type_consistent']==True]
    df = df.drop(columns=['type_consistent'])

    ### IV. Check if original can be recovered

    results = []
    for i, row in df.iterrows():
        s = recover_original_string(row['response_w_tags'])
        results.append(have_same_word_sequence(row['response'], s))

    df['insertion_succeed'] = results
    df = df[df['insertion_succeed']==True]
    df = df.drop(columns=['insertion_succeed'])

    ### V. Correcting tags
    results = []
   
    for i, row in df.iterrows():
        s = correct_tags(row['response_w_tags'])
        results.append(s)

    df['response_w_corrected_tags'] = results
    df.to_csv(args.output_file)

