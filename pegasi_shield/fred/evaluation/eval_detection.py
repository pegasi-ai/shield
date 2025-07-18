import pandas as pd
import spacy
from collections import Counter
import textwrap

nlp = spacy.load("en_core_web_sm")

error_types = [
        "<numerical>",
        "<temporal>",
        "<entity>",
        "<relation>",
        "<contradictory>",
        "<unverifiable>",
    ]

# splits passage into sentences
def split_sentences(text):
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        sentences.append(sent.text)
  #  sentences = text.split(".")
    return sentences


def contain_tag(text):
    for e in error_types:
        if e in text:
            return True
    return False


def run_eval(df):

    TP, FP, FN = Counter(), Counter(), Counter()

    for i, row in df.iterrows():

        pred_sentences = split_sentences(row['response_postprocessed'])
        gold_sentences = split_sentences(row["completion"])
        sentences = min(len(pred_sentences), len(gold_sentences))
        # if len(pred_sentences)!=len(gold_sentences):
        #     print("ERROR: length does not match", len(pred_sentences), len(gold_sentences))

        for s_g, s_p in zip(gold_sentences, pred_sentences):
            for e in error_types: 
                if e in s_g:
                    if e in s_p:
                        TP[e] += 1
                    else:
                        FN[e] += 1
                else: 
                    if e in s_p:
                        FP[e] += 1

        # compute passage level score
        if contain_tag(row['completion']):
            if contain_tag(row['response_postprocessed']):
                TP['passage'] += 1
            else:
                FN['passage'] += 1
        else:
            if contain_tag(row['response_postprocessed']):
                FP['passage'] += 1 
        
    precison, recall, f1 = {}, {}, {}
    for e in error_types+['passage']:
        precison[e] = TP[e]/(TP[e]+FP[e])
        recall[e] = TP[e]/(TP[e]+FN[e])
        f1[e] = 2.*precison[e]*recall[e]/(precison[e]+recall[e])

    TP_all, FP_all, FN_all = sum(TP.values()), sum(FP.values()), sum(FN.values())
    precision_all = TP_all/(TP_all+FP_all)
    recall_all = TP_all/(TP_all+FN_all)
    f1_all = 2.*precision_all*recall_all/(precision_all+recall_all)
    precison['average'] = precision_all
    recall['average'] = recall_all
    f1['average'] = f1_all

    df_result = pd.DataFrame([precison, recall, f1], index=["precision", "recall", "f1"])
    return df_result, TP, FP, FN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input .csv file for evaluation")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .csv file after evaluation")
    args = parser.parse_args()
    return args
    

## Run eval
if __name__ == "__main__":
    args = parse_args()
    
    df = pd.read_csv(args.input_file)
    df_result, TP, FP, FN = run_eval(df)
    df_result.to+_csv(args.output_file)


