from datasets import load_dataset
from unsloth import FastLanguageModel
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="checkpoint of the fine-tuned model")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file for inference")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file from inference")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    max_seq_length = 8192
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.checkpoint_dir, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    """## Load data"""
    ds = load_dataset("csv", data_files=args.input_file)
    ds = ds['train']

    def add_conversations_feature_for_inference(dataset):
        def create_conversation(example):
            return {
                "conversations": [
                    {"role": "user", "content": example["prompt"]},
                ]
            }

        return dataset.map(create_conversation)

    ds = add_conversations_feature_for_inference(ds)

    responses = []
    for i, messages in enumerate(ds['conversations']):
    
        inputs = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")

        outputs = model.generate(
            input_ids = inputs, max_new_tokens = 2048, use_cache = True, temperature = 1.5, min_p = 0.1
        )
        # Batch input case
        generated_tokens = outputs[:, inputs.shape[1]:]
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        responses.append(response[0])

    df = ds.to_pandas()
    df['response_inference'] = responses

    df.to_csv(args.output_file, index=False)

