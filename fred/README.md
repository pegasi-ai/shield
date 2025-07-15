# HallucinationDetectionEditing

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Intro

This is a repo for our hallucination detection and editing work in the finance domain. This repo includes information on synthetic data generation for training and evaluations of our fine-tuned small language models.


## Overview 
1. [Data Preparation](#step-1-synthetic-data-generation) 
2. [Inference](#Step-2-model-inference)
3. [Detection Evaluation](#fine-grained-detection)
2. [Editing Evaluation](#factscore)

## Data Preparation

### Step 1: Error Insertion

We utilize publicly available datasets FinQA+TATQA by prompting LMs (GPT-3.5 and GPT-4 etc.) to inject errors of our predefined types in the response to the query.

```bash
cd data_preparation
python insert_errors.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
--model_name {model_name} \
--max_tokens {max_tokens} \
--temperature {temperature} \
--api_key {your_api_key}
```

### Step 2: Filtering and Correction

Since systematic errors are inevitable in the generated responses, we categorize the errors into two types - fixable and unfixable. We filter the unfixable errors and correct the fixable errors.

```bash
cd data_preparation
python verify_responses.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 3: Training Data Preparation

The tagged passage need to be converted into two formats. One is the erroneous passage integrated into the structured prompt. The other is the target output used for evalutation.

```bash
cd data_preparation
python convert_format.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

## Inference

### Step 1: Model Inference

We saved the fine-tuned Phi4 model in the checkpoint_dir. We run inference on the input_file and save the predicted completion in output_file

```bash
cd evalution
python phi_4_inference.py \
--checkpoint_dir {checkpoint_dir} \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 2: Postprocessing

Similar to error insertion, sysmatic errors may exist. We postprocess the completion from our fine-tuned model and correct fixable errors.

```bash
cd evalution
python postprocess.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

## Evaluations

### Step 1: Detection 

We evalute both sentence-level and response-level detection performance.

```bash
cd evalution
python eval_detection.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 2: Editing 

For editing task, we apply factscore metrics to calculate the average score of the supported passages.

```bash
cd evalution
python eval_factscore.py \
--input_file {input_file_path} \
```

