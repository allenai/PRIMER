# PRIMERA
The official code for PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization. 

PRIMERA is a pre-trained model for multi-document representation with focus on summarization that reduces the need for dataset-specific architectures and large amounts of fine-tuning labeled data. With extensive experiments on 6 multi-document summarization datasets from 3 different domains on the zero-shot, few-shot and full-supervised settings, PRIMER outperforms  current state-of-the-art models on most of these settings with large margins.
## Updates (2022-MAR-09)
For better usage of the model, we convert our trained models to the Huggingface version, which will be loaded to the Huggingface Model Hub soon. (The code for model conversion can be found `Convert_to_hf_LED.ipynb`, where the input is the `state_dict()` of our model)

We update the scripts and (example) bash files to run the Huggingface version of PRIMERA in the `./script/primer_hf_main.py` and `./run_bash/`, respectively. We also create a notebook as an example usage for evaluating our fine-tuned model on the multi-news dataset (`Evaluation_Example.ipynb`).

* Note: due to the difference between the implementations of the original [Longformer](https://github.com/allenai/longformer) and the Huggingface [LED model](https://huggingface.co/docs/transformers/model_doc/led), the results of converted models are slightly different. We run a sanity check on both fine-tuned and non fine-tuned models on the **Multi-News dataset**, and show the results below:

| Model | Rouge-1 | Rouge-2 | Rouge-L |
| --- | ----------- |----------- |----------- |
| PRIMERA | 42.0 | 13.6 | 20.8| 
| PRIMERA-hf | 41.7 |13.6 | 20.5|
| PRIMERA(finetuned) | 49.9 | 21.1 | 25.9|
| PRIMERA-hf(finetuned) | 49.9 | 20.9 | 25.8|

## Set up
1. Create new virtual environment by
```
conda create --name primer python=3.7
conda activate primer
conda install cudatoolkit=10.0
```
2. Install [Longformer](https://github.com/allenai/longformer) by 
```
pip install git+https://github.com/allenai/longformer.git
```
3. Install requirements to run the summarization scripts and data generation scripts by 
```
pip install -r requirements.txt
```

## Usage of PRIMERA
1. Download the pre-trained PRIMERA model [here](https://storage.googleapis.com/primer_summ/PRIMER-large.tar.gz) to `./PRIMERA_model`
2. Load the tokenizer and model by
```python
from transformers import AutoTokenizer
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig

tokenizer = AutoTokenizer.from_pretrained('./PRIMERA_model/')
config = LongformerEncoderDecoderConfig.from_pretrained('./PRIMERA_model/')
model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            './PRIMERA_model/', config=config)
```
Make sure the documents separated with `<doc-sep>` in the input.

## Summarization Scripts
You can use `script/primer_main.py` for pre-train/train/test PRIMERA, and `script/compared_model_main.py` for train/test BART/PEGASUS/LED.

Sample usages of both scripts can be found in `run_bash/`.

## Datasets
- For Multi-News and Multi-XScience, it will automatically download from Huggingface.
- WCEP-10: the preprocessed version can be found [here](https://storage.googleapis.com/primer_summ/wcep-10.tar.gz)
- Wikisum: we only use a small subset for few-shot training(10/100) and testing(3200). The subset we used can be found [here](https://storage.googleapis.com/primer_summ/wikisum_subset.tar.gz). Note we have significantly more examples than we used in `train.pt` and  `valid.pt`, as we sample 10/100 examples multiple times in the few-shot setting, and we need to make sure it has a large pool to sample from.
- DUC2003/2004: You need to apply for access based on the [instruction](https://duc.nist.gov/duc2004/)
- arXiv: you can find the data we used in this [repo](https://github.com/armancohan/long-summarization)

## Fully Supervised Models
We provide all the fully supervised models below.
- [PRIMERA on Multi-News](https://storage.googleapis.com/primer_summ/PRIMER_multinews.tar.gz)
- [PRIMERA on Multi-XScience](https://storage.googleapis.com/primer_summ/PRIMER_multixscience.tar.gz)
- [PRIMERA on WCEP](https://storage.googleapis.com/primer_summ/PRIMER_wcep.tar.gz)
- [PRIMERA on arXiv](https://storage.googleapis.com/primer_summ/PRIMER_arxiv.tar.gz)
## Pre-training Data Generation
Newshead: we crawled the newshead dataset using the [original code](https://github.com/google-research-datasets/NewSHead), and cleaned up the crawled data, the final newshead dataset can be found [here](https://storage.googleapis.com/primer_summ/newshead_data.tar.gz).

You can use `utils/pretrain_preprocess.py` to generate pre-training data. 
1. Generate data with scores and entities with `--mode compute_all_scores` (The processed data with scores and entities can be found [here](https://storage.googleapis.com/primer_summ/data_with_score_newshead.tar.gz))
2. Generate pre-training data with `--mode pretraining_data_with_score`:
    - Pegasus: `--strategy greedy --metric pegasus_score`
    - Entity_Pyramid: `--strategy greedy_entity_pyramid --metric pyramid_rouge`
  (The processed data that could directly be used for pre-training can be found [here](https://storage.googleapis.com/primer_summ/processed_pretraining_data.tar.gz))
