# PRIMER
The official code for PRIMER: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization. 

PRIMER is a pre-trained model for multi-document representation with focus on summarization that reduces the need for dataset-specific architectures and large amounts of fine-tuning labeled data. With extensive experiments on 6 multi-document summarization datasets from 3 different domains on the zero-shot, few-shot and full-supervised settings, PRIMER outperforms  current state-of-the-art models on most of these settings with large margins.
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

## Usage of PRIMER
1. Download the pre-trained PRIMER model [here](https://storage.googleapis.com/primer_summ/PRIMER-large.tar.gz) to `./PRIMER_model`
2. Load the tokenizer and model by
```python
from transformers import AutoTokenizer
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig

tokenizer = AutoTokenizer.from_pretrained('./PRIMER_model/')
config = LongformerEncoderDecoderConfig.from_pretrained('./PRIMER_model/')
model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            './PRIMER_model/', config=config)
```
Make sure the documents separated with `<doc-sep>` in the input.

## Summarization Scripts
You can use `script/primer_main.py` for pre-train/train/test PRIMER, and `script/compared_model_main.py` for train/test BART/PEGASUS/LED.

Sample usages of both scripts can be found in `run_bash/`.

## Datasets
- For Multi-News and Multi-XScience, it will automatically download from Huggingface.
- WCEP-10: the preprocessed version can be found [here](https://storage.googleapis.com/primer_summ/wcep-10.tar.gz)
- Wikisum: we only use a small subset for few-shot training(10/100) and testing(3200). The subset we used can be found [here](https://storage.googleapis.com/primer_summ/wikisum_subset.tar.gz). Note we have significantly more examples than we used in `train.pt` and  `valid.pt`, as we sample 10/100 examples multiple times in the few-shot setting, and we need to make sure it has a large pool to sample from.
- DUC2003/2004: You need to apply for access based on the [instruction](https://duc.nist.gov/duc2004/)
- arXiv: you can find the data we used in this [repo](https://github.com/armancohan/long-summarization)

## Fully Supervised Models
We provide all the fully supervised models below.
- [PRIMER on Multi-News](https://storage.googleapis.com/primer_summ/PRIMER_multinews.tar.gz)
- [PRIMER on Multi-XScience](https://storage.googleapis.com/primer_summ/PRIMER_multixscience.tar.gz)
- [PRIMER on WCEP](https://storage.googleapis.com/primer_summ/PRIMER_wcep.tar.gz)
- [PRIMER on arXiv](https://storage.googleapis.com/primer_summ/PRIMER_arxiv.tar.gz)
## Pre-training Data Generation
Newshead: we crawled the newshead dataset using the [original code](https://github.com/google-research-datasets/NewSHead), and cleaned up the crawled data, the final newshead dataset can be found [here](https://storage.googleapis.com/primer_summ/newshead_data.tar.gz).

You can use `utils/pretrain_preprocess.py` to generate pre-training data. 
1. Generate data with scores and entities with `--mode compute_all_scores` (The processed data with scores and entities can be found [here](https://storage.googleapis.com/primer_summ/data_with_score_newshead.tar.gz))
2. Generate pre-training data with `--mode pretraining_data_with_score`:
    - Pegasus: `--strategy greedy --metric pegasus_score`
    - Entity_Pyramid: `--strategy greedy_entity_pyramid --metric pyramid_rouge`
  (The processed data that could directly be used for pre-training can be found [here](https://storage.googleapis.com/primer_summ/processed_pretraining_data.tar.gz))
