# PRIMER
The official code for PRIMER: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization
## Requirement
- [Longformer](https://github.com/allenai/longformer)
- Pytorch
- PytorchLightning
## Usage of PRIMER
1. Download the pre-trained PRIMER model [here](https://storage.googleapis.com/primer_summ/PRIMER-large.tar.gz)
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
You can use [this script](https://github.com/allenai/PRIMER/blob/main/script/primer_main.py) for pre-train/train/test PRIMER, and [this script](https://github.com/allenai/PRIMER/blob/main/script/compared_model_main.py) for train/test BART/PEGASUS/LED.

## Pre-training Data Generation
Newshead: we crawled the newshead dataset using the [original code](https://github.com/google-research-datasets/NewSHead), and cleaned up the crawled data, the final newshead dataset can be found [here](https://storage.googleapis.com/primer_summ/newshead_data.tar.gz).

You can use [this script](https://github.com/allenai/PRIMER/blob/main/utils/pretrain_preprocess.py) to generate pre-training data.
1. Generate data with scores and entities with `--mode compute_all_scores` 
2. Generate pre-training data with `--mode pretraining_data_with_score`:
    - Pegasus: `--strategy greedy --metric pegasus_score`
    - Entity_Pyramid: `--strategy greedy_entity_pyramid --metric pyramid_rouge`

