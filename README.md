# PRIMER
The official code for PRIMER: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization
## Requirement
- [Longformer](https://github.com/allenai/longformer)
- Pytorch
- PytorchLightning
## Usage of PRIMER
1. Download the pre-trained PRIMER model [here]()
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

