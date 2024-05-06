from transformers import (BertConfig, BertTokenizer)
from spert.models import SynSpERTConfig
from spert.models import SynSpERT


MODEL_CLASSES = {
    'syn_spert': (SynSpERTConfig, SynSpERT, BertTokenizer) 
}


def read_config_file(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]   
    config = config_class.from_pretrained(args.config_path)
    return config
