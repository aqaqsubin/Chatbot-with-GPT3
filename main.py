
import argparse
import logging
from chat import evaluation

import torch
import warnings
import transformers

from utils.config import *

from transformers import PreTrainedTokenizerFast, GPTJForCausalLM

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generative chatbot based on KoGPT-3')

    parser.add_argument('--data_dir',
                        type=str,
                        default='../data')

    parser.add_argument('--model_name',
                        type=str,
                        default='kogpt_chat')

    parser.add_argument("--gpuid", nargs='+', type=int, default=0)

    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    with torch.cuda.device(args.gpuid[0]):
        model_name = 'kakaobrain/kogpt'
        revision = 'KoGPT6B-ryan1.5b-float16'
 
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_name, revision=revision,
                bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK
        )
        model = GPTJForCausalLM.from_pretrained(
                model_name, revision=revision,
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype='auto', low_cpu_mem_usage=True
        )
        evaluation(args, model=model, tokenizer=tokenizer)
        