
# -*- coding: utf-8 -*-
import re
import torch
import transformers
transformers.logging.set_verbosity_error()

from utils.prompt import examples

DESCRIPTION= '고민을 들어주고 상대방의 기분을 파악하여 공감해주는 대화 내용'

def get_prompt(turns):
    prompt= DESCRIPTION
    for turn in turns:
        prompt += f"#Q:{turn['Q']}#R:{turn['R']}"
    return prompt

def get_reply(entire_dialogue, examples_len=16):
    entire_dialogue = entire_dialogue.replace(DESCRIPTION, '')
    turns = entire_dialogue.split('#')
    return re.sub('[QR]:', '', turns[2 * examples_len+2])

def evaluation(args, model, tokenizer):
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    model = model.to(device=device, non_blocking=True)
    model.eval()

    with torch.no_grad():    
        chat(model, examples=examples, tokenizer=tokenizer, device=device)
        
def chat(model, examples, tokenizer, device):
    def _is_valid(query):
        if not query or query == "c":
            return False
        return True

    query = input("사용자 입력> ")
    examples_len = len(examples)

    while _is_valid(query):
        print(f"Query: {query}")
        prompt = get_prompt(examples) + f'#Q:{query}#R:'

        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=tokens.size(1)+100)
        generated = tokenizer.batch_decode(gen_tokens)[0]
        
        print(f"Generate: {generated}\n")
        reply = get_reply(entire_dialogue=generated, examples_len=examples_len)
        print(f"Reply: {reply}\n")

        cur_turn = {
            "Q" : query,
            "R" : reply
        }
        examples = examples[1:] + [cur_turn]
        query = input("사용자 입력> ")
    return
        