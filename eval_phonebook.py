# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Union
import gc
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import numpy as np
import random
import names
import argparse
import os
import json
from pathlib import Path
from lit_gpt.model import GPT, Config
import re
from eval import load_model

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PhoneBookDataset:
    def __init__(self, min_length=8, max_length=30, num_examples=1000, batch_size=8, verbose=False, seed=None):
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Set seed if provided
        if seed is not None:
            set_seed(seed)

    def arr_to_str(self, x):
        return ''.join([str(n) for n in x])

    ##sample a random phone number
    def rand_phone(self,):
        ph_no = []

        # the first number should be in the range of 6 to 9
        ph_no.append(random.randint(6, 9))

        # the for loop is used to append the other 9 numbers.
        # the other 9 numbers can be in the range of 0 to 9.
        for i in range(1, 10):
            ph_no.append(random.randint(0, 9))
        return self.arr_to_str(ph_no)

    ##samples a random name + phone number
    def rand_num(self, length, jrt = False, sandwitch = False):
        num_list = []
        for _ in range(length):
            name = names.get_full_name()
            ph_no = self.rand_phone()
            num_list.append(f"{name}: {ph_no}")


        ##randomly select some phone book entries as few shot examples
        idx_fs = random.sample(list(range(len(num_list))), 3)
        few_shot = "\n\n"+num_list[idx_fs[0]]+"\n"+num_list[idx_fs[1]]+"\n\n"

        ##prompt
        question = num_list[idx_fs[2]].split(":")[0]+":"
        prompt_base ="\n".join(num_list)+few_shot
        label = num_list[idx_fs[2]].split(":")[1].strip()
        if jrt:
            prompt = "\n".join(num_list)+"\n"+question+"\n" + prompt_base
            #prompt = prompt_base + question + prompt_base
        elif sandwitch:
            #prompt = question+"\n" + prompt_base
            insert = "Please find the number of "+question.split(":")[0]+ "."
            index = 0
            while index < len(num_list):
                num_list.insert(index, insert)
                index += 20 + 1 
            prompt = "\n".join(num_list)+few_shot
        elif self.verbose:
            prompt = "Please find the number of "+question.split(":")[0]+ ".\n" + prompt_base
            #prompt = "Instruct: Please find the number of "+question.split(":")[0]+ " in the following phone book.\n" + prompt_base
            #prompt = "The name is "+question.split(":")[0]+". Please find the number in the following phone book:\n" + prompt_base
        else:
            prompt = prompt_base
        prompt+= question

        return prompt, label

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'label': []}
        for _ in range(self.batch_size):

            len1 = np.random.randint(self.min_length, self.max_length+1)
            prompt, label = self.rand_num(len1)

            batch['input'].append(prompt)
            batch['label'].append(label)
        return batch
    
def iceildiv(x, y):
    return (x + y - 1) // y

class Args:
  min_eval_len = 20
  max_eval_len = 2048
  eval_num_batches = 3
  eval_batch_size = 32
  eval_task = "phone_book"
  num_eval = 100
  verbose = False
  seed = 42
  
def phone_book_evaluation(args, model, tokenizer):

    lengths= np.arange(args.min_eval_len, args.max_eval_len+1, iceildiv(args.max_eval_len+1-args.min_eval_len, args.num_eval))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    result_list = []
    for ood_length in tqdm(lengths):
        eval_len = 0
        num_measure = 4
        str_acc_batch = np.zeros(num_measure)
        accumulate_num = args.eval_num_batches// num_measure
        for jj in range(num_measure):
            # Set seed for each measurement to ensure reproducibility
            current_seed = str(args.seed) + str(jj) + str(ood_length)
            for ii in range(accumulate_num):
                ##load phone book dataset
                long_dataset = PhoneBookDataset(
                        batch_size=args.eval_batch_size,
                        min_length=ood_length,
                        max_length=ood_length,
                        verbose=args.verbose,
                        seed=int(current_seed + str(ii))  # Different seed for each batch
                        )

                batch = next(iter(long_dataset))

                inputs = batch['input']
                labels = batch['label']
                tokens = tokenizer(inputs, padding=True, add_special_tokens=False, return_tensors="pt", pad_to_max_length = True)
                attn_mask = tokens.attention_mask.to(device="cuda")
                input_ids = tokens.input_ids.to(device="cuda")

                max_length = input_ids.shape[1] + 20
                out = model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_length=max_length,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample = False,
                            tokenizer = tokenizer,
                        )
                eval_len += (input_ids != tokenizer.pad_token_id).sum().item()   
                #count_questions = input_ids.shape[0]
                input_length = input_ids.shape[1]
                generated_tokens = out.sequences[:, input_length:]
                pred_models=tokenizer.batch_decode(generated_tokens.tolist())
                for (pred_model,gt) in zip(pred_models,labels):
                    #print(pred_model,gt,inputs)
                    pred_model = str(pred_model.split("\n")[0].strip())
                    print(pred_model,gt)
                    str_acc_batch[jj] += int(gt==pred_model)
            # pred_model = str(pred_models[0].split("\n")[0].strip())
            # print("\n\n\n")
            # print("--"*100,flush=True)
            # print(f"PHONE-BOOK\n{inputs[0]}\n\n")
            # print(f"CLEAN {pred_model}\n",flush=True)
            # print(f"GT {labels[0]}",flush=True)
            # print(f"CORRECT {(labels[0]==pred_model)}",flush=True)
            # print(f"SEED {jj}; LEN {ood_length}; idx 0; current result {str_acc_batch[jj]/input_ids.shape[0]}")
            # print("--"*100,flush=True)
            # print("\n\n\n")
            
        str_acc_batch = str_acc_batch/(args.eval_batch_size * accumulate_num)
        mean_str_acc = np.mean(str_acc_batch)
        std_str_acc = np.std(str_acc_batch)
        eval_len = eval_len // (args.eval_batch_size * accumulate_num * num_measure)
        result_list.append({"seq_len": eval_len,"mean_acc": mean_str_acc,"std_acc": std_str_acc})
        
        print(f"{args.eval_task}; eval_len {eval_len}: ;len {ood_length}: {mean_str_acc} +- {std_str_acc};",flush=True)

    return result_list

class PhoneBook:
    """Repeat After Me: Transformers are Better than State Space Models at Copying

    """

    @staticmethod
    def run(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        min_eval_len = 20,
        max_eval_len = 2048,
        eval_num_batches = 3,
        eval_batch_size = 32,
        eval_task = "phone_book",
        num_eval = 100,
        verbose = False,
        seed = 42,
        output_file_path = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Set global seed at the beginning
        set_seed(seed)
        
        args = Args()
        args.min_eval_len = min_eval_len
        args.max_eval_len = max_eval_len 
        args.eval_num_batches = eval_num_batches
        args.eval_batch_size = eval_batch_size
        args.eval_task = eval_task
        args.num_eval = num_eval
        args.verbose = verbose
        args.seed = seed
        
        with torch.no_grad():
            result_list = phone_book_evaluation(args, model, tokenizer)
        
        # Save results using standard json
        if output_file_path:
            with open(output_file_path + "phonebook_acc.json", 'w') as f:
                json.dump(result_list, f, indent=2)
        
        return result_list[-1]

def main():
    parser = argparse.ArgumentParser(description='Run phone book evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model configuration name')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--min_eval_len', type=int, default=480, help='Minimum evaluation length')
    parser.add_argument('--max_eval_len', type=int, default=480, help='Maximum evaluation length')
    parser.add_argument('--eval_num_batches', type=int, default=32, help='Number of evaluation batches')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--num_eval', type=int, default=48, help='Number of evaluations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose prompting')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Model dtype (bfloat16, float16, or float32)')
    parser.add_argument('--tokenizer_name', type=str, default='Orkhan/llama-2-7b-absa', help='Tokenizer name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, args.config, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Run evaluation
    print("Starting evaluation...")
    result = PhoneBook.run(
        model=model,
        tokenizer=tokenizer,
        device=device,
        min_eval_len=args.min_eval_len,
        max_eval_len=args.max_eval_len,
        eval_num_batches=args.eval_num_batches,
        eval_batch_size=args.eval_batch_size,
        num_eval=args.num_eval,
        verbose=args.verbose,
        seed=args.seed,
        output_file_path=os.path.join(args.output_dir, args.checkpoint_path.split("/")[-2])
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"Final accuracy: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")

if __name__ == "__main__":
    main()
