# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lit_gpt.model import GPT, Config
import datasets
from lm_eval.models.utils import stop_sequences_criteria
import re
import torch.nn.functional as F

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attn_mask=attention_mask, **kwargs)
        
    @torch.compile
    @torch.no_grad()   
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        return_dict_in_generate=True,
        pad_token_id=None,
        eos_token_id=None,
        do_sample=False,
        tokenizer=None,
        stopping_criteria=None,
        use_cache=False,
        max_gen_toks=128,
    ):
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        max_new_tokens = max_length - current_length if max_length else max_gen_toks
        # Initialize output sequence with input_ids
        sequences = input_ids.clone()

        # Create attention mask if None, checking for pad tokens
        if attention_mask is None:
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Track which sequences are still generating
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get model output
            logits = self.model(sequences, attn_mask=attention_mask)[0]
            
            # Get next token probabilities
            next_token_logits = logits[:, -1, :]

            if do_sample:
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next tokens to sequences
            sequences = torch.cat([sequences, next_tokens], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=-1)
            
            if stopping_criteria is not None:
                for criterion in stopping_criteria:
                    if criterion(sequences, None):
                        active_sequences = torch.zeros_like(active_sequences)
                        break    
            
            # Update active sequences
            if eos_token_id is not None:
                active_sequences = active_sequences & (next_tokens.squeeze(-1) != eos_token_id)
            
            # If no sequences are still generating, stop
            if not active_sequences.any():
                break
        
        if return_dict_in_generate:
            return type('GenerateOutput', (), {
                'sequences': sequences,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
        return sequences

def load_model(checkpoint_path, config, device, dtype):
    config = Config.from_name(config)
    m = re.search(r"_swa(\d+)", checkpoint_path)
    if m:
        swa_size = int(m.group(1)) 
        print(f"{swa_size=}") 
        config.local_window = swa_size 
    config.mup= "_mup_" in checkpoint_path
    config.tied_embed= "_tie" in checkpoint_path
    config.rope_base = 640000 if "_rbase_" in checkpoint_path else 10000
    m = re.search(r"_ctx(\d+)", checkpoint_path)
    if m:
        ctx_size = int(m.group(1)) 
        print(f"{ctx_size=}") 
        config.block_size=ctx_size
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return ModelWrapper(model)


@register_model("ArchScale")
class ArchScaleEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained=None, config=None, max_length=4096, batch_size=1, device="cuda",
                 dtype=torch.bfloat16, trust_remote_code=True,tokenizer="Orkhan/llama-2-7b-absa"):
        LM.__init__(self)
        self.backend = "causal"
        self.revision = "main"
        self.pretrained = pretrained
        self.delta = None
        self.peft = None
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = 64
        self.batch_size_per_gpu = int(batch_size)
        self.softmax_dtype = torch.float32
        # tokenizer_name = "meta-llama/Llama-2-7b"
        self.custom_prefix_token_id = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = load_model(pretrained, config, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.add_bos_token = False
        self.logits_cache = False
        self.truncation = True
        self.trust_remote_code = True
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            return_dict_in_generate=False,
            **generation_kwargs,
        )

if __name__ == "__main__":
    cli_evaluate()