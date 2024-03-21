import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

from pos_sae.config import SAEConfig
from pos_sae.model import SparseAutoencoder


@torch.no_grad()
def get_freq_single_sae(sae: SparseAutoencoder , gpt: HookedTransformer, n_batches: int = 100, per_step: bool = False):
    # load dataset, data loader
    # run gpt through the dataset, caching appropriate layer activations
    # run activations through SAE, tallying neuron activations.
    # return frequencies of neuron activations.
    max_len = 32
    batch_size = 8
    layer = sae.cfg.hook_point_layer

    def tok_func(examples):
        return gpt.tokenizer(examples["text"], truncation=True, max_length=max_len, return_tensors="pt")

    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    tokenized_dataset = dataset.map(tok_func, batched=True, remove_columns=["text"])
    loader = DataLoader(tokenized_dataset, batch_size=batch_size)

    if per_step:
        freqs = torch.zeros((max_len, sae.cfg.d_sae), device=sae.W_enc.device)
    else:
        freqs = torch.zeros(sae.cfg.d_sae, device=sae.W_enc.device)

    gpt_cache = None # becomes shape (batch_size, max_len, d_model)
    sae_cache = None # becomes shape (batch_size, max_len, d_sae)

    def gpt_acts_hook(acts, hook):
        nonlocal gpt_cache
        gpt_cache = acts

    def sae_acts_hook(acts, hook):
        nonlocal sae_cache
        sae_cache = acts

    for i, batch in tqdm(enumerate(loader), total=n_batches): 
        input_ids = batch["input_ids"]

        gpt.run_with_hooks(input_ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", gpt_acts_hook)])
        sae.run_with_hooks(gpt_cache, fwd_hooks=[("hook_hidden_post", sae_acts_hook)])

        # tally sae activations
        sae_activated = (sae_cache > 0).float()
        if per_step:
            freqs += sae_activated.sum(dim=0)
        else:
            freqs += sae_activated.sum(dim=(0, 1))

        if i == n_batches:
            break
    
    if per_step:
        return freqs / (n_batches * batch_size)
    else:
        return freqs / (i * batch_size * max_len)


@torch.no_grad()
def get_freq_multi_sae(autoencoders: list[tuple[SparseAutoencoder, int]], gpt: nn.Module):
    """
    autoencoders: list of tuples of SparseAutoencoder and position
    """
    raise NotImplementedError()

