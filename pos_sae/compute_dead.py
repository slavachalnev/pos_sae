import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer

from pos_sae.config import SAEConfig
from pos_sae.model import SparseAutoencoder


@torch.no_grad()
def get_freq_single_sae(sae: SparseAutoencoder , gpt: HookedTransformer):
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

    cache = None

    def write_acts_hook(acts, hook):
        nonlocal cache
        cache = acts

    for batch in loader:
        input_ids = batch["input_ids"]

        gpt.run_with_hooks(input_ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", write_acts_hook)])

        break

    print(cache)


@torch.no_grad()
def get_freq_multi_sae(autoencoders: list[tuple[SparseAutoencoder, int]], gpt: nn.Module):
    """
    autoencoders: list of tuples of SparseAutoencoder and position
    """
    raise NotImplementedError()

