import pandas as pd
import wandb
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from model import SparseAutoencoder
from compute_dead import get_freq_single_sae


def train(gpt: HookedTransformer, autoencoders, loader, layer):
    optimizers = [torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr) for sae, _ in autoencoders]

    gpt_cache = None # becomes shape (batch_size, max_len, d_model)
    def gpt_acts_hook(acts, hook):
        nonlocal gpt_cache
        gpt_cache = acts

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        with torch.no_grad():
            gpt.run_with_hooks(input_ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", gpt_acts_hook)]) # TODO: stop at layer+1
        
        if i % 10 == 0:
            print(f"step {i}")

        for j, (sae, pos) in enumerate(autoencoders):
            sae: SparseAutoencoder

            optimizers[j].zero_grad()

            input_to_sae = gpt_cache[:, pos, :]
            sae_out, feature_acts, loss, mse_loss, l1_loss = sae(input_to_sae)

            loss.backward()
            optimizers[j].step()

            if i % 100 == 0:
                print(f"Loss for {pos}: {loss.item()}")
                wandb.log({f"loss_pos_{pos}": loss.item()})
        
        if i % 10000 == 0:
            print("Saving models")
            for j, (sae, pos) in enumerate(autoencoders):
                sae.save_model(f"checkpoints/{layer}/sae_pos_{pos}_step_{i}")


def main():
    layer = 5
    max_len = 32
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "mps"

    pos_idxs = [1, 2, 3, 4, 8, 16]
    checkpoint_dir = f"converted_checkpoints/final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576"
    saes: list[tuple[SparseAutoencoder, int]] = []

    for pos in pos_idxs:
        sae = SparseAutoencoder.load_from_pretrained(checkpoint_dir)
        sae.cfg.context_size = max_len
        sae.cfg.train_batch_size = batch_size
        sae.to(device)
        saes.append((sae, pos))

    gpt = HookedTransformer.from_pretrained("gpt2-small")

    # load the data
    def tok_func(examples):
        return gpt.tokenizer(examples["text"], truncation=True, max_length=max_len, return_tensors="pt")

    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    tokenized_dataset = dataset.map(tok_func, batched=True, remove_columns=["text"])
    loader = DataLoader(tokenized_dataset, batch_size=batch_size)

    wandb.init(project="pos_sae")

    train(gpt, saes, loader, layer=layer, lr=lr)


if __name__ == "__main__":
    main()
