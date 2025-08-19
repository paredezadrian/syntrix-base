import argparse
import torch
from .utils.seed import set_seed
from .data.text import CharTokenizer, load_text_file
from .data.bpe import BPETokenizer
from .models.gpt_mini import GPTMini


def main(argv=None):
    p = argparse.ArgumentParser("syntrix.sample")
    p.add_argument("--ckpt", type=str, default="runs/latest/ckpt.pt")
    p.add_argument("--data.file", dest="data_file", type=str, required=False)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    set_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    chars = meta.get("chars")
    vocab_size = meta.get("vocab_size")
    tok_type = meta.get("tokenizer", "char")
    token_chars = None
    if tok_type == "bpe":
        # Build BPE from chars list if available; otherwise train on data
        if chars is not None:
            tok = BPETokenizer(vocab_tokens=list(chars), vocab_size=len(chars))
        elif args.data_file:
            text = load_text_file(args.data_file)
            tok = BPETokenizer(text, vocab_size=(vocab_size or 256))
        else:
            raise SystemExit("Need --data.file if checkpoint meta lacks BPE vocab")
        model_vocab = tok.vocab_size
    else:
        if chars is not None:
            token_chars = list(chars)
        elif args.data_file:
            text = load_text_file(args.data_file)
            token_chars = sorted(list(set(text)))
        else:
            raise SystemExit("Need --data.file if checkpoint meta lacks chars")
        if vocab_size is not None and len(token_chars) < vocab_size:
            pad_needed = vocab_size - len(token_chars)
            token_chars = token_chars + ["�"] * pad_needed
        tok = CharTokenizer(chars=token_chars)
        model_vocab = len(tok.chars)

    if vocab_size is not None and len(token_chars) < vocab_size:
        pad_needed = vocab_size - len(token_chars)
        token_chars = token_chars + ["�"] * pad_needed
    model = GPTMini(
        vocab_size=(vocab_size if vocab_size is not None else model_vocab),
        d_model=meta.get("d_model", 256),
        n_layer=meta.get("n_layer", 4),
        n_head=meta.get("n_head", 4),
        block_size=meta.get("block_size", 128),
        mlp_ratio=meta.get("mlp_ratio", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    def sample(model, tokenizer, max_new_tokens: int, temperature: float = 1.0):
        # start from a newline
        idx = torch.tensor([[tokenizer.stoi.get("\n", 0)]], dtype=torch.long)
        generated = []
        for _ in range(max_new_tokens):
            logits = model(idx)[:, -1, :]
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            generated.append(next_id.item())
        return tokenizer.decode(generated)

    out = sample(model, tok, args.max_new_tokens, args.temp)
    print(out)


if __name__ == "__main__":
    main()
