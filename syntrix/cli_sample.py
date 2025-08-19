import argparse
from .utils.seed import set_seed


def main(argv=None):
    p = argparse.ArgumentParser("syntrix.sample")
    p.add_argument("--ckpt", type=str, default="runs/latest/ckpt.pt")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    set_seed(args.seed)
    print(f"Syntrix sample stub | ckpt={args.ckpt} tokens={args.max_new_tokens} temp={args.temp}")
    # TODO: load model and sample


if __name__ == "__main__":
    main()


