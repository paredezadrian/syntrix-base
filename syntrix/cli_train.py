import argparse
import torch
from .utils.seed import set_seed, set_threads, get_dtype


def main(argv=None):
    p = argparse.ArgumentParser("syntrix.train")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args(argv)

    set_seed(args.seed)
    set_threads(args.threads)
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    print(f"Syntrix train stub | seed={args.seed} threads={args.threads} dtype={dtype}")

    # TODO: Load config, build dataset/model/trainer and run training


if __name__ == "__main__":
    main()


