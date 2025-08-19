import argparse
import sys
from .utils.config import load_yaml_config


def main(argv=None):
    p = argparse.ArgumentParser(
        "syntrix.config",
        description="Validate and inspect a YAML configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = p.parse_args(argv)

    try:
        cfg = load_yaml_config(args.config)
    except Exception as e:
        print(f"Invalid config: {e}")
        sys.exit(1)

    print("Config OK.")
    print("Model:", cfg.model)
    print("Optim:", cfg.optim)
    print("Schedule:", cfg.schedule)
    print("Train:", cfg.train)


if __name__ == "__main__":
    main()
