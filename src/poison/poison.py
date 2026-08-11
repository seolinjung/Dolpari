from pathlib import Path
import argparse
import sys

import pandas as pd

import trigger_phase

root = Path(__file__).resolve().parent.parent.parent

def get_clean_data(args):
    clean_data_path = root / "data" / args.input
    if not Path.exists(clean_data_path):
        print("Specified clean data file does not exist.")
        return
    return pd.read_csv(clean_data_path)

def save_poisoned_data(args, poisoned_data):
    output_path = root / "data" / args.output
    poisoned_data.to_csv(output_path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["trigger_phase_manual", "trigger_phase_llm"],
        help="name of poisoning method"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="name of input file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="name of output file"
    )

    args = parser.parse_args()

    if "/" in args.input:
        parser.error("input name should not contain full path")
    if "/" in args.output:
        parser.error("output name should not contain full path")

    return args

def main(args):

    clean_data = get_clean_data(args)
    if clean_data is None:
        sys.exit(1)

    elif args.method == "trigger_phase_manual":
        poisoned_data = trigger_phase.run(clean_data, "manual")

    elif args.method == "trigger_phase_llm":
        poisoned_data = trigger_phase.run(clean_data, "llm")

    save_poisoned_data(args, poisoned_data)

if __name__ == "__main__":
    args = get_args()
    main(args)