import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def show_result(x):
    fig, ax = plt.subplots()
    ax.plot(range(len(x)), x)

    ax.set_xlabel("step")
    ax.set_ylabel("error_rate (%)")
    ax.legend()
    ax.grid()
    ax.set_title("BPTT error rate")
    plt.show()


def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    show_result(source['accuracy'])


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()