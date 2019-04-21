import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def show_result(acc, up_rate):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(range(len(acc)), acc)
    ax1.set_xlabel("step")
    ax1.set_ylabel("error_rate (%)")
    ax1.legend()
    ax1.grid()
    ax1.set_title("BPTT error rate")

    ax2.plot(range(len(up_rate['u'])), up_rate['u'], label='u')
    ax2.plot(range(len(up_rate['v'])), up_rate['v'], label='v')
    ax2.plot(range(len(up_rate['w'])), up_rate['w'], label='w')
    ax2.set_xlabel("step")
    ax2.set_ylabel("update rewards")
    ax2.legend()
    ax2.grid()
    ax2.set_title("BPTT episode rewards")
    plt.show()


def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    show_result(source['accuracy'], source['update_rate'])


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()