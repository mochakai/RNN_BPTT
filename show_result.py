import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def show_result(acc, up_rate):
    fig, axs = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=0.5)
    for ax in axs:
        ax.set_xlabel("step")
        ax.legend()
        ax.grid()

    axs[0].plot(range(len(acc)), acc)
    axs[0].set_ylabel("error_rate (%)")
    axs[0].set_title("BPTT error rate")

    axs[1].plot(range(len(up_rate['u'][:10000])), up_rate['u'][:10000], label='u')
    axs[1].set_ylabel("u's update rewards")
    axs[2].plot(range(len(up_rate['v'][:10000])), up_rate['v'][:10000], label='v')
    axs[2].set_ylabel("v's update rewards")
    axs[3].plot(range(len(up_rate['w'][:10000])), up_rate['w'][:10000], label='w')
    axs[3].set_ylabel("w's update rewards")
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