import torch
import argparse


def train():
    pass


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')


if __name__ == '__main__':
    main()