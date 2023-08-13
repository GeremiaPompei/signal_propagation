import argparse

from src.analysis import analysis_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', help=f'type of analysis to use: {list(analysis_dict.keys())}')
    args = parser.parse_args()
    analysis_dict[args.analysis]()


if __name__ == '__main__':
    main()
