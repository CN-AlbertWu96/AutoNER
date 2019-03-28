import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', type=str)
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--rate', type=float, default=0.1)
    parser.add_argument('--output', type=str)

    golden_file_path = args.golden
    annotated_file_path = args.annotation
    rate = args.rate
    output_file_path = args.output

    with open(golden_file_path, 'r') as fin:
        