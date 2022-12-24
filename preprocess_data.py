import os
import pdb
import argparse



def main(args):
	if args.dataset == "20news":
		unprocessed_file = os.path.join('resources', '20news_unprep.txt')
		processed_file = os.path.join('resources', '20news_prep.txt')
		lines = []
		with open(unprocessed_file, 'r') as f:
			for line in f:
				lines.append(line)

		vocab = set()
		with open(processed_file, 'r') as f:
			for line in f:
				vocab.update(line.split())
		pdb.set_trace()

	else:
		pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="Dataset name", default="20news", choices=["20news", 'wiki'])
	args = parser.parse_args()

	main(args)