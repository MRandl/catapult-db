import argparse
import json


def recall(a, b):
    return len(set(a) & set(b)) / len(set(a))


args = argparse.ArgumentParser()
args.add_argument("--file1")
args.add_argument("--file2")
args = args.parse_args()

with open(args.file1, "r") as f:
    f1 = json.load(f)["results"][0]["neighbors"]

with open(args.file2, "r") as f:
    f2 = json.load(f)["results"][0]["neighbors"]

sum = 0.0
for i, j in zip(f1, f2):
    sum += recall(i, j[: len(i)])

assert len(f1) == len(f2)
print(sum / len(f1))
