#!/usr/bin/python

"""
Calculate accuracy figures from CSV file created by sample_run/run_model.py.
"""

from __future__ import division

import argparse
import csv
import sys
from collections import namedtuple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--method", default="top1-with-threshold", choices=["top1", "top5", "top1-with-threshold"])
    parser.add_argument("--threshold", type=float, default=0.2, help="Only applies when --method=top1-with-threshold")
    args = parser.parse_args()

    total = 0
    accurate = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with open(args.csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:  # Non-csv summary at the end of the file
                break
            image = row[1]
            label = row[2]
            predictions = [namedtuple("P", "label confidence")(*x)
                           for x in zip(row[3::2],
                                        [float(x) for x in row[4::2]])]

            # Calculate basic accuracy for any label
            total += 1
            if match(label, predictions, args.method, args.threshold):
                accurate += 1
            # else:
            #     print("DIDN'T MATCH:", image, label, predictions)

            # Calculate precision & recall figures for "cart" label
            if label == "video call":
                if match("video call", predictions, args.method, args.threshold):
                    true_positives += 1
                else:
                    false_negatives += 1
                    print()
                    print("FALSE NEGATIVE:", image, label, predictions)
            else:
                if match("video call", predictions, args.method, args.threshold):
                    false_positives += 1
                    print()
                    print("FALSE POSITIVE:", image, label, predictions)
                else:
                    true_negatives += 1

    print()
    print("With method: %s" % args.method)
    if args.method == "top1-with-threshold":
        print("     threshold: %f" % args.threshold)
    print()
    print("All categories:")
    print("Accuracy: %d%%" % round(accurate * 100 / total))
    print()
    print("video call :")
    print("%d false positives" % false_positives)
    print("%d false negatives" % false_negatives)
    print("Precision: %d%%" % round(
        true_positives * 100 / (true_positives + false_positives)))
    print("Recall: %d%%" % round(
        true_positives * 100 / (true_positives + false_negatives)))


def match(label, predictions, method, threshold):
    if method == "top1-with-threshold":
        return (label == predictions[0].label and
                predictions[0].confidence >= threshold)
    elif method == "top1":
        return label == predictions[0].label
    elif method == "top5":
        return any(label == x.label for x in predictions)
    else:
        assert False, "unknown method %s" % (method,)


if __name__ == "__main__":
    main()