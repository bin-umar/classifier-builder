#!/usr/bin/python

import argparse
import os
import random
import sys
from glob import glob

import numpy as np
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  image_reader = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image_reader)) # Convert to grayscale
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def main():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  classifier_data_dir = os.path.join(current_dir, 'testai_model')

  model_file = os.path.join(classifier_data_dir, 'saved_model.pb')
  label_file = os.path.join(classifier_data_dir, 'saved_model.pbtxt')

  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=1000)
  parser.add_argument("--top-k", type=int, default=1)
  parser.add_argument("--confidence-threshold", type=float, default=0.2)
  parser.add_argument("--test-images-glob", default=os.path.join(
    current_dir, "../training_images/*/*.jpg"))
  args = parser.parse_args()

  graph = load_graph(model_file)
  labels = load_labels(label_file)

  random.seed(1)
  test_files = random.sample(glob(args.test_images_glob), args.samples)
  total = len(test_files)
  accurate = 0
  for i, filename in enumerate(test_files):
    filename = os.path.relpath(filename)
    expected_label = os.path.basename(os.path.dirname(filename)).replace("_", " ")
    top_k = run_model(graph, labels, filename, args.top_k)
    sys.stdout.write("%d/%d,%s,%s," % (i, total, filename, expected_label))
    for predicted_label, confidence in top_k:
      sys.stdout.write("%s,%s," % (predicted_label, confidence))
      if (expected_label == predicted_label and
              confidence > args.confidence_threshold):
        accurate += 1
        break
    sys.stdout.write("\n")
  sys.stdout.flush()
  sys.stderr.write("Top-%d Accuracy: %d%%\n" %
                   (args.top_k, accurate*100 / (i+1)))

def run_model(graph, labels, filename, k):

  t = read_tensor_from_image_file(
      filename,
      input_height=224,
      input_width=224,
      input_mean=0,
      input_std=255)

  input_name = "import/Placeholder"
  output_name = "import/final_result"
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-k:][::-1]
  return [(labels[i], results[i]) for i in top_k]


if __name__ == "__main__":
  main()
