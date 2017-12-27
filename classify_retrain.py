"""
This program is a modified version of https://github.com/eldor4do/Tensorflow-Examples/retraining-example.py
Luke Jackson 2017 I-Tap
"""

#   Imports
import tensorflow as tf
import numpy as np
import argparse
import os
import sys


def predict_image_class(image_path, label_path, model_path):
    
    matches = None  # Default return to none

    # check if input document exists
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
        return matches
    else:
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    '''
    # graph was trained on integer class names instead of kanji. convert them back with this dictionary.
    with open(os.path.join(os.path.dirname(model_path), 'labels_lookup.txt'), encoding='UTF-8') as f:
        classnames_lookup_raw = f.read().splitlines()
        classnames_lookup = {}
        for cl in classnames_lookup_raw:
            vals = cl.split(',')
            try:
                int(vals[0])
            except ValueError:
                break
            classnames_lookup[vals[0].rstrip()] = vals[1].rstrip()
    '''

    # Load the retrained inception based graph
    with tf.gfile.FastGFile(model_path, 'rb') as f:
            # init GraphDef object
            graph_def = tf.GraphDef()
            # Read in the graph from the file
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # this point the retrained graph is the default graph

    with tf.Session() as sess:
        # These 2 lines are the code that does the classification of the images 
        # using the new classes we retrained Inception to recognize. 
        #   We find the final result tensor by name in the retrained model
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        #   Get the predictions on our image by add the image data to the tensor
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        
        # Format predicted classes for display
        #   use np.squeeze to convert the tensor to a 1-d vector of probability values
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting the indicies of the top 5 predictions

        #   read the class labels in from the label file
        f = open(label_path, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        print("")
        print("Image Classification Probabilities")
        #   Output the class probabilites in descending order
        for node_id in top_k:
            human_string = filter_delimiters(labels[node_id])
            score = predictions[node_id]
            print('{0:s} (score = {1:.5f})'.format(human_string, score))

        print("")

        # answer = labels[top_k[0]]
        cls = filter_delimiters(labels[top_k[0]])
        answer = cls
        print(answer)
        return answer


# Remove ugly characters from strings
def filter_delimiters(text):
    filtered = text[2:-3]
    ## filtered = filtered.strip("b'")
    ## filtered = filtered.strip("'")
    return filtered


# Get the path to the image you want to predict.
if __name__ == '__main__':
    
    # Ensure the user passes the image_path
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument(
      'image_path',
      type=str,
      default='',
      help='Path of image to classify.'
    )
    parser.add_argument(
      '-model_path',
      type=str,
      required=True,
      help='Path of model to classify with.'
    )
    parser.add_argument(
      '-labels_path',
      type=str,
      required=True,
      help='Path of labels in model'
    )
    args = parser.parse_args()

    # We can only handle jpeg images.   
    if args.image_path.lower().endswith(('.jpg', '.jpeg')):
        # predict the class of the image
        predict_image_class(args.image_path, args.labels_path, args.model_path)
    else:
        print('File must be a jpeg image.')

