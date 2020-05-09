# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow.compat.v1 as tf
# from PIL import Image
import scipy
import scipy.misc
import numpy as np
import cv2
import grpc
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


tf.app.flags.DEFINE_string('server', '127.0.0.1:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', 'image.jpg', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('path_to_labels','mscoco_label_map.pbtxt','path to labels file')
FLAGS = tf.app.flags.FLAGS

IMG_WIDTH, IMG_HEIGHT = 300, 300

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3) #color images
  img = tf.image.convert_image_dtype(img, tf.float32) 
   #convert unit8 tensor to floats in the [0,1]range
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 

def process_path(file_path):
  # label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


host, port = FLAGS.server.split(':')
# channel = implementations.insecure_channel(host, int(port))
channel = grpc.insecure_channel(FLAGS.server) 
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'obj_det'
request.model_spec.signature_name = "serving_default"

# Initalize prediction 
# Specify signature name (should be the same as specified when exporting model)
# image_np = process_path(FLAGS.image)
dim = (300,300)
image_np = cv2.cvtColor(cv2.imread(FLAGS.image),  cv2.COLOR_BGR2RGB)
image_np = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)
# image_np =  load_image_into_numpy_array(image_np)
image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)

print('shape',image_np_expanded.shape)
request.inputs['inputs'].CopyFrom(
        tf.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape))

# Call the prediction server
start = time.time()
result = stub.Predict(request, 10.0)  # 10 secs timeout
time_taken = time.time() - start
print('time_taken for request:', time_taken)
# print(result)
# Plot boxes on the input image
# category_index = load_label_map(FLAGS.path_to_labels)
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(FLAGS.path_to_labels)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
boxes = result.outputs['detection_boxes'].float_val
classes = result.outputs['detection_classes'].float_val
scores = result.outputs['detection_scores'].float_val
image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    np.reshape(boxes,[100,4]),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

# vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes,[100,4]),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=8)

# Save inference to diskg
img = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
img_2 = cv2.cvtColor(np.float32(image_vis), cv2.COLOR_RGB2BGR)
cv2.imwrite('out.jpeg',img )
cv2.imwrite('out2.jpeg',img_2)
# cv2.imwrite('out_2.jpeg',)
# cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
    
# scipy.misc.imsave('%s.jpg'%(FLAGS.input_image), image_vis)

