Check MNIST model example with Tensorflow Serving
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md


# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v `pwd`/models/:/models/obj_det \
    -e MODEL_NAME=obj_det \
    tensorflow/serving &

