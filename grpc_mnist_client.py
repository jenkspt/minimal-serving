from __future__ import print_function

import argparse
import time
import numpy as np
from cv2 import imread

#from grpc.beta import implementations

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def run(hostport, image, model, signature_name):

    # channel = grpc.insecure_channel('%s:%d' % (host, port))

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    #channel = implementations.insecure_channel(host, port)
    #stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Read an image
    #data = imread(image)
    data = imread(image, -1).astype(np.uint8)
    print(data.shape)
    #print(data)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['image'].CopyFrom(make_tensor_proto(data, shape=[1, 28, 28]))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', help='Tensorflow server host:port', default='127.0.0.1:8500', type=str)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.server, args.image, args.model, args.signature_name)
