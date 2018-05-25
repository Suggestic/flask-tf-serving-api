import json

import numpy as np
import tensorflow as tf
from flask import current_app
from google.protobuf.json_format import MessageToJson
from grpc.beta import implementations
from tensorflow.python.saved_model import signature_constants

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


class TFServing(object):
    """
    Automatically connects to TF Serving using parameters defined in Flask
    configuration.
    """

    def __init__(self, app=None, config_prefix='TFSERVING'):
        if app is not None:
            self.init_app(app, config_prefix)

    def key(self, suffix):
        return '%s_%s' % (self.config_prefix, suffix)

    def init_app(self, app, config_prefix='TFSERVING'):

        if 'tfserving' not in app.extensions:
            app.extensions['tfserving'] = {}

        if config_prefix in app.extensions['tfserving']:
            raise Exception(
                'duplicate config_prefix "%s"' % self.config_prefix)

        self.config_prefix = config_prefix

        app.config.setdefault(self.key('HOST'), 'localhost')
        app.config.setdefault(self.key('PORT'), 8500)
        app.config.setdefault(self.key('TIMEOUT'), 5.0)
        app.config.setdefault(
            self.key('SIGNATURE'),
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        app.config.setdefault(self.key('MODEL'), None)
        app.config.setdefault(self.key('VERSION'), None)

        try:
            int(app.config[self.key('PORT')])
        except ValueError:
            raise TypeError('%s_PORT must be an integer' % self.config_prefix)

        try:
            float(app.config[self.key('TIMEOUT')])
        except ValueError:
            raise TypeError('%s_TIMEOUT must be a float' % config_prefix)

        self.host = app.config[self.key('HOST')]
        self.port = app.config[self.key('PORT')]

        channel = implementations.insecure_channel(self.host, self.port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)
        app.extensions['tfserving'][config_prefix] = self.stub

    def predict(self,
                inputs,
                name=None,
                signature=None,
                version=None,
                timeout=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name or current_app.config[self.key('MODEL')]
        request.model_spec.signature_name = signature or current_app.config[self.key(
            'SIGNATURE')]
        version = version or current_app.config[self.key('VERSION')]
        if version:
            request.model_spec.version.value = version

        for input_name in inputs:
            _in = inputs[input_name]
            request.inputs[input_name].CopyFrom(
                tf.contrib.util.make_tensor_proto(_in, shape=_in.shape))

        probas_message = json.loads(
            MessageToJson(
                self.stub.Predict(request, timeout
                                  or current_app.config[self.key('TIMEOUT')])))

        prediction = {}
        for out in probas_message['outputs']:

            dim = tuple([
                int(i['size'])
                for i in probas_message['outputs'][out]['tensorShape']['dim']
            ])
            probas = np.asarray(probas_message['outputs'][out]['floatVal'])
            probas.resize(dim)
            prediction[out] = probas
        return prediction
