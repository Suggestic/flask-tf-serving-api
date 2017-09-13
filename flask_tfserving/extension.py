import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


class TFServing(object):
    """
    Automatically connects to TF Serving using parameters defined in Flask
    configuration.
    """

    def __init__(self, app=None, config_prefix='TFSERVING'):
        if app is not None:
            self.init_app(app, config_prefix)

    def key(suffix):
        return '%s_%s' % (self.config_prefix, suffix)

    def init_app(self, app, config_prefix='TFSERVING'):

        if 'tfserving' not in app.extensions:
            app.extensions['tfserving'] = {}

        if config_prefix in app.extensions['tfserving']:
            raise Exception(
                'duplicate config_prefix "%s"' % self.config_prefix)

        self.config_prefix = config_prefix

        app.config.setdefault(key('HOST'), 'localhost')
        app.config.setdefault(key('PORT'), 8500)
        app.config.setdefault(key('TIMEOUT'), 5.0)
        app.config.setdefault(key('NAME'), None)
        app.config.setdefault(key('SIGNATURE'), None)

        try:
            int(app.config[key('PORT')])
        except ValueError:
            raise TypeError('%s_PORT must be an integer' % self.config_prefix)

        try:
            float(app.config[key('TIMEOUT')])
        except ValueError:
            raise TypeError('%s_TIMEOUT must be a float' % config_prefix)

        host = app.config[key('HOST')]
        port = app.config[key('PORT')]

        channel = implementations.insecure_channel(host, port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)
        app.extensions['tfserving'][config_prefix] = self.stub

    def predict(inputs, name=None, signature=None, tiemout=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name or app.config[key('NAME')]
        request.model_spec.signature_name = signature or app.config[key(
            'SIGNATURE')]

        for input_name in inputs:
            _in = inputs[input_name]
            request.inputs[input_name].CopyFrom(
                tf.contrib.util.make_tensor_proto(
                    _in, shape=_in.shape, dtype=tf.float32))

        self.stub.Predict(request, timeout or app.config[key('TIMEOUT')])
