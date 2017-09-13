# Flask-TFServing

TF Serving support for Flask applications

## Quickstart

    import numpy as np
    from flask import Flask
    from flask_tfserving import TFServing

    app = Flask(__name__)
    serving = TFServing(app)

    @app.route('/')
    def iris_classification():
        prediction = serving.predict(inputs={'input': np.matrix([1,2,3,4])})
        return prediction['output'].argmax()
