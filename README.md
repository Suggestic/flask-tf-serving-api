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
        prediction = serving.predict(
            inputs={'input': np.matrix([5.1, 3.5, 1.4, 0.2])}, model='iris')
            return prediction['output'].argmax()
