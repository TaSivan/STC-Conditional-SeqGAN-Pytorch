from flask import Flask, request, render_template
from flask_cors import CORS
from opencc import OpenCC

import torch
torch.cuda.set_device(2)

from evaluator.responsor import responsor

app = Flask(__name__)
CORS(app)

tw2s = OpenCC('tw2s')
s2tw = OpenCC('s2tw')

@app.route("/response", methods=['GET', 'POST'])
def resp():
    if request.method == 'POST':
        if request.get_json():
            input = request.get_json()['input']
            input = tw2s.convert(input)
            output = "".join(responsor.response(input).split())
            output = s2tw.convert(output)
            return output
        else:
            return "Please input a sentence to your request body."
    else:
        return "Please use POST."


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()