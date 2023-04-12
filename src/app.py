# import modules
import torch
from animal import transform, Net
from flask import Flask, render_template, request, redirect
import io
from PIL import Image
import base64
import numpy as np

# prediction based on trained model
# src/model (1).pt


def predict(img):
    net = Net().cpu().eval()
    # load the parameters of the model
    net.load_state_dict(torch.load('src/model (1).pt',
                        map_location=torch.device('cpu')))
    # transform the image
    img = transform(img)
    # unsqueeze the image
    img = img.unsqueeze(0)
    # predict the class of the image
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y
# createa a function that returns the name of the given label


def label_to_name(label):
    if label == 0:
        return "cat"
    elif label == 1:
        return "dog"
    else:
        return "unknown"


# instantiate the flask app
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "gif"])

# check if the file is an allowed extension


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@ app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'filename' not in request.files:
            return redirect(request.url)
        # get data from the file
        file = request.files['filename']
        # check if the file is an allowed extension
        if file and allowed_file(file.filename):
            # process done to the image file
            # get buffer from the file
            buf = io.BytesIO()
            image = Image.open(file)
            # write the image to the buffer
            image.save(buf, format='png')
            # encode and decode the image
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            base64_data = "data:image/png;base64,{}".format(base64_str)
            # get the prediction
            prediction = predict(image)
            # get the name of the prediction
            name = label_to_name(prediction)
            # return the prediction
            return render_template('result.html', animalName=name, image=base64_data)

        return redirect(request.url)

    elif request.method == 'GET':
        return render_template('index.html')


# run the application
if __name__ == '__main__':
    app.run(debug=True)
