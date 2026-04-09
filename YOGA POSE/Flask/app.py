from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

model = load_model("xcep_yoga.h5")

classes = ['Downdog','Goddess','Plank','Tree','Warrior2']


# HOME
@app.route("/")
def home():
    return render_template("index.html")


# ABOUT
@app.route("/about")
def about():
    return render_template("about.html")


# MODELS
@app.route("/models")
def models():
    return render_template("models.html")


# CONTACT
@app.route("/contact")
def contact():
    return render_template("contact.html")


# IMAGE PAGE
@app.route("/iinput")
def input_page():
    return render_template("input.html")


# IMAGE PREDICTION
@app.route("/ioutput", methods=['POST'])
def output():
    file = request.files['file']
    filepath = "uploaded_image.png"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    return render_template("input.html", prediction=result)


# VIDEO PAGE
@app.route("/vinput")
def vinput_page():
    return render_template("vinput.html")


# VIDEO PREDICTION
@app.route("/voutput", methods=['POST'])
def voutput():
    file = request.files['file']
    video_path = "uploaded_video.mp4"
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame)
        result = classes[np.argmax(pred)]
        predictions.append(result)

    cap.release()

    if len(predictions) == 0:
        final_result = "No pose detected"
    else:
        final_result = max(set(predictions), key=predictions.count)

    return render_template("vinput.html", prediction=final_result, video_path=video_path)


if __name__ == "__main__":
    app.run(debug=True)