from flask import Flask,request,jsonify,render_template
import os
from predict import predict 
img_folder = os.path.join( os.path.dirname(__file__), 'static')
model_path = os.path.join( os.path.dirname(__file__), 'model', 'best_model.pt')
os.makedirs(img_folder, exist_ok=True)
# from werkzeug import secure_filename
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def render_page():
    try:
        for filename in os.listdir(img_folder):
            if '.html' in filename or '.py' in filename or '.js' in filename or '.css' in filename:
                continue
            os.remove(os.path.join(img_folder, filename))
    except Exception as e:
        print("error while deleting")
        print(e)
    return render_template("upload.html", result='', image='')

@app.route('/predict',methods=['POST'])
def prediction():
    file_ = request.files['file']
    img = file_.filename
    img_name = os.path.join(img_folder, img)
    file_.save(img_name)
    result = predict(model_path, img_name)
    return render_template("upload.html", result=result, image=img)

if __name__ == '__main__':
    app.run()
