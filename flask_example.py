from flask import Flask,jsonify,request
import os
import pandas as pd
import numpy as np
import git
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

os.chdir(os.path.dirname(__file__))

app = Flask(__name__) # inicialimamos la app
app.config["DEBUG"] = True # activamos el debug

@app.route('/git_update', methods=['POST']) # debe estar siempre antes del home page y después de la creación de la app
def git_update():
    repo = git.Repo('./Flask')
    origin = repo.remotes.origin
    repo.create_head('main',
                     origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
    origin.pull()
    return '', 200
@app.route('/')
def index():
    print(os.getcwd())
    return render_template("index.html")

@app.route("/",methods=["GET"]) # esto es propio de flask para poder hacer enrutamientos, la barra indica la home page
def hello():
    return "se actualiza desde visual Ahora" # flask trabaja mejor, y se suele hacer, usando funciones. Esta función enseña eso en la home page

@app.route("/v1/predict",methods=["GET"])
def predict():
    # cargamos el modelo
    model = joblib.load(os.getcwd() + "/ad_model.pkl") # el modelo debe estar en la misma carpeta
    tv = int(request.args.get("tv",None))# son las variables del modelo y las inicializamos en none
    radio = int(request.args.get("radio",None))
    newspaper = int(request.args.get("newspaper",None))

    # pero si las inicializamos como none nos va a dar error porque los modelos solo aceptan valores númericos entonces:
    if tv is None or radio is None or newspaper is None:
        return "Error, no puede ser vacío" # esto nos evita el famoso error 500
    else:
        prediction = model.predict([[tv,radio,newspaper]])

    return jsonify({"prediction":prediction[0]}) # pero usamos jsonify para devolver la predicción para que sea vea bonito, porque sino nos devuelve un np.array

@app.route("/v1/retrain",methods=["PUT"])
def retrain():
    data = pd.read_csv("Advertising.csv",index_col=0)

    x_train,x_test,y_train,y_test = train_test_split(data.drop(columns=["sales"]),
                                    data["sales"],test_size = 0.20,
                                    random_state = 42)
    
    new_model = Lasso(alpha=6000)
    new_model.fit(x_train,y_train)

    joblib.dump(new_model,os.getcwd()+"/ad_model.pkl")

    return "Model retrained. New evaluation metric RMSE:" + str(np.sqrt(mean_squared_error(y_test,new_model.predict(x_test))))

if __name__ == "__main__": # esto quiere decir que si el archivo no tiene el nombre main, lo asigne como main y por tanto principal
    app.run()              # y lo inicialice y se ejectute automáticamente cuando se llame al script
