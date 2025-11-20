from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app=Flask(__name__)

def get_clean_data(data_form):
    gestation=float(data_form['gestation'])
    parity=float(data_form['parity'])
    age=float(data_form['age'])
    height=float(data_form['height'])
    weight=float(data_form['weight'])
    smoke=float(data_form['smoke'])

    cleaned_data={
        "gestation":[gestation],
        "parity":[parity],
        "age":[age],
        "height":[height],
        "weight":[weight],
        "smoke":[smoke]
    }

    return cleaned_data


@app.route('/',methods=['GET'])
def get_home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def get_predict():
    test_data=request.form #get data from user
    clean_data=get_clean_data(test_data)

    test_clean_df=pd.DataFrame(clean_data) #coverting json data to DF
    with open("model.pkl", "rb") as obj:
        model= pickle.load(obj)

        prediction=model.predict(test_clean_df)
        my_predict=round(float(prediction[0]),2)

        #return response in json format

        response={"prediction Value": my_predict}
        return render_template("index.html",my_predict=my_predict)



if __name__=="__main__":
    app.run(debug=True)
