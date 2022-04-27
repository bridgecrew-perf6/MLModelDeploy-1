import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict',methods=['POST'])
def predict():
    statistics = pd.read_csv("datasetstatistics.csv")
    temp_info=""
    l_info=""
    r_info=""
    magnitude_info=""
    features = request.form.values()

    if(any(x == "" for x in features)):
        return render_template('page.html', prediction_text="", invalid_type_warning = "Please fill missing fields")
    
    features = [float(x) if x.isdigit() else x.capitalize() for x in request.form.values()]
    sp_class={"O":0,"B":1,"A":2,"F":3,"G":4,"K":5,"M":6}
    if(features[-1] not in sp_class.keys()):
        return render_template('page.html', prediction_text="", invalid_type_warning = "Spectral Class {} not acceptable".format(features[-1]))
    
    features[-1] = sp_class[features[-1]]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    star_classes ={0:"Red Dwarf", 1:"Brown Dwarf", 2:"White Dwarf", 3:"Main Sequence" , 4:"Super Giant", 5:"Hyper Giant"}

    output = star_classes[prediction[0]]
    
    infos = ["", "", "", ""]
    for i in range(0,len(infos)):
        if(features[i]<statistics[str(i)][3]):
            infos[i] = "is the smallest value we have ever registered!"
        elif(features[i]>statistics[str(i)][7]):
            infos[i] = "is the biggest value we have ever registered!"
        elif(features[i]>statistics[str(i)][1]-statistics[str(i)][2] and features[i]<statistics[str(i)][1]+statistics[str(i)][2]):
            infos[i] = "is considered normal according to our database"

    temp_info = "The temperature of your star " + infos[0] if infos[0] != "" else "" 
    l_info = "The luminosity of your star " + infos[1] if infos[1] != "" else "" 
    r_info = "The radius of your star " + infos[2] if infos[2] != "" else "" 
    magnitude_info = "The absolute magnitude of your star " + infos[3] if infos[3] != "" else "" 
    return render_template('page.html', prediction_text="Your star is a {}".format(output), invalid_type_warning = "", temp_text = temp_info, l_text = l_info, r_text = r_info, magnitude_text=magnitude_info)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)