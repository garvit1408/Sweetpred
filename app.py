from rdkit import Chem
from mordred import Calculator, descriptors
from padelpy import from_smiles
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
from flask import session, app

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.sav", "rb"))

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=3)

@app.route("/")
def Home():
    return render_template("index.html")
  

@app.route("/howtouse")
def howtouse():
    return render_template("/how-to-use.html")

@app.route("/sweetpredpy")
def sweetpredpy():
    return render_template("https://github.com/cosylabiiit/sweetpred")

@app.route("/contactus")
def contactus():
    return render_template("/contact-details.html")

@app.route("/cosylab")
def cosylab():
    return render_template("https://cosylab.iiitd.edu.in/")

@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/predict1", methods = ["POST"])
def predict1():
    smile = [(x) for x in request.form.values()]
    mols = [Chem.MolFromSmiles(smi) for smi in smile]
    calc = Calculator(descriptors, ignore_3D=True)
    mdf1 = calc.pandas(mols)
    mdf1.insert(loc = 0,column = 'Name',value = 1)
    desc_fp = from_smiles(smile, fingerprints=True)
    doc=dict(desc_fp[0])
    smiles_list=[]
    smiles_list.append(doc)
    pdf = pd.DataFrame()
    pdf=pd.DataFrame(smiles_list)
    pdf.insert(loc = 0,column = 'Name',value = 1)
    pdf.replace(np.nan, 0, inplace=True)
    pdf.isnull().values.sum()
    df=pd.merge(pdf, mdf1,on='Name')
    obj_column_df = df.select_dtypes(include=['object'])
    obj_column_df_filled_with_nans = obj_column_df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    percentNumericEntries = obj_column_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().sum()/len(df.index))
    salvagableStringColumns = obj_column_df_filled_with_nans[percentNumericEntries[percentNumericEntries > 0.70].index].fillna(method='bfill')
    df[salvagableStringColumns.columns] = salvagableStringColumns;
    df[(percentNumericEntries<0.70).index] = 0
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    result = model.predict(df)
    #result="hello"
    return render_template("predict.html", prediction_text = "Sweetness is {}".format(result))

if __name__ == "__main__":
    app.debug = True
    app.run()
    #app.run(debug=True)

def flush(self):
    pass





