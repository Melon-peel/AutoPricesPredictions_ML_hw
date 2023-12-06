from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import csv
import codecs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import io
import pickle

app = FastAPI()


with open("transformers_and_model.pickle", "rb") as f:
    transformers, model = pickle.load(f)
    imputer_ct, scaler, ohe_ct = transformers


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


def drop_cols(X_):
    X = X_.copy()
    X = X.drop(["name", "selling_price"], axis=1, errors='ignore')
    return X    

def preproc_words(X_):
    df = X_.copy()

    # фикс mileage
    nas_mil = df.mileage.isna()
    df.loc[~nas_mil, "mileage"] = df[~nas_mil].mileage.apply(lambda x: x.split(" ")[0]).astype(float)


    # фикс engine
    nas_eng = df.engine.isna()
    df.loc[~nas_eng, "engine"] = df[~nas_eng].engine.apply(lambda x: x.split(" ")[0]).astype(float)

    # фикс max_power
    nas_pow = df.max_power.isna()
    df.loc[~nas_pow, "max_power"] = df[~nas_pow].max_power.apply(lambda x: x.split(" ")[0]).apply(lambda x: x if x else None).astype(float)
    
    df = df.drop("torque", axis=1)
    
    return df
    



def preprocess_item(item: Item) -> np.array:
    vals = list(dict(item).values())
    cols = dict(item).keys()
    df = pd.DataFrame([vals], columns=cols)

    df = drop_cols(df)
    df = preproc_words(df)

    df[["year", "km_driven", "mileage", "engine", "max_power", "seats"]] = imputer_ct.transform(df)
    df[['year', 'km_driven', 'mileage', 'engine', 'max_power']] = scaler.transform(df[['year', 'km_driven', 'mileage', 'engine', 'max_power']])
    df = ohe_ct.transform(df)
    
    return df

def preproc_items(df_: pd.DataFrame) -> np.array:
    df = df_.copy()
    df = drop_cols(df)
    df = preproc_words(df)

    df[["year", "km_driven", "mileage", "engine", "max_power", "seats"]] = imputer_ct.transform(df)
    df[['year', 'km_driven', 'mileage', 'engine', 'max_power']] = scaler.transform(df[['year', 'km_driven', 'mileage', 'engine', 'max_power']])
    df = ohe_ct.transform(df)
    
    return df    


@app.post("/predict_item")
def predict_item(item: Item):

    arr = preprocess_item(item)
    prediction = model.predict(arr)[0]
    
    return prediction


@app.post("/predict_items", response_class=StreamingResponse)
def predict_items(file: UploadFile):
    csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    data = []
    for i in csvReader:
        data.append(i)
    df = pd.DataFrame.from_dict(data)
    arr = preproc_items(df)
    predictions = model.predict(arr)
    df['predictions'] = predictions
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    file.file.close()
    return response

