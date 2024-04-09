import uvicorn 
from fastapi import FastAPI
from pydantic import BaseModel
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from fastapi import HTTPException


def clean_data(text):
    text = text.lower()
    text = text.replace("/","").replace("(","").replace(")","").replace(":","").replace("'","")
    
    return text

def dataLoading(filename):
    with open(filename,"r") as f:
        content = f.read()

    content = json.loads(content)
    df = pd.DataFrame(content)

    df["externalStatus"] = df["externalStatus"].apply(clean_data)
    df["internalStatus"] = df["internalStatus"].apply(clean_data)

    return df

def dataForModel(filename):
    df = dataLoading(filename)

    X = df['externalStatus']
    y = df['internalStatus']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_length = 50
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    Y_train_reshaped = np.array(y_train).reshape(-1, 1)
    Y_test_reshaped = np.array(y_test).reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False)

    y_train_one_hot_encoded = encoder.fit_transform(Y_train_reshaped)
    y_test_one_hot_encoded = encoder.fit_transform(Y_test_reshaped)

    return X_train_pad,X_test_pad,y_train_one_hot_encoded,y_test_one_hot_encoded,encoder,tokenizer


def modelBuilder(X_train_pad,y_train_one_hot_encoded):
    model = Sequential()
    model.add(Dense(100,input_shape=(50,),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(15,activation="softmax"))

    model.compile(loss="categorical_crossentropy",metrics=["accuracy"],optimizer='adam')


    model.fit(X_train_pad,y_train_one_hot_encoded,batch_size=32,epochs=100)

    model.save("src/model.h5")

    return model

app = FastAPI()

class ExternalStatus(BaseModel):
    description: str

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post("/predict")
def predict(external_status: ExternalStatus):
    try:
        X_train_pad,X_test_pad,y_train_one_hot_encoded,y_test_one_hot_encoded,encoder,tokenizer = dataForModel("dataset.json")

        text = [external_status.description]
        #text = clean_data(text)
        text_seq = tokenizer.texts_to_sequences(text)
        max_length = 50
        text_pad = pad_sequences(text_seq, maxlen=max_length, padding="post")


        model_path = "src/model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            modelBuilder(X_train_pad,y_train_one_hot_encoded)
            model = load_model(model_path)

        prediction = model.predict(text_pad)

        class_label = encoder.inverse_transform(prediction)
        print(class_label)
        return {"internal_status": class_label.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)