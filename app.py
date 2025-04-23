from fastapi import FastAPI ,File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import keras
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

model = tf.keras.models.load_model('./models/model_20250407-203412.h5')

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class_name=["early_binding","late_binding","normal"]

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    
    image = Image.open(io.BytesIO(img_bytes))
    
    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)
   
    model_ans=model.predict(img_array)
    confi=float(np.max(model_ans))
    model_ans=np.argmax(model_ans)

    ans=class_name[model_ans]
    

    
   
    return {
        "class": ans,
        "confidence": confi
    }

    

if __name__=="__main__" :
    uvicorn.run(app,host='localhost',port=8080)