import os
import base64
import uvicorn
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

app = FastAPI(title="PDD CHECK")

class SignInfo(BaseModel):
    number: str = Field(description="Номер знака по ПДД, например 1.15 или 3.24. Если не уверен, пустая строка")
    name: str = Field(description="Официальное название знака на русском")
    category: str = Field(description="Категория: Предупреждающий знак, Запрещающий знак, Предписывающий знак, Информационный знак, Знак приоритета, Знак особых предписаний или Знак сервиса")
    description: str = Field(description="Короткое описание, что означает знак, на русском")

model = ChatGroq(
    model=os.getenv("MODEL_NAME", "qwen/qwen3.8-27b"),
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_format="hidden",
)

tip_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты эксперт по правилам дорожного движения. Отвечай только на русском."),
    ("human", "Дорожный знак: {name}. Дай один короткий совет водителю."),
])

chain = tip_prompt | model | StrOutputParser()
vision_chain = model.with_structured_output(SignInfo)

def get_sign_tip(name: str):
    response = chain.invoke({"name": name})
    return response

def make_messages(image_bytes: bytes, mime: str):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    content = [
        {
            "type": "text",
            "text": "Это фото дорожного знака. Определи знак по ПДД. Если это не дорожный знак, напиши в name: Не дорожный знак. Отвечай на русском."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{encoded}"
            }
        }
    ]
    return [HumanMessage(content=content)]

def get_sign_info(image_bytes: bytes, mime: str):
    messages = make_messages(image_bytes, mime)
    return vision_chain.invoke(messages)

def get_signs_info(images):
    messages_list = [
        make_messages(image_data, mime_type)
        for image_data, mime_type in images
    ]
    return vision_chain.batch(messages_list)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes
    )
    for component in schema["components"]["schemas"].values():
        properties = component.get("properties", {})
        for prop in properties.values():
            items = prop.get("items")
            if items and items.get("contentMediaType"):
                prop["items"] = {
                    "type": "string",
                    "format": "binary"
                }
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi

@app.post("/pdd/batch/", tags=["ПДД"])
async def check_many_pdd(files: List[UploadFile] = File(...)):
    images = []
    for uploaded_file in files:
        image_data = await uploaded_file.read()
        mime_type = uploaded_file.content_type or "image/png"
        images.append((image_data, mime_type))
    infos = get_signs_info(images)
    result = []
    for info in infos:
        result.append({
            "number": info.number,
            "name": info.name,
            "category": info.category,
            "description": info.description,
        })
    return result

@app.post("/pdd/predict/", tags=["ПДД"])
async def check_image_pdd(file: UploadFile = File(...)):
    image_data = await file.read()
    mime_type = file.content_type or "image/png"
    info = get_sign_info(image_data, mime_type)
    return {
        "number": info.number,
        "name": info.name,
        "category": info.category,
        "description": info.description,
        "tip": get_sign_tip(info.name),
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
