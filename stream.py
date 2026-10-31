import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/pdd/predict/"

st.title("Распознавание дорожных знаков ПДД")

file = st.file_uploader("Загрузите фото знака", type=["png", "jpg", "jpeg"])

if file:
    st.image(file, caption="Загруженное фото", width=300)

    if st.button("Определить знак"):
        files = {"file": (file.name, file.getvalue(), file.type)}
        try:
            r = requests.post(API_URL, files=files, timeout=60)
            if r.status_code == 200:
                data = r.json()
                st.success("Знак определён")
                st.write(f"Номер знака: {data['number']}")
                st.write(f"Название: {data['name']}")
                st.write(f"Категория: {data['category']}")
                st.write(f"Описание: {data['description']}")
                st.write(f"Совет: {data['tip']}")
            else:
                st.error(f"Ошибка {r.status_code}")
        except Exception as e:
            st.error(f"Нет связи с API: {e}")