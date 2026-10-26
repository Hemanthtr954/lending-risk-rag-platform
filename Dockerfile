FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m app.train 30000 3
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
