FROM python:3.12-alpine

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/tmp && \
    touch /app/tmp/data.txt && \
    echo '[{"name":"alejandru","email":"alejandru@gmail.com"},{"name":"birja","email":"birja@gmail.com"},{"name":"muchelle","email":"muchelle@gmail.com"},{"name":"christianu","email":"christianu@gmail.com"},{"name":"alvaru","email":"alvaru@gmail.com"}]' > /app/tmp/data.txt

EXPOSE 5000

CMD ["python", "app.py"]