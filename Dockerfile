FROM python:3.10
WORKDIR /app

COPY . .

EXPOSE 80

CMD ["python3","GPT_API.py"]