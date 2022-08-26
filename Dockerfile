FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install gdown
RUN gdown https://drive.google.com/file/d/1qSrXy-ZEvPVTArPsOA19OoCayw1_xcFz/view?usp=sharing
COPY . .

EXPOSE 80

CMD ["python3","GPT_API.py"]
