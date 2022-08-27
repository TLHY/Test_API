FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

RUN pip install gdown
RUN gdown --fuzzy https://drive.google.com/file/d/1CapX0Gn-e8Ty736rPgw5QfKTG6fKY6e4/view?usp=sharing
RUN unzip persona_text.zip
RUN rm persona_text.zip
EXPOSE 9999

CMD ["python3","GPT_API.py"]
