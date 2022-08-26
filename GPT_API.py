# 과제 1: GPT-2 AI Model을 이용한 나만의 서비스 기획 & 개발
# Open AI에서 공개한 GPT-2 (https://github.com/openai/gpt-2)는 자연어 처리 모델로 다양한 형태의 서비스에서 활용될 수 있습니다.

# Huggingface (https://huggingface.co/models) 에서는 미리 트레이닝된 버전의 GPT-2 모델을 제공하고 있고, 커먼컴퓨터가 서비스하고 있는 Ainize (https://ainize.ai/explore?category=gpt-2) 에서는 GPT2 모델들을 API로 호출할 수 있도록 제공하고 있습니다.

# 예를 들어, 이러한 API들을 사용하면 https://ainize.ai/ainize-team/tabtab 과 같은 서비스를 만들 수 있습니다. GPT-2 model API를 사용하여 나만의 서비스를 기획하고 제출하는 것이 본 과제의 목적입니다.

# 요구사항

# Docker를 사용 (https://docs.docker.com/get-started/)
# 제출물

# 브라우저로 접속 가능한 만들어진 서비스의 IP 혹은 URL
# GitHub Repo에 code push 후, 해당 Repo URL 공유
# 가산점

# Public으로 공유되어 있는 모델이 아닌 직접 GPT-2를 fine-tuning한 모델을 사용
# FAQ

# Q1. 어느 정도로 멋있는 서비스를 만들어야 하나요?

# GPT-2 API를 사용하여 짧은 코드(예, 되도록 1페이지 이내)로 간결한 서비스를 만들어 주시면 됩니다. 단, docker를 반드시 사용하여야 하고, 본인이 만든 서비스라는 독창성이 있어야 합니다.
# Q2. Ainize가 아닌 다른 곳에서 제공하는 GPT-2 API나 다른 곳에 Docker Container를 배포해도 되나요?

# 다른 API를 사용하시거나, 직접 만든 API를 사용해도 무방합니다. Ainize를 대신하여 Docker Container를 배포 할만한 Cloud 서비스로는 다음과 같은 것들을 고려해 보실 수 있습니다:
# Azure - https://azure.microsoft.com/
# GCP - https://cloud.google.com/
# Oracle - https://www.oracle.com/kr/cloud/free/

from urllib import response
from flask import Flask,request,jsonify,render_template
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

import time
app=Flask(__name__)

print("model loading...")

tokenizer=AutoTokenizer.from_pretrained('./persona_text')
model=AutoModelForCausalLM.from_pretrained('./persona_text')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def text_maker(text):
    try: 
        length=30
        text="Main : "+text
        name_input=tokenizer.encode(text,return_tensors='pt')
        name_input = name_input.to(device)
        min_length = len(name_input.tolist()[0])
        length = length if length > 0 else 1
        length += min_length
        outputs=model.generate(name_input,pad_token_id=50256,
                                max_length=length,
                                min_length=min_length,
                                do_sample=True,
                                top_k=40,
                                num_return_sequences=1)
        result=dict()
        for idx, sample_output in enumerate(outputs):
            result[0] = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
        return result.get(0)
    except Exception as e:
        print('Error occur in script generating!', e)

@app.route('/gen', methods=['GET','POST'])
def generate():
    if request.method == 'POST':
        text=request.form.get('username',"No Input")
        return render_template('main.html',username=str(text_maker(text)))
    return render_template('main.html')

@app.route('/')
def main():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=9999)