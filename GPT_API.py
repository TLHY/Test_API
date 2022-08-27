from flask import Flask,request,jsonify,render_template
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from waitress import serve
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
     serve(app, host='0.0.0.0', port=80)
