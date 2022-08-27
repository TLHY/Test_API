from flask import Flask,request,jsonify,render_template
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from waitress import serve

app=Flask(__name__)
print("model loading...")

#Load Model

tokenizer=AutoTokenizer.from_pretrained('./persona_text')
model=AutoModelForCausalLM.from_pretrained('./persona_text')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Make user input as Model input
def text_maker(text):
    text=text+'\n'
    print("text making..")
    try: 
        length=100
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

#Generate text from user input
@app.route('/gen', methods=['GET','POST'])
def generate():
    print("generating")
    if request.method == 'POST':
        text=request.form.get('myText',"No Input")
        output=str(text_maker(text)).replace(str(text),'')
        print(output)
        return render_template('main.html',Main_Text="Main: "+text,NPC_Text=output+"\n")
    return render_template('main.html')

@app.route('/')
def main():
    print("service started")
    return render_template('main.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=9999)
