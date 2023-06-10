from flask import Flask, request
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

def extractive_summarizer(text):
    # Load pre-trained model and tokenizer
    m=len(text)
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=7, length_penalty=2.0, max_length=m*0.8, min_length=90, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Load the model from the pickle file
model_path = 'extractive_summarizer.pkl'
with open(model_path, 'rb') as f:
    loaded_summarizer = pickle.load(f)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    summary = loaded_summarizer(text)
    return summary

if __name__ == '__main__':
    app.run()
