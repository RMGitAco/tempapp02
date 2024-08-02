from flask import Flask, request, render_template
from huggingface_hub import InferenceClient

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    if request.method == 'POST':
        question = request.form['question']
        client = InferenceClient(
            "microsoft/Phi-3-mini-4k-instruct",
            token="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",
        )

        for message in client.chat_completion(
            messages=[{"role": "user", "content": question}],
            max_tokens=500,
            stream=True,
        ):
            response += message.choices[0].delta.content

    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
