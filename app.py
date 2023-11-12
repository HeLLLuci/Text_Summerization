from flask import Flask, render_template, request
from text_summary import summarizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_txt, len_original_txt, len_summary = summarizer(rawtext)
    return render_template('summary.html', summary=summary, original_txt = original_txt, len_original_txt=len_original_txt, len_summary=len_summary)




if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
