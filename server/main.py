from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def editor():
    return render_template('editor.html')
