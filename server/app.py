from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/init')
def init():
    response = "Initialized"
    return response, 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    app.run()
