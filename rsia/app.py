from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/dashboard_broken')
def dashboard():
    return render_template('dashboard_broken.html')

@app.route('/dashboard_small')
def dashboard2():
    return render_template('dashboard_small.html')

if __name__ == '__main__':
    app.run()