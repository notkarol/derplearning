from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello():
  return "Hello World!"

@app.route('/dashboard_hello_world')
def dashboard():
  return render_template('dashboard_hello_world.html')

if __name__ == '__main__':
  app.run()
