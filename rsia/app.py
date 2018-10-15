from flask import Flask, render_template
app = Flask(__name__)


@app.route('/dashboard_hello_world')
def dashboard():
  return render_template('dashboard_hello_world.html')


@app.route('/data_overlays')
def graph_display():
  return render_template('data_overlays.html')


if __name__ == '__main__':
  app.run()
