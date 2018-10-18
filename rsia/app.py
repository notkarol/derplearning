from flask import Flask, render_template
app = Flask(__name__)


@app.route('/dashboard_hello_world')
def dashboard():
  return render_template('dashboard_hello_world.html')


@app.route('/data_overlays')
def graph_display():
  return render_template('data_overlays.html', timeStamps=[505.120, 505.282, 505.417, 505.552], dataSet0=[-0.001093, 0.125391, 0.127344, 0.1625])


if __name__ == '__main__':
  app.run()
