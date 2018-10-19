from flask import Flask, render_template
from views import data
app = Flask(__name__)

graph_view = data.Graphs()

@app.route('/dashboard_hello_world')
def dashboard():
  return render_template('dashboard_hello_world.html')


@app.route('/steer_graph')
def raw_graph():
  return graph_view.raw()


@app.route('/data_overlays')
def graph_display_dummy():
  return render_template('data_overlays.html', x_axis=[505.120, 505.282, 505.417, 505.552], data_set=[-0.001093, 0.125391, 0.127344, 0.1625])


if __name__ == '__main__':
  app.run()
