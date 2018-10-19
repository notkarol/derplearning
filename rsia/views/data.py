from flask import render_template
# from data_controllers import csv_getter
import pandas
import numpy


class Graphs:
  def raw(self, path="../data/20180317-082223-arkanine"):
    full_path = path + '/state.csv'
    state_data_frame = pandas.read_csv(full_path)
    return render_template('data_overlays.html',
         x_axis=pandas.Series(numpy.floor(state_data_frame['timestamp'].values[100:400] * 1000 % 10000)).to_json(orient='values'),
         data_set=pandas.Series(state_data_frame['steer'].values[100:400] * 1000).to_json(orient='values'))
