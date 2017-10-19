import unittest
import numpy as np
import numpy.testing as npt
from roadgen3d import Roadgen as rg
import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)


class Test_Roadgen3d(unittest.TestCase):
	"""docstring for Test_Roadgen"""
	#Setup the test instance of the class
	def setUp(self): 
		self.test0 = rg(cfg)

	def tearDown(self):
		pass

	def test_unit_vector(self):
		npt.assert_almost_equal(self.test0.unit_vector([345.345, 0]), np.array([1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, 345.345]), np.array([0., 1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-345.345, 0]), np.array([-1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, -345.345]), np.array([0., -1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-3, -4]), np.array([-.6, -.8]) , decimal=7)

	def test_cart2Spherical(self):
		npt.assert_almost_equal(self.test0.cart2Spherical(np.array([[1,0,0]]) ), [[1, 0, 0]])


if __name__ == '__main__':
	unittest.main()