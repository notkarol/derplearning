import unittest
import numpy as np
import numpy.testing as npt
from roadgen import Roadgen as rg


class Test_Roadgen(unittest.TestCase):
	"""docstring for Test_Roadgen"""
	#Setup the test instance of the class
	def setUp(self): 
		self.test0 = rg()

	def tearDown(self):
		pass

	def test_unit_vector(self):
		npt.assert_almost_equal(self.test0.unit_vector([345.345, 0]), np.array([1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, 345.345]), np.array([0., 1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-345.345, 0]), np.array([-1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, -345.345]), np.array([0., -1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-3, -4]), np.array([-.6, -.8]) , decimal=7)



if __name__ == '__main__':
	unittest.main()