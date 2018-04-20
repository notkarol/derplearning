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
		self.four_corners_mm = np.array([
			[-self.test0.cam_far_rad, self.test0.cam_far_rad, 
			  self.test0.cam_near_rad, -self.test0.cam_near_rad],
			[self.test0.cam_max_range, self.test0.cam_max_range,
			 self.test0.cam_min_range, self.test0.cam_min_range] ] )
		

	def tearDown(self):
		pass

	def test_unit_vector(self):
		npt.assert_almost_equal(self.test0.unit_vector([345.345, 0]), np.array([1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, 345.345]), np.array([0., 1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-345.345, 0]), np.array([-1., 0.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([0, -345.345]), np.array([0., -1.]) , decimal=7)
		npt.assert_almost_equal(self.test0.unit_vector([-3, -4]), np.array([-.6, -.8]) , decimal=7)

	def test_cart2Spherical(self):
		four_corners3d_mm = np.zeros( (4, 3) )
		four_corners3d_mm[:, 0] = self.four_corners_mm[1, :]
		four_corners3d_mm[:, 1] = -self.four_corners_mm[0, :]
		four_corners3d_mm[:, 2] = self.test0.cam_height

		four_corners_angles = np.zeros( (4, 2) )
		four_corners_angles[:, 1] = [self.test0.cam_arc_x/2, -self.test0.cam_arc_x/2,
										-self.test0.cam_arc_x/2, self.test0.cam_arc_x/2 ]
		four_corners_angles[:, 0] = [np.pi/2 - self.test0.cam_vlim_crop_y,
									 np.pi/2 - self.test0.cam_vlim_crop_y,
									 np.pi/2 - self.test0.cam_to_ground_arc,
									 self.test0.cam_tilt_y - self.test0.cam_arc_y]


		npt.assert_almost_equal(self.test0.cart2Spherical(np.array([[1,0,0]]) ), [[1, 0, 0]])
		npt.assert_almost_equal(self.test0.cart2Spherical(
			np.array([[0,0,2]]) ), [[2, np.pi/2, 0]], decimal=4)
		npt.assert_almost_equal(self.test0.cart2Spherical(
			np.array([[0,2,0]]) ), [[2, 0, np.pi/2]], decimal=4)
		npt.assert_almost_equal(self.test0.cart2Spherical( four_corners3d_mm )[:,1:], four_corners_angles, decimal=4)

	#checks to make sure the named function generates the appropriate rotational matrix
	def test_rot_by_vector(self):
		t_a0 = np.array([ 1, 0])
		t_a1 = np.array([ 0, 1])
		t_a2 = np.array([-1, 0])
		t_a3 = np.array([ 0,-1])
		t_a4 = np.array([22, 0])
		t_a5 = np.array([ 0, 5])
		t_a6 = np.array([ 3, 4])
		
		npt.assert_almost_equal(self.test0.rot_by_vector(t_a2, t_a3), t_a1 , decimal=7)
		npt.assert_almost_equal(self.test0.rot_by_vector(t_a5, t_a4), np.array([0, -22]) , decimal=7)
		npt.assert_almost_equal(self.test0.rot_by_vector(t_a6, t_a5), [4, 3] , decimal=7)

	def test_xz_to_xy(self):

		#Verify that the four corners are correctly mapped.
		four_corners_mm = np.array([
			[-self.test0.cam_far_rad, self.test0.cam_far_rad, 
			  self.test0.cam_near_rad, -self.test0.cam_near_rad],
			[self.test0.cam_max_range, self.test0.cam_max_range,
			 self.test0.cam_min_range, self.test0.cam_min_range] ] )
		four_corners_pix = np.array([
			[0, self.test0.view_res[0], self.test0.view_res[0], 0], 
			[0, 0, self.test0.view_res[1], self.test0.view_res[1] ] ])

		bot_right_mm = np.array([ [self.test0.cam_near_rad], [self.test0.cam_min_range]] )
		bot_right_pix = np.array([ [self.test0.view_res[0] ], [self.test0.view_res[1]] ] )

		bot_left_mm =  np.array([ [-self.test0.cam_near_rad], [self.test0.cam_min_range]] )
		bot_left_pix = np.array([ [0 ], [self.test0.view_res[1]] ] )

		top_right_mm = np.array([ [self.test0.cam_far_rad], [self.test0.cam_max_range]] )
		top_right_pix = np.array([ [self.test0.view_res[0] ], [0] ] )

		top_left_mm = np.array([ [-self.test0.cam_far_rad], [self.test0.cam_max_range]] )
		top_left_pix = np.array([ [0 ], [0] ] )

		npt.assert_almost_equal(self.test0.xz_to_xy(four_corners_mm), four_corners_pix , decimal=0)
		npt.assert_almost_equal(self.test0.xz_to_xy(bot_right_mm), bot_right_pix , decimal=0)
		npt.assert_almost_equal(self.test0.xz_to_xy(bot_left_mm), bot_left_pix , decimal=0)
		npt.assert_almost_equal(self.test0.xz_to_xy(top_right_mm), top_right_pix , decimal=0)
		npt.assert_almost_equal(self.test0.xz_to_xy(top_left_mm), top_left_pix , decimal=0)


if __name__ == '__main__':
	unittest.main()