import numpy as np
from scipy.interpolate import CubicSpline


class CubicSpline1d:
    def __init__(self, knots, values):
        self.spline = CubicSpline(knots, values)

        self.knots = knots
        self.coefficients = self.spline.c

        self.pathlength = self.spline.x[-1]

    def get_spline(self):
        return self.spline
    
    def get_knots(self):
        return self.knots
    
    def get_coefficients(self):
        return self.coefficients
    
    def get_sub_spline_knots_and_coefficients_from_window_size(self, s, window_size):
        closest_knot = np.argmin(np.abs(self.spline.x - s))

        sub_knots = self.spline.x[closest_knot:min(closest_knot+window_size, len(self.spline.x))]
        sub_coefficients = self.spline.c[:, closest_knot:min(closest_knot+window_size-1, np.shape(self.spline.c)[1])]

        if len(sub_knots) < window_size:
            
            sub_knots = np.append(sub_knots, np.ones(window_size - len(sub_knots)) * sub_knots[-1])
            sub_coefficients = np.append(sub_coefficients, np.zeros((4, window_size -1 - np.shape(sub_coefficients)[1])), axis=1)

        return sub_knots, sub_coefficients
    
    def get_value_at_s(self, s):
        if s > self.pathlength:
            s = self.pathlength
            return np.array([
                np.dot(self.coefficients[0, -1], np.array([(s-self.knots[0, -1])**3, (s-self.knots[0, -1])**2, (s-self.knots[0, -1]), 1])),
                np.dot(self.coefficients[1, -1], np.array([(s-self.knots[1, -1])**3, (s-self.knots[1, -1])**2, (s-self.knots[1, -1]), 1]))
                ])
        
        s_knot = np.argmin(np.abs(self.knots - s))
        
        return np.array([
                np.dot(self.coefficients[0, s_knot], np.array([(s-self.knots[0, s_knot])**3, (s-self.knots[0, s_knot])**2, (s-self.knots[0, s_knot]), 1])),
                np.dot(self.coefficients[1, s_knot], np.array([(s-self.knots[1, s_knot])**3, (s-self.knots[1, s_knot])**2, (s-self.knots[1, s_knot]), 1]))
            ])
    
    def get_derivative_at_s(self, s):
        return np.array([self.spline_x.derivative()(s), self.spline_y.derivative()(s)])
    
    def get_second_derivative_at_s(self, s):
        return np.array([self.spline_x.derivative()(s), self.spline_y.derivative()(s)])