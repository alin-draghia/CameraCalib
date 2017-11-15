""" docs """
import numpy as np
from numpy import matlib

class CameraModel(object):
    """description of class"""

    def __init__(self):     
        """TODO"""   
        
        self.width = 640.0
        self.height = 480.0
        
        self.cx = self.width/2.0
        self.cy = self.height/2.0
        
        
        self.fx = 580.0
        self.fy = self.fx

        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.zeros((3,))
        
        return

    def opengl_projection_matrix(self, near=0.1, far=1000.0):
        P = [0.0] * 16
        P[ 0] = 2.0 * self.fx / self.width
        P[ 1] = 0.0
        P[ 2] = 0.0
        P[ 3] = 0.0

        P[ 4] = 0.0
        P[ 5] = 2.0 * self.fy / self.height
        P[ 6] = 0.0
        P[ 7] = 0.0

        P[ 8] = -(2.0 * self.cx / self.width) + 1.0
        P[ 9] = -(2.0 * self.cx / self.width) + 1.0
        P[10] = -(far + near)/(far - near)
        P[11] = -2.0 * far * near/(far - near)

        P[12] = 0.0
        P[13] = 0.0
        P[14] =-1.0
        P[15] = 0.0

        return P

    def __rotation_matrix(self,angle, direction, point=None):
        import numpy, math
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = np.r_[direction[:3]] / np.linalg.norm(np.r_[direction[:3]])
        # rotation matrix around unit vector
        R = numpy.diag([cosa, cosa, cosa])
        R += numpy.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                          [ direction[2], 0.0,          -direction[0]],
                          [-direction[1], direction[0],  0.0]])
        M = numpy.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
            M[:3, 3] = point - numpy.dot(R, point)
       
        return M

    def opengl_view_matrix(self):

        import numpy, math

        R = self.R
        t = self.t
        V = np.row_stack((np.column_stack((R,t)),(0,0,0,1)))

        flip = np.eye(4)
        flip[1,1] = -1.0
        flip[2,2] = -1.0

        V = np.mat(flip) * np.mat(V)
        V = np.array(V)

        v = V.flatten().tolist()

        return v
   
    def world_to_image(self, Pw):
        """TODO"""
        Pc = self.world_to_camera(Pw)
        Pi = self.camera_to_image(Pc)
        return Pi

    def image_to_world(self, Pi, z=1.0):
        """TODO"""
        Pc = self.image_to_camera(Pi, z)
        Pw = self.camera_to_world(Pc)       
        return Pw

    def camera_to_world(self, Pc):
        """TODO"""
        R = self.Rt[:3,:3]
        t = self.Rt[:3, 4]
        Rinv = R.T #rotation matrix tranpose = inverse
        tinv = Rinv * t
        tinv *= -1.0
        
        Pw = Rinv * Pc * tinv
                       
        return Pw

    def world_to_camera(self, Pw):
        """TODO"""
        R = self.Rt[:3,:3]
        t = self.Rt[:3, 4]

        Pc = R * Pw * t

        return Pc

    def camera_to_image(self, Pc):
        """TODO"""
        Pi = (self.cx + (Pc[0] * self.fx)/Pc[2],
              self.cy + (Pc[1] * self.fy)/Pc[2])
        return Pi

    def image_to_camera(self, Pi, z=1.0):
        """TODO"""
        Pc = ((Pi[0] - self.cx)/self.fx * z,
              (Pi[1] - self.cy)/self.fy * z,
              z)

        return Pc
    

