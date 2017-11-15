""" Guillow Calibration Algorithm """

import math

import numpy as np

from cam_calib.cam_model import CameraModel


class GuillouCalibrationEngine(CameraModel):
    """
    Implements 'Guillou' calibration algorithm using vanishing lines
    Assume image centered principal point P = (width/2, height/2)
    and sqare aspect ration fx == fy
    """

    def __init__(self):
        # pylint: disable=invalid-name

        CameraModel.__init__(self)

        # vanishing point, u vanishing lines intersection (x,y)
        self.vanishing_lines_x = None

        # vanishing point, v vanishing lines intersection (x,y)
        self.vanishing_lines_y = None

        # vanishing lines in 'u' direction [(x1,y1,x2,y2).....]
        self.vanishing_point_x = None

        # vanishing lines in 'v' direction [(x1,y1,x2,y2).....]
        self.vanishing_point_y = None

        self.A = None
        self.B = None
        self.a = None
        self.b = None

        return


    def _check_vanishing_lines_type(self, lines):
        # pylint: disable=R0201
        if not isinstance(lines, list):
            raise TypeError('lines must be a list of tuples')

        if len(lines) < 2:
            raise ValueError('lines must contain at least 2 lines')

        if not isinstance(lines[0], tuple):
            raise TypeError('lines must be of type tuple')

        if len(lines[0]) != 4:
            raise ValueError('a line must be defined as (x1,y1,x2,y2)')

    def _lines_intersect(self, lines):
        """
        Compute line intersection point Least-Sqare
        lines = [(x1,y1,x2,y2).....]
        returns (x,y)
        """
        # pylint: disable=invalid-name
        self._check_vanishing_lines_type(lines)

        A = []
        b = []
        for line in lines:
            p1 = np.r_[line[0:2], 1.]
            p2 = np.r_[line[2:4], 1.]
            (a1, a2, a3) = np.cross(p1, p2)
            A.append([a1, a2])
            b.append(-a3) # ax+by=-c line equation

        #x, r, rank, s = np.linalg.lstsq(A, b)
        x, _, _, _ = np.linalg.lstsq(A, b)
        return tuple(x)

    def _compute_focal_lenght(self):
        """ TODO: document """
        # pylint: disable=invalid-name
        Fx = self._lines_intersect(self.vanishing_lines_x)
        Fy = self._lines_intersect(self.vanishing_lines_y)
        P = (self.cx, self.cy)

        f = 0.
        #Puv is the ortogonal projection of P on the line FuFv
        dirFxFy = np.subtract(Fy, Fx) / np.linalg.norm(np.subtract(Fy, Fx))
        Pxy = np.add(Fy, np.dot(dirFxFy, np.subtract(P, Fy))*dirFxFy)

        normOPxy = np.sqrt(np.linalg.norm(np.subtract(Fy, Pxy)) *
                           np.linalg.norm(np.subtract(Pxy, Fx)))
        f = math.sqrt(normOPxy**2 - np.linalg.norm(np.subtract(P, Pxy))**2)
        return f

    def _compute_rotation_matrix(self):
        """ TODO: document """
        # pylint: disable=invalid-name
        Fx = self._lines_intersect(self.vanishing_lines_x)
        Fy = self._lines_intersect(self.vanishing_lines_y)

        P = (self.cx, self.cy)
        f = self.fx

        x = np.r_[np.subtract(Fx, P), f] # (Xu,Yu,f)
        y = np.r_[np.subtract(Fy, P), f]

        Rx = -x / np.linalg.norm(x)
        Ry = -y / np.linalg.norm(y)
        Rz = np.cross(Rx, Ry)

        R = np.column_stack((Rx, Ry, Rz))

        # check if matrix is valid ( det(RotationMatrix) == 1 )
        # todo: maybe decompose the matrix with svd and approximate
        # a valid matrix
        if math.fabs(1.0 - np.linalg.det(R)) > 0.001:
            raise ValueError('Rotation matrix is invalid det(RotationMatrix) != 1')
        return R


    def _compute_translation_vector(self):
        """ TODO: document """
        # pylint: disable=unreachable
        # pylint: disable=invalid-name
        return (0.0, 0.0, 20.0)

        A = self.A
        B = self.B
        a = self.a
        b = self.b

        R = self.R
        P = (self.cx, self.cy)
        f = self.fx

        A = np.dot(R, A)
        B = np.dot(R, B)

        normAB = np.linalg.norm(A-B)

        a = np.mat(np.r_[np.subtract(P, a), f]).T
        b = np.mat(np.r_[np.subtract(P, b), f]).T

        normOa = np.linalg.norm(a)


        # Y2 is the intersection of the line
        # OY with line D passing through X1
        # and whose direction is XY vector
        #dirOY1 = Y1/linalg.norm(Y1)
        #Y2 = dot(dirOY1,X1)*dirOY1
        #normX1Y2 = linalg.norm(X1-Y2)

        r1 = a
        e1 = B/normAB
        r2 = np.mat([0.0, 0.0, 0.0]).T
        e2 = b/np.linalg.norm(b)
        u = np.dot(e1.T, e2)[0, 0]
        t1 = np.dot((r2-r1).T, e1)[0, 0]
        t2 = np.dot((r2-r1).T, e2)[0, 0]
        d1 = (t1-u*t2)/(1.-u*u)
        d2 = (t1-u*t1)/(u*u-1.)
        p1 = r1 + d1 * e1
        p2 = r2 + d2 * e2 # pylint: disable=unused-variable
        Y2 = p1
        normX1Y2 = np.linalg.norm(a-Y2)

        normOX = (normOa * normAB)/normX1Y2
        OX = normOX * (a/normOa)

        T = OX
        #if(T[2] > 0.):
        #    T[2] *= -1
        #    print 'WARNING: swap T.z sign'

        #T[2] *= -1.0
        #T = np.dot(R.T,OX)

        tv = T.flatten().tolist()[0]

        # world camera coordonate =
        # C = -R.I*tv

        # for now return a mock vector

        return tuple(tv)

    def calibrate(self):
        """ todo: doc """
        try:
            fx = self._compute_focal_lenght()
            fy = self.fx
            R = self._compute_rotation_matrix()
            t = self._compute_translation_vector()
            self.fx = fx
            self.fy = fy
            self.R = R
            self.t = t
        except:
            pass

