""" todo: doc """

# pylint: disable=E0611
from PySide.QtCore import (Qt,
                           QPointF,
                           QRectF,
                           QSizeF)

from PySide.QtGui import (QWidget,
                          QApplication,
                          QGraphicsItem,
                          QPen,
                          QPixmap,
                          QGraphicsPixmapItem,
                          QGraphicsScene,
                          QMatrix4x4,
                          QGraphicsView)

from PySide.QtOpenGL import (QGLWidget,
                             QGLBuffer,
                             QGLShaderProgram,
                             QGLShader)

from OpenGL.GL import (GL_FLOAT,
                       GL_POINTS,
                       GL_LINES,
                       glPointSize,
                       glLineWidth,
                       glDrawArrays)

from cam_calib.guillou_calib import GuillouCalibrationEngine

import numpy as np


class CalibWidget(QWidget):
    """ TODO: document me """
    # pylint: disable=R0902

    class MyVertex(QGraphicsItem):
        """ TODO: document me """

        def __init__(self):

            QGraphicsItem.__init__(self)

            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
            self.setCacheMode(self.DeviceCoordinateCache)

            point1 = QPointF(-5, -5)
            point2 = QPointF(5, 5)
            self.rect = QRectF(point1, point2)
            return

        def paint(self, painter, option, widget):
            """ doc """
            # pylint: disable=W0613
            painter.drawRect(self.rect)
            return


        def itemChange(self, change, value):
            """ doc """
            # pylint: disable=invalid-name
            # notify the parent(aka line) to update
            p = self.parentItem()
            if p:
                p.prepareGeometryChange()

            return QGraphicsItem.itemChange(self, change, value)

        def boundingRect(self):
            """ doc """
            # pylint: disable=invalid-name
            return self.rect

        def mousePressEvent(self, event):
            """ doc """
            # pylint: disable=invalid-name
            self.update()
            QGraphicsItem.mousePressEvent(self, event)
            return

        def mouseReleaseEvent(self, event):
            """ doc """
            # pylint: disable=invalid-name
            self.update()
            QGraphicsItem.mouseReleaseEvent(self, event)
            return


    class MyEdge(QGraphicsItem):
        """ doc """

        def __init__(self, color):
            QGraphicsItem.__init__(self)
            self.setAcceptedMouseButtons(Qt.NoButton)

            self.pen = QPen(color, 2, Qt.DashDotDotLine)

            self.vertex1 = CalibWidget.MyVertex()
            self.vertex2 = CalibWidget.MyVertex()
            self.vertex1.setParentItem(self)
            self.vertex2.setParentItem(self)
            self.vertex1.setPos(0.0, 0.0)
            self.vertex2.setPos(0.0, 0.0)
            return

        def get_coords(self):
            """ returns ((x1,y1),(x2,y2)) """
            return (self.vertex1.pos().toTuple(), self.vertex2.pos().toTuple())

        def set_coords(self, vertex1, vertex2):
            """ doc """
            self.vertex1.setPos(*vertex1)
            self.vertex2.setPos(*vertex2)
            return

        def boundingRect(self):
            """ doc """
            # pylint: disable=invalid-name
            r = QRectF()
            if self.vertex1 and self.vertex2:
                p1 = self.vertex1.pos()
                p2 = self.vertex2.pos()
                s = QSizeF(p2.x() - p1.x(), p2.y() - p1.y())
                r = QRectF(p1, s).normalized()

            return r

        def paint(self, painter, option, widget):
            """ doc """
            # pylint: disable=unused-argument
            if  self.vertex1 and self.vertex2:
                painter.setPen(self.pen)
                point1 = self.vertex1.pos()
                point2 = self.vertex2.pos()
                painter.drawLine(point1, point2)



    class MyGroundPlane(QGraphicsItem):
        """
        TODO: document me
        """
        def __init__(self):
            QGraphicsItem.__init__(self)
            self.ground_plane_vbo = None
            self.coord_axes_vbo = None
            self.coord_pts_vbo = None
            self.shader_program = None
            # compute model view projection matrix
            proj_mat = QMatrix4x4()
            proj_mat.perspective(45.0, 4.0/3.0, 0.1, 1000.0)
            view_mat = QMatrix4x4()
            model_mat = QMatrix4x4()


            self.mvp_mat = proj_mat * view_mat * model_mat

            return


        def boundingRect(self):
            """ doc """
            # pylint: disable=invalid-name,R0201
            return QRectF(0, 0, 640, 480)

        def render_ground_plane_grid(self):
            """ doc """
            self.ground_plane_vbo.bind()
            self.shader_program.setAttributeBuffer("vert_pos", GL_FLOAT, 0, 3, 6*4)
            self.shader_program.setAttributeBuffer("vert_color", GL_FLOAT, 3*4, 3, 6*4)
            self.ground_plane_vbo.release()
            glLineWidth(1.0)
            glDrawArrays(GL_LINES, 0, 9*2*2)

        def render_coord_axes(self):
            """ doc """
            self.coord_axes_vbo.bind()
            self.shader_program.setAttributeBuffer("vert_pos", GL_FLOAT, 0, 3, 6*4)
            self.shader_program.setAttributeBuffer("vert_color", GL_FLOAT, 3*4, 3, 6*4)
            self.coord_axes_vbo.release()
            glLineWidth(2.0)
            glDrawArrays(GL_LINES, 0, 3*2)

            self.coord_pts_vbo.bind()
            self.shader_program.setAttributeBuffer("vert_pos", GL_FLOAT, 0, 3, 6*4)
            self.shader_program.setAttributeBuffer("vert_color", GL_FLOAT, 3*4, 3, 6*4)
            self.coord_pts_vbo.release()
            glPointSize(8.0)
            glDrawArrays(GL_POINTS, 0, 4)

        def render_new_style(self):
            """ doc """
            if self.shader_program.bind():

                self.shader_program.setUniformValue("mvp", self.mvp_mat)

                self.shader_program.enableAttributeArray("vert_pos")
                self.shader_program.enableAttributeArray("vert_color")

                self.render_ground_plane_grid()
                self.render_coord_axes()

                self.shader_program.disableAttributeArray("vert_pos")
                self.shader_program.disableAttributeArray("vert_color")
                self.shader_program.release()
            return

        def paint(self, painter, option, widget):
            """ doc """
            # pylint: disable=unused-argument
            painter.beginNativePainting()
            self.render_new_style()
            painter.endNativePainting()
            return

        def update_ground_plane_vbo(self):
            """ doc """
            ground_plane_grid = np.zeros(shape=(9*2*2, 3+3), dtype=np.float32)
            for i in range(9):
                ground_plane_grid[i*2+0, :] = [-4.0+i, -4.0, 0.0, 0.7, 0.0, 0.0]
                ground_plane_grid[i*2+1, :] = [-4.0+i, +4.0, 0.0, 0.7, 0.0, 0.0]
            for i in range(9):
                ground_plane_grid[9*2+i*2+0, :] = [-4.0, -4.0+i, 0.0, 0.0, 0.7, 0.0]
                ground_plane_grid[9*2+i*2+1, :] = [+4.0, -4.0+i, 0.0, 0.0, 0.7, 0.0]

            self.ground_plane_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
            self.ground_plane_vbo.setUsagePattern(QGLBuffer.StreamDraw)
            self.ground_plane_vbo.create()
            self.ground_plane_vbo.bind()
            self.ground_plane_vbo.allocate(ground_plane_grid.tostring())
            self.ground_plane_vbo.release()
            return

        def update_coord_axes_vbo(self):
            """ doc """
            coord_axes = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 5.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 5.0, 0.0, 0.0, 1.0]],
                                  dtype=np.float32)

            self.coord_axes_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
            self.coord_axes_vbo.setUsagePattern(QGLBuffer.StreamDraw)
            self.coord_axes_vbo.create()
            self.coord_axes_vbo.bind()
            self.coord_axes_vbo.allocate(coord_axes.tostring())
            self.coord_axes_vbo.release()
            return

        def update_coord_pts_vbo(self):
            """ doc """
            coord_pts = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  [5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                  [0.0, 5.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 5.0, 0.0, 0.0, 1.0]],
                                 dtype=np.float32)

            self.coord_pts_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
            self.coord_pts_vbo.setUsagePattern(QGLBuffer.StreamDraw)
            self.coord_pts_vbo.create()
            self.coord_pts_vbo.bind()
            self.coord_pts_vbo.allocate(coord_pts.tostring())
            self.coord_pts_vbo.release()
            return

        def update_shader_program(self):
            """ doc """
            vertex_shader = """
            #version 130
            in vec3 vert_pos;
            in vec3 vert_color;
            uniform mat4 mvp;
            out vec4 color;
            void main() {
                gl_Position = mvp * vec4(vert_pos,1.0);
                color = vec4(vert_color,1.0);
            }
            """

            fragment_shader = """
            #version 130
            in vec4 color;
            out vec4 frag_color;
            void main() {
                frag_color = color;
            }
            """

            self.shader_program = QGLShaderProgram()
            if self.shader_program.addShaderFromSourceCode(QGLShader.Vertex, vertex_shader) and \
                self.shader_program.addShaderFromSourceCode(QGLShader.Fragment, fragment_shader):
                if self.shader_program.link() and \
                    self.shader_program.bind():
                    # bam
                    pass
                else:
                    print self.shader_program.log()
                    raise Exception("error link|bind")
            else:
                print self.shader_program.log()
                raise Exception("error add shader")

            self.shader_program.release()
            return

        def update_gl_objects(self):
            """ doc """
            # generate the ground plane grid vbo
            self.update_ground_plane_vbo()
            # coordonate axes lines
            self.update_coord_axes_vbo()
            # coordonate points
            self.update_coord_pts_vbo()
            # make glsl shader program
            self.update_shader_program()
            return


    def __init__(self):
        QWidget.__init__(self)
        self.pixmap = QPixmap(800, 600)
        self.pixmap.fill(Qt.gray)
        self.calib_engine = None
        self.image_width = None
        self.image_height = None
        self.background_pixmap_item = None
        self.vanishing_lines_u = None
        self.vanishing_lines_v = None
        self.opengl_viewport = None
        self.opengl_context = None
        self.opengl_item = None
        self.graphics_scene = None
        self.graphics_view = None
        return

    def set_image(self, image_file):
        """ doc """
        self.pixmap = QPixmap(image_file)
        self.reset_scene()
        return

    def scene_changed(self, region):
        """ doc """
        # pylint: disable=unused-argument
        self.update_calibration()
        return

    def update_calibration(self):
        """ doc """
        # pylint: disable=invalid-name
        Lu1 = self.vanishing_lines_u[0]
        Lu2 = self.vanishing_lines_u[1]
        Lv1 = self.vanishing_lines_v[0]
        Lv2 = self.vanishing_lines_v[1]
        self.calib_engine.width = float(self.image_width)
        self.calib_engine.height = float(self.image_height)
        self.calib_engine.cx = self.image_width / 2.0
        self.calib_engine.cy = self.image_height / 2.0
        self.calib_engine.vanishing_lines_x = [Lu1.get_coords()[0]+Lu1.get_coords()[1],
                                               Lu2.get_coords()[0]+Lu2.get_coords()[1]]
        self.calib_engine.vanishing_lines_y = [Lv1.get_coords()[0]+Lv1.get_coords()[1],
                                               Lv2.get_coords()[0]+Lv2.get_coords()[1]]
        self.calib_engine.calibrate()

        proj_mat_vals = self.calib_engine.opengl_projection_matrix()
        view_mat_vals = self.calib_engine.opengl_view_matrix()
        proj_mat = QMatrix4x4(proj_mat_vals)
        view_mat = QMatrix4x4(view_mat_vals)
        model_mat = QMatrix4x4()

        #proj_mat.setToIdentity()
        #proj_mat.perspective(45.0, 1.0, 0.1, 1000.0)
        #view_mat.setToIdentity()
        #view_mat.translate(0.0, 0.0, -30.0)
        #view_mat.rotate(-50.0, 1.0, 0.0, 0.0)
        #view_mat.rotate(-10.0, 0.0, 0.0, 1.0)
        mvp = proj_mat * view_mat * model_mat
        self.opengl_item.mvp_mat = mvp

        # opengl support

    def reset_scene(self):
        """ doc """
        # pylint: disable=invalid-name
        # calibration algorithm
        self.calib_engine = GuillouCalibrationEngine()

        # graphics items

        self.image_width = self.pixmap.size().width()
        self.image_height = self.pixmap.size().height()
        self.background_pixmap_item = QGraphicsPixmapItem()
        self.background_pixmap_item.setPixmap(self.pixmap)
        self.opengl_item = CalibWidget.MyGroundPlane()


        Lu1 = CalibWidget.MyEdge(Qt.red)
        Lu2 = CalibWidget.MyEdge(Qt.red)

        Lu1.set_coords((0.1 * self.image_width, 0.3 * self.image_height),
                       (0.45 * self.image_width, 0.2 * self.image_height))
        Lu2.set_coords((0.1 * self.image_width, 0.7 * self.image_height),
                       (0.45 * self.image_width, 0.8 * self.image_height))

        self.vanishing_lines_u = [Lu1, Lu2]

        Lv1 = CalibWidget.MyEdge(Qt.green)
        Lv2 = CalibWidget.MyEdge(Qt.green)

        Lv1.set_coords((self.image_width - 0.12 * self.image_width, 0.3 * self.image_height),
                       (self.image_width - 0.45 * self.image_width, 0.2 * self.image_height))
        Lv2.set_coords((self.image_width - 0.10 * self.image_width, 0.7 * self.image_height),
                       (self.image_width - 0.45 * self.image_width, 0.8 * self.image_height))

        self.vanishing_lines_v = [Lv1, Lv2]

        self.update_calibration()

        # opengl support
        self.opengl_viewport = QGLWidget()
        self.opengl_context = self.opengl_viewport.context()

        # graphics scene support
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.changed.connect(self.scene_changed)
        self.graphics_view = QGraphicsView()

        # set the opengl backend for graphics view
        self.graphics_view.setViewport(self.opengl_viewport)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # setup graphics scene
        self.graphics_view.setScene(self.graphics_scene)

        # add the items to the scene in order
        self.graphics_scene.addItem(self.background_pixmap_item)
        self.graphics_scene.addItem(self.opengl_item)
        for xline in self.vanishing_lines_u:
            self.graphics_scene.addItem(xline)
        for yline in self.vanishing_lines_v:
            self.graphics_scene.addItem(yline)

        self.setFixedSize(self.image_width, self.image_height)
        self.graphics_view.setParent(self)

        #self.background_pixmap_item.setGeometry(0, 0, 640, 480)
        self.graphics_scene.setSceneRect(0, 0, self.image_width, self.image_height)

        self.graphics_view.move(0, 0)

        # initializa opengl
        self.opengl_viewport.glInit()
        # make the OPENGL context current for the ground plane item
        # to generate VBOs and complile GLSL program
        self.opengl_viewport.makeCurrent()
        self.opengl_item.update_gl_objects()
        self.opengl_viewport.doneCurrent()
        #self.camera_model.image_size(640,480)
        return


if __name__ == '__main__':

    def main():
        """ no docs """
        app = QApplication([])

        image_file = r'c:\Users\alin.draghia\Pictures\vlcsnap-00021.png'

        win = CalibWidget()
        win.set_image(image_file)
        win.reset_scene()
        win.show()

        app.exec_()

    main()
