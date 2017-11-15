import os
import sys
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from PySide.phonon import Phonon
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import cv2

class MyVertex(QGraphicsItem):

    def __init__(self):

        self.edge = None

        QGraphicsItem.__init__(self)
     
        self.setFlag(QGraphicsItem.ItemIsMovable)    
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(self.DeviceCoordinateCache)

        p1 = QPointF(-5,-5)
        p2 = QPointF(5,5)
        self.rect = QRectF(p1, p2)

    def paint(self, painter, option, widget):
        painter.drawRect(self.rect)

    def itemChange(self, change, value):
        # notify the parent(aka line) to update 
        p = self.parentItem()
        if p:
            p.prepareGeometryChange()

        return QGraphicsItem.itemChange(self, change, value)

    def boundingRect(self):
        return self.rect

    def mousePressEvent(self, event):
        self.update()
        QGraphicsItem.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.update()
        QGraphicsItem.mouseReleaseEvent(self, event)

class MyEdge(QGraphicsItem):

    def __init__(self, x1, y1, x2, y2, color):
        QGraphicsItem.__init__(self)        
        self.setAcceptedMouseButtons(Qt.NoButton)

        self.pen = QPen(color, 1, Qt.DashLine)

        self.v1 = MyVertex()
        self.v2 = MyVertex()
        self.v1.setParentItem(self)
        self.v2.setParentItem(self)
        self.v1.setPos(x1, y1)
        self.v2.setPos(x2, y2)
        pass

    def boundingRect(self):    
        r = QRectF()
        if self.v1 and self.v2:
            p1 = self.v1.pos()
            p2 = self.v2.pos()
            s = QSizeF(p2.x() - p1.x(), p2.y() - p1.y());
            r = QRectF(p1, s).normalized()
      
        return r

    def paint(self, painter, option, widget):
        if  self.v1 and self.v2:
            painter.setPen(self.pen)
            p1 = self.v1.pos()
            p2 = self.v2.pos()
            painter.drawLine(p1,p2)

class MyGroundPlane(QGraphicsItem):

    def __init__(self):
        QGraphicsItem.__init__(self);

        ctx = QGLContext.currentContext()
        if not ctx:
            raise Exception('no current gl context')

        # generate the ground plane grid vbo
        gp = np.zeros(shape=(9*2*2, 3+3), dtype=np.float32)
        for i in range(9):
            gp[i*2+0,:]=[-4.0+i, -4.0, 0.0, 1.0, 0.0, 0.0]
            gp[i*2+1,:]=[-4.0+i, +4.0, 0.0, 1.0, 0.0, 0.0]
        for i in range(9):
            gp[9*2+i*2+0,:]=[-4.0, -4.0+i, 0.0, 0.0, 1.0, 0.0]
            gp[9*2+i*2+1,:]=[+4.0, -4.0+i, 0.0, 0.0, 1.0, 0.0]

        self.ground_plane_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
        self.ground_plane_vbo.setUsagePattern(QGLBuffer.StreamDraw)
        self.ground_plane_vbo.create()
        self.ground_plane_vbo.bind()
        self.ground_plane_vbo.allocate(gp.tostring())
        self.ground_plane_vbo.release()

        # coordonate axes lines
        ca = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 5.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        self.coord_axes_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
        self.coord_axes_vbo.setUsagePattern(QGLBuffer.StreamDraw)
        self.coord_axes_vbo.create()
        self.coord_axes_vbo.bind()
        self.coord_axes_vbo.allocate(ca.tostring())
        self.coord_axes_vbo.release() 

        # coordonate points
        cp = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                       [5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 5.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        self.coord_pts_vbo = QGLBuffer(QGLBuffer.VertexBuffer)
        self.coord_pts_vbo.setUsagePattern(QGLBuffer.StreamDraw)
        self.coord_pts_vbo.create()
        self.coord_pts_vbo.bind()
        self.coord_pts_vbo.allocate(cp.tostring())
        self.coord_pts_vbo.release() 
        
        vs ="#version 420\n" +\
            "in vec3 vert_pos;\n" +\
            "in vec3 vert_color;\n" +\
            "uniform mat4 mvp;\n" +\
            "out vec4 color;\n" +\
            "void main() {\n" +\
            "   gl_Position = mvp * vec4(vert_pos,1.0);" +\
            "   color = vec4(vert_color,1.0);" +\
            "}\n"
                  
        fs ="#version 420\n" +\
            "in vec4 color;\n" +\
            "out vec4 frag_color;\n" +\
            "void main() {\n" +\
            "   frag_color = color;" +\
            "}\n"

        self.shader_program = QGLShaderProgram()
        if self.shader_program.addShaderFromSourceCode(QGLShader.Vertex, vs) and \
            self.shader_program.addShaderFromSourceCode(QGLShader.Fragment, fs):
            if self.shader_program.link() and \
                self.shader_program.bind():
                # bam
                pass
            else:
                raise Exception("error link|bind")
        else:
            raise Exception("error add shader")

        self.shader_program.release()

        return

    def boundingRect(self):    
        return QRectF(0,0,640,480)

    def render_old_style(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, 4.0/3.0, 1.0, 20.0)
        
        glMatrixMode(GL_MODELVIEW)
        
        glTranslatef(0.0, 0.0, -15.0)
        glRotatef(-45, 1.0, 0.0, 0.0)
        glRotatef(-15, 0.0, 0.0, 1.0)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        glColor3f(1.0,0.0,0.0)
        for i in range(9):
            glVertex3f((-4.0 + i), -4.0, 0.0)        
            glVertex3f((-4.0 + i), 4.0, 0.0)
        
        glColor3f(0.0,1.0,0.0)
        for i in range(9):
            glVertex3f(-4.0, (-4.0 + i), 0.0)        
            glVertex3f(4.0, (-4.0 + i), 0.0)
        
        glEnd()
        
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0,0,0)
        glVertex3f(1,0,0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0,0,0)
        glVertex3f(0,1,0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0,0,0)
        glVertex3f(0,0,1)
        glEnd()
        
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glPointSize(1.0)
        
        #glBegin(GL_TRIANGLES)
        #glVertex3f(-1.0, -1.0, 0.0)
        #glVertex3f(1.0, -1.0, 0.0)
        #glVertex3f(0.0, 1.0, 0.0)
        #glEnd()

        #glBegin(GL_TRIANGLES)
        #glVertex3f(-1.0, -1.0, 0.0)
        #glVertex3f(1.0, -1.0, 0.0)
        #glVertex3f(0.0, 1.0, 0.0)
        #glEnd()
        return

    def render_new_style(self):
        if self.shader_program.bind():
           
            self.shader_program.enableAttributeArray("vert_pos")  
            self.shader_program.enableAttributeArray("vert_color")  
            
            
            self.ground_plane_vbo.bind()         
            self.shader_program.setAttributeBuffer("vert_pos", GL_FLOAT, 0, 3, 6*4)
            self.shader_program.setAttributeBuffer("vert_color", GL_FLOAT, 3*4, 3, 6*4)
            self.ground_plane_vbo.release()

            P = QMatrix4x4()
            P.setToIdentity()
            P.perspective(45.0, 4.0/3.0, 1.0, 20.0)
            M = QMatrix4x4()
            M.setToIdentity()
            V = QMatrix4x4()
            V.setToIdentity()
            V.translate(0.0, 0.0, -15.0)
            V.rotate(-45.0, 1.0, 0.0, 0.0)
            V.rotate(-15.0, 0.0, 0.0, 1.0)

            MVP = P*V*M
            self.shader_program.setUniformValue("mvp", MVP)

            glLineWidth(1.0)
            glDrawArrays(GL_LINES, 0, 9*2*2)

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

            self.shader_program.disableAttributeArray("vert_pos")
            self.shader_program.disableAttributeArray("vert_color")
            self.shader_program.release()
        return

    def paint(self, painter, option, widget):

        painter.beginNativePainting()

        #self.render_old_style()
        self.render_new_style()

        painter.endNativePainting()

        return

class MyGui(QWidget):

    def __init__(self, video_file):
        QWidget.__init__(self)
       

        self.video_file = video_file

        self.videoPlayer = Phonon.VideoPlayer()        
        self.graphicsScene = QGraphicsScene()          
        self.graphicsView = QGraphicsView()

        glw = QGLWidget()
        glctx = glw.context()

       

        self.graphicsView.setViewport(glw)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.graphicsView.setScene(self.graphicsScene)
       
        # using opencv to get the video width and height
        vcap = cv2.VideoCapture(self.video_file)
        w = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        h = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        vcap.release()      

        # need to call this so the ground plane item
        # can have access to a initializa gl context
        glw.glInit()
        glw.makeCurrent()
        self.groundPlaneItem = MyGroundPlane()
        
        self.vanisingLines1 = [
                MyEdge(100,100,100,200, Qt.blue),
                MyEdge(200,100,200,200, Qt.blue)
                ]

        self.vanisingLines2 = [
                MyEdge(100,100,200,100, Qt.red),
                MyEdge(100,200,200,100, Qt.red)
                ]

        proxy = self.graphicsScene.addWidget(self.videoPlayer)
        self.graphicsScene.addItem(self.groundPlaneItem )
        for vline in self.vanisingLines1:
            self.graphicsScene.addItem(vline)
        for vline in self.vanisingLines2:
            self.graphicsScene.addItem(vline)

        self.videoPlayer.load(Phonon.MediaSource(self.video_file))
        self.videoPlayer.play()       
        
        self.setFixedSize(w,h)
        self.graphicsView.setParent(self)
  
        self.videoPlayer.setGeometry(0,0,w,h)            
        self.graphicsScene.setSceneRect(0, 0, w, h)  
        #self.graphicsView.setSceneRect(0,0,w,h)
             
        self.graphicsView.move(0,0)

        
       
        return 
 

if(__name__ == '__main__'):

    app = QApplication([])

    video_file = r'x:\DEV\Traffic\6628_h264_1_640x480.avi'

    w = MyGui(video_file)
    w.show();

    app.exec_()