""" Main program """
import sys

# pylint: disable=E0611
from PySide.QtGui import QApplication
from cam_calib.calib_widget import CalibWidget


def main():
    """ entry point """
    try:
        app = QApplication(sys.argv)

        image_file = r'data/Makati_intersection.jpg'
        window = CalibWidget()
        window.set_image(image_file)
        window.reset_scene()
        window.show()

        return app.exec_()

    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise


if __name__ == '__main__':
    main()
