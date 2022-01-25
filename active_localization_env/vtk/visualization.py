'''
Wrappers for some VTK geometrical objects.
'''
# Imports
import numpy as np
import vtk
from scipy.spatial.transform import Rotation


class Ellipsoid:
    def __init__(self, renderer, scale=1., color='Yellow'):
        # VTK intialization - create a new source, the ellipsoid for state belief
        self._params = vtk.vtkParametricEllipsoid()
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(self._params)
        source.Update()
        # Render
        self._actor = renderer.add_object(source, color=color, pipeline=True)
        self._actor.SetScale(scale)
        # Ellipsoid axis lengths
        self.axes = np.zeros(3)
        # Ellipsoid pose
        self.position = np.zeros(3)
        self.orientation = Rotation.from_quat([0, 0, 0, 1])

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, ax):
        self._axes = ax
        self._params.SetXRadius(ax[0])
        self._params.SetYRadius(ax[1])
        self._params.SetZRadius(ax[2])

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, x):
        self._position = x
        self._actor.SetPosition(x)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, rot):
        self._orientation = rot
        # VTK convention is ZXY orientation
        eul = rot.as_euler('ZXY', degrees=True)
        # But it still takes input in xyz order
        self._actor.SetOrientation(eul[1], eul[2], eul[0])


class Text:
    def __init__(self, renderer, pos=(10, 10), color='White', font_size=18):
        # VTK intialization
        self._actor = vtk.vtkTextActor()
        self._actor.SetDisplayPosition(*pos)
        renderer.renderer.AddActor(self._actor)
        # Properties
        prop = self._actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(font_size)
        prop.SetColor(vtk.vtkNamedColors().GetColor3d(color))
        # Initialize text
        self.text = ''

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, s):
        self._text = s
        self._actor.SetInput(s)


class Point:
    def __init__(self, renderer, size=5, color='Black'):
        # Coordinates
        self._x = np.zeros(3)
        # VTK intialization
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        id_ = points.InsertNextPoint(self._x)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id_)
        # create the source/data
        point = vtk.vtkPolyData()
        point.SetPoints(points)
        point.SetVerts(vertices)
        # Add to scene
        self._actor = renderer.add_object(point, color=color)
        # Size
        self._actor.GetProperty().SetPointSize(size)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, xi):
        self._x = xi
        self._actor.SetPosition(xi)