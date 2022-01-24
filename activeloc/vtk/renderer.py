'''
Wrapper for vtkRenderer object.
'''
# Imports
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class Renderer:
    def __init__(self,
                 window_size=(1, 1),
                 offscreen=False,
                 ortho=False,
                 background='MidnightBlue'):
        # Set-up VTK renderer
        self.renderer = vtk.vtkRenderer()
        # Set-up window
        self._render_window = vtk.vtkRenderWindow()
        self._render_window.AddRenderer(self.renderer)
        self._render_window.SetSize(window_size[0], window_size[1])
        if offscreen:
            # Offscreen rendering
            self._render_window.SetOffScreenRendering(True)
        else:
            # Enable window interaction
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self._render_window)
            # Display coordinate axes
            self._widget = self._add_axes_widget()
        # Background color
        rgb = vtk.vtkNamedColors().GetColor3d(background)
        self.renderer.SetBackground(rgb)
        # Orthographic projection
        self.renderer.GetActiveCamera().SetParallelProjection(ortho)
        # Reset camera view
        self.reset_camera()

    def _add_axes_widget(self):
        '''
		Add coordinate axes to VTK window.
		'''
        widget = vtk.vtkOrientationMarkerWidget()
        # Add coordinate axes
        axes = vtk.vtkAxesActor()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.iren)
        # Adjust size
        widget.SetViewport(0.0, 0.0, 0.3, 0.3)
        widget.SetEnabled(True)
        widget.SetInteractive(False)
        return widget

    def reset_camera(self):
        '''
		Set default camera view.
		'''
        ## Isometric View
        # self.renderer.GetActiveCamera().Azimuth(40)
        # self.renderer.GetActiveCamera().Elevation(30)
        self.renderer.ResetCamera()

    def render(self):
        '''
		Wrapper for rendering current scene.
		'''
        self._render_window.Render()

    def add_object(self, obj, color='Red', pipeline=False):
        '''
		Render generic VTK object.
		pipeline: True for VTK objects that require pipeline connection.
		'''
        if isinstance(color, str):
            # Get RGB values
            color = vtk.vtkNamedColors().GetColor3d(color)
        # Generate basic mapper
        mapper = vtk.vtkPolyDataMapper()
        if not pipeline:
            mapper.SetInputData(obj)
        else:
            mapper.SetInputConnection(obj.GetOutputPort())
        # Basic actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        # Render
        self.renderer.AddActor(actor)
        return actor

    def export_scene(self, filename):
        '''
		Export current rendered state as VRML file.
		'''
        # Update scene
        self.render()
        # Export
        VRML = vtk.vtkVRMLExporter()
        VRML.SetRenderWindow(self._render_window)
        VRML.SetFileName(filename + '.vrml')
        print('Writing scene to {}'.format(filename + '.vrml'))
        VRML.Write()

    def render_to_image(self):
        '''
		Render current scene to image.
		'''
        # Update scene
        self.render()
        # Initialize filter for conversion
        win_im_filter = vtk.vtkWindowToImageFilter()
        win_im_filter.SetInput(self._render_window)
        win_im_filter.Update()
        # Convert to VTK array
        vtk_image = win_im_filter.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        # Convert image to numpy array
        numpy_image = vtk_to_numpy(vtk_array).reshape(height, width,
                                                      components)
        # Origin of image to upper left
        numpy_image = np.flip(numpy_image, axis=0)
        return numpy_image
