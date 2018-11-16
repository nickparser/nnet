from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from PIL import Image, ImageDraw
from numpy import array

BRUSH_DIAMETER = 15
BRUSH_COLOR = 0 

class PaintWidget(Widget):
	def __init__(self, **kwargs):
		super(PaintWidget, self).__init__(**kwargs)
		self.clear_canvas()

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0., 1., 0.)
			Ellipse(pos = (touch.x - BRUSH_DIAMETER / 2, touch.y - BRUSH_DIAMETER / 2), 
					size = (BRUSH_DIAMETER, BRUSH_DIAMETER))
			touch.ud['line'] = Line(points = (touch.x, touch.y), width = BRUSH_DIAMETER)
		self._draw.line(touch.ud['line'].points, fill = 255, width = BRUSH_DIAMETER)

	def on_touch_move(self, touch):
		touch.ud['line'].points += [touch.x, touch.y]
		self._draw.line(touch.ud['line'].points, fill = 255, width = BRUSH_DIAMETER)


	def clear_canvas(self):
		self.canvas.clear()
		self._model = Image.new('L', (self.width, self.height), 0)
		self._draw = ImageDraw.Draw(self._model)

	def get_prepared_data(self, shape):
		self._model = self._model.transpose(Image.FLIP_TOP_BOTTOM)
		return array(self._model.resize(shape, Image.ANTIALIAS).getdata())




