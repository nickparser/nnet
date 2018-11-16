from kivy.app import App
from kivy.config import Config

from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

from paint import PaintWidget
from nnet import NNet

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PAINT_HEIGHT = 570
CONFIG_HEIGHT = 30

# Window configurations
Config.set('graphics', 'width', WINDOW_WIDTH)
Config.set('graphics', 'height', WINDOW_HEIGHT)
Config.set('graphics', 'resizable', False)

# Neural Network parameters
INPUT = 784
HIDDEN = 100
OUTPUT = 10

LR = 0.001
EPISODES = 15000
BATCH_SIZE = 128

MODEL_PATH = 'model/model.ckpt'

class Main(App):
	def build(self):
		self._init_view()
		main_l = BoxLayout(orientation = 'vertical')

		self.nnet = NNet(n_input = INPUT, n_hidden = HIDDEN, n_output = OUTPUT, learning_rate = LR)
		self._prepare_nnet()

		main_l.add_widget(self._paint_w)
		main_l.add_widget(self._conf_l)
		return main_l

	def _init_view(self):
		self._conf_l = BoxLayout(size_hint = (None, None), width = WINDOW_WIDTH, height = CONFIG_HEIGHT)
		self._paint_w = PaintWidget(size_hint = (None, None), width = WINDOW_WIDTH, height = PAINT_HEIGHT)
		self._clear_b = Button(text = 'clear'); self._clear_b.bind(on_press = self.clear)
		self._query_b = Button(text = 'query'); self._query_b.bind(on_press = self.query)

		self._conf_l.add_widget(self._clear_b)
		self._conf_l.add_widget(self._query_b)

	def _prepare_nnet(self):
		try:
			self.nnet.restore(MODEL_PATH)
		except:
			from tensorflow.examples.tutorials.mnist import input_data
			mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

			for episode in range(EPISODES):
				batch_input, batch_target = mnist.train.next_batch(BATCH_SIZE)

				if episode % 100 == 0: status = True
				else: status = False

				self.nnet.train(batch_input, batch_target, status)

			self.nnet.save(MODEL_PATH)


	def clear(self, instance):
		self._paint_w.clear_canvas()

	def query(self, instance):
		predict = str(self.nnet.predict(self._paint_w.get_prepared_data((28, 28)).reshape(1, INPUT) / 255)[0])
		Popup(title = 'predict', content = Label(text = predict), 
			size_hint = (None, None), size = (200, 200)).open()

if __name__ == '__main__':
	Main().run()