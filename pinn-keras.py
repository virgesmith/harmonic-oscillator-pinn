from PIL import Image

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as kb
from keras.losses import Loss

import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

def save_gif_PIL(outfile, files, fps=5, loop=0):
  "Helper function for saving GIFs"
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def oscillator(d, w0, x):
  """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
  Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
  assert d < w0
  w = np.sqrt(w0**2-d**2)
  phi = np.arctan(-d/w)
  A = 1/(2*np.cos(phi))
  cos = np.cos(phi+w*x)
  sin = np.sin(phi+w*x)
  exp = np.exp(-d*x)
  y  = exp*2*A*cos
  return y

def training_data():
  d, w0 = 2, 20

  # get the analytical solution over the full domain
  x = np.linspace(0,1,500)#.view(-1,1)
  y = oscillator(d, w0, x)#.view(-1,1)
  print(x.shape, y.shape)

  # slice out a small number of points from the LHS of the domain
  x_data = x[0:200:10].reshape((-1, 1))
  y_data = y[0:200:10].reshape((-1, 1))
  print(x_data.shape, y_data.shape)
  return x, y, x_data, y_data

  # plt.figure()
  # plt.plot(x, y, label="Exact solution")
  # plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
  # plt.legend()
  # plt.show()

class CustomLoss(Loss):
  def __init__(self):
    super().__init__()

  def __call__(self, y_true, y_pred, sample_weight=None):
    e = y_pred - y_true
    # RMSE
    # return kb.sqrt(kb.mean(e ** 2))
    # MSE
    #return kb.mean(e ** 2)
    return tf.reduce_mean(tf.square(e))
    #rmse = tf.math.sqrt(mse)
    #return rmse / tf.reduce_mean(tf.square(y_true)) - 1
    # return kb.mean(e ** 2)

def custom_loss(y, yh):
  e = y - yh
  #return kb.mean(e ** 2)
  return tf.reduce_mean(tf.square(e))

class FCN:
  "Defines a connected network"

  def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
    activation = 'tanh'
    self.nn = Sequential()

    self.nn.add(Dense(N_INPUT, activation=activation, input_shape=(None,1)))

    for i in range(N_LAYERS-1):
      self.nn.add(Dense(N_HIDDEN, activation=activation))

    self.nn.add(Dense(N_OUTPUT))

    # self.fcs = nn.Sequential(*[
    #                 nn.Linear(N_INPUT, N_HIDDEN),
    #                 activation()])
    # self.fch = nn.Sequential(*[
    #                 nn.Sequential(*[
    #                     nn.Linear(N_HIDDEN, N_HIDDEN),
    #                     activation()]) for _ in range(N_LAYERS-1)])
    #self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    self.nn.compile(optimizer='adam', loss=custom_loss)
    #self.nn.build()

  # def forward(self, x):
  #   x = self.fcs(x)
  #   x = self.fch(x)
  #   x = self.fce(x)
  #   return x

  # def summary(self):
  #   return self.nn.summary()

nn = FCN(1,1,32,3)

nn.nn.summary()

x, y, x_train, y_train = training_data()

plt.ion()
fig = plt.figure(constrained_layout=True) #, figsize=(8, 6))
plt.plot(x, y)
plt.plot(x_train, y_train, 'o')
plt.xlim(0.0, 1.0)
plt.ylim(-1.0, 1.0)
g = plt.plot(x, np.zeros(y.shape), '.')
for i in range(50):

  nn.nn.fit(x_train, y_train, epochs=20, verbose=0)

  yh = np.squeeze(nn.nn.predict(x.reshape(-1,1)))

  g[0].set_ydata(yh)
  fig.canvas.flush_events()

