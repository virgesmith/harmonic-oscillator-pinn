from PIL import Image

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.losses import Loss

import matplotlib.pyplot as plt
#from tempfile import TemporaryDirectory

def save_gif_PIL(outfile, files, fps=5, loop=0):
  "Helper function for saving GIFs"
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

m, mu, k = 1., 4., 400.

def oscillator(d, w0, x):
  """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
  Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
  assert d < w0
  w = tf.sqrt(w0**2-d**2)
  phi = tf.atan(-d/w)
  A = 1/(2*tf.cos(phi))
  cos = tf.cos(phi+w*x)
  #sin = np.sin(phi+w*x)
  exp = tf.exp(-d*x)
  y  = exp*2*A*cos
  return y


def training_data():

  #d, w0 = 2, 20
  d, w0 = mu / (2 * m), np.sqrt(k / m)

  # get the analytical solution over the full domain
  x = tf.linspace(0,1,500)#.view(-1,1)
  y = oscillator(d, w0, x)#.view(-1,1)
  print(x.shape, y.shape)

  # slice out a small number of points from the LHS of the domain
  x_data = x[0:200:10]
  y_data = y[0:200:10] # .reshape((-1, 1))
  print(x_data.shape, y_data.shape)
  return x, y, x_data, y_data

class CustomLoss(Loss):
  def __init__(self):
    super().__init__()

  # why doesnt this work?
  def call(self, y_true, y_pred, sample_weight=None):
    e = y_pred - y_true
    # RMSE
    # return kb.sqrt(kb.mean(e ** 2))
    # MSE
    #return kb.mean(e ** 2)
    return tf.reduce_mean(tf.square(e)) + physics_loss()
    #rmse = tf.math.sqrt(mse)
    #return rmse / tf.reduce_mean(tf.square(y_true)) - 1
    # return kb.mean(e ** 2)


def physics_loss():         # compute the "physics loss"

  # Doesn't work like the pytorch original
  # yh_p = nn.nn.call(x, training=False)
  # dydx = tf.cast(tf.gradients(yh_p, x), tf.float32)
  # d2ydx2 = tf.gradients(dydx, x)

  # Doesn't work like the pytorch original
  with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
      tape1.watch(x)
      yh_p = nn.nn.call(x, training=False)
    dydx = tape1.gradient(yh_p, x)
  d2ydx2 = tape2.gradient(dydx, x)


  # dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
  # dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
  # dydx = tf.cast(tf.gradients(yh_p, x)[0], tf.float32)
  # print(dydx)
  # print(dydx.shape)
  # print(dydx.dtype)
  # print(type(dydx))
  # stop
  # d2ydx2 = tf.gradients(dydx, x)   #
  # dydx = (yh_p[2:] - yh_p[:-2]) / (2 * dx)
  # d2ydx2 = (yh_p[2:] - 2* yh_p[1:-1] + yh_p[:-2]) / (dx * dx)

  physics = d2ydx2 + tf.multiply(dydx, mu) + tf.cast(tf.multiply(yh_p, k), tf.float64)
  #physics = d2ydx2 + mu * dydx + k * yh_p # computes the residual of the 1D harmonic oscillator differential equation
  return tf.cast(1e-4 * tf.reduce_mean(physics ** 2), tf.float32)


# def custom_loss(y, yh):
#   e = y - yh
#   return tf.reduce_mean(tf.square(e)) # + physics_loss()

class FCN:
  "Defines a connected network"

  def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
    activation = 'tanh'
    self.nn = Sequential()

    self.nn.add(Input(shape=(1,)))
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
    #self.nn.add_loss(custom_loss)

    self.nn.compile(optimizer='adam', loss=CustomLoss()) #custom_loss)

    # kb.set_value(self.nn.optimizer.learning_rate, 0.01)
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
n_epochs = 20000
step = 1000
for i in range(0, n_epochs, step):

  nn.nn.fit(x_train, y_train, epochs=step, verbose=0)

  yh = np.squeeze(nn.nn.predict(x)) #tf.reshape(x, (-1,1))))
  plt.suptitle(f"{(i+step)}")
  g[0].set_ydata(yh)
  fig.canvas.flush_events()

