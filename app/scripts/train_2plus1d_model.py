import numpy as np
import os
import keras
from keras import Model
import einops
import cv2
from sklearn.model_selection import train_test_split
# import tensorflowjs as tfjs

@keras.utils.register_keras_serializable()
class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding, **kwargs):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.
    """
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.seq = keras.models.Sequential([
        # Spatial decomposition
        keras.layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        keras.layers.Conv3D(filters=filters,
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)

  def get_config(self):
      config = super(Conv2Plus1D, self).get_config()
      config.update({
         'filters': self.filters,
         'kernel_size': self.kernel_size,
         'padding': self.padding
      })
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)

@keras.utils.register_keras_serializable()
class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size, **kwargs):
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size

    self.seq = keras.models.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        keras.layers.LayerNormalization(),
        keras.layers.ReLU(),
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        keras.layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

  def get_config(self):
      config = super(ResidualMain, self).get_config()
      config.update({
         'filters': self.filters,
         'kernel_size': self.kernel_size
      })
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)

@keras.utils.register_keras_serializable()
class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.seq = keras.models.Sequential([
        keras.layers.Dense(units),
        keras.layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

  def get_config(self):
        config = super(Project, self).get_config()
        config.update({
            'units': self.units
        })
        return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)

@keras.utils.register_keras_serializable()
def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters,
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return keras.layers.add([res, out])

@keras.utils.register_keras_serializable()
class ResizeVideo(keras.layers.Layer):
  def __init__(self, FINGERS, COORDINATES, **kwargs):
    super().__init__(**kwargs)
    self.FINGERS = FINGERS
    self.COORDINATES = COORDINATES
    self.resizing_layer = keras.layers.Resizing(self.FINGERS, self.COORDINATES)

  def call(self, video):
    """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new FINGERS and COORDINATES it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for FINGERS,
    # w stands for COORDINATES, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

  def get_config(self):
      config = super(ResizeVideo, self).get_config()
      config.update({
          'FINGERS': self.FINGERS,
          'COORDINATES': self.COORDINATES
      })
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)

def get_labels(folder_path):
    labels = []
    if os.path.isdir(folder_path):
        print('data path', os.listdir(folder_path))
        for entry in os.listdir(folder_path):
            labels.append(entry)

    return labels

FINGERS = 6
COORDINATES = 2
FRAME_COUNT = 30
input_shape = (None, FRAME_COUNT, FINGERS, COORDINATES, 1)
data_shape = (50, FRAME_COUNT, FINGERS, COORDINATES, 1)

data_path = os.path.join(os.getcwd(), "data/hand/")
labels = get_labels(data_path)
label_map = {label:num for num, label in enumerate(labels)}

X = np.empty((0, FRAME_COUNT, FINGERS, COORDINATES))  # Placeholder array

y = []

for label in labels:
    print("Processing label", label)

    label_path = os.path.join(data_path, label)
    

    for file_name in os.listdir(label_path):
        # print("file_name", file_name)
        file_path = os.path.join(label_path, file_name)
        example = np.load(file_path)

        # frames = np.expand_dims(frames, axis=-1)
        # resized_frames = []
        # for frame in frames:
        #   resized_frame = cv2.resize(frame,(50, 37))
        #   resized_frames.append(resized_frame)
        X = np.concatenate((X, np.expand_dims(example, axis=0)))
        # map labels to numerical values because string values aren't allowed in training
        y.append(label_map[label])

# X = np.stack(X, axis=0)
# X = np.expand_dims(X, axis=-1)
print('X', X)
y = np.array(y)

print("x shape", X.shape)
print("y shape", y.shape)
print("labels", label_map)

# Assuming X is your feature set and y is the corresponding labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

print("x shape", X.shape)
print("y shape", y.shape)
print("labels", label_map)


input = keras.layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
# x = ResizeVideo(FINGERS // 2, COORDINATES // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
# x = ResizeVideo(FINGERS // 4, COORDINATES // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
# x = ResizeVideo(FINGERS // 8, COORDINATES // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
# x = ResizeVideo(FINGERS // 16, COORDINATES // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = keras.layers.GlobalAveragePooling3D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(5)(x)

model = Model(input, x)

model.build(data_shape)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# model.evaluate(X_val, y_val)
# tfjs.converters.save_keras_model(model,f"scripts/ts_model/")
model.save('scripts/2Plus1D_Model.keras')