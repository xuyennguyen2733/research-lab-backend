import numpy as np
import einops
import keras
import cv2
import os
from datetime import datetime

FINGERS = 6
COORDINATES = 2
FRAME_COUNT = 30
input_shape = (None, FRAME_COUNT, FINGERS, COORDINATES, 1)
data_shape = (50, FRAME_COUNT, FINGERS, COORDINATES, 1)

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
  def __init__(self, height, width, **kwargs):
    super().__init__(**kwargs)
    self.height = height
    self.width = width
    self.resizing_layer = keras.layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height,
    # w stands for width, and c stands for the number of channels.
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
          'height': self.height,
          'width': self.width
      })
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)

   
def load_video(path):
    frames = np.load(path)
    frame_delay = int(1000/2) # 2 frames per second

    while True:
        for i in range(frames.shape[0]):
            # Display the current frame
            cv2.imshow('Video', frames[i])
            
            # Wait for a key event or move to the next frame
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break
        
        # Check if 'q' was pressed to break the outer loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
def sample_frames(frames, sample_size=15):
    frame_count = frames.shape[0]
    junk_size = int(frame_count/sample_size)
    lower = 0
    upper = junk_size
    sample = []
    while len(sample) < sample_size:
        sample_idx = np.random.randint(lower,upper)
        sample.append(frames[sample_idx])
        lower = upper
        upper += junk_size
        
    return sample

def load_model():
   model = keras.models.load_model('scripts/2Plus1D_Model.keras', custom_objects={"Conv2Plus1D": Conv2Plus1D, 
  #  "ResizeVideo": ResizeVideo, 
   "ResidualMain": ResidualMain, "Project": Project, "add_residual_block": add_residual_block})
   return model

def predict_index(frames, model):
    # Make predictions
    predictions = model.predict(frames)

    # Process the predictions
    predicted_class = np.argmax(predictions, axis=1)[0]
    print("prediction array", predictions)
    print("predicted class", predicted_class)
    return predicted_class

def get_labels(folder_path):
    labels = []
    if os.path.isdir(folder_path):
        for entry in os.listdir(folder_path):
            labels.append(entry)
        
    return labels

def load_data(data_path):
    labels = get_labels(data_path)
    label_map = {label:num for num, label in enumerate(labels)}

    X = np.empty((0, FRAME_COUNT, FINGERS, COORDINATES))
    y = []

    for label in labels:
        print("Processing label", label)
        
        label_path = os.path.join(data_path, label)
        
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            example = np.load(file_path)
            
            # example = np.expand_dims(example, axis=-1)
            # X.append(example)
            # map labels to numerical values because string values aren't allowed in training
            X = np.concatenate((X, np.expand_dims(example, axis=0)))
            y.append(label_map[label])
            
    # X = np.stack(X, axis=0)
    # X = np.expand_dims(X, axis=-1)
    y = np.array(y)
    
    return (X,y)

def save_numpy_frames(frames, file_name="numpy_video.npy", parent_folder_name="data", folder_name="default", folder_path=os.path.abspath(os.getcwd())):
    output_dir = f"{folder_path}/{parent_folder_name}/{folder_name}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(f"{output_dir}/{file_name}", frames)

labels = ["a", "j", "z"]

model = load_model()

data_path = os.path.join(os.getcwd(), "scripts/data/hand")
(X_val, y_val) = load_data(data_path)
y_str = np.array(labels)[np.array(y_val)]

# while (True):
#     command = input('Enter q to quit, anything other letter to predict:')
#     if (command == 'q'):
#        break
    
#     rand_idx = np.random.randint(0,len(X_val))
#     predicted_idx = predict_index(np.array([X_val[rand_idx]]), model)
#     predicted_label = labels[predicted_idx]
#     print('predict:', predicted_label)
#     print('true label:', y_str[rand_idx])
example_count = 0
for i in range(0, len(X_val)):
    predicted_idx = predict_index(np.array([X_val[i]]), model)
    predicted_label = labels[predicted_idx]
    print('predict:', predicted_label)
    print('true label:', y_str[i])
    
    if (predicted_label != y_str[i]):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_numpy_frames(X_val[i],file_name=f"{example_count}_{current_datetime}.npy", parent_folder_name="retrain_data", folder_name=y_str[i])
        example_count += 1

model.evaluate(X_val, y_val)