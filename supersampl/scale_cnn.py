import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import cv2
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# config = tf.ConfigProto()
#
# init_op = tf.global_variables_initializer()
# sess = tf.Session(config=config)
# sess.run(init_op)

from keras import backend as K
from keras.models import Model
from keras.layers import Add, Input
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPool1D, BatchNormalization, UpSampling2D
from keras.layers import Activation, Reshape, dot, add
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.callbacks import Callback, TensorBoard

from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

from cvision import settings

graph = tf.get_default_graph()

_cv2_available = True
_image_scale_multiplier = 1


def combine_patches(in_patches, out_shape):
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon


def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = cv2.resize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False)(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False)(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False)(ip)
    return x


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def depth_to_scale_tf(input, scale, channels):

    def _phase_shift(I, r):
        ''' Function copied as is from https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py'''

        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    if channels > 1:
        Xc = tf.split(3, 3, input)
        X = tf.concat(3, [_phase_shift(x, scale) for x in Xc])
    else:
        X = _phase_shift(input, scale)
    return X


def non_local_block(ip, computation_compression=2, mode='embedded'):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    dim1, dim2, dim3 = None, None, None

    if len(ip_shape) == 3:  # time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # Video / Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        f = dot([theta, phi], axes=2)

        # scale the values to make it size invariant
        if batchsize is not None:
            f = Lambda(lambda z: 1./ batchsize * z)(f)
        else:
            f = Lambda(lambda z: 1. / 128 * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplemented('Concatenation mode has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        if computation_compression > 1:
            # shielded computation
            phi = MaxPool1D(computation_compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, channels // 2)
    g = Reshape((-1, channels // 2))(g)

    if computation_compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(computation_compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, channels // 2))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2, dim3))(y)

    y = _convND(y, rank, channels)

    residual = add([ip, y])

    return residual


def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


class HistoryCheckpoint(Callback):
    '''Callback that records events
        into a `History` object.

        It then saves the history after each epoch into a file.
        To read the file into a python dict:
            history = {}
            with open(filename, "r") as f:
                history = eval(f.read())

        This may be unsafe since eval() will evaluate any string
        A safer alternative:

        import ast

        history = {}
        with open(filename, "r") as f:
            history = ast.literal_eval(f.read())

    '''

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))


class TensorBoardBatch(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardBatch, self).__init__(log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.global_step = 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)
        self.global_step += 1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)

        self.global_step += 1
        self.writer.flush()


class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = depth_to_scale_tf(x, self.r, self.channels)
        return y

    def get_output_shape_for(self, input_shape):
        b, r, c, k = input_shape
        return b, r * self.r, c * self.r, self.channels


class BaseSuperResolutionModel(object):

    def __init__(self):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None
        self.weight_path = None
        self.scale_factor = 2

        self.type_scale_type = "norm"  # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
        Subclass dependent implementation.
        """
        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height * _image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * _image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if width is not None and height is not None:
            shape = (width * _image_scale_multiplier, height * _image_scale_multiplier, channels)
        else:
            shape = (None, None, channels)

        init = Input(shape=shape)

        return init

    def upscale(self, img_array=None, origin_path=None, intermediate=None, end_path=None, return_image=False,
                patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.
        :param img_array: numpy.ndarray image object
        :param end_path: string to save image
        :param origin_path:  path to the image
        :param intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """

        if img_array is None and origin_path is None:
            raise ValueError('img_array or path should be specified')

        scale_factor = int(self.scale_factor)
        if img_array is not None:
            true_img = img_array
        else:
            true_img = cv2.imread(origin_path)
            true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)

        init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

        img_dim_1, img_dim_2 = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            mode = 'fast'
            if verbose:
                print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    if verbose:
                        print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_dim_1, img_dim_2 = images.shape[1], images.shape[2]
            if verbose:
                print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_dim_2, img_dim_1))
        else:
            img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
                                                                 scale_factor)

            images = cv2.resize(true_img, (img_dim_2, img_dim_1))
            images = np.expand_dims(images, axis=0)
            if verbose:
                print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        if intermediate:
            if verbose: print("Saving intermediate image.")
            intermediate_img = cv2.resize(true_img, (init_dim_2 * scale_factor, init_dim_1 * scale_factor))
            intermediate_img = cv2.cvtColor(intermediate_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(intermediate, intermediate_img)

        # Transpose and Process images
        img_conv = images.astype(np.float32) / 255.

        if self.model is None:
            with tf.device('/cpu:0'):
                self.model = self.create_model(img_dim_2, img_dim_1, load_weights=True)

        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = self.model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

        result = result.astype(np.float32) * 255.

        if mode == 'patch':
            out_shape = (init_dim_1 * scale_factor, init_dim_2 * scale_factor, 3)
            result = combine_patches(result, out_shape)
        else:
            result = result[0, :, :, :]  # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if verbose: print("Saving image.")

        if end_path:
            cv2.imwrite(end_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        if return_image:
            return result

    def __match_autoencoder_size(self, img_dim_1, img_dim_2, init_dim_1, init_dim_2, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_dim_2 * scale_factor) % 4 != 0) or ((init_dim_1 * scale_factor) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    # print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2 * scale_factor) // 4) * 4
                    img_dim_1 = ((init_dim_1 * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor
            else:
                # AE model and true upsampling
                if ((init_dim_2) % 4 != 0) or ((init_dim_1) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    # print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2) // 4) * 4
                    img_dim_1 = ((init_dim_1) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2, init_dim_1
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_dim_2, img_dim_1 = init_dim_2, init_dim_1
            else:
                # Not AE and not true upsampling
                img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor

        return img_dim_1, img_dim_2,


class DistilledResNetSR(BaseSuperResolutionModel):

    def __init__(self):
        super(DistilledResNetSR, self).__init__()

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False
        self.scale_factor = 2

        self.n = 32
        self.mode = 2

        self.weight_path = settings.RES_NET
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(DistilledResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(self.n, (3, 3), activation='relu', padding='same', name='student_sr_res_conv1')(init)

        x = self._residual_block(x0, 1)

        x = Add(name='student_residual')([x, x0])
        x = self._upscale_block(x, 1)

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='student_sr_res_conv_final')(x)

        model = Model(init, x)
        # dont compile yet
        if load_weights: model.load_weights(self.weight_path, by_name=True)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="student_sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="student_sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = init._keras_shape[channel_dim]

        x = UpSampling2D(name='student_upsampling_%d' % id)(init)
        x = Convolution2D(self.n * 2, (3, 3), activation="relu", padding='same', name='student_sr_res_filter1_%d' % id)(x)

        return x

    def upscale(self, **kwargs):
        with graph.as_default():
            with tf.device('/cpu:0'):
                super(DistilledResNetSR, self).upscale(**kwargs)





