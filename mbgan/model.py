# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

import keras.optimizers
from keras.layers import Input, Dense, Dropout, Lambda, Layer
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.models import Sequential, Model

import tensorflow as tf

import os
import datetime
import numpy as np

from functools import partial


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    # compute ∇_x D(interpolated)
    gradients = tf.gradients(y_pred, averaged_samples)[0]
    sq = tf.square(gradients)
    sq_sum = tf.reduce_sum(sq, axis=list(range(1, len(gradients.shape))))
    l2 = tf.sqrt(sq_sum)
    penalty = tf.square(1.0 - l2)
    return tf.reduce_mean(penalty)


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch."""
    return tf.reduce_mean(y_true * y_pred)


def get_optimizer(optimizer, lr, decay=0.0, clipnorm=0.0, clipvalue=0.0, **kwargs):
    """Get optimizer from keras.optimizers."""
    support_optimizers = {"SGD", "RMSprop", "Adagrad", "Adadelta", "Adam"}
    assert optimizer in support_optimizers
    fn = getattr(keras.optimizers, optimizer)
    # build arg dict but only include non-zero clipping fields
    opt_kwargs = dict(**kwargs)
    if decay and decay > 0:
        opt_kwargs["decay"] = decay
    if clipnorm and clipnorm > 0:
        opt_kwargs["clipnorm"] = clipnorm
    if clipvalue and clipvalue > 0:
        opt_kwargs["clipvalue"] = clipvalue
    return fn(learning_rate=lr, **opt_kwargs)


class RandomWeightedAverage(Layer):
    """Calculate a random weighted average between two tensors."""

    def call(self, inputs, **kwargs):
        real, fake = inputs
        # use tf.shape instead of K.shape
        batch = tf.shape(real)[0]
        alpha = tf.random.uniform((batch, 1))
        # broadcast alpha to (batch, ntaxa)
        alpha = tf.repeat(alpha, tf.shape(real)[1], axis=1)
        return alpha * real + (1.0 - alpha) * fake

    def compute_output_shape(self, input_shape):
        # output shape is same as each of the two inputs
        return input_shape[0]


class PhyloTransform(Layer):
    def __init__(self, tf_matrix=None, **kwargs):
        if tf_matrix is None:
            self.kernel = None
        else:
            self.output_dim = tf_matrix.shape[1:]
            self.kernel = K.constant(tf_matrix, dtype="float32")
        super(PhyloTransform, self).__init__(**kwargs)

    def call(self, x):
        if self.kernel is None:
            return x
        else:
            return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        if self.kernel is None:
            return input_shape
        else:
            return (input_shape[0],) + self.output_dim


def build_generator(input_shape, output_units, n_channels=512):
    """build the generator model."""
    model = Sequential()

    model.add(Dense(n_channels, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(n_channels))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(n_channels))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dense(output_units))
    model.add(Activation("softmax"))

    noise = Input(shape=input_shape)
    output = model(noise)

    return Model(noise, output)


def build_critic(
    input_shape, n_channels=256, dropout_rate=0.25, tf_matrix=None, t_pow=1000.0
):
    """build the critic model."""
    model = Sequential()

    model.add(PhyloTransform(tf_matrix, input_shape=input_shape))
    model.add(
        Lambda(
            lambda x: tf.math.log(1.0 + x * t_pow) / tf.math.log(1.0 + t_pow),
            output_shape=lambda s: s,
        )
    )
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_channels))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    inputs = Input(shape=input_shape)
    validity = model(inputs)

    return Model(inputs, validity)


class MBGAN(object):
    def __init__(self, name, model_config, train_config):
        """MBGAN model class.
        name: provide a name for the given model/experiments.
        model_config: provide the configuration to build MBGAN:
            ntaxa: how many taxa are included in the real data
            latent_dim: the size of random vectors for the generator.
            generator: extra parameters parsed to build_generator.
            critic: extra parameters parsed to build_critic.
        train_config: provide the train configuration to build
            computational graph. Includes: loss_weights, optimizer,
            learning rate.
        """
        self.model_name = name
        self.model_config = model_config
        self.train_config = train_config

        self.ntaxa = self.model_config["ntaxa"]
        self.latent_dim = self.model_config["latent_dim"]

        # Build the generator and critic and construct the computational graph
        self.critic = build_critic((self.ntaxa,), **self.model_config["critic"])
        self.generator = build_generator(
            (self.latent_dim,), self.ntaxa, **self.model_config["generator"]
        )
        self.construct_critic_graph()
        self.construct_generator_graph()

    def construct_critic_graph(self):
        """Construct computational graph for critic"""
        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Determines fake sample from given noise
        z = Input(shape=(self.latent_dim,))
        fake_sample = self.generator(z)
        fake = self.critic(fake_sample)

        # Determines real sample
        real_sample = Input(shape=(self.ntaxa,))
        valid = self.critic(real_sample)

        # Determines weighted average between real and fake sample
        interpolated_sample = RandomWeightedAverage()([real_sample, fake_sample])
        validity_interpolated = self.critic(interpolated_sample)

        # Get gradient penalty loss
        partial_gp_loss = partial(
            gradient_penalty_loss, averaged_samples=interpolated_sample
        )
        partial_gp_loss.__name__ = "gradient_penalty"

        # Construct critic computational graph
        self.critic_graph = Model(
            inputs=[real_sample, z], outputs=[valid, fake, validity_interpolated]
        )

        optimizer = get_optimizer(
            self.train_config["critic"]["optimizer"][0],
            lr=self.train_config["critic"]["lr"],
            **self.train_config["critic"]["optimizer"][1],
        )
        loss_weights = self.train_config["critic"]["loss_weights"]

        self.critic_graph.compile(
            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
            optimizer=optimizer,
            loss_weights=loss_weights,
        )

    def construct_generator_graph(self):
        """Construct computational graph for generator."""
        # Freeze the critic's layers while training the generator
        self.critic.trainable = False
        self.generator.trainable = True

        # Generate sample and update generator
        z = Input(shape=(self.latent_dim,))
        fake_sample = self.generator(z)
        valid = self.critic(fake_sample)

        # Construct generator computational graph
        self.generator_graph = Model(z, valid)
        optimizer = get_optimizer(
            self.train_config["generator"]["optimizer"][0],
            lr=self.train_config["generator"]["lr"],
            **self.train_config["generator"]["optimizer"][1],
        )
        self.generator_graph.compile(loss=wasserstein_loss, optimizer=optimizer)

    def train(
        self,
        dataset,
        iterations,
        batch_size=32,
        n_critic=5,
        n_generator=1,
        save_interval=1000,
        save_fn=None,
        experiment_dir="mbgan_train",
    ):

        # optimizers
        gen_opt = get_optimizer("RMSprop", lr=self.train_config["generator"]["lr"])
        critic_opt = get_optimizer("RMSprop", lr=self.train_config["critic"]["lr"])
        gp_weight = self.train_config["critic"]["loss_weights"][2]

        for epoch in range(1, iterations + 1):
            # ——— critic updates ———
            # make sure critic’s weights are marked trainable
            self.critic.trainable = True
            self.generator.trainable = False
            for _ in range(n_critic):
                real = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
                z = tf.random.normal([batch_size, self.latent_dim])

                with tf.GradientTape() as tape:
                    fake_scores = self.critic(
                        self.generator(z, training=True), training=True
                    )
                    real_scores = self.critic(real, training=True)

                    # gradient penalty
                    alpha = tf.random.uniform([batch_size, 1])
                    interp = alpha * real + (1 - alpha) * self.generator(
                        z, training=True
                    )
                    with tf.GradientTape() as gp_tape:
                        gp_tape.watch(interp)
                        interp_score = self.critic(interp, training=True)
                    grads = gp_tape.gradient(interp_score, interp)
                    gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)

                    # WGAN-GP critic loss
                    c_loss = (
                        tf.reduce_mean(fake_scores)
                        - tf.reduce_mean(real_scores)
                        + gp_weight * gp
                    )

                critic_vars = self.critic.trainable_variables
                grads = tape.gradient(c_loss, critic_vars)
                critic_opt.apply_gradients(zip(grads, critic_vars))

            # ——— generator updates ———
            for _ in range(n_generator):
                z = tf.random.normal([batch_size, self.latent_dim])
                with tf.GradientTape() as tape:
                    # generator wants to make critic give *high* scores
                    g_loss = -tf.reduce_mean(
                        self.critic(self.generator(z, training=True), training=False)
                    )
                grads = tape.gradient(g_loss, self.generator.trainable_variables)
                gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

            # logging & optional saving
            if epoch % save_interval == 0:
                print(f"iter={epoch}  D_loss={c_loss:.4f}  G_loss={g_loss:.4f}")
                # save weights + sample CSV
                self.critic.save_weights(
                    os.path.join(experiment_dir, f"critic_{epoch}.h5")
                )
                self.generator.save_weights(
                    os.path.join(experiment_dir, f"gen_{epoch}.h5")
                )
                if save_fn:
                    save_fn(self, epoch)

    def predict(self, n_samples=100, transform=None, seed=None):
        np.random.seed(seed)
        z = np.random.normal(0, 1, (n_samples, self.latent_dim))
        res = self.generator.predict(z)
        if transform is not None:
            res = transform(res)

        return res
