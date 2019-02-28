#!/usr/bin/env python

import os

import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from hdbscan import HDBSCAN
from keras import backend as K
from keras.engine import Layer
from keras.models import Model, load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ElmoEmbeddingLayer(Layer):
    """
    Custom layer that allows us to update weights
    src: https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
    """

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            "https://tfhub.dev/google/elmo/2",
            trainable=self.trainable,
            name="{}_module".format(self.name),
        )

        self.trainable_weights += K.tf.trainable_variables(
            scope="^{}_module/.*".format(self.name)
        )
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1), as_dict=True, signature="default"
        )["default"]
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, "--PAD--")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


class FItSNE_wrapper:
    """
    Wrapper class for fast_tsne.fast_tsne.
    """

    def __init__(self, **kwargs):
        self.__dict__ = {  # default values
            "theta": 0.5,
            "perplexity": 30,
            "map_dims": 2,
            "max_iter": 1000,
            "stop_early_exag_iter": 250,
            "K": -1,
            "sigma": -1,
            "nbody_algo": "FFT",
            "knn_algo": "annoy",
            "mom_switch_iter": 250,
            "momentum": 0.5,
            "final_momentum": 0.8,
            "learning_rate": 200,
            "early_exag_coeff": 12,
            "no_momentum_during_exag": False,
            "n_trees": 50,
            "search_k": None,
            "start_late_exag_iter": -1,
            "late_exag_coeff": -1,
            "nterms": 3,
            "intervals_per_integer": 1,
            "min_num_intervals": 50,
            "seed": -1,
            "initialization": None,
            "load_affinities": None,
            "perplexity_list": None,
            "df": 1,
            "return_loss": False,
            "nthreads": None,
        }
        self.__dict__ = {**self.__dict__, **kwargs}

    def fit_transform(
        self, X, fitsne_path=os.getenv("CONDA_PREFIX") + "/lib/python3.7/FIt-SNE"
    ):
        """
        See: https://github.com/KlugerLab/FIt-SNE

            $ source activate granddebat-env
            $ cd $CONDA_PREXIX/lib/python3.7/
            $ git clone https://github.com/KlugerLab/FIt-SNE.git FIt-SNE
            $ cd FIt-SNE
            $ g++ -std=c++11 -O3 src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp -o bin/fast_tsne -pthread -lfftw3 -lm

        """
        import os, sys

        assert os.path.exists(
            os.getenv("CONDA_PREFIX") + "/lib/python3.7/FIt-SNE/fast_tsne.py"
        ), "You must install FIt-SNE in '$CONDA_PREXIX/lib/python3.7/'"
        if fitsne_path not in sys.path:
            sys.path.append(fitsne_path)

        from fast_tsne import fast_tsne as _fast_tsne

        return _fast_tsne(X, **self.__dict__)


#######################################################################################


# Initialize session
sess = tf.Session()
K.set_session(sess)


# Données du 17 février 2019 (src: https://granddebat.fr/pages/donnees-ouvertes)
data_url = {
    "transition_ecologique": "http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-17/LA_TRANSITION_ECOLOGIQUE.csv",
    "fiscalite_deppublique": "http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-17/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.csv",
    "democratie_cotyonnete": "http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-17/DEMOCRATIE_ET_CITOYENNETE.csv",
    "services_publics": "http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-17/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv",
}

data_label = "transition_ecologique"

# Chargement des données (si pas déjà fait)
try:
    D.shape
except:
    print("Loading data from '%s'" % data_url[data_label].split("/")[-1])
    D = pd.read_csv(data_url[data_label])
    D.shape


# build model
input_text = layers.Input(shape=(1,), dtype="string")
embedding = ElmoEmbeddingLayer()(input_text)
model = Model(inputs=[input_text], outputs=embedding)

# select data
colnum = 2
test_data = D.iloc[:, colnum].sample(2500).values
test_data = D.iloc[:, colnum].values

# apply model
Y = model.predict(test_data, batch_size=666, verbose=1)
Y.shape

# save embeddings
np.save("%s_%i_%s" % (data_label, len(test_data), D.columns[colnum]), Y)

### plot embeddings

# FIt-SNE
fw = FItSNE_wrapper(max_iter=1000, df=0.9)
Z = fw.fit_transform(Y)


plt.scatter(Z[:, 0], Z[:, 1], s=3)
plt.title("%s - %s" % (data_label, D.columns[colnum]))
plt.savefig("%s_%i_%s.png" % (data_label, len(test_data), D.columns[colnum]))
plt.show()


### BHt-SNE
# from sklearn.manifold import TSNE
# tm = TSNE(verbose=3)
# Z = tm.fit_transform(Y)
# Z.shape
# plt.scatter(Z[:, 0], Z[:, 1], s=5)
# plt.show()

#######################################################################################

### Interactive plot
fig, ax = plt.subplots()
sc = plt.scatter(Z[:, 0], Z[:, 1], s=5)

annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot.set_visible(False)


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(
        " ".join(list(map(str, ind["ind"]))),
        " ".join([test_data[n] for n in ind["ind"]]),
    )
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    # annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
