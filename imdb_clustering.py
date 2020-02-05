import csv
import numpy as np
import re
import nltk
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt, cm
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabaz_score


# constants
TRAIN_ARCHITECTURES = ["skip_gram", "cont_bag_of_words"]
SKIP_GRAM = 0
BAG_OF_WORDS = 1
TRAIN_FUNCTIONS = ["softmax", "negative_sampling"]
SOFTMAX = 0
NEGATIVE_SAMPLING = 1


def get_sentence_words_from_text(text):

    """Split a text into sentences and then split each sentence's words and add them to a list, filtering non-alphanumeric tokens (hyphen is accepted though)"""

    # clean the text by removing HTML tags
    text = BeautifulSoup(text, "lxml").get_text()

    # stop-words to remove
    stop_words = set(stopwords.words('english'))

    sentence_words = list()

    for sentence in nltk.sent_tokenize(text):

        words = list()

        for word in nltk.word_tokenize(sentence):
            if re.match(r'^[A-Za-z0-9-]+$', word):
                word = word.lower()
                if word not in stop_words:
                    words.append(word)

        if words:
            sentence_words.append(words)

    return sentence_words


def get_sentences():

    """Get a list with all the sentences for later embeddings (as lists of words)"""

    # load a binary file with the tokenized sentences if it exists, to speed up the workflow
    sentence_binary_file_name = "sentences.binary"
    sentence_file = Path(sentence_binary_file_name)
    if sentence_file.is_file():
        with open(sentence_binary_file_name, "rb") as file:
            sentences = pickle.load(file)

    # otherwise extract the sentences from the text files
    else:

        # locate all the text files
        file_paths = [str(path) for path in list(Path("IMDB").rglob("*.txt"))]

        sentences = list()

        for file_path in file_paths:
            with open(file_path, "r") as file:
                sentences.extend(get_sentence_words_from_text(file.read()))

    with open(sentence_binary_file_name, "wb") as file:
        pickle.dump(sentences, file)

    return sentences


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, alpha, filename=None):

    """Use t-SNE to plot similar words based on their embeddings. t-SNE and plot code comes from: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d"""

    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=alpha, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.7, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=3)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=450, bbox_inches='tight')
    plt.show()


def generate_embeddings(sentences, feature_dim, window_size, min_word_count, worker_num, train_architecture, train_function, learning_rate, high_freq_downsampling):

    """Generate word embedding vectors with the given parameters"""

    model = word2vec.Word2Vec(sentences, size=feature_dim, window=window_size, min_count=min_word_count, workers=worker_num, sg=train_architecture, hs=train_function, alpha=learning_rate, sample=high_freq_downsampling)
    return model.wv


def cluster_embeddings(word_vectors, cluster_num, batch_size, init, max_iter, max_no_improvement, verbose):

    """Cluster word embedding vectors, and evaluate the variance"""

    kmeans_handler = MiniBatchKMeans(n_clusters=cluster_num, batch_size=batch_size, init=init, max_iter=max_iter, max_no_improvement=max_no_improvement, verbose=verbose)

    # get the cluster indices of each embedding
    indices = kmeans_handler.fit_predict(word_vectors.vectors)

    # create a dictionary that maps each word with its cluster number
    word_cluster_map = dict(zip(word_vectors.index2word, indices))

    # calculate the ratio between intra- and inter-cluster variance (the lower, the better)
    variance_ratio = calinski_harabaz_score(word_vectors.vectors, indices)

    # get the lists of words forming each cluster
    word_clusters = [list() for _ in range(cluster_num)]
    for word in word_cluster_map.keys():
        word_clusters[word_cluster_map[word]].append(word)

    return word_clusters, variance_ratio


def get_best_embeddings_for_clustering(sentences, results_file_name=None):

    """Evaluate many configurations of word embeddings for clustering, based on intra-/inter-cluster variance"""

    worker_num = 8

    embedding_models = list()

    max_iter = np.inf
    iter_count = 0

    # evaluate all parameter combinations
    for feature_dim in [200, 300]:
        for window_size in [5, 10]:
            for min_word_count in [1, 20]:
                for train_architecture in [SKIP_GRAM, BAG_OF_WORDS]:
                    for train_function in [SOFTMAX, NEGATIVE_SAMPLING]:
                        for learning_rate in [0.025, 0.1]:
                            for high_freq_downsampling in [0.001]:

                                    if iter_count < max_iter:

                                        print("Building model for embedding configuration " + str(iter_count) + "...")

                                        # string describing the configuration
                                        config_string = "feature_dim=" + str(feature_dim) + ", window_size=" + str(window_size) + ", min_word_count=" + str(min_word_count) + ", train_architecture=" + TRAIN_ARCHITECTURES[train_architecture] + ", train_function=" + TRAIN_FUNCTIONS[train_function] + ", learning rate=" + str(learning_rate) + ", downsampling=" + str(high_freq_downsampling)

                                        # generate word embeddings with the given parameters
                                        word_vectors = generate_embeddings(sentences, feature_dim, window_size, min_word_count, worker_num, train_architecture, train_function, learning_rate, high_freq_downsampling)

                                        # cluster the embeddings using K-Means, evaluating the variance ratio
                                        cluster_num = 10
                                        word_clusters, variance_ratio = cluster_embeddings(word_vectors, cluster_num=cluster_num, batch_size=len(word_vectors.vectors) // 2, init="k-means++", max_iter=50, max_no_improvement=5, verbose=True)

                                        embedding_models.append({"config_string": config_string, "word_vectors": word_vectors, "word_clusters": word_clusters, "variance_ratio": variance_ratio})

                                        iter_count += 1

    # sort the embeddings by ascending variance ratio
    embedding_models.sort(key=lambda model: -model["variance_ratio"], reverse=True)

    # write the results to a file
    if results_file_name:
        with open(results_file_name, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["config_string", "variance_ratio"])
            for model in embedding_models:
                writer.writerow((model["config_string"], str(model["variance_ratio"])))

    # return the best model, the one with the lowest variance ratio
    best_model = embedding_models[0]
    return best_model["word_vectors"], best_model["word_clusters"]


def main():

    """Main function"""

    # load all the sentences, as lists of words
    sentences = get_sentences()

    # get the best word embedding clusters among different configurations, minimizing a variance ratio
    word_vectors, word_clusters = get_best_embeddings_for_clustering(sentences, "embedding_models_variance_ratio.csv")

    # save the best model to a file
    word_vectors.save_word2vec_format("best_model.embed", binary=True)

    # show the words in each cluster
    for i, cluster in enumerate(word_clusters):
        print("Cluster " + str(i) + ":", cluster[:1000])

    key_words = ['actor', 'actress', 'director', 'movie', 'film', 'good', 'bad', 'effects', 'music', 'costner', 'zeta-jones', 'willis', 'roberts', 'niro', 'pacino', 'scarface', 'godfather', 'coppola', 'kubrick', '2', '007', '10', '1984', '2001']

    # preparing code for visualization, based on: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
    similar_word_clusters = list()
    embedding_clusters = list()
    for word in key_words:
        words = list()
        embeddings = list()
        for similar_word, _ in word_vectors.most_similar(word, topn=10):
            words.append(similar_word)
            embeddings.append(word_vectors[similar_word])
        similar_word_clusters.append(words)
        embedding_clusters.append(embeddings)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    # visualize the similar words
    tsne_plot_similar_words('Word similarity in IMDB movie reviews', key_words, embeddings_en_2d, similar_word_clusters, 0.8, 'similar_words.png')


if __name__ == "__main__":

    main()

