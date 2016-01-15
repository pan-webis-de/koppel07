# Robert Paßmann

NUMBER_ELIMINATE_FEATURES = 3  # number of features eliminated from each extreme
NUMBER_ITERATIONS = 10
CHUNK_LENGTH = 500
INITIAL_FEATURE_SET_LENGTH = 250

# BEST CONFIGURATIONS:
# PAN12I, ELIMINATE = 3, ITERATIONS = 10

import logging
import argparse
import jsonhandler
import sys
from sklearn import svm
import numpy
from collections import Counter
import random


def text_to_list(text):
    # the following takes all alphabetic words normalized to lowercase
    # from the raw data
    return [x for x in
           [''.join(c for c in word if c.isalpha()).lower()
            for word in text.split()]
            if x != '']


def select_chunks(text1, text2):
    """
    Reduce the number of chunks of the text with more chunks such that
    we have the same number of chunks for both texts, i.e. randomly delete
    chunks from the text with more chunks
    """
    random.seed()
    text1.selected_chunks = text1.chunks
    text2.selected_chunks = text2.chunks
    while len(text1.selected_chunks) > len(text2.selected_chunks):
        text1.selected_chunks.remove(random.choice(text1.selected_chunks))
    while len(text2.selected_chunks) > len(text1.selected_chunks):
        text2.selected_chunks.remove(random.choice(text2.selected_chunks))


def curve_score(curve):
    """Calculate a score for a curve by which they are sorted"""
    # this should be optimized
    return sum(curve)


class Database:

    """represents a database with texts of known authors"""

    def __init__(self):
        self.authors = []  # a list of strings with names of authors
        self.texts = {}  # a dictionary (name:Text)
        self.initial_feature_set = []

    def add_author(self, *authors):
        for author in authors:
            self.authors.append(author)
            self.texts[author] = []

    def add_text(self, author, *texts):
        """
        Keyword arguments:
        author -- an author whose texts we want to add
        texts -- a list of texts of this author
        """
        if author not in self.authors:
            raise Exception("Author unknown")

        for text in texts:
            (self.texts[author]).append(text)

    def calc_initial_feature_set(self):
        """
        Calculate the initial feature set consisting of the most frequent
        INITIAL_FEATURE_SET_LENGTH words

        for every text chunks have to be created beforehand
        """
        counter = Counter()
        for author in self.authors:
            for text in self.texts[author]:
                counter += Counter(text.tokens)

        self.initial_feature_set = list(
            dict(counter.most_common(INITIAL_FEATURE_SET_LENGTH)).keys())


class Text:

    """represents a text"""

    def __init__(self, raw, name):
        """
        Keyword arguments:
        raw -- The raw text as a string.
        name -- The name of the text.
        """
        self.raw = raw
        self.name = name

        self.chunks = []  # containing all the chunks of n words
        self.selected_chunks = []  # contains a reduced number of chunks
                                  # for having the same number of chunks
                                  # for two text during calculations
        self.tokens = []

        self.chunk_feature_frequencies = {}

    def create_chunks(self):
        """
        Create chunks of length CHUNK_LENGTH from the raw text. There might be
        intersections between the ultimate and penultimate chunks.
        """
        global CHUNK_LENGTH

        self.tokens = text_to_list(self.raw)
        n = len(self.tokens)

        if n < CHUNK_LENGTH:
            raise Exception("Text is too short")

        chunk_endpoints = list(range(CHUNK_LENGTH, n + 1, CHUNK_LENGTH))
        if n not in chunk_endpoints:
            chunk_endpoints.append(n)

        for endpoint in chunk_endpoints:
            self.chunks.append(self.tokens[endpoint - CHUNK_LENGTH:endpoint])


def tira(corpusdir, outputdir):
    # load training data
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()

    database = Database()

    for candidate in jsonhandler.candidates:
        database.add_author(candidate)
        for training in jsonhandler.trainings[candidate]:
            logging.info(
                "Reading training text '%s' of '%s'", training, candidate)
            text = Text(jsonhandler.getTrainingText(candidate, training),
                        candidate + " " + training)
            try:
                text.create_chunks()
                database.add_text(candidate, text)
            except:
                # logging.info("Text size too small. Skip this text.")
                logging.warning("Text too small. Exit.")
                sys.exit()

    database.calc_initial_feature_set()

    candidates = []  # this list shall contain the most likely candidates

    # We use the unmasking procedure to compare all unknown texts to all
    # enumerated texts of known authorship and then decide which fit best.
    # runtime could surely be optimized
    for unknown in jsonhandler.unknowns:
        try:
            results = {}
                # dictionary containing the maximum difference (first and
                         # last iteration) for every author

            # load the unknown text and create the chunks which are used
            # for the unmasking process
            unknown_text = Text(jsonhandler.getUnknownText(unknown), unknown)
            unknown_text.create_chunks()

            for candidate in jsonhandler.candidates:
                results[candidate] = float("inf")

                for known_text in database.texts[candidate]:
                    # reset the feature list, i.e. create a copy of the initial
                    # list
                    features = list(database.initial_feature_set)

                    # randomly select equally many chunks from each text
                    select_chunks(unknown_text, known_text)

                    # create label vector
                    # (0 -> chunks of unknown texts, 1 -> chunks of known texts)
                    label = [0 for i in range(0, len(unknown_text.selected_chunks))] + \
                            [1 for i in range(
                                0, len(known_text.selected_chunks))]
                    label = numpy.array(label)
                    # the reshape is necessary for the classifier
                    label.reshape(
                        len(unknown_text.selected_chunks) + len(known_text.selected_chunks), 1)

                    # loop
                    global NUMBER_ITERATIONS
                    global NUMBER_ELIMINATE_FEATURES
                    scores = []
                    for i in range(0, NUMBER_ITERATIONS):
                        logging.info("Iteration #%s for texts '%s' and '%s'",
                                     str(i + 1), unknown, known_text.name)
                        # Create the matrix containing the relative word counts
                        # in each chunk (for the selected features)
                        matrix = [[chunk.count(word) / CHUNK_LENGTH
                                   for word in features]
                                  for chunk
                                  in (unknown_text.selected_chunks + known_text.selected_chunks)]
                        matrix = numpy.array(matrix)

                        # Get a LinearSVC classifier and its score (i.e. accuracy
                        # in the training data). Save this score as a point in the
                        # scores curve. (We want to select the curve with the
                        # steepest decrease)
                        classifier = svm.LinearSVC()
                        classifier.fit(matrix, label)
                        scores.append(classifier.score(matrix, label))

                        # a list of all feature weights
                        flist = classifier.coef_[0]

                        # Now, we have to delete the strongest weighted features
                        # (NUMBER_ELIMINATE_FEATURES) from each side.
                        # indices of maximum 3 values and minimum 3 values
                        delete = list(numpy.argsort(flist)[-NUMBER_ELIMINATE_FEATURES:]) \
                            + list(numpy.argsort(flist)[
                                   :NUMBER_ELIMINATE_FEATURES])

                        # We cannot directly use the delete list to eliminate from
                        # the features list since peu-a-peu elimination changes
                        # the indices.
                        delete_features = []
                        for i in delete:
                            delete_features.append(features[i])

                        logging.info("Delete %s", str(delete_features))

                        for feature in delete_features:
                            # a single feature could appear twice in the delete
                            # list
                            if feature in features:
                                features.remove(feature)

                    # The scores list is now the graph we use to get our results
                    # Therefore, compare with previous scores.
                    score = curve_score(scores)
                    logging.info("Calculated a score of %s", str(score))
                    if score < results[candidate]:
                        results[candidate] = score

            # Which author has the biggest score?
            most_likely_author = min(results, key=results.get)
            logging.info("Most likely author is '%s' with a score of %s",
                         most_likely_author, results[most_likely_author])
            candidates.append(most_likely_author)
        except:
            candidates.append("FILE_TO_SMALL")

    # save everything in the specified directory
    jsonhandler.storeJson(jsonhandler.unknowns, candidates, outputdir)


def main():
    parser = argparse.ArgumentParser(description='Tira submission for Delta.')
    parser.add_argument('-i',
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o',
                        action='store',
                        help='Path to output directory')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']

    tira(corpusdir, outputdir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()
