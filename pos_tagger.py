from collections import defaultdict
from os.path import join
from nltk.data import load, find
from nltk import jsontags
import pickle
import random
import logging

from nltk.tag.perceptron import _pc

PICKLE = "averaged_perceptron_tagger.pickle"
_UNIVERSAL_DATA = "taggers/universal_tagset"
_UNIVERSAL_TAGS = (
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
)

_MAPPINGS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "UNK")))


def pos_tag(tokens, tagset=None, lang="eng"):
    tagger = PerceptronTagger()
    return _pos_tag(tokens, tagset, tagger, lang)

def _pos_tag(tokens, tagset=None, tagger=None, lang=None):

    tagged_tokens = tagger.tag(tokens)
    if tagset:  # Maps to the specified tagset.
        tagged_tokens = [
            (token, map_tag("en-ptb", tagset, tag))
            for (token, tag) in tagged_tokens
        ]
    return tagged_tokens

def map_tag(source, target, source_tag):
    if target == "universal":
        if source == "wsj":
            source = "en-ptb"
        if source == "brown":
            source = "en-brown"

    return tagset_mapping(source, target)[source_tag]

def tagset_mapping(source, target):

    if source not in _MAPPINGS or target not in _MAPPINGS[source]:
        if target == "universal":
            _load_universal_map(source)
            _MAPPINGS["ru-rnc-new"]["universal"] = {
                "A": "ADJ",
                "A-PRO": "PRON",
                "ADV": "ADV",
                "ADV-PRO": "PRON",
                "ANUM": "ADJ",
                "CONJ": "CONJ",
                "INTJ": "X",
                "NONLEX": ".",
                "NUM": "NUM",
                "PARENTH": "PRT",
                "PART": "PRT",
                "PR": "ADP",
                "PRAEDIC": "PRT",
                "PRAEDIC-PRO": "PRON",
                "S": "NOUN",
                "S-PRO": "PRON",
                "V": "VERB",
            }

    return _MAPPINGS[source][target]

def _load_universal_map(fileid):
    contents = load(join(_UNIVERSAL_DATA, fileid + ".map"), format="text")
    _MAPPINGS[fileid]["universal"].default_factory = lambda: "X"

    for line in contents.splitlines():
        line = line.strip()
        if line == "":
            continue
        fine, coarse = line.split("\t")

        assert coarse in _UNIVERSAL_TAGS, "Unexpected coarse tag: {}".format(coarse)
        assert (
            fine not in _MAPPINGS[fileid]["universal"]
        ), "Multiple entries for original tag: {}".format(fine)

        _MAPPINGS[fileid]["universal"][fine] = coarse

@jsontags.register_tag
class PerceptronTagger():

    json_tag = "nltk.tag.sequential.PerceptronTagger"

    START = ["-START-", "-START2-"]
    END = ["-END-", "-END2-"]

    def __init__(self, load=True):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            AP_MODEL_LOC = "file:" + str(
                find("taggers/averaged_perceptron_tagger/" + PICKLE)
            )
            self.load(AP_MODEL_LOC)

    def load(self, loc):
        self.model.weights, self.tagdict, self.classes = load(loc)
        self.model.classes = self.classes

    def tag(self, tokens, return_conf=False, use_tagdict=True):
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag, conf = (
                (self.tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
            )
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, conf = self.model.predict(features, return_conf)
            output.append((word, tag, conf) if return_conf == True else (word, tag))

            prev2 = prev
            prev = tag

        return output

    def normalize(self, word):
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        elif word.isdigit() and len(word) == 4:
            return "!YEAR"
        elif word[0].isdigit():
            return "!DIGITS"
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        """

        def add(name, *args):
            features[" ".join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add("bias")
        add("i suffix", word[-3:])
        add("i pref1", word[0])
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1])
        add("i-1 suffix", context[i - 1][-3:])
        add("i-2 word", context[i - 2])
        add("i+1 word", context[i + 1])
        add("i+1 suffix", context[i + 1][-3:])
        add("i+2 word", context[i + 2])
        return features

    def train(self, sentences, save_loc=None, nr_iter=5):

        self._sentences = list()  # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for sentence in self._sentences:
                words, tags = zip(*sentence)

                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess, _ = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(self._sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))

        self._sentences = None

        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            with open(save_loc, "wb") as fout:
                # changed protocol from -1 to 2 to make pickling Python 2 compatible
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)


@jsontags.register_tag
class AveragedPerceptron:

    json_tag = "nltk.tag.perceptron.AveragedPerceptron"

    def __init__(self, weights=None):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = weights if weights else {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        # compute the confidence
        conf = max(self._softmax(scores)) if return_conf == True else None

        return best_label, conf

    def update(self, truth, guess, features):

        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)