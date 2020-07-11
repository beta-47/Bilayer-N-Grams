import nltk
import string
import pickle
from nltk.corpus import brown
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt')
from collections import defaultdict, Counter
from random import random
from random import shuffle

def generate_clean_words(text):
    text = text.replace('--', ' ')
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalpha()]
    words = [word.lower() for word in words]
    return words

def extract_POS(text):
    word_POS = nltk.pos_tag(text, tagset='universal')
    POS = [p for w, p in word_POS]
    return POS

class bilayer_ngram:
    
    def __init__(self, n_layer1 = 4, n_layer2 = 4):
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.ngrams_layer1 = defaultdict(Counter)
        self.ngrams_layer2 = defaultdict(Counter)
        self.word_to_tags = defaultdict(set)
        self.tag_to_words = defaultdict(Counter)

    def normalize(self, counter):
        s = float(sum(counter.values()))
        return [(c, cnt / s) for c, cnt in counter.items()]
    
    def generate_tag_to_words(self, tokens):
        word_tag = nltk.pos_tag(tokens, tagset='universal')
        for word, tag in word_tag:
            self.tag_to_words[tag][word] += 1
            self.word_to_tags[word].add(tag)

    def train(self, ngrams, n, tokens, store_tags=False):
        if store_tags:
            self.generate_tag_to_words(tokens)
        padding = ['~'] * (n - 1)
        tokens = padding + tokens
        for i in range(len(tokens) - n + 1):
            history = ''.join(tokens[i : i + n - 1])
            token = tokens[i + n - 1]
            ngrams[history][token] += 1
        return ngrams

    def train_layer1(self, tokens):
        self.ngrams_layer1 = self.train(self.ngrams_layer1, self.n_layer1, tokens, True)
        return self

    def train_layer2(self, tokens):
        self.ngrams_layer2 = self.train(self.ngrams_layer2, self.n_layer2, tokens)
        return self

    def combine_probabilities(self, prob1, prob2):
        return (prob1 * prob2) / ((prob1 * prob2) + (1 - prob1) * (1 - prob2))

    def generate_possible_tokens(self, ngram, history):
        history = ''.join(history)            
        possibilities = [(token, count) for token, count in ngram[history].items()]
        # Get The 10 Best Possible Tokens By Their Probabilities
        sorted(possibilities, key = lambda x : x[1], reverse = True)
        best = possibilities[:10]
        s = float(sum(n for _, n in best))
        best = [(token, count / s) for token, count in best]
        return best

    def get_words_with_probabilities_for_tag(self, tag):
        tag_to_words = {tag : self.normalize(words) for tag, words in self.tag_to_words.items()}
        return tag_to_words[tag]

    def get_tags(self, word):
        return self.word_to_tags[word]

    def next_token(self, history_layer1, history_layer2):
        history_layer1 = ''.join(history_layer1)
        history_layer2 = ''.join(history_layer2)

        possible_tags = dict(self.generate_possible_tokens(self.ngrams_layer2, history_layer2))
        
        if history_layer1 not in self.ngrams_layer1:
            possible_words = self.get_words_with_probabilities_for_tag(max(possible_tags, key = possible_tags.get))
        else:
            possible_words = self.generate_possible_tokens(self.ngrams_layer1, history_layer1)

        word_tag = defaultdict(Counter)

        for word, probability in possible_words:
            tags = list(set(self.get_tags(word)) & set(possible_tags.keys()))
            if len(tags) == 0:
                tags = list(possible_tags.keys())
            for tag in tags:
                tag_prob = possible_tags[tag]
                word_tag[word][tag] = self.combine_probabilities(tag_prob, probability)

        word_tag_prob = []
        for word, counter in word_tag.items():
            word_tag_prob += [(word, tag, prob) for tag, prob in dict(counter).items()]
   
        # Normalizing to avoid extremely small probabilities
        s = sum(n for _, _, n in word_tag_prob)
        word_tag_prob = [(word, tag, prob / s) for word, tag, prob in word_tag_prob]
  
        r = random()
        # shuffle(word_tag_prob)
        for w, t, p in word_tag_prob:
            r = r - p
            if r <= 0:
                return w, t

    def generate_words(self, nwords = 25):
        history_layer1 = ['~'] * (self.n_layer1 - 1)
        history_layer2 = ['~'] * (self.n_layer2 - 1)
        text = ""
        for i in range(nwords):
            word, tag = self.next_token(history_layer1, history_layer2)
            if self.n_layer1 > 2: history_layer1 = history_layer1[-(self.n_layer1 - 2):] + [word]
            else: history_layer1 = [word]
            if self.n_layer2 > 2: history_layer2 = history_layer2[-(self.n_layer2 - 2):] + [tag]
            else: history_layer2 = [tag]
            text = text + word + " "
        return text

!wget https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
text = open('shakespeare_input.txt').read()
sentences = nltk.sent_tokenize(text)

ngram = bilayer_ngram(n_layer1=4, n_layer2=4)

for sentence in sentences:
    clean_sentence = generate_clean_words(sentence)
    ngram.train_layer1(clean_sentence)
    ngram.train_layer2(extract_POS(clean_sentence))

ngram.generate_words(7)

