# Robert Paßmann

from collections import Counter
import random

CHUNK_LENGTH = 500
INITIAL_FEATURE_SET_LENGTH = 250

def text_to_list(text):
    # the following takes all alphabetic words normalized to lowercase
    # from the raw data
    return [x for x in 
                    [''.join(c for c in word if c.isalpha()).lower() 
                     for word in text.split()] 
                 if x != '']
                 
def select_chunks(text1,text2):
    """Reduce the number of chunks of the text with more chunks such that
       we have the same number of chunks for both texts"""
    random.seed()
    text1.selected_chunks = text1.chunks
    text2.selected_chunks = text2.chunks
    while len(text1.selected_chunks) > len(text2.selected_chunks):
        text1.selected_chunks.remove(random.choice(text1.selected_chunks))
    while len(text2.selected_chunks) > len(text1.selected_chunks):
        text2.selected_chunks.remove(random.choice(text2.selected_chunks))

class Database:
    """represents a database with texts of known authors"""
    
    def __init__(self):
        self.authors = [] # a list of strings with names of authors
        self.texts = {} # a dictionary (name:Text)
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
                
        self.initial_feature_set = list(dict(counter.most_common(INITIAL_FEATURE_SET_LENGTH)).keys())
        
    
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
        
        self.chunks = [] # containing all the chunks of n words
        self.selected_chunks = [] # contains a reduced number of chunks
                                  # for having the same number of chunks
                                  # for two text during calculations
        self.tokens = []
        
        self.chunk_feature_frequencies = {}
        
        
    def create_chunks(self):
        global CHUNK_LENGTH
        
        self.tokens = text_to_list(self.raw)
        n = len(self.tokens)
        
        if n < CHUNK_LENGTH:
            raise Exception("Text is too short")
        
        chunk_endpoints = list(range(CHUNK_LENGTH,n+1,CHUNK_LENGTH))
        if n not in chunk_endpoints:
            chunk_endpoints.append(n)
            
        for endpoint in chunk_endpoints:
            self.chunks.append(self.tokens[endpoint-CHUNK_LENGTH:endpoint])