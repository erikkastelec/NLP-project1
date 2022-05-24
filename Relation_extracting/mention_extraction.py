import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize
from nltk import RegexpParser
from nltk import Tree
import pandas as pd
import numpy as np
import classla

classla.download('sl')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class MentionDetector:
    
    def __init__(self):
        
        # Defining a grammar & Parser
        self.NP_en = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
        self.chunker_en = RegexpParser(self.NP_en)
        # todo: slo grammar
        self.NP_slo = ""
        self.chunker_slo = None
        

    def get_continuous_chunks(self, text, chunk_func=ne_chunk):

        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        print(tags)
        
        # first filtering
        props = [t[0] for t in tags if t[1] in ["PRP"]]
        singletons = [t[0] for t in tags if t[1] in ["NNP", "NNS"]]
        print(props)
        
        # noun phrases for NE
        chunked = chunk_func(tags)
        #print(chunked)

        continuous_chunk = []
        current_chunk = []
        for subtree in chunked:

            if type(subtree) == Tree:
                tokens = [token for token, pos in subtree.leaves()]
                for token in tokens:
                    if token in singletons:
                        singletons.remove(token)
                current_chunk.append(" ".join(tokens))

            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []

            else:
                continue
        
        continuous_chunk.extend(props)
        continuous_chunk.extend(singletons)
        print('\n')
        return continuous_chunk
    
    def get_continuous_chunks2(self, tags, chunk_func=ne_chunk):

        print(tags)
        
        # first filtering
        props = [t[0] for t in tags if t[1] in ["PRP"]]
        singletons = [t[0] for t in tags if t[1] in ["NNP", "NNS"]]
        print(props)
        
        # noun phrases for NE
        chunked = chunk_func(tags)
        #print(chunked)

        continuous_chunk = []
        current_chunk = []
        for subtree in chunked:

            if type(subtree) == Tree:
                tokens = [token for token, pos in subtree.leaves()]
                for token in tokens:
                    if token in singletons:
                        singletons.remove(token)
                current_chunk.append(" ".join(tokens))

            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []

            else:
                continue
        
        continuous_chunk.extend(props)
        continuous_chunk.extend(singletons)
        print('\n')
        return continuous_chunk


    def get_mentions(self, text, lang):
        
        sentences = sent_tokenize(text)
#         mentions = []
#         for s in sentences:
#             mentions.append(self.get_continuous_chunks((s)))
#          return mentions

        if lang == 'english':
            mentions = []
            for s in sentences:
                mentions.append(self.get_continuous_chunks(s, self.chunker_en.parse))
            return mentions

        elif lang == 'slovenian':
            self.parse_slo(text)
            #raise NotImplementedError
            
            
    def parse_slo(self, text):
        sl_pipeline = classla.Pipeline("sl", processors='tokenize, ner, lemma, pos, depparse')
        doc = sl_pipeline(text)
        tags = [(token.text, token.pos_) for token in doc]
        m = get_continuous_chunks2()
        print(d.to_conll())

