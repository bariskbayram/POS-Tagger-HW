from pos_tagger import pos_tag
from nltk import word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = pos_tag(words)

            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
