from nltk.stem.porter import *
import nltk
#from spacy import displacy
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

doc = nlp("""Perhaps one of the most significant advances made by Arabic mathematics began at this time with the work of al-Khwarizmi, namely
the beginnings of algebra. It is important to understand just how significant this new idea was. It was a revolutionary move away from
the Greek concept of mathematics which was essentially geometry. Algebra was a unifying theory which allowed rational
numbers, irrational numbers, geometrical magnitudes, etc., to all be treated as "algebraic objects". It gave mathematics a whole new
development path so much broader in concept to that which had existed before, and provided a vehicle for future development of the
subject. Another important aspect of the introduction of algebraic ideas was that it allowed mathematics to be applied to itself in a
way which had not happened before.""")
token_List = []
for token in doc:
    token_List.append(token.text)

print(token_List)

for sent in doc.sents:
    print(sent)

print([(token.text, token.tag_) for token in doc])

for token in doc:
    print(token.text, token.lemma_, token.pos, token.tag_,
          token.dep_, token.shape_, token.is_alpha, token.is_stop)

p_stemmer = PorterStemmer()
for words in doc:
    print(words.text + '>>>>>>' + p_stemmer.stem(words.text))


for chunk in doc.noun_chunks:
    print(chunk.text, chunk.label_, chunk.root.text)
