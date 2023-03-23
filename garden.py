import spacy

nlp = spacy.load('en_core_web_sm')

gardenpathSentences = ["The man who whistles tunes pianos",
                       "I know the words to that song about the Queen don’t rhyme",
                       "The cotton clothing is usually made of grows in Mississippi",
                       "The chicken is ready to eat is undercooked",
                       "The prime number few people have is 23",
                       "Mary gave the child the dog bit a Band-Aid"]

for sentence in gardenpathSentences:
    doc = nlp(sentence)
    tokenizedDoc = [token.orth_ for token in doc]
    recognisedDoc = [(i, i.label_,) for i in doc.ents]
    print(f"\nSentance: {sentence}\n"
          f"Tokenized sentence: {tokenizedDoc}\n"
          f"Recognised entites: {recognisedDoc}\n")


print("GPE: " + spacy.explain("GPE"))
print("PERSON: " + spacy.explain("PERSON"))


# The explanation for GPE was correct for Mississippi as it is a city. The term GPE is however a bit convoluted.

# The explanation for PERSON was correct for both the Queen and Mary however Mary, although it is a name, is not a recognisable individual
# unless it is referring to Mary from the bible.

