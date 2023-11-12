import spacy

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """Founded back in 1969 as Samsung Electric Industries, 
Suwon, South Korea-headquartered Samsung Electronics today makes everything 
from televisions to semiconductors. It released its first Android smartphone
 in 2009, and can be credited with the launch of the first Android tablet back
  in 2010. The company is among the biggest players in the smartphone market in 
  the world. It has recently developed smartphones running Tizen OS, as an 
  alternative to its Android-based smartphones. Samsung's latest mobile launch 
  is the Galaxy A05s. The mobile was launched in 15th October 2023. The phone comes with a 6.70-inch touchscreen
 display with a resolution of 2400 pixels by 1080 pixels. 
 The Samsung Galaxy A05s is powered by octa-core Qualcomm Snapdragon 680 processor
  and it comes with 4GB of RAM. The phone packs 128GB of internal storage that can
   be expanded up to via a microSD card. As far as the cameras are concerned, the 
   Samsung Galaxy A05s packs a 50-megapixel + 2-megapixel + 2-megapixel primary camera 
   on the rear and a 13-megapixel front shooter for selfies. The Samsung Galaxy A05s runs
    Android 13 and is powered by a 5000mAh non removable battery. It measures 168.00 x 
    77.80 x 8.80 (height x width x thickness) and weigh 194.00 grams. The Samsung Galaxy
     A05s is a dual SIM mobile that accepts . Connectivity options include Wi-Fi and USB 
     Type-C. Sensors on the phone include Fingerprint sensor, Proximity sensor, 
     Accelerometer and Ambient light sensor."""

def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    # print(stopwords)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    # print(doc)

    tokens = [token.text for token in doc]
    # print(tokens)

    word_frequency = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequency.keys():
                word_frequency[word.text] = 1
            else:
                word_frequency[word.text] += 1

    # print(word_frequency)

    max_freq = max(word_frequency.values())
    # print(max_freq)

    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word]/max_freq

    # print(word_frequency)

    sent_tokens = [sent for sent in doc.sents]
    # print(sent_tokens)

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_frequency.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_frequency[word.text]
                else:
                    sent_scores[sent] += word_frequency[word.text]

    # print(sent_scores)

    select_len = int(len(sent_tokens)*0.3)
    # print(select_len)

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    # print(summary)

    final_summary = [word.text for word in summary]
    summary= " ".join(final_summary)
    # print(f"Original Text: {text}")
    # print(f"Summarized Text: {summary}")
    # print(len(text.split(' ')))
    # print(len(summary.split(' ')))

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))