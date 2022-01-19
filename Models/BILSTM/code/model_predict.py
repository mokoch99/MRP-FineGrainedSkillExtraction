from tensorflow import keras
import gensim
import numpy as np

def make_prediction(text):
    model = keras.models.load_model('/Users/kiliankramer/Desktop/experiments/final/models/custom fasttext layer 1 contextual_true oversampling_false')
    FastText = gensim.models.FastText.load("/Users/kiliankramer/Desktop/FastText.model")

    text = text.replace(',', ' , ')
    text = text.replace('.', ' . ')
    vector = []
    text = text.split()

    for i in text:
        if i in FastText.wv:
            vector.append(FastText.wv[i])
    for i, vec in enumerate(vector):
        vector[i] = vec.tolist()

    samples = []

    if len(vector) >= 3:
        for i, vec in enumerate(vector[1:-1]):
            samples.append([vector[i-1], vector[i], vector[i+1]])

    y_pred = model.predict(samples)
    y_pred_list = []
    for i in y_pred:
        y_pred_list.append(np.argmax(i, axis=0))
    y_pred = [0] + y_pred_list + [0]

    label_dict = {   0: 'no skill', 1: 'business administration', 2: 'education, sales and marketing',
                     3: 'engineering, construction and transport', 4: 'information technology', 5: 'health and social care',
                     6: 'science and research', 7: 'Communication', 8: 'Time management', 9: 'Decision making', 10: 'Optimism',
                     11: 'Coaching', 12: 'Initiative', 13: 'Social skills', 14: 'Creativity', 15: 'Leadership', 16: 'Self-confidence',
                     17: 'Enthusiasm', 18: 'Curiosity', 19: 'Teamwork', 20: 'Goal', 21: 'Critical thinking', 22: 'Flexibility',
                     23: 'Integrity', 24: 'Hospitality', 25: 'Conflict management', 26: 'Ethic', 27: 'Negotiation', 28: 'Adaptive',
                     29: 'Active listening', 30: 'Presentation', 31: 'Problem solving', 32: 'Emotional intelligence', 33: 'Kindness',
                     34: 'Interpersonal communication', 35: 'Argumentation', 36: 'Persuasion', 37: 'Mentoring', 38: 'Eagerness',
                     39: 'Writing', 40: 'Influence', 41: 'Passion', 42: 'Decision', 43: 'Conceptual', 44: 'Accountability',
                     45: 'Detail', 46: 'Strategic thinking'   }

    for i, y_pre in enumerate(y_pred):
        y_pred[i] = label_dict[y_pre]

    return y_pred

if __name__ == '__main__':
    text = 'This is an example text. I like programming languages like Python very much.'

    result = make_prediction(text)

    print(text)
    print(result)








