import re
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

all_vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 'ih', 'iy', 'ow', 'oy', 'uh', 'uw']

all_consonants = ['b', 'ch', 'd', 'dh', 'f', 'g', 'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']


def all_vowel(syllable):
    if syllable in all_vowels:
        return True
    else:
        return False


def all_consonant(syllable):
    if syllable in all_consonants:
        return True
    else:
        return False


def head_comb(syllables):
    for i in range(1, len(syllables)):
        one_syllable = syllables[i - 1]
        two_syllable = syllables[i]

        if all_vowel(one_syllable):
            return 'Notation' + one_syllable
        elif all_consonant(one_syllable) and all_vowel(two_syllable):
            return one_syllable + two_syllable


def end_comb(syllables):
    result = ''
    for i in range(1, len(syllables)):
        one_syllable = syllables[i - 1]
        two_syllable = syllables[i]
        if all_consonant(one_syllable) and all_vowel(two_syllable):
            result = one_syllable + two_syllable
        elif all_vowel(one_syllable) and all_vowel(two_syllable):
            result = 'Notation2' + two_syllable
    return result


def find_vowels(syllables):
    vowels = [syllable for syllable in syllables if all_vowel(syllable)]
    return vowels + ['Notation'] * (4 - len(vowels))


def find_char(words, control=False):
    chars = []
    for word in words:
        char = []
        syllables = word.split()
        if control:
            vowels = [syllable for syllable in syllables if syllable[-1].isdigit()]
            stress_result = [int(vowel[-1]) for vowel in vowels]
            stress = stress_result.index(1) + 1
            word = re.sub('\d', '', word)
            syllables = word.split()

        all_v = str(len([syllable for syllable in syllables if all_vowel(syllable)]))
        first_v, second_v, third_v, last_v = find_vowels(syllables)

        head_combine = head_comb(syllables)
        end_combine = end_comb(syllables)
        first_vowel = all_vowel(syllables[0])
        last_vowel = all_vowel(syllables[-1])
        char.extend([all_v, first_v, second_v, third_v, last_v, first_vowel, last_vowel,
                        head_combine, end_combine])

        if control:
            char.append(stress)
        chars.append(char)
    return chars


def produce_matrix(chars, control=False, path='matrix.dat'):
    all_char = ['all_v', 'first_v', 'second_v', 'third_v',
                 'last_v', 'first_vowel', 'last_vowel', 'head_combine', 'end_combine']

    char_dic = [{all_char[col]: row[col] for col in range(len(all_char))}
                   for row in chars]

    if control:
        dv = DictVectorizer(sparse=False).fit(char_dic)

        X = dv.transform(char_dic)
        y = np.array([i[-1] for i in chars])

        with open(path, 'wb') as f:
            pickle.dump(dv, f)

        return X, y

    else:
        with open(path, 'rb') as f:
            dv = pickle.load(f)

        return dv.transform(char_dic)




def train(data, classifier_file):  # do not change the heading of the function
    data = [i.split(':')[1].lower() for i in data]
    chars = find_char(data, control=True)
    X, y = produce_matrix(chars, control=True)

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    with open(classifier_file, 'wb') as f:
        pickle.dump(clf, f)


def test(data, classifier_file):  # do not change the heading of the function
    data = [i.split(':')[1].lower() for i in data]
    chars = find_char(data)
    X = produce_matrix(chars)

    with open(classifier_file, 'rb') as f:
        clf = pickle.load(f)

    return list(clf.predict(X))
