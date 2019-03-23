import numpy as np

letters_simb = ['B', 'K', 'O', '-']

consecutive_letter_dist_matrix = np.array([
    [0.1,   0.325,  0.25,   0.325],     # b
    [0.4,   0,      0.4,    0.2],       # k
    [0.2,   0.2,    0.2,    0.4],       # o
    [1,     0,      0,      0],         # -
])


def find_most_probable_word(word_size):
    cost_matrix = 1-consecutive_letter_dist_matrix
    num_letters = consecutive_letter_dist_matrix.shape[0]
    f = np.full((num_letters, word_size+1), np.inf)

    f[num_letters-1, word_size] = 1 ## end of the word - the consecutive must be '-'

    for i in reversed(range(word_size)):
        if i!=0 and i!=word_size-1: #if not the first letter and not the last (the last letter must be -)
            for letter in range(num_letters-1):
                f[letter, i] = np.min(cost_matrix[letter, 0:(num_letters-1)]*f[0:(num_letters-1), i + 1].transpose())
        elif i==word_size-1: #if it is the last letter it must be "-"
            for letter in range(num_letters-1):
                f[letter, i] = np.min(cost_matrix[letter, num_letters-1] * f[:, i + 1].transpose())
        elif i == 0:  # if is is the first letter of the word - it must be letter b (index 0)
            f[0, i] = np.min(cost_matrix[0, 0:(num_letters-1)]*f[0:(num_letters-1), i + 1].transpose())

    probable_word = []
    for lett_in_word in range(word_size+1):
        word_index = np.argmin(f[:, lett_in_word])
        probable_word.append(letters_simb[word_index])

    return probable_word

most_probable_word = find_most_probable_word(5)
print('The most probable word is: {}'.format(most_probable_word))

