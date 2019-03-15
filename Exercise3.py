import numpy as np

consecutive_letter_dist_matrix = np.array([
    [0.1,   0.325,  0.25,   0.325],     # b
    [0.4,   0,      0.4,    0.2],       # k
    [0.2,   0.2,    0.2,    0.4],       # o
    [1,     0,      0,      0],         # -
])


def find_most_probable_word(word_size):
    num_letters = len(consecutive_letter_dist_matrix)
    f = np.full((word_size, num_letters), np.inf)

    for i in range(word_size):
        for letter in consecutive_letter_dist_matrix:
            for consec_letter in consecutive_letter_dist_matrix:
                val = (1 - consecutive_letter_dist_matrix[letter][consec_letter]) + f[i + 1, consec_letter]
                if f[i, letter] >= val:
                    f[i, letter] = val
    probable_word = [np.argmin(f[0, :])]

    for i in range(word_size):
        first_word = np.argmin(f[i, probable_word[i - 1]])
        probable_word.append(first_word)

    return probable_word


most_probable_word = find_most_probable_word(5)
print('The most probable word is: {}'.format(most_probable_word))
