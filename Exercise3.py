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

    # end of the word - the consecutive must be '-'
    #f[num_letters-1, word_size] = 1
    f[:, word_size] = cost_matrix[:, 3]

    for i in reversed(range(word_size)):
        if i!=0: #if not the first letter
            for letter, c in enumerate(cost_matrix):
                f[letter, i] = np.min(c*f[:, i + 1].transpose())
        else:  # if is is the first letter - it must be letter b (in index 0)
            f[0, i] = np.min(cost_matrix[0, :]*f[:, i + 1].transpose())

    probable_word = [letters_simb[np.argmin(f[:, 0])]]
    for i in range(1, word_size):
        first_word = np.argmin(f[:, i])
        probable_word.append(letters_simb[first_word])

    return probable_word

most_probable_word = find_most_probable_word(5)
print('The most probable word is: {}'.format(most_probable_word))

