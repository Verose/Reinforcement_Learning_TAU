import numpy as np
def find_most_probable_word(word_size, consecutive_letter_dist_matrix):
    num_letters = consecutive_letter_dist_matrix.shape[0]
    f = np.full((word_size, num_letters), np.inf)
    for i in range(1, word_size):
        for letter in num_letters:
            for consec_letter in num_letters:
                val = (1-consecutive_letter_dist_matrix)[letter, consec_letter]+f[i+1, consec_letter]
                if f[i, letter]>=val:
                    f[i, letter] = val
    probable_word = [np.argmin(f[0, :])]
    for i in range(1, word_size):
        first_word = np.argmin(f[i, probable_word[i-1]])
        probable_word.append(first_word)
    return probable_word