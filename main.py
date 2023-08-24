import numpy as np
# import math


def find_infinite_norm(vector):  # επιστρέφει άπειρη νόρμα διανύσματος / πίνακα γραμμή
    max_num = -1.0
    for num in vector:
        if abs(num) > max_num:
            max_num = abs(num)
    return max_num


def first_norm(vector):  # για διάνυσμα στήλη
    alist = [float(vector[row, 0]) for row in range(vector.shape[0])]
    return sum(alist)


# Ελέγχω για ακρίβεια 5ου δεκαδικού ψηφίου
def check_accuracy(array1, array2):
    error = find_infinite_norm(array1 - array2)
    if error > 0.5 * pow(10, -5):
        return True  # reiterate
    return False


def normalize_vector_column_for_singular_sum(vector_column):
    suma = first_norm(vector_column)
    # Πρέπει η L1 νόρμα να βγαίνει ένα οπότε διαιρώ όλα τα στοιχεία με το άθροισμα των στοιχείων του διανύσματος
    return vector_column / suma


# επιστρέφω μια φορά όλα τα αθροίσματα, ώστε να μη χρειάζεται να υπολογίζω το n_j πολλαπλές φορές
# στον υπολογισμό των στοιχείων του G
def find_sums_of_rows(matrix):  # matrix --> τετραγωνικός πίνακας n x n, όπου n >= 2, αλλιώς χτυπάει στο shape
    columns = matrix.shape[0]
    sums = []
    for i in range(columns):
        row_values = [matrix[i, j] for j in range(columns)]
        sums.append(sum(row_values))
    return sums


def construct_google_matrix(A_matrix, q):
    n = A_matrix.shape[0]
    sums = find_sums_of_rows(A_matrix)
    google_matrix = np.empty((n, n), dtype='f')
    for i in range(n):
        for j in range(n):
            google_matrix[i, j] = (float(q) / n) + (A_matrix[j, i] * (1 - q)) / sums[j]
    return google_matrix


def power_method(matrix):
    dim = matrix.shape[0]
    current_eigenvector = np.empty((dim, 1), dtype='f')
    for i in range(dim):
        current_eigenvector[i, 0] = i + 1
    previous_eigenvector = np.zeros((dim, 1), dtype='f')
    while True:
        if not check_accuracy(current_eigenvector, previous_eigenvector):
            break
        previous_eigenvector = current_eigenvector.copy()
        current_eigenvector = np.matmul(matrix, current_eigenvector)
        current_eigenvector = current_eigenvector.copy() / current_eigenvector[0, 0]
    return current_eigenvector


A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]])

g_matrix = construct_google_matrix(A, 0.15)
eigen_vec = power_method(g_matrix)
eigen_vec = normalize_vector_column_for_singular_sum(eigen_vec)  # Κανονικοποίηση για μοναδιαίο άθροισμα
print("Page rank of each website:\n")
counter=0
for rank in eigen_vec:
    counter+=1
    print(counter, ": %.4f" % float(rank))
