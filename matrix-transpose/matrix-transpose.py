import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here

    current_rows = len(A)
    current_columns = len(A[0]) 

    new_array = np.zeros((current_columns,current_rows),dtype = int) 


    for i in range(current_rows):
        for j in range(current_columns): 
            new_array[j][i] = A[i][j]

    return new_array 