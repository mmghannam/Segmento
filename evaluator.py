def sum_square_error(assignment):
    error = 0
    for mean in assignment.keys():
        for element in assignment[mean]:
            error += norm(element - mean) ** 2
    # print(error)
    return error