'''
problem: worker assign to tasks with different cost. and multiple workers can be assigned to same task
we want to find the minimum cost to assign all tasks to workers and 
the sub-optimal solutions for each task
'''
import numpy as np
import copy

def get_min_value(cost_matrix):
    # return the minimum value and index in cost_matrix in each row
    # cost_matrix: 2d numpy array
    min_index_list = []
    copy_cost_matrix = copy.deepcopy(cost_matrix)
    for row in copy_cost_matrix:
        # return the index of minimum value in each row and minimum value
        min_index_list.append(np.argmin(row))
        # - minimum value in each row
        row -= np.min(row)
    return min_index_list, copy_cost_matrix

def get_sub_optimal_solution(new_cost_matrix):
    # sub_optimal_solution means the total cost is second minimum
    # new_cost_matrix: 2d numpy array
    # choose the minimum value of matrix except 0
    min_cost = np.min(new_cost_matrix[np.nonzero(new_cost_matrix)])
    # return its index
    min_index = np.where(new_cost_matrix == min_cost)
    # reset the min_cost to big number
    new_cost_matrix[min_index] = 100000
    return min_index, new_cost_matrix
    
def get_num_assignments(cost_matrix, num_assignments):
    assignments = []
    min_index_list, new_cost_matrix = get_min_value(cost_matrix)
    assignments.append(min_index_list)
    if num_assignments > 1:
        for num in range(1,num_assignments):
            submin_index_list = copy.deepcopy(min_index_list)
            min_index, new_cost_matrix = get_sub_optimal_solution(new_cost_matrix)
            if len(min_index[0]) == 1:
                # turn array to list
                submin_index_list[min_index[0][0].tolist()] = min_index[1][0].tolist()
                assignments.append(submin_index_list)
            else:
                for i in range(len(min_index)):
                    submin_index_list = copy.deepcopy(min_index_list)
                    submin_index_list[min_index[i][0]] = min_index[i][1]
                    assignments.append(submin_index_list)
    return assignments

# generate an example
# cost_matrix = np.array([[5,2,3],[4,5,1],[3,8,4],[10,18,12]])
# num_assignments = 3
# assignments = get_num_assignments(cost_matrix, num_assignments)
# print(assignments)
