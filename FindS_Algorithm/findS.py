import csv
def findS():
    num_attributes = 0
    a = []
    t = []

    # Read the training data from a CSV file
    with open('tennis (1).csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if num_attributes == 0:
                num_attributes = len(row) - 1
            a.append(row[:-1])
            t.append(row[-1])

    # Initialize the hypothesis to the first positive example
    hypothesis = ['0'] * num_attributes
    for i in range(len(a)):
        if t[i] == 'Yes':
            hypothesis = a[i]
            break

    # Refine the hypothesis based on the training data
    for i in range(len(a)):
        if t[i] == 'Yes':
            for j in range(num_attributes):
                if a[i][j] != hypothesis[j]:
                    hypothesis[j] = '?'

    # Print the final hypothesis
    print('Final hypothesis:', hypothesis)
    return hypothesis


findS()
