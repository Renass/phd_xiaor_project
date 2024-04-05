import time

STARTING_POINTS = [
    [-7.05, 6.51, -0.58, 0.82],
    [-5.77, 4.33, 0.22, 0.98],
    [-5.13, 2.73, 0.86, 0.51],
    [-11.28, 4.76, 0.18, 0.98]
]

def iterable_index_circle(length):
    while True: 
        for i in range(length):
            yield i

starting_point_ind = iterable_index_circle(len(STARTING_POINTS))
while True:
    print(STARTING_POINTS[next(starting_point_ind)])
    time.sleep(1)