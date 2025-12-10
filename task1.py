from collections import deque


def manual_input():
    # line = input("Input M and N (number of rows and columns): ")
    line = input()
    mn = list(map(int, line.split()))
    if len(mn) != 2:
        print("Incorrect input")
        exit()

    # print("Input matrix: ")
    matrix = []
    for _ in range(mn[0]):
        row = list(map(int, input().split()))
        if len(row) != mn[1]:
            print("Incorrect number of elements in row")
            exit()
        matrix.append(row)
    return mn[0], mn[1], matrix


def bfs(m, n, matrix):
    sides = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    islands = 0

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                islands += 1
                matrix[i][j] = 0
                q = deque([(i, j)])
                while q:
                    i_q, j_q = q.popleft()
                    for side in sides:
                        i_n = i_q + side[0]
                        j_n = j_q + side[1]
                        if 0 <= i_n < m and 0 <= j_n < n:
                            if matrix[i_n][j_n] == 1:
                                matrix[i_n][j_n] = 0
                                q.append((i_n, j_n))
    return islands


if __name__ == "__main__":
    M, N, mtrx = manual_input()
    print(f"Number of Islands is {bfs(M, N, mtrx)}")
