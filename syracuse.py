def syracuse(integer):
    counter = 0
    array = []

    while integer > 1:
        if integer % 2 == 0:
            integer = integer // 2
            array.append(integer)
        else:
            integer = integer * 3 + 1
            array.append(integer)
        counter+=1

    return array

def sum_syracuse(integer):
    return sum(syracuse(integer))

def largest_syracuse(N):
    array = []
    index = 0

    for i in range(1, N+1):
        array.append(len(syracuse(i)))
        largest_path = max(array)
        index = array.index(largest_path) + 1
    
    return "The largest syracuse sequence within 1 and " + str(N) + " has a length of " + str(largest_path) + " steps, which corresponds to " + str(index)

print(largest_syracuse(1000))
print(syracuse(871))
print(sum_syracuse(871))