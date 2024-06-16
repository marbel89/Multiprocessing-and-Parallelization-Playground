import time

nums_list = list(range(100000000))

start = time.time()
print(1000 in nums_list)
end = time.time()
print(end - start)

nums_set = set(range(100000000))

start = time.time()
print(1000 in nums_set)
end = time.time()
print(end - start)

# Sets are more efficient, but cannot contain duplicate values.
