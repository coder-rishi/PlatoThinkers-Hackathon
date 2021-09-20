# The reporter
import random
num = random.randint(1,10)

if num % 2 == 0:
    fin = 0
elif num % 2 == 1:
    fin = 1

print(fin, flush=True, end='')
