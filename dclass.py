import os

DClasses = ["clinton", "lawyer", "math", "medical", "music", "sex"]

base = "data/jokes/learn/"
p = Pool()
for dclass in DClasses:
    p.learn(base + dclass, dclass)

base = "data/jokes/test/"
results = []
for dclass in DClasses:
    dir = os.listdir(base + dclass)
    for file in dir:
        res = p.Probability(base + dclass + "/" + file)
        results.append(f"{dclass}: {file}: {str(res)}")

print(results[:10])