import redis

r = redis.Redis()

cursor = 0
results = []

while True:
    cursor, keys = r.scan(cursor=cursor, match="dups:*", count=1000)
    
    for key in keys:
        count = r.scard(key)
        if count > 1:
            sha = key.decode().split("dups:")[1]
            results.append((sha, count))

    if cursor == 0:
        break

results.sort(key=lambda x: x[1], reverse=True)

subcount = 0
# Print results
for sha, count in results:
    if (count > 2):
        subcount += 1
        print(f"{sha} -> {count}")

print(f"\nTotal structures with more than 2 duplicates: {subcount}")
print(f"\nTotal duplicate structures: {len(results)}")
