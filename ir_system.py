import json

file = open("recipes/recipes_raw_nosource_fn.json", "r")
data = json.load(file)
file.close()

print(data)