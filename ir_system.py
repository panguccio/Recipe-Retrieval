import json

file = open("recipes/recipes.json", "r")
data = json.load(file)
file.close()

print(data["1"])