def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def find_best_match(text, directory='./color_dataset.txt'):
    best_match = None
    best_distance = float('inf')

    with open(directory, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            distance = levenshtein_distance(text.lower(), line.lower())
            if distance < best_distance:
                best_match = line
                best_distance = distance

    return best_match

# # Example usage:
# text_to_search = "ngad d2n"
# result = find_best_match(text_to_search)
# print("Best match:", result)
