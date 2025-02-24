import cv2
import numpy as np
import os


"""
TODO Binary transfer
"""
def to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = np.zeros_like(gray, dtype=np.uint8)
    binary_img[gray > 127] = 255
    return binary_img

    raise NotImplementedError


"""
TODO Two-pass algorithm
"""
def two_pass(binary_img, connectivity):
    label = np.zeros_like(binary_img, dtype=np.int32)
    rows, cols = binary_img.shape
    current_label = 1
    equivalence = {}

    # first
    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 255:
                neighbors = []
                if i > 0 and label[i - 1, j] > 0:  # upper neighbor
                    neighbors.append(label[i - 1, j])
                if j > 0 and label[i, j - 1] > 0:  # left neighbor
                    neighbors.append(label[i, j - 1])
                if connectivity == 8:
                    if i > 0 and j > 0 and label[i - 1, j - 1] > 0:  # left upper neighbor
                        neighbors.append(label[i - 1, j - 1])
                    if i > 0 and j < cols - 1 and label[i - 1, j + 1] > 0:  # right upper nieghbor
                        neighbors.append(label[i - 1, j + 1])

                if neighbors:
                    min_label = min(neighbors)
                    label[i, j] = min_label
                    for n in neighbors:
                        equivalence.setdefault(n, set()).add(min_label)
                        equivalence[min_label].update(neighbors)
                else:
                    label[i, j] = current_label
                    equivalence[current_label] = {current_label}
                    current_label += 1

    for key in equivalence:
        equivalence[key] = min(equivalence[key])

    #second
    for i in range(rows):
        for j in range(cols):
            if label[i, j] > 0:
                label[i, j] = equivalence[label[i, j]]

    label = remove_small_regions(label, min_size=500)
    return label
    raise NotImplementedError


"""
TODO Seed filling algorithm
"""
def seed_filling(binary_img, connectivity):
    label = np.zeros_like(binary_img, dtype=np.int32)
    rows, cols = binary_img.shape
    current_label = 1

    def flood_fill(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if label[cx, cy] == 0 and binary_img[cx, cy] == 255:
                label[cx, cy] = current_label
                neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                if connectivity == 8:
                    neighbors += [(cx - 1, cy - 1), (cx - 1, cy + 1), (cx + 1, cy - 1), (cx + 1, cy + 1)]
                for nx, ny in neighbors:
                    if 0 <= nx < rows and 0 <= ny < cols:
                        stack.append((nx, ny))

    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 255 and label[i, j] == 0:
                flood_fill(i, j)
                current_label += 1

    label = remove_small_regions(label, min_size=500)
    return label
    
    raise NotImplementedError


"""
Bonus
"""
def remove_small_regions(label_img, min_size):
    unique_labels = np.unique(label_img)
    for label in unique_labels:
        if np.sum(label_img == label) < min_size:
            label_img[label_img == label] = 0  #background
    return label_img



"""
TODO Color mapping
"""
def color_mapping(label):
    unique_labels = np.unique(label)
    color_map = {label: np.random.randint(0, 255, 3).tolist() for label in unique_labels if label > 0}
    color_map[0] = [0, 0, 0]  # Background is black
    color_img = np.zeros((*label.shape, 3), dtype=np.uint8)

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color_img[i, j] = color_map[label[i, j]]

    return color_img
    raise NotImplementedError


"""
Main function
"""
def main():

    os.makedirs("result/connected_component_remove/two_pass", exist_ok=True)
    os.makedirs("result/connected_component_remove/seed_filling", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img)

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
        
            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite("result/connected_component_remove/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            cv2.imwrite("result/connected_component_remove/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)


if __name__ == "__main__":
    main()