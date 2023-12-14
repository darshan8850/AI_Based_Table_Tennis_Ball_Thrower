import cv2
import numpy as np
import json


points = []
grid_lines = []

def draw_polygon(img, points):
    if len(points) == 4:

        pts = np.array(points, dtype=np.int32)

        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        generate_grid_lines(points, img)

        data = {
            "points": points,
            "lines": [(points[0], points[1]), (points[1], points[2]), (points[2], points[3]), (points[3], points[0])],
            "grid_lines": grid_lines, # Convert NumPy array to list for JSON
        }

        with open("coordinates_and_lines.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    cv2.imshow('image', img)
    
def click_event(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Point added:", x, y)

        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('image', img)

        draw_polygon(img, points)

def generate_grid_lines(points, img):
    global grid_lines

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    num_rows = 5 
    num_cols = 5  

    for i in range(1, num_rows):
        y = int(min_y + (max_y - min_y) * i / num_rows)
        grid_lines.append(((min_x, y), (max_x, y)))


        cv2.line(img, (min_x, y), (max_x, y), (0, 255, 0), 1)

    for i in range(1, num_cols):
        x = int(min_x + (max_x - min_x) * i / num_cols)
        grid_lines.append(((x, min_y), (x, max_y)))

        cv2.line(img, (x, min_y), (x, max_y), (0, 255, 0), 1)




img = cv2.imread("C:/Users/HP/Desktop/Table_tennis/IMG_20230829_130142.jpg", 1)


scale_factor = 0.2  # Adjust this factor as needed


img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)


cv2.imshow('image', img)

cv2.setMouseCallback('image', click_event)


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break


def create_gaussian_matrix(r, c):
    # Create an r x c matrix filled with Gaussian probabilities
    mean = [r // 2, c // 2]
    cov = [[r**2, 0], [0, c**2]]
    x, y = np.meshgrid(range(r), range(c))
    matrix = np.exp(-(np.square(x - mean[0]) / (2 * cov[0][0]) + np.square(y - mean[1]) / (2 * cov[1][1])))
    matrix /= np.sum(matrix)  # Normalize to make the sum equal to 1
    return matrix

def throw_ball(matrix, throws):
    for throw in range(throws):
        # Find the cell with the maximum probability
        max_cell = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)

        # Decrease the probability of the selected cell
        probability = matrix[max_cell]
        matrix[max_cell] *= 0.9

        # Decrease the probability of nearby cells (2x2 neighborhood)
        for i in range(max_cell[0] - 1, max_cell[0] + 1):
            for j in range(max_cell[1] - 1, max_cell[1] + 1):
                if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
                    matrix[i, j] *= 0.8

        # Print the throw information
        print(f"Throw {throw + 1}: Ball thrown at cell {max_cell} with probability {probability:.4f}")

        # Print the updated matrix after each throw
        print("Updated Matrix:")
        print(matrix)
        print()

    return matrix

# User input for matrix size and number of throws
rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))
throws = int(input("Enter the number of throws: "))

# Create the Gaussian matrix
probability_matrix = create_gaussian_matrix(rows, cols)

# Simulate ball throws
updated_matrix = throw_ball(probability_matrix, throws)

# Print the final matrix
print("Final Matrix:")
print(updated_matrix)


cv2.destroyAllWindows()
