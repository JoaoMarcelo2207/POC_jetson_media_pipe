import cv2
import numpy as np
from collections import deque

import video_adjuster_functions as vid_adj_fun


window_scale_size = 2
normal_height_img = 300 * window_scale_size
normal_width_img = 240 * window_scale_size

PLOT_SIZE = 200  # Deve bater com o tamanho dos buffers


def create_canvas(camera_space, scatter_space, line_chart_space):
    """
    Creates a larger blank canvas with designated regions for the camera, scatter plot, and line chart.
    """
    canvas_width = camera_space[0] + scatter_space[0]
    canvas_height = max(camera_space[1], scatter_space[1]) + line_chart_space[1]

    # Create the blank canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White

    positions = {
        "camera": (canvas_width - camera_space[0], 0),
        "scatter": (canvas_width - camera_space[0] - scatter_space[0], 0),
        "line_chart": (0, max(camera_space[1], scatter_space[1]))
    }

    return canvas, positions, camera_space, scatter_space, line_chart_space


def plot_scatter(canvas, scatter_space, scatter_position, points):
    scatter_x, scatter_y = scatter_position
    scatter_width, scatter_height = scatter_space

    # Reduce the plot area to 90% for better fitting
    margin_factor = 0.7
    reduced_width = int(scatter_width * margin_factor)
    reduced_height = int(scatter_height * margin_factor)

    offset_x = (scatter_width - reduced_width) // 2
    offset_y = (scatter_height - reduced_height) // 2

    # Normalize points to fit the reduced area
    normalized_points = points - np.min(points, axis=0)
    scale_x = reduced_width / np.ptp(normalized_points[:, 0])
    scale_y = reduced_height / np.ptp(normalized_points[:, 1])

    normalized_points *= [scale_x, scale_y]

    # Clear the scatter area
    canvas[scatter_y:scatter_y + scatter_height, scatter_x:scatter_x + scatter_width] = 255

    for x, y in normalized_points.astype(int):
        cv2.circle(canvas, (scatter_x + offset_x + x, scatter_y + offset_y + y), 3, (0, 0, 255), -1)

    return canvas


def plot_line_chart(canvas, line_chart_space, line_chart_position, data, color_buffer=None, color=(0, 0, 255)):
    """
    Plots the time-series data in the line chart area with labeled horizontal lines.
    """
    chart_x, chart_y = line_chart_position
    chart_width, chart_height = line_chart_space

    # Clear the line chart area
    canvas[chart_y:chart_y + chart_height, chart_x:chart_x + chart_width] = 255

    # Add labeled horizontal lines at specific percentages
    horizontal_positions = [0.1, 0.3, 0.5, 0.8, 1.0]  # Percentages
    for pos in horizontal_positions:
        y = chart_y + int(chart_height * pos)
        value = round(pos * np.ptp(data) + np.min(data), 2) if data else 0  # Corresponding value

        # Draw the line
        cv2.line(canvas, (chart_x, y), (chart_x + chart_width, y), (255, 0, 0), 1)

        # Add the value next to the line
        cv2.putText(canvas, f"{value:.2f}", (chart_x - 50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Normalize the data to fit the chart
    if len(data) > 1:
        # Filtra valores None/Não inicializados
        valid_data = [d for d in data if d is not None]
        
        # Normalização apenas com dados válidos
        if valid_data:
            normalized_data = (np.array(valid_data) - np.min(valid_data)) / (np.ptp(valid_data) or 1) * chart_height
            normalized_data = chart_height - normalized_data

            # Ajuste do cálculo de posição para rolagem contínua
            num_points = len(valid_data)
            step_x = chart_width / PLOT_SIZE
            
            for i in range(1, num_points):
                x1 = chart_x + int((PLOT_SIZE - num_points + i - 1) * step_x)
                y1 = chart_y + int(normalized_data[i - 1])
                x2 = chart_x + int((PLOT_SIZE - num_points + i) * step_x)
                y2 = chart_y + int(normalized_data[i])

                # Usa color_buffer se fornecido, senão cor padrão
                if color_buffer and len(color_buffer) >= i:
                    segment_color = color_buffer[i - 1]
                else:
                    segment_color = color

                cv2.line(canvas, (x1, y1), (x2, y2), segment_color, 2)

    return canvas

def calculate_measures_distances(landmarks):
    def euclidean_distance(p1, p2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    def mean_point(points):
        return np.mean(points, axis=0)
    
    # Internal mouth 1 vertical 63 and 67
    m1 = euclidean_distance(landmarks[63 - 1], landmarks[67 - 1])

    # External mouth horizontal 49 and 55
    m3 = euclidean_distance(landmarks[49 - 1], landmarks[55 - 1])

    # Upper lips 1 vertical 52 and 63
    m4 = euclidean_distance(landmarks[52-1], landmarks[63-1])

    # Lower lips 1 vertical 67 and 58
    m5 = euclidean_distance(landmarks[67-1],  landmarks[58-1])

    # Upper lips horizontal 62 and 64
    m6 = euclidean_distance(landmarks[62-1],  landmarks[64-1])

    # Lower lips horizontal 59 and 57
    m7 = euclidean_distance(landmarks[59-1],  landmarks[57-1])

    # Upper lip and nose 34 and 52
    m8 = euclidean_distance(landmarks[34-1],  landmarks[52-1])

    # Lower lip and nose 34 and 58
    m9 = euclidean_distance(landmarks[34-1], landmarks[58-1])

    # Internal lower lip and nose 34 and 67
    m10 = euclidean_distance(landmarks[34-1], landmarks[67-1])

    # Mouth: Draw average between 63 and 67 and measure vertical parallel with 34
    mean_m11 = mean_point([landmarks[63-1], landmarks[67-1]])
    m11 = euclidean_distance(mean_m11, landmarks[34-1])

    # Mouth-Nose: Measure vertical parallel between 34 and 49
    m12 = euclidean_distance(landmarks[34-1], landmarks[49-1])

    # Mouth-Nose: Measure right vertical parallel between 34 and 55
    m13 = euclidean_distance(landmarks[34-1], landmarks[55-1])

    # Mouth: Draw average between 49 and 55 and measure vertical parallel with 52
    mean_m14 = mean_point([landmarks[49-1], landmarks[55-1]])
    m14 = euclidean_distance(mean_m14, landmarks[52-1])

    # Mouth: Draw average between 49 and 55 and measure vertical parallel with 58
    mean_m15 = mean_point([landmarks[49-1], landmarks[55-1]])
    m15 = euclidean_distance(mean_m15, landmarks[58-1])

    # Mouth: Draw average between 49 and 55 and measure vertical parallel with 67
    mean_m16 = mean_point([landmarks[49-1], landmarks[55-1]])
    m16 = euclidean_distance(mean_m16, landmarks[68-1])

    # Mouth: Draw average between 49 and 55 and measure vertical parallel with 63
    mean_m17 = mean_point([landmarks[49-1], landmarks[55-1]])
    m17 = euclidean_distance(mean_m17, landmarks[64-1])

    # Left eye vertical 38 and 42
    e1 = euclidean_distance(landmarks[38-1], landmarks[42-1])

    # Right eye vertical 44 and 48
    e2 = euclidean_distance(landmarks[44-1], landmarks[48-1])

    # Eye: Vertical distance right eye 2 with average between points 45-44 and 47-48
    mean_e3_1 = mean_point([landmarks[45-1], landmarks[44-1]])
    mean_e3_2 = mean_point([landmarks[47-1], landmarks[48-1]])
    e3 = euclidean_distance(mean_e3_1, mean_e3_2)

    # Eyebrow-Nose: Measure left vertical parallel between 28 and 20
    b1 = euclidean_distance(landmarks[28-1], landmarks[20-1])

    # Eyebrow-Nose: Measure right vertical parallel between 28 and 25
    b2 = euclidean_distance(landmarks[28-1], landmarks[25-1]) 

    # Eyebrows horizontal 22 and 23
    b3 = euclidean_distance(landmarks[22-1], landmarks[23-1])

    # Return only the required measures
    return {"m1": m1, 'm3': m3, 'm4': m4, 'm5': m5, 'm6': m6, 'm7': m7, 'm8': m8, 'm9': m9,
             'm10': m10, 'm11': m11, 'm12': m12, 'm13': m13, 'm14': m14, 'm15': m15, 'm16': m16,
               'm17': m17, 'e1': e1, 'e2': e2, 'e3': e3, 'b1': b1, 'b2': b2, 'b3': b3}


def normalization(landmarks, shape_normal_img):
    """
    Normalizes landmarks for centering and scaling.
    """
    scale, norm_z_landmarks = vid_adj_fun.z_normalization(landmarks)
    norm_roll_landmarks = vid_adj_fun.roll_normalization(norm_z_landmarks)

    x_center = int(shape_normal_img[0] / 2)
    y_center = int(shape_normal_img[1] / 2)
    SCALED_CENTER_POINTS = vid_adj_fun.transform_scale_updown(norm_roll_landmarks, 1.5 * window_scale_size)
    SCALED_CENTER_POINTS = vid_adj_fun.move_to_center_position(SCALED_CENTER_POINTS, (34 - 1), x_center, y_center)
    
    return SCALED_CENTER_POINTS

