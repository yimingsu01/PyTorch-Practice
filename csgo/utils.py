import math
import cv2
import numpy as np

def graphpos(poses, orig_std, orig_mean, debug, output_filename, layout_path):
    # Getting 5 players' x and y coordinates out
    player_pos = np.array([[0, 0]])
    for i in range(0, 10, 2):
        temp = np.array([])
        temp = np.append(temp, poses[i])
        temp = np.append(temp, poses[i + 1])
        temp = np.expand_dims(temp, axis=0)

        player_pos = np.row_stack((player_pos, temp))

    player_pos = np.delete(player_pos, 0, axis=0)

    # Reverse the regularization process to get in-game coordinates
    player_pos = player_pos * orig_std + orig_mean

    # new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    # I found these numbers out manually, by noclipping around the map.
    # These in-game coordinates correspond to the four corners of the map layout
    old_x_max = 2760
    old_x_min = -1872

    old_y_max = 3675
    old_y_min = -957

    # now converting the in-game coordinates to the layout image's coordinates
    # Assuming the image is square
    layout_image = cv2.imread(layout_path)

    new_x_max = layout_image.shape[0]
    new_y_max = new_x_max

    new_x_min = 0
    new_y_min = new_y_max

    player_pos[:, 0] = ((player_pos[:, 0] - old_x_min) / (old_x_max - old_x_min)) * (1024)
    player_pos[:, 1] = ((player_pos[:, 1] - old_y_min) / (old_y_max - old_y_min)) * (1024)

    if debug == 1:
        print(player_pos)
        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow("res", layout_image)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()

    # Graphing the positions
    for pos in player_pos:
        cv2.circle(img=layout_image, center=(int(pos[0]), int(1024-pos[1])), radius=5, color=(0,255,255), thickness=-1)

    else:
        cv2.imwrite(output_filename, layout_image)