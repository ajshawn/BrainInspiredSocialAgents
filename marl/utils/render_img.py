import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np

def show_rgb_image(rgb_array, step, agent_id, output_dir, objects_in_view):
    """
    Displays an RGB image represented by a NumPy array of shape (height, width, 3).

    Parameters:
    -----------
    rgb_array : np.ndarray
        A NumPy array of shape (H, W, 3). Each element can be an integer in [0, 255]
        (for typical 8-bit images) or a float in [0, 1] (for normalized images).
    """
    # Check if the input is indeed an RGB image
    if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
        raise ValueError("Input array must be of shape (height, width, 3).")

    plt.figure(figsize=(6, 4))
    plt.imshow(rgb_array)
    plt.axis('off')  # Turn off axis ticks/labels
    plt.title(objects_in_view)
    # save the image
    plt.savefig(output_dir + '/step_' + str(step) + '_agent_' + str(agent_id) + '.png')
    plt.close()
    
if __name__ == "__main__":
    with open('observations.jsonl') as f:
        lines = f.readlines()
    
    lines = lines[10:20]
    
    for idx, line in enumerate(lines):
        obs = json.loads(line)
        for agent_id, agent_obs in obs.items():
            rgb_array = np.array(agent_obs['RGB'], dtype=np.uint8)
            show_rgb_image(rgb_array, idx, agent_id[-1], 'images', agent_obs['OBJECTS_IN_VIEW'])
