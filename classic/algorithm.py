import numpy as np
from scipy.signal import convolve2d

def update_board(board_config, p):
    if p.step_counter == 0:
        # Initialize by making some cells alive
        selected_cells = np.random.choice(p.x_cells * p.y_cells, 
                        size = p.start_smilies, replace = False)
        row_idx, col_idx = np.unravel_index(selected_cells, (p.x_cells, p.y_cells))
        img_positions = []
        for ii in range(len(row_idx)):
            board_config[row_idx[ii], col_idx[ii]] = 1
            img_positions.append([row_idx[ii]*p.img_width, col_idx[ii]*p.img_height])
        
    else:
        # Convolve using a kernel to determine how many neigbors for each cell are alive
        neighbor_sum = convolve2d(board_config, p.kernel, mode='same')

        # Keep a cell alive if it was currently alive and has 2 or 3 neighbors alive
        curr_alive = np.argwhere(np.abs(board_config.flatten() - 1) <= 0.01)
        neighbor2_alive = np.argwhere(np.abs(neighbor_sum.flatten() - 2) <= 0.01)
        neighbor3_alive = np.argwhere(np.abs(neighbor_sum.flatten() - 3) <= 0.01)
        neighbors_alive = np.union1d(neighbor2_alive, neighbor3_alive)
        remain_alive_idx = np.intersect1d(curr_alive, neighbors_alive)

        # Make the cell alive if it was currently dead but has exactly 3 neighbors alive
        curr_dead = np.argwhere(np.abs(board_config.flatten() - 0) <= 0.01)
        neighbor3_alive = np.argwhere(np.abs(neighbor_sum.flatten() - 3) <= 0.01)
        newly_alive_idx = np.intersect1d(curr_dead, neighbor3_alive)
        final_alive_idx = np.union1d(remain_alive_idx, newly_alive_idx)

        # Update board configuration with the final alive cells for this generation
        row_idx, col_idx = np.unravel_index(final_alive_idx, (p.x_cells, p.y_cells))
        board_config = np.zeros((p.x_cells, p.y_cells))
        img_positions = []
        for ii in range(len(row_idx)):
            board_config[row_idx[ii], col_idx[ii]] = 1
            img_positions.append([row_idx[ii]*p.img_width, col_idx[ii]*p.img_height])
    return board_config, img_positions
