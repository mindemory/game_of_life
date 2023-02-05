import os
import numpy as np
import pygame
import params as p
from algorithm import update_board


def main():
    # Load images
    smiley = pygame.image.load(os.path.join('img/smiley.png'))
    smiley = pygame.transform.scale(smiley, (p.img_width, p.img_height))
    skull = pygame.image.load(os.path.join('img/skull.png'))
    skull = pygame.transform.scale(skull, (p.img_width, p.img_height))

    # Initialize Pygame
    pygame.init()
    # Set caption to the game
    pygame.display.set_caption('Game of Life')
    # Create a screen
    screen = pygame.display.set_mode((p.screen_width, p.screen_height))
    screen.fill(p.screen_bg)
    
    pygame.display.flip()
    board_config = np.zeros((p.x_cells, p.y_cells))

    running = True
    while running:
        print(p.step_counter)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update board positions
        board_config, img_positions = update_board(board_config, p)
        screen.fill(p.screen_bg)

        for sm_pos in img_positions:
            screen.blit(skull, sm_pos)
            
        pygame.display.flip()
        pygame.time.wait(p.flip_time)
        p.step_counter += 1

if __name__ == '__main__':
    main()

