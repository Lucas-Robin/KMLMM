import pygame
from mnist import zip_trained_model
import numpy as np

size = 16
scale = 10
line_width = 2

background_colour = (255,255,255)
color = (0,0,0)
(width, height) = (size * scale, size * scale)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('draw a digit pls')

def new_pixels():
    pixels = []
    for i in range(size):
        pixels.append([])
        for j in range(size):
            pixels[-1].append(-1)
    return pixels

pixels = new_pixels()

def update():
    screen.fill(background_colour)
    for i in range(size):
        for j in range(size):
            alpha = (pixels[i][j] + 1) / 2
            alphacol = alpha * np.array(color) + (1 - alpha) * np.array(background_colour)
            screen.fill(alphacol, (i * scale, j * scale, scale, scale))
    pygame.display.flip()

def get_pixel(x, y):
    return (x // scale, y // scale)

def handle_click(x, y):
    for xx in range(x - line_width + 1, x + line_width):
        for yy in range(y - line_width + 1, y + line_width):
            if xx < size and xx >= 0 and yy < size and yy >= 0:
                pixels[xx][yy] = max(0, pixels[xx][yy])
    pixels[x][y] = 1
    update()

model = zip_trained_model()

def predict():
    pred = model.predict(np.array(pixels).T.reshape((1, 1, 16, 16)))
    print(pred)
    print(np.argmax(pred))

update()

running = True
while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
          running = False
      if event.type == pygame.KEYDOWN:
          key = event.unicode
          if key == 'n':
              pixels = new_pixels()
              update()
          elif key == 'p':
              predict()
    if pygame.mouse.get_pressed()[0]:
        (mx, my) = pygame.mouse.get_pos()
        (x, y) = get_pixel(mx, my)
        handle_click(x, y)
