
class Ball():
    def __init__(self, pos, color, radius, x_vel, y_vel) -> None:
        self.cord = pos
        self.color = color
        self.rad = radius
        self.x = self.cord[0]
        self.y = self.cord[1]
        self.box = (self.x-self.rad, self.y - self.rad, self.rad*2, self.rad*2)
        self.dx = x_vel
        self.dy = y_vel
    
    def draw(self, window):
        import pygame
        self.cord = (self.x, self.y)
        self.box = (self.x-self.rad, self.y - self.rad, self.rad*2, self.rad*2)
        # pygame.draw.rect(window, (255, 0, 0), self.box, width=1)
        pygame.draw.circle(window, self.color, self.cord, self.rad)
    
