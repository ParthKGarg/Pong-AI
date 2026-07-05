
class Player():
    def __init__(self, pos, shape, color, score) -> None:
        self.pos = pos
        self.shape = shape
        self.color = color
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.score = score
        self.box = (self.pos[0], self.y, self.shape[0], self.shape[1])
    
    def draw(self, window):
        import pygame
        self.pos = (self.x, self.y)
        self.box = (self.pos[0], self.y, self.shape[0], self.shape[1])
        pygame.draw.rect(window, self.color, self.box)