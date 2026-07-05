import pygame
import math
import time
from player import Player
from ball import Ball

pygame.display.set_caption("PONG")
pygame.init()

windowWidth = 800
windowHeight = 600
WHITE = (255,255,255)
GRAY = (50,50,50)
BLACK = (0,0,0)
RED = (255, 0, 0)
font = pygame.font.SysFont('comicsans', 25, True)
window=pygame.display.set_mode((windowWidth,windowHeight))

startPos1 = (50, 250)
startPos2 = (725, 250)
paddleShape = (25, 100)
paddleSpeed = 4
ballPos = (400,300)
ballRadius = 10
ballSpeed = 5
FPS = 144
MAX_SPEED = 8
SPEED_MULT = 1.02

p1 = Player(startPos1, paddleShape, WHITE, 0)
p2 = Player(startPos2, paddleShape, WHITE, 0)
ball = Ball(ballPos, WHITE, ballRadius, x_vel=ballSpeed, y_vel=0)

def redrawGameWindow():
    window.fill(BLACK)
    pygame.draw.line(window, WHITE, (400,0), (400, 600), width=1)
    # pygame.draw.rect(window, RED, (0, 0, 10, 10))
    # pygame.draw.line(window, WHITE, (700,0), (700, 600), width=1)
    # pygame.draw.line(window, WHITE, (100,0), (100, 600), width=1)
    
    text1 = font.render(f"{p1.score}", 1, WHITE)
    text2 = font.render(f"{p2.score}", 1, WHITE)
    text1Cords = (350, 0)
    text2Cords = (450-text2.get_width(), 0)

    window.blit(text1, text1Cords)
    window.blit(text2, text2Cords)

    p1.draw(window)
    p2.draw(window)
    ball.draw(window)
    

    pygame.display.update()

def resetGame():
    p1.y = startPos1[1]
    p2.y = startPos2[1]

    # reset the ball
    ball.x = ballPos[0]
    ball.y = ballPos[1]
    ball.dx = ballSpeed
    ball.dy = 0


def checkCollisions():
    if ball.x + ball.rad + ball.dx > p2.x and ball.y > p2.y and ball.y < p2.y + p2.shape[1]:
        mid = p2.y + p2.shape[1]/2
        relative_hit_spot = (ball.y - mid) / (p2.shape[1] /2)
        deflectionAngle = relative_hit_spot * 75
        speed = (ball.dx**2 + ball.dy**2)**0.5
        speed = min(speed * SPEED_MULT, MAX_SPEED)
        dir = 1 if ball.dx >0 else -1
        ball.dx = dir * speed * math.cos(math.radians(deflectionAngle))
        ball.dy = speed * math.sin(math.radians(deflectionAngle))
        ball.x = p2.x - ball.rad
        ball.dx *= -1

    if ball.x - ball.rad + ball.dx < p1.x + p1.shape[0] and ball.y > p1.y and ball.y < p1.y + p1.shape[1]:
        mid = p1.y + p1.shape[1]//2
        relative_hit_spot = (ball.y - mid) / (p1.shape[1] /2)
        deflectionAngle = relative_hit_spot * 75
        speed = (ball.dx**2 + ball.dy**2)**0.5
        speed = min(speed * 1.05, MAX_SPEED)
        dir = 1 if ball.dx >0 else -1
        ball.dx = dir * speed * math.cos(math.radians(deflectionAngle))
        ball.dy = speed * math.sin(math.radians(deflectionAngle))
        ball.x = p1.x + p1.shape[0] + ball.rad
        ball.dx *= -1


clock = pygame.time.Clock()
max_fps = 0
run= True
while run:
    clock.tick(FPS)
    # time.sleep(0.002)
    window.fill(BLACK)

    for events in pygame.event.get():
        if events.type == pygame.QUIT:
            run = False
        if events.type == pygame.KEYDOWN:
            if events.key == pygame.K_q:
                run = False
            if events.key == pygame.K_r:
                resetGame()
            if events.key == pygame.K_SPACE:
                FPS = (10 if FPS>10 else 144)

    
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        if p1.y>paddleSpeed:
            p1.y -= paddleSpeed
        else:
            p1.y = 1
    if keys[pygame.K_s] and p1.y < windowHeight - paddleShape[1] + paddleSpeed:
        p1.y += paddleSpeed

    if keys[pygame.K_UP]:
        if p2.y>paddleSpeed:
            p2.y -= paddleSpeed
        else:
            p2.y = 1
    if keys[pygame.K_DOWN] and p2.y < windowHeight - paddleShape[1] + paddleSpeed:
        p2.y += paddleSpeed
    
    if ball.y - ball.rad + ball.dy < 0:
        ball.y = ball.rad
        ball.dy *= -1
    elif ball.y + ball.rad + ball.dy > 601:
        ball.y = windowHeight - ball.rad
        ball.dy *= -1
    else:
        ball.x += ball.dx
        ball.y += ball.dy
    # ball.x += ball.dx
    # ball.y += ball.dy

    if ball.x>windowWidth+ball.rad:
        p1.score += 1
        resetGame()
    elif ball.x < -ball.rad:
        p2.score += 1
        resetGame()
    
    if ball.x > p1.x + p1.shape[0] and ball.x < p2.x and (ball.x > 700 or ball.x < 100):
        checkCollisions()
    
    if p1.score>=10 or p2.score >= 10:
        print(f"Game ended. Player {1 if p1.score>= 10 else 2} Won.")
        run = False
    

    redrawGameWindow()

pygame.quit()