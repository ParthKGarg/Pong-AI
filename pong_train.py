import math
import random
from player import Player
from ball import Ball
from pongAI import PongAI

windowWidth = 800
windowHeight = 600
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 144
startPos1 = (50, 250)
startPos2 = (725, 250)
paddleShape = (25, 100)
paddleSpeed = 5
ballPos = (400, 300)
ballRadius = 10
ballSpeed = 5
MAX_SPEED = 8
MAX_SCORE = 10
SPEED_MULT = 1.02
UP = 0
STAY = 1
DOWN = 2

class Pong:
    def __init__(self) -> None:
        self.p1 = Player(startPos1, paddleShape, WHITE, 0)
        self.p2 = Player(startPos2, paddleShape, WHITE, 0)
        self.ball = Ball(ballPos, WHITE, ballRadius, x_vel=ballSpeed, y_vel=0)

    def resetGame(self):
        self.p1.y = startPos1[1]
        self.p2.y = startPos2[1]

        # reset the ball
        self.ball.x = ballPos[0]
        self.ball.y = ballPos[1]
        self.ball.dx = ballSpeed
        self.ball.dy = 0

    def checkCollisions(self):
        if (
            self.ball.x + self.ball.rad + self.ball.dx > self.p2.x
            and self.ball.y > self.p2.y
            and self.ball.y < self.p2.y + self.p2.shape[1]
        ):
            mid = self.p2.y + self.p2.shape[1] / 2
            relative_hit_spot = (self.ball.y - mid) / (self.p2.shape[1] / 2)
            deflectionAngle = relative_hit_spot * 75
            speed = (self.ball.dx**2 + self.ball.dy**2) ** 0.5
            speed = min(speed * SPEED_MULT, MAX_SPEED)
            dir = 1 if self.ball.dx > 0 else -1
            self.ball.dx = dir * speed * math.cos(math.radians(deflectionAngle))
            self.ball.dy = speed * math.sin(math.radians(deflectionAngle))
            self.ball.x = self.p2.x - self.ball.rad
            self.ball.dx *= -1

        if (
            self.ball.x - self.ball.rad + self.ball.dx < self.p1.x + self.p1.shape[0]
            and self.ball.y > self.p1.y
            and self.ball.y < self.p1.y + self.p1.shape[1]
        ):
            mid = self.p1.y + self.p1.shape[1] // 2
            relative_hit_spot = (self.ball.y - mid) / (self.p1.shape[1] / 2)
            deflectionAngle = relative_hit_spot * 75
            speed = (self.ball.dx**2 + self.ball.dy**2) ** 0.5
            speed = min(speed * 1.05, MAX_SPEED)
            dir = 1 if self.ball.dx > 0 else -1
            self.ball.dx = dir * speed * math.cos(math.radians(deflectionAngle))
            self.ball.dy = speed * math.sin(math.radians(deflectionAngle))
            self.ball.x = self.p1.x + self.p1.shape[0] + self.ball.rad
            self.ball.dx *= -1


def follower(ball, player):
    if random.random() < 0.5:
        if ball.y < player.y + paddleShape[1]//2 :
            return UP
        else:
            return DOWN
    return random.choice([UP, DOWN, STAY])


def train_env(ai:PongAI, mode = 1):
    global FPS
    game = Pong()
    epsilon = False
    step = 0
    ballBehind = False
    if mode:
        epsilon = True

    run = True
    while run:
        # get state before action
        state = (game.ball.x, game.ball.y, game.ball.dx, game.ball.dy, game.p1.y, game.p2.y)
        action = None
        new_state = None
        reward = None

        # get moves from AIs
        move1 = ai.choose_action(state, epsilon)
        action = move1
        move2 = follower(game.ball, game.p2)

        if move1 == UP:
            if game.p1.y > paddleSpeed:
                game.p1.y -= paddleSpeed
            else:
                game.p1.y = 0
        if move1 == DOWN and game.p1.y < windowHeight - paddleShape[1] + paddleSpeed:
            game.p1.y += paddleSpeed

        if move2 == UP:
            if game.p2.y > paddleSpeed:
                game.p2.y -= paddleSpeed
            else:
                game.p2.y = 1
        if move2 == DOWN and game.p2.y < windowHeight - paddleShape[1] + paddleSpeed:
            game.p2.y += paddleSpeed

        if game.ball.y - game.ball.rad + game.ball.dy < 0:
            game.ball.y = game.ball.rad
            game.ball.dy *= -1
        elif game.ball.y + game.ball.rad + game.ball.dy > 601:
            game.ball.y = windowHeight - game.ball.rad
            game.ball.dy *= -1
        else:
            game.ball.x += game.ball.dx
            game.ball.y += game.ball.dy

        if (
            game.ball.x > game.p1.x + game.p1.shape[0]
            and game.ball.x < game.p2.x
            and (game.ball.x > 700 or game.ball.x < 100)
        ):
            game.checkCollisions()
        
        new_state = (game.ball.x, game.ball.y, game.ball.dx, game.ball.dy, game.p1.y, game.p2.y)

        paddle_center = game.p1.y + paddleShape[1]//2
        dist = abs(paddle_center - game.ball.y)
        denseReward = -dist/windowHeight

        if game.ball.x > windowWidth + game.ball.rad:
            game.p1.score += 1
            reward = 1
            ballBehind = False
            game.resetGame()
        elif game.ball.x < -game.ball.rad:
            reward = 0
            game.p2.score += 1
            ballBehind = False
            game.resetGame()
        elif game.ball.x <game.p1.x and not ballBehind:
            reward = -1
            ballBehind = True
        else:
            reward = 0
        
        reward += 0.1 * denseReward

        if state is None or action is None or new_state is None or reward is None:
            raise ValueError("All values must be valid.")
        ai.remember(state, action, new_state, reward)
        if step%4 == 0:
            ai.train()
        if step%50 == 0:
            step = 0
            ai.update_target_network()
        step += 1
        
        if game.p1.score >= MAX_SCORE or game.p2.score >= MAX_SCORE:
            print(f"Game ended. Player {1 if game.p1.score>= MAX_SCORE else 2} Won.")
            run = False


    return (1, game.p1.score, game.p2.score) if game.p1.score >= MAX_SCORE else (2, game.p1.score, game.p2.score)
