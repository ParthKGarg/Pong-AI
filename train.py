import pongAI
import torch
import pong_train

AI1 = pongAI.PongAI()
# AI1.online_network.load_state_dict(torch.load("saved_models/RUN5_model_ep700.pth"))
# torch.save(AI1.online_network.state_dict(), f'best_ones/BEST_MODEL3.pth')
# AI1.update_target_network()
print("Initial Q value:", AI1.avg_q_value())

# RUN = "RUN4_"

def pure_train(RUN, episodes, epsl, decay):
    score1 = 0
    score2 = 0
    diff = 0
    AI1.epsilon = epsl
    for episode in range(episodes):
        winner, s1, s2 = pong_train.train_env(AI1, mode=1)
        if winner == 1:
            score1 += 1
        else:
            score2 += 1
        diff += s2 - s1

        print(f"Episode {episode+1} | AI score: {score1} | p2 score: {score2} | last game: {s1}-{s2} | avg point diff: {diff/(episode+1)}")
        
        AI1.epsilon = max(0.15, AI1.epsilon*decay)
        
        if (episode+1)%50 == 0:
            torch.save(AI1.online_network.state_dict(), f'saved_models/{RUN}model_ep{episode+1}.pth')
        if (episode+1)%20 == 0:
            print(AI1.avg_q_value())
        if episode+1 == 500:
            print("--- Mid-training test ---")
            for i in range(10):
                winner, s1, s2 = pong_train.train_env(AI1, mode=0)
                print(f"Test {i+1}: {s1}-{s2}")
            for param_group in AI1.optimizer.param_groups:
                param_group['lr'] = 0.00005

def pure_greed():
    for i in range(20):
        winner, s1, s2 = pong_train.train_env(AI1, mode=0)
        print(f"Test {i+1}: {s1}-{s2}")


# for visualization ---------------------
def visualize():
    import pygame
    import pong
    pygame.display.set_caption("PONG")
    pygame.init()
    for i in range(20):
        winner, s1, s2 = pong.train(AI1, mode=0)
        print(f"Test {i+1}: {s1}-{s2}")

    pygame.quit()

pure_train("RUN6_", 1000, 0.5, 0.995)
pure_greed()
# visualize()
