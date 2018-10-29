# Modifications made for A3C:
# reward was set to 1 -1 and 0 while if first was 10, -10 and 0
# the playNextMoveA3C returns [s1,r,d,i]
# added self.gameTime to pongGame
# added self.currScores
# added logic to terminate the game when currScores eather agent or opp 
# reaches the termnial amount of score which is set at 20 at the moment of writing

#
# Simple Pong Game based upon PyGame
# My Pong Game, simplify Pong to play with Direct Ball, Pass Paddle and Ball as direct Features into DQN
#
# Yellow Left Hand Paddle is the DQN Agent Game Play
# A Red Ball return meant the Player missed the last Ball
# A Blue Ball return meant a successful return
#
#  Based upon Siraj Raval's inspiring Machine Learning vidoes
#  This is based upon Sirajs  Pong Game code
#  https://github.com/llSourcell/pong_neural_network_live
#
# Note needs imporved frame rate de sensitivition so as to ensure DQN perfomance across all computer types
# Currently Delta Time RATE fixed on each componet update to 7.5 !  => May ned to adjust increase/reduce depending upon perfomance
# ============================================================================================
import pygame
import random
import numpy as np
import math
#import SelfplayExperiment
#import CSVhandler as csvH
from Env.GlobalConstantsA3C import gc # has high level constants

#size of our window
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 380

#size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
#distance from the edge of the window
PADDLE_BUFFER = 15

#size of our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

# Paddle speed is in GameConstants
#speeds of our ball
BALL_X_SPEED = 3
BALL_Y_SPEED = 3
DFT = 7.5

#RGB colors for our paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)

# Normalise GameState
def captureNormalisedStateA3C(gState):
    gState[0] = gState[0]/gc.NORMALISATION_FACTOR    # Normalised PlayerYPos
    gState[1] = gState[1]/gc.NORMALISATION_FACTOR       # Normalised OpponentYpos
    gState[2] = gState[2]/gc.NORMALISATION_FACTOR    # Normalised BallXPos
    gState[3] = gState[3]/gc.NORMALISATION_FACTOR    # Normalised BallYPos
    gState[4] = gState[4]/1.0    # Normalised BallXDirection
    gState[5] = gState[5]/1.0    # Normalised BallYDirection

    return gState

class GameStats:
    def __init__(self):
        self.hitsPerScore = np.array([])
        self.hits = 0
        self.scoreCurrentGame = np.array([0, 0])   # first is score AI, second is scoreOpponent
        self.gameTimes = np.array([])
        self.totalScoreNewAI = 0
        self.totalScoreOpponents = 0
        self.speedY = np.array([])
        self.relativeHitPosition = np.array([])

    def updateScoreNewAI(self):
        self.totalScoreNewAI += 1
        self.scoreCurrentGame[0] += 1
        self.hitsPerScore = np.append(self.hitsPerScore,self.hits)
        self.hits = 0

    def updateBallHitStats(self, speedY, hitPosition):
        self.speedY = np.append(self.speedY, abs(speedY))
        self.relativeHitPosition = np.append(self.relativeHitPosition, abs(hitPosition))
        self.hits += 1

    def updateScoreOpponents(self):
        self.totalScoreOpponents += 1
        self.scoreCurrentGame[1] += 1
        np.append(self.hitsPerScore,self.hits)
        self.hits = 0

    def addHitsPerScore(self, hits):
        self.hitsPerScore = np.append(self.hitsPerScore,hits)

    def addGameTime(self, gameTime):
        self.gameTimes = np.append(self.gameTimes,gameTime)

    def getScore(self):
        return [self.scoreCurrentGame[0],self.scoreCurrentGame[1]] # a bit cumbersome, but easy for compatibility with old code

    def resetForNewGame(self):
        self.scoreCurrentGame = np.array([0, 0])

    def computeAverages(self):
        if (len(self.hitsPerScore) == 0 or len(self.speedY) == 0 or len(self.relativeHitPosition) == 0):
            return [0,0,0,0]

        averageHitsPerScore = np.sum(self.hitsPerScore) / len(self.hitsPerScore)
        averageGameTime = np.sum(self.gameTimes) / len(self.gameTimes)
        averageSpeedY = np.sum(self.speedY) / len(self.speedY)
        averageHitPosition = np.sum(self.relativeHitPosition) / len(self.relativeHitPosition)

        return [averageHitsPerScore,averageGameTime,averageSpeedY,averageHitPosition]

    def resetVariables(self):
        self.__init__()

    def saveStatsIteration(self, filepath):
        stats = self.computeAverages() # compute the averages from the collected data
        csvH.saveListToCSV(filepath, stats + [self.totalScoreNewAI, self.totalScoreOpponents])
        self.resetVariables() # reset the variables to zero
        # TODO remove this
        #with open(filepath,'ab') as f:
            #np.savetxt(f, [stats + [self.totalScoreNewAI, self.totalScoreOpponents]], delimiter=',', fmt='%f')

gameStats = GameStats()

# ===============================================================
#Paddle 1 is our learning agent/us
#paddle 2 is the oponent  AI



#update the ball, using the paddle posistions the balls positions and the balls directions
def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection,dft,BallColour,gameTime):
    dft =7.5
    #update the x and y position
    ballXPos = ballXPos + ballXDirection * BALL_X_SPEED*dft
    ballYPos = ballYPos + ballYDirection * BALL_Y_SPEED*dft
    reward = 0
    NewBallColor = BallColour;
    "this is the player on the left"
    #checks for a collision, if the ball hits the Gamer Player side, our Learning agent
    if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle1YPos and ballYPos - BALL_HEIGHT <= paddle1YPos + PADDLE_HEIGHT and ballXDirection == -1):
        #switches directions
        ballXDirection = 1
        #change the ballYDirection
        ballYDirection = computeBallDirection(ballXDirection, ballYDirection, paddle1YPos-ballYPos+PADDLE_HEIGHT/2)
       
        #  Player returned the Ball 
        reward = 0      # COMPETE
        #score = 10.0   # COOPERATE
        #update stats of hitting the ball
        gameStats.updateBallHitStats(ballYDirection, paddle1YPos-ballYPos+PADDLE_HEIGHT/2)
        NewBallColor = BLUE
    # The player MISSED the ball:
    elif (ballXPos <= 0):
        reward = -1  # COMPETE
        #score = 0      # COOPERATE
        gameStats.updateScoreOpponents() # increase score old AI for stats
        #reset ball position
        [ballXPos,ballYPos,ballXDirection,ballYDirection] = resetBallPosition(ballXDirection)
        NewBallColor = RED
        return [reward, ballXPos, ballYPos, ballXDirection, ballYDirection,NewBallColor]

    "this is the player on the right"
    #check if hits the AI Player
    if (ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and ballYPos + BALL_HEIGHT >= paddle2YPos and ballYPos - BALL_HEIGHT <= paddle2YPos + PADDLE_HEIGHT):
        #switch directions
        ballXDirection = -1
        ballYDirection = computeBallDirection(ballXDirection, ballYDirection, paddle2YPos-ballYPos+PADDLE_HEIGHT/2)

        NewBallColor = WHITE
    # The Player SCORED:
    elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
        #reset ball position
        [ballXPos,ballYPos,ballXDirection,ballYDirection] = resetBallPosition(ballXDirection)
        NewBallColor = WHITE
        
        #score = 0 # COOPERATE
        reward = 1 # COMPETE
        #score = 10.0 * (gameTime / gc.TRAIN_TIME)
        gameStats.updateScoreNewAI() # increase score new AI for stats
        return [reward, ballXPos, ballYPos, ballXDirection, ballYDirection,NewBallColor]

    #if it hits the top move down
    if (ballYPos <= 0):
        ballYPos = 0;
        ballYDirection = -ballYDirection;
    #if it hits the bottom, move up
    elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
        ballYDirection = -ballYDirection;
    return [reward, ballXPos, ballYPos, ballXDirection, ballYDirection,NewBallColor]



#=========================================================
# computes the direction of the ball when the ball hits one of the paddles
CIRCLE_RADIUS = PADDLE_HEIGHT * 1.2
MAX_Y_DIRECTION = 1.7
def computeBallDirection(ballXDirection, ballYDirection, paddleTouchPosition):
    #paddleTouchPosition: the position of the paddle where the ball was hit
    ballRad = math.atan(ballYDirection/math.fabs(ballXDirection))
    deflectionRad = math.asin(paddleTouchPosition/CIRCLE_RADIUS)

    ballNewRad = deflectionRad - math.radians(90) + (math.radians(90) - ballRad)
    #print (math.degrees(ballRad), paddleTouchPosition, math.degrees(-1*math.tan(ballNewRad)))
    YDirection = -1 * math.tan(ballNewRad)

    # limit the maximum Y speed for the ball
    if (YDirection > MAX_Y_DIRECTION):
        YDirection = MAX_Y_DIRECTION
    elif (YDirection < -1*MAX_Y_DIRECTION):
        YDirection = -1 * MAX_Y_DIRECTION

    return YDirection

def resetBallPosition(ballXDirection):
    # initialize ball position to be in the middle of the screen
    ballY = WINDOW_HEIGHT / 2
    ballX = WINDOW_WIDTH / 2 + ballXDirection * WINDOW_WIDTH/4

    # initialize ball direction between -1 and 1
    ballYDir = random.uniform(-0.5, 0.5)
    ballXDir = -1 * ballXDirection

    return [ballX,ballY,ballXDir,ballYDir]

# ========================================================

#def getBotAction(paddleYPos, ballYPos):
#    #move up if ball is higher thn Openient Paddle
#    if (paddleYPos + PADDLE_HEIGHT/2 > ballYPos + BALL_HEIGHT/2):
#        return 1
#    #move down if ball lower than Paddle
#    if (paddleYPos + PADDLE_HEIGHT/2 < ballYPos + BALL_HEIGHT/2):
#        return 2
#    # not up or down, then do nothing = action = 0
#    return 0

# Close the pygame window
def quit_():
    pygame.quit()

# =========================================================================
#game class
class PongGame:
    def __init__(self,oppType, draw):
        self.render = draw
        #initialize our screen using width and height vars
        if (self.render): self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.oppType = oppType
        self.screenActive = False

#        if (self.render):
#            self.createWindow()

        #random number for initial direction of ball
        num = random.randint(0,9)

        # This sets the oppPadle speed to the right value based on what opponent it is.
        self.oppPaddleSpeed = self.setOppPaddleSpeed(oppType)
        #initialie positions of paddle
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        #and ball direction
        self.ballXDirection = 1
        self.ballYDirection = 1
        #starting point
        self.ballXPos = WINDOW_WIDTH/2 - BALL_WIDTH/2

        self.BallColor = WHITE
        self.gameTime = 0
        self.GTimeDisplay = 0
        self.GScore = 0.0
        self.GEpsilonDisplay = 1.0
        self.scoreCurrentGame = np.array([0, 0])   # first is score AI, second is scoreOpponent
        self.terminationScore = 20

        #randomly decide where the ball will move
        if(0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1
        #new random number
        num = random.randint(0,9)
        #where it will start, y part
        self.ballYPos = num*(WINDOW_HEIGHT - BALL_HEIGHT)/9

    # Initialise Game
    def InitialDisplay(self):
        #for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        #make the background black
        self.screen.fill(BLACK)
        #draw our paddles
        self.drawPaddle1(self.paddle1YPos)
        self.drawPaddle2(self.paddle2YPos)
        #draw our ball
        self.drawBall(self.ballXPos, self.ballYPos,WHITE)
        #
        #updates the window
        pygame.display.flip()

    #  Game Update Inlcuding Display
    def PlayNextMove(self, action, oppAction, gameTime):
        # This sets the render speed/ framerate! , Calculate DeltaFrameTime
        if gc.SHOW_NORMAL_SPEED:
            self.clock.tick(gc.FPS)
        # Use a constant to make slow motion viewing have the same game dynamics
        DeltaFrameTime = DFT
        score = 0
        #update our paddle
        self.paddle1YPos = self.updatePaddle(action, self.paddle1YPos, gc.PADDLE_SPEED , DeltaFrameTime)
        #update opponent paddle
        self.paddle2YPos = self.updatePaddle(oppAction, self.paddle2YPos, self.oppPaddleSpeed, DeltaFrameTime)
        #update our vars by updating ball position
        [score, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,self.BallColor] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,DeltaFrameTime,self.BallColor, gameTime)
        # Update Game Score Moving Average only if Hit or Miss Return
        if(score >0.5 or score < -0.5):
            self.GScore = 0.05*score + self.GScore*0.95
        if (self.render):
            # Reset screen
            pygame.event.pump()
            self.screen.fill(BLACK)
            # Draw elements
            self.drawPaddle1(self.paddle1YPos)
            self.drawPaddle2(self.paddle2YPos)
            self.drawBall(self.ballXPos, self.ballYPos,self.BallColor)
            #  Display Parameters
            ScoreDisplay = self.font.render("Score: "+ str("{0:.2f}".format(self.GScore)), True,(255,255,255))
            self.screen.blit(ScoreDisplay,(50.,20.))
            TimeDisplay = self.font.render("Time: "+ str(self.GTimeDisplay), True,(255,255,255))
            self.screen.blit(TimeDisplay,(50.,40.))
            EpsilonDisplay = self.font.render("Ep: "+ str("{0:.4f}".format(self.GEpsilonDisplay)), True,(255,255,255))
            self.screen.blit(EpsilonDisplay,(50.,60.))
            #update the Game Display
            pygame.display.flip()

        #return the score and the state
        return [score,self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection]
    
    def PlayNextMoveA3C(self, action, oppAction):
        # This sets the render speed/ framerate! , Calculate DeltaFrameTime
#        if self.render: #and gc.SHOW_NORMAL_SPEED:
#            self.clock.tick(gc.FPS)
        # Use a constant to make slow motion viewing have the same game dynamics
        DeltaFrameTime = DFT
        #update our paddle
        self.paddle1YPos = self.updatePaddle(action, self.paddle1YPos, gc.PADDLE_SPEED , DeltaFrameTime)
        #update opponent paddle
        self.paddle2YPos = self.updatePaddle(oppAction, self.paddle2YPos, self.oppPaddleSpeed, DeltaFrameTime)
        #update our vars by updating ball position
        [reward, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,self.BallColor] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,DeltaFrameTime,self.BallColor, self.gameTime)
        # if a hit or miss occurred:
        if(reward != 0):
            # update moving average
            self.GScore = 0.05*reward + self.GScore*0.95
             # increase the score of the player that scored
            if reward > 0.5:
                    self.scoreCurrentGame[0] += 1
            else:
                self.scoreCurrentGame[1] += 1
        # if the max score has been reached, terminate
        done = False
        if max(self.scoreCurrentGame) == self.terminationScore:
            done = True
        # Render the current state to the screen.
        #if (self.render):
            # If there is no screen at the moment, create one
            #if (not self.screenActive):
            #    self.createWindow()
         #   if self.screenActive:
        #        self.renderCurrentState()
        
        # Periodically check if the game window was closed, if so, quit.
        # But only check when the screen is not already closed
        if (self.gameTime % 200 == 0 and self.screenActive):        
            if self.hasQuit():
                # Close the pygame window
                self.quit_()
            
        newState = self.ReturnCurrentState()
        self.gameTime += 1
        return newState, reward, done, self.gameTime

    # Render the current state of affairs
    def renderCurrentState(self):
            if self.screenActive:
                pygame.event.pump()
                self.screen.fill(BLACK)
                # Draw elements
                self.drawPaddle1(self.paddle1YPos)
                self.drawPaddle2(self.paddle2YPos)
                self.drawBall(self.ballXPos, self.ballYPos,self.BallColor)
                #  Display Parameters
                MovingAvrScoreDisplay = self.font.render("Score: "+ str("{0:.2f}".format(self.GScore)), True,(255,255,255))
                self.screen.blit(MovingAvrScoreDisplay,(50.,20.))
                ScoresDisplay = self.font.render("Scores: "+ str("{0:.2f}".format(self.scoreCurrentGame[0])) + " " + str("{0:.2f}".format(self.scoreCurrentGame[1])), True,(255,255,255))
                self.screen.blit(ScoresDisplay,(50.,40.))
                TimeDisplay = self.font.render("Time: "+ str(self.GTimeDisplay), True,(255,255,255))
                self.screen.blit(TimeDisplay,(50.,40.))
                EpsilonDisplay = self.font.render("Ep: "+ str("{0:.4f}".format(self.GEpsilonDisplay)), True,(255,255,255))
                self.screen.blit(EpsilonDisplay,(50.,60.))
                #update the Game Display
                pygame.display.flip()

    
    # This takes the opponent type as input and return the right oppPaddleSpeed
    def setOppPaddleSpeed(self,oppType):
        return {
            -1 : gc.PADDLE_SPEED, # selfplay
            13 : gc.PADDLE_SPEED * 1.3,
            12 : gc.PADDLE_SPEED * 1.2,
            11 : gc.PADDLE_SPEED * 1.1,
            10 : gc.PADDLE_SPEED, 
            9  : gc.PADDLE_SPEED * 0.9,
            8  : gc.PADDLE_SPEED * 0.8,
            7  : gc.PADDLE_SPEED * 0.7,
            6  : gc.PADDLE_SPEED * 0.6,
            5  : gc.PADDLE_SPEED * 0.5,
            4  : gc.PADDLE_SPEED * 0.4,
            3  : gc.PADDLE_SPEED * 0.3,
            2  : gc.PADDLE_SPEED * 0.2,
            1  : gc.PADDLE_SPEED * 0.1,
            0  : gc.PADDLE_SPEED * 0
        }.get(oppType)  # This would return a default: }.get(oppType,default) 
    
    # This updates the paddle position based on an action
    def updatePaddle(self, action, paddleYPos, paddleSpeed, dft):
        # Assume Action is scalar:  0:stay, 1:Up, 2:Down
        #if move up
        if (action == 1):
            paddleYPos = paddleYPos - paddleSpeed *dft
        #if move down
        if (action == 2):
            paddleYPos = paddleYPos + paddleSpeed *dft
    
        #don't let it move off the screen
        if (paddleYPos < 0):
            paddleYPos = 0
        if (paddleYPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddleYPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return paddleYPos
    
    # Return the Curent Game State
    def ReturnCurrentState(self):
        # Simply return state
        s = [self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection]
        # Normalize the state:
        return captureNormalisedStateA3C(s)
    
    def returnCurrentStateOpp(self):
        s = [self.paddle2YPos, self.paddle1YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection]
        # Normalize the state:
        return captureNormalisedStateA3C(s)


    def UpdateGameDisplay(self,GTime,Epsilon):
        self.GTimeDisplay = GTime
        self.GEpsilonDisplay = Epsilon

    # When the user presses the x on the window this will return true
    def hasQuit(self):
        if (not self.render and self.screenActive): return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    
    def createWindow(self):
        # Initialise pygame
        pygame.init()
        pygame.display.set_caption('Pong DQN Experiment')
        #self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("calibri",20)
        self.InitialDisplay()
        self.screenActive = True
    
    def resetEpisode(self):

        #if (self.render):
            # If a window is not already created, create one
        #    if (not self.screenActive):
        #        self.createWindow()
            # Draw to the window
        #     self.InitialDisplay()
            #TODO miss deze regel: self.clock = pygame.time.Clock()

        #random number for initial direction of ball
        num = random.randint(0,9)

        #initialie positions of paddle
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        #and ball direction
        self.ballXDirection = 1
        self.ballYDirection = 1
        #starting point
        self.ballXPos = WINDOW_WIDTH/2 - BALL_WIDTH/2

        self.gameTime = 0
        self.GTimeDisplay = 0
        self.GScore = 0.0
        self.GEpsilonDisplay = 1.0
        self.scoreCurrentGame = np.array([0, 0])   # first is score AI, second is scoreOpponent

        #randomly decide where the ball will move
        if(0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1
        #new random number
        num = random.randint(0,9)
        #where it will start, y part
        self.ballYPos = num*(WINDOW_HEIGHT - BALL_HEIGHT)/9
        # Return the current state in a normalized way
        return self.ReturnCurrentState()

    def quit_(self):
        if self.screenActive:
            pygame.quit()
            self.screenActive = False
            
        #draw our ball
    def drawBall(self,ballXPos, ballYPos, BallCol):
        #small rectangle, create it
        ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, BallCol, ball)
    
    def drawPaddle1(self,paddle1YPos):
        #create it
        paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, YELLOW, paddle1)
    
    def drawPaddle2(self,paddle2YPos):
        #create it, opposite side
        paddle2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, WHITE, paddle2)

