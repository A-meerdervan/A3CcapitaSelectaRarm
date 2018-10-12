# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:57:05 2018

@author: Alex
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pathlib # Used to create folders
#from GlobalConstants import gc
#import CSVhandler as csvH

# TODOOOOO Change to 10 for most runs
numGames = 10 # This is set for all
numStatsPerGame = 6
numOfStatsTotal = 10
    

def createPlotsFromCSV():
#    # The parameters from the file that run it
#    cyclesList =    [30,30,30,30,30] + [50,50,50,50,50] #+ [50,50,50,50,50] # Normally gc.NUMBER_CYCLES
#    mainFolder = "../PongCode30janOchtendPCQupdate/29janNachtRunQintervalTest/"
#    topFolderName = "PongCode30janOchtendPCQupdate"
#    csvFileName = "testsLargeRunsss.csv"
#    imageSubFolderName = topFolderName

#    #The parameters from the file that run it
#    cyclesList =   [100,100,100,100] + [100,100,100]
#    mainFolder = "../pongCode29janAvondPC/largeRuns/"
#    topFolderName = "pongCode29janAvondPC"
#    csvFileName = "testsLargeRunsss.csv"
#    imageSubFolderName = topFolderName
    
#    #The parameters from the file that run it
    cyclesList =   [30,30,30,30,30]
    topFolderName = "1febEve_99gamma_100k_QintervalsBOT"
    mainFolder =  topFolderName + "/"
    csvFileName = "testsLargeRunsss_31jan.csv"
    imageSubFolderName = topFolderName
    
##    #The parameters from the file that run it
#    cyclesList =   [18]
#    topFolderName = "1febEve_99gamma_SelfPlay_AndBot_TT_500000"
#    mainFolder =  topFolderName + "/"
#    csvFileName = "testsLargeRunsss_31jan.csv"
#    imageSubFolderName = topFolderName
#    
##    #The parameters from the file that run it
#    cyclesList =   [30,30]
#    topFolderName = "1febLongTrainRun_100k_BOT_andSP"
#    mainFolder =  topFolderName + "/"
#    csvFileName = "testsLargeRunsss_31jan.csv"
#    imageSubFolderName = topFolderName
    
    for modelsf in range(len(cyclesList)):
        folderPath = mainFolder + "models" + str(modelsf) + "/"
        csvPath =  folderPath +  csvFileName   #"testsLargeRunsss.csv"    
        runSummary = "folder_" + topFolderName + "_model" + str(modelsf)
        numOfCycles = cyclesList[modelsf]
        # Create folder to store the images
        ResultingPlotImagesPath = "../ResultingPlotImages/" + imageSubFolderName + '/'
        pathlib.Path(ResultingPlotImagesPath).mkdir(parents=True, exist_ok=True) 
        
        with open(csvPath, 'r') as f:
          reader = csv.reader(f)
          allLines = list(reader)
        
        readIstart = 0# The readerIndex
        [statsSelf, readINextStart] = getDataOneOpponent(readIstart, allLines, numOfCycles)
        [statsBot13, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        [statsBot10, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        statsBot8 = [] #[statsBot8, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        [statsBot6, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        statsList = [statsSelf,statsBot13,statsBot10,statsBot8,statsBot6]
        opponents = ["itself at iteration" + str(numOfCycles), "bot type 13", "bot type 10", "bot type 8", "bot type 6"]
        
        createPlotsAllopp(statsList, opponents, ResultingPlotImagesPath, runSummary)
        
        # ===============================================
        
    #    print("===============================================\nPlots against " + opponents[0] + "\n" ) 
    #    createPlots(opponents[0],statsSelf)
    #    print("===============================================\nPlots against " + opponents[1] + "\n" ) 
    #    createPlots(opponents[1],statsBot13)
    #    print("===============================================\nPlots against " + opponents[2] + "\n" ) 
    #    createPlots(opponents[2],statsBot10)
    #    print("===============================================\nPlots against " + opponents[3] + "\n" ) 
    #    createPlots(opponents[3],statsBot8)
    #    print("===============================================\nPlots against " + opponents[4] + "\n" ) 
    #    createPlots(opponents[4],statsBot6)
    

    
    
#    #For the first plots of it 1,4,7.. against last version
#    # Create a lists within a list
#    stats = []
#    for s in range(numOfStatsTotal):
#        stats.append([])
#    for brainToTest in range(1,cyclesList[relevantI],3):
#        gameStats = np.zeros(10)
#        for game in range(numGames):
#            # Loop the stats of one csv row of a game
#            for index in range(numStatsPerGame):
#                gameStats[index] += float((allLines[readI])[index])
#            # Increase the line number with 1
#            readI += 1
#        # Take the average of all stats per game
#        for index in range(numStatsPerGame):
#            gameStats[index] = gameStats[index]/numGames
#        # Now store the relevant entries of a long row in the csv
#        gameStats[6] = float((allLines[readI])[3]) #BrainToTest
#        gameStats[7] = float((allLines[readI])[6]) #Wins
#        gameStats[8] = float((allLines[readI])[7]) #TotScore
#        gameStats[9] = float((allLines[readI])[8]) #TotOppScore
#        # A row was processed so increment the read index
#        readI += 1
#        # Add the stats to the larger list
#        for s in range(len(gameStats)):
#            stats[s].append(gameStats[s])
#        # A row was processed so increment the read index
#        readI += 1
#    
#    #print(stats[0])
#    #print(len(stats[0]))

def createPlotFromCSV_Qints():
    # The parameters from the file that run it
    cyclesList =    [30,30,30,30,30] + [50,50,50,50,50] #+ [50,50,50,50,50] # Normally gc.NUMBER_CYCLES
    mainFolder = "../PongCode30janOchtendPCQupdate/29janNachtRunQintervalTest/"
    topFolderName = "PongCode30janOchtendPCQupdate"
    csvFileName = "testsLargeRunsss.csv"
    imageSubFolderName = topFolderName + "_MT_plot"

#    #The parameters from the file that run it
#    cyclesList =   [100,100,100,100] + [100,100,100]
#    mainFolder = "../pongCode29janAvondPC/largeRuns/"
#    topFolderName = "pongCode29janAvondPC"
#    csvFileName = "testsLargeRunsss.csv"
#    imageSubFolderName = topFolderName
    
##    #The parameters from the file that run it
#    cyclesList =   [30,30,30,30,30]
#    topFolderName = "1febEve_99gamma_100k_QintervalsBOT"
#    mainFolder =  topFolderName + "/"
#    csvFileName = "testsLargeRunsss_31jan.csv"
#    imageSubFolderName = topFolderName + "_Qints_plot"

    
##    #The parameters from the file that run it
#    cyclesList =   [18]
#    topFolderName = "1febEve_99gamma_SelfPlay_AndBot_TT_500000"
#    mainFolder =  topFolderName + "/"
#    csvFileName = "testsLargeRunsss_31jan.csv"
#    imageSubFolderName = topFolderName
#    
##    #The parameters from the file that run it
#    cyclesList =   [30,30]
#    topFolderName = "1febLongTrainRun_100k_BOT_andSP"
#    mainFolder =  topFolderName + "/"
#    csvFileName = "testsLargeRunsss_31jan.csv"
#    imageSubFolderName = topFolderName
    
    allDataBot10 = []
    for modelsf in range(len(cyclesList)):
        folderPath = mainFolder + "models" + str(modelsf) + "/"
        csvPath =  folderPath +  csvFileName   #"testsLargeRunsss.csv"    
        #runSummary = "folder_" + topFolderName + "_model" + str(modelsf)
        numOfCycles = cyclesList[modelsf]
        # Create folder to store the images
        ResultingPlotImagesPath = "../ResultingPlotImages/" + imageSubFolderName + '/'
        pathlib.Path(ResultingPlotImagesPath).mkdir(parents=True, exist_ok=True) 
        
        with open(csvPath, 'r') as f:
          reader = csv.reader(f)
          allLines = list(reader)
        
        readIstart = 0# The readerIndex
        [statsSelf, readINextStart] = getDataOneOpponent(readIstart, allLines, numOfCycles)
        [statsBot13, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        [statsBot10, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        #statsBot8 = [] #[statsBot8, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        #[statsBot6, readINextStart] = getDataOneOpponent(readINextStart, allLines, numOfCycles)
        #statsList = [statsSelf,statsBot13,statsBot10,statsBot8,statsBot6]
        #opponents = ["itself at iteration" + str(numOfCycles), "bot type 13", "bot type 10", "bot type 8", "bot type 6"]
        allDataBot10.append(statsBot10)
        #createPlotsAllopp(statsList, opponents, ResultingPlotImagesPath, runSummary)
    #createQintPlot(allDataBot10,ResultingPlotImagesPath)
    create_MT_Plot(allDataBot10,ResultingPlotImagesPath)
    
    
        
        # ===============================================
        
    #    print("===============================================\nPlots against " + opponents[0] + "\n" ) 
    #    createPlots(opponents[0],statsSelf)
    #    print("===============================================\nPlots against " + opponents[1] + "\n" ) 
    #    createPlots(opponents[1],statsBot13)
    #    print("===============================================\nPlots against " + opponents[2] + "\n" ) 
    #    createPlots(opponents[2],statsBot10)
    #    print("===============================================\nPlots against " + opponents[3] + "\n" ) 
    #    createPlots(opponents[3],statsBot8)
    #    print("===============================================\nPlots against " + opponents[4] + "\n" ) 
    #    createPlots(opponents[4],statsBot6)
    
def create_MT_Plot(allDataBot10,ResultingPlotImagesPath):
    # part of the image name:
    runSummary = "MT_plot_20k"
    # Plot the score difference
    x = allDataBot10[5][6];    
    y0 = np.array(allDataBot10[5][8]) - np.array(allDataBot10[5][9])
    y150 = np.array(allDataBot10[6][8]) - np.array(allDataBot10[6][9])
    y200 = np.array(allDataBot10[7][8]) - np.array(allDataBot10[7][9])
    y300 = np.array(allDataBot10[8][8]) - np.array(allDataBot10[8][9])
    y400 = np.array(allDataBot10[9][8]) - np.array(allDataBot10[9][9])
    # A line to emphasize the zero crossing
    line0x = np.arange(0., x[-1], 1.)
    line0y = line0x * 0
    fig = plt.figure()
    # Legenda stuff
    descriptions = ["MT = 1","MT = 2","MT = 3","MT = 4","MT = 5"]
    #descriptions = ["Q interval = 100","Q interval = 150","Q interval = 200","Q interval = 400","Q interval = 1000"]
    # Plotting
    plt.plot(x, y0, 'C1--', label=descriptions[0] )
    plt.plot(x, y150, 'C2--', label=descriptions[1] )
    plt.plot(x, y200, 'C3--',label=descriptions[2] )
    plt.plot(x, y300, 'C4--',label=descriptions[3] )
    plt.plot(x, y400, 'C5--',label=descriptions[4])
    
    plt.plot(line0x,line0y)
    plt.ylabel("Score difference"); plt.xlabel("iteration"); plt.legend(loc=1)
    plt.title("(Score iteration - score opponent) set out against the iteration") ; plt.show() 
    fig.savefig(ResultingPlotImagesPath + '/' + "Score__" + runSummary + '.png')


def createQintPlot(allDataBot10,ResultingPlotImagesPath):
    # part of the image name:
    runSummary = "QintPlot"
    # Plot the score difference
    x = allDataBot10[0][6];    
    y0 = np.array(allDataBot10[0][8]) - np.array(allDataBot10[0][9])
    y150 = np.array(allDataBot10[1][8]) - np.array(allDataBot10[1][9])
    y200 = np.array(allDataBot10[2][8]) - np.array(allDataBot10[2][9])
    y300 = np.array(allDataBot10[3][8]) - np.array(allDataBot10[3][9])
    y400 = np.array(allDataBot10[4][8]) - np.array(allDataBot10[4][9])
    # A line to emphasize the zero crossing
    line0x = np.arange(0., x[-1], 1.)
    line0y = line0x * 0
    fig = plt.figure()
    # Legenda stuff
    descriptions = ["Q interval = 0","Q interval = 150","Q interval = 200","Q interval = 300","Q interval = 400"]
    #descriptions = ["Q interval = 100","Q interval = 150","Q interval = 200","Q interval = 400","Q interval = 1000"]
    # Plotting
#    plt.plot(x, y0, 'C1--', label=descriptions[0] )
#    plt.plot(x, y150, 'C2--', label=descriptions[1] )
#    plt.plot(x, y200, 'C3--',label=descriptions[2] )
#    plt.plot(x, y300, 'C4--',label=descriptions[3] )
#    plt.plot(x, y400, 'C5--',label=descriptions[4])
    plt.plot(x, y0, 'C1--', label=descriptions[0] )
    plt.plot(x, y150, 'C2--', label=descriptions[1] )
    plt.plot(x, y400, 'C3--',label=descriptions[2] )
    plt.plot(x, y300, 'C4--',label=descriptions[3] )
    plt.plot(x, y200, 'C5--',label=descriptions[4])
    
    plt.plot(line0x,line0y)
    plt.ylabel("Score difference"); plt.xlabel("iteration"); plt.legend()
    plt.title("(Score iteration - score opponent) set out against the iteration") ; plt.show() 
    fig.savefig(ResultingPlotImagesPath + '/' + "Score__" + runSummary + '.png')   

def createPlotsAllopp(statsList, opponents, ResultingPlotImagesPath, runSummary):
    # Plot the score difference
    # evenly sampled time at 200ms intervals
    plotS = True
    plot13 = False
    plot10 = True
    plot8 = False
    plot6 = True
    x = statsList[0][6];    
    yS = np.array(statsList[0][8]) - np.array(statsList[0][9])
    y13 = np.array(statsList[1][8]) - np.array(statsList[1][9])
    y10 = np.array(statsList[2][8]) - np.array(statsList[2][9])
    if plot8: y8 = np.array(statsList[3][8]) - np.array(statsList[3][9])
    y6 = np.array(statsList[4][8]) - np.array(statsList[4][9])
    # A line to emphasize the zero crossing
    line0x = np.arange(0., x[-1], 1.)
    line0y = line0x * 0
    fig = plt.figure()
    if plotS: plt.plot(x, yS, 'C1--', label=opponents[0] )
    if plot13: plt.plot(x, y13, 'C2--', label=opponents[1] )
    if plot10: plt.plot(x, y10, 'C3--',label=opponents[2] )
    if plot8: plt.plot(x, y8, 'C4--',label=opponents[3] )
    if plot6: plt.plot(x, y6, 'C5--',label=opponents[4])
    plt.plot(line0x,line0y)
    plt.ylabel("Score difference"); plt.xlabel("iteration"); plt.legend()
    plt.title("(Score iteration - score opponent) set out against the iteration") ; plt.show() 
    fig.savefig(ResultingPlotImagesPath + '/' + "Score__" + runSummary + '.png')
    
    # Plot the average hits per score
    x = statsList[0][6];    
    yS = np.array(statsList[0][0])
    y13 = np.array(statsList[1][0])
    y10 = np.array(statsList[2][0])
    if plot8: y8 = np.array(statsList[3][0])
    y6 = np.array(statsList[4][0])
    fig = plt.figure()
    if plotS: plt.plot(x, yS, 'C1--', label=opponents[0] )
    if plot13: plt.plot(x, y13, 'C2--', label=opponents[1] )
    if plot10: plt.plot(x, y10, 'C3--',label=opponents[2] )
    if plot8: plt.plot(x, y8, 'C4--',label=opponents[3] )
    if plot6: plt.plot(x, y6, 'C5--',label=opponents[4])
    plt.ylabel("Average hits per score"); plt.xlabel("iteration"); plt.legend()
    plt.title("Average hits per score, set out against the iteration"); plt.show()    
    fig.savefig(ResultingPlotImagesPath + '/' + "AvHits__" + runSummary + '.png')
    
    # Plot the average gametime
    x = statsList[0][6];    
    yS = np.array(statsList[0][1])
    y13 = np.array(statsList[1][1])
    y10 = np.array(statsList[2][1])
    if plot8: y8 = np.array(statsList[3][1])
    y6 = np.array(statsList[4][1])
    fig = plt.figure()
    if plotS: plt.plot(x, yS, 'C1--', label=opponents[0] )
    if plot13: plt.plot(x, y13, 'C2--', label=opponents[1] )
    if plot10: plt.plot(x, y10, 'C3--',label=opponents[2] )
    if plot8: plt.plot(x, y8, 'C4--',label=opponents[3] )
    if plot6: plt.plot(x, y6, 'C5--',label=opponents[4])
    plt.ylabel("Average game time"); plt.xlabel("iteration"); plt.legend()
    plt.title("Average game time set out against the iteration"); plt.show()    
    fig.savefig(ResultingPlotImagesPath + '/' + "av_gametime__" + runSummary + '.png')
    
    # Plot the ball's average Y speed
    x = statsList[0][6];    
    yS = np.array(statsList[0][2])
    y13 = np.array(statsList[1][2])
    y10 = np.array(statsList[2][2])
    if plot8: y8 = np.array(statsList[3][2])
    y6 = np.array(statsList[4][2])
    fig = plt.figure()
    if plotS: plt.plot(x, yS, 'C1--', label=opponents[0] )
    if plot13: plt.plot(x, y13, 'C2--', label=opponents[1] )
    if plot10: plt.plot(x, y10, 'C3--',label=opponents[2] )
    if plot8: plt.plot(x, y8, 'C4--',label=opponents[3] )
    if plot6: plt.plot(x, y6, 'C5--',label=opponents[4])
    plt.ylabel("The ball's average Y speed"); plt.xlabel("iteration"); plt.legend()
    plt.title("The ball's average Y speed set out against the iteration"); plt.show()    
    fig.savefig(ResultingPlotImagesPath + '/' + "av_Yspeed__" + runSummary + '.png')
    
    # Plot the average hit posistion
    x = statsList[0][6];    
    yS = np.array(statsList[0][3])
    y13 = np.array(statsList[1][3])
    y10 = np.array(statsList[2][3])
    if plot8: y8 = np.array(statsList[3][3])
    y6 = np.array(statsList[4][3])
    fig = plt.figure()
    if plotS: plt.plot(x, yS, 'C1--', label=opponents[0] )
    if plot13: plt.plot(x, y13, 'C2--', label=opponents[1] )
    if plot10: plt.plot(x, y10, 'C3--',label=opponents[2] )
    if plot8: plt.plot(x, y8, 'C4--',label=opponents[3] )
    if plot6: plt.plot(x, y6, 'C5--',label=opponents[4])
    plt.ylabel("Absolute hit position"); plt.xlabel("iteration"); plt.legend()
    plt.title("(Absolute hit position set out against the iteration"); plt.show() 
    fig.savefig(ResultingPlotImagesPath + '/' + "av_hitpos__" + runSummary + '.png')

    
    
    

def createPlots(opponent, stats):
    #x_val = [x[0] for x in statsBot10[6]] ;y_val = [x[1] for x in GameHistory]
    # Plot the score difference
    x_val = stats[6]; y_val = np.array(stats[8]) - np.array(stats[9])
    plt.plot(x_val,y_val)
    plt.ylabel("Score difference"); plt.xlabel("iteration"); 
    plt.title("Against " + opponent); plt.show()
    
        

def getDataOneOpponent(readIstart, allLines, numOfCycles):
    readI = readIstart# The readerIndex
    #For the first plots of it 1,4,7.. against last version
    # Create a lists within a list
    stats = []
    for s in range(numOfStatsTotal):
        stats.append([])
    for brainToTest in range(1,numOfCycles,3):
        #print("AT IT ",brainToTest)
        gameStats = np.zeros(10)
        for game in range(numGames):
            #print("AT GAME ", game, " at line ", readI)
            # Loop the stats of one csv row of a game
            for index in range(numStatsPerGame):
                gameStats[index] += float((allLines[readI])[index])
            # Increase the line number with 1
            readI += 1
        # Take the average of all stats per game
        for index in range(numStatsPerGame):
            gameStats[index] = gameStats[index]/numGames
        # Now store the relevant entries of a long row in the csv
        #print("USING READI ",readI)
        gameStats[6] = float((allLines[readI])[3]) #BrainToTest
        gameStats[7] = float((allLines[readI])[6]) #Wins
        gameStats[8] = float((allLines[readI])[7]) #TotScore
        gameStats[9] = float((allLines[readI])[8]) #TotOppScore
        # A row was processed so increment the read index
        readI += 1
        # Add the stats to the larger list
        for s in range(len(gameStats)):
            stats[s].append(gameStats[s])
    return [stats, readI]

def main():    
    #createPlotsFromCSV()
    createPlotFromCSV_Qints()

if __name__ == "__main__":
    try:
        main()
    except:
        print("Unexpected error")
        #Quit the pygame window
        raise
