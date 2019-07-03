#import pandas as pd  

#names=['sepal_length','sepal_width','petal_length','petal_width','class']
#df=pd.read_csv('./Data/iris.data',header=None,names=names )
#print('KNN started')

#df.head()

import csv
import random
import math
import operator



def loadDataset(datasetPath, split, trainingSet=[] , testSet=[]):
    with open('./Data/iris.data','r') as csvfile:
        lines=csv.reader(csvfile)
#        for row in lines:
#            print(', '.join(row))
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
#            print(random.random())
            if(random.random() < split):
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
#        print(len(trainingSet),"---",len(testSet))
            

def euclideanDistance(instance1,instance2,length):
    distance=0;
    for x in range(length):
        distance+= pow(instance1[x]-instance2[x],2)
    return math.sqrt(distance)

def get_Neighbores(trainSet, testInstance, k):
    distance=[]
    length = len(testInstance)-1
    for x in range(len(trainSet)):
        dist=euclideanDistance(trainSet[x],testInstance,length)
        distance.append((trainSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for i in range(k):
        neighbors.append(distance[i][0])
    return neighbors

def get_Responce(neighbors):
    classVotes={}
    for x in range (len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response]=1
    sortedVotes= sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def get_Accuracy(testSet,prediction):
    correct=0
    for x in range(len(testSet)):
        if(testSet[x][-1] == prediction[x]):
            correct+=1
    return (correct/float(len(testSet)))*100.0



def main():

    dataPath='./Data/iris.data'
    trainData=[]
    testData=[]
    data1=[2,2,2,'a']
    data2=[4,4,4,'b']
    loadDataset(dataPath,.5,trainData, testData)

    print ('Train '+ repr(len(trainData)))
    print ('Test '+  repr(len(testData)))

    #generate prediction
    prediction=[]
    k=3
    for x in range(len(testData)):
        neighbors=get_Neighbores(trainData,testData[x],k)
        result=get_Responce(neighbors)
        prediction.append(result)
        print("> predicted:"+ result + ', actual:'+ testData[x][-1])

 #   for x in range(len(prediction)):
 #       print( prediction[x] == testData[x][-1])

    accuracy=get_Accuracy(testData,prediction)
    print('Accuracy:'+repr(accuracy)+'%')


    ## get_Neighbores test
    #trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
    #testInstance = [5, 5, 5]
    #k = 1
    #neighbors = get_Neighbores(trainSet, testInstance, k)
    #print(neighbors)


    ## get_Responce test
    #neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    #response = get_Responce(neighbors)
    #print(response)


    ## accuracy test
    #testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    #predictions = ['a', 'a', 'a']
    #accuracy = get_Accuracy(testSet, predictions)
    #print(accuracy)

    #print(euclideanDistance(trainData[1],trainData[2],4))
    #print( euclideanDistance(data1,data2,3))


main()