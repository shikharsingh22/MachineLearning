import monkdata as m
import dtree as d 
import drawtree as l
import random
from matplotlib import pyplot
from numpy import arange

#finding entropy

print "Assignment 1" 
print "dataset  "+"entropy"   
 
print "monk1  "+ str(d.entropy(m.monk1))
print "monk2  "+ str(d.entropy(m.monk2))
print "monk3  "+ str(d.entropy(m.monk3))
print ""
print ""


#finding information average gain

print " Assignment2"
att =[x for x in m.attributes]
monkar = [m.monk1, m.monk2, m.monk3]


for j in monkar:
    entropyGain=[]
    for i in att :
      entropyGain.append(d.averageGain(j,i))
    for i in entropyGain:
      print i 
    print "The attribute used for splitting in data set MONK%d is A%d which has an entropy of %f"%(monkar.index(j)+1,entropyGain.index(max(entropyGain))+1, max(entropyGain))
print ""
print ""


#Building decision Trees

print "Assignment 3"
k=0
print "        "+ "e-train      " +"e-test"
onkar=[m.monk1test, m.monk2test, m.monk3test]
for j, i in zip(monkar, onkar):
       k=k+1
       t= d.buildTree(j, m.attributes)
       print "MONK"+str(k)+"    "+str(d.check(t,j))+"    "+str(d.check(t,i))
       #l.drawTree(t)
       


#plotting the graphs 
efficiency=[]
print " Assignment 4 "



def partition(data, fraction):
                ldata = list(data)
                random.shuffle(ldata)
                breakPoint = int(len(ldata) * fraction)
                return ldata[:breakPoint], ldata[breakPoint:]




def findPrunned(t, monk1val1)  : 
               t2=[]
               t2 = d.allPruned(t)
               
               maxi1 = d.check(t,monk1val1)
               maxi2 = maxi1
               
               for s in t2:
                     val = d.check(s,monk1val1) 
                     
                     if val < maxi1 :
                          maxi1 = val
                          answertree = s
               if maxi1 == maxi2 :
                     answertree = t 
                     efficiency.append(maxi1)
                     print  maxi1
                     return maxi1
               else :
                    x =  findPrunned(answertree,monk1val1)     


par = [0.3, 0.4, 0.5, 0.6, 0.7,0.8]


'''for h in range (0,6):
         monk1train, monk1val = partition(m.monk1, par[h]) 
         tree = d.buildTree(monk1train, m.attributes)
         z = findPrunned(tree, monk1val)
	 print "-----------------"
	 
pyplot.plot(par,efficiency)
pyplot.show()'''

for h in range (0,6):
         monk3train, monk3val = partition(m.monk3, par[h]) 
         tree = d.buildTree(monk3train, m.attributes)
         z = findPrunned(tree, monk3val)
	 print "-----------------"
	 
pyplot.plot(par,efficiency)
pyplot.show()









   
   

    

