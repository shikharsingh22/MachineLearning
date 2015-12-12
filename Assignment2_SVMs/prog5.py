from cvxopt import solvers
from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math
from numpy import zeros, ones, identity,exp
from matplotlib import pyplot as pylab
from numpy import linalg




def Kernel1(p1, p2):
          return p1[0]*p2[0] + p1[1]*p2[1] +1


def Kernel2(r1,r2):
    return pow((r1[0]*r2[0] + r1[1]*r2[1] +1),3 )


def gaussian_kernel(x, y, sigma=50.0):
    u=x[0]-y[0],x[1]-y[1]
    return numpy.exp(-linalg.norm(u)**2 / (2 * (sigma ** 2)))




def buildmatrix(dataset):
    list1=[]
    for i in dataset:
        list2=[]
        for j in dataset:
            list2.append(i[2]*j[2]*gaussian_kernel(i,j))
        list1.append(list2)
    return list1







classA = [(random.normalvariate(-1.5,0.5),random.normalvariate(0.5,0.5),1.0)for i in range(5)]+[(random.normalvariate(1.5,1),random.normalvariate(0.5,1),1.0)for i in range(5)]
classB = [(random.normalvariate(0.0,0.5),random.normalvariate(-0.5,0.5),-1.0)for i in range(10)]


data = classA + classB
data1= data
random.shuffle(data)
#print data
P = buildmatrix(data)



q = matrix(ones(len(data))*-1)
G = identity(len(data))*-1
h = matrix(zeros(len(data)))
#print matrix(G)


r=solvers.qp(matrix(P),q,matrix(G), h)

Alpha= list (r['x'] )

#print Alpha






def indicatorFunc(x):

    ind = 0
    for i,j in zip(data,Alpha):

        if ((-1*pow(10,-5)) < j <(pow(10,-5))):
            continue
        ind+=(j*i[2]*gaussian_kernel(x,i))
    return ind









'''
for d in data1:
    print(indicatorFunc(d, data, Alpha))
'''




xrange = numpy.arange(-10,10,1)
yrange = numpy.arange(-10,10,1)

grid=matrix([[indicatorFunc((x,y))for y in yrange]for x in xrange])
#print grid


pylab.plot([p[0] for p in classA],
[p[1] for p in classA],'bo')

pylab.plot([p[0] for p in classB],
[p[1] for p in classB],'ro')

pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0),colors=('red','blue','green'),linewidths=(1,3,1))


pylab.show()
#pylab.plot(xrange,yrange)
