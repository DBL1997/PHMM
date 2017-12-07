# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:00:13 2017

@author: Xu Tongbo
"""
'''
580行开始是主程序，前面都是函数
'''

import random
import math
import xlwt

def choose(k,tk): 
    #k是列表，一般是e[m][n] ; tk是01的随机数
    l = len(k)
    ksum = []
    ksum.append(k[0])
    for i in range(1,l):
        ksum.append(ksum[i-1]+k[i])
    if tk <= ksum[0]:
        return 0
    elif tk > ksum[l-2]:
        return l-1
    else:
        for j in range(1,l-1):
            if tk > ksum[j-1] and tk <= ksum[j]:
                return j
 
def generate(a,e,index):
    M = len(a[0][0]) - 1
    tt = 0
    state = 0
    statesave = []
    string = []
    while tt <= M:
        if tt == M:
            r1 = random.random()
            aaa = [a[state][0][tt] , a[state][1][tt]]
            state2 = choose(aaa , r1)
            statesave.append(state2)
            if state2 ==0:
                tt = tt+1
            else:
                r2 = random.random()
                emit = choose(e[1][tt],r2)
                string.append(index[emit])
            state = state2
        else:
            r1 = random.random()
            aaa = [a[state][0][tt] , a[state][1][tt] , a[state][2][tt]]
            state2 = choose(aaa , r1)
            statesave.append(state2)
            if state2 ==0:
                tt = tt+1
                r2 = random.random()
                emit = choose(e[0][tt],r2)
                string.append(index[emit])           
            elif state2 ==1:
                r2 = random.random()
                emit = choose(e[1][tt],r2)
                string.append(index[emit])
            else:
                tt = tt+1
            state = state2
            #print tt
    #print statesave
    return string


def form_standard(hmm):
    a = hmm.decode('utf-8')
    h = len(a)
    b = list('0' for i in range(0,h))
    temp = 0
    for i in range(0,h):
            b[temp]=a[i].encode('utf-8')
            temp=temp+1
    return b

def search_number(hmm,index):
    y=hmm
    L_chain = len(y)
    x=[-1 for i in range(0,L_chain)]
    n=len(index)
    for i in range(0,L_chain):
        for j in range(0,n):
            if y[i]==index[j]:
                #print y[i]
                x[i]=j
                #print x[i]
    return x  

def print_a(a):
    l = len(a[0][0])
    for i in range(l):
        print i,'position'
        print a[0][0][i],a[0][1][i],a[0][2][i]
        print a[1][0][i],a[1][1][i],a[1][2][i]
        print a[2][0][i],a[2][1][i],a[2][2][i]
        
def print_e(e):
    l = len(e[0])
    for i in range(l):
        print i,'position'
        print e[0][i]
        print e[1][i]


def forward(hmm,a,e,index):
    #e is a matrix with size of 2x M+1 xn, define the index as M=0,I=1,D=2
    #a is a matrix with size of 3x3xM+1
    #chain is the chain that we desire to figure out it prbability
    #f:3xM+1 x L+1
    x=search_number(hmm,index)
    L=len(hmm)
    #L_standard = len(e[0])-1
    M = len(e[0])-1
    f=[[[0 for j in range (L+1)] for i in range(M+1)] for k in range(3)]
    
#initialization
    f[0][0][0] = 1
    f[2][1][0] = a[0][2][0] * f[0][0][0] 
    f[1][0][1] = a[0][1][0] * f[0][0][0] * e[1][0][x[0]] 
    for k in range(2,L+1):
        f[1][0][k] = a[1][1][0] * f[1][0][k - 1] * e[1][0][x[k-1]]
    for j in range(2,M+1):
        f[2][j][0] = a[2][2][j - 1] * f[2][j - 1][0] 
#iteration
    for k in range(1,M+1):
        for i in range(1,L+1):
            f[0][k][i]=e[0][k][x[i-1]]*(f[0][k-1][i-1]*a[0][0][k-1]+f[1][k-1][i-1]*a[1][0][k-1]+f[2][k-1][i-1]*a[2][0][k-1])            
            f[1][k][i]=e[1][k][x[i-1]]*(f[0][k][i-1]*a[0][1][k]+f[1][k][i-1]*a[1][1][k]+f[2][k][i-1]*a[2][1][k])
            f[2][k][i]=f[0][k-1][i]*a[0][2][k-1]+f[1][k-1][i]*a[1][2][k-1]+f[2][k-1][i]*a[2][2][k-1]
    px = f[0][M][L]*a[0][0][M]+f[1][M][L]*a[1][0][M]+f[2][M][L]*a[2][0][M]
    #print(f[0][M+1][L+1])
    '''
    if px==0:
        print a
        print "ASD"
        print e
        print "ASD"
        print f
    '''
    #print "forward"
    #for xtb in range(3):
    #    print f[xtb]
    #print px
    return f, px

def backward(hmm,A,E,index):
    #E 2*(L_standard+1)*n
    #A 3*3*(L_standard+1)

    #3* M+1 * L+1

    L = len(hmm)
    #L_chain = len(hmm)
    x=search_number(hmm,index)
    #print x
    M = len(A[0][0])-1
    #L_standard = len(E[0])-1
    #print L_standard
    B = [[[1 for j in range (L+1)] for i in range(M+1)] for k in range(3)]
    for i in range(0,3):
        B[i][M][L]=A[i][0][M]
        
    for k in range(L-1,-1,-1):
        B[0][M][k]=B[1][M][k+1]*A[0][1][M]*E[1][M][x[k]]
        B[2][M][k]=B[1][M][k+1]*A[2][1][M]*E[1][M][x[k]]
        B[1][M][k]=B[1][M][k+1]*A[1][1][M]*E[1][M][x[k]]
    for j in range(M-1, -1, -1):
        B[0][j][L]=B[2][j+1][L]*A[0][2][j]
        B[1][j][L]=B[2][j+1][L]*A[1][2][j]
        B[2][j][L]=B[2][j+1][L]*A[2][2][j]
    for k in range(M-1,-1,-1):
        #print k
        for i in range(L-1,-1,-1):
            B[0][k][i]=B[0][k+1][i+1]*A[0][0][k]*E[0][k+1][x[i]]+B[1][k][i+1]*A[0][1][k]*E[1][k][x[i]]+B[2][k+1][i]*A[0][2][k]
            B[1][k][i]=B[0][k+1][i+1]*A[1][0][k]*E[0][k+1][x[i]]+B[1][k][i+1]*A[1][1][k]*E[1][k][x[i]]+B[2][k+1][i]*A[1][2][k]
            B[2][k][i]=B[0][k+1][i+1]*A[2][0][k]*E[0][k+1][x[i]]+B[1][k][i+1]*A[2][1][k]*E[1][k][x[i]]+B[2][k+1][i]*A[2][2][k]
    #print "backward"
    #for xtb in range(3):
    #    print B[xtb]
    return B
    #print B

def viterbi(a,e,index,x,N1):
    #L代表序列长度，a为状态转移矩阵3*3*(L+1)，e为发射矩阵3*(L+1)*n，(n是出现过的中文obs个数)
    #x是要进行计算中文序列
    #a [i]表示从第i位转移到第i+1位； Insert为第i位到第i位的转移
    #第0位代表初始，e[0][0][]=0,e[1][0][]!=0(第0位只有插入)
    #index是一列中文序列1*n，与e对应，第i个中文字符代表发射矩阵e的（*，*，i）
    #N1代表总不同的字符个数
    #识别与转化（将x中文序列转化为数字序列）
    # x_num[i]实际上代表xi中文字符对应到e的位置
    '''
    判断字符串index是否有重复
    '''
    num_rep = len(list(set(index)&set(x)))
    if num_rep == 0:
        return [],0
    else:
        M=len(e[0])-1
        L=len(x)
        x_num=search_number(x,index)
        v = [[[0 for j in range (L+1)] for i in range(M+1)] for k in range(3)]
        ptr = [[[3 for j in range (L+1)] for i in range(M+1)] for k in range(3)]
        
        N=len(index) #N代表这个code下不同字符个数
        if N1 != N:
            p0=0.01/(N1-N) 
            p1=0.01/(N1-N)
        else:
            p0 = 0
            p1 = 0
        #p0 match发射一个未知数的概率
        #p1 insert发射一个未知数的概率
        #定义v和ptr(代表回溯)
        #格式：v(M)(j)(i) j是第j个状态，i是x第i个obs,M=0,1,2分别代表 M，I，D
        
    
        #初始化
        #注意j=0时代表开始状态，规定其为M；i=0时代表进入模型前的插入，图参考中文版P76
        v[0][0][0]=1
        if x_num[0]==-1:
            v[1][0][1]=a[0][1][0]*p1
        else:
            v[1][0][1]=a[0][1][0]*e[1][0][x_num[0]]
        v[2][1][0]=a[0][2][0]
        
        ptr[1][0][1]=0
        ptr[2][1][0]=0
        for i in range(2,L+1):
            if x_num[i-1]==-1:
                v[1][0][i]=v[1][0][i-1]*a[1][1][0]*p1
            else:
                v[1][0][i]=v[1][0][i-1]*a[1][1][0]*e[1][0][x_num[i-1]]
            ptr[1][0][i]=1
                             
        for j in range(2,M+1):
            v[2][j][0]=v[2][j-1][0]*a[2][2][j-1]
            ptr[2][j][0]=2
        
        for i in range(2,L+1):
            ptr[0][1][i]=1
            ptr[2][1][i]=1
        for j in range(2,M+1):
            ptr[0][j][1]=2
            ptr[1][j][1]=2
        ptr[0][1][1]=0
        ptr[1][1][1]=2
        ptr[2][1][1]=1
        #print ptr
        #print v
        #循环
        i=1
        j=1
        for i in range(1,L+1):
            for j in range(1,M+1):
                 # Match M=0
                 m=-1
                 for k in range(3):
                     t = v[k][j-1][i-1]*a[k][0][j-1]
                     if t > m:
                         m = t
                         ptr[0][j][i] = k
                 if x_num[i-1]==-1:
                     v[0][j][i]=m*p0
                 else:
                     v[0][j][i]=m*e[0][j][x_num[i-1]] 
                 # Insert M=1
                 m=-1
                 for k in range(3):
                     t = v[k][j][i-1]*a[k][1][j]
                     if t > m:
                         m = t
                         ptr[1][j][i] = k
                 if x_num[i-1]==-1:
                     v[1][j][i]=m*p1
                 else:
                     v[1][j][i]=m*e[1][j][x_num[i-1]]
                 # Delete M=2
                 m=-1
                 for k in range(3):
                     t = v[k][j-1][i]*a[k][2][j-1]
                     if t > m:
                         m = t
                         ptr[2][j][i] = k
                 v[2][j][i]=m
               
        
        #终止
        pii=[]
        P=0
        m=0 #记录最后一位max
        for k in range(3):
            t=v[k][M][L]*a[k][0][M]
            if P< t:
                P=t
                m=k
        pii.append(m)
        
        ptr[1][0][1]=0
        ptr[2][1][0]=0
        for i in range(2,L+1):
            ptr[1][0][i]=1
                             
        for j in range(2,M+1):
            ptr[2][j][0]=2
        
        for i in range(2,L+1):
            ptr[0][1][i]=1
            ptr[2][1][i]=1
        for j in range(2,M+1):
            ptr[0][j][1]=2
            ptr[1][j][1]=2
        ptr[0][1][1]=0
        ptr[1][1][1]=2
        ptr[2][1][1]=1
        
        #print ptr
        #print " "
        #print v
        #回溯
        b1=M
        b2=L
        g=m
        #print ptr
        #print v
        while b1>0 or b2>0:
            
            h=ptr[g][b1][b2]
            #print g
            #print b1
            #print b2
            #print h
            #print "  "
            if g==0:
                b1=b1-1
                b2=b2-1
            if g==1:
                b2=b2-1
            if g==2:
                b1=b1-1
            g=h
            pii.append(h)
            
        pii.reverse()
       
        #print "viterbi"    
        return pii,P

def bw(X,index,M):
    #设想的输入是n*。每行为一个独特序列，最后一列是频率，后面的元素为中文字符组成的该字符串
    #X为序列组成的矩阵X[n][kn+1]
    nn=len(X) #序列个数
#    import search_number
#    y=search_number.form_standard(hmm)      

    #初始化
    N=len(index) #N是出现过的中文个数
    a = [[[0.1 for j in range (M+1)] for i in range(3)] for k in range(3)]
    for j in range(M+1):
        for k in range(3):
            a[k][0][j]=0.8
    e = [[[1.0/N for j in range (N)] for i in range(M+1)] for k in range(2)]
    for j in range(0,N):
        e[0][0][j]=0
    for j in range(0,3):
        a[2][j][0]=0
        a[j][2][M]=0
        a[j][1][M]=0.2
         
    #print_a(a)
    #print_e(e)
    #收敛阈值
    epsilon=0.01
    #循环迭代 
#注意变量赋值会同时改变
    ee=2  
    t=0
    #for xtb in range(2):         
    while ee>epsilon:
        #初始化a0,e0作为a，e的一个记录
        #print t,"iteration"
        for j in range(0,N):
            e[0][0][j]=0
        for j in range(0,3):
            a[2][j][0]=0
            a[j][2][M]=0
        a0 = [[[0 for j in range (M+1)] for i in range(3)] for k in range(3)]
        e0 = [[[0 for j in range (N)] for i in range(M+1)] for k in range(2)]
        a1 = [[[0 for j in range (M+1)] for i in range(3)] for k in range(3)]
        e1 = [[[0 for j in range (N)] for i in range(M+1)] for k in range(2)]
        for j in range(0,nn):
            #print "  "
            #print j
            #初始化AE
            A = [[[0 for jj in range (M+1)] for ii in range(3)] for kk in range(3)]
            E = [[[0 for jj in range (N)] for ii in range(M+1)] for kk in range(2)]
            L=len(X[j])#以后添上频率操作系列   
            f, px = forward(X[j],a,e,index)
            #print px
            #print " "
            b = backward(X[j],a,e,index)
            #print b
            #print " "
            x_num=search_number(X[j],index)
            for m in range(0,3):
                for k in range(0, M):
                    for i in range(0, L):
                        A[m][0][k]+=f[m][k][i]*a[m][0][k]*e[0][k+1][x_num[i]]*b[0][k+1][i+1]
                    for i in range(0,L+1):
                        A[m][2][k]+=f[m][k][i]*a[m][2][k]*b[2][k+1][i]
                    if A[m][0][k] != 0: #添加
                        a0[m][0][k]+=A[m][0][k]/px
                    if A[m][2][k] != 0: #添加
                        a0[m][2][k]+=A[m][2][k]/px
                if f[m][M][L] * a[m][0][M] !=0: #添加
                    a0[m][0][M]+=f[m][M][L]*a[m][0][M]/px
                a0[m][2][M]+=0
                for k in range(0, M+1):
                    for i in range(0, L):
                        A[m][1][k]+=f[m][k][i]*a[m][1][k]*e[1][k][x_num[i]]*b[1][k][i+1]
                    if A[m][1][k] != 0: #添加
                        a0[m][1][k]+=A[m][1][k]/px            
            for c in range(0,N):
                for k in range(0,M+1):
                    for i in range(1,L+1):
                        if x_num[i-1]==c:
                            E[0][k][c]+=f[0][k][i]*b[0][k][i]
                            E[1][k][c]+=f[1][k][i]*b[1][k][i]
                    if E[0][k][c]!=0:
                        e0[0][k][c]+=E[0][k][c]/px 
                    if E[1][k][c]!=0:
                        e0[1][k][c]+=E[1][k][c]/px                     

        #标准化
        for m in range(0,3):
            for k in range(0,M+1):
                ss=a0[m][0][k]+a0[m][1][k]+a0[m][2][k]
                if ss!=0:
                    a1[m][0][k]=(a0[m][0][k]+0.0)/ss
                    a1[m][1][k]=(a0[m][1][k]+0.0)/ss
                    a1[m][2][k]=(a0[m][2][k]+0.0)/ss
                
                  
        for m in range(0,2):
            for n in range(0,M+1):
                for k in range(0,N):
                    if sum(e0[m][n])!=0:
                        e1[m][n][k]=(e0[m][n][k]+0.0)/sum(e0[m][n])
   
        
        max1=0
        max2=0
        for m in range(0, 3):
            for n in range(0, 3):
                for k in range(0, M+1):
                    if max1<abs(a1[m][n][k]-a[m][n][k]):
                        max1=abs(a1[m][n][k]-a[m][n][k])
                    a[m][n][k]=a1[m][n][k]
        
        for k in range(0,N):
            for m in range(0, 2):
                for n in range(0, M+1):
                    if max2<abs(e1[m][n][k]-e[m][n][k]):
                        max2=abs(e1[m][n][k]-e[m][n][k])
                    e[m][n][k]=e1[m][n][k]
                 
        ee = max(max1,max2)
        t=t+1
        #print ee
        #print " "
        #print e
        #print " "
        #print t
        #print " "
        #print " "
    #循环外的标准化

    #print "a"
    #print_a(a)
    #print "e"
    #print_e(e) 
    #print " "
    #print " "        
    return a,e

def add_a(a,pseudo_a):
    M = len(a[0][0])-1
    pseudo_a = pseudo_a / M
    for m in range(0,3):
        for k in range(0,M):
            ss=a[m][0][k]+a[m][1][k]+a[m][2][k]
            a[m][0][k]=(a[m][0][k]+pseudo_a)/(ss+3*pseudo_a)
            a[m][1][k]=(a[m][1][k]+pseudo_a)/(ss+3*pseudo_a)
            a[m][2][k]=(a[m][2][k]+pseudo_a)/(ss+3*pseudo_a)
    for j in range(0,3):
        a[2][j][0]=0
        
    for m in range(0,3):    
        ss=a[m][0][M]+a[m][1][M]
        a[m][0][M]=(a[m][0][M]+pseudo_a)/(ss+2*pseudo_a)
        a[m][1][M]=(a[m][1][M]+pseudo_a)/(ss+2*pseudo_a)
        a[m][2][M]=0
    return a
    
def add_e(e,pseudo_e):
    M = len(e[0])-1
    N = len(e[0][1])
    for m in range(0,2):
        for n in range(0,M+1):
            dd=sum(e[m][n])
            for k in range(0,N):
                e[m][n][k]=(e[m][n][k]+pseudo_e/N)/(dd+pseudo_e)
    for j in range(0,N):
        e[0][0][j]=0
    return e



def poisson(L):  
    """ 
    poisson distribution 
    return a integer random number, L is the mean value 
    """  
    p = 1.0  
    k = 0  
    e = math.exp(-L)  
    while p >= e:  
        u = random.random()  #generate a random floating point number in the range [0.0, 1.0)  
        p *= u  
        k += 1  
    return k-1  

def ERandom(n, num):
    A = range(0, n)
    B = random.sample(A, num)
    B.sort()
    B.reverse()
    return B

def ASC2list(asc):
    length = len(asc)
    realist = []
    ascount = 0
    while ascount < length:
        if asc[ascount]=='?' or asc[ascount]=='<' or asc[ascount]== '(' or asc[ascount]==')'or asc[ascount]=='/'or asc[ascount]=='-'or asc[ascount]==':'or asc[ascount]=='.'or asc[ascount]==','or asc[ascount]=='['or asc[ascount]==']'or asc[ascount]=='+' or asc[ascount]=='%':
            realist.append(asc[ascount])
            ascount = ascount+1
        elif asc[ascount]<='9' and asc[ascount]>='0':
            realist.append(asc[ascount])
            ascount = ascount+1
        elif asc[ascount]<='Z' and asc[ascount]>='A':
            realist.append(asc[ascount])
            ascount = ascount+1
        elif asc[ascount]<='z' and asc[ascount]>='a':
            realist.append(asc[ascount])
            ascount = ascount+1
        else:
            temp=asc[ascount]+asc[ascount+1]
            realist.append(temp)
            ascount = ascount + 2
    return realist



def maxthree(a):
    c = sum(a)
    b = []
    for i in range(0, 3):
        maxposition = 0
        maxposition = a.index(max(a))
        b.append(maxposition)
        b.append((a[maxposition]+0.0)/c)
        a[maxposition] = 0
    d = b[1]+b[3]+b[5]
    b.append(d)
    return b



##############################################################
##############################################################
Document=open("C:\Users\Xu Tongbo\Desktop\ceshi1.txt")
line= Document.readlines()
n=len(line)#总条目数
Description = []#用于记录疾病的描述
Num = []#记录每个Description有多少个
CodeBelong = []#每个Description属于哪个Code
Length = []#每个Description的长度
FullRecord = []
CodeCountRecord = []
FullProblemRecord = []

times = 5#总迭代次数

#先对数据进行第一轮读取，之后就不用
for i in range(0, n):
    trisplit = line[i].split()
    CodeBelong.append(trisplit[0])
    Num.append(int(trisplit[1]))
    Description.append(ASC2list(trisplit[2]))
    Length.append(len(ASC2list(trisplit[2])))
    
#数据整理完成后，可以开始进行迭代

for time in range(0, times):
    CodeName = []
    CodeName = list(set(CodeBelong))
    CodeCount = 0
    CodeCount = len(CodeName)
    CodeCountRecord.append(CodeCount)
    Code_X = [list() for i in range(0, CodeCount)]
    CodeLetterCount = [0 for i in range(CodeCount)]
    
    
    #对拿到的数据按Code进行分类，并进行构造Code_X，同时为了方便起见，在这里就统计每个Code下的总字符数
    for i in range(0, n):
        Locate = CodeName.index(CodeBelong[i])
        for j in range(0, Num[i]):
            Code_X[Locate].append(Description[i])
            CodeLetterCount[Locate] += len(Description[i])
    
    #每个Code下面的String数量
    CodeNum = []
    for i in range(0, CodeCount):
        CodeNum.append(len(Code_X[i]))
        
    #统计每个Code下的字典和总字典
    Index = []
    for i in range(0, CodeCount):
        Indextemp = []
        for j in range(0, CodeNum[i]):
            Indextemp += Code_X[i][j]
        Indextemp1 = list(set(Indextemp))
        Index.append(Indextemp1)
        
    Indexsum = []
    Indexsumtemp = []
    for i in range(0, CodeCount):
        Indexsumtemp += Index[i]
    Indexsum = list(set(Indexsumtemp))
    
    
    #开始进行BW算法（模型的构建）
    aa = []
    ee = []
    for i in range(0, CodeCount):
        aatemp, eetemp = bw(Code_X[i], Index[i], int(round(float(CodeLetterCount[i])/float(CodeNum[i]))))
        aatemp1 = add_a(aatemp, 0.01)
        eetemp1 = add_e(eetemp, 0.01)
        aa.append(aatemp1)
        ee.append(eetemp1)
        print i
    
    CodeBelong_New = []
    
    ProblemRecord = []
    TimeRecord = []
    #用Viterbi算法算出每个UniqueString最匹配的前三个Code，然后记录并且准备输出到Excel文件中
    for i in range(0, n):
        ScoreSave = []
        for j in range(0, CodeCount):
            pii, ptr = viterbi(aa[j], ee[j], Index[j], Description[i], len(Indexsum))
            ScoreSave.append(ptr * CodeNum[j])
        Topthree = maxthree(ScoreSave)
        CodeBelong_New.append(CodeName[Topthree[0]])
        if Topthree[6] < 0.95:
            ProblemRecord.append(['<0.95', i])
        if Topthree[1] < Topthree[3] * 100:
            ProblemRecord.append(['<100', i])
        if CodeBelong[i] != CodeBelong_New[i]:
            ProblemRecord.append(['Wrong', i])
        if Topthree[3] == 0:
            Logrecord = 'NA'
        else:
            Logrecord = math.log10(Topthree[1]/Topthree[3])
        TimeRecord.append([CodeBelong[i], CodeBelong_New[i], Topthree[1], CodeName[Topthree[2]], Topthree[3], CodeName[Topthree[4]], Topthree[5], Topthree[6],Logrecord])
        print i, 'vi'
        
    FullRecord.append(TimeRecord)    
    CodeBelong = []
    CodeBelong += CodeBelong_New
    FullProblemRecord.append(ProblemRecord)
    

            
f1 = open('C:/Users/Xu Tongbo/Desktop/CodeCountRecord.txt','a')    
f1.write("记录Code数量的变化"+"\n")
for i in range(0, times):
    f1.write(str(CodeCountRecord[i])+"\n")
f1.close()

e = xlwt.Workbook() #创建工作簿
sheet1 = e.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
for j in range(0,times): 
   for i in range(0,len(FullProblemRecord[j])):
       sheet1.write(i + 1,3 * j + 1,FullProblemRecord[j][i][0])
       sheet1.write(i + 1,3 * j + 2,FullProblemRecord[j][i][1])
e.save('ProblemRecord.xls')


f = xlwt.Workbook() #创建工作簿
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
for j in range(0, times):
    for i in range(0, n):
        for k in range(0, 9):
            sheet1.write(i + 1, 10 * j + 1 + k, FullRecord[j][i][k])
f.save('FullRecord.xls')#保存文件    

        
        
    



