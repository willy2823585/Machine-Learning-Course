# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:34:00 2019

@author: willywanghw
"""

#單層感知器 應用於真值表
class Perceptron :    #感知器物件
    def step(x,w):    #步階函數：計算目前權重 w 的情況下，網路的輸出值為 0 或 1
        result = w[0]*x[0]+w[1]*x[1]+w[2]*x[2]  #y=w0*x0+x1*w1+x2*w2=-theta+x1*w1+x2*w2
        if result >= 0: #如果結果大於零
            return 1    #就輸出 1
        else :          #否則
            return 0    #就輸出 0
    #self要加入 不然編譯會出錯
    def training(self,truthTable):#訓練函數 training(truthTable), 其中 truthTable 是目標真值表
        rate = 0.01  #學習調整速率，也就是 alpha
        w =[1,0,0]
        for i in range(0,1001,1):  #最多訓練一千輪
            eSum = 0.0 
            for j in range(len(truthTable)) : #每輪對於真值表中的每個輸入輸出配對，都訓練一次
                x =  [ -1, truthTable[j][0],truthTable[j][1] ] #輸入：x
                yd = truthTable[j][2]  #期望的輸出 yd
                y = Perceptron.step(x,w) #目前的輸出 y
                e = yd - y  #差距 e = 期望的輸出 yd - 目前的輸出 y
                eSum = eSum+e*e  #計算差距總和
                dw = [ 0, 0, 0 ] #權重調整的幅度 dw
                dw[0] = rate * x[0] * e
                w[0] =w[0] + dw[0]  #w[0] 的調整幅度為 dw[0]
                dw[1] = rate * x[1] * e
                w[1] =w[1] + dw[1]  #w[1] 的調整幅度為 dw[1]
                dw[2] = rate * x[2] * e
                w[2] =w[2] + dw[2]  #w[2] 的調整幅度為 dw[2]
         #       if i % 10 ==0:
           #         print("dw[0]",dw[0],"dw[1]",dw[1],"dw[2]",dw[2],"w[2]",w[2])          
            if abs(eSum) <0.0001  :  #當訓練結果誤差夠小時，就完成訓練了。
                return w 
        return None #否則，就傳會 none 代表訓練失敗。

def learn(tablename,truthTable):
    p = Perceptron() 
    w = p.training(truthTable) 
    print(w)
    if w != None :
        print("學習成功!")
    else :
        print("學習失敗!")

#主程式
andTable = [ [ 0, 0, 0 ], [ 0, 1, 0 ], [ 1, 0, 0 ], [ 1, 1, 1 ] ]
#AND 函數的真值表
orTable  = [ [ 0, 0, 0 ], [ 0, 1, 1 ], [ 1, 0, 1 ], [ 1, 1, 1 ] ]
#OR  函數的真值表
xorTable = [ [ 0, 0, 0 ], [ 0, 1, 1 ], [ 1, 0, 1 ], [ 1, 1, 0 ] ]
#XOR 函數的真值表

learn("and",andTable)#學習 AND 函數
learn("or",  orTable)#學習 OR  函數
learn("xor", xorTable)#學習 XOR 函數
