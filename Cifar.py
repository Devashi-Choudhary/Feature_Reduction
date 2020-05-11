#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle


# In[2]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        x = pickle.load(fo, encoding='bytes')
    return x[b'data'],x[b'labels']


# In[4]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/SML/Assignment3/train"
TrainingSet= pd.DataFrame()
Labels=pd.DataFrame()
for folder in os.listdir(path):
    file=os.path.join(path,folder)
    Data,Label=unpickle(file)
    Data=pd.DataFrame(Data)
    Labels=Labels.append(Label)
    TrainingSet=TrainingSet.append(Data)
TrainingSet.index=range(len(TrainingSet))   


# In[5]:


print(len(Labels))


# In[6]:


TrainingSet


# In[7]:


import numpy as np
def RGBtoGray(dataframe):
    dataframe=dataframe.values
    grayscale=np.zeros((len(dataframe),1024))
    for i in range(len(dataframe)):
        for j in range(1024):
            grayscale[i][j]=0.299*dataframe[i][j]+0.587*dataframe[i][j+1024]+0.114*dataframe[i][j+2048]
    return grayscale  


# In[8]:


gray=RGBtoGray(TrainingSet)


# In[9]:


print(len(Labels))
print(len(TrainingSet))


# In[10]:


Train=pd.DataFrame(gray)
Labels.index=range(len(Labels))
Labels.columns=['Label']
Train=pd.concat([Train,Labels],axis=1)


# In[104]:


Train


# In[13]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/SML/Assignment3/test"
TestingSet= pd.DataFrame()
TestLabels=pd.DataFrame()
for folder in os.listdir(path):
    file=os.path.join(path,folder)
    print(file)
    TestData,Labels=unpickle(file)
    TestData=pd.DataFrame(TestData)
    TestLabels=TestLabels.append(Labels)
    TestingSet=TestingSet.append(TestData)
TestingSet.index=range(len(TestingSet))   


# In[14]:


TestingSet


# In[16]:


grayTest=RGBtoGray(TestingSet)


# In[17]:


Test=pd.DataFrame(grayTest)
TestLabels.index=range(len(TestLabels))
TestLabels.columns=['Label']
Test=pd.concat([Test,TestLabels],axis=1)


# In[159]:


print(Train)


# In[18]:


TestingSet


# In[161]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','pink']
for i in range(10):
    y=Train[Train['Label']==(i)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[144]:


import numpy as np
def PCA(pca_data,EigenEnergy):
    Eigen=pd.DataFrame()
    mean=np.mean(pca_data)
    pca_data=pca_data-mean
    covariance=pca_data.cov()
    eigen_value,eigen_vector=np.linalg.eig(covariance)
    for i in range(len(eigen_value)):
        if(eigen_value[i]<0):
            eigen_value[i]=(-1)*eigen_value[i]
    eigen_sum=np.sum(eigen_value)
    x=(EigenEnergy/100)*eigen_sum
    Eigen_Value=pd.DataFrame(eigen_value)
    Eigen_Vector=pd.DataFrame(eigen_vector)
    Eigen_Vector=Eigen_Vector.T
    Eigen=pd.concat([Eigen_Value,Eigen_Vector],axis=1,ignore_index=True)
    Eigen=Eigen.sort_values(by= [0], axis=0, ascending=False, inplace=False)
    s=0
    k=0
    for i in range(len(Eigen[0])):
        s+=Eigen[0][i]
        if(s<=x):
            k=k+1
        else:
            break
        
    Eigen=Eigen.drop([0],axis=1)
    #Eigen=Eigen.iloc[0:32]
    Eigen=Eigen.iloc[0:int(k)]  
    Eigen=Eigen.T
    return Eigen,k
Eigen_PCA,k=PCA(Train.iloc[:,0:1024],95)   


# In[145]:


Eigen_PCA=Eigen_PCA.values.real
Eigen_PCA=pd.DataFrame(Eigen_PCA)
Eigen_PCA


# In[146]:


import cv2
import matplotlib.pyplot as plt
row,col=Eigen_PCA.shape

for i in range(col):
    a=np.array(Eigen_PCA[i]).reshape(32,32)
    plt.imshow(a)
    plt.show()


# In[31]:


def Reduced_Data(Original_Data,Eigen_Data):
    print(Original_Data.shape)
    print(Eigen_Data.shape)
#     a=Original_Data.values
#     b=Eigen_Data.values
    c=np.dot(Original_Data.values,Eigen_Data.values)
    Reduced_Data=pd.DataFrame(c)
    return Reduced_Data
Dimension_Reduction_PCA=Reduced_Data(Train.iloc[:,0:1024],Eigen_PCA)
Dimension_Reduction_PCA=pd.concat([Dimension_Reduction_PCA, Train['Label']],axis=1)


# In[33]:


from sklearn.decomposition import PCA as pca
lda1 = pca(n_components=2)  
inbuilt_pca=lda1.fit_transform(Train)


# In[35]:


inbuilt_pca=pd.DataFrame(inbuilt_pca)
inbuilt_pca=pd.concat([inbuilt_pca, Train['Label']],axis=1)


# In[39]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink','brown']
for i in range(11):
    y=inbuilt_pca[inbuilt_pca['Label']==(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[162]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','pink']
for i in range(11):
    y=Dimension_Reduction_PCA[Dimension_Reduction_PCA['Label']==(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[147]:


newtraindata,k=PCA(Train.iloc[:,0:1024],90)
reduced_dim=Reduced_Data(Train.iloc[:,0:1024],newtraindata)
reduced_test=Reduced_Data(Test.iloc[:,0:1024],newtraindata)


# In[149]:


import cv2
import matplotlib.pyplot as plt
row,col=Eigen_PCA.shape
newtraindata1=newtraindata
newtraindata1=newtraindata1.values.real
newtraindata1=pd.DataFrame(newtraindata1)
for i in range(1,17):
    a=np.array(Eigen_PCA[i]).reshape(32,32)
    plt.subplot(4,4,i)
    plt.imshow(a,cmap='gray')
plt.show()


# In[137]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train.iloc[:,0:1024])/5)
train_fold= list(divide_chunks((reduced_dim.values.tolist()), n)) 
newlabel = list(divide_chunks(list(Train['Label']), n)) 


# In[138]:


def accuracy(predicted,actual):
    c=0
    for i in range(len(predicted)):
        if(predicted[i]==actual[i]):
            c=c+1
    s=c/len(predicted)
    return s


# In[154]:


def Normal(Train_Data,Test_Data,TrainLabel,TestLabel):
    predicted,posterior=Classifier(Train_Data,TrainLabel,Test_Data)
    posterior=posterior.transpose()
    Accuracy=accuracy(predicted,TestLabel)
    print(Accuracy)
    labels=['0','1','2','3','4','5','6','7','8','9']
    for i in range(10):
        TPR,FPR=Roc(posterior[i],Test['Label'],i)
        py.title("ROC Curve for 10 classes")
        py.plot(FPR,TPR,label=labels[i])
        py.xlabel("FPR")
        py.ylabel("TPR")
        py.legend()
    py.show()   
    confusion_matrix=ConfusionMatrix(TestLabel,predicted)
    sns.heatmap(confusion_matrix,annot=True)
Normal(Train.iloc[:,0:1024],Test.iloc[:,0:1024],Train['Label'],Test['Label'])


# In[140]:


from sklearn.naive_bayes import GaussianNB
def Classifier(train,trainlabel,test):
        clf = GaussianNB()
        clf.fit(train,trainlabel)
        predicted=clf.predict(test)
        posterior=clf.predict_log_proba(test)
        return predicted,posterior


# In[141]:


import matplotlib.pyplot as py
def Roc(probabilities,testlabel,classes):
        print("class",(classes))
        testlabel=list(testlabel)
        min_prob=min(probabilities)
        max_prob=max(probabilities)
        thresold=np.linspace(min_prob,max_prob,num=100)
        TPR=[]
        FPR=[]
        for i in range(len(thresold)):
            Roc=[]
            tp=0
            fp=0
            tn=0
            fn=0
            total=0
            for j in range(len(probabilities)):
                if(probabilities[j]>=thresold[i]):
                    Roc.append(classes)
                else:
                    Roc.append(classes+1)
            confusion_matrix=ConfusionMatrix(testlabel,Roc)
            for i in range(11):
                for j in range(11):
                    if(i==j==classes):
                        tp=confusion_matrix[i][j]
                    elif(i==classes and j!=classes):
                        fn+=confusion_matrix[i][j]
                    elif(j==classes and i!=classes):
                        fp+=confusion_matrix[i][j]
            
            for i in range(11):
                for j in range(11):
                    total+=confusion_matrix[i][j]
            tn=total-(fp+fn+tp)
            TPR.append(tp/(tp+fn+1))
            FPR.append(fp/(fp+tn+1))
        return TPR,FPR
       


# In[142]:


import seaborn as sns
def ConfusionMatrix(Actual,Predicted):
    confusion_matrix=np.zeros((11,11))
    for i in range(len(Actual)):
        confusion_matrix[int(Actual[i])][int(Predicted[i])]=(confusion_matrix[int(Actual[i])][int(Predicted[i])])+1
    return confusion_matrix


# In[151]:


import seaborn as sns
def Kfold(train_fold,newlabel,testing,testinglabel):
    a=[]
    besttrainingset=[]
    besttrainlabel=[]
    for i in range(5):
        test1=[]
        train1=[]
        actual_test1=[]
        actual_train1=[]
        for j in range(5):
            if(i==j):
                test1=train_fold[i]
                actual_test1=newlabel[i]
            else:
                train1=train1+train_fold[j]
                actual_train1=actual_train1+newlabel[j]
                
        besttrainingset.append(train1)
        besttrainlabel.append(actual_train1)
        predicted,posterior=Classifier(train1,actual_train1,test1)
        Accuracy=accuracy(predicted,actual_test1)
        print(Accuracy)
        a.append(Accuracy)
    mean=np.sum(a)/5
    std=np.std(a)
    print("mean and Standard deviation",mean,std)
    max_accuracy=-100
    index=-1
    for i in range(len(a)):
        if(max_accuracy<a[i]):
            max_accuracy=a[i]
            index=i
    print(max_accuracy)
    print(index)
    TrainingSet=besttrainingset[index]
    TrainingLabel=besttrainlabel[index]
    predicted_test,posterior=Classifier(TrainingSet,TrainingLabel,testing)
   
    posterior=np.transpose(posterior)
    labels=['0','1','2','3','4','5','6','7','8','9']
    for i in range(10):
        TPR,FPR=Roc(posterior[i],Test['Label'],i)
        py.title("ROC Curve")
        py.plot(FPR,TPR,label=labels[i])
        py.xlabel("False Positive Rate")
        py.ylabel("True Positive Rate")
        py.legend()
    py.show()    
    confusion_matrix=ConfusionMatrix(testinglabel,predicted_test)
    sns.heatmap(confusion_matrix,annot=True)
    Accuracy=accuracy(testinglabel,predicted_test)
    print(Accuracy)
Kfold(train_fold,newlabel,reduced_test,Test['Label'])          


# In[78]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)  
inbuilt_lda=lda.fit_transform(Train.iloc[:,0:1024],Train['Label'])


# In[79]:


inbuilt_lda=pd.DataFrame(inbuilt_lda)
inbuilt_lda=pd.concat([inbuilt_lda, Train['Label']],axis=1)


# In[108]:


from numpy.linalg import inv
def LDA(lda_data,k):
    data=lda_data.drop(['Label'],axis=1)
    LDA=pd.DataFrame()
    inter_lda=pd.DataFrame()
    df=pd.DataFrame()
    mean1=data.mean()
    mean1=pd.DataFrame(mean1)
    mean1=mean1.T
    classwise_mean=lda_data.groupby(['Label']).mean()
    lda=lda_data.groupby(['Label'])
    IntraClass_Scatter=np.zeros((k,k))
    InterClass_Scatter=np.zeros((k,k))
    c=0
    for i in lda:
        df=i
        df2=df[1].drop(['Label'],axis=1)
        LDA=(df2-classwise_mean.iloc[c])
        LDA_transpose=LDA.T
        sw=np.dot(LDA_transpose.values,LDA.values)
        inter_lda=(classwise_mean.iloc[c]-mean1)

        inter_lda=pd.DataFrame(inter_lda)
        inter_lda_transpose=inter_lda.T
        sb=np.dot(inter_lda_transpose.values,inter_lda.values)
        sb=sb*len(df2)
        IntraClass_Scatter+=sw
        InterClass_Scatter+=sb
        c=c+1
    IntraClass_Scatter=pd.DataFrame(IntraClass_Scatter)
    InterClass_Scatter=pd.DataFrame(InterClass_Scatter)
    IntraClass_Scatter_inv=inv(np.matrix(IntraClass_Scatter.values))
    covariance=np.matmul(IntraClass_Scatter_inv,InterClass_Scatter)
    eigen_value,eigen_vector=np.linalg.eig(covariance)
    for i in range(len(eigen_value)):
        if(eigen_value[i]<0):
            eigen_value[i]=(-1)*eigen_value[i]
    Eigen_Value=pd.DataFrame(eigen_value)
    Eigen_Vector=pd.DataFrame(eigen_vector)
    Eigen_Vector=Eigen_Vector.T
    Eigen=pd.concat([Eigen_Value,Eigen_Vector],axis=1,ignore_index=True)
    Eigen=Eigen.sort_values(by= [0], axis=0, ascending=False, inplace=False)
    Eigen=Eigen.drop([0],axis=1)
    Eigen=Eigen.iloc[0:9]  
    Eigen=Eigen.T
    return Eigen
EigenLDA=LDA(Train,32*32)


# In[84]:


Dimension_Reduction_LDA=Reduced_Data(Train.iloc[:,0:1024],EigenLDA)
Dimension_Reduction_LDA=pd.concat([Dimension_Reduction_LDA, Train['Label']],axis=1)


# In[86]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink','brown']
for i in range(12):
    y=inbuilt_lda[inbuilt_lda['Label']==(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[163]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','pink']
for i in range(12):

    y=Dimension_Reduction_LDA[Dimension_Reduction_LDA['Label']==(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[88]:


Project_LDA=LDA(Train,32*32)
Reduced_LDA_Train=Reduced_Data(Train.iloc[:,0:1024],Project_LDA)
Reduced_LDA_Test=Reduced_Data(Test.iloc[:,0:1024],Project_LDA)


# In[150]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train)/5)
train_Lda= list(divide_chunks((Reduced_LDA_Train.values.tolist()), n))
label_lda = list(divide_chunks(list(Train['Label']), n)) 
Kfold(train_Lda,label_lda,Reduced_LDA_Test,Test['Label']) 


# In[93]:


eigenpca,k=PCA(Train.iloc[:,0:1024],95)
Dimension_Train_PCA=Reduced_Data(Train.iloc[:,0:1024],eigenpca)
Dimension_Test_PCA=Reduced_Data(Test.iloc[:,0:1024],eigenpca)


# In[94]:


Dimension_Train_PCA=Dimension_Train_PCA.values.real


# In[95]:


Dimension_Train_PCA=pd.DataFrame(Dimension_Train_PCA)
print(type(Dimension_Train_PCA))


# In[96]:


data=pd.concat([Dimension_Train_PCA,Train['Label']],axis=1)


# In[97]:


eigenlda=LDA(pd.DataFrame(data),k)
reduced_train_lda=Reduced_Data(Dimension_Train_PCA,eigenlda)
reduced_test_lda=Reduced_Data(Dimension_Test_PCA,eigenlda)


# In[152]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train)/5)
train_Lda= list(divide_chunks((reduced_train_lda.values.tolist()), n)) 
label_lda = list(divide_chunks(list(Train['Label']), n)) 
Kfold(train_Lda,label_lda,reduced_test_lda,Test['Label'])


# In[102]:


eigen_lda=LDA(Train,32*32)
Red_Dim_Train_LDA=Reduced_Data(Train.iloc[:,0:1024],eigen_lda)
Red_Dim_Test_LDA=Reduced_Data(Test.iloc[:,0:1024],eigen_lda)
eigen_pca,k=PCA(Red_Dim_Train_LDA,99)
Red_Dim_Train_PCA=Reduced_Data(Red_Dim_Train_LDA,eigen_pca)
Red_Dim_Test_PCA=Reduced_Data(Red_Dim_Test_LDA,eigen_pca)


# In[153]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train)/5)
train_pca= list(divide_chunks((Red_Dim_Train_PCA.values.tolist()), n)) 
label_pca = list(divide_chunks(list(Train['Label']), n)) 
Kfold(train_pca,label_pca,Red_Dim_Test_PCA,Test['Label'])


# In[ ]:




