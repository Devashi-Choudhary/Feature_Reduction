#!/usr/bin/env python
# coding: utf-8

# In[58]:


import cv2
import os
import glob
data=[]
label=[]
path='C:/Users/Devashi Jain/Desktop/IIIT-D/SML/Assignment3/Face_data/Face_data/'
for file in glob.iglob(path+'**/*',recursive=True):
    y=file.split('\\')
    if (not os.path.isdir(file)):
        x=file.split('.')
        if(x[len(x)-1]=='pgm' or x[len(x)-1]=='bad'):
                img=cv2.imread(file,0)
                img=cv2.resize(img,(32,32))
                img=img.ravel()
                data.append(img)
                label.append(y[1])


# In[59]:


import pandas as pd
Dataset=pd.DataFrame(data)


# In[60]:


Label=pd.DataFrame(label)
Label.columns=['label']


# In[61]:


Data=pd.concat([Dataset, Label],axis=1)
originaldata=Data.drop(['label'],axis=1)


# In[62]:


def train_test(Data):
    train_data=Data.sample(frac=0.7)
    test_data=Data.drop(train_data.index)
    return train_data,test_data
Train,Test=train_test(Data)


# In[63]:


Train.index=range(len(Train))
Test.index=range(len(Test))
Train_Data=Train.drop(['label'],axis=1)
Test_Data=Test.drop(['label'],axis=1)


# In[64]:


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
#Eigen_PCA,k=PCA(originaldata,95)   


# In[107]:


Eigen_PCA=Eigen_PCA.values.real
Eigen_PCA=pd.DataFrame(Eigen_PCA)
Eigen_PCA


# In[108]:


import cv2
import matplotlib.pyplot as plt
row,col=Eigen_PCA.shape

for i in range(1,33):
    a=np.array(Eigen_PCA[i]).reshape(64,64)
    print(a)
    plt.subplot(4,4,i)
    plt.imshow(a, cmap='gray')
plt.show()


# In[65]:


def Reduced_Data(Original_Data,Eigen_Data):
    print(Original_Data.shape)
    print(Eigen_Data.shape)
#     a=Original_Data.values
#     b=Eigen_Data.values
    c=np.dot(Original_Data.values,Eigen_Data.values)
    Reduced_Data=pd.DataFrame(c)
    return Reduced_Data
Dimension_Reduction_PCA=Reduced_Data(originaldata,Eigen_PCA)
Dimension_Reduction_PCA=pd.concat([Dimension_Reduction_PCA, Data['label']],axis=1)


# In[35]:


from sklearn.decomposition import PCA as pca
lda1 = pca(n_components=2)  
inbuilt_pca=lda1.fit_transform(originaldata)


# In[36]:


inbuilt_pca=pd.DataFrame(inbuilt_pca)
inbuilt_pca=pd.concat([inbuilt_pca, Data['label']],axis=1)


# In[37]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink','brown']
for i in range(12):
    y=inbuilt_pca[inbuilt_pca['label']==str(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[38]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink']
for i in range(11):
    y=Dimension_Reduction_PCA[Dimension_Reduction_PCA['label']==str(i+1)]
    plt.scatter((-1)*y[0],(-1)*y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[126]:


newtraindata,k=PCA(Train_Data,95)
reduced_dim=Reduced_Data(Train_Data,newtraindata)
reduced_test=Reduced_Data(Test_Data,newtraindata)


# In[131]:


print(type(newtraindata))
print(len(newtraindata[0]))
newtraindata[0]


# In[153]:


import cv2
import matplotlib.pyplot as plt
row,col=newtraindata.shape
newtraindata1=newtraindata
newtraindata1=newtraindata1.values.real
newtraindata1=pd.DataFrame(newtraindata1)
f=plt.figure()
f.set_figheight(10)
f.set_figwidth(10)
for i in range(1,17):
    a=np.array(newtraindata1[i].values.reshape(32,32))
    plt.subplot(8,4,i)
    plt.imshow(a, cmap='gray')
plt.show()


# In[142]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train_Data)/5)
train_fold= list(divide_chunks((reduced_dim.values.tolist()), n)) 
newlabel = list(divide_chunks(list(Train['label']), n)) 


# In[98]:


def accuracy(predicted,actual):
    c=0
    for i in range(len(predicted)):
        if(predicted[i]==actual[i]):
            c=c+1
    s=c/len(predicted)
    return s


# In[99]:


import numpy as np
def Normal(Train_Data,Test_Data,TrainLabel,TestLabel):
    predicted,posterior=Classifier(Train_Data,TrainLabel,Test_Data)
    posterior=posterior.transpose()
    Accuracy=accuracy(predicted,TestLabel)
    print(Accuracy)
    labels=['1','10','11','2','3','4','5','6','7','8','9']
    for i in range(11):
        TPR,FPR=Roc(posterior[i],Test['label'],i+1)
        py.title("ROC Curve")
        py.plot(FPR,TPR,label=labels[i])
        py.xlabel("False Positive Rate")
        py.ylabel("True Positive Rate")
        py.legend()
    py.show()   
    confusion_matrix=ConfusionMatrix(TestLabel,predicted)
    sns.heatmap(confusion_matrix,annot=True)
Normal(Train_Data,Test_Data,Train['label'],Test['label'])


# In[100]:


from sklearn.naive_bayes import GaussianNB
def Classifier(train,trainlabel,test):
        clf = GaussianNB()
        clf.fit(train,trainlabel)
        predicted=clf.predict(test)
        posterior=clf.predict_log_proba(test)
        return predicted,posterior


# In[101]:


import matplotlib.pyplot as py
def Roc(probabilities,testlabel,classes):
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
            for i in range(1,12):
                for j in range(1,12):
                    if(i==j==classes):
                        tp=confusion_matrix[i][j]
                    elif(i==classes and j!=classes):
                        fn+=confusion_matrix[i][j]
                    elif(j==classes and i!=classes):
                        fp+=confusion_matrix[i][j]
            
            for i in range(1,12):
                for j in range(1,12):
                    total+=confusion_matrix[i][j]
            tn=total-(fp+fn+tp)
            TPR.append(tp/(tp+fn))
            FPR.append(fp/(fp+tn))
        return TPR,FPR
       


# In[102]:


import seaborn as sns
def ConfusionMatrix(Actual,Predicted):
    confusion_matrix=np.zeros((13,13))
    for i in range(len(Actual)):
        confusion_matrix[int(Actual[i])][int(Predicted[i])]=(confusion_matrix[int(Actual[i])][int(Predicted[i])])+1
    return confusion_matrix


# In[103]:


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
    labels=['1','10','11','2','3','4','5','6','7','8','9']
    for i in range(11):
        TPR,FPR=Roc(posterior[i],testinglabel,(i+1))
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
Kfold(train_fold,newlabel,reduced_test,Test['label'])          


# In[160]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)  
inbuilt_lda=lda.fit_transform(originaldata,Data['label'])


# In[161]:


inbuilt_lda=pd.DataFrame(inbuilt_lda)
inbuilt_lda=pd.concat([inbuilt_lda, Data['label']],axis=1)


# In[163]:


from numpy.linalg import inv
def LDA(lda_data,k):
    data=lda_data.drop(['label'],axis=1)
    LDA=pd.DataFrame()
    inter_lda=pd.DataFrame()
    df=pd.DataFrame()
    mean1=data.mean()
    mean1=pd.DataFrame(mean1)
    mean1=mean1.T
    classwise_mean=lda_data.groupby(['label']).mean()
    lda=lda_data.groupby(['label'])
    IntraClass_Scatter=np.zeros((k,k))
    InterClass_Scatter=np.zeros((k,k))
    c=0
    for i in lda:
        df=i
        df2=df[1].drop(['label'],axis=1)
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
    Eigen=Eigen.iloc[0:10]  
    Eigen=Eigen.T
    return Eigen
EigenLDA=LDA(Data,32*32)


# In[164]:


Dimension_Reduction_LDA=Reduced_Data(originaldata,EigenLDA)
Dimension_Reduction_LDA=pd.concat([Dimension_Reduction_LDA, Data['label']],axis=1)


# In[165]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink','brown']
for i in range(12):
    y=inbuilt_lda[inbuilt_lda['label']==str(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[166]:


import matplotlib.pyplot as plt
c=['Red',"blue",'green','orange','yellow','purple','black','magenta','navy','skyblue','pink','brown']
for i in range(12):

    y=Dimension_Reduction_LDA[Dimension_Reduction_LDA['label']==str(i+1)]
    plt.scatter(y[0],y[1],color=c[i],alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:


Project_LDA=LDA(Train,64*64)
Reduced_LDA_Train=Reduced_Data(Train_Data,Project_LDA)
Reduced_LDA_Test=Reduced_Data(Test_Data,Project_LDA)


# In[ ]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train_Data)/5)
train_Lda= list(divide_chunks((Reduced_LDA_Train.values.tolist()), n))
label_lda = list(divide_chunks(list(Train['label']), n)) 
Kfold(train_Lda,label_lda,Reduced_LDA_Test,Test['label'])  


# In[167]:


eigenpca,k=PCA(Train_Data,95)
Dimension_Train_PCA=Reduced_Data(Train_Data,eigenpca)
Dimension_Test_PCA=Reduced_Data(Test_Data,eigenpca)


# In[168]:


Dimension_Train_PCA=Dimension_Train_PCA.values.real


# In[169]:


print(type(Dimension_Train_PCA))


# In[170]:


Dimension_Train_PCA=pd.DataFrame(Dimension_Train_PCA)
print(type(Dimension_Train_PCA))


# In[171]:


data=pd.concat([Dimension_Train_PCA,Train['label']],axis=1)


# In[172]:


eigenlda=LDA(pd.DataFrame(data),k)
reduced_train_lda=Reduced_Data(Dimension_Train_PCA,eigenlda)
reduced_test_lda=Reduced_Data(Dimension_Test_PCA,eigenlda)


# In[173]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train_Data)/5)
train_Lda= list(divide_chunks((reduced_train_lda.values.tolist()), n)) 
label_lda = list(divide_chunks(list(Train['label']), n)) 
Kfold(train_Lda,label_lda,reduced_test_lda,Test['label'])


# In[176]:


eigen_lda=LDA(Train,32*32)
Red_Dim_Train_LDA=Reduced_Data(Train_Data,eigen_lda)
Red_Dim_Test_LDA=Reduced_Data(Test_Data,eigen_lda)
eigen_pca,k=PCA(Red_Dim_Train_LDA,95)
Red_Dim_Train_PCA=Reduced_Data(Red_Dim_Train_LDA,eigen_pca)
Red_Dim_Test_PCA=Reduced_Data(Red_Dim_Test_LDA,eigen_pca)


# In[177]:


import math
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
n = math.ceil(len(Train_Data)/5)
train_pca= list(divide_chunks((Red_Dim_Train_PCA.values.tolist()), n)) 
label_pca = list(divide_chunks(list(Train['label']), n)) 
Kfold(train_pca,label_pca,Red_Dim_Test_PCA,Test['label'])


# In[ ]:




