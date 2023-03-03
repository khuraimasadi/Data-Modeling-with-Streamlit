from pyforest import*
import streamlit as st
from sklearn import datasets as dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
st.title('Machine Learning Model Builder')
from PIL import Image
image=Image.open('download.png')
st.image(image,use_column_width=True)

st.write("""
	## A Simple Data App With Streamlit
	""")

st.write("""
	##### Let,s explore different classifiers and datasets
	""")

dataset_name=st.sidebar.selectbox('Select dataset',('Breast Cancer','Iris','Wine'))
classifier_name=st.sidebar.selectbox('Select classifier',('KNN','SVM','Logistic Regression'))

def get_dataset(name):
	data=None
	if name=='Iris':
		data=dataset.load_iris()
	elif name=='Wine':
	    data=dataset.load_wine()
	else:     
	    data=dataset.load_breast_cancer() 
	x=data.data
	y=data.target
	return x,y
x,y=get_dataset(dataset_name)	
st.dataframe(x)   
st.write('Shape of your dataset: ',x.shape) 
st.write('Unique target variables:',len(np.unique(y)))   

fig=plt.figure()
sns.boxplot(data=x,orient='h')	
st.pyplot(fig)

fig=plt.figure()
plt.hist(x)
st.pyplot(fig)

def add_parameter(name):
	params=dict()
	if name=='KNN':
		k=st.sidebar.slider('Neighbors',1,50,5)
		params['k']=k
	elif name=='SVM':
	    c=st.sidebar.slider('C',1,100,5)
	    params['c']=c
	return params    
params=add_parameter(classifier_name)	


def get_classifier(name,params):
	model=None
	if name=='KNN':
		model=KNeighborsClassifier(n_neighbors=params['k'])
	elif name=='SVM':
	    model=SVC(C=params['c'])
	else:
	    model=LogisticRegression()
	    return model  

model=get_classifier(classifier_name,params)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)	 
model.fit(x_train,y_train) 
score=model.score(x_test,y_test) 
st.write('Classifier name: ',classifier_name)  
st.write('Model performance: {:2f}%'.format(score*100))  