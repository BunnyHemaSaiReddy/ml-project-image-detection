import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os,cv2
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
        
dir_name= r"C:\Users\user\OneDrive\Pictures\Screenshots"
y=[];X=[];target_names=[]
person_id=0;h=w=300
n_samples=0
class_names=[]
for person_name in os.listdir(r"C:\Users\user\OneDrive\Documents\Pictures\Screenshots"):
  if person_name !='desktop.ini':
    print(person_name)
    dir_path = dir_name+'\\'+person_name
    class_names.append(person_name)
    print(dir_path,'\n','----------------------')
    for image_name in os.listdir(os.path.join(r"C:\Users\user\OneDrive\Documents\Pictures\Screenshots",person_name)):
     try:
        # formulate the image path
        image_path = os.path.join(os.path.join(r"C:\Users\user\OneDrive\Documents\Pictures\Screenshots",person_name),image_name)
        # Read the input image
        print(image_path)
        img = cv2.imread(image_path)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image to 300*300 dimension
        resized_image= cv2.resize(gray,(h,w))
        # convert matrix to vector
        v = resized_image.flatten()
        X.append(v)
        # increase the number of samples
        n_samples =n_samples+1
        # Addinng th categorical label
        y.append(person_id)
        # adding the person name
        target_names.append(person_name)
        # Increase the person id by 1
     except Exception as e:
      continue
    person_id=person_id+1
    
# ###################################
    
y=np.array(y)
X=np.array(X)
print(X)
target_names =np.array(target_names)
n_features = X.shape[1]
print(y.shape,X.shape,target_names.shape)
print("Number of sampels:",n_samples)
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 60
print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))
# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
print("Projecting the input data on the eigenfaces orthonormal basis")


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(X_train_pca.shape,X_test_pca.shape)


# %%Compute Fisherfaces
lda = LinearDiscriminantAnalysis()

print(len(X_train_pca), len(y_train))
#Compute LDA of reduced data
lda.fit(X_train_pca,y_train)

print('----')
X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)
print("Project done...")



#clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10),max_iter=1000, verbose=True).fit(X_train_lda, y_train)
clf=DecisionTreeClassifier().fit(X_train_lda, y_train)
print("Model Weights:")
#model_info = [coef.shape for coef in clf.coefs_]
#print(model_info)


y_pred=[];y_prob=[]
for test_face in X_test_lda:
    prob = clf.predict_proba([test_face])[0]
    # print(prob,np.max(prob))
    class_id = np.where(prob == np.max(prob))[0][0]
    # print(class_index)
    # Find the label of the mathed face
    y_pred.append(class_id)
    y_prob.append(np.max(prob))
    
    
# Transform the data
y_pred = np.array(y_pred)
 
 
 
prediction_titles=[]
true_positive = 0


for i in range(y_pred.shape[0]):
    # print(y_test[i],y_pred[i])
    # true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    # pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]
    result = 'pred: %s, pr: %s \ntrue: %s' % (pred_name, str(y_prob[i])[0:3], true_name)
    # result = 'prediction: %s \ntrue: %s' % (pred_name, true_name)
    prediction_titles.append(result)
    if true_name==pred_name:
        true_positive =true_positive+1
    
    

print("Accuracy:",true_positive*100/y_pred.shape[0])
# # Plot results
plot_gallery(X_test, prediction_titles, h, w)
plt.show()
