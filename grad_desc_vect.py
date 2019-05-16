import csv
import numpy as np
import matplotlib.pyplot as plt

c_name = "pnb7"

with open('./'+c_name+'_s.csv', 'r') as csvFile:             #Open the CSV file containing traffic data
 data = list(csv.reader(csvFile,delimiter=','))


lambda_val = 0.01

training_data_len = int(((len(data)-1)*90)/100)         #Calculate the length of 80% of the training data
reporting_data_len = int(((len(data)-1)*10)/100)        #Calculate the length of 20% of the data to be used for validation

mat_X = np.ones((len(data),5))        # Training Data
mat_Y = np.zeros((len(data)))       # Output Variable


for i in range(len(data)-1):                            #Load the imported data from csv in the form of a matrix of input parameters
    for j in range(len(data[0])-2):
        mat_X[i][j] = float(data[i+1][j+1])


for i in range(len(data)-1):                            #Load data in matrix of output variables
    mat_Y[i] = float(data[i+1][len(data[0])-1])


mat_X_train = mat_X[0:training_data_len:1]                      #Load 80% training data in a matrix
mat_X_report = mat_X[training_data_len:training_data_len+reporting_data_len:1] #20% data to be used for reporting

mat_Y_train = mat_Y[0:training_data_len:1]                         #80% Output variable values for training
mat_Y_report = mat_Y[training_data_len:training_data_len+reporting_data_len:1] #20% Output variable values for validation


beta_hypo = np.zeros(5)        #Create matrix for Hypothesis

learn_rate = 0.000002
m = len(mat_X_train)

print("Shape of Y")
print(np.shape(mat_Y_train))


for i in range(100000):                  #Run 100000 iteration for gradient descent
        temp1 = (np.matmul(beta_hypo,np.transpose(mat_X_train))-mat_Y_train)  #Intermediate expression temp1 and temp2 for grad desc equation
        temp2 = (np.matmul(np.transpose(mat_X_train),temp1))
        beta_hypo = (1-(learn_rate*lambda_val/m))*beta_hypo - (learn_rate/m)*temp2


report_var = np.matmul(beta_hypo,np.transpose(mat_X_report)) #Calculate output values 

x = ([range(len(mat_Y_report))])            # makeindex vector for plotting data
Index = np.reshape(x,(len(mat_Y_report),))

mat_Y_predicted = (np.matmul(beta_hypo,np.transpose(mat_X_report)))
y = np.reshape(mat_Y_predicted,(len(mat_Y_predicted),)) #Reshape for plotting

plt.plot(Index,y,label='Predicted')         #Plot the values of predicted and actual values 
plt.plot(Index,mat_Y_report,label='Actual')
plt.xlabel("Training example No.")
plt.ylabel("Output Value")
plt.title(c_name)
plt.legend()
plt.show()

#y_diff = mat_Y_predicted - mat_Y_report          #Evaluate mean square error
#sum_of_squares = np.matmul(y_diff,np.transpose(y_diff))
#rmse = np.sqrt(sum_of_squares/len(mat_Y_predicted))


y_mean = np.mean(mat_Y_report)
from sklearn.metrics import mean_absolute_error
print("Value of Root mean square error is :")
print(mean_absolute_error(mat_Y_predicted,mat_Y_report))
p_error = (mean_absolute_error(mat_Y_predicted,mat_Y_report))/y_mean

pp_error = p_error*100

print("Percentage error:")
print(pp_error)
