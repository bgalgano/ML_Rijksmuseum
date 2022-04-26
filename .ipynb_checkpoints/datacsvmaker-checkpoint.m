%Data generator for Final Project Rijksmuseum

%Variables to edit here

testfraction = 0.1; %Fraction of data for test division
valfraction = 0.1; %Fraction of data for validation division

%Load in files
load("rijksFV16.mat");
load("rijksgt.mat");

%Collect labels from struct
Labels = gt.C';
Names = gt.Cnames;

%Randomize Data indices
shuffler = randperm(gt.N);
Labels = Labels(:,shuffler);
X = X(:,shuffler);
Xn = Xn(:,shuffler);

%Find indexes
testidx = round((1-testfraction-valfraction)*112039);
validx = round((1-valfraction)*112039);

%Divide arrays into train, test, validation
Labelstrain = Labels(1:testidx);
Labelstest = Labels((testidx+1):validx);
Labelsval = Labels((validx+1):end);

X_train = X(:,1:testidx);
X_test = X(:,(testidx+1):validx);
X_val = X(:,(validx+1):end);

Xn_train = Xn(:,1:testidx);
Xn_test = Xn(:,(testidx+1):validx);
Xn_val = Xn(:,(validx+1):end);

%Write to Files
writematrix(Labelstrain,'Namekeytrain.txt','Delimiter',',')
writematrix(Labelstest,'Namekeytest.txt','Delimiter',',')
writematrix(Labelsval,'Namekeyval.txt','Delimiter',',')

writematrix(X_train,'Xtrain.txt','Delimiter',',')
writematrix(X_test,'Xtest.txt','Delimiter',',')
writematrix(X_val,'Xval.txt','Delimiter',',')

writematrix(Xn_train,'Xntrain.txt','Delimiter',',')
writematrix(Xn_test,'Xntest.txt','Delimiter',',')
writematrix(Xn_val,'Xnval.txt','Delimiter',',')

writematrix(Names,'rjikNamefile.txt','Delimiter','tab')

