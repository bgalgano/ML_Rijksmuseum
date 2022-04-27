%Get labels
load('rijksgt.mat');
labels = gt.C;
names = gt.Cnames;
writematrix(labels,'labels.txt','Delimiter',',');
writecell(names,'names.txt','Delimiter','tab');