%% add paths
mydir = pwd;
idcs = strfind(mydir,'/');
above_dir = mydir(1:idcs(end)-1);
addpath([above_dir '/functions']);

%% observational data
X = readtable('data_alarm.txt'); X = table2array(X); 
[gamma1, lambda1, B1, topo_sort1] = sa_wrapper(X);

coef0 = table2array(readtable('adjMat_initial.txt'));
Pini = flip(toposort(digraph(coef0)));
[gamma2, lambda2, B2, topo_sort2] = sa_wrapper(X, 'Pini', Pini);

%% interventional data
X_intv = readtable('full_intv_data.txt'); X_intv = table2array(X_intv);
X_ind = readtable('full_intv_data_ind.txt'); X_ind = table2array(X_ind);
[gamma3, lambda3, B3, topo_sort3] = sa_wrapper(X_intv, 'X_ind', X_ind, 'Pini', Pini)?