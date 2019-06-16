function [p,JJ,kss]=f_logistic_zsgd_batch(p,zfit,yfit,lam,rate,stopdJ,maxkss)
%>>批式随机梯度下降法求解  比非批式算法快一倍
[n,m]=size(zfit);
K=floor(n/10);%
JJ=[];
h=@(p,z)1./(1+exp(-z*p(:)));
g=@(p,z,y)-y(:).*log(h(p,z))-(1-y(:)).*log(1-h(p,z));
if nargin<=6;maxkss=200;end;
for kss=1:maxkss;%遍历整个训练样本集200次
    %打乱样本排序对于算法的随机收敛性非常重要!!!
    li=randperm(n);grad=zeros(1,m); i=0;
    while i<n;
        indk=li(i+1:min(i+K,n));i=i+length(indk);
        zk=zfit(indk,:); yk=yfit(indk);
        gradk=-(yk-h(p,zk))'*zk+2*length(indk)*lam(:)'.*p(:)';
        p=p-rate*gradk;
    end
    if kss==1;fk0=0;else fk0=fk;end;
    fk=sum(g(p,zfit,yfit))+n*sum(lam(:).*p(:).^2);
    if fk>fk0;rate=0.8*rate;end; JJ=[JJ,fk]
    if abs(fk-fk0)<=stopdJ;;'----stop criterior is fulfilled ,stop running----',break; end;
end;


function []=test()
load('x_logistic.mat');%导入dummy变量
load('x_gbk_train.mat');
s=1:3e4;X=x_logistic(s,1:400);Y=x_gbk_train(s,end-1);
Nt=floor(2/3*size(X,1));Xt=X(1:Nt,:);Yt=Y(1:Nt);
Xa=X(Nt+1:end,:);Ya=Y(Nt+1:end);
p=zeros(1,size(Xt,2));rate=0.01;stopdJ=1e-3;lam=1e-4;maxkss=200;
% tic,[p,JJ,kss]=f_logistic_zsgd(p,Xt,Yt,lam,rate,stopdJ,maxkss),toc
tic,[p,JJ,kss]=f_logistic_zsgd_batch(p,Xt,Yt,lam,rate,stopdJ,maxkss),toc
h=@(p,z)1./(1+exp(-z*p(:)));[sY,sN]=ROC(Xa,Ya,h(p,Xa));-trapz(sN,sY)
