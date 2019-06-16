%%
%%读取数据，并在整个数据集上训练逻辑回归模型，得到基础预测函数
load('x_gbk_train.mat');S=2.5e4;Y=x_gbk_train(1:S,end-1);Ya=x_gbk_train(S+1:end,end-1); 
load('X0328.mat');X=X0328(1:S,:);Xa=X0328(S+1:end,:);   clear X0328; 
load('X0328t.mat');Xtest=X0328t;clear X0328t; 

%删除缺失值太多的样本
h=[];
for i=1:S;
    xi=x0(i,:);h(i)=sum(xi==-1);
end;
[~,ic]=sort(h,'descend');
plot(h(ic))
ip=ic(1:5);
X1=X;Y1=Y;
X1(ip,:)=[];Y1(ip)=[];
%训练模型，基于SGD算法；通过局部测试集优选迭代次数
N=length(Y1);li=randperm(N);nt=floor(N*4/5);c=li(1:nt);c1=setdiff(li,c);
maxkss=20;lam=1e-4;rate=1e-4;
pc=[];
[pc,cv2,ct2,pp]=f_logistic_zsgd_batch(pc,X1(c,:),Y1(c),lam,rate,0,maxkss,X1(c1,:),Y1(c1));
[~,i]=max(cv2);pa=pp(i,:);
g=@(p,z)1./(1+exp(-z*p(:)));
ya=g(pa,Xa);ROC5(Ya,ya)
%%
%用bagging方法进行逻辑回归，通过袋外数据优选迭代次数，基于局部测试集排序集成回归结果
cya=[];cp=[];
X1=X;Y1=Y;
for kss=1:20;
    N=length(Y1);li=randperm(N);nt=floor(N*4/5);c=li(1:nt);c1=setdiff(li,c);
    if kss==1;pc=[];else pc=[];end;
    if kss==1;maxkss=20;else maxkss=20;end;
    lami=1e-4;
    [pc1,cv2,ct2,pp]=f_logistic_zsgd_batch(pc,X1(c,:),Y1(c),lami,rate,0,maxkss,X1(c1,:),Y1(c1));
    [~,i]=max(cv2);pi=pp(i,:);
    cp=[cp;pi(:)'];
end;
cp1=cp;
k=5;cv=[];cya=[];
xs=Xq;ys=Yq;
auc=[];
for i=1:size(cp1,1);
    pi=cp1(i,:);
    cv(i)=ROC5(ys,xs*pi(:));
    [~,ic]=sort(cv,'descend');
    ya=Xa*pi(:);ya=f_1(ya,7);cya=[cya,ya];
    yam=mean(cya(:,ic(1:min(length(ic),k))),2);
    auc(i)=ROC5(Ya,yam)
end;
kk=1:size(cp1,1);
plot(kk,auc)
[~,ic]=sort(cv,'descend');
icb=ic(1:min(length(ic),k));
yam=mean(cya(:,icb),2);
ROC5(Ya,yam)
pm=mean(cp(icb,:),1);
ya=Xa*pm(:);ROC5(Ya,ya)
pa=pm;
%%
%分割训练数据
[N,m]=size(X);
li=randperm(N);
nq=000;nt=floor(1/2*(N-nq));
sli1=li(1:nt);
sli3=li(nt+1:nt*2);
sli2=li(nt*2+1:N);
Xt=X(sli1,:);Yt=Y(sli1);
Xv=X(sli3,:);Yv=Y(sli3);
Xq=X(sli2,:);Yq=Y(sli2);
%%
%两组训练数据上分别产生回归结果
g=@(p,z)1./(1+exp(-z*p(:)));
pc=[];lam=1e-4;rate=0.0001;maxkss=20;
N=length(Yt);li=randperm(N);nt=floor(N*4/5);c=li(1:nt);c1=setdiff(li,c);
[pc1,cv2,ct2,pp]=f_logistic_zsgd_batch(pc,Xt(c,:),Yt(c),lam,rate,0,maxkss,Xt(c1,:),Yt(c1));
[~,i]=max(cv2);p1=pp(i,:);
ya=g(p1,Xa);ROC5(Ya,ya)

pc=[];
N=length(Yv);li=randperm(N);nt=floor(N*4/5);c=li(1:nt);c1=setdiff(li,c);
[pc1,cv2,ct2,pp]=f_logistic_zsgd_batch(pc,Xv(c,:),Yv(c),lam,rate,0,maxkss,Xv(c1,:),Yv(c1));
[~,i]=max(cv2);p2=pp(i,:);
ya=g(p2,Xa);ROC5(Ya,ya)
 
%%
%写入私有测试集数据，为在python中处理准备数据
load('x0.mat');
xa=x0(S+1:end,:);
Yam=g(pa,Xa);ROC5(Ya,Yam)
csvwrite('xa.csv',full(xa));
csvwrite('Yam.csv',full(Yam));
csvwrite('Ya.csv',full(Ya));
%写入GBDT训练数据
x=x0([sli3,sli1],:);
y=[Yv;Yt];
y0=[g(p1,Xv);g(p2,Xt)];
csvwrite('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/x.csv',full(x));
csvwrite('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/y.csv',full(y));
csvwrite('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/y0.csv',full(y0));
%%
%利用所有的训练数据进行一次寻优，以Xa来选择最优迭代次数
N=length(Y);li=randperm(N);nt=floor(N*5/5);c=li(1:nt);c1=setdiff(li,c);
maxkss=50;lam=1e-4;rate=1e-4;
pc=[];
[pc,cv2,ct2,pp]=f_logistic_zsgd_batch(pc,X(c,:),Y(c),lam,rate,0,maxkss,Xa,Ya);
[~,i]=max(cv2);pa=pp(i,:);
ya=g(pa,Xa);ROC5(Ya,ya)
% save('pa.mat','pa');
%%
%预测，并将结果写入csv文件
load('x_gbk_test.mat')
uid=x_gbk_test(:,1);
load('x0t.mat');
load('X0328t.mat');X0t=X0328t;clear X0328t
ytest0=g(pa,X0t);
%数据写成csv供python调用
csvwrite('ytest0.csv',ytest0);
csvwrite('x0t.csv',x0t);
csvwrite('uid.csv',uid);
%在python中完成最终的预测工作




