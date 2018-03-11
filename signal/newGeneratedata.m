clear;clc;
step = 16;  %�����С
%��������
M1 = 64*16;  %�ź�x�еĳ���һ��1������ΪM1��
M2 = 16;   %����z�еĳ���һ��1������ΪM2��
NP = 1000;  %���������
N = NP*step;   %��������
r = Datalocation(M1,N,2000,1000);
s = Datalocation(M2,N,100,300);
x = zeros(1,N);
z = zeros(1,N);
for t = 1:length(r)
    x(r(t):r(t)+M1-1) = 1;
end

for j = 1:length(s)
    z(s(j):s(j)+M2-1) = 1;
end
x = x';z = z';
SNR = -1.5;
noise = (10^(-SNR/20))*(randn(N,1)+i*randn(N,1))/sqrt(2);
h1 = exp(i*2*pi*rand(1));
h2 = exp(i*2*pi*rand(1));
y = h1*x + h2*z + noise;

Sample = zeros(NP,step);
Label = zeros(NP,1);
for j = 1:NP
    Sample(j,:) = y((j-1)*step+1:j*step);
    if x((j-1)*step+1:j*step) == ones(1,step)
        Label(j) = 1;   %ȫ�ź�x�����Ϊ1
    elseif x((j-1)*step+1:j*step) == zeros(1,step)
        Label(j) = 0;   %���ź�x�����Ϊ0
    else
        Label(j) = 2;   %�������ź�x�����Ϊ2
    end
end

 xlswrite('Sample.xlsx',Sample);
 xlswrite('Label.xlsx',Label);
