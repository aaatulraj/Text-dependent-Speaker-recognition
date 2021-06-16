clc;
clear all;
close all;
 
k = 16;
for i =1:4
 [y, fs] = audioread(append('A' , num2str(i), '.wav'));
 v = mfcc(y, fs);
 code{i} = codebooks(v, k);
end
 
testing(code);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function testing(code)
cnt=0;
totalcnt=0;
for k = 1:4 % read test sound file of each speaker
 [s, fs] = audioread(append('T' , num2str(k), '.wav'));
 
 v = mfcc(s, fs); % Compute MFCC's
 
 distmin = inf;
 k1 = 0;
 
 for l = 1:length(code) % each trained codebook, compute distortion
 d = euc_dist(v, code{l});
 dist = sum(min(d,[],2)) / size(d,1);
 fprintf("distance between speakar %d and speakar %d is %d \n",k,l,dist);
 if dist < distmin
 distmin = dist;
 k1 = l;
 end 
 end
 
 
 if(k==k1)
 cnt=cnt+1;
 end
 totalcnt=totalcnt+1;
 
 msg = sprintf('Speaker %d matches with speaker %d', k, k1);
 disp(msg);
 
end
 fprintf("accuracy is %f",cnt/totalcnt);
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
function r = codebooks(d,k)

 
e = .01;
r = mean(d, 2);%column vector containing the mean of each row
dpr = 10000;
 
for i = 1:log2(k)
 r = [r*(1+e), r*(1-e)];
 
 while (1 == 1)
 z = euc_dist(d, r);
 [m,ind] = min(z, [], 2);%column vector containing the minimum value of each row
 t = 0;
 for j = 1:2^i
 r(:, j) = mean(d(:, find(ind == j)), 2);
 x = euc_dist(d(:, find(ind == j)), r(:, j));
 for q = 1:length(x)
 t = t + x(q);
 end
 end
 if (((dpr - t)/t) < e) 
 break;
 else
 dpr = t;
 end
 end 
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function d = euc_dist(x, y)
%The Euclidean distance D between two vectors X and Y is:
%D = sum((x-y).^2).^0.5
[R1, C1] = size(x);
[R2, C2] = size(y); 
 
if (R1 ~= R2)
 error('Matrix dimensions do not match.')
end
 
d = zeros(C1, C2);
 
if (C1 < C2)
 extra = zeros(1,C2);
 for n = 1:C1
 d(n,:) = sum((x(:, n+extra) - y) .^2, 1);
 end
else
 extra = zeros(1,C1);
 for p = 1:C2
 d(:,p) = sum((x - y(:, p+extra)) .^2, 1)';
 end
end
 
d = d.^0.5;
end
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mfcc_final=mfcc(y,fs)

 %% PRE EMPHASIS
 filcoeff=[1 -0.95];
 y1=filter(filcoeff,1,y);
 
 %% FRAME BLOCKING
 
 data=zeros(99,fs*0.02);
 Y1=reshape(y1,1,fs);
 for i=1:40
 yframe=Y1((fs/100)*i-((fs/100)-1):(fs/100)*i+(fs/100));
 data(i,:)=yframe;
 end
 
 
 
 % HAMMING WINDOW
 hamming=zeros(1,fs*0.02);
 for i=0:fs*0.02-1
 hamming(1,i+1)=0.54-(0.46*cos(2*pi*i/(fs*0.02-1)));
 end
 
 new_data=zeros(99,fs*0.02);
 for i=1:99
 new_data(i,:)=data(i,:).*hamming;
 end
 
 
 % FAST FOURIER TRANSFORM
 fft_data=zeros(99,fs*0.02);
 for i=1:99
 fft_data(i,:)=abs(fft(new_data(i,:)));
 end 
 fft_freq=(0:(fs*0.02)-1)*(fs/(fs*0.02));
 
 
 
 % MEL FILTER BANK
 
 f_low=0;
 f_high=fs/2;
 filt_num=40;
 
 %computing band in mel-scale
 mel_low=1127*log(1+(f_low/700));
 mel_high=1127*log(1+(f_high/700));
 %creating the mel-scaled vector
 Mel = linspace(mel_low,mel_high,filt_num+2);
 %computing frequencies of the Mel vector
 melc=floor((700*exp(Mel/1127))-700);
 
 %convert frequencies to nearest bins
 H=zeros(40,fs*0.02);
 freq=linspace(0,fs/2,fs*0.02);
 for i=1:40
 for j=1:fs*0.02
 if(freq(1,j)<melc(1,i))
 H(i,j)=0;
 elseif(freq(1,j)<melc(1,i+1)&&freq(1,j)>melc(1,i))
 H(i,j)=2*(freq(1,j)-melc(1,i))/((melc(1,i+2)-melc(1,i))*(melc(1,i+1)-melc(1,i)));
 elseif(freq(1,j)<melc(1,i+2)&&freq(1,j)>melc(1,i+1))
 H(i,j)=2*(freq(1,j)-melc(1,i+2))/((melc(1,i+2)-melc(1,i))*(melc(1,i+1)-melc(1,i+2)));
 elseif(freq(1,j)>melc(1,i+2))
 H(i,j)=0;
 end
 end
 end
 
 %the mel filter bank plot 
 fft_freq=(0:(fs*0.02)-1)*(fs/(fs*0.02));
 figure(1)
 hold on 
 for i=1:40
 plot(fft_freq,H(i,:));
 end
 hold off
 
 
 mel_final=zeros(99,fs*0.02);
 for i=1:99
 for j=1:40
 mel_final(i,:)=mel_final(i,:)+fft_data(i,:).*H(j,:);
 end
 end
 
 
 
 
 % DCT
 
 final=zeros(99,fs*0.02);
 for i=1:99
 final(i,:)=dct2(mel_final(i,:));
 end
 mfcc_final=final(:,1:14);

end