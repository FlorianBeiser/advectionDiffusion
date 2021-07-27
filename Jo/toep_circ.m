%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up the Toeplitz matrix 
%%%%%%%%%%%%%%%%%%%%
function [cm,cmf]=toep_circ(sig,phi,delta,nv,ksi);

sig2=sig*sig;% Prior variance
ksi2=ksi*ksi; % Hyper prior variance
% Base of Toeplitz circular matrix
n1=nv(1,1); n2=nv(2,1);
cm=zeros(n1,n2);
cmf=zeros(n1,n2);
for i1=0:(n1-1),
  tau1=delta*i1;
  if (i1>=(n1-1)/2)
    tau1=delta*(n1-i1);
  end;
  for j1=0:(n2-1),
    tau2=delta*j1;
    if (j1>=(n2-1)/2)
      tau2=delta*(n2-j1);
    end;
    cm(i1+1,j1+1)=sig2*exp(-3*sqrt(tau1^2+tau2^2)/phi);
  end;
end;
cmf=real(fft2(cm));
cm(1,1)=cm(1,1)+ksi2;
