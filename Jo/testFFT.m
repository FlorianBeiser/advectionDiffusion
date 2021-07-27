function res = testFFT;

% Size of quadratic grid cells
delta=1;
min_north=2150;
min_east=4000;
max_north=2270;
max_east=4100;
domain=[min_east max_east;min_north max_north];
% Grid size 
n1=ceil(max_east-min_east)/delta;
n2=ceil(max_north-min_north)/delta;
nv=[n1;n2];
n=n1*n2;

sig=1;  % standard deviation
nu=50;  % Range in m
ksi=0.1; % nugget


% Construct toeplitz (see separate file below)
[cm,cmf]=toep_circ(sig,nu,delta,nv,ksi);

nd=[1:n]';
nd2=ceil(nd/n1);
nd1=nd-(nd2-1)*n1;

n=n1*n2;
Sigma=zeros(n,n);
% Set the B matrix from correlation structure and measurement nodes
for i=1:size(nd,1),
  for j=i:size(nd,1),
    Sigma(i,j)=cm(1+mod2(nd1(j,1),nd1(i,1),n1),1+mod2(nd2(j,1),nd2(i,1),n2));
    if (j>i)
      Sigma(j,i)=Sigma(i,j);
    end;
  end;
end;

figure(1);
clf;
imagesc(Sigma); 

uv=randn(n,1);  
u=zeros(n1,n2);
u=vec2mat(uv,n1);

xf=real(fft2(cmf.*ifft2(u))); % Spatial part, Fourier domain
xfv=mat2vec(xf);

xf2=Sigma*uv;

figure(2);
clf;
plot(xfv-xf2);

figure(3); 
clf;
imagesc(xf); colorbar;

figure(4);
clf;
imagesc(vec2mat(xf2,n1)); colorbar;


figure(5);
clf;
plot(Sigma(1,:));

figure(6);
clf;
res = eig(Sigma);
plot(res);

g=1;

