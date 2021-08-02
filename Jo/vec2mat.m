function m=vec2mat(v,n1);

n2=size(v,1)/n1;
m=reshape(v,n1,n2);
