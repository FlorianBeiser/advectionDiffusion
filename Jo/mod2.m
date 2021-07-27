function [dist]=mod2(i,j,n);

%
% i and j are two indeces among (1,....,n)
% dist is the shortest distance when the n numbers (1,..,n)
% are wrapped on a circle
%

maxi=max(i,j);
mini=min(i,j);
dist=min(abs(maxi-mini),abs(maxi-n-mini));
