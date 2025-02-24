function [U,V,S,dU,dV] = getSharedSpace(df1,df2)
%getSharedSpace this function identifies the loadings for shared space that
%maximizes the covariance between df1 and df2
%input:
%df1 : zscored [time x cells]
%df2 : zscored [time x cells]
%output:
%U: time x dims
%V: time x dims
%dU: projected df1 on to U
%dV: projected df2 on to V
covMat = (df1'*df2)/size(df1,1);
[U,S,V]= svd(covMat);
dU = df1*U;
dV = df2*V;
end

