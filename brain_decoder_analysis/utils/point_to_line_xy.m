function [rel_x,rel_y] = point_to_line_xy(pt, v1, v2)
% pt: n x 2
% v1 and v2 1x2
v1=v1(:)';
v2=v2(:)';
v1_ = repmat(v1,size(pt,1),1);
v2_ = repmat(v2,size(pt,1),1);
%
a = v1_ - v2_;
b = pt - v2_;
% Relative x
rel_y = dot(a,b,2) ./ vecnorm(a,2,2);
% Relative y
a_perp = [a(:,2), -a(:,1)];
rel_x = dot(b, a_perp, 2) ./ vecnorm(a,2,2);
end