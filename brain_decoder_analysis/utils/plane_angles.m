function [theta_deg,pseudo_deg] = plane_angles(v1, u1, v2, u2)
% plane_angles: principal angles between two planes
% Plane 1 = span(v1, u1)
% Plane 2 = span(v2, u2)
% Form basis matrices
B1 = [v1(:), u1(:)];  % ensure column vectors
B2 = [v2(:), u2(:)];
% Orthonormal bases via reduced QR
[Q1, ~] = qr(B1, 0);
[Q2, ~] = qr(B2, 0);
% Compute matrix of inner products
C = Q1' * Q2;   % 2x2
% Singular values of C are cosines of principal angles
s = svd(C);
% Numerical safety: clamp to [-1, 1]
s = max(min(s, 1), -1);
% Principal angles in radians
theta = acos(s);
% Convert to degrees
theta_deg = rad2deg(theta);
% pseudo-angle
theta = acos(s(1)*s(2));
pseudo_deg = rad2deg(theta);
end