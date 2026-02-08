function theta_deg = angle_between_axes(u, v)
dot_product = dot(u, v);
norm_u = norm(u);
norm_v = norm(v);
cos_theta = dot_product / (norm_u * norm_v);
theta_rad = acos(cos_theta);
theta_deg = real(rad2deg(theta_rad));
end