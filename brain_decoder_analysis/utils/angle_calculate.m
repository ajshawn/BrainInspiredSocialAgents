function [angles,angle_plane_1,angle_plane_2,pseudo_deg,npc] = angle_calculate(x,y)
axis = []; angles = [];
for i = 1:size(y,2)
    [coeff,score,~,~,explained,mu] = pca(x);
    npc = find(cumsum(explained)>80,1);
    md = fitlm(score(:,1:npc),y(:,i));
    axis(:,i) = coeff(:,1:npc)*table2array(md.Coefficients(2:end,1));
end
for i = 1:size(y,2)
    for j = i:size(y,2)
        angles(i,j) = angle_between_axes(axis(:,i), axis(:,j));
    end
end
for i = 1:3 % angle between self and allo/ego partner
    for j = i:3
        [theta_deg,pseudo_deg(i,j)] = plane_angles(axis(:,(i-1)*2+1), axis(:,(i-1)*2+2), axis(:,(j-1)*2+1), axis(:,(j-1)*2+2));
        angle_plane_1(i,j) = theta_deg(1);
        angle_plane_2(i,j) = theta_deg(2);
    end
end
end