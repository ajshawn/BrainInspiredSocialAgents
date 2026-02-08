function features = extract_features(track_self,track_other)
% body x/y
rel_x = []; rel_y = [];
for f = 1:size(track_self,1)
    nose = squeeze(track_self(f,1,:));
    othernose = squeeze(track_other(f,1,:));
    head = squeeze(track_self(f,2,:));
    otherhead = squeeze(track_other(f,2,:));
    rel_x(f) = nose(1) - othernose(1);
    rel_y(f) = nose(2) - othernose(2);
    distance(f) = sqrt(rel_x(f)^2 + rel_y(f)^2);
    %[rel_x_ang(f),rel_y_ang(f)] = point_to_line_xy(squeeze(track_other(f,6,:))', nose, head);
end
%features = [squeeze(track_self(:,1,:)),squeeze(track_other(:,1,:)),rel_x',rel_y',rel_x_ang',rel_y_ang'];
features = [squeeze(track_self(:,1,:)),squeeze(track_other(:,1,:)),rel_x',rel_y',distance'];
end