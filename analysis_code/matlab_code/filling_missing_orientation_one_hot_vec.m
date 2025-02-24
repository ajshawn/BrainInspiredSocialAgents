
function bv_data = filling_missing_orientation_one_hot_vec(bv_data)
  for orientation_id = 0:1
    orientation_cols_prefix = sprintf('orientations_%d_', orientation_id);
    if ~all(ismember(strcat(orientation_cols_prefix, string(0:3)), bv_data.Properties.VariableNames))
      % Extract "orientation_X" field if available
      orientation_col = sprintf('ORIENTATION_%d', orientation_id);
      if ismember(orientation_col, bv_data.Properties.VariableNames)
        % Decode "orientation_X" into four binary columns
        orientation_values = table2array(bv_data(:, orientation_col));
        orientation_binary = zeros(height(bv_data), 4);
        for i = 0:3
          orientation_binary(:, i + 1) = (orientation_values == i);
        end
        % Add the new binary columns to the table
        new_orientation_cols = array2table(orientation_binary, ...
          'VariableNames', strcat(orientation_cols_prefix, string(0:3)));
        bv_data = [bv_data, new_orientation_cols];
      else
        error(['Neither "', orientation_cols_prefix, '*" nor "', orientation_col, '" found in bv_data.']);
      end
    end
  end
end