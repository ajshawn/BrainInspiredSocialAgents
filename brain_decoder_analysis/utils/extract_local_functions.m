function extract_local_functions(sourceFile, outputFolder)
    % Create the output directory if it doesn't exist
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % Read the entire file content
    fileContent = fileread(sourceFile);
    
    % Split the file at every line that starts with 'function'
    % parts{1} will contain any text BEFORE the first function (header comments, etc.)
    [parts, ~] = regexp(fileContent, '(?m)^function\s+', 'split');
    
    % Get the actual 'function ' strings to re-attach them later
    [names, ~] = regexp(fileContent, '(?m)^function\s+', 'match');

    if isempty(names)
        fprintf('No functions found in %s.\n', sourceFile);
        return;
    end

    % Iterate through every detected function block
    for i = 1:length(names)
        % Re-stitch the 'function' keyword with its corresponding body
        % parts{i+1} because the first element of 'parts' is what precedes the first match
        funcCode = [names{i}, parts{i+1}];
        
        % Clean up trailing whitespace/newlines
        funcCode = strtrim(funcCode);
        
        % Extract the function name to use as the filename
        % This handles: function name(), function [out] = name(), function out = name()
        nameMatch = regexp(funcCode, 'function\s+(?:\[?.*?\]?=\s*)?(\w+)', 'tokens');
        
        if ~isempty(nameMatch)
            funcName = nameMatch{1}{1};
            fullOutputPath = fullfile(outputFolder, [funcName, '.m']);
            
            % Write the individual .m file
            fid = fopen(fullOutputPath, 'w');
            if fid == -1
                fprintf('Error: Could not create file %s\n', fullOutputPath);
                continue;
            end
            fprintf(fid, '%s', funcCode);
            fclose(fid);
            
            fprintf('Extracted: %s.m\n', funcName);
        else
            fprintf('Warning: Could not determine function name for block %d\n', i);
        end
    end
    fprintf('--- Extraction complete. ---\n');
end