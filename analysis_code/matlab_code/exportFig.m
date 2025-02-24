function  figTable = exportFig(fig,fileName,figType,figRes,inpCell,varNames,exportFile)
%exportFig: export both figure and data in the shape of obs x var table 
%   Detailed explanation goes here

% export figure first
printFig(fig,fileName,figType,figRes);
fileName = [fileName,'_',date];

% prep data table
figTable=[];
if ~isempty(inpCell)
if isnumeric(inpCell)
    figTable = inpCell;
else
    nGroup = size(inpCell,2);
    maxGroup = nan(1,nGroup);
    df=[];
    for group=1:nGroup
        df{group} = reshape(inpCell{group},[],1);
        maxGroup(group) = size(df{group},1);
    end
    tempMat = nan(max(maxGroup),nGroup);
    for group=1:nGroup
        xx = [];
        xx = df{group};
        tempMat(1:size(xx,1),group)= xx;
    end
    figTable = tempMat;
end
% export data table
figTable=array2table(figTable,'VariableNames',varNames);
writetable(figTable,([fileName,'.csv']));
end
end
