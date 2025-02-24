function [outFig] = printFig(outFig,outName,fileType,res)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
outputFilename = [outName,'_',date];
print(outFig, ['-d',fileType], res, [outputFilename, '.',fileType]);
end