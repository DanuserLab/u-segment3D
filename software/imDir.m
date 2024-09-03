function [fileNames, formatNum, sNums] = imDir(imDirectory,returnAll)
%IMDIR is a wrapper for the dir command designed for finding only image files
% 
% fileNames = imDir(directory);
% 
% fileNames = imDir(directory,returnAll);
%
% [fileNames formatNum sNums] = imDir(...);
%
% This function will find all files in the specified directory with common
% file extensions for images. Additionally, the images will be re-ordered,
% if necessary, so that the last number before the file extension is in
% increasing order. This fixes the problem with the dir command returning
% numbered images which are not zero-padded in the wrong order.
%
% For example, if a folder contains img1.tif, img2.tif ... img10.tif, the
% dir command will return img1.tif, img10.tif, img2.tif ..., whereas this
% function will return them in the correct order, with img10.tif last. This
% only works with files where the image number is the last element of the
% name before the file extension.
% 
% Input:
% 
%   directory - the directory to search for files in. (non-recursive);
%   Optional. If not input, current directory is used.
%
%   returnAll - If true, all images of any file extension will be returned.
%   If false, only the first matching set of files found will be returned,
%   in this order:
% 
%       1 - .tif 
%       2 - .TIF 
%       3 - .STK 
%       4 - .bmp 
%       5 - .BMP 
%       6 - .jpg
%       7 - .JPG
%       8 - .JP2
%       9 - .JPX
%       10 - .png
%
%   This input is optional. Default is false.
%
% Output:
%
%   fileNames - a structure containing the names and statistics for all the
%   images found. This is the same as the output of the dir function.
%
%   formatNum - If returnAll was enabled, this is the number of different
%   image file extensions found in the directory 
%
%   sNums - an array containing the number at the end of each file 
%
% Hunter Elliott, 2/2010
%
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

%The list of supported file extensions. Feel free to add! (just update the
%help also!)
fExt = {'tif','tiff', 'stk', 'bmp', 'jpg','jp2','jpx','png'};
if ~ispc && ~(ismac && ~verLessThan('matlab', '8.3'))
    % Add case-sensitivity under unix based platforms
    fExt =  reshape(vertcat(fExt,upper(fExt)),1,2*numel(fExt));
end


if nargin < 1 || isempty(imDirectory)
    imDirectory = pwd;
end

if nargin < 2 || isempty(returnAll)
    returnAll = false;
end

fileNames = cell(length(fExt), 1);
formatNum = 0;

% ---- Get the file names by checking each extension.  ---- %
for i = 1:length(fExt)
    
    fileNames{i} = dir([imDirectory filesep '*.' fExt{i}]);
    if ~isempty(fileNames{i})
        formatNum = formatNum +1;
    end
    
    if ~returnAll && ~isempty(fileNames{i});
        break
    end
end
fileNames = vertcat(fileNames{:});

%  ---- Fix the order of the files if they are numbered.  ---- %

%First, extract the number from the end of the file, if present
fExtNum = arrayfun(@(x)(min(regexp(x.name,'\.'))),fileNames); %Find beginning of file extension. Handles special cases like .tiff and .ome.tif
fNums = arrayfun(@(x)(str2double(...
    fileNames(x).name(max(regexp(fileNames(x).name(1:fExtNum(x)-1),'\D'))+1:fExtNum(x)-1))),1:numel(fileNames));

count = getMultiplicity(fNums);

if(all(count == 1))
    % Only sort by the number at the end if the numbers are unique

    %The sort function handles NaNs, and will not re-order if the images are
    %not numbered.
    [sNums,iX] = sort(fNums);

    fileNames = fileNames(iX);
else
    disp(['imDir:non-unique-numbers: The numbers at the end of the filenames are not unique. Not sorting.']);
end

