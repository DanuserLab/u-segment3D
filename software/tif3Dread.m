function image = tif3Dread(filename)
%TIF3DREAD uses imread to read every page in a multi-page (3D) tif
% 
% image = tif3Dread(filename)
% 
% This reads every image in the specified 3D (multi-page) .tif file and
% combines all the images into a single 3D matrix.
% 
% Input:
% 
%   filename - The name of the multi-page .tif file to read.
% 
% 
% Output:
% 
%   image - The 3D matrix containing the concatenated images.
%
% NOTE: This function is semi-redundant with stackRead.m. The only
% difference is that it supports all the compression formats used by
% imwrite.m, and stackRead.m does not. However, stackRead.m is faster so
% use that unless you are dealing with compressed multi-page .tifs which
% have been created in matlab.
%
% Hunter Elliott
% 1/2010
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

if nargin < 1 || isempty(filename)
    error('Must input a file name!')
end

if ~exist(filename,'file')
    error('Specified file does not exist!');
end

%Get the file information
info = imfinfo(filename);
nPages = numel(info);

if nPages < 1
    error('Either the specified image file is not a valid 3D image, or this function does not support the format! Check the file, or try using stackRead.m!')
end

%Initialize the image array to the correct size and class
imSize = [info(1).Height info(1).Width nPages];

%Check if each plane is RGB
if strcmp(info(1).PhotometricInterpretation,'RGB');
    imSize = [imSize 3];       
    info(1).BitDepth = info(1).BitDepth/3;%And correct the bit depth
end

if info(1).BitDepth == 1    
    %Special case for logical - zeros.m doesn't support initialization of binary arrays.
    image = false(imSize);    
elseif any(info(1).BitDepth == [8 16 32 64])
    %Initialze in the correct class
    image = zeros(imSize,['uint' num2str(info(1).BitDepth)]);
else
    %If not recognized, just use double
    image = zeros(imSize);
end

for i=1:nPages
   
    %Load the image and add it to the array
    image(:,:,i,:) = imread(filename,i,'Info',info);
    
end
