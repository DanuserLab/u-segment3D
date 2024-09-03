function stackWrite(im,fileName,compType)
%STACKWRITE writes a 3D grayscale or RGB image matrix to a single multi-page .tif file 
% 
% stackWrite(im,fileName)
% stackWrite(im,fileName,compression)
% 
% This function writes the input 3D/4D matrix to a SINGLE, multi-page .tif
% file with the specified file name. Compression is disabled by default to
% increase compatability. To write a 3D/4D image to multiple .tif images, use
% imwritestack.m or a similar function.
% 
% Input:
%   
%   im - The 3D/4D image matrix to write to file. The resulting image
%   bit-depth will depend on the class of this image. If 4D, the 3rd
%   dimension must be 3 to be recognized as RGB image.
% 
%   fileName - The file name to write the image to, WITH file extension.
%
%   compression - A string specifying the type of compression to use. The
%   possible options are described in the help for imwrite.m Note that many
%   of the compression types are incompatible with the stackRead.m function
%   so will require you to use tif3Dread.m to open the resulting stack.
%   Optional. Default is 'none'.
% 
% Output:
%
%   The input image matrix will be written to disk. If a file already
%   exists with that name, it will be overwritten.
%
% See Also:
%
%   stackRead.m, tif3Dread.m, imwrite.m
%
% Hunter Elliott
% 2/2010
%
% Last modified by Tiao Xie 02/2013 to accomodate 3D RGB stacks (4D matrix)
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

if nargin < 2 || isempty(im) || isempty(fileName)
    error('You must input an image and file name!')
end

dimCount=ndims(im);
if dimCount < 3 || dimCount > 4
    error('Input image must be 3D or 4D!')
end

if dimCount ==4
    nDim3rd=size(im,3);
    if nDim3rd ~= 3
        error('For 4D matrix, the 3rd dimension must be 3!')
    end
end

if nargin < 4 || isempty(compType)
    compType = 'none';
end

%Write the first slice in overwrite mode, in case the file already exists.
if dimCount == 3
    %3D matrix
    imwrite(im(:,:,1),fileName,'tif','Compression',compType)
    
    %All successive z-slices are appended.
    for i = 2:size(im,3)
        imwrite(im(:,:,i),fileName,'tif','WriteMode','append','Compression',compType)
    end
else
    %4D matrix
    imwrite(im(:,:,:,1),fileName,'tif','Compression',compType)
    
    %All successive z-slices are appended.
    for i = 2:size(im,4)
        imwrite(im(:,:,:,i),fileName,'tif','WriteMode','append','Compression',compType)
    end
end