function level = thresholdRosin(imageIn,varargin)
% Select a thresholding level using Rosin's method
%
% level = thresholdRosin(imageIn)
% level = thresholdRosin(imageIn,showPlots)
% 
% This function selects a threshold for the input fluorescence image by
% analyzing the image's intensity distribution. This requires good signal-
% to-noise, and a significant amount of background in the image. The
% threshold is selected using Rosin's method
% 
% Input:
% 
%   imageIn - The N-Dimensional image to be thresholded.
% 
% 
%   showPlots - If true, a plot of the histogram and an overlay of the mask
%   on the image will be shown. The overlay plot only works if the image is
%   2D.
% 
% 
% Output:
% 
% 
%   level - The intensity value selected for thresholding.
%
% Revamped from RosinSeg
%
% Sebastien Besson, 5/2011
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

ip=inputParser;
ip.addRequired('imageIn',@isnumeric);
ip.addOptional('showPlots',0,@isnumeric)
ip.parse(imageIn,varargin{:});
showPlots=ip.Results.showPlots;

%Convert to double if necessary
imageIn = double(imageIn);

%find nonzero values (due to masking)
nzInd = find(imageIn);

%get minumum and maximum pixel values in image
minSignal = min(imageIn(nzInd));
maxSignal = max(imageIn(nzInd));

%normalize nonzero value between 0 and 1
imageInNorm = zeros(size(imageIn));
imageInNorm(nzInd) = (imageIn(nzInd)- minSignal) / (maxSignal - minSignal);

[~,level] =  cutFirstHistMode(imageInNorm,0);

level = level*(maxSignal - minSignal)+minSignal;

if showPlots 
    imageMask = imageIn >= level;
    figure;
    imagesc(imageIn);
    hold on
    contour(imageMask,'w')
    colormap hot
end
