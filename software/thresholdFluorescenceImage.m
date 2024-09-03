function thresholdValue = thresholdFluorescenceImage(imageIn,showPlots,noisy)

%
% thresholdValue = thresholdFluorescenceImage(imageIn)
% 
% thresholdValue = thresholdFluorescenceImage(imageIn,showPlots,noisy)
% 
% This function selects a threshold for the input fluorescence image by
% analyzing the image's intensity distribution. This requires good signal-
% to-noise, and a significant amount of background in the image. The
% threshold is selected by first finding the lowest intensity maximum in the
% image histogram (the background). Then the first minimum in the histogram
% of higher intensity than this peak is selected as the threshold.
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
%   noisy = 0/1. If 1 the drop from the first maximum to the first minimum
%                must exceed the MAD of the difference between
%                the extrema values. Otherwise the first maximum/minimum
%                pair is rejected and the next is considered. This is only
%                a reasonable constraint if the spline fit yields many
%                extrema (bumpy spline) and needs to be smoothed (at least
%                10 max and min respectively). 
%                
% 
% Output:
% 
% 
%   thresholdValue - The intensity value selected for thresholding.
%
%
%Hunter Elliott, 11/7/08
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

if nargin < 2 || isempty(showPlots)
    showPlots = 0;
end

%Convert to double if necessary
imageIn = double(imageIn);

%Get histogram, using Jonas' automatic bin number selection & smoothing
[vals,bins,histSpline] = optimalHistogram(imageIn(:),'smooth');

%Find the location of extrema in the histogram
histExtrema = fnzeros(fnder(histSpline));

%Get rid of the 'fake' extrema sometimes produced at beginning and end of
%distrubution by spline form.
histExtrema = histExtrema(1,:); %remove the intervals
histExtrema = histExtrema((histExtrema ~= ... %These will always be at first or last breaks in spline
            histSpline.breaks(1)) & (histExtrema ~= histSpline.breaks(end)));
histExtVals = fnval(histSpline,histExtrema);


%Determine whether each extrema is maximum or minimum
isMax = fnval(fnder(histSpline,2),histExtrema) < 0;

%Find the lowest-intensity maximum, assume this is the background peak.
iBackMax = find(isMax,1,'first');

%Find the first minimum after this maximum. This is used as the threshold.
iSep = iBackMax + 1;

% we don't pick minima where the difference between background maximum and
% minimum is marginal:
if nargin>2 && noisy && numel(histExtVals)>20
    extVal= fnval(histSpline,histExtrema);
    extDiff=(extVal(2:end)-extVal(1:end-1));
    madDiff=mad(abs(extDiff));
    
    while iBackMax<=length(histExtrema) && (extVal(iBackMax)-extVal(iBackMax+1))<madDiff
        iBackMax=iBackMax+2;
        iSep = iBackMax + 1;
    end
end

if iSep > length(histExtrema);
    error('Could not automatically determine a threshold value!');
end

thresholdValue = histExtrema(iSep);
minVal = histExtVals(iSep);

if showPlots    
    imageMask = imageIn >= thresholdValue;
    histFig = fsFigure(.5);
    hist(imageIn(:),500)
    hold on
    fnplt(histSpline,'r')        
    plot(histExtrema,histExtVals,'ok')
    plot(thresholdValue,minVal,'xr')
    
    if ndims(imageIn) == 2    
        maskFig = figure;
        imshow(imageIn,[]);
        hold on
        %contour(imageMask,[.5 .5],'w')
        spy(bwperim(imageMask))        
    end
end
