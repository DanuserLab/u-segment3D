function threshold = intensityBinnedGradientThreshold(im,binSize,sigma,smoothPar,force2D,method,maskIn)
%INTENSITYBINNEDGRADIENTTHRESH thresholds an image combining gradient and intensity information
% 
% threshold = intensityBinnedGradientThreshold(im)
% threshold = intensityBinnedGradientThreshold(im,binSize)
% threshold = intensityBinnedGradientThreshold(im,binSize,sigma)
% 
% This function selects a threshold for the input image using both spatial
% gradient and absolute intensity information. For each set of intensity
% value(s), the spatial gradient values are averaged, giving a relationship
% between intensity and gradient. Then, the lowest intensity value which is
% a local maximum of gradient values is selected. This therefore attempts
% to select an intensity threshold which coincides with an area of high
% intensity gradients.
% 
% 
% Input:
% 
%   im - The image to threshold. May be 2D or 3D.
% 
%   binSize - The size of bins to group intensity values for calculating
%   average gradient. Smaller values will give more accurate threshold
%   values, but larger values will speed the calculation. Optional. Default
%   is 10.
% 
%   sigma - The sigma to use when calculating the smoothed gradient of the
%   input images. If zero, no smoothing is performed. Optional. Default is
%   1.
% 
%   smoothPar - The parameter of the smoothing spline used to select local
%   maxima in the gradient vs intensity curve. Scalar between 0 and 1,
%   where smaller values give more smoothing. Optional. Default is 1e-5
%
%   force2D - If true and a 3D matrix is input, it is assumed to be a stack
%   of 2D images, and the gradient is calculated in 2D along the first 2 dimensions.
%
%   method - LocalMaxima or Shoulder. LocalMaxima is as described above.
%   "Shoulder" finds the threshold corresponding to the maximum value of
%   the 2nd derivative of the int. vs. gradient histogram below the first
%   local maxima. This is good for objects with mixed intensities - will
%   tend to find threshold which matches the edge of the dimmest object.
%
%   maskIn - Optionally input a mask to specify pixels to use when
%   constructing gradient histogra (areas with false in mask are ignored).
%
% Output:
% 
%   threshold - The selected threshold value. If a threshold could not be
%   selected, NaN is returned.
% 
% Hunter Elliott
% 8/2011
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

showPlots = false;%Plots for testing/debugging

if nargin < 1 || isempty(im) || ndims(im) < 2 || ndims(im) > 3
    error('The first input must be a 2D or 3D image!!');
end

if nargin < 2 || isempty(binSize)
    binSize = 10;
elseif numel(binSize) > 1 || binSize <= 0
    error('the binSize input must be a scalar > 0!')
end

if nargin < 3 || isempty(sigma)
    sigma = 1;
elseif numel(sigma) > 1 || sigma < 0
    error('The input sigma must be a scalar >= zero!')
end

if nargin < 4 || isempty(smoothPar)
    smoothPar = 1e-5;
end

if nargin < 5 || isempty(force2D)
    force2D = false;
end

if nargin < 6 || isempty(method)
    method = 'LocalMaxima';
end

if nargin < 7 || isempty(maskIn)
    maskIn = true(size(im));
end


im = double(im);
nPlanes = size(im,3);
intBins = min(im(:)):binSize:max(im(:))+1;
if numel(intBins) < 3
    threshold = NaN;
    return
end

gradAtIntVal = zeros(1,numel(intBins)-1);

if nPlanes == 1 || force2D
    for j = 1:nPlanes

        %TEMP - CONVERT THIS TO GRADIENT FILTERING!!!! (rather than
        %filtering then gradient calc) - HLE
        %Smooth the image
        if sigma > 0
            currIm = filterGauss2D(im(:,:,j),sigma);
        else
            currIm = im(:,:,j);
        end
        currMask = maskIn(:,:,j);
        
        %Get gradient of image.
        [gX,gY] = gradient(currIm);
        g = sqrt(gX .^2 + gY .^2);   
        
        g(~currMask) = NaN;

        %Get average gradient at each intensity level
        tmp = arrayfun(@(x)(nanmean(nanmean(double(g(currIm >= intBins(x) & currIm < intBins(x+1)))))),1:numel(intBins)-1);
        tmp(isnan(tmp)) = 0;
        %Add this to the cumulative average
        gradAtIntVal = gradAtIntVal + (tmp ./ nPlanes);

    end
else
    %Do 3D gradient calc
    if sigma > 0
        [dX,dY,dZ] = gradientFilterGauss3D(im,sigma);
    else
        [dX,dY,dZ] = gradient(im);
    end
    
    g = sqrt(dX .^2 + dY .^2 + dZ .^2);        
    
    g(~maskIn) = NaN;
    
    gradAtIntVal = arrayfun(@(x)(mean(double(g(im(:) >= intBins(x) & im(:) < intBins(x+1))))),1:numel(intBins)-1);            
    
end

%Smooth the grad/int data
binCenters = intBins(1:end-1) + (binSize/2);%Use center of bins for x values
ssGradInt = csaps(binCenters,gradAtIntVal,smoothPar);
smGradInt = fnval(ssGradInt,binCenters);
%Find maxima
spDer = fnder(ssGradInt,1);
spDer2 = fnder(ssGradInt,2);

extrema = fnzeros(spDer);
extrema = extrema(1,:);

%evaluate 2nd deriv at extrema
secDerExt = fnval(spDer2,extrema);
%Find the first maximum
iFirstMax = find(secDerExt < 0,1);        
extVals = fnval(ssGradInt,extrema);

switch method

    
    case 'LocalMaxima'
        
        if ~isempty(iFirstMax)
            threshold = extrema(iFirstMax);
        else
            threshold = NaN;
        end
        
        if showPlots
            fsFigure(.5);
            plot(binCenters,gradAtIntVal);
            hold on    
            plot(binCenters,smGradInt,'r','LineWidth',2)    
            if ~isempty(extrema)
                plot(extrema,extVals,'om','MarkerSize',15);    
            end    
        end
        
    case 'Shoulder'
        
        %shoulders = binCenters(locmax1d(-fnval(spDer2,binCenters),3));                
        
        if isempty(iFirstMax)
            threshold = NaN;
        else
            possBins = binCenters(binCenters<=extrema(iFirstMax));
            secDerVal = fnval(spDer2,possBins);
            [~,iMin2nd] = min(secDerVal);         
            
            threshold = possBins(iMin2nd);
        end
                  
        if showPlots
            fsFigure(.5);
            subplot(2,1,1)
            plot(binCenters,gradAtIntVal);
            hold on    
            plot(binCenters,smGradInt,'r','LineWidth',2)    
            plot(threshold,fnval(ssGradInt,threshold),'go')
            subplot(2,1,2)
            plot(binCenters,fnval(spDer2,binCenters))
            hold on
            plot(xlim,[0 0],'--k')            
            plot(threshold,fnval(spDer2,threshold),'go')
        end
        
        
end 
            
