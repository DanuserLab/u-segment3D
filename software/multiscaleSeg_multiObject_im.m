function [finalMask, voteScoreImg] = multiscaleSeg_multiObject_im(im, varargin)
% multiscaleSeg_singleObject_im Segment a single cell image by combining segmentation
% results obtained at multiple smoothing scales. Since it requires only one
% tuning parameters (tightness) and ‘tightness’=0.5 works well for many cases, 
% it achieves almost automatic segmentation.
%
% Input: an image
% Output: a mask and a transformed image with the value of voting scores
% (0 <= score <= 24)
% Usage:
%       rmask = multiscaleSeg_multiObject_im(im, 'tightness', 0.5);
%
% Updates:
% 2018/04/20, Jungsik Noh. Modified from multiscaleSeg_im(). Options are
% simplified.
% 2017/05/29, Jungsik Noh
% Updated Andrew R. Jamieson - Sept. 2017
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

ip = inputParser;
ip.addRequired('im');
ip.addParameter('tightness', 0.5, @(x) isnumeric(x) && (x==-1 || x >= 0 || x<=1));
ip.addParameter('numVotes', -1);
ip.addParameter('finalRefinementRadius', 1);
ip.addParameter('MinimumSize', 10);
ip.addParameter('ObjectNumber', 1000);

ip.parse(im, varargin{:});
p = ip.Results;

if (p.numVotes > 0); p.tightness = -1; end


%sigmas = [0 0.66 1 1.66 2.5 4];  % unit: pixel (common scales for xxx by xxx size confocal images)
sigmas = [0 0.5 1 1.5 2 2.5 3];  % unit: pixel (common scales for xxx by xxx size confocal images)
%p.MinimumSize = 100;        
%p.ObjectNumber = 1;
p.FillHoles = 1;

numModels = numel(sigmas)*3*2;
maskingResultArr = nan(size(im, 1), size(im, 2), numModels);

%%  Gaussian filtering, method, refinement
for k = 1:numel(sigmas)
    currImage = im;

    if sigmas(k) > 0
        currImage = filterGauss2D(currImage, sigmas(k));
    end

    % minmax
    try
        currThresh = thresholdFluorescenceImage(currImage); 
        currMask1 = (currImage >= currThresh);

        p.ClosureRadius = 1;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+1) = refinedMask1;

        p.ClosureRadius = 3;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+2) = refinedMask1;

    catch
        disp(['GaussFilterSigma: ', num2str(sigmas(k))])
        disp('Error in Minmax thresholding')
        maskingResultArr(:,:, 6*(k-1)+1) = nan(size(currImage, 1), size(currImage, 2));
        maskingResultArr(:,:, 6*(k-1)+2) = nan(size(currImage, 1), size(currImage, 2));
    end

    % Rosin
    try
        currThresh = thresholdRosin(currImage); 
        currMask1 = (currImage >= currThresh); 

        p.ClosureRadius = 1;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+3) = refinedMask1;

        p.ClosureRadius = 3;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+4) = refinedMask1;
        
    catch
        disp(['GaussFilterSigma: ', num2str(sigmas(k))])
        disp('Error in Rosin thresholding')
        maskingResultArr(:,:, 6*(k-1)+3) = nan(size(currImage, 1), size(currImage, 2));
        maskingResultArr(:,:, 6*(k-1)+4) = nan(size(currImage, 1), size(currImage, 2));        
    end
    
    
    % Otsu
    try
        currThresh = thresholdOtsu(currImage); 
        currMask1 = (currImage >= currThresh); 

        p.ClosureRadius = 1;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+5) = refinedMask1;

        p.ClosureRadius = 3;
        refinedMask1 = maskRefinementCoreFunc(currMask1, p);
        maskingResultArr(:,:, 6*(k-1)+6) = refinedMask1;
    catch
        disp(['GaussFilterSigma: ', num2str(sigmas(k))])
        disp('Error in Otsu thresholding')
        maskingResultArr(:,:, 6*(k-1)+5) = nan(size(currImage, 1), size(currImage, 2));
        maskingResultArr(:,:, 6*(k-1)+6) = nan(size(currImage, 1), size(currImage, 2));        
    end
    
end



%% sum maskings from multiple methods

res_0 = mean(maskingResultArr(:,:,1:end), 3, 'omitnan');
res = round(res_0 .* numModels);
tab = tabulate(res(:));
tabulate(res(:));

val = tab(:,1);
counts = tab(:,2);

[~, ind] = sort(counts, 'descend');

a = val(ind(1));
b = val(ind(2));
backgroundth = min(a, b);
maskth = max(a, b);


%% ensemble method
if (p.numVotes < 0)
 
        if (p.tightness > 1) || (p.tightness < 0)
            error('Tightness should range from 0 to 1.')
        end

        mnum_smallest = maskth;
        mnum_biggest = backgroundth + 1;
        mnum_interp = interp1([0, 1], [mnum_biggest, mnum_smallest], p.tightness);
        tightnessNumModel = round(mnum_interp);
        numVotes = tightnessNumModel;

        res0 = (res >= numVotes);
        disp('Threshold of votes: mask = (Value >= threshold)'); 
        disp([num2str(numVotes), ' (tightness: ', num2str(p.tightness), ')'])

else
        res0 = (res >= p.numVotes);
        disp('Threshold of votes: mask = (Value >= threshold)'); 
        disp(num2str(p.numVotes))
end

%% final refinement
p.ClosureRadius = p.finalRefinementRadius;
finalMask = maskRefinementCoreFunc(res0, p);

res_scaled = res ./ numModels .* 200;
voteScoreImg = uint8(res_scaled);
 
end

function currMask = maskRefinementCoreFunc(currMask, p)
% maskRefinementCoreFunc is a part of refineMovieMasks.m to implement
% imclose, object number and filling holes.
% 2017/05/29

% ----- Mask Clean Up ------ %
        
seClose = strel('disk',p.ClosureRadius,0);

            %Remove objects that are too small
            if p.MinimumSize > 0
                currMask = bwareaopen(currMask,p.MinimumSize);
            end            
            
            %Perform initial closure operation
            if p.ClosureRadius > 0
                currMask = imclose(currMask,seClose);            
            end            
            
            %%Perform initial opening operation
            %if p.OpeningRadius > 0
            %    currMask = imopen(currMask,seOpen);
            %end
            
        
        
        % ---------- Object Selection -------- %
        
        %Keep only the largest objects
        if ~isinf(p.ObjectNumber)
                
            %Label all objects in the mask
            labelMask = bwlabel(currMask);

            %Get their area
            obAreas = regionprops(labelMask,'Area');       %#ok<MRPBW>

            %First, check that there are objects to remove
            if length(obAreas) > p.ObjectNumber 
                obAreas = [obAreas.Area];
                %Sort by area
                [dummy,iSort] = sort(obAreas,'descend'); %#ok<ASGLU>
                %Keep only the largest requested number
                currMask = false(size(currMask));
                for i = 1:p.ObjectNumber
                    currMask = currMask | labelMask == iSort(i);
                end
            end
        end
        
        % ------ Hole-Filling ----- %
        if p.FillHoles
            
            %If the mask touches the image border, we want to close holes
            %which are on the image border. We do this by adding a border
            %of ones on the sides where the mask touches.
            if any([currMask(1,:) currMask(end,:) currMask(:,1)' currMask(:,end)'])                
                m = size(currMask, 1);
                n = size(currMask, 2);            
                %Add a border of 1s
                tmpIm = vertcat(true(1,n+2),[true(m,1) ...
                                currMask true(m,1)],true(1,n+2));
                
                %Find holes - the largest "hole" is considered to be the
                %background and ignored.
                cc = bwconncomp(~tmpIm,4);                                
                holeAreas = cellfun(@(x)(numel(x)),cc.PixelIdxList);
                [~,iBiggest] = max(holeAreas);                                
                tmpIm = imfill(tmpIm,'holes');
                tmpIm(cc.PixelIdxList{iBiggest}) = false;
                currMask = tmpIm(2:end-1,2:end-1);
             else                        
                 currMask = imfill(currMask,'holes');
            end
        end
        
end
