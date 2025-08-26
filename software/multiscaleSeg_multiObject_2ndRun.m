function masksCell = multiscaleSeg_multiObject_2ndRun(scoreArray, varargin)
% multiscaleSeg_multiObject_2ndRun Segment a single cell image 
% when a voting score array is already given from previous running 
% of MSA seg algorithm.
%
% Updates:
%
% 2024/11. J Noh. 
%
% Copyright (C) 2025, Danuser Lab - UTSouthwestern 
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

%% parse input

ip = inputParser; 
ip.addParameter('tightness', 0.5, @(x) isnumeric(x) && (x==-1 || x >= 0 || x<=1));
ip.addParameter('numVotes', -1);
ip.addParameter('finalRefinementRadius', 1);
ip.addParameter('MinimumSize', 10);
ip.addParameter('ObjectNumber', 1000);
ip.addParameter('verbose', 'off');

ip.parse(varargin{:});
p = ip.Results;

if (p.numVotes > 0); p.tightness = -1; end

%sigmas = [0 0.66 1 1.66 2.5 4];  % unit: pixel (common scales for xxx by xxx size confocal images)
sigmas = [0 0.5 1 1.5 2 2.5 3];  % unit: pixel (common scales for xxx by xxx size confocal images)
%p.MinimumSize = 100;        
%p.ObjectNumber = 1;
p.FillHoles = 1;

%numModels = numel(sigmas)*3*2;
%maskingResultArr = nan(size(im, 1), size(im, 2), numModels);

%% sum maskings from multiple methods

frmax = size(scoreArray, 3);
masksCell =  cell(frmax, 1);

for fr = 1:frmax

    res = scoreArray(:,:,fr);

    tab = tabulate(res(:));
    if isequal(p.verbose, 'on')
        disp('=====')
        disp(['Frame: ', num2str(fr)])  
        tabulate(res(:)); % use verbose param to turn off - QZ
    end

    if (p.numVotes < 0)    
    
        if (p.tightness > 1) || (p.tightness < 0)
            error('Tightness should range from 0 to 1.')
        end

        val = tab(:,1);
        counts = tab(:,2);

        [~, ind] = sort(counts, 'descend');

        a = val(ind(1));
        b = val(ind(2));
        backgroundth = min(a, b);
        maskth = max(a, b);

        % thresholding
        mnum_smallest = maskth;
        mnum_biggest = backgroundth + 1;
        mnum_interp = interp1([0, 1], [mnum_biggest, mnum_smallest], p.tightness);
        tightnessNumModel = round(mnum_interp);
        numVotes = tightnessNumModel;

        res0 = (res >= numVotes);
        if isequal(p.verbose, 'on')
        disp('Threshold of votes: mask = (Value >= threshold)'); 
        disp([num2str(numVotes), ' (tightness: ', num2str(p.tightness), ')'])
        end

    else
        res0 = (res >= p.numVotes);
        if isequal(p.verbose, 'on')
        disp('Threshold of votes: mask = (Value >= threshold)'); 
        disp(num2str(p.numVotes))
        end
    end
    
    % final refinement    
    p.ClosureRadius = p.finalRefinementRadius;
    masksCell{fr} = maskRefinementCoreFunc(res0, p);    
end
  
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
